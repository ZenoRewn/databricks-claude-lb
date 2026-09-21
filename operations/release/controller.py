"""Two-replica coordinator; durable state survives the submitting client. Author: Zeno Ren."""
import argparse
import base64
from collections import Counter
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
import signal
import threading
import time
import uuid

from .backend import Backend
from .engine import Engine,TERMINAL
from .kube import KubeAPI,Lease,RecordStore
from .model import digest,validate_plan


class Controller:
    def __init__(self,api,namespace,targets,identity,*,lab=False):
        self.api,self.namespace,self.targets,self.identity=api,namespace,set(targets),identity
        self.lab=lab;self.stop=threading.Event();self.leader=None;self.jobs={};self.ready=True;self.last_error=None
        self.phase_counts={};self.oldest_maintenance=None;self.last_scan=None
        self.faults=None
        if lab:
            if not all(ns.startswith('lb-lab') for ns,_ in self.targets):raise ValueError('Lab targets must be isolated')
            from .lab_faults import LabFaults
            self.faults=LabFaults(api,namespace)
    def tick(self):
        if self.leader is None or self.leader.lost:
            if self.leader:self.leader.close()
            for lease in self.jobs.values():lease.close()
            self.jobs={};self.leader=Lease(self.api,self.namespace,'lb-release-coordinator',self.identity)
            if not self.leader.acquire():return
            self.leader.start()
        self.leader.check()
        items=self.api.list('configmap',self.namespace,'app=lb-release-record')
        states=[json.loads(o['data']['record.json']) for o in items]
        self.phase_counts=dict(Counter(r['phase'] for r in states))
        starts=[r['maintenance_started_at'] for r in states if r.get('maintenance_started_at') is not None
                and r.get('maintenance_finished_at') is None and r['phase'] not in ('succeeded','rolled_back','cancelled')]
        self.oldest_maintenance=min(starts) if starts else None;self.last_scan=time.time()
        for obj in sorted(items,key=lambda x:x['metadata']['creationTimestamp']):
            self.leader.check();name=obj['metadata']['name'];store=RecordStore(self.api,self.namespace,name)
            record=store.read()
            if record['phase'] in TERMINAL:
                if name in self.jobs:self.jobs.pop(name).close()
                continue
            plan=validate_plan(record['plan'])
            if (plan['namespace'],plan['deployment']) not in self.targets:raise ValueError('target_not_allowlisted')
            immutable=self.api.get('configmap',self.namespace,'lb-plan-'+plan['release_id'])
            if not immutable.get('immutable') or json.loads(immutable['data']['plan.json'])!=plan:
                raise ValueError('immutable_plan_changed')
            if obj['metadata']['annotations'].get('plan-uid')!=immutable['metadata']['uid']:
                raise ValueError('plan_identity_changed')
            if name not in self.jobs:
                lock='lb-target-'+digest([plan['namespace'],plan['deployment']])[:24]
                lease=Lease(self.api,self.namespace,lock,self.identity+':'+plan['release_id'])
                if not lease.acquire():continue
                lease.start();self.jobs[name]=lease
            lease=self.jobs[name]
            if lease.lost:
                lease.close();del self.jobs[name];continue
            secret=self.api.get('secret',self.namespace,'lb-backup-'+plan['release_id'])
            snapshot=json.loads(base64.b64decode(secret['data']['snapshot.json'],validate=True))
            if digest(snapshot)!=plan['snapshot_sha256']:raise ValueError('backup_identity_changed')
            backend=Backend(self.api,store,plan,snapshot,[self.leader,lease],controller_namespace=self.namespace,lab=self.lab)
            if self.faults:
                self.faults.pause_at(record,self.stop)
                if self.stop.is_set():return
                original=backend.perform
                def perform(action,current):
                    self.faults.before_action(action,current)
                    return original(action,current)
                backend.perform=perform
            # Quiescence observations must survive loop iterations, not be reset
            # by an adapter constructor. They need not survive a leader change.
            state=getattr(lease,'observations',{})
            backend.quiet_since=state.get('quiet_since');backend.last_started=state.get('last_started')
            result=Engine(store,backend).tick()
            lease.observations={'quiet_since':backend.quiet_since,'last_started':backend.last_started}
            if result['phase']!=record['phase']:
                print(json.dumps({'kind':'release_phase','release_id':plan['release_id'],'phase':result['phase']}),flush=True)
            if result['phase']=='needs_attention':
                print(json.dumps({'kind':'release_needs_attention','release_id':plan['release_id'],'cleanup_errors':result['cleanup_errors']}),flush=True)
                try:
                    now=datetime.now(timezone.utc).isoformat()
                    self.api.create('event',self.namespace,{'apiVersion':'v1','kind':'Event',
                        'metadata':{'generateName':'release-'+plan['release_id']+'-'},
                        'involvedObject':{'apiVersion':'v1','kind':'ConfigMap','namespace':self.namespace,'name':name,'uid':obj['metadata']['uid']},
                        'reason':'ReleaseNeedsAttention','message':'Release '+plan['release_id']+' requires state inspection; no success was declared.',
                        'type':'Warning','source':{'component':'lb-release-controller'},'firstTimestamp':now,'lastTimestamp':now,'count':1})
                except Exception:
                    print(json.dumps({'kind':'attention_event_write_failed','release_id':plan['release_id']}),flush=True)
    def serve(self,port):
        controller=self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ('/health/live','/health/ready'):
                    code=200 if controller.ready and not controller.stop.is_set() else 503
                    body=json.dumps({'ready':code==200,'leader':bool(controller.leader and not controller.leader.lost)}).encode()
                elif self.path=='/metrics':
                    code=200
                    values={'lb_release_controller_ready':int(controller.ready),
                            'lb_release_controller_leader':int(bool(controller.leader and not controller.leader.lost)),
                            'lb_release_needs_attention':controller.phase_counts.get('needs_attention',0),
                            'lb_release_maintenance_oldest_age_seconds':max(0,time.time()-controller.oldest_maintenance) if controller.oldest_maintenance else 0,
                            'lb_release_last_scan_timestamp_seconds':controller.last_scan or 0}
                    body=''.join(f'# TYPE {name} gauge\n{name} {value}\n' for name,value in values.items()).encode()
                else:code=404;body=b'{}'
                self.send_response(code);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('0.0.0.0',port),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        return server
    def run(self):
        while not self.stop.is_set():
            try:self.tick();self.last_error=None
            except Exception as exc:
                # API errors can contain resource bodies; only emit the class.
                kind=type(exc).__name__
                if kind!=self.last_error:print(json.dumps({'kind':'coordinator_error','error_type':kind}),flush=True)
                self.last_error=kind
            self.stop.wait(1)
        if self.leader:self.leader.close()
        for lease in self.jobs.values():lease.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace',default=os.getenv('POD_NAMESPACE'))
    parser.add_argument('--targets',default=os.getenv('LB_RELEASE_TARGETS'))
    parser.add_argument('--lab',action='store_true');parser.add_argument('--port',type=int,default=9000)
    args=parser.parse_args()
    if not args.namespace or not args.targets:parser.error('namespace and allowlisted targets are required')
    targets=[tuple(value.split('/')) for value in args.targets.split(',')]
    if not all(len(t)==2 and all(t) for t in targets):parser.error('targets must be namespace/deployment')
    identity=(os.getenv('POD_UID') or 'local')+':'+uuid.uuid4().hex[:12]
    controller=Controller(KubeAPI(in_cluster=True),args.namespace,targets,identity,lab=args.lab)
    signal.signal(signal.SIGTERM,lambda *_:controller.stop.set())
    signal.signal(signal.SIGINT,lambda *_:controller.stop.set())
    server=controller.serve(args.port)
    try:controller.run()
    finally:server.shutdown()


if __name__=='__main__':main()
