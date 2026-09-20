"""Assert real lab state after releases and faults. Author: Zeno Ren."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import time
import threading

from kind_lab import lab_api,NS,OPS
from operations.release.engine import TERMINAL
from operations.release.model import APP_FILES

cli=importlib.import_module('operations.release.__main__')


class Scenarios:
    def __init__(self,api,images,directory,output):
        self.api,self.images,self.directory,self.output=api,images,Path(directory),Path(output)
        self.directory.mkdir(mode=0o700,parents=True,exist_ok=True);self.output.mkdir(parents=True,exist_ok=True)
        self.hashes={}
        for image in (images['new'],images['alternate']):
            script='import pathlib,hashlib,json;print(json.dumps({n:hashlib.sha256(pathlib.Path("/app",n).read_bytes()).hexdigest() for n in '+repr(list(APP_FILES))+'}))'
            self.hashes[image]=json.loads(subprocess.check_output(['docker','run','--rm','--platform','linux/arm64','--entrypoint','python',image,'-c',script]))
    def record(self,rid):return json.loads(self.api.get('configmap',OPS,'lb-release-'+rid)['data']['record.json'])
    def fault(self,value):
        obj=self.api.optional('configmap',OPS,'lb-lab-fault')
        if obj:self.api.patch('configmap',OPS,obj,[{'op':'replace','path':'/data/fault.json','value':json.dumps(value)}])
        else:self.api.create('configmap',OPS,{'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'lb-lab-fault'},'data':{'fault.json':json.dumps(value)}})
    def current_image(self):return self.api.get('deployment',NS,'claude-lb')['spec']['template']['spec']['containers'][0]['image']
    def submit(self,rid,**overrides):
        current=self.current_image();image=self.images['alternate'] if current==self.images['new'] else self.images['new']
        plan={'release_id':rid,'namespace':NS,'deployment':'claude-lb','container':'claude-lb','source_revision':'0'*40,
              'source_hashes':self.hashes[image],'image':image,'lab':True,'direct_pod_access':False,
              'storage_compatibility':'additive-compatible','legacy_bootstrap':True,
              'public_urls':['http://claude-lb.lb-lab.svc','http://claude-lb-internal.lb-lab.svc'],
              'business_probes':[{'api':'messages','model':'claude-opus-5','stream':False,'max_tokens':32},
                                 {'api':'responses','model':'gpt-test','stream':True,'max_tokens':32},
                                 {'api':'chat','model':'gpt-test','stream':True,'max_tokens':32}]}
        plan.update(overrides)
        path=self.directory/rid;cli.plan_release(self.api,plan,path,'.');cli.submit(self.api,OPS,path)
        return current,image
    def wait(self,rid,phase=None,timeout=340):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            record=self.record(rid)
            if phase and record['phase']==phase:return record
            if record['phase'] in TERMINAL:
                if phase:raise AssertionError(f'{rid} ended at {record["phase"]}, expected {phase}')
                return record
            time.sleep(.4)
        raise TimeoutError(rid)
    def upstream_count(self):
        pod=self.api.list('pod',NS,'app=lab-upstream')[0]
        script='import urllib.request,json; print(json.load(urllib.request.urlopen("http://127.0.0.1:8000/"))["requests"])'
        return int(self.api.exec(NS,pod['metadata']['name'],'upstream',['python','-c',script]))
    def verify(self,rid,record,expected,image,initial_sends,expected_sends):
        if record['phase']!=expected:raise AssertionError(f'{rid}: '+record['phase']+' '+str(record['cleanup_errors']))
        if self.current_image()!=image:raise AssertionError('Unexpected final image')
        for name in ('claude-lb','claude-lb-internal'):
            svc=self.api.get('service',NS,name)
            if svc['spec']['selector']!={'app':'claude-lb'}:raise AssertionError('Orphaned routing gate')
            endpoints=[e for s in self.api.list('endpointslice',NS,'kubernetes.io/service-name='+name) for e in (s.get('endpoints') or []) if e.get('conditions',{}).get('ready')]
            if len(endpoints)!=1:raise AssertionError('Expected one restored ready endpoint')
        sends=self.upstream_count()-initial_sends
        if sends!=expected_sends:raise AssertionError(f'Unexpected provider replay: {sends} != {expected_sends}')
        receipt={'author':'Zeno Ren','scope':'Disposable Kind, synthetic provider, no production calls',
                 'case':rid,'expected':expected,'provider_send_delta':sends,'record':record}
        (self.output/(rid+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'case':rid,'phase':record['phase'],'provider_sends':sends}),flush=True)
    def ordinary(self,rid,mode=None,phase=None):
        self.fault({'release_id':rid,'mode':mode,'phase':phase} if mode else {})
        sends=self.upstream_count();old,new=self.submit(rid);record=self.wait(rid)
        expected='rolled_back' if mode in ('journal_failure','verification_failure') else 'succeeded'
        self.verify(rid,record,expected,old if expected=='rolled_back' else new,sends,0 if expected=='rolled_back' else 3)
        self.fault({})
    def crash_at(self,rid,phase):
        self.fault({'release_id':rid,'mode':'pause_phase','phase':phase})
        sends=self.upstream_count();old,new=self.submit(rid);self.wait(rid,phase)
        lease=self.api.get('lease',OPS,'lb-release-coordinator');identity=lease['spec']['holderIdentity'];uid=identity.split(':')[0]
        pod=next(p for p in self.api.list('pod',OPS,'app=lb-release-controller') if p['metadata']['uid']==uid)
        if not pod['spec']['nodeName'].startswith('lb-release-lab-'):raise AssertionError('Not a task-owned node')
        container_id=pod['status']['containerStatuses'][0]['containerID'].split('://',1)[1]
        # Kill only the coordinator's container inside a task-owned Kind node;
        # application writers are never force-deleted to fabricate drain proof.
        subprocess.run(['docker','exec',pod['spec']['nodeName'],'crictl','stop','--timeout','0',container_id],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            current=self.api.get('lease',OPS,'lb-release-coordinator')['spec']
            if current['holderIdentity']!=identity and current['leaseTransitions']>lease['spec']['leaseTransitions']:break
            time.sleep(.5)
        else:raise AssertionError('Coordinator takeover did not occur')
        self.fault({});record=self.wait(rid)
        self.verify(rid,record,'succeeded',new,sends,3)

    def long_stream(self,rid):
        self.fault({})
        # Allow the previous target lease to expire before starting the timed request.
        time.sleep(31)
        upstream=self.api.list('pod',NS,'app=lab-upstream')[0]
        def delay(seconds):
            script='import urllib.request,json; r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/control",data=json.dumps({"delay":'+str(seconds)+'}).encode(),headers={"Content-Type":"application/json"}));print(r.status)'
            self.api.exec(NS,upstream['metadata']['name'],'upstream',['python','-c',script])
        pod=self.api.list('pod',NS,'app=claude-lb')[0];result=[];failures=[];sends=self.upstream_count()
        script='''import urllib.request,json,yaml,pathlib
key=yaml.safe_load(pathlib.Path('/app/config.yaml').read_text())['auth']['api_key']
body={'model':'claude-opus-5','messages':[{'role':'user','content':'Reply LB_OK.'}],'max_tokens':32}
r=urllib.request.urlopen(urllib.request.Request('http://claude-lb.lb-lab.svc/v1/messages',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}),timeout=90)
print(json.dumps({'status':r.status,'marker':'LB_OK' in r.read().decode()}))
'''
        def request():
            try:result.append(json.loads(self.api.exec(NS,pod['metadata']['name'],'claude-lb',['python','-c',script],timeout=95)))
            except Exception as exc:failures.append(type(exc).__name__)
        try:
            delay(40);thread=threading.Thread(target=request);thread.start()
            deadline=time.monotonic()+10
            while self.upstream_count()==sends and time.monotonic()<deadline:time.sleep(.2)
            if self.upstream_count()!=sends+1:raise AssertionError('Long request did not begin')
            old,new=self.submit(rid,drain_seconds=10,forward_seconds=50,maintenance_seconds=200)
            record=self.wait(rid);thread.join(timeout=95)
            if failures or not result or result[0]!={'status':200,'marker':True}:raise AssertionError('Long request was interrupted')
            if self.api.list('pod',NS,'app=claude-lb')[0]['metadata']['uid']!=pod['metadata']['uid']:raise AssertionError('Old writer was replaced despite drain timeout')
            self.verify(rid,record,'rolled_back',old,sends,1)
        finally:delay(0)

    def manual_takeover(self,rid):
        self.fault({'release_id':rid,'mode':'pause_phase','phase':'draining'})
        sends=self.upstream_count();old,new=self.submit(rid);self.wait(rid,'draining')
        service=self.api.get('service',NS,'claude-lb');manual={'app':'claude-lb','manual-route':'hold'}
        self.api.patch('service',NS,service,[{'op':'replace','path':'/spec/selector','value':manual}])
        self.fault({});record=self.wait(rid)
        if record['phase']!='needs_attention' or self.api.get('service',NS,'claude-lb')['spec']['selector']!=manual:
            raise AssertionError('Manual route change was overwritten')
        (self.output/(rid+'-attention.json')).write_text(json.dumps(record,indent=2)+'\n')
        service=self.api.get('service',NS,'claude-lb')
        self.api.patch('service',NS,service,[{'op':'replace','path':'/spec/selector','value':{'app':'claude-lb','lb.zeno.ink/release-gate':rid}}])
        cli.change_request(self.api,OPS,rid,recover=True);record=self.wait(rid)
        self.verify(rid,record,'rolled_back',old,sends,0)

    def unavailable_rollback(self,rid):
        self.fault({'release_id':rid,'mode':'rollback_failure'})
        sends=self.upstream_count();old,new=self.submit(rid);record=self.wait(rid)
        if record['phase']!='needs_attention':raise AssertionError('Unavailable recovery was declared successful')
        if self.api.get('service',NS,'claude-lb')['spec']['selector'].get('lb.zeno.ink/release-gate')!=rid:
            raise AssertionError('Unverified backend was opened')
        (self.output/(rid+'-attention.json')).write_text(json.dumps(record,indent=2)+'\n')
        self.fault({});cli.change_request(self.api,OPS,rid,recover=True);record=self.wait(rid)
        self.verify(rid,record,'rolled_back',old,sends,0)

    def scheduling(self,rid):
        self.fault({'release_id':rid,'mode':'pause_phase','phase':'starting'})
        sends=self.upstream_count();old,new=self.submit(rid,drain_seconds=10,forward_seconds=50,maintenance_seconds=200)
        self.wait(rid,'starting');nodes=[n for n in self.api.list('node',None) if 'control-plane' not in n['metadata']['name']]
        try:
            for node in nodes:self.api.patch('node',None,node,[{'op':'add','path':'/spec/unschedulable','value':True}])
            self.fault({});self.wait(rid,'recovering',timeout=120)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                record=self.record(rid)
                if any(v.get('never_scheduled_after_deletion') for v in record['action_receipts'].get('writer_termination',{}).values()):break
                if record['phase'] in TERMINAL:raise AssertionError('Scheduling recovery ended before writer proof')
                time.sleep(.5)
            else:raise AssertionError('Never-scheduled pod was not safely retired')
        finally:
            for old_node in nodes:
                current=self.api.get('node',None,old_node['metadata']['name'])
                self.api.patch('node',None,current,[{'op':'add','path':'/spec/unschedulable','value':old_node['spec'].get('unschedulable',False)}])
            self.fault({})
        record=self.wait(rid);self.verify(rid,record,'rolled_back',old,sends,0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',required=True);parser.add_argument('--kubeconfig',required=True)
    parser.add_argument('--images',required=True);parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--case',required=True);parser.add_argument('--mode');parser.add_argument('--phase')
    args=parser.parse_args();runner=Scenarios(lab_api(args.context,args.kubeconfig),json.loads(Path(args.images).read_text()),args.directory,args.output)
    if args.mode=='crash':runner.crash_at(args.case,args.phase)
    elif args.mode=='long_stream':runner.long_stream(args.case)
    elif args.mode=='manual_takeover':runner.manual_takeover(args.case)
    elif args.mode=='unavailable_rollback':runner.unavailable_rollback(args.case)
    elif args.mode=='scheduling':runner.scheduling(args.case)
    else:runner.ordinary(args.case,args.mode,args.phase)
