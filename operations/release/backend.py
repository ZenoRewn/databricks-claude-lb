"""Kubernetes effects: conditional, bounded and recoverable by a new leader."""
import base64
import copy
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit

from .engine import OwnershipLost,PendingOperation,UnsafeState
from .model import APP_FILES,CONTROL,EPOCH,FINALIZER,GATE,RECOVER,SPEC,digest,target_pod_spec
from .snapshot import protected_spec,matches,verify_pod_image

def escape(value):return value.replace('~','~0').replace('/','~1')


class Backend:
    def __init__(self,api,record_store,plan,snapshot,leases,*,controller_namespace,controller_selector='app=lb-release-controller',lab=False):
        self.api,self.store,self.plan,self.snapshot,self.leases=api,record_store,plan,snapshot,leases
        self.namespace=plan['namespace'];self.deployment=plan['deployment'];self.release_id=plan['release_id']
        self.epoch=leases[-1].epoch;self.controller_namespace=controller_namespace
        self.controller_selector=controller_selector;self.lab=lab;self.quiet_since=None;self.last_started=None
        self.legacy=False
        self.store.guard=self.check_owner
    def check_owner(self):
        for lease in self.leases:lease.check()
    def deployment_object(self):return self.api.get('deployment',self.namespace,self.deployment)
    def _verified_old_image(self,pod=None):
        image=self.snapshot['deployment']['spec']['template']['spec']['containers'][0]['image']
        try:
            verify_pod_image(self.snapshot['old_pods'][0],image,self.plan['container'])
            if pod is not None:verify_pod_image(pod,image,self.plan['container'])
        except ValueError as exc:raise UnsafeState(str(exc)) from None
        return image
    def _annotations(self,obj):return obj['metadata'].get('annotations',{})
    def _claim(self,kind,obj):
        self.check_owner();annotations=dict(self._annotations(obj));owner=annotations.get(CONTROL)
        if owner not in (None,self.release_id):raise UnsafeState('object_owned_by_another_release')
        if int(annotations.get(EPOCH,'0'))>self.epoch:raise OwnershipLost('newer_execution_epoch')
        annotations.update({CONTROL:self.release_id,EPOCH:str(self.epoch)})
        if kind=='deployment':annotations.setdefault(SPEC,digest(obj['spec']))
        if annotations!=self._annotations(obj):
            obj=self.api.patch(kind,self.namespace,obj,[{'op':'add','path':'/metadata/annotations','value':annotations}])
        return obj
    def _owned(self,kind,name):
        self.check_owner();obj=self.api.get(kind,self.namespace,name)
        if self._annotations(obj).get(CONTROL)!=self.release_id:
            raise UnsafeState('release_ownership_removed_or_changed')
        return self._claim(kind,obj)
    def _patch(self,kind,obj,operations):
        self.check_owner()
        guards=[{'op':'test','path':'/metadata/annotations/'+escape(CONTROL),'value':self.release_id},
                {'op':'test','path':'/metadata/annotations/'+escape(EPOCH),'value':str(self.epoch)}]
        if kind=='deployment' and any(op['path'].startswith('/spec/') for op in operations):
            preview=self.api.request('PATCH',kind,self.namespace,obj['metadata']['name'],
                [{'op':'test','path':'/metadata/resourceVersion','value':obj['metadata']['resourceVersion']}]+guards+operations,
                query={'dryRun':'All'},patch=True)
            operations=operations+[{'op':'add','path':'/metadata/annotations/'+escape(SPEC),'value':digest(preview['spec'])}]
        return self.api.patch(kind,self.namespace,obj,guards+operations)
    def _pods(self):
        selector=self.snapshot['deployment']['spec']['selector']['matchLabels']
        pods=self.api.list('pod',self.namespace,','.join(f'{k}={v}' for k,v in selector.items()))
        for pod in pods:
            owners=pod['metadata'].get('ownerReferences',[])
            owner=next((x for x in owners if x.get('kind')=='ReplicaSet' and x.get('controller')),None)
            if owner is None:raise UnsafeState('foreign_pod_matches_application_selector')
            rs=self.api.get('replicaset',self.namespace,owner['name'])
            if rs['metadata']['uid']!=owner['uid'] or not any(x.get('kind')=='Deployment' and x.get('uid')==self.snapshot['deployment']['metadata']['uid'] for x in rs['metadata'].get('ownerReferences',[])):
                raise UnsafeState('pod_deployment_owner_mismatch')
        return pods
    def _new_pod(self):
        old={p['metadata']['uid'] for p in self.snapshot['old_pods']}
        pods=[p for p in self._pods() if p['metadata']['uid'] not in old and not p['metadata'].get('deletionTimestamp')]
        if len(pods)!=1:raise PendingOperation('new_pod_not_unique')
        p=pods[0]
        if p['spec']['containers'][0]['image']!=self.plan['image']:raise UnsafeState('unexpected_new_image')
        return p
    def _old_pod(self):
        saved=self.snapshot['old_pods'][0]
        p=self.api.optional('pod',self.namespace,saved['metadata']['name'])
        if p and p['metadata']['uid']!=saved['metadata']['uid']:raise UnsafeState('old_pod_identity_changed')
        return p
    def _probe(self,pod,action,*,extra=()):
        def invoke(operation,timeout):
            self.check_owner()
            current=self.api.get('pod',self.namespace,pod['metadata']['name'])
            if current['metadata']['uid']!=pod['metadata']['uid']:raise UnsafeState('probe_pod_changed')
            return self.api.exec(self.namespace,pod['metadata']['name'],self.plan['container'],
                                 ['python','/app/release_probe.py',operation,*extra],timeout=timeout)
        try:out=invoke(action,15 if action!='business' else 85)
        except RuntimeError as failure:
            if action!='business':raise
            try:out=invoke('receipt',10)
            except (OwnershipLost,UnsafeState):raise
            except Exception:raise failure from None
        try:return json.loads(out.strip().splitlines()[-1])
        except (ValueError,IndexError):raise RuntimeError('probe_result_invalid') from None
    def _legacy_diagnostics(self,pod):
        script="""import urllib.request,json
def get(p):
 try:r=urllib.request.urlopen('http://127.0.0.1:8000'+p,timeout=3)
 except urllib.error.HTTPError as e:r=e
 with r:return r.status,r.read(2097153).decode()
status,body=get('/health/accepting');_,metrics=get('/metrics')
values={}
for line in metrics.splitlines():
 if line and not line.startswith('#'):
  name,value=line.rsplit(' ',1)
  if name.startswith(('lb_requests_active','lb_requests_started_total','lb_admission_','lb_usage_')):values[name]=float(value)
print(json.dumps({'accepting_status':status,'accepting':json.loads(body),'metrics':values}))
"""
        self.check_owner()
        return json.loads(self.api.exec(self.namespace,pod['metadata']['name'],self.plan['container'],['python','-c',script]))
    def _diagnostics(self,pod):
        # This read-only diagnostic also works on 892e397, before the new probe exists.
        return self._legacy_diagnostics(pod)
    def _control(self,pod,action):
        self.check_owner()
        status=self.api.exec(self.namespace,pod['metadata']['name'],self.plan['container'],['python','/app/gateway_lifecycle.py','status'])
        state=json.loads(status)
        if action=='resume' and not state['paused'] and state['release_id']==self.release_id and state['epoch']==self.epoch:return
        if state['release_id']==self.release_id and state['epoch']==self.epoch and state['paused']==(action=='pause'):return
        self.check_owner()
        self.api.exec(self.namespace,pod['metadata']['name'],self.plan['container'],
                      ['python','/app/gateway_lifecycle.py',action,'--release-id',self.release_id,
                       '--epoch',str(self.epoch),'--expected-revision',str(state['revision'])])
        state=self._diagnostics(pod)['accepting']
        if bool(state.get('maintenance_paused'))!=(action=='pause'):raise PendingOperation('pause_state_not_observed')
    def _routes(self):
        restored=True;gated_all=True
        for name,saved in self.snapshot['services'].items():
            current=self.api.get('service',self.namespace,name)
            if current['metadata']['uid']!=saved['metadata']['uid']:raise UnsafeState('service_identity_changed')
            if {k:v for k,v in current['spec'].items() if k!='selector'}!={k:v for k,v in saved['spec'].items() if k!='selector'}:
                raise UnsafeState('protected_service_spec_drift')
            selector=current['spec'].get('selector')
            expected=saved['spec']['selector'];gated={**expected,GATE:self.release_id}
            if selector not in (expected,gated):raise UnsafeState('service_selector_drift')
            restored=restored and selector==expected
            gated_all=gated_all and selector==gated
            if self._annotations(current).get(CONTROL)==self.release_id and int(self._annotations(current).get(EPOCH,'0'))<self.epoch:
                self._claim('service',current)
        self.routes_gated=gated_all
        return restored
    def _guard_spec(self):
        current=self.deployment_object();saved=self.snapshot['deployment']
        if current['metadata']['uid']!=saved['metadata']['uid'] or protected_spec(current,self.plan['container'])!=protected_spec(saved,self.plan['container']):
            raise UnsafeState('protected_deployment_drift')
        expected=self._annotations(current).get(SPEC)
        if expected and expected!=digest(current['spec']):raise UnsafeState('controlled_spec_changed_outside_release')
        if current['spec']['template']['spec']['containers'][0]['image'] not in (self.plan['image'],saved['spec']['template']['spec']['containers'][0]['image']):
            raise UnsafeState('unexpected_image_drift')
        return current
    def observe(self,record):
        current=self._guard_spec();routes=self._routes();old=self._old_pod()
        if self._annotations(current).get(CONTROL)==self.release_id and int(self._annotations(current).get(EPOCH,'0'))<self.epoch:
            current=self._claim('deployment',current)
        recovery_only=self._annotations(current).get(RECOVER)==self.release_id
        if record['phase'] in ('pausing','draining','stopping','starting','verifying_backend') and not self.routes_gated and not recovery_only:
            raise UnsafeState('routing_changed_outside_release')
        old_running=bool(old and not old['metadata'].get('deletionTimestamp') and old['status'].get('phase')=='Running')
        new_running=any(p['metadata']['uid'] not in {x['metadata']['uid'] for x in self.snapshot['old_pods']}
                        and p['status'].get('phase')=='Running' for p in self._pods())
        if new_running and record['phase'] in ('preflight','gating','pausing','draining','stopping'):
            raise UnsafeState('unexpected_concurrent_writer')
        result={'routes_restored':routes,'old_running':old_running,'new_running':new_running,
                'old_stopped':all(p['metadata']['uid'] in record.get('action_receipts',{}).get('writer_termination',{}) for p in self.snapshot['old_pods']),
                'recovery_only':recovery_only,'quiescent':False}
        if old_running and record['phase'] in ('draining','stopping','recovering'):
            diagnostics=self._diagnostics(old);metrics=diagnostics['metrics'];state=diagnostics['accepting']
            self.legacy=state.get('release_control_version',0)<2
            if not self.legacy and state.get('maintenance_paused'):
                self._control(old,'pause')  # Fence commands from the previous execution epoch.
            required=('lb_admission_active','lb_admission_queued','lb_usage_pending_events','lb_usage_inflight_batch','lb_usage_backend_ready')
            if not all(k in metrics for k in required):raise UnsafeState('missing_drain_metrics')
            active=[v for k,v in metrics.items() if k.startswith('lb_requests_active{')]
            started=[v for k,v in metrics.items() if k.startswith('lb_requests_started_total{')]
            quiet=(len(active)==3 and len(started)==3 and sum(active)==0 and metrics['lb_admission_active']==0
                   and metrics['lb_admission_queued']==0 and metrics['lb_usage_pending_events']==0
                   and metrics['lb_usage_inflight_batch']==0 and metrics['lb_usage_backend_ready']==1)
            if not quiet or self.last_started!=sum(started):self.quiet_since=None
            self.last_started=sum(started)
            if quiet and self.quiet_since is None:self.quiet_since=time.monotonic()
            result['quiescent']=quiet and time.monotonic()-self.quiet_since>=4
            result['paused']=state.get('maintenance_paused',False)
            result['backend_healthy']=metrics['lb_usage_backend_ready']==1
        return result
    def _public_health(self):
        for base in self.plan['public_urls']:
            with urllib.request.urlopen(base.rstrip('/')+'/health/accepting',timeout=5) as response:
                if response.status!=200:raise RuntimeError('public_health_failed')
                body=json.loads(response.read(65537))
                if body.get('status')!='accepting' or body.get('draining'):raise RuntimeError('public_backend_not_accepting')
    def _verify_public_targets(self):
        allowed=set()
        for ingress in self.snapshot['ingresses']:
            for rule in ingress['spec'].get('rules',[]):
                if any(path.get('backend',{}).get('service',{}).get('name') in self.plan['services'] for path in rule.get('http',{}).get('paths',[])):
                    if rule.get('host'):allowed.add(rule['host'].lower())
        if self.lab:
            for name in self.plan['services']:
                allowed.update((name+'.'+self.namespace+'.svc',name+'.'+self.namespace+'.svc.cluster.local'))
        if any(urlsplit(url).hostname.lower() not in allowed for url in self.plan['public_urls']):
            raise UnsafeState('public_probe_host_not_bound_to_target_routing')
    def _verify_references(self):
        for ref in self.snapshot['references']:
            obj=self.api.get(ref['kind'],self.namespace,ref['name'])
            if obj['metadata']['uid']!=ref['uid'] or digest({k:obj.get(k) for k in ('data','binaryData','type','immutable')})!=ref['sha256']:
                raise UnsafeState('configuration_or_credentials_changed')
        current={x['metadata']['uid']:x['spec'] for x in self.api.list('ingress',self.namespace)}
        expected={x['metadata']['uid']:x['spec'] for x in self.snapshot['ingresses']}
        if current!=expected:raise UnsafeState('ingress_drift')
    def _image_preflight(self,image,suffix,hashes=None):
        name='lb-check-'+self.release_id+'-'+suffix
        pod=self.api.optional('pod',self.namespace,name)
        if pod is None:
            spec=copy.deepcopy(self.snapshot['deployment']['spec']['template']['spec'])
            spec.pop('nodeName',None)
            # Validation is read-only; it must never mount a usage writer's PVC.
            removed={v['name'] for v in spec.get('volumes',[]) if 'persistentVolumeClaim' in v}
            spec['volumes']=[v for v in spec.get('volumes',[]) if v['name'] not in removed]
            spec.pop('initContainers',None)
            c=spec['containers'][0];c['image']=image
            c['volumeMounts']=[v for v in c.get('volumeMounts',[]) if v['name'] not in removed]
            for key in ('livenessProbe','readinessProbe','startupProbe','lifecycle','args'):c.pop(key,None)
            c['command']=['python','/app/release_probe.py','image','--hashes',json.dumps(hashes)] if hashes else ['python','-c','import main; print("OLD_IMAGE_IMPORT_OK")']
            spec.update(restartPolicy='Never',activeDeadlineSeconds=90,automountServiceAccountToken=False)
            self.check_owner();self.api.create('pod',self.namespace,{'apiVersion':'v1','kind':'Pod','metadata':{'name':name,
                'labels':{'app':'lb-release-preflight'},'annotations':{CONTROL:self.release_id,EPOCH:str(self.epoch)}},'spec':spec})
            raise PendingOperation('image_preflight_created')
        if self._annotations(pod).get(CONTROL)!=self.release_id:raise UnsafeState('preflight_pod_owner_changed')
        phase=pod['status'].get('phase')
        if phase=='Failed':raise RuntimeError('image_preflight_failed')
        if phase!='Succeeded':raise PendingOperation('image_preflight_pending')
    def _stop_writer(self,record,pod,key):
        name=pod['metadata']['name'];current=self.api.optional('pod',self.namespace,name)
        receipts=record['action_receipts'].setdefault('writer_termination',{})
        if current is None:
            if pod['metadata']['uid'] not in receipts:raise UnsafeState('writer_disappeared_without_termination_evidence')
            return
        if current['metadata']['uid']!=pod['metadata']['uid']:raise UnsafeState('writer_identity_changed')
        current=self._claim('pod',current)
        if current['metadata']['uid'] not in record.setdefault('writer_stop_requested',[]):
            record['writer_stop_requested'].append(current['metadata']['uid'])
            self.store.save(record)
        finalizers=current['metadata'].get('finalizers',[])
        if FINALIZER not in finalizers:
            current=self._patch('pod',current,[{'op':'add','path':'/metadata/finalizers','value':finalizers+[FINALIZER]}])
        d=self._owned('deployment',self.deployment)
        if d['spec'].get('replicas',1)!=0:self._patch('deployment',d,[{'op':'replace','path':'/spec/replicas','value':0}])
        current=self.api.get('pod',self.namespace,name)
        statuses=current['status'].get('containerStatuses',[])
        terminated=[c for c in statuses if c.get('state',{}).get('terminated')]
        if current['metadata'].get('deletionTimestamp') and not current['spec'].get('nodeName') and not statuses:
            receipts[current['metadata']['uid']]={'never_scheduled_after_deletion':True,'containers':[]}
            self.store.save(record)
            current=self._owned('pod',name)
            self._patch('pod',current,[{'op':'replace','path':'/metadata/finalizers','value':[x for x in current['metadata'].get('finalizers',[]) if x!=FINALIZER]}])
            return
        if not current['metadata'].get('deletionTimestamp') or len(terminated)!=len(current['spec']['containers']):
            if not current['spec'].get('nodeName'):
                raise PendingOperation('unscheduled_writer_deletion_not_observed')
            node=self.api.get('node',None,current['spec']['nodeName'])
            if not any(c['type']=='Ready' and c['status']=='True' for c in node['status'].get('conditions',[])):
                raise UnsafeState('writer_node_unavailable_without_fencing')
            raise PendingOperation('writer_still_terminating')
        if any(c['state']['terminated']['exitCode'] not in (0,143) for c in terminated):raise UnsafeState('writer_unclean_exit')
        receipts[current['metadata']['uid']]={'node':current['spec']['nodeName'],'containers':[
            {'name':c['name'],'finishedAt':c['state']['terminated'].get('finishedAt'),'exitCode':c['state']['terminated']['exitCode']} for c in terminated]}
        self.store.save(record)  # Durable proof precedes removal of the observation finalizer.
        current=self._owned('pod',name)
        self._patch('pod',current,[{'op':'replace','path':'/metadata/finalizers','value':[x for x in current['metadata'].get('finalizers',[]) if x!=FINALIZER]}])
    def _set_image(self,image,*,rollback=False):
        d=self._owned('deployment',self.deployment);pod=copy.deepcopy(d['spec']['template']['spec']);c=pod['containers'][0]
        if rollback:
            saved=self.snapshot['deployment']['spec']['template']['spec'];original=saved['containers'][0]
            for key in ('image','livenessProbe','readinessProbe','startupProbe','lifecycle'):
                if key in original:c[key]=copy.deepcopy(original[key])
                else:c.pop(key,None)
            pod['terminationGracePeriodSeconds']=saved.get('terminationGracePeriodSeconds',30)
        else:
            pod=target_pod_spec(d,self.plan)
        if d['spec']['template']['spec']==pod and d['spec'].get('replicas')==1:return
        self._patch('deployment',d,[{'op':'replace','path':'/spec/template/spec','value':pod},
                                   {'op':'replace','path':'/spec/replicas','value':1}])
    def _restore_routes(self):
        self._routes()
        self._verify_references()
        for name,saved in self.snapshot['services'].items():
            service=self._owned('service',name);current=service['spec']['selector'];original=saved['spec']['selector']
            if current==original:continue
            if current!={**original,GATE:self.release_id}:raise UnsafeState('selector_changed_before_restore')
            self._patch('service',service,[{'op':'test','path':'/spec/selector','value':current},
                                           {'op':'replace','path':'/spec/selector','value':original}])
        for name in self.snapshot['services']:
            ready=[e for s in self.api.list('endpointslice',self.namespace,'kubernetes.io/service-name='+name)
                   for e in (s.get('endpoints') or []) if e.get('conditions',{}).get('ready') is True
                   and not e.get('conditions',{}).get('terminating')]
            if not ready:raise PendingOperation('restored_endpoints_pending')
        self._public_health()
    def perform(self,action,record):
        self.check_owner();self._guard_spec()
        if action=='preflight':
            if bool(self.plan.get('lab'))!=self.lab:raise UnsafeState('lab_profile_mismatch')
            if digest(self.snapshot)!=self.plan['snapshot_sha256']:raise UnsafeState('private_backup_hash_mismatch')
            self._verified_old_image()
            self._verify_references()
            self._verify_public_targets()
            d=self.deployment_object()
            preview=self.api.request('PATCH','deployment',self.namespace,self.deployment,
                [{'op':'test','path':'/metadata/uid','value':d['metadata']['uid']},
                 {'op':'test','path':'/metadata/resourceVersion','value':d['metadata']['resourceVersion']},
                 {'op':'replace','path':'/spec/template/spec','value':target_pod_spec(d,self.plan)}],query={'dryRun':'All'},patch=True)
            if protected_spec(preview,self.plan['container'])!=protected_spec(d,self.plan['container']):
                raise UnsafeState('server_dry_run_changed_protected_fields')
            if any(h.get('spec',{}).get('scaleTargetRef',{}).get('name')==self.deployment for h in self.api.list('hpa',self.namespace)):
                raise UnsafeState('hpa_controlled_workload_requires_separate_strategy')
            annotations=self.snapshot['deployment']['metadata'].get('annotations',{})
            if any(k.startswith(('argocd.argoproj.io/','kustomize.toolkit.fluxcd.io/','helm.toolkit.fluxcd.io/')) for k in annotations):
                raise UnsafeState('gitops_coordination_required')
            controllers=self.api.list('pod',self.controller_namespace,self.controller_selector)
            ready=[p for p in controllers if not p['metadata'].get('deletionTimestamp') and all(c.get('ready') for c in p['status'].get('containerStatuses',[])) and p['status'].get('containerStatuses')]
            if len({p['spec'].get('nodeName') for p in ready})<2:raise PendingOperation('two_coordinators_on_distinct_nodes_required')
            old=self._old_pod()
            if old is None:raise UnsafeState('old_pod_missing')
            self._verified_old_image(old)
            diagnostics=self._diagnostics(old);version=diagnostics['accepting'].get('release_control_version',0)
            if version<2 and not self.plan['legacy_bootstrap']:raise UnsafeState('bootstrap_not_allowed')
            if version<2:
                expected=json.loads(Path(__file__).with_name('legacy-892e397.json').read_text())['files']
                script='import hashlib,json,pathlib; print(json.dumps({n:hashlib.sha256(pathlib.Path("/app",n).read_bytes()).hexdigest() for n in '+repr(list(expected))+'}))'
                actual=json.loads(self.api.exec(self.namespace,old['metadata']['name'],self.plan['container'],['python','-c',script]))
                if actual!=expected:raise UnsafeState('legacy_runtime_not_certified_for_bootstrap')
            if diagnostics['accepting_status']!=200 or diagnostics['metrics'].get('lb_usage_backend_ready')!=1:
                raise RuntimeError('old_backend_not_healthy')
            required=('lb_admission_active','lb_admission_queued','lb_usage_pending_events','lb_usage_inflight_batch')
            if not all(k in diagnostics['metrics'] for k in required) or sum(k.startswith('lb_requests_active{') for k in diagnostics['metrics'])!=3 or sum(k.startswith('lb_requests_started_total{') for k in diagnostics['metrics'])!=3:
                raise UnsafeState('incomplete_drain_observability')
            self._public_health()
            discovered=[s['metadata']['name'] for s in self.api.list('service',self.namespace) if matches(s['spec'].get('selector'),old['metadata']['labels'])]
            if set(discovered)!=set(self.plan['services']):raise UnsafeState('entrypoint_inventory_changed')
            self._image_preflight(self.plan['image'],'new',self.plan['source_hashes'])
            self._image_preflight(self.snapshot['deployment']['spec']['template']['spec']['containers'][0]['image'],'old')
            self._claim('deployment',self.deployment_object())
            for name in self.plan['services']:self._claim('service',self.api.get('service',self.namespace,name))
            return {'verified':True,'runtime_control_version':version,'services':self.plan['services'],'server_dry_run_verified':True}
        if action=='gate':
            for name,saved in self.snapshot['services'].items():
                service=self._owned('service',name);selector=service['spec']['selector'];target={**saved['spec']['selector'],GATE:self.release_id}
                if selector==target:continue
                if selector!=saved['spec']['selector']:raise UnsafeState('selector_drift_before_gate')
                self._patch('service',service,[{'op':'test','path':'/spec/selector','value':selector},{'op':'replace','path':'/spec/selector','value':target}])
            for name in self.plan['services']:
                if any((s.get('endpoints') or []) for s in self.api.list('endpointslice',self.namespace,'kubernetes.io/service-name='+name)):
                    raise PendingOperation('route_removal_not_observed')
        elif action=='pause':
            old=self._old_pod()
            if old is None:raise UnsafeState('old_writer_missing')
            if self._diagnostics(old)['accepting'].get('release_control_version',0)>=2:self._control(old,'pause')
        elif action=='stop_old':
            if not record.get('writer_stop_requested') and not self.observe(record)['quiescent']:
                raise PendingOperation('quiescence_lost')
            self._stop_writer(record,self.snapshot['old_pods'][0],'old')
        elif action=='start_new':self._set_image(self.plan['image'])
        elif action=='verify_backend':
            pod=self._new_pod()
            if not all(c.get('ready') for c in pod['status'].get('containerStatuses',[])) or not pod['status'].get('containerStatuses'):
                raise PendingOperation('new_backend_not_ready')
            result=self._probe(pod,'image',extra=('--hashes',json.dumps(self.plan['source_hashes'])))
            diagnostics=self._diagnostics(pod)
            if not result.get('verified') or diagnostics['accepting_status']!=200 or diagnostics['metrics'].get('lb_usage_backend_ready')!=1:
                raise RuntimeError('new_backend_verification_failed')
            return {'verified':True,'pod':pod['metadata']['name'],'uid':pod['metadata']['uid'],'imageID':pod['status']['containerStatuses'][0]['imageID']}
        elif action=='restore_routes':self._restore_routes()
        elif action=='verify_business':
            pod=self._new_pod()
            result=self._probe(pod,'business',extra=('--plan',json.dumps({k:self.plan[k] for k in ('release_id','public_urls','business_probes')})))
            if result.get('pending'):raise PendingOperation('business_probe_still_running')
            if result.get('verified') is not True:result={**result,'verified':False,'failed':True}
            return result
        elif action=='resume_old':
            old=self._old_pod()
            if old is None or old['metadata'].get('deletionTimestamp'):raise UnsafeState('old_backend_cannot_resume')
            self._verified_old_image(old)
            diagnostic=self._diagnostics(old)
            if diagnostic['accepting'].get('permanent_draining'):raise UnsafeState('permanent_drain_cannot_resume')
            if diagnostic['accepting'].get('release_control_version',0)>=2:self._control(old,'resume')
            restored=self._diagnostics(old)
            if restored['accepting_status']!=200 or restored['metrics'].get('lb_usage_backend_ready')!=1:
                raise UnsafeState('old_backend_unhealthy')
            self._restore_routes()
        elif action=='rollback':return self._rollback(record)
        elif action=='cleanup':
            if not self._routes():raise UnsafeState('cleanup_requires_restored_routes')
            for kind,name in [('deployment',self.deployment)]+[('service',n) for n in self.plan['services']]:
                obj=self.api.get(kind,self.namespace,name);annotations=dict(self._annotations(obj))
                if CONTROL not in annotations:continue  # ACK lost after our cleanup.
                if annotations[CONTROL]!=self.release_id:raise UnsafeState('cleanup_owner_changed')
                obj=self._owned(kind,name);annotations=dict(self._annotations(obj))
                for key in (CONTROL,EPOCH,RECOVER,SPEC):annotations.pop(key,None)
                self._patch(kind,obj,[{'op':'replace','path':'/metadata/annotations','value':annotations}])
            for suffix in ('old','new'):
                pod=self.api.optional('pod',self.namespace,'lb-check-'+self.release_id+'-'+suffix)
                if pod and self._annotations(pod).get(CONTROL)==self.release_id:
                    if pod.get('status',{}).get('phase')=='Failed':
                        record.setdefault('retained_diagnostics',[]).append(pod['metadata']['name'])
                        continue
                    self.check_owner();self.api.delete('pod',self.namespace,pod)
        else:raise ValueError('Unsupported release action')
        return {'verified':True}
    def _rollback(self,record):
        # Rollback is itself journalled, and never deletes the usage ledger.
        old_image=self._verified_old_image()
        self.perform('gate',record)
        current=self.deployment_object()
        for old in self.snapshot['old_pods']:
            if old['metadata']['uid'] not in record['action_receipts'].get('writer_termination',{}):
                self._stop_writer(record,old,'old')
        if current['spec']['template']['spec']['containers'][0]['image']!=old_image:
            if 'rollback_writers' not in record:
                record['rollback_writers']=[{'metadata':{'name':p['metadata']['name'],'uid':p['metadata']['uid']}} for p in self._pods()]
                self.store.save(record)
            for saved in record['rollback_writers']:
                pod=self.api.optional('pod',self.namespace,saved['metadata']['name'])
                if pod and not pod['metadata'].get('deletionTimestamp') and (pod['spec'].get('nodeName') or pod['status'].get('containerStatuses')):
                    diagnostics=self._diagnostics(pod)
                    if diagnostics['accepting'].get('release_control_version',0)<2:raise UnsafeState('rollback_writer_cannot_pause')
                    self._control(pod,'pause');metrics=self._diagnostics(pod)['metrics']
                    if any(metrics.get(k)!=0 for k in ('lb_admission_active','lb_admission_queued','lb_usage_pending_events','lb_usage_inflight_batch')):
                        raise PendingOperation('rollback_waiting_for_writer')
                self._stop_writer(record,saved,'new')
            self._set_image(old_image,rollback=True)
            raise PendingOperation('old_image_restarting')
        if current['spec'].get('replicas')==0:
            self._set_image(old_image,rollback=True);raise PendingOperation('old_image_restarting')
        pods=[p for p in self._pods() if not p['metadata'].get('deletionTimestamp')]
        if len(pods)!=1 or not pods[0]['status'].get('containerStatuses') or not all(c.get('ready') for c in pods[0]['status'].get('containerStatuses',[])):
            raise PendingOperation('rollback_backend_pending')
        self._verified_old_image(pods[0])
        diagnostics=self._diagnostics(pods[0])
        if diagnostics['accepting_status']!=200 or diagnostics['metrics'].get('lb_usage_backend_ready')!=1:
            raise PendingOperation('rollback_backend_unhealthy')
        self._restore_routes();return {'verified':True,'image':old_image,'data_restored':False,
                                       'public_health_verified':True,'storage_backend_ready':True,
                                       'business_probes_replayed':False}
    def emergency_restore(self,record):
        self.check_owner();d=self._owned('deployment',self.deployment)
        annotations=dict(self._annotations(d));annotations[RECOVER]=self.release_id
        self._patch('deployment',d,[{'op':'replace','path':'/metadata/annotations','value':annotations}])
        if self._old_pod() is not None and not record.get('action_receipts',{}).get('stop_old'):
            self.perform('resume_old',record);return True
        return False
