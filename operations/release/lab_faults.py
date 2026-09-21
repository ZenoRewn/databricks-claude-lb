"""Opt-in fault injection against lb-lab namespaces only. Author: Zeno Ren.

Not installed or enabled by production manifests. Effects happen through the
real Kubernetes API; only the acknowledgement or selected client call fails.
"""
import json

from .engine import UncertainOperation


class LabFaults:
    def __init__(self,api,namespace):
        self.api,self.namespace=api,namespace
        self.original=api.request
        self.original_exec=api.exec
        api.request=self.request
        api.exec=self.exec

    def configuration(self):
        from kubernetes.client.exceptions import ApiException
        try:
            obj=self.original('GET','configmap',self.namespace,'lb-lab-fault')
            return obj,json.loads(obj.get('data',{}).get('fault.json','{}'))
        except ApiException as exc:
            if exc.status==404:return None,{}
            raise

    def consume(self,obj,fault):
        fault=dict(fault);fault['consumed']=True
        operations=[{'op':'test','path':'/metadata/resourceVersion','value':obj['metadata']['resourceVersion']},
                    {'op':'replace','path':'/data/fault.json','value':json.dumps(fault)}]
        self.original('PATCH','configmap',self.namespace,obj['metadata']['name'],operations,patch=True)

    def request(self,method,kind,namespace,name=None,body=None,query=None,patch=False):
        if method!='PATCH' or query and query.get('dryRun'):
            return self.original(method,kind,namespace,name,body,query,patch)
        obj,fault=self.configuration()
        if fault.get('consumed') or not fault.get('release_id'):
            return self.original(method,kind,namespace,name,body,query,patch)
        release_id=fault['release_id']
        if fault.get('mode')=='ack_loss_after_gate' and namespace.startswith('lb-lab') and kind=='service':
            closes=any(op.get('path')=='/spec/selector' and op.get('value',{}).get('lb.zeno.ink/release-gate')==release_id for op in body)
            if closes:
                result=self.original(method,kind,namespace,name,body,query,patch)
                self.consume(obj,fault)
                raise UncertainOperation('lab_ack_loss_after_real_service_write')
        if fault.get('mode')=='journal_failure' and namespace==self.namespace and kind=='configmap' and name=='lb-release-'+release_id:
            values=[op.get('value') for op in body if op.get('path')=='/data/record.json']
            if values and json.loads(values[-1]).get('phase')==fault.get('phase','draining'):
                self.consume(obj,fault)
                raise OSError('lab_journal_unavailable')
        return self.original(method,kind,namespace,name,body,query,patch)

    def exec(self,namespace,pod,container,command,timeout=10):
        result=self.original_exec(namespace,pod,container,command,timeout)
        if namespace.startswith('lb-lab') and command[:3]==['python','/app/release_probe.py','business']:
            obj,fault=self.configuration()
            if fault.get('mode')=='exec_ack_loss' and not fault.get('consumed'):
                plan=json.loads(command[command.index('--plan')+1])
                if plan['release_id']==fault.get('release_id'):
                    self.consume(obj,fault);raise UncertainOperation('lab_exec_ack_loss_after_business_checks')
        return result

    def pause_at(self,record,stop):
        import time
        while not stop.is_set():
            obj,fault=self.configuration()
            if fault.get('mode')!='pause_phase' or fault.get('release_id')!=record['plan']['release_id'] or fault.get('phase')!=record['phase']:
                return
            time.sleep(.2)

    def before_action(self,action,record):
        obj,fault=self.configuration()
        if fault.get('release_id')!=record['plan']['release_id'] or fault.get('consumed'):return
        if fault.get('mode')=='verification_failure' and action=='verify_backend':
            self.consume(obj,fault);raise RuntimeError('lab_backend_verification_failed')
        if fault.get('mode')=='rollback_failure':
            if action in ('verify_backend','rollback'):raise RuntimeError('lab_unavailable_rollback_target')
