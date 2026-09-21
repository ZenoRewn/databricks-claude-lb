"""Explicit Kubernetes contexts, conditional writes and expiring ownership."""
import copy
from datetime import datetime,timezone
import json
import threading
import time

from .engine import OwnershipLost, UncertainOperation, PendingOperation

RESOURCES={'deployment':('/apis/apps/v1','deployments'),'replicaset':('/apis/apps/v1','replicasets'),
           'namespace':('/api/v1','namespaces'),
           'hpa':('/apis/autoscaling/v2','horizontalpodautoscalers'),
           'pod':('/api/v1','pods'),'service':('/api/v1','services'),'configmap':('/api/v1','configmaps'),
           'secret':('/api/v1','secrets'),'node':('/api/v1','nodes'),'event':('/api/v1','events'),
           'endpointslice':('/apis/discovery.k8s.io/v1','endpointslices'),
           'ingress':('/apis/networking.k8s.io/v1','ingresses'),
           'lease':('/apis/coordination.k8s.io/v1','leases')}


def conditional_conflict(error,method,body,readback):
    status=getattr(error,'status',None)
    if status==409:return True
    if status!=422 or method!='PATCH' or not isinstance(body,list):return False
    expected={op['path'].rsplit('/',1)[-1]:op.get('value') for op in body if op.get('op')=='test'
              and op.get('path') in ('/metadata/uid','/metadata/resourceVersion')}
    if not expected:return False
    current=readback()
    return current is None or any(current['metadata'].get(key)!=value for key,value in expected.items())


class KubeAPI:
    def __init__(self,*,context=None,kubeconfig=None,in_cluster=False):
        from kubernetes import client,config
        configuration=client.Configuration()
        if in_cluster:config.load_incluster_config(client_configuration=configuration)
        else:
            if not context or not kubeconfig:raise ValueError('Explicit context and kubeconfig required')
            config.load_kube_config(config_file=kubeconfig,context=context,client_configuration=configuration,persist_config=False)
        self.client=client.ApiClient(configuration)
        self.core=client.CoreV1Api(self.client)
    def path(self,kind,namespace,name=None):
        prefix,plural=RESOURCES[kind]
        return prefix+(f'/namespaces/{namespace}' if namespace else '')+'/'+plural+('/'+name if name else '')
    def request(self,method,kind,namespace,name=None,body=None,query=None,patch=False):
        from kubernetes.client.exceptions import ApiException
        try:
            return self.client.call_api(self.path(kind,namespace,name),method,
                header_params={'Accept':'application/json','Content-Type':'application/json-patch+json' if patch else 'application/json'},
                query_params=list((query or {}).items()),body=body,response_type='object',
                auth_settings=['BearerToken'],_return_http_data_only=True,_request_timeout=(3,8))
        except ApiException as exc:
            if conditional_conflict(exc,method,body,lambda:self.optional(kind,namespace,name)):
                raise PendingOperation('conditional_write_requires_fresh_readback') from exc
            raise
        except Exception as exc:
            if method!='GET':raise UncertainOperation(type(exc).__name__) from exc
            raise
    def get(self,kind,namespace,name):return self.request('GET',kind,namespace,name)
    def optional(self,kind,namespace,name):
        from kubernetes.client.exceptions import ApiException
        try:return self.get(kind,namespace,name)
        except ApiException as exc:
            if exc.status==404:return None
            raise
    def list(self,kind,namespace,selector=None):
        return self.request('GET',kind,namespace,query={'labelSelector':selector} if selector else {})['items']
    def create(self,kind,namespace,body):return self.request('POST',kind,namespace,body=body)
    def replace(self,kind,namespace,body):return self.request('PUT',kind,namespace,body['metadata']['name'],body)
    def delete(self,kind,namespace,obj):
        return self.request('DELETE',kind,namespace,obj['metadata']['name'],body={'apiVersion':'v1','kind':'DeleteOptions',
            'preconditions':{'uid':obj['metadata']['uid'],'resourceVersion':obj['metadata']['resourceVersion']}})
    def patch(self,kind,namespace,obj,operations):
        metadata=obj['metadata']
        guards=[{'op':'test','path':'/metadata/uid','value':metadata['uid']},
                {'op':'test','path':'/metadata/resourceVersion','value':metadata['resourceVersion']}]
        return self.request('PATCH',kind,namespace,metadata['name'],guards+operations,patch=True)
    def exec(self,namespace,pod,container,command,timeout=10):
        from kubernetes.stream import stream
        # Never mutate the shared REST ApiClient used by Lease renewal: stream()
        # temporarily substitutes its request transport.
        from kubernetes import client
        core=client.CoreV1Api(client.ApiClient(copy.deepcopy(self.client.configuration)))
        socket=stream(core.connect_get_namespaced_pod_exec,pod,namespace,container=container,
                      command=command,stderr=True,stdin=False,stdout=True,tty=False,_preload_content=False)
        out='';err='';deadline=time.monotonic()+timeout
        try:
            while socket.is_open() and time.monotonic()<deadline:
                socket.update(timeout=1)
                if socket.peek_stdout():out+=socket.read_stdout()
                if socket.peek_stderr():err+=socket.read_stderr()
                if len(out)+len(err)>1024*1024:raise ValueError('Probe output too large')
            if socket.is_open():raise UncertainOperation('exec_timeout')
            if socket.returncode!=0:raise RuntimeError('bounded_probe_failed')
            return out
        finally:
            socket.close();core.api_client.close()


class Lease:
    def __init__(self,api,namespace,name,identity,*,duration=30):
        self.api,self.namespace,self.name,self.identity=api,namespace,name,identity
        self.duration=duration;self.epoch=0;self.lost=True;self.stop=threading.Event();self.thread=None
    @staticmethod
    def now():return datetime.now(timezone.utc)
    def acquire(self):
        obj=self.api.optional('lease',self.namespace,self.name);now=self.now();stamp=now.isoformat()
        if obj:
            spec=obj.get('spec',{});renew=spec.get('renewTime') or spec.get('acquireTime')
            expiry=datetime.fromisoformat(renew.replace('Z','+00:00')).timestamp()+spec.get('leaseDurationSeconds',self.duration) if renew else 0
            if spec.get('holderIdentity')!=self.identity and now.timestamp()<expiry:return False
            epoch=spec.get('leaseTransitions',0)+1
            obj['spec']={**spec,'holderIdentity':self.identity,'renewTime':stamp,'leaseDurationSeconds':self.duration,'leaseTransitions':epoch}
            self.api.replace('lease',self.namespace,obj)
        else:
            epoch=1
            self.api.create('lease',self.namespace,{'apiVersion':'coordination.k8s.io/v1','kind':'Lease',
                'metadata':{'name':self.name},'spec':{'holderIdentity':self.identity,'acquireTime':stamp,'renewTime':stamp,
                                                  'leaseDurationSeconds':self.duration,'leaseTransitions':epoch}})
        self.epoch=int(epoch);self.lost=False;return True
    def check(self):
        if self.lost:raise OwnershipLost('lease_lost')
        obj=self.api.get('lease',self.namespace,self.name);spec=obj['spec']
        renew=datetime.fromisoformat(spec['renewTime'].replace('Z','+00:00')).timestamp()
        if spec.get('holderIdentity')!=self.identity or spec.get('leaseTransitions')!=self.epoch or time.time()-renew>=self.duration-2:
            self.lost=True;raise OwnershipLost('lease_owner_changed')
    def start(self):
        def renew():
            while not self.stop.wait(3):
                try:
                    self.check();obj=self.api.get('lease',self.namespace,self.name)
                    if obj['spec'].get('holderIdentity')!=self.identity or obj['spec'].get('leaseTransitions')!=self.epoch:
                        raise OwnershipLost('lease_owner_changed')
                    obj['spec']['renewTime']=self.now().isoformat();self.api.replace('lease',self.namespace,obj)
                except Exception:self.lost=True;return
        self.thread=threading.Thread(target=renew,daemon=True);self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=4)


class RecordStore:
    def __init__(self,api,namespace,name):self.api,self.namespace,self.name=api,namespace,name;self.obj=None;self.guard=None
    def read(self):
        self.obj=self.api.get('configmap',self.namespace,self.name)
        return json.loads(self.obj['data']['record.json'])
    def save(self,record):
        if self.guard:self.guard()
        if self.obj is None:raise OSError('journal_not_loaded')
        try:
            self.obj=self.api.patch('configmap',self.namespace,self.obj,
                                   [{'op':'replace','path':'/data/record.json','value':json.dumps(record,sort_keys=True)}])
        except Exception as exc:raise OSError('journal_write_failed') from exc
