"""Read-only discovery; keep complete selectors and private rollback material."""
import copy

from .model import CONTROL,GATE,digest,public_snapshot


def matches(selector,labels):
    return bool(selector) and all(labels.get(k)==v for k,v in selector.items())


def capture(api,namespace,deployment,container):
    d=api.get('deployment',namespace,deployment)
    if d['spec']['selector'].get('matchExpressions'):
        raise ValueError('Expression selectors require an explicit adapter')
    if d['spec'].get('replicas',1)!=1:
        raise ValueError('v1 release mode requires one application replica')
    if d['metadata'].get('annotations',{}).get(CONTROL):
        raise ValueError('Existing release ownership must be resolved before planning')
    containers=d['spec']['template']['spec']['containers']
    if len(containers)!=1 or containers[0]['name']!=container:
        raise ValueError('v1 supports one explicitly named application container')
    labels=d['spec']['template']['metadata']['labels']
    selector=','.join(f'{k}={v}' for k,v in d['spec']['selector']['matchLabels'].items())
    pods=api.list('pod',namespace,selector)
    if any(p['metadata'].get('deletionTimestamp') and any(c.get('state',{}).get('running') for c in p['status'].get('containerStatuses',[])) for p in pods):
        raise ValueError('Another writer is still terminating')
    running=[p for p in pods if not p['metadata'].get('deletionTimestamp') and p['status'].get('phase')=='Running']
    if len(running)!=1:raise ValueError('A single healthy old pod is required')
    old=running[0]
    if len(old['spec']['containers'])!=1:
        raise ValueError('Injected sidecars require a separate writer/termination contract')
    if not all(c.get('ready') for c in old['status'].get('containerStatuses',[])):
        raise ValueError('Old pod is not ready')
    services={}
    for service in api.list('service',namespace):
        spec=service['spec'];name=service['metadata']['name']
        if matches(spec.get('selector'),old['metadata']['labels']):
            if not matches(spec['selector'],labels) or GATE in spec['selector']:
                raise ValueError('Version-pinned or gated Service cannot safely restore unchanged')
            services[name]=service
        elif not spec.get('selector'):
            slices=api.list('endpointslice',namespace,'kubernetes.io/service-name='+name)
            if any(e.get('targetRef',{}).get('uid')==old['metadata']['uid'] for s in slices for e in (s.get('endpoints') or [])):
                raise ValueError('Selectorless direct routing is unsupported')
    if not services:raise ValueError('No application Services discovered')
    references=set()
    pod_spec=d['spec']['template']['spec']
    for volume in pod_spec.get('volumes',[]):
        if 'configMap' in volume:references.add(('configmap',volume['configMap']['name']))
        if 'secret' in volume:references.add(('secret',volume['secret']['secretName']))
    for c in pod_spec.get('containers',[])+pod_spec.get('initContainers',[]):
        for env in c.get('env',[]):
            value=env.get('valueFrom',{})
            for key,kind in [('configMapKeyRef','configmap'),('secretKeyRef','secret')]:
                if key in value:references.add((kind,value[key]['name']))
        for env in c.get('envFrom',[]):
            for key,kind in [('configMapRef','configmap'),('secretRef','secret')]:
                if key in env:references.add((kind,env[key]['name']))
    fingerprints=[]
    for kind,name in sorted(references):
        obj=api.get(kind,namespace,name)
        fingerprints.append({'kind':kind,'name':name,'uid':obj['metadata']['uid'],
                             'resourceVersion':obj['metadata']['resourceVersion'],
                             'sha256':digest({k:obj.get(k) for k in ('data','binaryData','type','immutable')})})
    # Credentials are not copied; detect drift, never restore someone else's configuration.
    return {'deployment':d,'services':services,'container':container,'old_pods':[old],
            'references':fingerprints,'ingresses':api.list('ingress',namespace),
            'original_replicas':d['spec'].get('replicas',1)}


def protected_spec(deployment,container):
    spec=copy.deepcopy(deployment['spec'])
    spec.pop('replicas',None)
    pod=spec['template']['spec'];pod.pop('terminationGracePeriodSeconds',None)
    annotations=spec['template'].get('metadata',{}).get('annotations',{})
    annotations.pop('lb.zeno.ink/source-revision',None)
    if not annotations:spec['template'].get('metadata',{}).pop('annotations',None)
    for c in pod['containers']:
        if c['name']==container:
            for key in ('image','livenessProbe','readinessProbe','startupProbe','lifecycle'):c.pop(key,None)
    return spec
