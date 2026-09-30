"""Validated release plans and bounded receipts. Author: Zeno Ren."""
import hashlib
import copy
import json
import re
from urllib.parse import urlsplit

SCHEMA='lb-release-plan-v1'
CONTROL='lb.zeno.ink/release'
EPOCH='lb.zeno.ink/release-epoch'
RECOVER='lb.zeno.ink/recovery-only'
SPEC='lb.zeno.ink/controlled-spec-sha256'
FINALIZER='lb.zeno.ink/observe-termination'
GATE='lb.zeno.ink/release-gate'
DNS=re.compile(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?')
SHA=re.compile(r'[a-f0-9]{40}')
DIGEST=re.compile(r'[^\s@]+@sha256:[a-f0-9]{64}')
APP_FILES=('main.py','effort_compat.py','request_telemetry.py','safe_diagnostics.py','request_budget.py','admission.py',
           'gateway_lifecycle.py','upstream_body.py','usage_store.py','otel_setup.py','dashboard.html',
           'response_semantics.py','cleanup_observability.py','copilot_pricing.py','release_probe.py')


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def default_target_settings():
    return {'livenessProbe':{'httpGet':{'path':'/health/live','port':8000,'scheme':'HTTP'},'periodSeconds':30,'timeoutSeconds':2,'failureThreshold':3,'successThreshold':1},
            'readinessProbe':{'httpGet':{'path':'/health/accepting','port':8000,'scheme':'HTTP'},'periodSeconds':5,'timeoutSeconds':2,'failureThreshold':1,'successThreshold':1},
            'startupProbe':{'httpGet':{'path':'/health/live','port':8000,'scheme':'HTTP'},'periodSeconds':2,'timeoutSeconds':2,'failureThreshold':30,'successThreshold':1},
            'lifecycle':{'preStop':{'exec':{'command':['python','/app/gateway_lifecycle.py','--wait-seconds','45']}}},
            'terminationGracePeriodSeconds':90}


def target_pod_spec(current,plan):
    pod=copy.deepcopy(current['spec']['template']['spec'])
    settings=plan.get('target_settings',default_target_settings())
    pod['containers'][0]['image']=plan['image']
    for key in ('livenessProbe','readinessProbe','startupProbe','lifecycle'):pod['containers'][0][key]=copy.deepcopy(settings[key])
    pod['terminationGracePeriodSeconds']=settings['terminationGracePeriodSeconds']
    return pod


def validate_plan(plan):
    if plan.get('schema_version')!=SCHEMA:
        raise ValueError('Unsupported release plan schema')
    if plan.get('lab') and not plan.get('namespace','').startswith('lb-lab'):
        raise ValueError('Lab mode is restricted to explicitly isolated lb-lab namespaces')
    if 'target_settings' not in plan and not plan.get('lab'):
        raise ValueError('Target lifecycle/probes must be fixed in the reviewed plan')
    if plan.get('target_settings',default_target_settings())!=default_target_settings():
        raise ValueError('v1 supports only the reviewed LB lifecycle/probe contract')
    for key in ('release_id','namespace','deployment','container'):
        value=plan.get(key)
        if not isinstance(value,str) or not DNS.fullmatch(value) or len(value)>(40 if key=='release_id' else 63):
            raise ValueError('Invalid '+key)
    if not isinstance(plan.get('services'),list) or not plan['services'] or len(set(plan['services']))!=len(plan['services']):
        raise ValueError('All selected Services must be listed once')
    if not all(isinstance(x,str) and DNS.fullmatch(x) for x in plan['services']):
        raise ValueError('Invalid Service name')
    if not SHA.fullmatch(plan.get('source_revision','')) or not DIGEST.fullmatch(plan.get('image','')):
        raise ValueError('Fixed source SHA and immutable registry digest are required')
    hashes=plan.get('source_hashes',{})
    if set(hashes)!=set(APP_FILES) or not all(re.fullmatch(r'[a-f0-9]{64}',v) for v in hashes.values()):
        raise ValueError('Complete runtime file hashes are required')
    for key,default in [('maintenance_seconds',300),('forward_seconds',150),('drain_seconds',60)]:
        value=plan.setdefault(key,default)
        if type(value) is not int or value<=0:raise ValueError('Invalid time budget')
    if plan['maintenance_seconds']-plan['forward_seconds']<150 or plan['drain_seconds']>plan['forward_seconds']-30:
        raise ValueError('Preserve at least 150 seconds for recovery and 30 for switching')
    if plan.get('direct_pod_access') is not False:
        raise ValueError('Direct/bypassing callers must be ruled out before maintenance')
    if plan.get('storage_compatibility')!='additive-compatible':
        raise ValueError('Storage compatibility requires an explicit review; automatic DDL is unsupported')
    urls=plan.get('public_urls',[])
    if not urls:raise ValueError('Public baseline and acceptance URLs are required')
    for value in urls:
        url=urlsplit(value)
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ValueError('Invalid public probe URL')
        if url.scheme!='https' and not (plan.get('lab') is True and url.scheme=='http'):
            raise ValueError('Public probes require verified HTTPS')
    probes=plan.get('business_probes',[])
    if not 1<=len(probes)<=8:raise ValueError('Provide 1 to 8 bounded synthetic probes')
    for probe in probes:
        if set(probe)-{'api','model','stream','max_tokens'} or probe.get('api') not in ('messages','responses','chat'):
            raise ValueError('Only fixed synthetic API probes are supported')
        if not isinstance(probe.get('model'),str) or not 1<=len(probe['model'])<=128:
            raise ValueError('Probe model required')
        if type(probe.get('max_tokens')) is not int or not 1<=probe['max_tokens']<=512 or type(probe.get('stream')) is not bool:
            raise ValueError('Invalid probe budget/mode')
    if sum(p['max_tokens'] for p in probes)>2048:raise ValueError('Probe token budget exceeded')
    if plan.get('legacy_bootstrap') not in (True,False):raise ValueError('Explicit bootstrap policy required')
    if not re.fullmatch(r'[a-f0-9]{64}',plan.get('snapshot_sha256','')):
        raise ValueError('Private snapshot identity is required')
    return plan


def public_snapshot(snapshot):
    d=snapshot['deployment'];c=next(c for c in d['spec']['template']['spec']['containers'] if c['name']==snapshot['container'])
    return {'deployment_uid':d['metadata']['uid'],'generation':d['metadata'].get('generation'),
            'image':c['image'],'replicas':d['spec'].get('replicas',1),
            'services':{name:{'uid':s['metadata']['uid'],'selector':s['spec'].get('selector')} for name,s in snapshot['services'].items()}}
