"""Validated release plans and bounded receipts. Author: Zeno Ren."""
import hashlib
import copy
import json
import re
from urllib.parse import urlsplit

SCHEMA='lb-release-plan-v1'
ENV_SCHEMA='lb-release-plan-v2'
IDENTITY_MODE='COPILOT_RESPONSES_ITEM_ID_MODE'
IDENTITY_MODELS='COPILOT_RESPONSES_ITEM_ID_MODELS'
ENV_KEYS=(IDENTITY_MODE,IDENTITY_MODELS)
MODEL_NAME=re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}')
CONTROL='lb.zeno.ink/release'
EPOCH='lb.zeno.ink/release-epoch'
RECOVER='lb.zeno.ink/recovery-only'
SPEC='lb.zeno.ink/controlled-spec-sha256'
FINALIZER='lb.zeno.ink/observe-termination'
GATE='lb.zeno.ink/release-gate'
SOURCE='lb.zeno.ink/source-revision'
DNS=re.compile(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?')
SHA=re.compile(r'[a-f0-9]{40}')
DIGEST=re.compile(r'[^\s@]+@sha256:[a-f0-9]{64}')
APP_FILES=('main.py','build_metadata.py','chat_adapter.py','claude_model_identity.py','model_capabilities.py','effort_compat.py','request_telemetry.py','safe_diagnostics.py','request_timing.py','request_budget.py','admission.py',
           'gateway_lifecycle.py','upstream_body.py','usage_store.py','otel_setup.py','dashboard.html',
           'response_semantics.py','responses_item_identity.py','cleanup_observability.py','copilot_pricing.py','release_probe.py','key_policy.py','provider_credentials.py')


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def environment_overrides(plan):
    """Only reviewed, literal identity settings; never arbitrary env/commands."""
    values=plan.get('target_env',{})
    if not isinstance(values,dict) or set(values)-set(ENV_KEYS):
        raise ValueError('Only fixed Responses identity environment overrides are supported')
    if any(not isinstance(value,str) for value in values.values()):
        raise ValueError('Identity environment overrides must be literal strings')
    if IDENTITY_MODE in values and values[IDENTITY_MODE] not in ('off','observe','stabilize'):
        raise ValueError('Invalid Responses identity mode')
    if IDENTITY_MODELS in values:
        models=values[IDENTITY_MODELS].split(',') if values[IDENTITY_MODELS] else []
        if (len(values[IDENTITY_MODELS])>16384 or len(models)>128 or len(set(models))!=len(models)
                or any(not MODEL_NAME.fullmatch(model) for model in models)):
            raise ValueError('Identity models must be unique exact names in a bounded CSV')
    return values


def apply_environment_overrides(container,plan):
    """Mutate only a caller-owned copy, preserving all unrelated env entries."""
    overrides=environment_overrides(plan)
    if not overrides:return
    env=container.get('env')
    if env is None:env=[]
    if not isinstance(env,list) or any(not isinstance(entry,dict) for entry in env):
        raise ValueError('Invalid original container environment')
    seen=set();result=[]
    for entry in env:
        name=entry.get('name')
        if name in overrides:
            if name in seen:raise ValueError('Duplicate targeted environment variable')
            seen.add(name);result.append({'name':name,'value':overrides[name]})
        else:result.append(copy.deepcopy(entry))
    result.extend({'name':name,'value':overrides[name]} for name in ENV_KEYS if name in overrides and name not in seen)
    container['env']=result


def verify_pod_environment(pod,expected_container):
    containers=pod.get('spec',{}).get('containers',[])
    if len(containers)!=1 or containers[0].get('name')!=expected_container.get('name'):
        raise ValueError('environment_container_identity_changed')
    for key in ('env','envFrom'):
        if ((key in containers[0])!=(key in expected_container)
                or containers[0].get(key)!=expected_container.get(key)):
            raise ValueError('pod_environment_does_not_match_reviewed_image')


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
    apply_environment_overrides(pod['containers'][0],plan)
    for key in ('livenessProbe','readinessProbe','startupProbe','lifecycle'):pod['containers'][0][key]=copy.deepcopy(settings[key])
    pod['terminationGracePeriodSeconds']=settings['terminationGracePeriodSeconds']
    return pod


def source_annotations(current,revision):
    annotations=copy.deepcopy(current['spec']['template'].get('metadata',{}).get('annotations') or {})
    if revision is None:annotations.pop(SOURCE,None)
    else:annotations[SOURCE]=revision
    return annotations


def verify_target_template(deployment,pod_spec,annotations):
    if not isinstance(deployment,dict):raise ValueError('deployment_readback_missing')
    template=deployment.get('spec',{}).get('template',{})
    if (template.get('spec')!=pod_spec
            or (template.get('metadata',{}).get('annotations') or {})!=annotations):
        raise ValueError('deployment_template_readback_mismatch')


def validate_plan(plan):
    overrides=environment_overrides(plan)
    if plan.get('schema_version') not in (SCHEMA,ENV_SCHEMA):
        raise ValueError('Unsupported release plan schema')
    if overrides and plan.get('schema_version')!=ENV_SCHEMA:
        raise ValueError('Environment overrides require release plan v2')
    if plan.get('schema_version')==ENV_SCHEMA and not overrides:
        raise ValueError('Release plan v2 requires fixed environment overrides')
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
