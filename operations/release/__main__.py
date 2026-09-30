"""Plan/submit/status/cancel/recover a fixed namespace-scoped release. Author: Zeno Ren."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from .engine import new_record,TERMINAL
from .kube import KubeAPI
from .model import APP_FILES,SCHEMA,encoded,digest,validate_plan,public_snapshot,default_target_settings,target_pod_spec,source_annotations,verify_target_template
from .snapshot import capture


def private_write(path,data):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def plan_release(api,profile,output,source_root):
    profile=dict(profile)
    snapshot=capture(api,profile['namespace'],profile['deployment'],profile['container'])
    if profile['image']==snapshot['deployment']['spec']['template']['spec']['containers'][0]['image']:
        raise ValueError('Target image is already deployed; no maintenance needed')
    discovered=sorted(snapshot['services'])
    if 'services' in profile and sorted(profile['services'])!=discovered:raise ValueError('Configured services differ from discovery')
    profile['services']=discovered;profile['schema_version']=SCHEMA;profile['snapshot_sha256']=digest(snapshot)
    profile.setdefault('target_settings',default_target_settings())
    if not profile.get('lab'):
        import hashlib
        hashes={}
        for filename in APP_FILES:
            raw=subprocess.check_output(['git','show',profile['source_revision']+':'+filename],cwd=source_root)
            hashes[filename]=hashlib.sha256(raw).hexdigest()
        if profile.get('source_hashes') and profile['source_hashes']!=hashes:raise ValueError('Git source does not match supplied manifest')
        profile['source_hashes']=hashes
    validate_plan(profile)
    deployment=snapshot['deployment']
    operations=[{'op':'test','path':'/metadata/uid','value':deployment['metadata']['uid']},
                {'op':'test','path':'/metadata/resourceVersion','value':deployment['metadata']['resourceVersion']},
                {'op':'replace','path':'/spec/template/spec','value':target_pod_spec(deployment,profile)},
                {'op':'add','path':'/spec/template/metadata/annotations','value':source_annotations(deployment,profile['source_revision'])}]
    preview=api.request('PATCH','deployment',profile['namespace'],profile['deployment'],operations,query={'dryRun':'All'},patch=True)
    verify_target_template(preview,target_pod_spec(deployment,profile),source_annotations(deployment,profile['source_revision']))
    from .snapshot import protected_spec
    if protected_spec(preview,profile['container'])!=protected_spec(deployment,profile['container']):
        raise ValueError('Server dry-run changed protected fields')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    private_write(output/'snapshot.json',encoded(snapshot))
    private_write(output/'plan.json',encoded(profile))
    private_write(output/'review.json',encoded({'author':'Zeno Ren','before':public_snapshot(snapshot),
        'target_image':profile['image'],'source_revision':profile['source_revision'],
        'target_source_annotation':profile['source_revision'],
        'target_settings':profile['target_settings'],'server_dry_run_verified':True,
        'services':discovered,'maintenance_seconds':profile['maintenance_seconds'],
        'legacy_bootstrap':profile['legacy_bootstrap'],'public_urls':profile['public_urls'],
        'maximum_probe_output_tokens':sum(p['max_tokens'] for p in profile['business_probes']),
        'production_changed':False}))
    return {'plan_directory':str(output.resolve()),'plan_sha256':digest(profile),'production_changed':False}


def submit(api,namespace,directory):
    directory=Path(directory);plan=validate_plan(json.loads((directory/'plan.json').read_text()))
    snapshot=json.loads((directory/'snapshot.json').read_text())
    if digest(snapshot)!=plan['snapshot_sha256']:raise ValueError('Private snapshot changed')
    rid=plan['release_id']
    immutable={'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'lb-plan-'+rid,'labels':{'app':'lb-release-plan'}},
               'immutable':True,'data':{'plan.json':encoded(plan).decode()}}
    existing=api.optional('configmap',namespace,immutable['metadata']['name'])
    if existing:
        if existing.get('data')!=immutable['data'] or not existing.get('immutable'):raise ValueError('Release ID already belongs to another plan')
    else:existing=api.create('configmap',namespace,immutable)
    backup={'apiVersion':'v1','kind':'Secret','metadata':{'name':'lb-backup-'+rid,'labels':{'app':'lb-release-backup'}},
            'immutable':True,'type':'Opaque','data':{'snapshot.json':base64.b64encode(encoded(snapshot)).decode()}}
    old=api.optional('secret',namespace,backup['metadata']['name'])
    if old:
        if old.get('data')!=backup['data'] or not old.get('immutable'):raise ValueError('Release backup identity mismatch')
    else:api.create('secret',namespace,backup)
    name='lb-release-'+rid
    previous=api.optional('configmap',namespace,name)
    if previous:
        if json.loads(previous['data']['record.json'])['plan']!=plan:raise ValueError('Existing record differs')
        return {'release_id':rid,'phase':json.loads(previous['data']['record.json'])['phase'],'already_submitted':True}
    api.create('configmap',namespace,{'apiVersion':'v1','kind':'ConfigMap',
        'metadata':{'name':name,'labels':{'app':'lb-release-record'},'annotations':{'plan-uid':existing['metadata']['uid']}},
        'data':{'record.json':encoded(new_record(plan)).decode()}})
    return {'release_id':rid,'phase':'preflight'}


def change_request(api,namespace,rid,recover=False):
    obj=api.get('configmap',namespace,'lb-release-'+rid);record=json.loads(obj['data']['record.json'])
    if recover:
        if record['phase']!='needs_attention':raise ValueError('Recover only an explicitly inspected needs_attention release')
        record.update(phase='recovering',final_result='not_completed',
                      recovery_requested_at=time.time(),recovery_deadline=time.time()+150)
    elif record['phase'] in TERMINAL:raise ValueError('Release is already terminal')
    else:record['cancel_requested']=True
    api.patch('configmap',namespace,obj,[{'op':'replace','path':'/data/record.json','value':encoded(record).decode()}])
    return {'release_id':rid,'requested':'recover' if recover else 'cancel'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',required=True);parser.add_argument('--kubeconfig',required=True)
    parser.add_argument('--controller-namespace',default='lb-release-system')
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('--profile',required=True);p.add_argument('--output',required=True);p.add_argument('--source-root',default='.')
    p=sub.add_parser('submit');p.add_argument('--directory',required=True)
    for name in ('status','cancel','recover'):
        p=sub.add_parser(name);p.add_argument('--release-id',required=True)
    args=parser.parse_args();api=KubeAPI(context=args.context,kubeconfig=args.kubeconfig)
    if args.action=='plan':result=plan_release(api,yaml.safe_load(Path(args.profile).read_text()),args.output,args.source_root)
    elif args.action=='submit':result=submit(api,args.controller_namespace,args.directory)
    elif args.action=='status':result=json.loads(api.get('configmap',args.controller_namespace,'lb-release-'+args.release_id)['data']['record.json'])
    else:result=change_request(api,args.controller_namespace,args.release_id,recover=args.action=='recover')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 2 if args.action=='status' and result['phase']=='needs_attention' else 0


if __name__=='__main__':raise SystemExit(main())
