"""Reproducible lab fixture; hard-refuses non-loopback/non-Kind contexts. Author: Zeno Ren."""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from urllib.parse import urlsplit

import yaml

from operations.release.kube import KubeAPI
from operations.release.manifests import objects

NS='lb-lab';OPS='lb-release-system'


def lab_api(context,kubeconfig):
    if not context.startswith('kind-lb-release-lab-'):raise ValueError('Only task-owned Kind contexts are permitted')
    api=KubeAPI(context=context,kubeconfig=kubeconfig)
    if urlsplit(api.client.configuration.host).hostname not in ('127.0.0.1','localhost'):
        raise ValueError('Refusing a non-loopback API server')
    return api


def wait_ready(api,name,timeout=180):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        d=api.get('deployment',NS,name)
        if d.get('status',{}).get('readyReplicas')==d['spec']['replicas']:return
        time.sleep(2)
    raise TimeoutError(name+' readiness')


def setup(api,images,directory):
    directory=Path(directory);directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    certs=directory/'certs';certs.mkdir(mode=0o700,exist_ok=True)
    commands=[['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout','ca.key','-out','ca.crt','-days','2','-subj','/CN=LB release lab CA'],
              ['openssl','req','-newkey','rsa:2048','-nodes','-keyout','server.key','-out','server.csr','-subj','/CN=mysql.lb-lab.svc'],
              ['openssl','x509','-req','-in','server.csr','-CA','ca.crt','-CAkey','ca.key','-CAcreateserial','-out','server.crt','-days','2','-extfile','server.ext']]
    (certs/'server.ext').write_text('subjectAltName=DNS:mysql.lb-lab.svc,DNS:mysql.lb-lab.svc.cluster.local\n')
    if not (certs/'server.crt').exists():
        for command in commands:subprocess.run(command,cwd=certs,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for p in certs.iterdir():os.chmod(p,0o600)
    if api.optional('configmap',NS,'lab-fixture'):
        raise ValueError('Existing lab fixture must not be overwritten')
    nodes=api.list('node',None)
    if len(nodes)!=3 or not all(n['metadata']['name'].startswith('lb-release-lab-') for n in nodes):
        raise ValueError('Unexpected lab node identity')
    # Namespace requests use kubectl only with the already validated explicit config.
    from kubernetes.client.exceptions import ApiException
    try:api.create('namespace',None,{'apiVersion':'v1','kind':'Namespace','metadata':{'name':NS,'labels':{'lb-release-lab':'true'}}})
    except ApiException as exc:
        if exc.status!=409:raise
    root_password=secrets.token_hex(16);password=secrets.token_hex(16);api_key='synthetic-'+secrets.token_hex(12)
    cfg={'auth':{'api_key':api_key},'load_balancer':{'strategy':'least_requests'},
         'endpoints':[{'name':'synthetic-db','api_base':'http://upstream.lb-lab.svc:8000','token':'synthetic','models':['databricks-claude-opus-5']}],
         'azure_openai':{'endpoints':[{'name':'synthetic-azure','endpoint':'http://upstream.lb-lab.svc:8000','api_key':'synthetic','deployments':['gpt-test']}]},
         'usage_storage':{'type':'mysql','host':'mysql.lb-lab.svc','port':3306,'user':'lb','password':password,'database':'lb_lab','pool_size':3,'retention_days':0}}
    def create(kind,obj):return api.create(kind,NS,obj)
    def metadata(name):return {'name':name,'namespace':NS}
    def service(name,selector,port,target_port,**spec):
        create('service',{'apiVersion':'v1','kind':'Service','metadata':metadata(name),
                         'spec':{'selector':selector,'ports':[{'port':port,'targetPort':target_port}],**spec}})
    create('secret',{'apiVersion':'v1','kind':'Secret','metadata':metadata('lab-config'),
        'stringData':{'config.yaml':yaml.safe_dump(cfg),'mysql-root':root_password,'mysql-user':password}})
    create('secret',{'apiVersion':'v1','kind':'Secret','metadata':metadata('lab-tls'),
        'data':{n:base64.b64encode((certs/n).read_bytes()).decode() for n in ('ca.crt','server.crt','server.key')}})
    create('configmap',{'apiVersion':'v1','kind':'ConfigMap','metadata':metadata('lab-fixture'),
        'data':{'upstream.py':Path('tests/release/upstream_fixture.py').read_text()}})
    service('mysql',{'app':'lab-mysql'},3306,3306,publishNotReadyAddresses=True)
    def env_secret(name,key):return {'name':name,'valueFrom':{'secretKeyRef':{'name':'lab-config','key':key}}}
    create('deployment',{'apiVersion':'apps/v1','kind':'Deployment','metadata':metadata('mysql'),
        'spec':{'replicas':1,'selector':{'matchLabels':{'app':'lab-mysql'}},'template':{'metadata':{'labels':{'app':'lab-mysql'}},'spec':{
            'containers':[{'name':'mysql','image':images['mysql'],'args':['--ssl-ca=/certs/ca.crt','--ssl-cert=/certs/server.crt','--ssl-key=/certs/server.key','--require-secure-transport=ON'],
                'env':[env_secret('MYSQL_ROOT_PASSWORD','mysql-root'),env_secret('MYSQL_PASSWORD','mysql-user'),{'name':'MYSQL_DATABASE','value':'lb_lab'},
                       {'name':'MYSQL_USER','value':'lb'},{'name':'MYSQL_ROOT_HOST','value':'%'}],
                'resources':{'requests':{'cpu':'100m','memory':'256Mi'},'limits':{'cpu':'1000m','memory':'1Gi'}},
                'volumeMounts':[{'name':'data','mountPath':'/var/lib/mysql'},{'name':'tls','mountPath':'/certs','readOnly':True}],
                'readinessProbe':{'exec':{'command':['sh','-c','MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysql -h mysql.lb-lab.svc --ssl-mode=VERIFY_IDENTITY --ssl-ca=/certs/ca.crt -uroot -Nse "SELECT 1"']},'timeoutSeconds':3,'periodSeconds':3}}],
            'volumes':[{'name':'data','emptyDir':{}},{'name':'tls','secret':{'secretName':'lab-tls'}}]}}}})
    wait_ready(api,'mysql')
    service('upstream',{'app':'lab-upstream'},8000,8000)
    create('deployment',{'apiVersion':'apps/v1','kind':'Deployment','metadata':metadata('upstream'),
        'spec':{'replicas':1,'selector':{'matchLabels':{'app':'lab-upstream'}},'template':{'metadata':{'labels':{'app':'lab-upstream'}},'spec':{
            'containers':[{'name':'upstream','image':images['new'],'command':['python','/fixture/upstream.py'],
                'resources':{'requests':{'cpu':'10m','memory':'32Mi'},'limits':{'cpu':'250m','memory':'128Mi'}},
                'volumeMounts':[{'name':'script','mountPath':'/fixture','readOnly':True}],
                'readinessProbe':{'httpGet':{'path':'/','port':8000},'periodSeconds':2}}],
            'volumes':[{'name':'script','configMap':{'name':'lab-fixture'}}]}}}})
    wait_ready(api,'upstream')
    create('deployment',{'apiVersion':'apps/v1','kind':'Deployment','metadata':metadata('claude-lb'),
        'spec':{'replicas':1,'strategy':{'type':'RollingUpdate','rollingUpdate':{'maxSurge':0,'maxUnavailable':1}},
            'selector':{'matchLabels':{'app':'claude-lb'}},'template':{'metadata':{'labels':{'app':'claude-lb'}},'spec':{
                'terminationGracePeriodSeconds':90,'containers':[{'name':'claude-lb','image':images['old'],
                    'env':[{'name':'SSL_CERT_FILE','value':'/certs/ca.crt'},{'name':'LOG_LEVEL','value':'WARNING'}],
                    'resources':{'requests':{'cpu':'30m','memory':'128Mi'},'limits':{'cpu':'500m','memory':'512Mi'}},
                    'volumeMounts':[{'name':'config','mountPath':'/app/config.yaml','subPath':'config.yaml','readOnly':True},
                                    {'name':'tls','mountPath':'/certs','readOnly':True}],
                    'readinessProbe':{'httpGet':{'path':'/health/accepting','port':8000},'periodSeconds':2,'timeoutSeconds':2},
                    'livenessProbe':{'httpGet':{'path':'/health/live','port':8000},'periodSeconds':10,'timeoutSeconds':2}}],
                'volumes':[{'name':'config','secret':{'secretName':'lab-config'}},{'name':'tls','secret':{'secretName':'lab-tls','items':[{'key':'ca.crt','path':'ca.crt'}]}}]}}}})
    service('claude-lb',{'app':'claude-lb'},80,8000);service('claude-lb-internal',{'app':'claude-lb'},80,8000)
    wait_ready(api,'claude-lb')
    state={'images':images,'namespace':NS,'controller_namespace':OPS,'created_at':time.time()}
    (directory/'lab-state.json').write_text(json.dumps(state,indent=2));os.chmod(directory/'lab-state.json',0o600)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',required=True);parser.add_argument('--kubeconfig',required=True)
    parser.add_argument('--images',required=True);parser.add_argument('--directory',required=True)
    args=parser.parse_args();api=lab_api(args.context,args.kubeconfig)
    result=setup(api,json.loads(Path(args.images).read_text()),args.directory)
    print(json.dumps({'lab_ready':True,'namespace':NS,'images':result['images']}))
