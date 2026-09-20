"""Fixed, bounded in-pod release checks; never prints credentials. Author: Zeno Ren."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import time
import uuid

import httpx
import yaml


def configuration():
    value=yaml.safe_load(Path(os.getenv('CONFIG_PATH','/app/config.yaml')).read_text())
    if not isinstance(value,dict):raise ValueError('invalid_configuration')
    return value


def expand(value):
    return re.sub(r'\$\{([^}]+)\}',lambda m:os.environ.get(m[1],''),value)


async def image_check(hashes):
    if not hashes or any('/' in name or '..' in name for name in hashes):raise ValueError('invalid_hash_manifest')
    actual={name:hashlib.sha256(Path('/app',name).read_bytes()).hexdigest() for name in hashes}
    if actual!=hashes:raise ValueError('runtime_hash_mismatch')
    import main
    if 'Dashboard unavailable' in main.DASHBOARD_HTML or not main.DASHBOARD_HTML.rstrip().endswith('</html>'):
        raise ValueError('dashboard_incomplete')
    cfg=configuration();clients=[]
    try:
        db,az,cp,storage=main.load_config(os.getenv('CONFIG_PATH','/app/config.yaml'));clients=[db,az,cp]
        if not any(p and p.load_balancer.endpoints for p in clients):raise ValueError('no_configured_routes')
        if storage.get('type')=='mysql':
            await check_schema(storage)
        else:raise ValueError('release_event_acceptance_requires_mysql')
    finally:
        for client in clients:
            if client:await client.close()
    return {'verified':True,'source_hashes':actual,'configuration_valid':True,'schema_read_only':True}


async def mysql_connection(storage):
    import aiomysql
    return await aiomysql.connect(host=storage['host'],port=storage.get('port',3306),user=storage['user'],
        password=expand(storage['password']),db=storage['database'],ssl=ssl.create_default_context(),
        connect_timeout=5,autocommit=True,charset='utf8mb4')


async def check_schema(storage):
    conn=await mysql_connection(storage)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT TABLE_NAME,ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME IN ('usage_daily','usage_batch_ledger')")
            schema={'engines':dict(await cur.fetchall())}
            await cur.execute("SELECT COLUMN_NAME,DATA_TYPE,IS_NULLABLE,CHARACTER_MAXIMUM_LENGTH,COLLATION_NAME,DATETIME_PRECISION FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='usage_batch_ledger'")
            schema['ledger_columns']={row[0]:list(row[1:]) for row in await cur.fetchall()}
            await cur.execute("SELECT COLUMN_NAME,DATA_TYPE,IS_NULLABLE,CHARACTER_MAXIMUM_LENGTH FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='usage_daily'")
            schema['daily_columns']={row[0]:list(row[1:]) for row in await cur.fetchall()}
            await cur.execute("SELECT TABLE_NAME,INDEX_NAME,COLUMN_NAME FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME IN ('usage_daily','usage_batch_ledger') AND NON_UNIQUE=0 ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX")
            indexes={}
            for table,index,column in await cur.fetchall():indexes.setdefault(table,{}).setdefault(index,[]).append(column)
            schema['unique_indexes']=indexes
            await cur.execute("SELECT COLUMN_NAME FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='usage_batch_ledger' AND INDEX_NAME='usage_batch_event_date' ORDER BY SEQ_IN_INDEX")
            schema['event_date_index']=[row[0] for row in await cur.fetchall()]
            validate_schema(schema)
    finally:conn.close()


def validate_schema(schema):
    if schema['engines']!={'usage_daily':'InnoDB','usage_batch_ledger':'InnoDB'}:raise ValueError('usage_schema_engine_mismatch')
    expected={'batch_id':['char','NO',36,'ascii_bin',None], 'event_date':['date','NO',None,None,None],
              'payload_sha256':['char','NO',64,'ascii_bin',None], 'payload':['json','YES',None,None,None],
              'created_at':['timestamp','NO',None,None,6]}
    if any(schema['ledger_columns'].get(name)!=value for name,value in expected.items()):raise ValueError('usage_schema_columns_mismatch')
    expected={**{name:['bigint','NO',None] for name in ('input_tokens','output_tokens','cache_creation_tokens','cache_read_tokens')},
              **{name:['int','NO',None] for name in ('requests','errors')},'date':['date','NO',None],'model':['varchar','NO',128]}
    if any(schema['daily_columns'].get(name)!=value for name,value in expected.items()):raise ValueError('daily_schema_columns_mismatch')
    if schema['unique_indexes']!={'usage_batch_ledger':{'PRIMARY':['batch_id']},'usage_daily':{'PRIMARY':['date','model']}}:
        raise ValueError('usage_schema_unique_keys_mismatch')
    if schema['event_date_index']!=['event_date']:raise ValueError('usage_schema_retention_index_missing')


def success_payload(payload,api):
    from response_semantics import assess_json
    return assess_json(payload,api).outcome=='completed'


async def business_check(plan):
    cfg=configuration();auth=cfg.get('auth',{});key=auth.get('api_key')
    if not key:key=next(iter(auth.get('api_keys',{}).values()),'')
    key=expand(key)
    if not key:raise ValueError('missing_probe_credential')
    operation='release-'+plan['release_id'];request_ids=[];results=[]
    cache=Path('/tmp/lb-release-acceptance-'+plan['release_id']+'.json')
    # A previous operation may have generated output before its exec ACK was
    # lost. Never automatically repeat probes whose completion is unknown.
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved.get('plan_hash')!=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest():raise ValueError('probe_plan_changed')
        if saved.get('verified'):return saved
        if saved.get('failed'):return saved
        pid=saved.get('execution_pid')
        try:
            command=Path('/proc',str(pid),'cmdline').read_bytes()
            if b'release_probe.py' in command and plan['release_id'].encode() in command and time.time()-saved['started_at']<80:
                return {**saved,'pending':True}
        except (OSError,ValueError):pass
        raise ValueError('probe_execution_uncertain_requires_review')
    start={'plan_hash':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),'verified':False,'started_at':time.time(),'execution_pid':os.getpid()}
    with cache.open('x') as f:
        os.chmod(cache,0o600);json.dump(start,f);f.flush();os.fsync(f.fileno())
    routes={'messages':'/v1/messages','responses':'/v1/responses','chat':'/v1/chat/completions'}
    async with httpx.AsyncClient(timeout=httpx.Timeout(20,connect=5),follow_redirects=False) as client:
        for index,case in enumerate(plan['business_probes']):
            api=case['api'];body={'model':case['model'],'stream':case['stream']}
            if api=='responses':body.update(input='Reply with exactly LB_OK.',max_output_tokens=case['max_tokens'])
            else:body.update(messages=[{'role':'user','content':'Reply with exactly LB_OK.'}],max_tokens=case['max_tokens'])
            base=plan['public_urls'][index%len(plan['public_urls'])].rstrip('/')
            headers={'Authorization':'Bearer '+key,'anthropic-version':'2023-06-01','X-LB-Operation-Id':operation}
            async with client.stream('POST',base+routes[api],json=body,headers=headers) as response:
                if response.status_code!=200:raise ValueError('public_inference_failed')
                request_id=response.headers.get('x-lb-request-id')
                if not request_id:raise ValueError('request_identity_missing')
                request_ids.append(request_id)
                if case['stream']:
                    import main
                    observation=main._SSEObservation(api);wire=bytearray()
                    async for chunk in response.aiter_bytes():
                        wire.extend(chunk)
                        if len(wire)>1024*1024:raise ValueError('probe_output_too_large')
                    for block in bytes(wire).replace(b'\r\n',b'\n').split(b'\n\n'):
                        if block:observation.observe(block+b'\n\n')
                    completed=observation.terminal is not None and observation.outcome=='completed'
                    text=bytes(wire).decode(errors='replace')
                else:
                    raw=await response.aread()
                    if len(raw)>1024*1024:raise ValueError('probe_output_too_large')
                    payload=json.loads(raw);completed=success_payload(payload,api);text=raw.decode(errors='replace')
                if not completed or 'LB_OK' not in text:raise ValueError('public_protocol_acceptance_failed')
                results.append({'api':api,'stream':case['stream'],'model':case['model'],'request_id':request_id,'completed':True})
    storage=cfg.get('usage_storage',{'type':'json','path':cfg.get('usage_data_dir','/app/usage_data')})
    if storage.get('type')!='mysql':raise ValueError('event_level_acceptance_requires_mysql')
    # The application flush cadence is 30s. This probe is bounded separately
    # from inference requests; its caller retains Lease renewal throughout.
    deadline=time.monotonic()+40;found={}
    while time.monotonic()<deadline:
        conn=await mysql_connection(storage)
        try:
            async with conn.cursor() as cur:
                await cur.execute('SELECT batch_id,payload_sha256,payload FROM usage_batch_ledger WHERE created_at >= FROM_UNIXTIME(%s)',(start['started_at']-2,))
                rows=await cur.fetchall()
                found={}
                for batch_id,payload_hash,payload in rows:
                    if not payload:continue
                    batch=json.loads(payload);encoded=json.dumps(batch,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                    if hashlib.sha256(encoded).hexdigest()!=payload_hash:raise ValueError('ledger_hash_mismatch')
                    for event in batch.get('events',[]):
                        if event.get('request_id') in request_ids:found.setdefault(event['request_id'],[]).append(event)
        finally:conn.close()
        if all(len(found.get(r,[]))==1 for r in request_ids):break
        if any(len(found.get(r,[]))>1 for r in request_ids):raise ValueError('duplicate_probe_usage')
        await asyncio.sleep(1)
    if not all(len(found.get(r,[]))==1 and found[r][0].get('generation_outcome')=='completed' for r in request_ids):raise ValueError('probe_usage_missing')
    result={**start,'verified':True,'checks':results,'persistence_verified':True,'usage_events':len(request_ids)}
    with cache.open('w') as f:json.dump(result,f);f.flush();os.fsync(f.fileno())
    return result


async def run(args):
    if args.action=='image':
        async with asyncio.timeout(12):return await image_check(json.loads(args.hashes))
    async with asyncio.timeout(75):
        return await business_check(json.loads(args.plan))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('image','business'));parser.add_argument('--hashes');parser.add_argument('--plan')
    args=parser.parse_args()
    try:
        result=asyncio.run(run(args));print(json.dumps(result));return 0
    except Exception as exc:
        if args.action=='business':
            try:
                plan=json.loads(args.plan);cache=Path('/tmp/lb-release-acceptance-'+plan['release_id']+'.json')
                saved=json.loads(cache.read_text())
                if saved.get('execution_pid')==os.getpid():
                    saved.update(failed=True,error_type=type(exc).__name__)
                    with cache.open('w') as f:json.dump(saved,f);f.flush();os.fsync(f.fileno())
            except (ValueError,OSError,KeyError):pass
        message=str(exc)
        print(json.dumps({'verified':False,'error_type':type(exc).__name__,
                          'error_code':message if re.fullmatch(r'[a-z0-9_]{1,80}',message) else None}));return 1


if __name__=='__main__':raise SystemExit(main())
