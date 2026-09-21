"""Fixed, bounded in-pod release checks; never prints credentials. Author: Zeno Ren."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import tempfile
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


def _text_blocks(content,kind):
    if not isinstance(content,list):return []
    return [part['text'] for part in content if isinstance(part,dict)
            and part.get('type')==kind and isinstance(part.get('text'),str)]


def _answer_texts(payload,api):
    if not isinstance(payload,dict):return []
    if api=='messages':return _text_blocks(payload.get('content'),'text')
    if api=='responses':
        output=payload.get('output',[])
        if not isinstance(output,list):return []
        return [text for item in output if isinstance(item,dict) and item.get('type')=='message'
                and item.get('role','assistant')=='assistant'
                for text in _text_blocks(item.get('content'),'output_text')]
    choices=payload.get('choices',[])
    if not isinstance(choices,list):return []
    for choice in choices:
        if not isinstance(choice,dict) or choice.get('index',0)!=0:continue
        message=choice.get('message')
        if not isinstance(message,dict):continue
        content=message.get('content')
        return [content] if isinstance(content,str) else _text_blocks(content,'text')
    return []


def assess_probe_response(raw,api,streaming):
    """Return bounded diagnostics, not response bodies or reasoning/tool data."""
    if len(raw)>1024*1024:raise ValueError('probe_output_too_large')
    if not streaming:
        from response_semantics import decode_json_response
        payload=decode_json_response(httpx.Response(200,content=raw))
        completed=success_payload(payload,api);texts=_answer_texts(payload,api)
    else:
        import main
        observation=main._SSEObservation(api)
        framer=main._SSEFramer(main._RetainedStreamBudget(1024*1024),limit=1024*1024)
        parts={};snapshot=None;invalid=False;first=True

        def observe(frame):
            nonlocal snapshot,invalid,first
            text=frame.decode('utf-8-sig' if first else 'utf-8',errors='replace');first=False
            _,data,done=main._parse_sse_event_block(text,responses=api=='responses')
            if data is None:
                if not (done and api=='chat') and any(line.partition(':')[0]=='data' for line in text.splitlines()):invalid=True
            elif data.get('type') in ('error','response.failed') or isinstance(data.get('error'),dict):invalid=True
            ended=observation.terminal is not None
            observation.observe(frame)
            if ended or not data:return
            kind=data.get('type')
            if api=='messages':
                index=data.get('index',0)
                if kind=='message_start':
                    message=data.get('message',{})
                    if isinstance(message,dict):
                        for i,block in enumerate(message.get('content',[]) or []):
                            if isinstance(block,dict) and block.get('type')=='text' and isinstance(block.get('text'),str):parts[i]=block['text']
                elif kind=='content_block_start':
                    block=data.get('content_block',{})
                    if isinstance(block,dict) and block.get('type')=='text' and isinstance(block.get('text'),str):parts[index]=block['text']
                elif kind=='content_block_delta':
                    delta=data.get('delta',{})
                    if isinstance(delta,dict) and delta.get('type')=='text_delta' and isinstance(delta.get('text'),str):parts[index]=parts.get(index,'')+delta['text']
            elif api=='responses':
                index=(data.get('output_index',0),data.get('content_index',0))
                if kind=='response.output_text.delta' and isinstance(data.get('delta'),str):parts[index]=parts.get(index,'')+data['delta']
                elif kind=='response.output_text.done' and isinstance(data.get('text'),str):parts[index]=data['text']
                elif kind=='response.completed':
                    final=_answer_texts(data.get('response'),'responses')
                    if final:snapshot=final  # Full snapshots replace, never duplicate deltas.
            else:
                choices=data.get('choices',[])
                for choice in choices if isinstance(choices,list) else []:
                    if not isinstance(choice,dict) or choice.get('index',0)!=0:continue
                    delta=choice.get('delta',{})
                    if isinstance(delta,dict) and isinstance(delta.get('content'),str):parts[0]=parts.get(0,'')+delta['content']

        try:
            for frame in framer.feed(raw):observe(frame)
            for frame in framer.eof():observe(frame)
        finally:framer.close()
        completed=not invalid and observation.terminal is not None and observation.outcome=='completed'
        texts=snapshot if snapshot is not None else list(parts.values())
    return {'completed':completed,'marker_found':any('LB_OK' in text for text in texts),
            'text_chars':sum(map(len,texts)),
            'text_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()}


def _cache_path(plan):
    rid=plan.get('release_id','')
    if not isinstance(rid,str) or not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,39}',rid):raise ValueError('invalid_release_id')
    return Path('/tmp/lb-release-acceptance-'+rid+'.json')


def _save_receipt(cache,value):
    fd,temporary=tempfile.mkstemp(prefix=cache.name+'.',dir=cache.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,cache)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def read_receipt(plan):
    """Read-only recovery of a probe receipt; never performs model requests."""
    saved=json.loads(_cache_path(plan).read_text())
    if saved.get('plan_hash')!=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest():raise ValueError('probe_plan_changed')
    if saved.get('verified') or saved.get('failed'):return saved
    try:
        command=Path('/proc',str(saved.get('execution_pid')),'cmdline').read_bytes()
        if b'release_probe.py' in command and plan['release_id'].encode() in command and time.time()-saved['started_at']<80:
            return {**saved,'pending':True}
    except (OSError,ValueError,KeyError):pass
    return {**saved,'execution_uncertain':True,'error_code':'probe_execution_uncertain_requires_review'}


async def business_check(plan):
    cfg=configuration();auth=cfg.get('auth',{});key=auth.get('api_key')
    if not key:key=next(iter(auth.get('api_keys',{}).values()),'')
    key=expand(key)
    if not key:raise ValueError('missing_probe_credential')
    operation='release-'+plan['release_id'];request_ids=[];results=[]
    cache=_cache_path(plan)
    # A previous operation may have generated output before its exec ACK was
    # lost. Never automatically repeat probes whose completion is unknown.
    if cache.exists():
        saved=read_receipt(plan)
        if saved.get('verified'):return saved
        if saved.get('failed'):return saved
        if saved.get('pending'):return saved
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
                    wire=bytearray()
                    async for chunk in response.aiter_bytes():
                        wire.extend(chunk)
                        if len(wire)>1024*1024:raise ValueError('probe_output_too_large')
                    assessment=assess_probe_response(bytes(wire),api,True)
                else:
                    raw=await response.aread()
                    if len(raw)>1024*1024:raise ValueError('probe_output_too_large')
                    assessment=assess_probe_response(raw,api,False)
                results.append({'api':api,'stream':case['stream'],'model':case['model'],'request_id':request_id,**assessment})
                _save_receipt(cache,{**start,'checks':results})
                if not assessment['completed'] or not assessment['marker_found']:raise ValueError('public_protocol_acceptance_failed')
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
    _save_receipt(cache,result)
    return result


async def run(args):
    if args.action=='receipt':return read_receipt(json.loads(args.plan))
    if args.action=='image':
        async with asyncio.timeout(12):return await image_check(json.loads(args.hashes))
    async with asyncio.timeout(75):
        return await business_check(json.loads(args.plan))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('image','business','receipt'));parser.add_argument('--hashes');parser.add_argument('--plan')
    args=parser.parse_args()
    try:
        result=asyncio.run(run(args));print(json.dumps(result))
        if args.action=='business' and (result.get('failed') is True or not (
                result.get('verified') is True or result.get('pending') is True)):return 1
        return 0
    except Exception as exc:
        if args.action=='business':
            try:
                plan=json.loads(args.plan);cache=_cache_path(plan)
                saved=json.loads(cache.read_text())
                if saved.get('execution_pid')==os.getpid():
                    message=str(exc)
                    saved.update(failed=True,error_type=type(exc).__name__,
                                 error_code=message if re.fullmatch(r'[a-z0-9_]{1,80}',message) else None)
                    _save_receipt(cache,saved)
            except (ValueError,OSError,KeyError):pass
        message=str(exc)
        print(json.dumps({'verified':False,'error_type':type(exc).__name__,
                          'error_code':message if re.fullmatch(r'[a-z0-9_]{1,80}',message) else None}));return 1


if __name__=='__main__':raise SystemExit(main())
