"""Isolated exact-baked-main real HTTP/1.1 upstream AND downstream forensic cases.
No credentials, app lifespan, external network, source override, or production probes.
Print only case names, event names, counts/sizes/booleans/timings/outcomes.
"""
import asyncio, gzip, hashlib, importlib.metadata, json, logging, socket, time
from unittest.mock import AsyncMock
import httpx, uvicorn
from fastapi import FastAPI
import main

ROWS=[]
class Capture(logging.Handler):
    def emit(self,r):
        if getattr(r,'kind',None)=='copilot_stream_end':
            ROWS.append({k:getattr(r,k,None) for k in ['outcome','chunks','first_event','last_event','saw_completion','input_bytes','elapsed']})
logging.getLogger().handlers=[Capture()]
logging.getLogger().setLevel(logging.INFO)

def event(t='response.completed', nl=b'\n'):
    payload={'type':t,'response':{'id':'synthetic','usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}}}
    if t=='response.failed': payload['response']['error']={'code':'invalid_prompt','message':'synthetic'}
    if t=='response.incomplete': payload['response']['incomplete_details']={'reason':'max_output_tokens'}
    return b'data: '+json.dumps(payload,separators=(',',':')).encode()+nl+nl

def reference(data):
    # Independent complete-wire WHATWG framing oracle; no EOF dispatch.
    text=data.decode('utf-8-sig',errors='replace').replace('\r\n','\n').replace('\r','\n')
    results=[]
    for block in text.split('\n\n')[:-1]:
        values=[]; name='message'
        for line in block.split('\n'):
            if not line or line.startswith(':'): continue
            field,sep,value=line.partition(':')
            if value.startswith(' '): value=value[1:]
            if field=='event': name=value
            if field=='data': values.append(value)
        if not values: continue
        try: obj=json.loads('\n'.join(values))
        except Exception: obj=None
        results.append((name,obj))
    return results

async def run_case(name,wire,fragments='single',gzip_body=False,abrupt=False,provider='copilot',api_type='responses'):
    rows_start=len(ROWS); handlers=set(); requested=[]
    sent=gzip.compress(wire) if gzip_body else wire
    if fragments=='byte': parts=[sent[i:i+1] for i in range(len(sent))]
    elif isinstance(fragments,int): parts=[sent[i:i+fragments] for i in range(0,len(sent),fragments)]
    else: parts=[sent] if sent else []
    async def origin(reader,writer):
        task=asyncio.current_task(); handlers.add(task)
        try:
            raw=await reader.readuntil(b'\r\n\r\n')
            length=next((int(x.split(b':',1)[1]) for x in raw.split(b'\r\n') if x.lower().startswith(b'content-length:')),0)
            body=await reader.readexactly(length) if length else b''
            requested.append(len(body))
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n'+(b'Content-Encoding: gzip\r\n' if gzip_body else b'')+b'\r\n')
            for part in parts:
                writer.write(('%x\r\n'%len(part)).encode()+part+b'\r\n'); await writer.drain()
                if fragments!='single': await asyncio.sleep(.0005)
            if abrupt: writer.close()
            else:
                writer.write(b'0\r\n\r\n'); await writer.drain(); await reader.read()
        except (ConnectionError,asyncio.IncompleteReadError): pass
        finally:
            writer.close(); await writer.wait_closed(); handlers.discard(task)
    upstream=await asyncio.start_server(origin,'127.0.0.1',0)
    port=upstream.sockets[0].getsockname()[1]
    if provider=='copilot':
        ep=main.CopilotEndpoint(name='synthetic',github_token=''); cls=main.CopilotProxy
    elif provider=='azure':
        ep=main.AzureOpenAIEndpoint(name='synthetic',endpoint=f'http://127.0.0.1:{port}',api_key=''); cls=main.AzureOpenAIProxy
    else:
        ep=main.WorkspaceEndpoint(name='synthetic',api_base=f'http://127.0.0.1:{port}',token=''); cls=main.ClaudeProxy
    proxy=cls(main.LoadBalancer([ep]),'')
    proxy.load_balancer.on_request_end=AsyncMock(wraps=proxy.load_balancer.on_request_end)
    await proxy.client.aclose()
    proxy.client=httpx.AsyncClient(trust_env=False,timeout=httpx.Timeout(connect=2,read=3,write=2,pool=2),limits=httpx.Limits(max_connections=1,max_keepalive_connections=1))
    proxy._probe_upstream_connect=AsyncMock(return_value={'ok':True})
    app=FastAPI()
    @app.post('/responses')
    async def respond():
        await proxy.load_balancer.on_request_start(ep)
        if provider=='databricks':
            return await proxy._stream_request(ep,f'http://127.0.0.1:{port}/messages',{'input':'汉😀','stream':True},{},model='synthetic',start_time=time.time())
        return await proxy._stream_response(ep,f'http://127.0.0.1:{port}/responses',{'input':'汉😀','stream':True},{},'synthetic',api_type,time.time())
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); sock.listen(16); sock.setblocking(False)
    dport=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='critical',lifespan='off',access_log=False))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started: await asyncio.sleep(.001)
        async with httpx.AsyncClient(trust_env=False,timeout=8) as client:
            response=await client.post(f'http://127.0.0.1:{dport}/responses',json={})
            output=response.content
        await asyncio.sleep(.005)
        observed=ROWS[rows_start:]
        pool=proxy.client._transport._pool
        refs=reference(wire)
        outrefs=reference(output)
        counts=lambda rr:[o.get('type') for _,o in rr if isinstance(o,dict) and o.get('type')]
        result={'case':name,'input_stream_bytes':len(wire),'source_http_chunks':len(parts),'request_wire_bytes':requested,'status':response.status_code,'downstream_bytes':len(output),'prefix_byte_identical':output.startswith(wire),'oracle_events':counts(refs),'downstream_events':counts(outrefs),'added_truncation':b'upstream_truncated' in output,'added_network_error':b'upstream_network_error' in output,'terminal_logs':observed,'pool_requests_before_teardown':len(pool._requests),'business_active_before_teardown':ep.active_requests,'registry_before_teardown':len(getattr(proxy,'_stream_connections',{})),'endpoint_successes':ep.successful_requests,'endpoint_errors':ep.total_errors,'unexpected_external_probe_calls':proxy._probe_upstream_connect.await_count}
        result.update(provider=provider,api_type=api_type,exact_wire=output==wire,
                      recorded_input_tokens=ep.total_input_tokens,recorded_output_tokens=ep.total_output_tokens,
                      settlement_calls=proxy.load_balancer.on_request_end.await_count,
                      source_requests=len(requested),
                      downstream_error_events=sum(1 for n,o in outrefs if n=='error' or isinstance(o,dict) and (o.get('type') in ('error','response.failed') or 'error' in o)),
                      downstream_completed_events=sum(1 for _,o in outrefs if isinstance(o,dict) and o.get('type')=='response.completed' and isinstance(o.get('response'),dict) and isinstance(o['response'].get('id'),str)))
        return result
    finally:
        server.should_exit=True; await task; sock.close()
        await proxy.client.aclose(); upstream.close(); await upstream.wait_closed()
        await asyncio.gather(*list(handlers),return_exceptions=True)

async def work():
    completed=event(); delta=b'data: {"type":"response.output_text.delta","delta":"'+ '汉😀'.encode()+b'"}\n\n'
    cases=[
      ('lf_single',completed,'single',False,False),
      ('lf_every_byte_utf8',delta+completed,'byte',False,False),
      ('crlf_every_byte',event(nl=b'\r\n'),'byte',False,False),
      ('cr_single',event(nl=b'\r'),'single',False,False),
      ('lf_delta_crlf_terminal',delta+event(nl=b'\r\n'),7,False,False),
      ('multiline_json',b'event: response.completed\ndata: {"type":"response.completed",\ndata: "response":{"id":"synthetic"}}\n\n',3,False,False),
      ('comments_keepalive',b': heartbeat\n\n'+completed,3,False,False),
      ('trailing_eof_no_blank',completed[:-1],5,False,False),
      ('trailing_eof_no_newline',completed[:-2],5,False,False),
      ('empty_eof',b'','single',False,False),
      ('comment_only_eof',b': heartbeat\n\n','single',False,False),
      ('terminal_failed',event('response.failed'),4,False,False),
      ('terminal_incomplete',event('response.incomplete'),4,False,False),
      ('event_only_no_data',b'event: response.completed\n\n','single',False,False),
      ('named_terminal_invalid_json',b'event: response.completed\ndata: not-json\n\n','single',False,False),
      ('named_terminal_no_type',b'event: response.completed\ndata: {"id":"synthetic"}\n\n','single',False,False),
      ('event_message_payload_terminal',b'event: message\n'+completed,'single',False,False),
      ('leading_bom_single',b'\xef\xbb\xbf'+completed,'single',False,False),
      ('leading_bom_byte',b'\xef\xbb\xbf'+completed,'byte',False,False),
      ('non_json_error',b'event: error\ndata: not-json\n\n','single',False,False),
      ('json_error',b'event: error\ndata: {"type":"error","code":"server_error","message":"synthetic"}\n\n','single',False,False),
      ('responses_done_not_terminal',b'data: [DONE]\n\n','single',False,False),
      ('large_event_lf',delta+b'data: {"type":"response.output_text.delta","delta":"'+b'x'*150000+b'"}\n\n'+completed,'single',False,False),
      ('large_event_gzip',delta+b'data: {"type":"response.output_text.delta","delta":"'+b'x'*150000+b'"}\n\n'+completed,'single',True,False),
      ('completed_abrupt_http_close',completed,'single',False,True),
    ]
    print(json.dumps({'runtime':{'python':__import__('sys').version.split()[0],'main_sha256':hashlib.sha256(open('/app/main.py','rb').read()).hexdigest(),'packages':{x:importlib.metadata.version(x) for x in ['httpx','httpcore','anyio','starlette','uvicorn']}}}),flush=True)
    for args in cases:
        try:
            result=await asyncio.wait_for(run_case(*args),15); print(json.dumps(result),flush=True)
        except Exception as e: print(json.dumps({'case':args[0],'test_error_type':type(e).__name__}),flush=True); raise

if __name__=='__main__':
    import signal; signal.alarm(150); asyncio.run(work())
