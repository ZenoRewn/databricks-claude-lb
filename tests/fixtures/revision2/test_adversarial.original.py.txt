"""Independent baked-code oracles, loopback-only synthetic HTTP, no auth/source overlay."""
import asyncio,contextlib,gzip,hashlib,json,logging,time,unittest
from unittest.mock import AsyncMock,patch
import httpx,main
from protocol_socket_all import run_case
logging.disable(logging.CRITICAL)
MODES=[('copilot','responses'),('azure','responses'),('copilot','chat'),('azure','chat'),('databricks','messages')]
class Independent(unittest.IsolatedAsyncioTestCase):
 async def test_buffered_401_refresh_cancel_releases_half_open(self):
  for state in ('closed','half_open'):
   with self.subTest(state=state): await self.refresh_case(state,True)
 async def test_buffered_half_open_401_repair_can_complete(self):
  await self.refresh_case('half_open',False)
 async def refresh_case(self,state,cancel):
  calls=[];handlers=set();refresh=asyncio.Event();release=asyncio.Event()
  async def origin(r,w):
   handlers.add(asyncio.current_task())
   try:
    h=await r.readuntil(b'\r\n\r\n'); n=next((int(x.split(b':',1)[1]) for x in h.split(b'\r\n') if x.lower().startswith(b'content-length:')),0)
    if n: await r.readexactly(n)
    self.assertFalse(any(x.lower().startswith(b'authorization:') for x in h.split(b'\r\n')))
    calls.append(1);status=b'401 Unauthorized' if len(calls)==1 else b'200 OK';body=b'{"usage":{}}'
    w.write(b'HTTP/1.1 '+status+b'\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body);await w.drain()
   finally:w.close();await w.wait_closed();handlers.discard(asyncio.current_task())
  server=await asyncio.start_server(origin,'127.0.0.1',0); url='http://127.0.0.1:'+str(server.sockets[0].getsockname()[1])
  ep=main.CopilotEndpoint('synthetic','',models=['gpt-5.6-sol'],api_types=['responses']);ep.session_base_url=url
  lb=main.LoadBalancer([ep],circuit_breaker_timeout=60);p=main.CopilotProxy(lb,'');await p.client.aclose();p.client=httpx.AsyncClient(trust_env=False)
  p._build_headers=AsyncMock(return_value={})
  async def local_refresh(*args,**kw):
   refresh.set()
   if cancel: await release.wait()
   return 'synthetic-unused'
  p.get_session_token=local_refresh
  if state=='half_open':lb._open(ep);ep.circuit_retry_at=0
  task=asyncio.create_task(p.proxy_responses({'model':'gpt-5.6-sol'},stream=False))
  try:
   await asyncio.wait_for(refresh.wait(),2)
   if cancel:
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,2)
    print('REFRESH_CANCEL_STATE',json.dumps(dict(state=state,active=ep.active_requests,half_open=ep.half_open_in_flight,attempts=len(lb._attempts.get(id(ep),{})),cancelled=ep.cancelled_requests,errors=ep.total_errors,calls=len(calls),pool=len(p.client._transport._pool._requests))),flush=True)
    self.assertEqual(ep.active_requests,0,'cancel during buffered auth repair must settle original admission')
    self.assertEqual(ep.cancelled_requests,1);self.assertFalse(ep.half_open_in_flight);self.assertEqual(ep.total_errors,0)
   else:
    try:result=await asyncio.wait_for(task,2);status=result.status_code
    except main.HTTPException as e:status=e.status_code
    print('HALF_OPEN_REPAIR',json.dumps(dict(status=status,calls=len(calls),active=ep.active_requests,half_open=ep.half_open_in_flight)),flush=True)
    self.assertEqual(status,200,'successful one-time auth repair must retry the held half-open attempt')
    self.assertEqual(len(calls),2)
  finally:
   release.set()
   if not task.done():task.cancel()
   with contextlib.suppress(BaseException):await task
   await p.client.aclose();server.close();await server.wait_closed();await asyncio.gather(*list(handlers),return_exceptions=True)
 async def test_normal_asgi_gzip_request_cancellation_all_modes(self):
  for provider,api in MODES:
   with self.subTest(provider=provider,api=api):await self.cancel_case(provider,api)
 async def cancel_case(self,provider,api):
  handlers=set();sent=asyncio.Event();requests=[];peerclosed=asyncio.Event()
  async def origin(r,w):
   handlers.add(asyncio.current_task())
   try:
    h=await r.readuntil(b'\r\n\r\n');n=next((int(x.split(b':',1)[1]) for x in h.split(b'\r\n') if x.lower().startswith(b'content-length:')),0)
    if n:await r.readexactly(n)
    requests.append(1)
    import zlib
    z=zlib.compressobj(wbits=31);b=z.compress(b': ready\n\ndata: {"pending":"')+z.flush(zlib.Z_SYNC_FLUSH)
    w.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Encoding: gzip\r\nTransfer-Encoding: chunked\r\n\r\n'+f'{len(b):x}\r\n'.encode()+b+b'\r\n');await w.drain();await r.read()
   finally:w.close();await w.wait_closed();handlers.discard(asyncio.current_task());peerclosed.set()
  server=await asyncio.start_server(origin,'127.0.0.1',0);url='http://127.0.0.1:'+str(server.sockets[0].getsockname()[1])
  if provider=='copilot': ep=main.CopilotEndpoint('synthetic','');cls=main.CopilotProxy
  elif provider=='azure':ep=main.AzureOpenAIEndpoint('synthetic',url,'');cls=main.AzureOpenAIProxy
  else:ep=main.WorkspaceEndpoint('synthetic',url,'');cls=main.ClaudeProxy
  p=cls(main.LoadBalancer([ep]),'');await p.client.aclose();p.client=httpx.AsyncClient(trust_env=False,timeout=None)
  p.load_balancer._open(ep);ep.circuit_retry_at=0;await p.load_balancer.on_request_start(ep)
  if provider=='databricks':response=await p._stream_request(ep,url,{}, {},model='synthetic',start_time=time.time())
  else:response=await p._stream_response(ep,url,{}, {},'synthetic',api,time.time())
  async def send(message):
   if message['type']=='http.response.body' and message.get('body')==b': ready\n\n':sent.set()
  async def receive():await asyncio.Event().wait()
  scope={'type':'http','method':'POST','asgi':{'version':'3.0','spec_version':'2.4'}}
  task=asyncio.create_task(response(scope,receive,send))
  try:
   await asyncio.wait_for(sent.wait(),2);await asyncio.sleep(.02)
   self.assertGreater(main._STREAM_BUFFER_BUDGET.retained,0)
   task.cancel()
   with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,2)
   await asyncio.wait_for(peerclosed.wait(),2)
   self.assertEqual(len(p.client._transport._pool._requests),0);self.assertEqual(ep.active_requests,0);self.assertEqual(ep.cancelled_requests,1);self.assertEqual(ep.total_errors,0);self.assertFalse(ep.half_open_in_flight)
   self.assertEqual(main._STREAM_BUFFER_BUDGET.retained,0);self.assertEqual(getattr(p,'_stream_connections',{}),{})
   self.assertFalse([t for t in asyncio.all_tasks() if '_OwnedResponseStream' in t.get_coro().__qualname__]);self.assertEqual(len(requests),1)
   print('NORMAL_ASGI_GZIP_CANCEL_OK',provider,api,flush=True)
  finally:
   if not task.done():task.cancel()
   with contextlib.suppress(BaseException):await task
   await response.body_iterator.aclose();await p.client.aclose();server.close();await server.wait_closed();await asyncio.gather(*list(handlers),return_exceptions=True)
 async def test_nonfinite_overflow_in_completed_json_is_not_success(self):
  for provider in ('copilot','azure'):
   with self.subTest(provider=provider):
    wire=b'data: {"type":"response.completed","response":{"id":"synthetic","metadata":{"n":1e400}}}\n\n'
    r=await run_case('overflow_json',wire,provider=provider)
    print('OVERFLOW_JSON',json.dumps(r),flush=True)
    self.assertEqual(r['endpoint_successes'],0,'serde_json rejects an out-of-range JSON number anywhere in the event')
 async def test_extra_real_socket_schema_and_gzip_errors(self):
  wires=[b'data: {"type":"response.completed","response":{"id":"x","status":"incomplete"}}\n\n',b'data: {"type":"response.completed","response":{"id":"x","usage":{"input_tokens":true,"output_tokens":1,"total_tokens":2}}}\n\n',b'data: {"type":"response.completed","response":{"id":"x"},"call_id":3}\n\n']
  for provider in ('copilot','azure'):
   for i,wire in enumerate(wires):
    r=await run_case('independent_schema_'+str(i),wire,fragments=3,gzip_body=True,provider=provider)
    self.assertEqual(r['endpoint_successes'],0);self.assertEqual(r['endpoint_errors'],1);self.assertEqual(r['settlement_calls'],1);self.assertEqual(r['source_requests'],1)
    self.assertEqual(r['pool_requests_before_teardown'],0);self.assertEqual(r['business_active_before_teardown'],0)
  print('EXTRA_SCHEMA_SOCKET_CASES_OK',6,flush=True)
if __name__=='__main__':
 import signal;signal.alarm(90)
 assert main.__file__=='/app/main.py'
 assert hashlib.sha256(open(main.__file__,'rb').read()).hexdigest()=='75817f1f4b627e5dc9a1bce297d14b4f5081e091fdd6effd2e50ca59fb087648'
 unittest.main(verbosity=2)
