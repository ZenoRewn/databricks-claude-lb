"""P1 exact production token-lock and ordinary Sol adapter path, no auth traffic."""
import asyncio,contextlib,hashlib,json,logging,time,unittest
from unittest.mock import AsyncMock,patch
import httpx,main
logging.disable(logging.CRITICAL)
class Review(unittest.IsolatedAsyncioTestCase):
 async def test_real_token_lock_disconnect_must_not_strand_trial(self):
  handlers=set();calls=[]
  async def upstream(r,w):
   handlers.add(asyncio.current_task())
   try:
    h=await r.readuntil(b'\r\n\r\n');n=next((int(x.split(b':',1)[1]) for x in h.split(b'\r\n') if x.lower().startswith(b'content-length:')),0)
    b=await r.readexactly(n) if n else b''
    self.assertTrue(h.startswith(b'POST /responses HTTP/1.1'))
    self.assertNotIn(b'authorization:',h.lower());self.assertFalse(json.loads(b)['stream'])
    calls.append(1);w.write(b'HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}');await w.drain()
   finally:w.close();await w.wait_closed();handlers.discard(asyncio.current_task())
  s=await asyncio.start_server(upstream,'127.0.0.1',0);url='http://127.0.0.1:'+str(s.sockets[0].getsockname()[1])
  ep=main.CopilotEndpoint('synthetic','',models=['gpt-5.6-sol'],api_types=['responses']);ep.session_base_url=url
  lb=main.LoadBalancer([ep]);p=main.CopilotProxy(lb,'');await p.client.aclose();p.client=httpx.AsyncClient(trust_env=False)
  p._build_headers=AsyncMock(return_value={})
  # Guard forbids a real token exchange even if this oracle's synchronization fails.
  p._exchange_token=AsyncMock(side_effect=AssertionError('forbidden token exchange'))
  lock=p._get_token_lock(ep);await lock.acquire();lb._open(ep);ep.circuit_retry_at=0
  async def disconnected():return bool(lock._waiters)
  try:
   with patch.object(main,'copilot_proxy',p),patch.object(main,'azure_proxy',None):
    with self.assertRaises(main.HTTPException) as caught:
     await asyncio.wait_for(main._route_openai_chat({'model':'gpt-5.6-sol','messages':[]},True,disconnect_checker=disconnected),3)
   self.assertEqual(caught.exception.status_code,499);p._exchange_token.assert_not_awaited()
   snapshot=dict(route='ordinary Sol Chat -> buffered Responses',http_status=499,upstream_posts=len(calls),active=ep.active_requests,half_open_in_flight=ep.half_open_in_flight,owned_attempts=len(lb._attempts[id(ep)]),cancelled=ep.cancelled_requests,pool_requests=len(p.client._transport._pool._requests),token_exchange_calls=p._exchange_token.await_count)
   ep.circuit_retry_at=time.monotonic()-10000;snapshot['still_unavailable_after_expired_cooldown']=not lb.is_available(ep)
   print('REAL_LOCK_REPRO',json.dumps(snapshot),flush=True)
   self.assertEqual(ep.active_requests,0,'ordinary downstream disconnect must release HALF_OPEN attempt')
   self.assertFalse(ep.half_open_in_flight)
  finally:
   lock.release();await p.client.aclose();s.close();await s.wait_closed();await asyncio.gather(*list(handlers),return_exceptions=True)
if __name__=='__main__':
 import signal;signal.alarm(15)
 assert main.__file__=='/app/main.py'
 assert hashlib.sha256(open(main.__file__,'rb').read()).hexdigest()=='75817f1f4b627e5dc9a1bce297d14b4f5081e091fdd6effd2e50ca59fb087648'
 unittest.main(verbosity=2)
