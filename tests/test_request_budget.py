"""Budgets terminate the client protocol without replaying ambiguous generation."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import request_budget

import main
from request_budget import RequestBudgetMiddleware, startup_budget, headers_received, UpstreamStartupTimeout
from request_telemetry import RequestTelemetry, RequestTelemetryMiddleware


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self,app,api='responses'):
        metrics=RequestTelemetry();sent=[]
        async def send(message):sent.append(message)
        wrapped=RequestTelemetryMiddleware(RequestBudgetMiddleware(app,timeout=.02,error_frame_factory=main._sse_terminal_error),metrics)
        paths={'messages':'/v1/messages','responses':'/v1/responses','chat':'/v1/chat/completions'}
        await wrapped({'type':'http','method':'POST','path':paths[api],'headers':[]},AsyncMock(),send)
        return sent,metrics

    async def test_before_headers_returns_504_and_one_deadline_outcome(self):
        async def app(scope,receive,send):await asyncio.Event().wait()
        sent,metrics=await self.drive(app)
        self.assertEqual(sent[0]['status'],504)
        error=json.loads(sent[1]['body'])['error']
        self.assertEqual(error['code'],'request_deadline_exceeded')
        self.assertEqual(metrics.outcomes,{('responses','deadline_exceeded'):1})
        self.assertEqual(metrics.active['responses'],0)

    async def test_after_sse_headers_emits_protocol_error_without_second_response_start(self):
        async def app(scope,receive,send):
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'text/event-stream')]})
            await send({'type':'http.response.body','body':b': keep-alive\n\n','more_body':True})
            await asyncio.Event().wait()
        for api in ('messages','responses','chat'):
            with self.subTest(api=api):
                sent,metrics=await self.drive(app,api)
                self.assertEqual(len([m for m in sent if m['type']=='http.response.start']),1)
                wire=b''.join(m.get('body',b'') for m in sent)
                self.assertIn(b'request_deadline_exceeded',wire)
                self.assertEqual(b'data: [DONE]' in wire,api=='chat')
                if api=='messages':self.assertIn(b'event: error',wire)
                if api=='responses':self.assertIn(b'response.failed',wire)
                self.assertFalse(sent[-1].get('more_body',False))
                self.assertEqual(metrics.outcomes[(api,'deadline_exceeded')],1)

    async def test_external_cancellation_is_not_relabelled_deadline(self):
        entered=asyncio.Event()
        async def app(scope,receive,send):
            entered.set();await asyncio.Event().wait()
        sent=[]
        async def send(message):sent.append(message)
        task=asyncio.create_task(RequestBudgetMiddleware(app,timeout=10,error_frame_factory=main._sse_terminal_error)(
            {'type':'http','method':'POST','path':'/v1/responses','headers':[]},AsyncMock(),send))
        await entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(sent,[])

    async def test_completed_body_does_not_get_an_extra_error_during_cleanup(self):
        async def app(scope,receive,send):
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
            await asyncio.Event().wait()
        sent,_=await self.drive(app)
        self.assertEqual(len(sent),2)

    async def test_startup_timeout_does_not_authorize_post_replay(self):
        with self.assertRaises(UpstreamStartupTimeout) as captured:
            async with startup_budget(.01):await asyncio.Event().wait()
        self.assertFalse(main._replay_safe_transport_failure(captured.exception))

    async def test_headers_disable_startup_timer_before_long_success_body(self):
        async with startup_budget(.005):
            headers_received()
            await asyncio.sleep(.02)

    async def test_unrelated_timeout_error_is_not_claimed_as_our_deadline(self):
        with self.assertRaisesRegex(TimeoutError,'synthetic'):
            async with startup_budget(.05):raise TimeoutError('synthetic')


class ProxyBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def proxy(self,url,transport=None):
        ep=main.WorkspaceEndpoint('fixture',url,'synthetic')
        proxy=main.ClaudeProxy(main.LoadBalancer([ep]),'synthetic')
        hooks=proxy.client.event_hooks
        await proxy.client.aclose()
        proxy.client=httpx.AsyncClient(transport=transport,trust_env=False,event_hooks=hooks,
                                      timeout=httpx.Timeout(connect=1,read=None,write=1,pool=1))
        self.addAsyncCleanup(proxy.close)
        return proxy,ep

    async def test_normal_post_can_keep_reading_after_headers_past_startup_budget(self):
        class SlowBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"type":"message",'
                await asyncio.sleep(.03)
                yield b'"stop_reason":"end_turn","usage":{}}'
        async def upstream(request):return httpx.Response(200,stream=SlowBody())
        proxy,_=await self.proxy('https://fixture.invalid',httpx.MockTransport(upstream))
        with patch.object(request_budget,'STARTUP_TIMEOUT',.01),patch.object(main,'usage_store',None):
            response=await proxy.proxy_request({'model':'claude-opus-5','messages':[]})
        self.assertEqual(response.status_code,200)

    async def stalled_server(self,send_headers):
        disconnected=asyncio.Event();calls=[]
        async def peer(reader,writer):
            try:
                head=await reader.readuntil(b'\r\n\r\n')
                size=next((int(line.split(b':',1)[1]) for line in head.split(b'\r\n') if line.lower().startswith(b'content-length:')),0)
                if size:await reader.readexactly(size)
                calls.append(1)
                if send_headers:
                    frame=b'event: message_start\ndata: {"type":"message_start","message":{"usage":{}}}\n\n'
                    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n')
                    writer.write(f'{len(frame):x}\r\n'.encode()+frame+b'\r\n')
                    await writer.drain()
                await reader.read()
                disconnected.set()
            finally:
                writer.close();await writer.wait_closed()
        server=await asyncio.start_server(peer,'127.0.0.1',0)
        async def close():server.close();await server.wait_closed()
        self.addAsyncCleanup(close)
        return f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}',disconnected,calls

    async def test_real_header_stall_is_closed_and_never_replayed(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                url,closed,calls=await self.stalled_server(False)
                proxy,ep=await self.proxy(url)
                with patch.object(request_budget,'STARTUP_TIMEOUT',.04):
                    if stream:
                        response=await proxy.proxy_request({'model':'claude-opus-5','messages':[]},stream=True)
                        wire=b''.join([chunk async for chunk in response.body_iterator])
                        self.assertIn(b'UpstreamStartupTimeout',wire)
                    else:
                        with self.assertRaises(main.HTTPException) as captured:
                            await proxy.proxy_request({'model':'claude-opus-5','messages':[]})
                        self.assertEqual(captured.exception.status_code,502)
                        self.assertEqual(captured.exception.detail['error']['failure_type'],'UpstreamStartupTimeout')
                await asyncio.wait_for(closed.wait(),1)
                self.assertEqual(len(calls),1)
                self.assertEqual(ep.active_requests,0)

    async def test_real_stream_total_deadline_preserves_cleanup_and_cancel_attribution(self):
        url,closed,calls=await self.stalled_server(True)
        proxy,ep=await self.proxy(url)
        async def app(scope,receive,send):
            response=await proxy.proxy_request({'model':'claude-opus-5','messages':[]},stream=True)
            await response(scope,receive,send)
        metrics=RequestTelemetry();sent=[]
        async def send(message):sent.append(message)
        wrapped=RequestTelemetryMiddleware(RequestBudgetMiddleware(app,timeout=.08,error_frame_factory=main._sse_terminal_error),metrics)
        scope={'type':'http','method':'POST','path':'/v1/messages','headers':[],
               'asgi':{'version':'3.0','spec_version':'2.4'}}
        await wrapped(scope,AsyncMock(),send)
        await asyncio.wait_for(closed.wait(),1)
        self.assertEqual(len(calls),1)
        self.assertEqual(ep.active_requests,0)
        self.assertEqual(ep.cancelled_requests,1)
        self.assertEqual(ep.total_errors,0)
        self.assertEqual(metrics.outcomes[('messages','deadline_exceeded')],1)
        self.assertIn(b'request_deadline_exceeded',b''.join(m.get('body',b'') for m in sent))
