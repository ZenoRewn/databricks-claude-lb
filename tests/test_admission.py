"""Queue bounds, tenant fairness, cancellation races, and request-body budgets."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from starlette.requests import Request
import main
from admission import AdmissionController, AdmissionError, CURRENT_LEASE, AdmissionMiddleware


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    def controller(self,**kwargs):
        return AdmissionController(max_active=kwargs.get('max_active',1),max_queued=kwargs.get('max_queued',1),
                                   wait_timeout=kwargs.get('wait_timeout',.1),body_budget=kwargs.get('body_budget',100),
                                   tenant_limits=kwargs.get('tenant_limits',{}))
    async def test_queue_full_and_exactly_once_release(self):
        c=self.controller();first=await c.acquire('a')
        pending=asyncio.create_task(c.acquire('b'));await asyncio.sleep(0)
        with self.assertRaises(AdmissionError) as caught:await c.acquire('c')
        self.assertEqual(caught.exception.reason,'queue_full')
        first.release();second=await pending
        self.assertEqual(c.active,1)
        second.release();second.release()
        self.assertEqual(c.active,0);self.assertEqual(len(c.waiters),0)
    async def test_cancel_before_and_after_slot_offer_does_not_leak(self):
        for offered in (False,True):
            with self.subTest(offered=offered):
                c=self.controller();first=await c.acquire('a')
                pending=asyncio.create_task(c.acquire('b'));await asyncio.sleep(0)
                if offered:first.release()
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):await pending
                first.release()
                self.assertEqual(c.active,0);self.assertEqual(len(c.waiters),0)
    async def test_queue_timeout_removes_waiter(self):
        c=self.controller(wait_timeout=.01);lease=await c.acquire('a')
        with self.assertRaises(AdmissionError) as caught:await c.acquire('b')
        self.assertEqual(caught.exception.reason,'queue_timeout')
        self.assertEqual(len(c.waiters),0);lease.release()
    async def test_one_tenant_does_not_block_another_tenants_free_slot(self):
        c=self.controller(max_active=2,max_queued=3,tenant_limits={'monitor':1})
        first=await c.acquire('monitor')
        pending=asyncio.create_task(c.acquire('monitor'));await asyncio.sleep(0)
        other=await c.acquire('interactive')
        self.assertEqual(c.active,2);self.assertFalse(pending.done())
        other.release();first.release();(await pending).release()
        self.assertEqual(c.active,0)
    async def test_body_budget_is_process_wide_and_released_with_lease(self):
        c=self.controller(max_active=2)
        first=await c.acquire('a');second=await c.acquire('b')
        first.reserve_body(80)
        with self.assertRaises(AdmissionError) as caught:second.reserve_body(30)
        self.assertEqual(caught.exception.reason,'body_memory_budget')
        self.assertEqual(c.body_bytes,80)
        first.release();second.reserve_body(30);second.release()
        self.assertEqual(c.body_bytes,0)
    async def test_middleware_rejection_is_json_503_with_retry_after(self):
        c=self.controller(max_queued=0);first=await c.acquire('a');sent=[]
        async def send(message):sent.append(message)
        app=AsyncMock()
        scope={'type':'http','method':'POST','path':'/v1/responses','headers':[]}
        await AdmissionMiddleware(app,controller=c,tenant_for_scope=lambda s:'b')(scope,AsyncMock(),send)
        self.assertEqual(sent[0]['status'],503)
        self.assertEqual(dict(sent[0]['headers'])[b'retry-after'],b'1')
        self.assertEqual(json.loads(sent[1]['body'])['error']['code'],'lb_overloaded')
        app.assert_not_awaited();first.release()
    async def test_middleware_holds_slot_until_response_and_cleanup_end(self):
        c=self.controller();states=[]
        async def app(scope,receive,send):
            states.append((c.active,CURRENT_LEASE.get() is not None))
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
            states.append((c.active,True))
        await AdmissionMiddleware(app,controller=c,tenant_for_scope=lambda s:'a')(
            {'type':'http','method':'POST','path':'/v1/messages','headers':[]},AsyncMock(),AsyncMock())
        self.assertEqual(states,[(1,True),(1,True)])
        self.assertEqual(c.active,0);self.assertIsNone(CURRENT_LEASE.get())

    async def test_drain_rejects_waiters_without_interrupting_active_work(self):
        c=self.controller();active=await c.acquire('a')
        pending=asyncio.create_task(c.acquire('b'));await asyncio.sleep(0)
        c.drain()
        with self.assertRaises(AdmissionError) as caught:await pending
        self.assertEqual(caught.exception.reason,'draining')
        self.assertEqual(c.active,1)
        with self.assertRaises(AdmissionError):await c.acquire('a')
        active.release();self.assertEqual(c.active,0)


class BodyAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def request(self,chunks):
        read=[]
        async def receive():
            n=len(read);read.append(n)
            return {'type':'http.request','body':chunks[n],'more_body':n<len(chunks)-1}
        return Request({'type':'http','method':'POST','path':'/v1/messages','headers':[]},receive),read
    async def test_size_limit_stops_reading_before_buffering_rest(self):
        request,read=self.request([b'abcd',b'efgh',b'must-not-read'])
        with patch.object(main,'MAX_RAW_REQUEST_SIZE',6):
            with self.assertRaises(main.HTTPException) as caught:await main._read_bounded_request_body(request)
        self.assertEqual(caught.exception.status_code,413)
        self.assertEqual(len(read),2)
    async def test_global_body_budget_and_request_cache(self):
        c=AdmissionController(max_active=1,max_queued=0,wait_timeout=.1,body_budget=5,tenant_limits={})
        lease=await c.acquire('a');token=CURRENT_LEASE.set(lease)
        try:
            request,read=self.request([b'abc',b'def'])
            with self.assertRaises(main.HTTPException) as caught:await main._read_bounded_request_body(request)
            self.assertEqual(caught.exception.status_code,503)
            self.assertEqual(c.body_bytes,3)
            self.assertTrue(request.scope['state']['lb_overloaded'])
        finally:CURRENT_LEASE.reset(token);lease.release()
        self.assertEqual(c.body_bytes,0)
        request,read=self.request([b'abc',b'd'])
        self.assertEqual(await main._read_bounded_request_body(request),b'abcd')
        self.assertEqual(await request.body(),b'abcd')
        self.assertEqual(len(read),2)
    async def test_body_upload_timeout_precedes_upstream_admission(self):
        async def receive():await asyncio.Event().wait()
        request=Request({'type':'http','method':'POST','path':'/v1/messages','headers':[]},receive)
        with patch.object(main,'REQUEST_BODY_TIMEOUT_SECONDS',.01):
            with self.assertRaises(main.HTTPException) as caught:await main._read_bounded_request_body(request)
        self.assertEqual(caught.exception.status_code,408)
