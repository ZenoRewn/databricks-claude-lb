"""Hierarchical inference circuits; synthetic evidence. Author: Zeno Ren."""
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import httpx
from h2.events import ConnectionTerminated

import main
from tests.test_protocol_diagnostics import reset_error


class ScopedCircuitTests(unittest.IsolatedAsyncioTestCase):
    # Exact integer base: see the note in tests/test_recovery_preference.py. A real
    # monotonic reading makes cooldown arithmetic drift by one float ulp, so the
    # Retry-After assertion below would intermittently see 31 instead of 30.
    BASE_MONOTONIC = 1_000_000.0

    async def asyncSetUp(self):
        self.now = self.BASE_MONOTONIC
        self.clock = patch.object(main, 'time', SimpleNamespace(monotonic=lambda:self.now, time=time.time))
        self.clock.start()
        self.ep = main.CopilotEndpoint('synthetic', '', models=['a','b'])
        self.lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=2, circuit_breaker_timeout=30)

    async def asyncTearDown(self):
        self.clock.stop()

    async def fail(self, model='a', api='responses', scope='model_api'):
        lease = await self.lb.on_request_start(self.ep, model=model, api_type=api)
        await self.lb.on_request_end(self.ep, False, lease=lease, failure_scope=scope)

    async def test_model_and_api_failures_do_not_block_siblings(self):
        await self.fail();await self.fail()
        self.assertFalse(self.lb.is_available(self.ep, model='a', api_type='responses'))
        self.assertTrue(self.lb.is_available(self.ep, model='b', api_type='responses'))
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='chat'))
        self.assertFalse(self.ep.circuit_open)
        self.assertEqual(self.ep.total_errors, 2)
        with self.assertRaises(main.HTTPException) as caught:
            await self.lb.on_request_start(self.ep, model='a', api_type='responses')
        self.assertEqual(caught.exception.headers['Retry-After'], '30')

    async def test_connection_failures_retain_shared_protection(self):
        await self.fail(scope='endpoint');await self.fail(scope='endpoint')
        self.assertFalse(self.lb.is_available(self.ep, model='b', api_type='chat'))
        self.assertTrue(self.ep.circuit_open)

    async def test_any_api_inspection_respects_only_configured_capabilities(self):
        self.ep.api_types=('responses',)
        await self.fail();await self.fail()
        self.assertFalse(self.lb.is_available(self.ep,model='a'))
        self.ep.api_types=('responses','chat')
        self.assertTrue(self.lb.is_available(self.ep,model='a'))

    async def test_local_failure_cannot_claim_shared_trial_recovery(self):
        self.lb._open(self.ep)
        self.now+=30
        lease=await self.lb.on_request_start(self.ep,model='a',api_type='responses')
        await self.lb.on_request_end(self.ep,False,lease=lease,failure_scope='model_api')
        self.assertTrue(self.ep.circuit_open)
        self.assertEqual(self.lb.circuit_state(self.ep),'OPEN')
        self.assertEqual(self.ep.consecutive_errors,0)

    async def test_stale_result_cannot_clear_scoped_trip_and_trials_are_atomic(self):
        stale = await self.lb.on_request_start(self.ep, model='a', api_type='responses')
        await self.fail();await self.fail()
        await self.lb.on_request_end(self.ep, True, lease=stale)
        self.assertFalse(self.lb.is_available(self.ep, model='a', api_type='responses'))
        self.now += 30
        winner = await self.lb.on_request_start(self.ep, model='a', api_type='responses')
        with self.assertRaises(main.HTTPException):
            await self.lb.on_request_start(self.ep, model='a', api_type='responses')
        await self.lb.on_request_end(self.ep, False, lease=winner, cancelled=True)
        self.assertFalse(self.lb.is_available(self.ep, model='a', api_type='responses'))
        self.now += 30
        winner = await self.lb.on_request_start(self.ep, model='a', api_type='responses')
        await self.lb.on_request_end(self.ep, True, lease=winner)
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='responses'))
        self.assertEqual(self.ep.active_requests,0)

    async def test_administrative_reset_covers_local_circuits_and_old_leases(self):
        await self.fail();await self.fail()
        self.now += 30
        old = await self.lb.on_request_start(self.ep, model='a', api_type='responses')
        self.lb.reset_circuit(self.ep)
        await self.lb.on_request_end(self.ep,False,lease=old,failure_scope='model_api')
        self.assertTrue(self.lb.is_available(self.ep,model='a',api_type='responses'))

    async def test_wildcard_route_cache_is_bounded_and_never_evicts_open_circuit(self):
        self.ep.models=[]
        await self.fail();await self.fail()
        for i in range(300):
            lease=await self.lb.on_request_start(self.ep,model=f'm{i}',api_type='responses')
            await self.lb.on_request_end(self.ep,True,lease=lease)
        self.assertLessEqual(len(self.lb._route_circuits),128)
        self.assertFalse(self.lb.is_available(self.ep,model='a',api_type='responses'))

    async def test_real_proxy_scopes_reset_without_replaying_or_switching_accounts(self):
        class ResetStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                raise reset_error()
                yield b''
            async def aclose(self):pass
        proxy=main.CopilotProxy(self.lb,'')
        await proxy.client.aclose()
        sends=[]
        def upstream(req):
            sends.append(req)
            if json_model(req)=='a':
                return httpx.Response(200,stream=ResetStream(),extensions={'http_version':b'HTTP/2'})
            return httpx.Response(200,json={'id':'synthetic','status':'completed','output':[]})
        proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream),trust_env=False)
        proxy._build_headers=AsyncMock(return_value={})
        proxy._probe_upstream_connect=AsyncMock(return_value={'ok':True})
        try:
            for _ in range(2):
                result=await proxy.proxy_responses({'model':'a'},stream=True)
                self.assertIn(b'response.failed',b''.join([part async for part in result.body_iterator]))
            with self.assertRaises(main.HTTPException) as caught:
                await proxy.proxy_responses({'model':'a'},stream=True)
            self.assertEqual(caught.exception.status_code,503)
            result=await proxy.proxy_responses({'model':'b'})
            self.assertEqual(result.status_code,200)
            self.assertEqual(len(sends),3)
            self.assertEqual(self.ep.active_requests,0)
        finally:await proxy.close()


def json_model(request):
    import json
    return json.loads(request.content)['model']


def test_scope_requires_typed_stream_evidence_and_preserves_shared_overload():
    assert main._copilot_failure_scope(exception=reset_error())=='model_api'
    assert main._copilot_failure_scope(exception=reset_error(7))=='endpoint'
    assert main._copilot_failure_scope(exception=httpx.RemoteProtocolError('unknown'))=='endpoint'
    assert main._copilot_failure_scope(reason='invalid_protocol')=='model_api'
    for reason in ('upstream_unavailable','authentication','rate_limited','upstream_truncated'):
        assert main._copilot_failure_scope(reason=reason)=='endpoint'
