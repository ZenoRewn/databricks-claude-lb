"""Bounded recovery preference using existing ingress only. Author: Zeno Ren."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main


class RecoveryPreferenceTests(unittest.IsolatedAsyncioTestCase):
    # The clock is fully mocked, so the base must be an exact integer rather than
    # a real monotonic reading. With an arbitrary float base, (base+30+10+30) minus
    # (base+30+10) is not exactly 30.0, and math.ceil in retry_after then yields 31.
    # That made cooldown assertions depend on the host's uptime. Ceil itself is the
    # correct conservative behaviour for Retry-After; only the base was at fault.
    BASE_MONOTONIC=1_000_000.0

    async def asyncSetUp(self):
        self.now=self.BASE_MONOTONIC
        self.clock=patch.object(main,'time',SimpleNamespace(monotonic=lambda:self.now,time=time.time))
        self.clock.start()
        self.ep=main.CopilotEndpoint('synthetic','',models=['a'])
        self.lb=main.LoadBalancer([self.ep],circuit_breaker_timeout=30)
        self.small={'model':'a','input':'small'}
        self.large={'model':'a','input':'large'*20000}
        self.lb._open(self.ep)
        self.now+=30

    async def asyncTearDown(self):self.clock.stop()

    async def test_large_request_waits_but_small_real_request_can_recover(self):
        with self.assertRaises(main.HTTPException) as caught:
            await self.lb.on_request_start(self.ep,model='a',api_type='responses',payload=self.large)
        self.assertEqual(caught.exception.headers['Retry-After'],'10')
        self.assertEqual(caught.exception.detail['error']['code'],'recovery_prefers_small_request')
        self.assertEqual(self.ep.total_requests,0)
        self.assertFalse(self.ep.half_open_in_flight)
        lease=await self.lb.on_request_start(self.ep,model='a',api_type='responses',payload=self.small)
        await self.lb.on_request_end(self.ep,True,lease=lease)
        self.assertFalse(self.ep.circuit_open)
        lease=await self.lb.on_request_start(self.ep,model='a',api_type='responses',payload=self.large)
        await self.lb.on_request_end(self.ep,True,lease=lease)

    async def test_window_expires_without_starving_large_or_opaque_input(self):
        opaque={'model':'a','input':[{'type':'reasoning','encrypted_content':'synthetic'}]}
        for body in (self.large,opaque):
            self.assertFalse(self.lb.is_available(self.ep,model='a',api_type='responses',payload=body))
        self.now+=10
        lease=await self.lb.on_request_start(self.ep,model='a',api_type='responses',payload=opaque)
        self.assertTrue(lease.probe)
        await self.lb.on_request_end(self.ep,False,lease=lease,cancelled=True)
        self.assertEqual(self.ep.active_requests,0)

    async def test_readiness_remains_pure_and_preference_does_not_switch_pinned_account(self):
        proxy=object.__new__(main.CopilotProxy);proxy.load_balancer=self.lb
        other=main.CopilotEndpoint('other','',models=['a']);self.lb.endpoints.append(other)
        for _ in range(5):self.assertTrue(self.lb.is_available(self.ep,readiness=True,payload=self.large))
        self.assertFalse(self.ep.half_open_in_flight)
        self.assertIsNone(proxy._select_endpoint('a','responses',pinned=self.ep,body=self.large))
        self.assertIs(proxy._select_endpoint('a','responses',body=self.large),other)

    async def test_shared_and_scoped_waits_use_the_longest_blocking_gate(self):
        # Route cooldown and account cooldown both apply to this exact route.
        self.lb._reset_circuit(self.ep)
        route=self.lb._route_circuit(self.ep,'a','responses',create=True)
        self.lb._open(route)
        self.now+=10
        self.lb._open(self.ep)
        self.assertEqual(self.lb.retry_after([self.ep],model='a',api_type='responses',payload=self.small),30)

    async def test_deferred_request_sends_no_inference_or_auth_call(self):
        proxy=main.CopilotProxy(self.lb,'')
        proxy._build_headers=AsyncMock(return_value={})
        proxy.client.send=AsyncMock()
        try:
            with self.assertRaises(main.HTTPException):
                await proxy.proxy_responses(self.large,stream=True)
            proxy._build_headers.assert_not_awaited()
            proxy.client.send.assert_not_awaited()
        finally:await proxy.close()

    async def test_local_preference_does_not_authorize_cross_provider_fallback(self):
        proxy=main.CopilotProxy(self.lb,'')
        azure=SimpleNamespace(load_balancer=main.LoadBalancer([main.AzureOpenAIEndpoint(
            'azure','https://fixture.invalid','',deployments=['a'])]),proxy_responses=AsyncMock())
        try:
            with patch.object(main,'copilot_proxy',proxy),patch.object(main,'azure_proxy',azure):
                with self.assertRaises(main.HTTPException) as caught:
                    await main._route_openai(self.large,True,'responses')
            self.assertEqual(caught.exception.detail['error']['code'],'recovery_prefers_small_request')
            azure.proxy_responses.assert_not_awaited()
        finally:await proxy.close()
