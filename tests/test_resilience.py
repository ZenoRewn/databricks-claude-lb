"""Deterministic breaker/accounting regressions; no provider network or lifespan."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
import httpx
import main
import test_copilot_request_lifecycle as fixtures


class ResilienceRegressionTests(unittest.IsolatedAsyncioTestCase):
    def make(self, threshold=2):
        return fixtures.CopilotRequestLifecycleTests()._make_proxy(Mock(), threshold)

    async def test_success_breaks_failure_streak(self):
        _, lb, ep = self.make()
        for ok in (False, True, False):
            await lb.on_request_start(ep)
            await lb.on_request_end(ep, ok)
        self.assertFalse(ep.circuit_open)
        self.assertEqual(ep.total_errors, 2)

    async def test_cooldown_does_not_erase_error_telemetry(self):
        _, lb, ep = self.make(1)
        with patch.object(main.time, 'time', return_value=100), patch.object(main.time, 'monotonic', return_value=100):
            await lb.on_request_start(ep)
            await lb.on_request_end(ep, False)
        with patch.object(main.time, 'time', return_value=161), patch.object(main.time, 'monotonic', return_value=161):
            self.assertIn(ep, lb.get_available_endpoints())
        self.assertEqual(ep.total_errors, 1)

    async def test_readiness_allows_real_recovery_without_reset(self):
        proxy, lb, ep = self.make(1)
        with patch.object(main.time, 'time', return_value=100), patch.object(main.time, 'monotonic', return_value=100):
            await lb.on_request_start(ep)
            await lb.on_request_end(ep, False)
        with patch.object(main.time, 'time', return_value=161), patch.object(main.time, 'monotonic', return_value=161):
            self.assertTrue(proxy.is_any_endpoint_healthy())
            self.assertTrue(ep.circuit_open, 'readiness must not declare recovered')

    async def test_known_model_unavailable_is_503_not_404(self):
        proxy, lb, ep = self.make(1)
        await lb.on_request_start(ep)
        await lb.on_request_end(ep, False)
        with patch.object(main, 'copilot_proxy', proxy), patch.object(main, 'azure_proxy', None):
            with self.assertRaises(main.HTTPException) as ctx:
                await main._route_openai({'model':'gpt-test'}, False, 'responses')
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.headers['Retry-After'], '60')

    async def test_unknown_model_stays_404(self):
        proxy, _, _ = self.make()
        with patch.object(main, 'copilot_proxy', proxy), patch.object(main, 'azure_proxy', None):
            with self.assertRaises(main.HTTPException) as ctx:
                await main._route_openai({'model':'not-allowed'}, False, 'responses')
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_ambiguous_post_read_failure_must_not_replay(self):
        proxy, lb, ep = self.make(5)
        proxy.client.post = AsyncMock(side_effect=httpx.ReadError('ambiguous execution'))
        with patch.object(main.asyncio, 'sleep', new=AsyncMock()):
            with self.assertRaises(main.HTTPException):
                await proxy.proxy_responses({'model':'gpt-test', 'input':[]})
        self.assertEqual(proxy.client.post.await_count, 1)
        self.assertEqual(ep.active_requests, 0)
