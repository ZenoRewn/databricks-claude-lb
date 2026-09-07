import asyncio
import unittest
from unittest.mock import patch
import main


class CircuitAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch.object(main.time, 'monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.ep = main.CopilotEndpoint('test', 'synthetic', models=['a','b'])
        self.lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=2, circuit_breaker_timeout=30)

    async def trip(self):
        for _ in range(2):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, lease=lease)
        self.assertEqual(self.lb.circuit_state(self.ep), 'OPEN')

    async def test_concurrent_half_open_claim_is_bounded(self):
        await self.trip()
        self.now += 30
        async def claim():
            await asyncio.sleep(0)
            return await self.lb.on_request_start(self.ep)
        leases = await asyncio.gather(*(claim() for _ in range(100)), return_exceptions=True)
        admitted = [x for x in leases if isinstance(x, main.RequestAttempt)]
        self.assertEqual(len(admitted), 1)
        self.assertEqual(sum(isinstance(x, main.HTTPException) for x in leases), 99)
        self.assertEqual(self.ep.active_requests, 1)
        self.assertTrue(self.ep.circuit_open)
        self.assertEqual(self.ep.rejected_requests, 99)
        self.assertTrue(self.lb.is_available(self.ep, readiness=True))
        self.assertFalse(self.lb.is_available(self.ep))
        await self.lb.on_request_end(self.ep, True, lease=admitted[0])
        self.assertEqual(self.lb.circuit_state(self.ep), 'CLOSED')
        self.assertEqual(self.ep.total_errors, 2)
        self.assertEqual(self.ep.consecutive_errors, 0)

    async def test_late_completions_cannot_close_or_reopen_new_generation(self):
        old_ok = await self.lb.on_request_start(self.ep)
        old_bad = await self.lb.on_request_start(self.ep)
        await self.trip()
        await self.lb.on_request_end(self.ep, True, lease=old_ok)
        self.assertTrue(self.ep.circuit_open)
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, True, lease=trial)
        await self.lb.on_request_end(self.ep, False, lease=old_bad)
        self.assertFalse(self.ep.circuit_open)
        self.assertEqual(self.ep.total_errors, 3)
        self.assertEqual(self.ep.consecutive_errors, 0)
        self.assertEqual(self.ep.active_requests, 0)

    async def test_cancelled_trial_releases_once_without_health_failure(self):
        await self.trip()
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, False, lease=trial, cancelled=True)
        await self.lb.on_request_end(self.ep, False, lease=trial, cancelled=True)
        self.assertEqual(self.ep.cancelled_requests, 1)
        self.assertEqual(self.ep.total_errors, 2)
        self.assertEqual(self.ep.active_requests, 0)
        self.assertEqual(self.lb.circuit_state(self.ep), 'OPEN')
        self.assertEqual(self.lb.retry_after(), 30)
        self.now += 30
        next_trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, True, lease=next_trial)
        self.assertFalse(self.ep.circuit_open)

    async def test_trial_failure_reopens_for_fresh_cooldown(self):
        await self.trip()
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        self.now += 4
        await self.lb.on_request_end(self.ep, False, lease=trial)
        self.assertEqual(self.lb.retry_after(), 30)
        self.assertEqual(self.ep.total_errors, 3)
        self.assertFalse(self.ep.half_open_in_flight)

    async def test_neutral_result_does_not_reset_or_increment_failure_streak(self):
        lease = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, False, lease=lease)
        for kw in ({'is_client_error':True}, {'cancelled':True}):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, lease=lease, **kw)
        self.assertEqual(self.ep.total_errors, 1)
        self.assertEqual(self.ep.consecutive_errors, 1)
        self.assertEqual(self.ep.neutral_requests, 1)
        self.assertEqual(self.ep.cancelled_requests, 1)

    async def test_neutral_trial_does_not_claim_inference_recovered(self):
        await self.trip()
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, False, lease=trial, is_client_error=True)
        self.assertEqual(self.lb.circuit_state(self.ep), 'OPEN')
        self.assertEqual(self.ep.total_errors, 2)

    async def test_probe_not_abandoned_when_request_runs_longer_than_cooldown(self):
        await self.trip()
        self.now += 30
        lease = await self.lb.on_request_start(self.ep)
        self.now += 10000
        self.assertFalse(self.lb.get_available_endpoints())
        self.assertEqual(self.ep.active_requests, 1)
        await self.lb.on_request_end(self.ep, False, lease=lease, cancelled=True)
        self.assertEqual(self.ep.active_requests, 0)

    async def test_inspection_is_pure_and_does_not_advance_round_robin(self):
        await self.trip()
        self.now += 30
        proxy = object.__new__(main.CopilotProxy)
        proxy.load_balancer = self.lb
        for _ in range(10):
            self.assertTrue(proxy.can_handle('a'))
            self.assertIn(self.ep, self.lb.get_available_endpoints())
            self.assertEqual(self.lb.get_stats()['endpoints'][0]['circuit_state'], 'HALF_OPEN')
        self.assertEqual(self.ep.total_requests, 2)
        self.assertEqual(self.lb._rr_index, 0)
        self.assertFalse(self.ep.half_open_in_flight)
        self.assertTrue(self.ep.circuit_open)

    async def test_monotonic_cooldown_unaffected_by_wallclock(self):
        await self.trip()
        with patch.object(main.time, 'time', return_value=2**40):
            self.assertFalse(self.lb.is_available(self.ep))
        self.now += 29.2
        self.assertEqual(self.lb.retry_after(), 1)
        self.assertFalse(self.lb.is_available(self.ep))
        self.now += .8
        self.assertTrue(self.lb.is_available(self.ep))

    async def test_endpoint_state_is_independent(self):
        sibling = main.CopilotEndpoint('other', 'synthetic', models=['a'])
        self.lb.endpoints.append(sibling)
        await self.trip()
        self.assertEqual(self.lb.get_available_endpoints(), [sibling])
        lease = await self.lb.on_request_start(sibling)
        await self.lb.on_request_end(sibling, True, lease=lease)
        self.assertTrue(self.ep.circuit_open)
        self.assertFalse(sibling.circuit_open)

    async def test_legacy_completion_requires_unambiguous_identity(self):
        a = await self.lb.on_request_start(self.ep)
        b = await self.lb.on_request_start(self.ep)
        with self.assertRaises(RuntimeError):
            await self.lb.on_request_end(self.ep, True)
        self.assertEqual(self.ep.active_requests, 2)
        await self.lb.on_request_end(self.ep, True, lease=a)
        await self.lb.on_request_end(self.ep, True, lease=b)
        self.assertEqual(self.ep.active_requests, 0)
