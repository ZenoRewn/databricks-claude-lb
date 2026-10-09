"""Preference window requires recent small-input evidence. Author: Zeno Ren.

A workload made only of large requests was deferred in every HALF_OPEN window
during the 2026-10-09 incident, because each cooldown re-armed the window. The
"window expires" guarantee holds per window, not across a circuit that keeps
reopening. Preferring a small request is only meaningful when small requests
actually arrive, so the preference is gated on observed evidence.
"""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class RecoveryWorkloadEvidenceTests(unittest.IsolatedAsyncioTestCase):
    # Exact integer base: see tests/test_recovery_preference.py.
    BASE_MONOTONIC = 1_000_000.0

    async def asyncSetUp(self):
        self.now = self.BASE_MONOTONIC
        self.clock = patch.object(main, 'time', SimpleNamespace(monotonic=lambda: self.now, time=time.time))
        self.clock.start()
        self.ep = main.CopilotEndpoint('synthetic', '', models=['a'])
        self.lb = main.LoadBalancer([self.ep], circuit_breaker_timeout=30)
        self.small = {'model': 'a', 'input': 'small'}
        self.large = {'model': 'a', 'input': 'large' * 20000}

    async def asyncTearDown(self):
        self.clock.stop()

    def _half_open(self):
        """Open the circuit and advance to the start of its HALF_OPEN window."""
        self.lb._open(self.ep)
        self.now += 30

    async def test_large_only_workload_is_not_deferred(self):
        # No small request ever arrived, so there is nothing worth waiting for.
        self._half_open()
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='responses', payload=self.large))
        lease = await self.lb.on_request_start(self.ep, model='a', api_type='responses', payload=self.large)
        self.assertTrue(lease.probe)

    async def test_small_input_during_cooldown_re_enables_the_preference(self):
        self.lb._open(self.ep)
        # A small request arrives while the circuit is still OPEN and is rejected.
        with self.assertRaises(main.HTTPException):
            await self.lb.on_request_start(self.ep, model='a', api_type='responses', payload=self.small)
        self.now += 30
        self.assertFalse(self.lb.is_available(self.ep, model='a', api_type='responses', payload=self.large))
        with self.assertRaises(main.HTTPException) as caught:
            await self.lb.on_request_start(self.ep, model='a', api_type='responses', payload=self.large)
        self.assertEqual(caught.exception.detail['error']['code'], 'recovery_prefers_small_request')
        # The small request itself still claims the trial.
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='responses', payload=self.small))

    async def test_evidence_does_not_survive_a_new_cooldown(self):
        self.lb._open(self.ep)
        with self.assertRaises(main.HTTPException):
            await self.lb.on_request_start(self.ep, model='a', api_type='responses', payload=self.small)
        self.now += 30
        # Trial fails; the circuit reopens and the previous evidence must not carry over.
        lease = await self.lb.on_request_start(self.ep, model='a', api_type='responses', payload=self.small)
        await self.lb.on_request_end(self.ep, False, lease=lease)
        self.now += 30
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='responses', payload=self.large))

    async def test_scoped_circuit_tracks_its_own_evidence(self):
        route = self.lb._route_circuit(self.ep, 'a', 'responses', create=True)
        self.lb._open(route)
        self.now += 30
        # Shared circuit is closed; only the route is recovering and saw no small input.
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='responses', payload=self.large))

    async def test_readiness_never_records_evidence(self):
        self.lb._open(self.ep)
        for _ in range(5):
            self.lb.is_available(self.ep, readiness=True, model='a', api_type='responses', payload=self.small)
        self.now += 30
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='responses', payload=self.large))

    async def test_incomplete_estimate_is_not_small_input_evidence(self):
        opaque = {'model': 'a', 'input': [{'type': 'reasoning', 'encrypted_content': 'synthetic'}]}
        self.lb._open(self.ep)
        with self.assertRaises(main.HTTPException):
            await self.lb.on_request_start(self.ep, model='a', api_type='responses', payload=opaque)
        self.now += 30
        self.assertTrue(self.lb.is_available(self.ep, model='a', api_type='responses', payload=self.large))


if __name__ == '__main__':
    unittest.main()
