"""Circuit history and volatile usage boundaries, without production access. Author: Zeno Ren."""
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import main
from usage_store import JsonUsageStore


class CircuitEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_transition_events_are_mutation_bound_and_readiness_stays_pure(self):
        ep = main.WorkspaceEndpoint('synthetic', 'https://fixture.invalid', 'synthetic')
        lb = main.LoadBalancer([ep], circuit_breaker_threshold=1, circuit_breaker_timeout=10)
        now = [1000.0]
        with patch.object(main.time, 'monotonic', side_effect=lambda: now[0]), self.assertLogs('main', level='INFO') as logs:
            lease = await lb.on_request_start(ep)
            await lb.on_request_end(ep, False, lease=lease)
            now[0] += 11
            for _ in range(10):
                self.assertTrue(lb.is_available(ep))
            trial = await lb.on_request_start(ep)
            await lb.on_request_end(ep, True, lease=trial)
        events = [r for r in logs.records if getattr(r, 'kind', '') == 'lb_circuit_transition']
        self.assertEqual([r.transition_reason for r in events], ['failure_threshold', 'trial_admitted', 'trial_succeeded'])
        self.assertEqual([r.circuit_state for r in events], ['OPEN', 'HALF_OPEN', 'CLOSED'])
        self.assertEqual(ep.total_errors, 1)
        self.assertEqual(ep.completed_requests, 1)


class UsageEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_v3_reports_accepted_acknowledged_and_oldest_pending_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonUsageStore(directory)
            store.record('synthetic', 11, 2)
            before = store.extended_persistence_stats()
            self.assertEqual(before['accepted_events_total'], 1)
            self.assertEqual(before['persisted_events_total'], 0)
            self.assertEqual(before['volatile_buffer'], 1)
            self.assertGreaterEqual(before['oldest_pending_age_seconds'], 0)
            await store._flush()
            after = store.extended_persistence_stats()
            self.assertEqual(after['persisted_events_total'], 1)
            self.assertEqual(after['oldest_pending_age_seconds'], 0)
            from operations.metrics_contract import parse_exposition
            parse_exposition(store.render_extended_metrics(), schema_version='lb-metrics-v3')

    async def test_unknown_pending_timestamp_is_not_zero_age(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonUsageStore(directory); store.record('synthetic', 1, 1)
            del store._buffer[0]['recorded_at_unix']
            self.assertIsNone(store.extended_persistence_stats()['oldest_pending_age_seconds'])
            self.assertIn('NaN', store.render_extended_metrics())

    async def test_hard_process_exit_before_flush_exposes_the_documented_loss_boundary(self):
        await self.crash_case(flush=False, expected=0)

    async def test_hard_process_exit_after_acknowledged_flush_preserves_the_ledger_total(self):
        await self.crash_case(flush=True, expected=11)

    async def crash_case(self, *, flush, expected):
        with tempfile.TemporaryDirectory() as directory:
            code = '''import asyncio,os,sys
from usage_store import JsonUsageStore
async def run():
    store=JsonUsageStore(sys.argv[1])
    store.record('synthetic',11,2)
    if sys.argv[2]=='flush': await store._flush()
    os._exit(73)
asyncio.run(run())
'''
            result = subprocess.run([sys.executable, '-c', code, directory, 'flush' if flush else 'pending'],
                                    cwd=Path(main.__file__).parent, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 73)
            recovered = JsonUsageStore(directory)
            data = await recovered._load_day(date.today())
            self.assertEqual((data.get('models', {}).get('synthetic') or {}).get('input_tokens', 0), expected)
