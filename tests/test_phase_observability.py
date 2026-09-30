"""Measured phase boundaries and opt-in metrics schema. Author: Zeno Ren."""
import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import main
import request_budget
import request_telemetry as telemetry
from operations.metrics_contract import parse_exposition


class TimingTests(unittest.TestCase):
    def test_overlapping_cleanup_owners_measure_union_even_if_first_exits_first(self):
        from request_timing import Timeline
        now = [0.0]
        timeline = Timeline(clock=lambda: now[0])
        first, second = timeline.measure('cleanup'), timeline.measure('cleanup')
        first.__enter__()
        now[0] = 1
        second.__enter__()
        now[0] = 2
        first.__exit__(None, None, None)
        now[0] = 4
        second.__exit__(None, None, None)
        self.assertEqual(timeline.snapshot()['phase_seconds']['cleanup'], 4)

    def test_nested_cleanup_is_not_double_counted_and_unknown_phases_stay_null(self):
        from request_timing import Timeline
        now = [10.0]
        timeline = Timeline(clock=lambda: now[0])
        with timeline.measure('cleanup'):
            now[0] = 11
            with timeline.measure('cleanup'):
                now[0] = 13
            now[0] = 14
        values = timeline.snapshot()['phase_seconds']
        self.assertEqual(values['cleanup'], 4)
        for stage in ('dns', 'tcp', 'tls', 'pool', 'first_content_from_ingress'):
            self.assertIsNone(values[stage])

    def test_attempt_header_time_is_distinct_from_buffered_send_duration(self):
        from request_timing import Timeline
        now = [0.0]
        timeline = Timeline(clock=lambda: now[0])
        attempt = timeline.begin_attempt(180)
        now[0] = 2
        timeline.headers_received()
        now[0] = 8
        timeline.end_send(attempt)
        self.assertEqual(attempt.header_seconds, 2)
        self.assertEqual(attempt.send_seconds, 8)
        self.assertEqual(timeline.snapshot()['phase_seconds']['send_to_headers'], 2)

    def test_send_result_counters_have_finite_zero_baselines(self):
        series = parse_exposition(telemetry.RequestTelemetry().render())
        results = [s for s in series if s['name'] == 'lb_upstream_send_finished_total']
        self.assertTrue(results)
        self.assertTrue(all(s['value'] == 0 for s in results))
        self.assertTrue(any(s['labels'] == {'provider': 'copilot', 'api_type': 'responses', 'result': 'startup_timeout'} for s in results))

    def test_budget_overrides_are_exact_route_scoped_and_validate_before_use(self):
        rules = request_budget.parse_startup_overrides([
            {'provider': 'copilot', 'api_type': 'responses', 'model': 'synthetic-model', 'seconds': 240}])
        with patch.object(request_budget, 'STARTUP_OVERRIDES', rules):
            self.assertEqual(request_budget.startup_seconds('copilot', 'responses', 'synthetic-model'), 240)
            self.assertEqual(request_budget.startup_seconds('copilot', 'chat', 'synthetic-model'), request_budget.STARTUP_TIMEOUT)
            self.assertEqual(request_budget.startup_seconds('databricks', 'messages', 'synthetic-model'), request_budget.STARTUP_TIMEOUT)
        for seconds in (0, -1, True, float('nan'), float('inf'), '240'):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                request_budget.parse_startup_overrides([
                    {'provider': 'copilot', 'api_type': 'responses', 'model': 'synthetic-model', 'seconds': seconds}])


class PhaseIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_does_not_mark_content_or_valid_event(self):
        from request_timing import CURRENT, Timeline
        timeline = Timeline()
        token = CURRENT.set(timeline)
        try:
            observation = main._SSEObservation('responses')
            observation.observe(b': keep-alive\n\n')
            observation.observe(b'data: not-json\n\n')
            values = timeline.snapshot()['phase_seconds']
            self.assertIsNone(values['first_event_from_ingress'])
            self.assertIsNone(values['first_content_from_ingress'])
            observation.observe(b'data: {"type":"response.output_text.delta","delta":"synthetic"}\n\n')
            self.assertIsNotNone(timeline.snapshot()['phase_seconds']['first_event_from_ingress'])
            self.assertTrue(observation.frame_has_content)
        finally:
            CURRENT.reset(token)

    async def test_v2_default_stays_compatible_and_v3_is_explicit(self):
        from operations.metrics_contract import CONTRACT_V3
        with patch.object(main, 'proxy', None), patch.object(main, 'azure_proxy', None), \
                patch.object(main, 'copilot_proxy', None), patch.object(main, 'usage_store', None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local') as client:
                legacy = await client.get('/metrics')
                extended = await client.get('/metrics?schema=lb-metrics-v3')
                invalid = await client.get('/metrics?schema=unknown')
        self.assertEqual(legacy.headers['x-lb-metrics-schema'], 'lb-metrics-v2')
        self.assertNotIn('lb_request_phase_duration_seconds', legacy.text)
        parse_exposition(legacy.text)
        self.assertEqual(extended.headers['x-lb-metrics-schema'], 'lb-metrics-v3')
        self.assertIn('lb_diagnostic_dropped_events_total', extended.text)
        self.assertIn('lb_request_phase_duration_seconds', extended.text)
        parse_exposition(extended.text, schema_version='lb-metrics-v3')
        self.assertEqual(invalid.status_code, 400)
        saved = json.loads((Path(__file__).parents[1] / 'operations/metrics-contract-v3.json').read_text())
        self.assertEqual(saved, CONTRACT_V3)

    async def test_deadline_records_observed_stage_without_changing_cancellation(self):
        from request_timing import observed_phase
        @observed_phase('body_read')
        async def waiting():
            await asyncio.Event().wait()
        async def app(scope, receive, send):
            await waiting()
        wrapped = telemetry.RequestTelemetryMiddleware(request_budget.RequestBudgetMiddleware(
            app, timeout=.01, error_frame_factory=main._sse_terminal_error), telemetry.RequestTelemetry())
        with self.assertLogs('main', level='INFO') as logs:
            await wrapped({'type': 'http', 'path': '/v1/responses', 'method': 'POST', 'headers': []}, AsyncMock(), AsyncMock())
        end = next(r for r in logs.records if getattr(r, 'kind', '') == 'lb_request_end')
        self.assertEqual(end.outcome, 'deadline_exceeded')
        self.assertEqual(end.deadline_phase, 'body_read')
        self.assertGreater(end.phase_seconds['body_read'], 0)
        self.assertIsNone(end.phase_seconds['dns'])
