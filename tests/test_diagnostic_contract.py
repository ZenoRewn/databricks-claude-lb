"""Safe correlated inference diagnostics. Synthetic fixtures only. Author: Zeno Ren."""
import asyncio
import json
import logging
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import main
import request_telemetry as telemetry


class Stream(httpx.AsyncByteStream):
    def __init__(self, fail=False):
        self.fail, self.closed = fail, False

    async def __aiter__(self):
        yield b'data: {"type":"response.output_text.delta","delta":"synthetic"}\n\n'
        await asyncio.sleep(0)
        if self.fail:
            raise httpx.RemoteProtocolError('SYNTHETIC_PRIVATE_EXCEPTION')
        yield b'data: {"type":"response.completed","response":{"id":"synthetic"}}\n\n'

    async def aclose(self):
        self.closed = True


class DiagnosticLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, *, repair=False):
        ep = main.CopilotEndpoint('synthetic', '')
        proxy = main.CopilotProxy(main.LoadBalancer([ep]), '')
        await proxy.client.aclose()
        calls, wire = [], []
        stream = Stream(fail=not repair)

        async def upstream(request):
            calls.append(request)
            if repair and len(calls) == 1:
                return httpx.Response(401, json={'error': {'code': 'unauthorized', 'message': 'synthetic'}})
            return httpx.Response(200, stream=stream, headers={'content-type': 'text/event-stream'})

        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False)
        proxy.get_session_token = AsyncMock(return_value='synthetic-token')
        proxy._build_headers = AsyncMock(return_value={})
        proxy._probe_upstream_connect = AsyncMock(return_value={'ok': True})

        async def app(scope, receive, send):
            await proxy.load_balancer.on_request_start(ep)
            response = await proxy._stream_response(ep, 'https://fixture.invalid/responses', {}, {},
                                                    'synthetic-model', 'responses', time.time(),
                                                    request_id='synthetic-client-id')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            async for part in response.body_iterator:
                wire.append(part)
                await send({'type': 'http.response.body', 'body': part, 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})

        metrics = telemetry.RequestTelemetry()
        try:
            with self.assertLogs('main', level='INFO') as logs, patch.object(main, 'usage_store', None):
                await telemetry.RequestTelemetryMiddleware(app, metrics)(
                    {'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []},
                    AsyncMock(), AsyncMock())
            return logs.records, b''.join(wire), calls, ep, stream
        finally:
            await proxy.close()

    async def test_protocol_failure_keeps_reason_and_single_send_cleanup(self):
        logs, wire, calls, ep, stream = await self.drive()
        ends = [r for r in logs if getattr(r, 'kind', '') == 'lb_request_end']
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0].outcome, 'failed')
        self.assertEqual(ends[0].failure_reason, 'transport_protocol_error')
        self.assertIn(b'upstream_network_error', wire)
        self.assertEqual(len(calls), 1)
        self.assertEqual(ep.active_requests, 0)
        self.assertTrue(stream.closed)

    async def test_successful_auth_repair_clears_final_reason_keeps_attempt_history(self):
        logs, wire, calls, ep, stream = await self.drive(repair=True)
        end = next(r for r in logs if getattr(r, 'kind', '') == 'lb_request_end')
        self.assertEqual((end.outcome, end.failure_reason), ('completed', 'none'))
        sends = [r for r in logs if getattr(r, 'kind', '') == 'lb_upstream_send_end']
        self.assertEqual([r.upstream_status for r in sends], [401, 200])
        self.assertEqual(len({r.upstream_attempt_id for r in sends}), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(ep.active_requests, 0)
        self.assertTrue(stream.closed)

    async def test_stream_and_send_share_server_identity_and_model_context(self):
        logs, _, _, _, _ = await self.drive()
        end = next(r for r in logs if getattr(r, 'kind', '') == 'lb_request_end')
        stream_end = next(r for r in logs if getattr(r, 'kind', '') == 'copilot_stream_end')
        send_end = next(r for r in logs if getattr(r, 'kind', '') == 'lb_upstream_send_end')
        self.assertEqual(stream_end.lb_request_id, end.lb_request_id)
        self.assertEqual(stream_end.upstream_attempt_id, send_end.upstream_attempt_id)
        self.assertEqual(send_end.forwarded_model, 'synthetic-model')
        self.assertEqual(send_end.provider, 'copilot')
        self.assertEqual(send_end.resolved_model, 'unknown')

    async def test_exception_text_never_reaches_logs(self):
        logs, _, _, _, _ = await self.drive()
        for record in logs:
            self.assertNotIn('SYNTHETIC_PRIVATE_EXCEPTION', main._JsonLogFormatter().format(record))
            self.assertNotIn('SYNTHETIC_PRIVATE_EXCEPTION', record.getMessage())

    async def test_upstream_error_body_never_reaches_logs(self):
        marker = 'SYNTHETIC_PRIVATE_ERROR_BODY'
        ep = main.WorkspaceEndpoint('synthetic', 'https://fixture.invalid', 'synthetic')
        proxy = main.ClaudeProxy(main.LoadBalancer([ep]), 'synthetic')
        await proxy.client.aclose()
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(400, json={'error': {'code': 'invalid_request_error', 'message': marker}})),
            trust_env=False)
        try:
            with patch.object(main, 'proxy', proxy), patch.object(main, 'usage_store', None), self.assertLogs('main', level='INFO') as logs:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local', trust_env=False) as client:
                    result = await client.post('/v1/messages', json={'model': 'claude-opus-5', 'messages': []},
                                               headers={'Authorization': 'Bearer synthetic'})
            self.assertEqual(result.status_code, 400)
            for record in logs.records:
                self.assertNotIn(marker, record.getMessage())
                self.assertNotIn(marker, main._JsonLogFormatter().format(record))
        finally:
            await proxy.close()


class DiagnosticSinkTests(unittest.TestCase):
    def test_allowlist_removes_raw_and_nested_secrets_and_rejects_injection(self):
        from safe_diagnostics import safe_fields
        cleaned = safe_fields({'kind': 'lb_upstream_error', 'error_origin': 'upstream_http',
                               'model': 'gpt\nFORGED', 'error_body': 'SYNTHETIC_SECRET',
                               'prompt': 'SYNTHETIC_SECRET', 'extra': {'authorization': 'SYNTHETIC_SECRET'},
                               'upstream_code': 'x' * 100000, 'upstream_status': 400})
        self.assertNotIn('SYNTHETIC_SECRET', json.dumps(cleaned))
        self.assertNotIn('FORGED', json.dumps(cleaned))
        self.assertLess(len(json.dumps(cleaned)), 2048)
        self.assertEqual(cleaned['upstream_status'], 400)

    def test_bounded_queue_does_not_wait_for_slow_or_failed_sink(self):
        from safe_diagnostics import BoundedLogHandler
        entered, release = threading.Event(), threading.Event()

        class Sink(logging.Handler):
            def emit(self, record):
                entered.set()
                release.wait(2)
                raise OSError('synthetic sink failure')

        handler = BoundedLogHandler(Sink(), capacity=2)
        try:
            record = logging.LogRecord('synthetic', logging.INFO, __file__, 1, 'safe', (), None)
            handler.emit(record)
            self.assertTrue(entered.wait(1))
            start = time.monotonic()
            for _ in range(10):
                handler.emit(record)
            self.assertLess(time.monotonic() - start, .2)
            self.assertEqual(handler.queue.qsize(), 2)
            self.assertEqual(handler.dropped['queue_full'], 8)
            release.set()
            self.assertTrue(handler.drain(timeout=2))
            self.assertEqual(handler.dropped['sink_error'], 3)
        finally:
            release.set()
            handler.close()
