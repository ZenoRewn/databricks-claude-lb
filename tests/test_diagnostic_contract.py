"""Safe correlated inference diagnostics. Synthetic fixtures only. Author: Zeno Ren."""
import asyncio
import json
import logging
import os
import subprocess
import sys
import textwrap
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
    async def test_adapter_attempt_uses_upstream_api_without_changing_ingress_api(self):
        for failure in (None, httpx.ReadTimeout('SYNTHETIC_PRIVATE_EXCEPTION')):
            with self.subTest(failure=failure is not None):
                async def app(scope, receive, send):
                    call = AsyncMock(return_value=httpx.Response(200), side_effect=failure)
                    try:
                        await telemetry.inference_call(call(), 'copilot', 'responses',
                                                       model='synthetic-model', endpoint='synthetic')
                    except httpx.ReadTimeout:
                        telemetry.note_generation('failed')
                    else:
                        telemetry.note_generation('completed')
                    await send({'type': 'http.response.start', 'status': 200, 'headers': []})
                    await send({'type': 'http.response.body', 'body': b'{}'})

                with self.assertLogs('main', level='INFO') as logs:
                    await telemetry.RequestTelemetryMiddleware(app, telemetry.RequestTelemetry())(
                        {'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions', 'headers': []},
                        AsyncMock(), AsyncMock())
                attempts = [r for r in logs.records if r.kind in
                            ('lb_upstream_send_start', 'lb_upstream_send_end', 'lb_upstream_error')]
                self.assertEqual(len(attempts), 3 if failure else 2)
                self.assertEqual({r.api_type for r in attempts}, {'responses'})
                self.assertEqual(len({r.upstream_attempt_id for r in attempts}), 1)
                end = next(r for r in logs.records if r.kind == 'lb_request_end')
                self.assertEqual(end.api_type, 'chat')
                self.assertEqual({r.lb_request_id for r in attempts}, {end.lb_request_id})

    async def test_local_admission_rejection_is_not_attributed_to_upstream(self):
        async def app(scope, receive, send):
            scope.setdefault('state', {})['lb_overloaded'] = True
            await send({'type': 'http.response.start', 'status': 503, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'{}'})
        with self.assertLogs('main', level='INFO') as logs:
            await telemetry.RequestTelemetryMiddleware(app, telemetry.RequestTelemetry())(
                {'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []}, AsyncMock(), AsyncMock())
        end = next(r for r in logs.records if getattr(r, 'kind', '') == 'lb_request_end')
        self.assertEqual((end.failure_reason, end.error_origin), ('local_overload', 'local'))

    async def test_end_record_has_start_time_size_bucket_and_only_reported_model(self):
        async def app(scope, receive, send):
            telemetry.note_request_context('synthetic-alias', body_size=2_000_000)
            telemetry.note_json_result({'id': 'synthetic', 'status': 'completed', 'output': [],
                                       'model': 'synthetic-upstream-version'}, 'responses')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'{}'})
        with self.assertLogs('main', level='INFO') as logs:
            await telemetry.RequestTelemetryMiddleware(app, telemetry.RequestTelemetry())(
                {'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []}, AsyncMock(), AsyncMock())
        end = next(r for r in logs.records if getattr(r, 'kind', '') == 'lb_request_end')
        self.assertGreater(end.started_at_unix, 0)
        self.assertEqual(end.body_size_bucket, 'large')
        self.assertEqual(end.requested_model, 'synthetic-alias')
        self.assertEqual(end.resolved_model, 'synthetic-upstream-version')

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
        generic = [r for r in logs if getattr(r, 'kind', '') == 'lb_stream_end']
        self.assertEqual(len(generic), 1)
        self.assertEqual(generic[0].lb_request_id, end.lb_request_id)
        self.assertFalse(generic[0].terminal_seen)

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
                                               headers={'Authorization': 'Bearer synthetic', 'X-Request-Id': 'client-auxiliary-id'})
            self.assertEqual(result.status_code, 400)
            self.assertEqual(result.json()['detail']['error']['lb_request_id'], result.headers['x-lb-request-id'])
            self.assertEqual(result.headers['x-request-id'], 'client-auxiliary-id')
            for record in logs.records:
                self.assertNotIn(marker, record.getMessage())
                self.assertNotIn(marker, main._JsonLogFormatter().format(record))
        finally:
            await proxy.close()

    async def test_untrusted_external_id_is_not_echoed_unbounded(self):
        from types import SimpleNamespace
        route = AsyncMock(return_value=main.JSONResponse({'type': 'message', 'stop_reason': 'end_turn'}))
        proxy = SimpleNamespace(verify_api_key=lambda key: key == 'synthetic', proxy_request=route)
        with patch.object(main, 'proxy', proxy):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local') as client:
                response = await client.post('/v1/messages', json={'model': 'claude-opus-5', 'messages': []},
                    headers={'Authorization': 'Bearer synthetic', 'X-Request-Id': 'x' * 10000})
        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(response.headers['x-request-id']), 128)
        self.assertNotEqual(response.headers['x-request-id'], 'x' * 10000)


class DiagnosticSinkTests(unittest.TestCase):
    def test_text_sink_preserves_legacy_copilot_summary_markers_and_safe_fields(self):
        from safe_diagnostics import DiagnosticFilter, DiagnosticTextFormatter
        for kind, label in (('copilot_stream_end', 'stream_end'), ('copilot_request_end', 'request_end')):
            with self.subTest(kind=kind):
                record = logging.LogRecord('main', logging.INFO, __file__, 1, 'SYNTHETIC_PRIVATE_BODY', (), None)
                record.__dict__.update(kind=kind, lb_request_id='synthetic-qa-request', outcome='failed')
                self.assertTrue(DiagnosticFilter().filter(record))
                rendered = DiagnosticTextFormatter('%(levelname)s:%(name)s:%(message)s').format(record)
                self.assertIn('[Copilot ' + label + ']', rendered)
                self.assertIn('lb_request_id=synthetic-qa-request', rendered)
                self.assertIn('outcome=failed', rendered)
                self.assertNotIn('SYNTHETIC_PRIVATE_BODY', rendered)

    def test_default_sink_retains_safe_fields_in_text_and_json(self):
        script = textwrap.dedent('''
            import logging
            import main
            from request_telemetry import log_event
            log_event({'kind': 'lb_request_end', 'lb_request_id': 'synthetic-qa-request',
                       'outcome': 'failed', 'failure_reason': 'read_timeout',
                       'provider': 'copilot', 'api_type': 'responses',
                       'prompt': 'SYNTHETIC_PRIVATE_BODY',
                       'extra': {'authorization': 'SYNTHETIC_PRIVATE_BODY'}})
            if not logging.getLogger().handlers[0].drain(timeout=2):
                raise RuntimeError('Synthetic diagnostic did not drain')
        ''')
        for format_name in ('text', 'json'):
            with self.subTest(format=format_name):
                env = {**os.environ, 'LOG_FORMAT': format_name, 'LOG_LEVEL': 'INFO',
                       'LB_MODEL_CAPABILITIES_PATH': '', 'PYTHONDONTWRITEBYTECODE': '1'}
                result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                        text=True, env=env, timeout=15, check=True)
                event = next(line for line in result.stderr.splitlines() if 'lb_request_end' in line)
                self.assertIn('synthetic-qa-request', event)
                self.assertNotIn('SYNTHETIC_PRIVATE_BODY', result.stderr)
                payload = json.loads(event.removeprefix('INFO:main:') if format_name == 'text' else event)
                self.assertEqual(payload['outcome'], 'failed')
                self.assertEqual(payload['failure_reason'], 'read_timeout')
                self.assertEqual(payload['provider'], 'copilot')
                self.assertEqual(payload['api_type'], 'responses')

    def test_interpreter_shutdown_does_not_wait_for_stuck_default_sink(self):
        script = textwrap.dedent('''
            import logging
            import threading
            from safe_diagnostics import DiagnosticStreamHandler, default_handler
            entered = threading.Event()
            class StuckStream:
                def write(self, text):
                    entered.set()
                    threading.Event().wait()
                def flush(self):
                    pass
            handler = default_handler(DiagnosticStreamHandler(StuckStream()))
            logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
            logging.info('synthetic')
            if not entered.wait(1):
                raise RuntimeError('Synthetic sink did not receive event')
            handler.close()
            print('bounded close returned', flush=True)
        ''')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=3, check=True,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertIn('bounded close returned', result.stdout)

    def test_timestamp_uses_event_creation_instead_of_delayed_sink_time(self):
        record = logging.LogRecord('synthetic', logging.INFO, __file__, 1, 'safe', (), None)
        record.created = 1.0
        self.assertEqual(json.loads(main._JsonLogFormatter().format(record))['ts'], '1970-01-01T00:00:01.000Z')

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
