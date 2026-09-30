"""Read-only behavioral assessment using synthetic inputs. Author: Zeno Ren.

No service startup, real upstream, credentials, database, or cluster operations.
These probes document existing behavior; they are not implementation acceptance.
Run from the repository root with: python3 docs/reviews/2026-09-30-openclaw-assessment/assessment_probes.py
"""
import asyncio
import json
import logging
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import httpx
import main
import request_telemetry as telemetry


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class SyntheticStream(httpx.AsyncByteStream):
    def __init__(self, error=None):
        self.error = error
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"type":"response.output_text.delta","delta":"synthetic"}\n\n'
        await asyncio.sleep(0)
        if self.error:
            raise self.error
        yield b'data: {"type":"response.completed","response":{"id":"synthetic"}}\n\n'

    async def aclose(self):
        self.closed = True


async def stream_case(capture, *, repair=False):
    start_record = len(capture.records)
    ep = main.CopilotEndpoint('synthetic', '')
    proxy = main.CopilotProxy(main.LoadBalancer([ep]), '')
    await proxy.client.aclose()
    calls = []
    stream = SyntheticStream(None if repair else httpx.RemoteProtocolError('synthetic protocol failure'))

    async def upstream(request):
        calls.append(request)
        if repair and len(calls) == 1:
            return httpx.Response(401, json={'error': {'code': 'unauthorized', 'message': 'synthetic'}})
        return httpx.Response(200, stream=stream, headers={'content-type': 'text/event-stream'})

    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False)
    proxy.get_session_token = AsyncMock(return_value='synthetic-token')
    proxy._build_headers = AsyncMock(return_value={})
    proxy._probe_upstream_connect = AsyncMock(return_value={'ok': True})
    wire = []

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
        await telemetry.RequestTelemetryMiddleware(app, metrics)(
            {'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []},
            AsyncMock(), AsyncMock())
    finally:
        await proxy.close()
    records = capture.records[start_record:]
    ends = [r for r in records if getattr(r, 'kind', None) == 'lb_request_end']
    stream_ends = [r for r in records if getattr(r, 'kind', None) == 'copilot_stream_end']
    if len(ends) != 1 or len(stream_ends) != 1:
        raise RuntimeError('Expected exactly one request and stream summary')
    end = ends[0]
    return {'outcome': end.outcome, 'failure_reason': end.failure_reason,
            'actual_mock_sends': len(calls), 'response_closed': stream.closed,
            'endpoint_active_after': ep.active_requests,
            'request_end_count': len(ends),
            'stream_log_has_lb_request_id': hasattr(stream_ends[0], 'lb_request_id'),
            'stream_log_has_upstream_attempt_id': hasattr(stream_ends[0], 'upstream_attempt_id'),
            'emitted_network_failure': b'upstream_network_error' in b''.join(wire)}


async def log_body_case(capture):
    marker = 'SYNTHETIC_PRIVATE_BODY_CANARY_20260930'
    start_record = len(capture.records)
    ep = main.WorkspaceEndpoint('synthetic', 'https://fixture.invalid', 'synthetic')
    proxy = main.ClaudeProxy(main.LoadBalancer([ep]), 'synthetic')
    await proxy.client.aclose()
    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(400, json={'error': {'code': 'invalid_request_error',
                                                         'message': marker}})), trust_env=False)
    try:
        with patch.object(main, 'proxy', proxy):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app),
                                         base_url='http://local', trust_env=False) as client:
                response = await client.post('/v1/messages',
                    json={'model': 'claude-opus-5', 'messages': [], 'stream': False},
                    headers={'Authorization': 'Bearer synthetic'})
        return {'http_status': response.status_code,
                'synthetic_error_body_canary_found_in_log': any(
                    marker in r.getMessage() for r in capture.records[start_record:])}
    finally:
        await proxy.close()


async def adapter_case():
    route = AsyncMock(return_value=main.JSONResponse({'id': 'synthetic', 'status': 'completed',
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'unconstrained'}]}]}))
    fake_auth = SimpleNamespace(verify_api_key=lambda key: key == 'synthetic')
    with patch.object(main, 'proxy', None), patch.object(main, 'azure_proxy', None), \
            patch.object(main, 'copilot_proxy', fake_auth), patch.object(main, '_route_openai_responses', route):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app),
                                     base_url='http://local', trust_env=False) as client:
            response = await client.post('/v1/chat/completions', json={
                'model': 'gpt-5.6-luna', 'messages': [{'role': 'user', 'content': 'synthetic'}],
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': 'synthetic', 'strict': True, 'schema': {'type': 'object'}}}},
                headers={'Authorization': 'Bearer synthetic', 'X-LB-Strict-Parameters': 'true'})
    payload = route.call_args.args[0] if route.call_args else {}
    return {'http_status': response.status_code, 'upstream_route_called': route.await_count,
            'policy_response_header': response.headers.get('x-lb-parameter-policy'),
            'drop_response_header': response.headers.get('x-lb-dropped-parameters'),
            'output_schema_forwarded': 'response_format' in payload or 'text' in payload,
            'upstream_stream': payload.get('stream')}


async def token_case():
    small = {'messages': [{'role': 'user', 'content': 'synthetic'}]}
    large = {**small, 'system': 'x' * 10000,
             'tools': [{'name': 'synthetic', 'description': 'y' * 10000,
                        'input_schema': {'type': 'object'}}]}
    results = [await main.count_tokens(SimpleNamespace(json=AsyncMock(return_value=body)))
               for body in (small, large)]
    return {'base_estimate': results[0]['input_tokens'], 'expanded_estimate': results[1]['input_tokens'],
            'system_chars_added': 10000, 'tool_description_chars_added': 10000,
            'estimates_equal': results[0] == results[1]}


def log_sink_case():
    class Slow(logging.Handler):
        def emit(self, record):
            time.sleep(.03)
    with patch.object(main.logger, 'handlers', [Slow()]):
        start = time.monotonic()
        telemetry.log_event({'kind': 'synthetic_assessment'})
        elapsed = time.monotonic() - start
    return {'injected_sink_delay_seconds': .03, 'log_call_seconds': round(elapsed, 6),
            'note': 'Synthetic blocking demonstration, not a production latency benchmark'}


async def assess():
    capture = Capture()
    with patch.object(main.logger, 'handlers', [capture]), patch.object(main.logger, 'propagate', False), \
            patch.object(main.logger, 'level', logging.INFO), patch.object(main, 'usage_store', None), \
            patch.object(socket.socket, 'connect', side_effect=RuntimeError('Assessment forbids network connects')):
        result = {'author': 'Zeno Ren', 'evidence_scope': 'local_synthetic_only',
                  'runtime': sys.version.split()[0]}
        result['partial_stream_protocol_error'] = await stream_case(capture)
        result['auth_repair_then_completed_stream'] = await stream_case(capture, repair=True)
        result['error_logging'] = await log_body_case(capture)
        result['strict_adapter_json_schema'] = await adapter_case()
        result['count_tokens_estimate'] = await token_case()
        result['log_sink_blocking'] = log_sink_case()
        exposition = telemetry.RequestTelemetry().render()
        result['fresh_send_result_series'] = sum(
            line.startswith('lb_upstream_send_finished_total{') for line in exposition.splitlines())
        images = {'input': [{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': 'https://fixture.invalid/one'},
            {'type': 'input_image', 'image_url': 'https://fixture.invalid/two'}]}]}
        removed = main.trim_excess_images(images, max_count=1)
        result['existing_image_trim'] = {'removed_images': removed,
            'first_image_replaced_by_text': images['input'][0]['content'][0]['type'] == 'input_text'}
        return result


if __name__ == '__main__':
    print(json.dumps(asyncio.run(assess()), ensure_ascii=False, indent=2))
