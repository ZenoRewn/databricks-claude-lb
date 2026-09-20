"""Shared protocol outcomes must agree with settlement and persisted usage."""
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import main
from usage_store import UsageDataStore
from request_telemetry import CURRENT, RequestRecord, RequestTelemetry


class ResultConsistencyTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, provider, api, payload, *, stream=False, content_type=None, status=200, headers=None):
        if provider == 'databricks':
            ep = main.WorkspaceEndpoint('fixture', 'https://fixture.invalid', 'synthetic')
            proxy = main.ClaudeProxy(main.LoadBalancer([ep]), 'synthetic')
            call = proxy.proxy_request
        elif provider == 'azure':
            ep = main.AzureOpenAIEndpoint('fixture', 'https://fixture.invalid', 'synthetic', deployments=['gpt-test'])
            proxy = main.AzureOpenAIProxy(main.LoadBalancer([ep]), 'synthetic')
            call = proxy.proxy_responses if api == 'responses' else proxy.proxy_chat_completions
        else:
            ep = main.CopilotEndpoint('fixture', 'synthetic')
            ep.session_base_url = 'https://fixture.invalid'
            proxy = main.CopilotProxy(main.LoadBalancer([ep]), 'synthetic')
            proxy._build_headers = AsyncMock(return_value={})
            call = proxy.proxy_responses if api == 'responses' else proxy.proxy_chat_completions
        hooks = proxy.client.event_hooks
        await proxy.client.aclose()
        calls = []
        async def upstream(request):
            calls.append(request)
            kwargs = {'content': payload} if isinstance(payload, bytes) else {'json': payload}
            return httpx.Response(status, headers={**(headers or {}), **({'content-type': content_type} if content_type else {})}, **kwargs)
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), event_hooks=hooks, trust_env=False)
        store = UsageDataStore()
        record = RequestRecord(api, RequestTelemetry())
        token = CURRENT.set(record)
        try:
            with patch.object(main, 'usage_store', store):
                try:
                    result = await call({'model': 'claude-opus-5' if provider == 'databricks' else 'gpt-test', 'messages': [], 'input': []}, stream=stream)
                    wire = b''.join([c.encode() if isinstance(c, str) else c async for c in result.body_iterator]) if stream else result.body
                    code, response_headers = result.status_code, dict(result.headers)
                except main.HTTPException as exc:
                    wire = json.dumps(exc.detail).encode()
                    code, response_headers = exc.status_code, exc.headers or {}
            return {'status': code, 'headers': response_headers, 'wire': wire, 'ep': ep,
                    'events': store._buffer, 'record': record, 'sends': len(calls)}
        finally:
            CURRENT.reset(token)
            await proxy.close()

    async def test_error_json_200_is_not_completed_or_zero_token_success(self):
        for provider, api in [('databricks','messages'), ('azure','responses'), ('azure','chat'), ('copilot','responses'), ('copilot','chat')]:
            with self.subTest(provider=provider, api=api):
                r = await self.drive(provider, api, {'error': {'message': 'synthetic failure'}})
                self.assertEqual(r['status'], 502)
                self.assertEqual(r['ep'].completed_requests, 0)
                self.assertEqual(r['events'], [])
                self.assertEqual(r['sends'], 1)
                self.assertEqual(r['ep'].active_requests, 0)
                self.assertEqual(r['headers'].get('X-Should-Retry'), 'false')

    async def test_failure_with_valid_usage_is_recorded_as_failure_once(self):
        for provider, api in [('databricks','messages'), ('azure','responses'), ('copilot','responses')]:
            with self.subTest(provider=provider):
                r = await self.drive(provider, api, {'error': {'message': 'failed'}, 'usage': {'input_tokens': 7, 'output_tokens': 2}})
                self.assertEqual(r['status'], 502)
                self.assertEqual(len(r['events']), 1)
                self.assertEqual(r['events'][0]['input_tokens'], 7)
                self.assertEqual(r['events'][0]['generation_outcome'], 'failed')
                self.assertTrue(r['events'][0]['is_error'])
                self.assertEqual(r['ep'].successful_requests, 0)

    async def test_html_terminal_cannot_be_forwarded_or_counted_successful(self):
        for provider, api, body in [
            ('databricks','messages',b'event: message_stop\ndata: {"type":"message_stop"}\n\n'),
            ('azure','responses',b'data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}\n\n'),
            ('copilot','chat',b'data: [DONE]\n\n'),
        ]:
            with self.subTest(provider=provider):
                r = await self.drive(provider, api, body, stream=True, content_type='text/html')
                self.assertIn(b'upstream_html_error', r['wire'])
                self.assertNotIn(b'message_stop', r['wire'])
                self.assertNotIn(b'response.completed', r['wire'])
                self.assertEqual(r['ep'].completed_requests, 0)
                self.assertEqual(r['events'], [])
                self.assertEqual(r['sends'], 1)

    async def test_stream_error_preserves_status_code_and_retry_after(self):
        for provider, api in [('databricks','messages'), ('azure','responses'), ('copilot','chat')]:
            with self.subTest(provider=provider):
                r = await self.drive(provider, api, {'error_code':'TEMPORARILY_UNAVAILABLE','message':'capacity'}, stream=True, status=503, headers={'Retry-After':'37'})
                events = [json.loads(l[5:]) for l in r['wire'].decode().splitlines() if l.startswith('data:') and l[5:].strip() != '[DONE]']
                error = events[-1].get('error') or events[-1]['response']['error']
                self.assertEqual(error['upstream_status'], 503)
                self.assertEqual(error['upstream_code'], 'TEMPORARILY_UNAVAILABLE')
                self.assertEqual(error['retry_after'], '37')
                self.assertFalse(error['retryable'])
                self.assertEqual(r['sends'], 1)

    async def test_unknown_empty_stream_has_both_retry_suppression_signals(self):
        for provider, api in [('databricks','messages'),('azure','responses'),('azure','chat'),('copilot','responses'),('copilot','chat')]:
            with self.subTest(provider=provider, api=api):
                r=await self.drive(provider,api,b'',stream=True,content_type='text/event-stream')
                self.assertEqual(r['headers'].get('x-should-retry'),'false')
                events=[json.loads(line[5:]) for line in r['wire'].decode().splitlines()
                        if line.startswith('data:') and line[5:].strip()!='[DONE]']
                error=events[-1].get('error') or events[-1]['response']['error']
                self.assertFalse(error['retryable'])
                self.assertEqual(error['execution_certainty'],'unknown')
                self.assertEqual(r['sends'],1)
                self.assertEqual(r['ep'].completed_requests,0)
                self.assertEqual(r['ep'].active_requests,0)
                self.assertEqual(r['events'],[])

    async def test_context_rejection_is_neutral_in_both_modes(self):
        for stream in (False, True):
            r = await self.drive('databricks','messages', {'error': {'code':'context_length_exceeded','message':'input exceeds context window'}}, stream=stream, status=400)
            self.assertIn(b'context_window_exceeded', r['wire'])
            self.assertEqual(r['ep'].total_errors, 0)
            self.assertEqual(r['ep'].neutral_requests, 1)

    async def test_failed_stream_retains_observed_usage(self):
        raw = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":9,"output_tokens":0}}}\n\n'
               b'event: error\ndata: {"type":"error","error":{"code":"server_error","message":"failed"}}\n\n')
        r = await self.drive('databricks','messages',raw,stream=True,content_type='text/event-stream')
        self.assertEqual(r['ep'].completed_requests, 0)
        self.assertEqual(len(r['events']),1)
        self.assertEqual(r['events'][0]['input_tokens'],9)
        self.assertEqual(r['events'][0]['generation_outcome'],'failed')

    async def test_explicit_http_failure_with_reported_usage_is_not_lost(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                r=await self.drive('databricks','messages',{'error':{'code':'server_error','message':'failed'},
                                                         'usage':{'input_tokens':3,'output_tokens':1}},status=503,stream=stream)
                self.assertEqual(len(r['events']),1)
                self.assertEqual(r['events'][0]['generation_outcome'],'failed')
                self.assertEqual(r['ep'].completed_requests,0)

    async def test_invalid_json_cannot_be_accounted_before_serialization(self):
        for raw in (b'{"type":"message","stop_reason":"end_turn","metadata":{"x":NaN}}',
                    b'{"type":"message","stop_reason":"end_turn","metadata":{"x":"\\ud800"}}',
                    b'{"type":"message","stop_reason":"end_turn","error":{"message":"failed"},"error":null}'):
            with self.subTest(raw=raw):
                r=await self.drive('databricks','messages',raw,content_type='application/json')
                self.assertEqual(r['status'],502)
                self.assertEqual(r['ep'].successful_requests,0)
                self.assertEqual(r['ep'].completed_requests,0)
                self.assertEqual(r['events'],[])

    async def test_streaming_json_error_200_is_structured_before_forwarding(self):
        for provider,api in [('databricks','messages'),('azure','responses'),('copilot','chat')]:
            with self.subTest(provider=provider):
                r=await self.drive(provider,api,{'error':{'code':'server_error','message':'synthetic failure'}},stream=True)
                self.assertTrue(r['wire'].startswith((b'event: error',b'data: ')))
                self.assertIn(b'upstream_status',r['wire'])
                self.assertEqual(r['ep'].completed_requests,0)
                self.assertEqual(r['sends'],1)

    async def test_incomplete_opaque_repair_does_not_claim_completed_recovery(self):
        from test_copilot_opaque_state_recovery import _make_proxy,stateful_body,ORPHAN_401
        proxy,lb,ep,forced=_make_proxy(threshold=99)
        request=httpx.Request('POST','https://fixture.invalid/responses')
        proxy.client.post=AsyncMock(side_effect=[httpx.Response(401,request=request,text=ORPHAN_401,headers={'content-type':'application/json'}),
            httpx.Response(200,request=request,json={'id':'r','status':'incomplete','output':[],
                'incomplete_details':{'reason':'max_output_tokens'},'usage':{'input_tokens':2,'output_tokens':1}})])
        with patch.object(main,'usage_store',None):result=await proxy.proxy_responses(stateful_body())
        self.assertEqual(result.status_code,200)
        self.assertEqual(proxy.client.post.await_count,2)
        self.assertEqual(proxy.opaque_state_recovery.get('succeeded',0),0)
        self.assertEqual(ep.completed_requests,0)
