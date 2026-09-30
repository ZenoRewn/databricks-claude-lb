"""No silent schema, tool, role, budget or image loss. Author: Zeno Ren."""
import base64
import copy
from io import BytesIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
import httpx
from PIL import Image

import main


def body(**extras):
    return {'model': 'gpt-5.6-luna', 'messages': [{'role': 'user', 'content': 'synthetic'}], **extras}


class AdapterContractTests(unittest.TestCase):
    def test_refusal_is_preserved_in_the_chat_message(self):
        response = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'refusal', 'refusal': 'synthetic refusal'}]}]}
        result = main._responses_json_to_chat_completion(response, 'synthetic')
        self.assertEqual(result['choices'][0]['message']['refusal'], 'synthetic refusal')

    def test_shared_cache_controls_are_forwarded_without_changing_retention(self):
        result = main._build_responses_payload_from_chat(body(prompt_cache_retention='24h',
            prompt_cache_options={'ttl': '30m'}, store=False))
        self.assertEqual(result['prompt_cache_retention'], '24h')
        self.assertEqual(result['prompt_cache_options'], {'ttl': '30m'})
        self.assertFalse(result['store'])

    def test_schema_reasoning_store_and_explicit_tool_choice_are_preserved(self):
        schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                  'required': ['ok'], 'additionalProperties': False}
        request = body(response_format={'type': 'json_schema', 'json_schema': {
            'name': 'synthetic', 'strict': True, 'schema': schema}}, reasoning_effort='high', store=False,
            tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}],
            tool_choice={'type': 'function', 'function': {'name': 'lookup'}}, parallel_tool_calls=False)
        before = copy.deepcopy(request)
        payload = main._build_responses_payload_from_chat(request)
        self.assertEqual(payload['text']['format'], {'type': 'json_schema', 'name': 'synthetic', 'strict': True, 'schema': schema})
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertFalse(payload['store'])
        self.assertFalse(payload['parallel_tool_calls'])
        self.assertEqual(payload['tool_choice'], {'type': 'function', 'name': 'lookup'})
        self.assertFalse(payload['tools'][0]['strict'], 'Chat non-strict default must not become Responses strict default')
        self.assertEqual(request, before)

    def test_tool_round_trip_preserves_ids_arguments_results_and_tool_definitions(self):
        request = body(messages=[
            {'role': 'developer', 'content': 'preserve instructions'},
            {'role': 'assistant', 'content': 'checking', 'tool_calls': [
                {'id': 'call_one', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"q":"synthetic"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_one', 'content': '{"ok":true}'}],
            tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}])
        result = main._build_responses_payload_from_chat(request)
        self.assertEqual(result['input'][0]['role'], 'developer')
        call = next(item for item in result['input'] if item.get('type') == 'function_call')
        output = next(item for item in result['input'] if item.get('type') == 'function_call_output')
        self.assertEqual(call['call_id'], output['call_id'])
        self.assertEqual(call['arguments'], '{"q":"synthetic"}')
        self.assertEqual(output['output'], '{"ok":true}')
        self.assertIn({'role': 'assistant', 'content': 'checking'}, result['input'])
        self.assertEqual(result['tools'][0]['name'], 'lookup')

    def test_orphan_duplicate_and_incomplete_tool_pairs_are_rejected(self):
        call = {'role': 'assistant', 'tool_calls': [
            {'id': 'call_one', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]}
        output = {'role': 'tool', 'tool_call_id': 'call_one', 'content': 'synthetic'}
        for messages in ([output], [call], [call, output, output], [call, call, output]):
            with self.subTest(messages=messages), self.assertRaises(HTTPException) as error:
                main._build_responses_payload_from_chat(body(messages=messages))
            self.assertEqual(error.exception.status_code, 400)

    def test_images_and_developer_role_keep_their_meaning(self):
        result = main._build_responses_payload_from_chat(body(messages=[
            {'role': 'developer', 'content': 'instructions'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'synthetic'},
                {'type': 'image_url', 'image_url': {'url': 'https://fixture.invalid/image', 'detail': 'high'}}]}]))
        self.assertEqual(result['input'][0]['role'], 'developer')
        self.assertEqual(result['input'][1]['content'], [
            {'type': 'input_text', 'text': 'synthetic'},
            {'type': 'input_image', 'image_url': 'https://fixture.invalid/image', 'detail': 'high'}])

    def test_unsupported_critical_parameters_and_conflicting_limits_are_not_dropped(self):
        for extras in ({'n': 2}, {'stop': ['synthetic']}, {'context_management': {}},
                       {'response_format': {'type': 'unknown'}}, {'max_tokens': 10, 'max_completion_tokens': 20},
                       {'max_completion_tokens': 1.5}, {'tools': [{'type': 'unsupported'}]}):
            with self.subTest(extras=extras), self.assertRaises(HTTPException):
                main._build_responses_payload_from_chat(body(**extras))

    def test_incomplete_buffered_generation_does_not_become_stop(self):
        payload = main._responses_json_to_chat_completion({'id': 'synthetic', 'status': 'incomplete',
            'incomplete_details': {'reason': 'max_output_tokens'}, 'output_text': 'partial'}, 'synthetic-model')
        self.assertEqual(payload['choices'][0]['finish_reason'], 'length')


class AdapterEntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_locally_echoed_model_is_not_misreported_as_upstream_resolution(self):
        import request_telemetry as telemetry
        route = AsyncMock(return_value=main.JSONResponse({'id': 'synthetic', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'synthetic'}]}]}))
        async def app(scope, receive, send):
            with patch.object(main, '_route_openai_responses', route):
                response = await main._route_chat_via_responses(body(), stream=False)
                await send({'type': 'http.response.start', 'status': response.status_code, 'headers': []})
                await send({'type': 'http.response.body', 'body': response.body})
        with self.assertLogs('main', level='INFO') as logs:
            await telemetry.RequestTelemetryMiddleware(app, telemetry.RequestTelemetry())(
                {'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions', 'headers': []}, AsyncMock(), AsyncMock())
        end = next(row for row in logs.records if getattr(row, 'kind', '') == 'lb_request_end')
        self.assertEqual(end.resolved_model, 'unknown')

    async def test_refusal_survives_buffered_sse(self):
        payload = main._responses_json_to_chat_completion({'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'synthetic refusal'}]}]}, 'synthetic')
        wire = b''.join([chunk async for chunk in main._chat_completion_sse_from_payload(payload)])
        self.assertIn(b'"refusal": "synthetic refusal"', wire)

    async def test_real_openai_sdk_reads_text_tools_usage_and_finish_reason(self):
        from openai import OpenAI
        response = {'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'checking'}]},
            {'type': 'function_call', 'call_id': 'call_one', 'name': 'lookup', 'arguments': '{}'}],
            'usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5}}
        payload = main._responses_json_to_chat_completion(response, 'gpt-5.6-luna')
        wire = b''.join([chunk async for chunk in main._chat_completion_sse_from_payload(payload, include_usage=True)])
        with OpenAI(api_key='synthetic', http_client=httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=wire, headers={'content-type': 'text/event-stream'})))) as client:
            chunks = list(client.chat.completions.create(model='gpt-5.6-luna', messages=[], stream=True))
        choices = [chunk.choices[0] for chunk in chunks if chunk.choices]
        self.assertEqual(''.join(c.delta.content or '' for c in choices), 'checking')
        call = next(c.delta.tool_calls[0] for c in choices if c.delta.tool_calls)
        self.assertEqual((call.index, call.id, call.function.arguments), (0, 'call_one', '{}'))
        self.assertEqual(choices[-1].finish_reason, 'tool_calls')
        self.assertEqual(chunks[-1].usage.total_tokens, 5)

    async def test_strict_equivalent_conversion_is_visible_and_buffered_stream_is_declared(self):
        route = AsyncMock(return_value=main.JSONResponse({'id': 'synthetic', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"ok":true}'}]}]}))
        with patch.object(main, 'proxy', None), patch.object(main, 'azure_proxy', None), \
                patch.object(main, 'copilot_proxy', SimpleNamespace(verify_api_key=lambda k: k == 'synthetic')), \
                patch.object(main, '_route_openai_responses', route):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local') as client:
                response = await client.post('/v1/chat/completions', json=body(stream=True,
                    response_format={'type': 'json_object'}), headers={
                    'Authorization': 'Bearer synthetic', 'X-LB-Strict-Parameters': 'true'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(route.call_args.args[0]['text']['format'], {'type': 'json_object'})
        self.assertIn('response_format', response.headers['x-lb-transformed-parameters'])
        self.assertEqual(response.headers['x-lb-stream-mode'], 'buffered-adapter')
        self.assertFalse(route.call_args.kwargs['stream'])
        self.assertEqual(route.await_count, 1)

    async def test_incomplete_stream_keeps_length_terminal(self):
        payload = main._responses_json_to_chat_completion({'status': 'incomplete',
            'incomplete_details': {'reason': 'max_output_tokens'}, 'output_text': 'partial'}, 'synthetic')
        wire = b''.join([chunk async for chunk in main._chat_completion_sse_from_payload(payload)])
        events = [json.loads(line[5:]) for line in wire.decode().splitlines()
                  if line.startswith('data:') and line[5:].strip() != '[DONE]']
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'length')

    async def test_image_trim_requires_explicit_policy_and_strict_always_preserves(self):
        image = Image.new('RGB', (2, 2))
        buffer = BytesIO(); image.save(buffer, format='PNG')
        block = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                            'data': base64.b64encode(buffer.getvalue()).decode()}}
        for headers, status, sends in (({}, 413, 0), ({'X-LB-Image-Trim': 'allow'}, 200, 1),
            ({'X-LB-Image-Trim': 'allow', 'X-LB-Strict-Parameters': 'true'}, 400, 0)):
            with self.subTest(headers=headers):
                route = AsyncMock(return_value=main.JSONResponse({'type': 'message', 'stop_reason': 'end_turn'}))
                proxy = SimpleNamespace(verify_api_key=lambda k: k == 'synthetic', proxy_request=route)
                with patch.object(main, 'proxy', proxy), patch.object(main, 'copilot_proxy', None), \
                        patch.object(main, 'azure_proxy', None), patch.object(main, '_IMG_MAX_COUNT', 1), \
                        patch.object(main, '_IMG_COMPRESS_THRESHOLD', 1):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local') as client:
                        response = await client.post('/v1/messages', json={'model': 'claude-opus-5',
                            'messages': [{'role': 'user', 'content': [block, block]}]},
                            headers={'Authorization': 'Bearer synthetic', **headers})
                self.assertEqual(response.status_code, status)
                self.assertEqual(route.await_count, sends)
                if sends:
                    self.assertIn('images.trimmed', response.headers['x-lb-dropped-parameters'])
                    sent = route.call_args.args[0]['messages'][0]['content']
                    self.assertEqual([part['type'] for part in sent], ['text', 'image'])
