"""Native GHCP identity compatibility integration. Author: Zeno Ren."""
import asyncio
from dataclasses import replace
import gzip
import json
import os
import unittest
from unittest.mock import patch

import main
import httpx
import test_integrated_protocol as protocol


def frame(kind, **fields):
    return ('data: ' + json.dumps({'type': kind, **fields}, ensure_ascii=False) + '\n\n').encode()


def message(item_id, text='same text'):
    return {'id': item_id, 'type': 'message', 'role': 'assistant', 'phase': 'commentary',
            'content': [{'type': 'output_text', 'text': text}]}


def drift_frames(terminal='completed'):
    return [
        frame('response.output_item.added', output_index=0, item=message('A')),
        frame('response.output_text.delta', output_index=0, content_index=0, item_id='B', delta='same text'),
        frame('response.output_item.done', output_index=0, item=message('B')),
        frame('response.' + terminal, response={'id': 'r', 'status': terminal,
              'output': [message('C')], 'usage': {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10}}),
    ]


def events(wire):
    return [json.loads(line[6:]) for line in wire.splitlines() if line.startswith(b'data: {')]


def settings(mode='stabilize', models=('synthetic',)):
    return replace(main.LB_SETTINGS, copilot_responses_item_id_mode=mode,
                   copilot_responses_item_id_models=models)


class IdentitySettingsTests(unittest.TestCase):
    def test_defaults_off_and_empty_allowlist(self):
        with patch.dict(os.environ, {}, clear=True):
            value = main.LBSettings.load()
        self.assertEqual(value.copilot_responses_item_id_mode, 'off')
        self.assertEqual(value.copilot_responses_item_id_models, ())

    def test_modes_allowlist_and_strip_requirement(self):
        with patch.dict(os.environ, {'COPILOT_RESPONSES_ITEM_ID_MODE': 'stabilize',
                                    'COPILOT_RESPONSES_ITEM_ID_MODELS': ' a, b ,a ',
                                    'COPILOT_STRIP_INPUT_ITEM_IDS': 'true'}):
            value = main.LBSettings.load()
        self.assertEqual(value.copilot_responses_item_id_models, ('a', 'b'))
        self.assertEqual(value.as_dict()['copilot_responses_item_id_mode'], 'stabilize')
        with self.assertRaisesRegex(ValueError, 'COPILOT_STRIP_INPUT_ITEM_IDS'):
            replace(value, copilot_strip_input_item_ids=False)
        with self.assertRaisesRegex(ValueError, 'COPILOT_RESPONSES_ITEM_ID_MODE'):
            replace(value, copilot_responses_item_id_mode='invalid')
        replace(value, copilot_responses_item_id_mode='observe', copilot_strip_input_item_ids=False)


class IdentityIntegrationTests(unittest.IsolatedAsyncioTestCase):
    make = protocol.StreamPolicyTests.make
    response = protocol.StreamPolicyTests.response
    run_mode = protocol.StreamPolicyTests.run_mode

    async def test_native_stream_stable_ids_single_send_and_usage(self):
        with patch.object(main, 'LB_SETTINGS', settings()):
            wire, proxy, endpoint = await self.run_mode('copilot', 'responses', drift_frames())
        data = events(wire)
        self.assertEqual([data[0]['item']['id'], data[1]['item_id'], data[2]['item']['id'],
                          data[3]['response']['output'][0]['id']], ['A'] * 4)
        self.assertEqual(endpoint.successful_requests, 1)
        self.assertEqual(endpoint.total_input_tokens, 7)
        self.assertEqual(endpoint.total_output_tokens, 3)
        self.assertEqual(data[3]['response']['id'], 'r')
        self.assertEqual(data[2]['item']['phase'], 'commentary')

    async def test_opt_in_model_and_provider_isolation(self):
        source = drift_frames()
        for mode, models, provider in [('off', ('synthetic',), 'copilot'),
                                       ('observe', ('synthetic',), 'copilot'),
                                       ('stabilize', (), 'copilot'),
                                       ('stabilize', ('synthetic-other',), 'copilot'),
                                       ('stabilize', ('synthetic',), 'azure')]:
            with self.subTest(mode=mode, models=models, provider=provider), \
                    patch.object(main, 'LB_SETTINGS', settings(mode, models)):
                wire, _, _ = await self.run_mode(provider, 'responses', source)
                self.assertEqual(wire, b''.join(source))

    async def test_failed_and_incomplete_keep_outcome_and_usage(self):
        for terminal in ('failed', 'incomplete'):
            with self.subTest(terminal=terminal), patch.object(main, 'LB_SETTINGS', settings()):
                wire, _, ep = await self.run_mode('copilot', 'responses', drift_frames(terminal))
            self.assertEqual(events(wire)[-1]['response']['output'][0]['id'], 'A')
            self.assertEqual(events(wire)[-1]['type'], 'response.' + terminal)
            self.assertEqual(ep.successful_requests, 0)
            self.assertEqual(ep.total_input_tokens, 7)

    async def test_conflict_is_neutral_no_replay_and_retains_terminal_usage(self):
        source = drift_frames()
        source[-1] = frame('response.completed', response={'id': 'r', 'status': 'completed',
            'output': [{'type': 'function_call', 'id': 'tool', 'call_id': 'call-1'}],
            'usage': {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10}})
        with patch.object(main, 'LB_SETTINGS', settings()):
            wire, _, ep = await self.run_mode('copilot', 'responses', source)
        self.assertIn(b'identity_conflict', wire)
        self.assertNotIn(b'"type": "response.completed"', wire)
        self.assertEqual(ep.total_input_tokens, 7)
        self.assertEqual(ep.total_output_tokens, 3)
        self.assertEqual(ep.successful_requests, 0)
        self.assertEqual(ep.neutral_requests, 1)
        self.assertEqual(ep.total_errors, 0)

    async def test_gzip_and_arbitrary_chunk_boundaries(self):
        source = gzip.compress(b''.join(drift_frames()))
        with patch.object(main, 'LB_SETTINGS', settings()):
            wire, _, _ = await self.run_mode('copilot', 'responses',
                [source[i:i+3] for i in range(0, len(source), 3)], headers={'content-encoding': 'gzip'})
        self.assertEqual(events(wire)[-1]['response']['output'][0]['id'], 'A')

    async def test_mode_frozen_before_body_iteration(self):
        p, ep, stream, calls = await self.make('copilot', 'responses', drift_frames())
        with patch.object(main, 'LB_SETTINGS', settings()):
            response = await self.response(p, ep, 'copilot', 'responses')
        with patch.object(main, 'LB_SETTINGS', settings('off')):
            wire = b''.join([part async for part in response.body_iterator])
        self.assertEqual(events(wire)[-1]['response']['output'][0]['id'], 'A')
        self.assertTrue(stream.closed)
        self.assertEqual(len(calls), 1)

    async def test_consumer_close_releases_rewritten_frame_and_state(self):
        created = []
        real = main.ResponsesItemIdentity
        def capture(*args, **kwargs):
            result = real(*args, **kwargs)
            created.append(result)
            return result
        with patch.object(main, 'LB_SETTINGS', settings()), patch.object(main, 'ResponsesItemIdentity', capture):
            p, ep, stream, calls = await self.make('copilot', 'responses', drift_frames()[:2], wait=asyncio.Event())
            response = await self.response(p, ep, 'copilot', 'responses')
            iterator = response.body_iterator
            await anext(iterator)
            await anext(iterator)
            self.assertGreater(main._STREAM_BUFFER_BUDGET.retained, 0)
            await iterator.aclose()
        self.assertEqual(main._STREAM_BUFFER_BUDGET.retained, 0)
        self.assertEqual(created[0].summary()['mapping_bytes'], 0)
        self.assertTrue(stream.closed)
        self.assertEqual(ep.active_requests, 0)
        self.assertEqual(ep.total_errors, 0)
        self.assertEqual(len(calls), 1)

    async def test_identity_implementation_errors_never_penalize_account(self):
        for method in ('transform', 'summary'):
            for mode in ('observe', 'stabilize'):
                with self.subTest(method=method, mode=mode), patch.object(main, 'LB_SETTINGS', settings(mode)), \
                        patch.object(main.ResponsesItemIdentity, method, side_effect=RuntimeError('private payload')):
                    wire, _, ep = await self.run_mode('copilot', 'responses', drift_frames())
                    self.assertEqual(ep.total_errors, 0)
                    if mode == 'observe':
                        self.assertEqual(wire, b''.join(drift_frames()))
                        self.assertEqual(ep.successful_requests, 1)
                    else:
                        self.assertIn(b'identity_transform_error', wire)
                        self.assertNotIn(b'private payload', wire)
                        self.assertEqual(ep.neutral_requests, 1)

    async def test_identity_cleanup_fault_does_not_skip_upstream_close(self):
        with patch.object(main, 'LB_SETTINGS', settings()), \
                patch.object(main.ResponsesItemIdentity, 'close', side_effect=RuntimeError('local close bug')):
            _, proxy, ep = await self.run_mode('copilot', 'responses', drift_frames())
        self.assertEqual(ep.successful_requests, 1)
        self.assertEqual(proxy.responses_item_identity_mapping_bytes, 0)

    async def test_stats_once_per_request_with_finite_labels_and_private_ids_absent(self):
        with patch.object(main, 'LB_SETTINGS', settings()), self.assertLogs('main', level='INFO') as logs:
            _, proxy, _ = await self.run_mode('copilot', 'responses', drift_frames())
        stats = proxy.responses_item_identity_stats['stabilize']
        self.assertEqual(stats['requests'], 1)
        self.assertEqual(stats['drift_items'], 1)
        self.assertEqual(stats['rewritten_requests'], 1)
        self.assertEqual(stats['rewritten_frames'], 3)
        records = [record for record in logs.records if getattr(record, 'kind', None) == 'copilot_response_identity']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].identity_drift_items, 1)
        self.assertNotIn('same text', str(records[0].__dict__))
        with patch.object(main, 'copilot_proxy', proxy):
            metrics = (await main.metrics()).body.decode()
        self.assertIn('copilot_response_identity_drift_items_total{mode="stabilize"} 1', metrics)
        self.assertIn('copilot_response_identity_mapping_bytes 0', metrics)

    async def test_native_chat_and_nonstream_responses_bypass_normalizer(self):
        with patch.object(main, 'LB_SETTINGS', settings()), \
                patch.object(main, 'ResponsesItemIdentity', side_effect=AssertionError('must bypass')) as normalizer:
            await self.run_mode('copilot', 'chat', [b'data: [DONE]\n\n'])
            proxy, ep, _, calls = await self.make('copilot', 'responses', [])
            payload = {'id': 'r', 'status': 'completed', 'output': [message('B')]}
            async def handler(request):
                calls.append(request)
                return httpx.Response(200, json=payload)
            await proxy.client.aclose()
            proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
            self.addAsyncCleanup(proxy.client.aclose)
            await proxy.load_balancer.on_request_start(ep)
            response = await proxy._normal_request(ep, 'http://synthetic/responses', {}, {}, 'synthetic', 'responses', 0)
            self.assertEqual(json.loads(response.body)['output'][0]['id'], 'B')
            normalizer.assert_not_called()
