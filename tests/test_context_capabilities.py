"""Channel-scoped capabilities and honest context estimates. Author: Zeno Ren."""
import copy
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
import httpx
import main


def catalog_data(**overrides):
    row = {'provider': 'copilot', 'api_type': 'responses', 'model': 'synthetic-model',
           'endpoint_alias': None, 'model_version': 'synthetic-v1',
           'verification': 'operator_verified', 'verified_at': '2026-09-29T00:00:00Z',
           'expires_at': '2026-10-02T00:00:00Z',
           'source': {'kind': 'channel_contract', 'url': 'https://fixture.invalid/contract'},
           'limits': {'input_tokens': 1000, 'context_tokens': 1200, 'output_tokens': 200},
           'features': {'tools': True, 'images': None, 'structured_output': False, 'opaque_state': None}}
    row.update(overrides)
    return {'schema_version': 1, 'entries': [row]}


class CapabilityTests(unittest.TestCase):
    def test_shipped_example_has_no_fabricated_verified_limits(self):
        from model_capabilities import CapabilityCatalog
        document = json.loads((Path(__file__).parents[1] / 'model-capabilities.example.json').read_text())
        catalog = CapabilityCatalog(document)
        self.assertTrue(all(row['capability_status'] != 'verified' for row in catalog.public_view()['entries']))
        self.assertTrue(all(value is None for row in document['entries'] for value in row['limits'].values()))

    def make(self, **overrides):
        from model_capabilities import CapabilityCatalog
        return CapabilityCatalog(catalog_data(**overrides), now=lambda: datetime(2026, 9, 30, tzinfo=timezone.utc))

    def test_system_tools_and_unicode_affect_estimate_without_mutating_content(self):
        from model_capabilities import estimate_payload
        simple = {'messages': [{'role': 'user', 'content': 'synthetic'}]}
        large = {**simple, 'system': '汉' * 10000, 'tools': [{'name': 'lookup', 'description': 'x' * 10000}]}
        before = copy.deepcopy(large)
        self.assertGreater(estimate_payload(large)['estimated_input_tokens'], estimate_payload(simple)['estimated_input_tokens'])
        self.assertEqual(estimate_payload(large)['estimate_confidence'], 'low')
        self.assertEqual(large, before)

    def test_images_opaque_and_server_side_history_are_explicitly_incomplete(self):
        from model_capabilities import estimate_payload
        result = estimate_payload({'previous_response_id': 'synthetic', 'input': [
            {'type': 'input_image', 'image_url': 'data:image/png;base64,synthetic'},
            {'type': 'reasoning', 'encrypted_content': 'synthetic'}]})
        self.assertFalse(result['estimate_complete'])
        self.assertEqual(set(result['unknown_components']), {'images', 'opaque_state', 'prior_state'})

    def test_unknown_expired_and_other_channel_do_not_gain_verified_limits(self):
        catalog = self.make()
        self.assertEqual(catalog.lookup('copilot', 'responses', 'synthetic-model')['capability_status'], 'verified')
        self.assertEqual(catalog.lookup('azure_openai', 'responses', 'synthetic-model')['capability_status'], 'unknown')
        self.assertEqual(catalog.lookup('copilot', 'chat', 'synthetic-model')['capability_status'], 'unknown')
        self.assertEqual(self.make(expires_at='2026-09-29T12:00:00Z').lookup(
            'copilot', 'responses', 'synthetic-model')['capability_status'], 'expired')

    def test_low_confidence_large_input_is_never_hard_rejected(self):
        from model_capabilities import evaluate_budget
        payload = {'input': 'synthetic ' * 10000, 'max_output_tokens': 100}
        result = evaluate_budget(self.make(), 'copilot', 'responses', 'synthetic-model', payload, mode='enforce')
        self.assertFalse(result['enforcement_allowed'])
        self.assertEqual(result['context_status'], 'estimated_over')

    def test_verified_output_limit_and_feature_rejection_are_precise(self):
        from model_capabilities import evaluate_budget
        for payload in ({'input': 'small', 'max_output_tokens': 201},
                        {'input': 'small', 'text': {'format': {'type': 'json_schema'}}}):
            with self.subTest(payload=payload), self.assertRaises(HTTPException) as error:
                evaluate_budget(self.make(), 'copilot', 'responses', 'synthetic-model', payload, mode='enforce')
            self.assertEqual(error.exception.status_code, 400)
            self.assertFalse(error.exception.detail['error']['retryable'])
        evaluate_budget(self.make(verification='unverified'), 'copilot', 'responses', 'synthetic-model',
                        {'max_output_tokens': 10000}, mode='enforce')

    def test_catalog_rejects_invalid_limits_duplicate_routes_and_sensitive_urls(self):
        from model_capabilities import CapabilityCatalog
        invalid = []
        item = catalog_data(); item['entries'][0]['limits']['input_tokens'] = True; invalid.append(item)
        item = catalog_data(); item['entries'] *= 2; invalid.append(item)
        item = catalog_data(); item['entries'][0]['source']['url'] = 'https://user:secret@fixture.invalid'; invalid.append(item)
        item = catalog_data(); item['entries'][0]['verified_at'] = '2026-09-29'; invalid.append(item)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                CapabilityCatalog(item)


class CapabilityEntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_models_schema_stays_compatible_and_admin_catalog_requires_auth(self):
        auth = SimpleNamespace(verify_api_key=lambda k: k == 'synthetic')
        with patch.object(main, 'proxy', None), patch.object(main, 'azure_proxy', None), patch.object(main, 'copilot_proxy', auth):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local') as client:
                unauth = await client.get('/admin/model-capabilities')
                catalog = await client.get('/admin/model-capabilities', headers={'Authorization': 'Bearer synthetic'})
        self.assertEqual(unauth.status_code, 401)
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json()['schema_version'], 1)
        self.assertEqual(catalog.json()['unknown_policy'], 'observe_without_input_rejection')

    async def test_token_count_covers_system_and_tools_and_declares_estimation(self):
        auth = SimpleNamespace(verify_api_key=lambda k: k == 'synthetic')
        with patch.object(main, 'proxy', auth), patch.object(main, 'azure_proxy', None), patch.object(main, 'copilot_proxy', None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local') as client:
                missing = await client.post('/v1/messages/count_tokens', json={'messages': []})
                results = [await client.post('/v1/messages/count_tokens', json=payload, headers={'Authorization': 'Bearer synthetic'})
                    for payload in ({'messages': []}, {'messages': [], 'system': 'x' * 10000,
                                                      'tools': [{'name': 'lookup', 'description': 'y' * 10000}]})]
        self.assertEqual(missing.status_code, 401)
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertGreater(results[1].json()['input_tokens'], results[0].json()['input_tokens'])
        self.assertEqual(results[1].headers['x-lb-token-count-confidence'], 'low')
        self.assertIn('estimate', results[1].headers['x-lb-token-count-method'])

    async def test_known_output_rejected_before_any_copilot_send_or_stream_headers(self):
        from model_capabilities import CapabilityCatalog
        import model_capabilities
        ep = main.CopilotEndpoint('synthetic', 'synthetic', models=['synthetic-model'])
        ep.session_token = 'synthetic'; ep.session_token_expires_at = 2**31
        proxy = main.CopilotProxy(main.LoadBalancer([ep]), 'synthetic')
        proxy._build_headers = AsyncMock(return_value={})
        calls = []
        await proxy.client.aclose()
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request)), trust_env=False)
        catalog = CapabilityCatalog(catalog_data(), now=lambda: datetime(2026, 9, 30, tzinfo=timezone.utc))
        try:
            with patch.object(model_capabilities, 'CATALOG', catalog), patch.object(model_capabilities, 'MODE', 'enforce'), \
                    patch.object(main, 'proxy', None), patch.object(main, 'azure_proxy', None), patch.object(main, 'copilot_proxy', proxy):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local') as client:
                    response = await client.post('/v1/responses', json={'model': 'synthetic-model', 'input': 'small',
                        'stream': True, 'max_output_tokens': 201}, headers={'Authorization': 'Bearer synthetic'})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(calls, [])
            self.assertEqual(ep.active_requests, 0)
            self.assertEqual(ep.total_errors, 0)
        finally:
            await proxy.close()
