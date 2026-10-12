"""Per-model provider preference, with fallback retained. Author: Zeno Ren.

A preference reorders the providers that can serve a model; it does not remove
any. The default order stays Copilot → Azure, and providers the operator did
not name follow the ones they did. Combined with the per-key allow-list the
effective candidates are:

    (providers that can serve the model) ∩ (providers the key allows),
    ordered by preference, then by the default order

Opening `claude-*` on the OpenAI-style entry points is opt-in per model: the
gateway still rejects them unless an explicit preference maps that model to a
non-Databricks provider. Without that, a client sending claude to
/v1/chat/completions keeps getting the same 400 as before.

Scope limit: this only covers the OpenAI-style entry points, where Copilot
serves claude natively. /v1/messages stays Databricks-only because serving the
Anthropic Messages protocol from Copilot needs a bidirectional adapter — the
repository can parse Anthropic SSE but has no emitter.
"""
import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import key_policy
import main


def _store():
    store = key_policy.KeyPolicyStore(None)
    asyncio.run(store.start())
    return store


class PreferenceValidationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()

    async def test_order_is_preserved_unlike_the_key_allow_list(self):
        # normalise_providers canonicalises order; a preference must not.
        await self.store.set_model_preference('gpt-5.4', ['azure_openai', 'copilot'])
        self.assertEqual(self.store.model_preference('gpt-5.4'), ('azure_openai', 'copilot'))

    async def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            await self.store.set_model_preference('gpt-5.4', ['gcp'])

    async def test_empty_and_non_list_are_rejected(self):
        for value in ([], 'copilot', None, 42):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    await self.store.set_model_preference('gpt-5.4', value)

    async def test_duplicates_are_rejected_because_order_is_meaningful(self):
        with self.assertRaises(ValueError):
            await self.store.set_model_preference('gpt-5.4', ['copilot', 'copilot'])

    async def test_unset_model_has_no_preference(self):
        self.assertEqual(self.store.model_preference('gpt-4o'), ())

    async def test_clearing_removes_the_preference(self):
        await self.store.set_model_preference('gpt-5.4', ['azure_openai'])
        await self.store.clear_model_preference('gpt-5.4')
        self.assertEqual(self.store.model_preference('gpt-5.4'), ())

    async def test_never_loaded_store_reports_no_preference(self):
        store = key_policy.KeyPolicyStore({'type': 'mysql'})
        with patch.object(store, '_backend_load', side_effect=OSError('db down')):
            await store.start()
        self.assertEqual(store.model_preference('gpt-5.4'), ())

    async def test_write_failure_leaves_the_enforced_preference_unchanged(self):
        await self.store.set_model_preference('gpt-5.4', ['copilot'])
        with patch.object(self.store, '_backend_write_preference', side_effect=OSError('db down')):
            with self.assertRaises(OSError):
                await self.store.set_model_preference('gpt-5.4', ['azure_openai'])
        self.assertEqual(self.store.model_preference('gpt-5.4'), ('copilot',))


class ProviderOrderTests(unittest.TestCase):
    def setUp(self):
        self.store = _store()
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def test_default_order_is_copilot_then_azure(self):
        self.assertEqual(main._openai_provider_order('gpt-4o'), ('copilot', 'azure_openai'))

    def test_preference_moves_a_provider_first_without_dropping_the_other(self):
        asyncio.run(self.store.set_model_preference('gpt-5.4', ['azure_openai']))
        self.assertEqual(main._openai_provider_order('gpt-5.4'), ('azure_openai', 'copilot'))

    def test_full_preference_sets_the_whole_order(self):
        asyncio.run(self.store.set_model_preference('gpt-5.4', ['azure_openai', 'copilot']))
        self.assertEqual(main._openai_provider_order('gpt-5.4'), ('azure_openai', 'copilot'))

    def test_databricks_in_a_preference_does_not_enter_the_openai_order(self):
        # Databricks serves /v1/messages, not the OpenAI-style entry points.
        asyncio.run(self.store.set_model_preference('claude-opus-5.5', ['databricks', 'copilot']))
        self.assertEqual(main._openai_provider_order('claude-opus-5.5'), ('copilot', 'azure_openai'))

    def test_no_store_keeps_the_default_order(self):
        with patch.object(main, 'key_policy_store', None):
            self.assertEqual(main._openai_provider_order('gpt-4o'), ('copilot', 'azure_openai'))


class ClaudeOnOpenAiEntryPointsTests(unittest.TestCase):
    def setUp(self):
        self.store = _store()
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def test_claude_is_still_rejected_by_default(self):
        with self.assertRaises(HTTPException) as ctx:
            main._reject_anthropic_on_openai('claude-opus-5.5')
        self.assertEqual(ctx.exception.status_code, 400)

    def test_an_explicit_preference_opens_the_model(self):
        asyncio.run(self.store.set_model_preference('claude-opus-5.5', ['copilot']))
        main._reject_anthropic_on_openai('claude-opus-5.5')  # must not raise

    def test_opening_one_model_does_not_open_the_family(self):
        asyncio.run(self.store.set_model_preference('claude-opus-5.5', ['copilot']))
        for other in ('claude-sonnet-5.5', 'claude-haiku-5.5', 'claude-opus-5'):
            with self.subTest(model=other):
                with self.assertRaises(HTTPException):
                    main._reject_anthropic_on_openai(other)

    def test_a_databricks_only_preference_does_not_open_the_model(self):
        asyncio.run(self.store.set_model_preference('claude-opus-5.5', ['databricks']))
        with self.assertRaises(HTTPException):
            main._reject_anthropic_on_openai('claude-opus-5.5')

    def test_the_messages_entry_point_is_unaffected(self):
        # Databricks name mapping must keep working regardless of preference.
        asyncio.run(self.store.set_model_preference('claude-opus-5.5', ['copilot']))
        self.assertEqual(main.get_databricks_model('claude-opus-5-5'),
                         'databricks-claude-opus-5-5')


class RoutingHonoursOrderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)
        self.calls = []

    def _providers(self, *, copilot_fails=None):
        calls = self.calls

        class _Copilot:
            def can_handle(self, m, api_type=None):
                return True

            def supports_model(self, m, api_type=None):
                return True

            async def proxy_responses(self, body, **kw):
                calls.append('copilot')
                if copilot_fails:
                    raise copilot_fails
                return 'copilot-response'

        class _Azure:
            async def proxy_responses(self, body, **kw):
                calls.append('azure')
                return 'azure-response'

        return _Copilot(), _Azure()

    async def _route(self, model, *, copilot_fails=None):
        copilot, azure = self._providers(copilot_fails=copilot_fails)
        main._CURRENT_TENANT.set('default')
        with patch.object(main, 'copilot_proxy', copilot), \
             patch.object(main, 'azure_proxy', azure), \
             patch.object(main, '_azure_supports_model', lambda m: True):
            return await main._route_openai({'model': model}, False, 'responses')

    async def test_default_order_tries_copilot_first(self):
        result = await self._route('gpt-4o')
        self.assertEqual(result, 'copilot-response')
        self.assertEqual(self.calls, ['copilot'])

    async def test_preference_makes_azure_first(self):
        await self.store.set_model_preference('gpt-5.4', ['azure_openai'])
        result = await self._route('gpt-5.4')
        self.assertEqual(result, 'azure-response')
        self.assertEqual(self.calls, ['azure'])

    async def test_fallback_is_retained_when_the_preferred_provider_rejects(self):
        await self.store.set_model_preference('gpt-5.4', ['copilot'])
        result = await self._route('gpt-5.4', copilot_fails=main._UnsupportedModelError('nope'))
        self.assertEqual(result, 'azure-response')
        self.assertEqual(self.calls, ['copilot', 'azure'])

    async def test_transient_copilot_unavailability_still_falls_back(self):
        failure = HTTPException(status_code=503, detail={'error': {'code': 'endpoint_unavailable',
                                                                   'message': 'all open'}})
        result = await self._route('gpt-4o', copilot_fails=failure)
        self.assertEqual(result, 'azure-response')

    async def test_a_client_error_is_not_retried_on_another_provider(self):
        failure = HTTPException(status_code=400, detail={'error': {'message': 'bad request'}})
        with self.assertRaises(HTTPException) as ctx:
            await self._route('gpt-4o', copilot_fails=failure)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(self.calls, ['copilot'])

    async def test_key_allow_list_still_subtracts_from_the_ordered_set(self):
        await self.store.set_model_preference('gpt-5.4', ['azure_openai'])
        await self.store.set_providers('locked', ['copilot'])
        main._CURRENT_TENANT.set('locked')
        copilot, azure = self._providers()
        with patch.object(main, 'copilot_proxy', copilot), \
             patch.object(main, 'azure_proxy', azure), \
             patch.object(main, '_azure_supports_model', lambda m: True):
            result = await main._route_openai({'model': 'gpt-5.4'}, False, 'responses')
        self.assertEqual(result, 'copilot-response', 'preference must not override the key policy')


class HistoryPricingAttributionTests(unittest.TestCase):
    """Databricks records the mapped `databricks-` name; Copilot records its own id.

    Verified against production on 2026-10-12: usage_daily holds 14 Claude rows,
    all prefixed `databricks-`, and zero bare `claude-*` rows. So a bare claude
    row can only come from Copilot and must be priced on the Copilot basis.
    """

    STATS = {'requests': 1, 'input_tokens': 1000, 'output_tokens': 200,
             'cache_creation_tokens': 100, 'cache_read_tokens': 400}

    def test_databricks_prefixed_rows_keep_the_anthropic_basis(self):
        quote = main.historical_model_cost('databricks-claude-sonnet-5-5', dict(self.STATS))
        self.assertEqual(quote['pricing_status'], 'complete')
        self.assertEqual(quote['estimated_cost_usd'],
                         main.calculate_cost('databricks-claude-sonnet-5-5', 1000, 200, 100, 400))

    def test_a_bare_claude_row_is_priced_on_the_copilot_basis(self):
        import copilot_pricing
        quote = main.historical_model_cost('claude-sonnet-5.5', dict(self.STATS))
        reference = copilot_pricing.estimate_history_reference('claude-sonnet-5.5', dict(self.STATS))
        self.assertIsNotNone(reference)
        self.assertEqual(quote['pricing_basis'], reference['pricing_basis'])

    def test_the_two_channels_do_not_collapse_to_the_same_number(self):
        # sonnet-5.5 cache_read differs between the public API price and the
        # Copilot table, so a mis-attributed row would be silently wrong.
        adb = main.historical_model_cost('databricks-claude-sonnet-5-5', dict(self.STATS))
        ghcp = main.historical_model_cost('claude-sonnet-5.5', dict(self.STATS))
        self.assertNotEqual(adb['estimated_cost_usd'], ghcp['estimated_cost_usd'])

    def test_legacy_bare_family_names_still_use_the_anthropic_basis(self):
        quote = main.historical_model_cost('opus-5-5', dict(self.STATS))
        self.assertEqual(quote['estimated_cost_usd'],
                         main.calculate_cost('opus-5-5', 1000, 200, 100, 400))


class RoutingAdminEndpointTests(unittest.TestCase):
    KEY = 'lb-routing-admin-key-01234'

    def setUp(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = RoutingAdminEndpointTests.KEY

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = _store()
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[self.KEY] = 'default'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        for attr, value in (('proxy', _Stub()), ('copilot_proxy', _Stub()),
                            ('key_policy_store', self.store)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.H = {'x-api-key': self.KEY}

    def test_all_three_require_a_valid_key(self):
        self.assertEqual(self.client.get('/admin/model-routing').status_code, 401)
        self.assertEqual(self.client.put('/admin/model-routing/gpt-5.4',
                                         json={'providers': ['copilot']}).status_code, 401)
        self.assertEqual(self.client.delete('/admin/model-routing/gpt-5.4').status_code, 401)

    def test_setting_a_preference_reports_the_effective_order(self):
        r = self.client.put('/admin/model-routing/gpt-5.4', headers=self.H,
                            json={'providers': ['azure_openai']})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body['providers'], ['azure_openai'])
        self.assertEqual(body['effective_openai_order'], ['azure_openai', 'copilot'],
                         'the unnamed provider must stay as a fallback')
        self.assertFalse(body['opens_claude_on_openai_entry_points'])

    def test_mapping_a_claude_model_reports_that_it_opens_the_entry_points(self):
        r = self.client.put('/admin/model-routing/claude-opus-5.5', headers=self.H,
                            json={'providers': ['copilot']})
        self.assertTrue(r.json()['opens_claude_on_openai_entry_points'])

    def test_invalid_preferences_are_rejected(self):
        for providers in (['gcp'], [], 'copilot', ['copilot', 'copilot']):
            with self.subTest(providers=providers):
                r = self.client.put('/admin/model-routing/gpt-5.4', headers=self.H,
                                    json={'providers': providers})
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.json()['detail']['error']['code'], 'invalid_model_routing')

    def test_listing_shows_defaults_and_stored_preferences(self):
        self.client.put('/admin/model-routing/gpt-5.4', headers=self.H,
                        json={'providers': ['azure_openai']})
        body = self.client.get('/admin/model-routing', headers=self.H).json()
        self.assertEqual(body['default_openai_order'], ['copilot', 'azure_openai'])
        entry = next(p for p in body['preferences'] if p['model'] == 'gpt-5.4')
        self.assertEqual(entry['providers'], ['azure_openai'])

    def test_clearing_restores_the_default_and_recloses_claude(self):
        self.client.put('/admin/model-routing/claude-opus-5.5', headers=self.H,
                        json={'providers': ['copilot']})
        r = self.client.delete('/admin/model-routing/claude-opus-5.5', headers=self.H)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['effective_openai_order'], ['copilot', 'azure_openai'])
        with self.assertRaises(HTTPException):
            main._reject_anthropic_on_openai('claude-opus-5.5')

    def test_write_failure_reports_503_and_keeps_the_preference(self):
        self.client.put('/admin/model-routing/gpt-5.4', headers=self.H,
                        json={'providers': ['copilot']})
        with patch.object(self.store, '_backend_write_preference', side_effect=OSError('db down')):
            r = self.client.put('/admin/model-routing/gpt-5.4', headers=self.H,
                                json={'providers': ['azure_openai']})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.store.model_preference('gpt-5.4'), ('copilot',))


if __name__ == '__main__':
    unittest.main()
