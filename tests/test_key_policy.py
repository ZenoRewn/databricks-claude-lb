"""Per-key provider allow-lists. Author: Zeno Ren.

An allow-list only ever *removes* routing candidates. It must not add a
provider, reorder Copilot-first → Azure-fallback, or let a provider serve a
model it does not support — those are existing invariants in CLAUDE.md.

The store holds tenant names and provider ids only. No client credential
material is ever written to it, matching the project rule that artifacts carry
identity and fingerprints rather than credential contents.

Enforcement reads an in-memory cache. If the store was never loaded the gateway
applies no restriction and reports the state, a deliberate fail-open: a
restriction layer must not turn a database outage into a total outage.
"""
import asyncio
import unittest
from unittest.mock import patch

import key_policy
import main


class ProviderVocabularyTests(unittest.TestCase):
    def test_ids_match_the_values_usage_records_already_use(self):
        # usage_store.record(provider=...) is called with exactly these.
        self.assertEqual(set(key_policy.PROVIDER_IDS), {'databricks', 'copilot', 'azure_openai'})

    def test_every_id_has_an_operator_facing_label(self):
        self.assertEqual(key_policy.PROVIDER_LABELS,
                         {'databricks': 'ADB', 'copilot': 'GHCP', 'azure_openai': 'Foundry'})

    def test_normalise_accepts_any_order_and_canonicalises(self):
        self.assertEqual(key_policy.normalise_providers(['copilot', 'databricks']),
                         ('databricks', 'copilot'))
        self.assertEqual(key_policy.normalise_providers(('azure_openai',)), ('azure_openai',))

    def test_normalise_rejects_unknown_or_empty(self):
        for value in (['nope'], [], 'copilot', None, ['copilot', 'copilot', 'nope']):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    key_policy.normalise_providers(value)

    def test_duplicates_collapse_rather_than_erroring(self):
        self.assertEqual(key_policy.normalise_providers(['copilot', 'copilot']), ('copilot',))


class FingerprintAndMaskTests(unittest.TestCase):
    def test_fingerprint_is_a_short_sha256_prefix(self):
        import hashlib
        key = 'lb-secret-value-1234'
        self.assertEqual(key_policy.fingerprint(key),
                         hashlib.sha256(key.encode()).hexdigest()[:12])

    def test_mask_shows_only_the_outer_characters(self):
        self.assertEqual(key_policy.mask('abcdefghijklmnop'), 'abcd…mnop')

    def test_short_keys_are_not_masked_but_withheld(self):
        # Revealing 8 of 10 characters would be most of the secret.
        for key in ('short', 'elevenchars', 'a' * 11):
            with self.subTest(key=key):
                self.assertEqual(key_policy.mask(key), '')

    def test_mask_never_returns_the_whole_key(self):
        for key in ('abcdefghijklmnop', 'a' * 64, 'short'):
            with self.subTest(key=key):
                self.assertNotIn(key, key_policy.mask(key))


class StoreStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_backend_starts_loaded_and_allows_everything(self):
        store = key_policy.KeyPolicyStore(None)
        await store.start()
        self.assertEqual(store.status()['state'], 'loaded')
        self.assertEqual(store.status()['backend'], 'memory')
        self.assertEqual(store.allowed('anyone'), key_policy.ALL_PROVIDERS)
        self.assertFalse(store.restricted('anyone'))

    async def test_setting_providers_restricts_only_that_tenant(self):
        store = key_policy.KeyPolicyStore(None)
        await store.start()
        await store.set_providers('team-a', ['azure_openai'])
        self.assertEqual(store.allowed('team-a'), frozenset({'azure_openai'}))
        self.assertTrue(store.restricted('team-a'))
        self.assertEqual(store.allowed('team-b'), key_policy.ALL_PROVIDERS)

    async def test_never_loaded_store_applies_no_restriction_and_says_so(self):
        store = key_policy.KeyPolicyStore({'type': 'mysql'})
        with patch.object(store, '_backend_load', side_effect=OSError('db down')):
            await store.start()
        self.assertEqual(store.status()['state'], 'unavailable')
        self.assertEqual(store.allowed('team-a'), key_policy.ALL_PROVIDERS)

    async def test_a_failed_refresh_keeps_the_last_known_policy_and_marks_it_stale(self):
        store = key_policy.KeyPolicyStore(None)
        await store.start()
        await store.set_providers('team-a', ['copilot'])
        with patch.object(store, '_backend_load', side_effect=OSError('db down')):
            await store.reload()
        self.assertEqual(store.status()['state'], 'stale')
        self.assertEqual(store.allowed('team-a'), frozenset({'copilot'}))

    async def test_write_failure_does_not_change_the_enforced_policy(self):
        store = key_policy.KeyPolicyStore(None)
        await store.start()
        await store.set_providers('team-a', ['copilot'])
        with patch.object(store, '_backend_write', side_effect=OSError('db down')):
            with self.assertRaises(OSError):
                await store.set_providers('team-a', ['databricks'])
        self.assertEqual(store.allowed('team-a'), frozenset({'copilot'}))

    async def test_snapshot_carries_no_credential_material(self):
        store = key_policy.KeyPolicyStore(None)
        await store.start()
        await store.set_providers('team-a', ['copilot'])
        blob = repr(store.snapshot()) + repr(store.status())
        for secret in ('sk-', 'password', 'token'):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, blob)
        self.assertEqual(set(store.snapshot()), {'team-a'})


class SubtractionOnlyTests(unittest.TestCase):
    """The allow-list must not widen routing."""

    def test_allowed_providers_defaults_to_everything_without_a_store(self):
        with patch.object(main, 'key_policy_store', None):
            self.assertEqual(main._allowed_providers('default'), key_policy.ALL_PROVIDERS)

    def test_allowed_providers_reads_the_store(self):
        store = key_policy.KeyPolicyStore(None)
        asyncio.run(store.start())
        asyncio.run(store.set_providers('team-a', ['databricks']))
        with patch.object(main, 'key_policy_store', store):
            self.assertEqual(main._allowed_providers('team-a'), frozenset({'databricks'}))

    def test_provider_allowed_helper_is_a_pure_membership_check(self):
        store = key_policy.KeyPolicyStore(None)
        asyncio.run(store.start())
        asyncio.run(store.set_providers('team-a', ['copilot']))
        with patch.object(main, 'key_policy_store', store):
            self.assertTrue(main._provider_allowed('copilot', 'team-a'))
            self.assertFalse(main._provider_allowed('azure_openai', 'team-a'))
            self.assertFalse(main._provider_allowed('databricks', 'team-a'))


class RouteFilteringTests(unittest.IsolatedAsyncioTestCase):
    """_route_openai must subtract candidates, never invent one."""

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    async def _route(self, model, *, tenant, copilot_can, azure_supports):
        calls = []

        class _Copilot:
            def can_handle(self, m, api_type=None):
                return copilot_can

            def supports_model(self, m, api_type=None):
                return copilot_can

            async def proxy_responses(self, body, **kw):
                calls.append('copilot')
                return 'copilot-response'

        class _Azure:
            async def proxy_responses(self, body, **kw):
                calls.append('azure')
                return 'azure-response'

        main._CURRENT_TENANT.set(tenant)
        with patch.object(main, 'copilot_proxy', _Copilot()), \
             patch.object(main, 'azure_proxy', _Azure() if azure_supports else None), \
             patch.object(main, '_azure_supports_model', lambda m: azure_supports):
            result = await main._route_openai({'model': model}, False, 'responses')
        return result, calls

    async def test_unrestricted_tenant_keeps_copilot_first(self):
        result, calls = await self._route('gpt-5.4', tenant='default',
                                          copilot_can=True, azure_supports=True)
        self.assertEqual(result, 'copilot-response')
        self.assertEqual(calls, ['copilot'])

    async def test_denying_copilot_routes_to_the_remaining_provider(self):
        await self.store.set_providers('team-a', ['azure_openai'])
        result, calls = await self._route('gpt-5.4', tenant='team-a',
                                          copilot_can=True, azure_supports=True)
        self.assertEqual(result, 'azure-response')
        self.assertEqual(calls, ['azure'])

    async def test_policy_rejection_is_403_and_distinguishable_from_no_capacity(self):
        await self.store.set_providers('team-a', ['azure_openai'])
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            # Only Copilot could serve it, and the tenant may not use Copilot.
            await self._route('gemini-3.8-flash', tenant='team-a',
                              copilot_can=True, azure_supports=False)
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.detail['error']['code'], 'provider_not_allowed')

    async def test_a_model_no_provider_serves_still_reports_no_provider_not_policy(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            await self._route('made-up-model', tenant='default',
                              copilot_can=False, azure_supports=False)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_allowing_a_provider_cannot_make_it_serve_an_unsupported_model(self):
        # team-a may use Azure, but Azure has no deployment for this model.
        await self.store.set_providers('team-a', ['azure_openai'])
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            await self._route('gemini-3.8-flash', tenant='team-a',
                              copilot_can=False, azure_supports=False)
        self.assertEqual(ctx.exception.status_code, 404)


class MessagesEnforcementTests(unittest.TestCase):
    def test_databricks_denied_tenant_cannot_use_the_messages_entry_point(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = 'k-messages-enforcement'

            def verify_api_key(self, key):
                return key == self.api_key

        store = key_policy.KeyPolicyStore(None)
        asyncio.run(store.start())
        asyncio.run(store.set_providers('team-a', ['copilot']))
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT['k-messages-enforcement'] = 'team-a'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        with patch.object(main, 'proxy', _Stub()), \
             patch.object(main, 'key_policy_store', store):
            client = TestClient(main.app, raise_server_exceptions=False)
            r = client.post('/v1/messages',
                            headers={'x-api-key': 'k-messages-enforcement'},
                            json={'model': 'claude-opus-5-5', 'max_tokens': 8,
                                  'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()['detail']['error']['code'], 'provider_not_allowed')


if __name__ == '__main__':
    unittest.main()
