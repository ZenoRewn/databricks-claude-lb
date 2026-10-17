"""Per-key provider allow-lists. Author: Zeno Ren.

An allow-list only ever *removes* routing candidates. It must not add a
provider, reorder Copilot-first → Azure-fallback, or let a provider serve a
model it does not support — those are existing invariants in CLAUDE.md.

The store holds tenant names and provider ids only. No client credential
material is ever written to it, matching the project rule that artifacts carry
identity and fingerprints rather than credential contents.

Enforcement reads an in-memory cache. Degradation follows the project's single
rule — an overlay that is missing or unloaded falls back to the config baseline
— which for this layer means no restriction at all: a restriction layer must
not turn a database outage into a total outage. See DegradationFallsBackToThe
ConfigBaselineTests at the bottom, which pins that rule across every layer so
it cannot drift back into per-layer folklore.
"""
import asyncio
import unittest
from unittest.mock import patch

import key_policy
import main


class ProviderVocabularyTests(unittest.TestCase):
    def test_ids_match_the_values_usage_records_already_use(self):
        # usage_store.record(provider=...) is called with the first three.
        # `azure_anthropic` is registered ahead of its request path so the
        # per-key allowlist and the Dashboard can already name it; no usage
        # record carries that value until /v1/messages can route there. The
        # vocabulary and the recorded values are allowed to differ only in
        # that direction — an id nobody records is inert, an id recorded but
        # unregistered would fail allowlist validation.
        self.assertEqual(set(key_policy.PROVIDER_IDS),
                         {'databricks', 'copilot', 'azure_openai', 'azure_anthropic'})

    def test_every_id_has_an_operator_facing_label(self):
        # Both Foundry faces are listed: one resource, two protocols, two
        # billing口径. See key_policy.PROVIDER_IDS for why they stay separate.
        self.assertEqual(key_policy.PROVIDER_LABELS,
                         {'databricks': 'ADB', 'copilot': 'GHCP',
                          'azure_openai': 'Foundry', 'azure_anthropic': 'Foundry Claude'})

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


class TenantProviderResetTests(unittest.TestCase):
    """Setting an allow-list had no inverse from the Dashboard.

    `clear_providers` existed in the store from the start but was never exposed,
    so a tenant restricted by mistake stayed restricted until someone edited the
    database. Reset returns that tenant to the config baseline — every provider
    config declares — which is the same place the degradation rule lands on.
    """

    KEY = 'lb-provider-reset-key-01234'

    def setUp(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = TenantProviderResetTests.KEY

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = key_policy.KeyPolicyStore(None)
        asyncio.run(self.store.start())
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[self.KEY] = 'tenant-a'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        for attr, value in (('proxy', _Stub()), ('copilot_proxy', _Stub()),
                            ('key_policy_store', self.store)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.H = {'x-api-key': self.KEY}

    def _restrict(self):
        asyncio.run(self.store.set_providers('tenant-a', ['copilot']))

    def test_reset_returns_the_tenant_to_the_config_baseline(self):
        self._restrict()
        self.assertTrue(self.store.restricted('tenant-a'))
        r = self.client.delete('/admin/api-keys/tenant-a/providers', headers=self.H)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(r.json()['providers']), sorted(key_policy.ALL_PROVIDERS))
        self.assertFalse(self.store.restricted('tenant-a'))

    def test_resetting_an_already_unrestricted_tenant_is_not_an_error(self):
        r = self.client.delete('/admin/api-keys/tenant-a/providers', headers=self.H)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(self.store.restricted('tenant-a'))

    def test_an_unknown_tenant_is_404_like_the_put(self):
        r = self.client.delete('/admin/api-keys/nobody/providers', headers=self.H)
        self.assertEqual(r.status_code, 404)

    def test_it_requires_a_valid_key(self):
        self.assertEqual(self.client.delete('/admin/api-keys/tenant-a/providers').status_code, 401)

    def test_a_write_failure_keeps_the_restriction_and_says_so(self):
        self._restrict()
        with patch.object(self.store, '_backend_delete', side_effect=OSError('db down')):
            r = self.client.delete('/admin/api-keys/tenant-a/providers', headers=self.H)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()['detail']['error']['code'], 'policy_write_failed')
        self.assertTrue(self.store.restricted('tenant-a'),
                        'a failed reset must not pretend the tenant is unrestricted')


class DegradationFallsBackToTheConfigBaselineTests(unittest.IsolatedAsyncioTestCase):
    """One rule, every layer: an unavailable overlay yields the config baseline.

    Four of these baselines happen to be permissive, which is why the rule has
    been mistaken for "fail-open" before. It is not: GHCP account enablement has
    a baseline that can be restrictive, and the same rule then keeps the account
    out of service. Asserting all five together is what stops the next reader
    from treating one of them as an exception.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        await self.store.set_providers('tenant-a', ['copilot'])
        await self.store.set_model_preference('gpt-4o', ['azure_openai'])
        await self.store.set_account_capability('acct', ('gpt-4o',))
        await self.store.set_account('acct', allowed_models=['gpt-4o'])
        await self.store.set_model_accounts('gpt-4o', ['acct'])
        patcher = patch.object(main, 'key_policy_store', self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _unavailable(self):
        self.store._state = 'unavailable'

    async def test_every_overlay_is_in_force_while_the_store_is_loaded(self):
        # Guards the test itself: if nothing were in force, the assertions below
        # would pass for the wrong reason.
        self.assertEqual(self.store.allowed('tenant-a'), frozenset({'copilot'}))
        self.assertEqual(self.store.model_preference('gpt-4o'), ('azure_openai',))
        self.assertFalse(self.store.account_allows('acct', 'kimi-k3'))
        self.assertFalse(self.store.model_allows_account('gpt-4o', 'other'))

    async def test_provider_allowlist_falls_back_to_every_configured_provider(self):
        self._unavailable()
        self.assertEqual(self.store.allowed('tenant-a'), key_policy.ALL_PROVIDERS)

    async def test_model_account_selection_falls_back_to_every_account(self):
        self._unavailable()
        self.assertEqual(self.store.model_accounts('gpt-4o'), ())
        self.assertTrue(self.store.model_allows_account('gpt-4o', 'other'),
                        'a per-model account limit must not outlive its store')

    async def test_model_preference_falls_back_to_the_default_order(self):
        self._unavailable()
        self.assertEqual(self.store.model_preference('gpt-4o'), ())
        self.assertEqual(main._openai_provider_order('gpt-4o'), main.OPENAI_STYLE_PROVIDERS)

    async def test_allowed_models_falls_back_to_the_endpoints_configured_models(self):
        self._unavailable()
        # The overlay stops subtracting, but config's own limit still holds.
        endpoint = main.CopilotEndpoint(name='acct', github_token='synthetic', models=['gpt-4o'])
        self.assertTrue(endpoint.supports('gpt-4o', 'responses'))
        self.assertFalse(endpoint.supports('kimi-k3', 'responses'),
                         "config's models list is the baseline, not the overlay")

    async def test_account_enablement_falls_back_to_the_config_declaration(self):
        self._unavailable()
        inert = main.CopilotEndpoint(name='acct', github_token='synthetic',
                                     enabled_in_config=False)
        live = main.CopilotEndpoint(name='acct', github_token='synthetic')
        self.assertFalse(main._copilot_account_enabled(inert),
                         'a restrictive baseline must survive the outage too')
        self.assertTrue(main._copilot_account_enabled(live))

    async def test_a_recorded_decision_survives_the_outage_in_both_directions(self):
        # Degradation drops the overlay's *cache*, not an explicit decision that
        # is still in memory — otherwise a disabled account would come back.
        await self.store.set_account('acct', enabled=False)
        self._unavailable()
        self.assertFalse(main._copilot_account_enabled(
            main.CopilotEndpoint(name='acct', github_token='synthetic')))

    async def test_the_status_view_reports_the_degraded_state_rather_than_hiding_it(self):
        self._unavailable()
        self.assertEqual(self.store.status()['state'], 'unavailable')


class ExclusionReasonTests(unittest.IsolatedAsyncioTestCase):
    """The booleans must be derived from the reasons, not computed beside them.

    The Dashboard has to answer «why is this account not receiving traffic for
    this model», and the only honest source for that is the same code that
    decides candidacy. Two implementations would drift, and the one that drifts
    is the diagnostic — which is worse than having none, because it would be
    believed. So the reason reader is primary and `account_allows` /
    `model_allows_account` delegate to it; these tests pin the delegation.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()

    async def test_no_record_at_all_excludes_nothing(self):
        self.assertIsNone(self.store.account_exclusion_reason('ghost', 'gpt-4o'))
        self.assertIsNone(self.store.model_accounts_exclusion_reason('gpt-4o', 'ghost'))

    async def test_a_disabled_account_names_its_own_layer(self):
        await self.store.set_account('acct', enabled=False)
        self.assertEqual(self.store.account_exclusion_reason('acct', 'gpt-4o'),
                         'account_disabled')

    async def test_a_model_outside_capability_is_not_confused_with_operator_intent(self):
        # capability is upstream fact; allowed_models is operator intent. Telling
        # them apart is the whole point — one is fixed by refreshing, the other
        # by changing a checkbox.
        await self.store.set_account_capability('acct', ('gpt-4o',))
        self.assertEqual(self.store.account_exclusion_reason('acct', 'kimi-k3'),
                         'not_in_capability')
        await self.store.set_account('acct', allowed_models=['gpt-4o'])
        self.assertIsNone(self.store.account_exclusion_reason('acct', 'gpt-4o'))

    async def test_allowed_models_reports_the_account_side_intent(self):
        await self.store.set_account_capability('acct', ('gpt-4o', 'gpt-5'))
        await self.store.set_account('acct', allowed_models=['gpt-4o'])
        self.assertEqual(self.store.account_exclusion_reason('acct', 'gpt-5'),
                         'excluded_by_allowed_models')

    async def test_a_model_limited_to_other_accounts_reports_the_model_side_intent(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertEqual(self.store.model_accounts_exclusion_reason('gpt-4o', 'acct-b'),
                         'not_selected_for_model')
        self.assertIsNone(self.store.model_accounts_exclusion_reason('gpt-4o', 'acct-a'))

    async def test_an_unloaded_store_reports_no_reason_because_it_enforces_nothing(self):
        await self.store.set_account('acct', enabled=False)
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.store._state = 'unavailable'
        self.assertIsNone(self.store.account_exclusion_reason('acct', 'gpt-4o'))
        self.assertIsNone(self.store.model_accounts_exclusion_reason('gpt-4o', 'acct-b'))

    async def test_the_booleans_agree_with_the_reasons_across_every_combination(self):
        """The anti-drift guard. If these ever disagree the diagnostic lies."""
        await self.store.set_account_capability('capped', ('gpt-4o', 'gpt-5'))
        await self.store.set_account('capped', allowed_models=['gpt-4o'])
        await self.store.set_account('off', enabled=False)
        await self.store.set_account_capability('open', ())
        await self.store.set_model_accounts('gpt-4o', ['capped'])
        await self.store.set_model_accounts('kimi-k3', ['ghost'])
        for state in ('loaded', 'stale', 'unavailable'):
            self.store._state = state
            for account in ('capped', 'off', 'open', 'never-seen'):
                for model in ('gpt-4o', 'gpt-5', 'kimi-k3', 'brand-new'):
                    with self.subTest(state=state, account=account, model=model):
                        self.assertEqual(
                            self.store.account_allows(account, model),
                            self.store.account_exclusion_reason(account, model) is None)
                        self.assertEqual(
                            self.store.model_allows_account(model, account),
                            self.store.model_accounts_exclusion_reason(model, account) is None)

    async def test_every_reason_is_in_the_documented_closed_set(self):
        # A free-text reason would be unusable in the UI and untestable here.
        # CLAUDE.md and docs/CONFIGURATION.md publish this set.
        self.assertEqual(key_policy.EXCLUSION_REASONS, {
            'account_disabled', 'not_in_capability', 'excluded_by_allowed_models',
            'not_selected_for_model',
        })


if __name__ == '__main__':
    unittest.main()
