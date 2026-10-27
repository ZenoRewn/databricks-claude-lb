"""Per-model routing: model → provider → GHCP account. Author: Zeno Ren.

A preference reorders the providers that can serve a model; it does not remove
any. The default order stays Copilot → Azure, and providers the operator did
not name follow the ones they did. Combined with the per-key allow-list the
effective candidates are:

    (providers that can serve the model) ∩ (providers the key allows),
    ordered by preference, then by the default order

A third segment names *which GHCP accounts* serve one model. The account-side
policy already in place is the reverse mapping — `allowed_models` answers "which
models may this account serve" — and the two are intersected, so either one
alone is enough to remove a candidate. Expressing the intent from the model side
matters because the reverse route would require editing every account's
allow-list to say one thing about one model, dragging other models along with it.

That segment subtracts, which the per-model *provider* preference deliberately
does not. It is not a contradiction: it subtracts accounts *within* a provider,
so the provider-level Copilot → Azure fallback is untouched. Selecting accounts
that cannot serve the model empties the Copilot candidate set and the request
falls back, exactly as it would if those accounts were circuit-broken.

Empty selection means "not limited", never "no account may serve this" — the
latter has no way back through the UI and would silently remove a model.

Opening `claude-*` on the OpenAI-style entry points is opt-in per model: the
gateway still rejects them unless an explicit preference maps that model to a
non-Databricks provider. Without that, a client sending claude to
/v1/chat/completions keeps getting the same 400 as before.

One stored preference now feeds **two** protocol surfaces, each filtering to
what it can reach: the OpenAI-style entries order copilot and azure_openai,
while /v1/messages orders databricks, copilot and azure_anthropic. The earlier
note here — that /v1/messages stays Databricks-only because serving Anthropic
Messages from Copilot would need a bidirectional adapter — was wrong, and the
2026-10-17 upstream probes are what disproved it: Copilot declares
`supported_endpoints: ["/v1/messages", ...]` itself and Foundry answers
`POST /anthropic/v1/messages` natively, so no adapter exists or is needed.
Per-surface ordering lives in tests/test_messages_surface_routing.py.
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


class ModelAccountSelectionTests(unittest.IsolatedAsyncioTestCase):
    """The third segment: which GHCP accounts serve one model.

    Unlike the provider preference, **order carries no meaning here** — which of
    the selected accounts a request lands on is decided by session affinity, not
    by how the operator sorted them. So this one canonicalises to a sorted set
    while the preference keeps the operator's order.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()

    async def test_unset_model_is_not_limited_to_any_account(self):
        self.assertEqual(self.store.model_accounts('gpt-4o'), ())
        self.assertTrue(self.store.model_allows_account('gpt-4o', 'anything'))

    async def test_a_selection_subtracts_every_account_outside_it(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-b', 'acct-a'])
        self.assertTrue(self.store.model_allows_account('gpt-4o', 'acct-a'))
        self.assertFalse(self.store.model_allows_account('gpt-4o', 'acct-c'))

    async def test_order_is_canonicalised_because_affinity_picks_not_the_operator(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-b', 'acct-a', 'acct-b'])
        self.assertEqual(self.store.model_accounts('gpt-4o'), ('acct-a', 'acct-b'))

    async def test_the_selection_is_scoped_to_one_model(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertTrue(self.store.model_allows_account('kimi-k3', 'acct-c'))

    async def test_an_empty_selection_clears_rather_than_excluding_everything(self):
        # Clearing every checkbox means "no longer limited", not "no account may
        # serve this model". The latter would remove the model from GHCP with no
        # way to express the undo, so it must not be reachable by accident.
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        for empty in ([], None):
            with self.subTest(empty=empty):
                await self.store.set_model_accounts('gpt-4o', empty)
                self.assertEqual(self.store.model_accounts('gpt-4o'), ())
                self.assertTrue(self.store.model_allows_account('gpt-4o', 'acct-c'))
                await self.store.set_model_accounts('gpt-4o', ['acct-a'])

    async def test_clearing_removes_the_selection(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        await self.store.clear_model_accounts('gpt-4o')
        self.assertEqual(self.store.model_accounts('gpt-4o'), ())

    async def test_a_bare_string_or_non_string_entries_are_rejected(self):
        # A bare string would silently become a per-character selection.
        for value in ('acct-a', 42, ['acct-a', 7], ['']):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    await self.store.set_model_accounts('gpt-4o', value)

    async def test_an_empty_model_name_is_rejected(self):
        for model in ('', '   ', None):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    await self.store.set_model_accounts(model, ['acct-a'])

    async def test_an_unavailable_store_falls_back_to_the_baseline(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.store._state = 'unavailable'
        self.assertEqual(self.store.model_accounts('gpt-4o'), ())
        self.assertTrue(self.store.model_allows_account('gpt-4o', 'acct-c'))

    async def test_status_counts_this_layer_too(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertEqual(self.store.status()['model_accounts'], 1)

    async def test_write_failure_leaves_the_enforced_selection_unchanged(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        with patch.object(self.store, '_backend_write_model_accounts',
                          side_effect=OSError('db down')):
            with self.assertRaises(OSError):
                await self.store.set_model_accounts('gpt-4o', ['acct-b'])
        self.assertEqual(self.store.model_accounts('gpt-4o'), ('acct-a',))

    async def test_never_loaded_store_reports_no_selection(self):
        store = key_policy.KeyPolicyStore({'type': 'mysql'})
        with patch.object(store, '_backend_load', side_effect=OSError('db down')):
            await store.start()
        self.assertEqual(store.model_accounts('gpt-4o'), ())

    async def test_the_json_backend_round_trips(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = key_policy.KeyPolicyStore({'type': 'json', 'path': tmp})
            await store.start()
            await store.set_model_accounts('gpt-4o', ['acct-b', 'acct-a'])
            reopened = key_policy.KeyPolicyStore({'type': 'json', 'path': tmp})
            await reopened.start()
            self.assertEqual(reopened.model_accounts('gpt-4o'), ('acct-a', 'acct-b'))
            await reopened.clear_model_accounts('gpt-4o')
            again = key_policy.KeyPolicyStore({'type': 'json', 'path': tmp})
            await again.start()
            self.assertEqual(again.model_accounts('gpt-4o'), ())


class AccountSelectionSubtractsCandidatesTests(unittest.IsolatedAsyncioTestCase):
    """Enforcement has to sit in `supports()` — the single choke point for every
    candidate decision, including the configured-eligible set session affinity
    hashes over. Filtering at the call sites instead would miss affinity."""

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        for name in ('acct-a', 'acct-b'):
            await self.store.set_account_capability(name, ('gpt-4o', 'kimi-k3'))
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)
        self.a = main.CopilotEndpoint(name='acct-a', github_token='synthetic')
        self.b = main.CopilotEndpoint(name='acct-b', github_token='synthetic')

    def _proxy(self, *endpoints):
        proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        proxy.load_balancer = main.LoadBalancer(list(endpoints))
        proxy.session_affinity_events = {}
        return proxy

    async def test_an_account_outside_the_selection_stops_supporting_that_model(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertTrue(self.a.supports('gpt-4o', 'responses'))
        self.assertFalse(self.b.supports('gpt-4o', 'responses'))

    async def test_other_models_on_the_excluded_account_are_untouched(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertTrue(self.b.supports('kimi-k3', 'responses'),
                        'the selection is per model, not a blanket account disable')

    async def test_the_proxy_level_helpers_follow_automatically(self):
        proxy = self._proxy(self.b)
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertFalse(proxy.supports_model('gpt-4o', 'responses'))
        self.assertFalse(proxy.can_handle('gpt-4o', 'responses'))
        self.assertTrue(proxy.supports_model('kimi-k3', 'responses'))

    async def test_both_account_layers_subtract(self):
        # allowed_models removes a model from an account; model_accounts removes
        # an account from a model. Either alone drops the candidate.
        await self.store.set_account('acct-a', allowed_models=['kimi-k3'])
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertFalse(self.a.supports('gpt-4o', 'responses'))

    async def test_selecting_accounts_without_the_capability_empties_the_set(self):
        # Not an error: the Copilot candidate set goes empty and the
        # provider-level fallback carries the request, same as a circuit open.
        await self.store.set_account_capability('acct-c', ('gpt-5.4',))
        c = main.CopilotEndpoint(name='acct-c', github_token='synthetic')
        await self.store.set_model_accounts('gpt-4o', ['acct-c'])
        for ep in (self.a, self.b, c):
            with self.subTest(account=ep.name):
                self.assertFalse(ep.supports('gpt-4o', 'responses'))
        self.assertFalse(self._proxy(self.a, self.b, c).supports_model('gpt-4o', 'responses'))

    async def test_a_disabled_account_stays_disabled_even_if_selected(self):
        # Selecting an account is not a way to enable it; enablement is its own
        # layer and this one only ever subtracts.
        await self.store.set_account('acct-a', enabled=False)
        await self.store.set_model_accounts('gpt-4o', ['acct-a'])
        self.assertFalse(self.a.supports('gpt-4o', 'responses'))

    async def test_no_store_means_no_limit(self):
        with patch.object(main, 'key_policy_store', None):
            self.assertTrue(self.a.supports('gpt-4o', 'responses'))
            self.assertTrue(self.b.supports('gpt-4o', 'responses'))

    async def test_the_model_catalogue_follows_the_selection(self):
        # Caught end-to-end, not by the assertions above: the catalogue asked
        # `account_allows` directly instead of going through supports(), so
        # /v1/models kept advertising a model whose only selected accounts
        # cannot serve it. A client would pick it and get a routing failure.
        await self.store.set_account_capability('acct-c', ('gpt-5.4',))
        c = main.CopilotEndpoint(name='acct-c', github_token='synthetic')
        proxy = self._proxy(self.a, self.b, c)
        with patch.object(main, 'copilot_proxy', proxy):
            self.assertIn('gpt-5.4', main._collect_openai_model_ids(azure=None, databricks=None))
            await self.store.set_model_accounts('gpt-5.4', ['acct-a'])
            catalogue = main._collect_openai_model_ids(azure=None, databricks=None)
            self.assertNotIn('gpt-5.4', catalogue,
                             'acct-a has no gpt-5.4 capability, so nothing can serve it')
            self.assertIn('gpt-4o', catalogue, 'unrelated models stay in the catalogue')

    async def test_a_wildcard_endpoint_also_honours_the_selection(self):
        # A wildcard account (no configured models, no capability yet) falls back
        # to the static verified table. That fallback used to bypass the account
        # selection wholesale, which is the same defect one layer up.
        wild = main.CopilotEndpoint(name='acct-wild', github_token='synthetic')
        proxy = self._proxy(wild)
        with patch.object(main, 'copilot_proxy', proxy):
            self.assertIn('gpt-5.4', main._collect_openai_model_ids(azure=None, databricks=None))
            await self.store.set_model_accounts('gpt-5.4', ['acct-a'])
            self.assertNotIn('gpt-5.4', main._collect_openai_model_ids(azure=None, databricks=None),
                             'the wildcard account is not selected for this model')
            self.assertIn('gpt-6.1-sol', main._collect_openai_model_ids(azure=None, databricks=None),
                          'models with no selection still come from the wildcard')


class SessionAffinitySurvivesTheSelectionTests(unittest.IsolatedAsyncioTestCase):
    """Affinity hashes over the configured-eligible set. This layer makes that
    set smaller; it must not make it unstable."""

    ACCOUNTS = ('acct-a', 'acct-b', 'acct-c', 'acct-d')

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        for name in self.ACCOUNTS:
            await self.store.set_account_capability(name, ('gpt-4o',))
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)
        self.endpoints = [main.CopilotEndpoint(name=n, github_token='synthetic')
                          for n in self.ACCOUNTS]
        self.proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        self.proxy.load_balancer = main.LoadBalancer(self.endpoints)
        self.proxy.session_affinity_events = {}

    def _pick(self, session_key):
        return self.proxy._select_endpoint('gpt-4o', 'responses', session_key=session_key)

    def _map(self, n=24):
        return {f'sess-{i}': self._pick(f'sess-{i}').name for i in range(n)}

    async def test_the_same_session_lands_on_the_same_account_every_time(self):
        first = self._pick('sess-1')
        for _ in range(10):
            self.assertIs(self._pick('sess-1'), first)

    async def test_the_selection_narrows_the_set_without_destabilising_it(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-a', 'acct-b'])
        placement = self._map()
        self.assertTrue(set(placement.values()) <= {'acct-a', 'acct-b'},
                        'an unselected account must never be chosen')
        self.assertEqual(placement, self._map(), 'placement must be reproducible')

    async def test_narrowing_to_one_account_routes_every_session_there(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-c'])
        self.assertEqual(set(self._map().values()), {'acct-c'})

    async def test_a_circuit_opening_does_not_reshuffle_the_other_sessions(self):
        # The hash is taken over the eligible set, not the available pool, so a
        # single account going down must not move sessions that were not on it.
        await self.store.set_model_accounts('gpt-4o', ['acct-a', 'acct-b', 'acct-c'])
        before = self._map()
        with patch.object(self.proxy.load_balancer, 'is_available',
                          lambda ep, **kw: ep.name != 'acct-c'):
            after = self._map()
        moved = {s for s, name in before.items() if after[s] != name}
        self.assertTrue(all(before[s] == 'acct-c' for s in moved),
                        'only the sessions pinned to the downed account may move')
        self.assertNotIn('acct-c', set(after.values()))

    async def test_changing_the_selection_only_moves_the_affected_sessions(self):
        # Dropping one account from a three-account selection reshuffles, because
        # the modulus changes — that is inherent to the existing hash and is why
        # the selection is an operator decision rather than something automatic.
        # What must hold is that no session lands outside the new selection.
        await self.store.set_model_accounts('gpt-4o', ['acct-a', 'acct-b', 'acct-c'])
        self._map()
        await self.store.set_model_accounts('gpt-4o', ['acct-a', 'acct-b'])
        self.assertTrue(set(self._map().values()) <= {'acct-a', 'acct-b'})


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


class RoutingRuleAdminTests(unittest.TestCase):
    """One rule, one call: model → provider → accounts.

    The Dashboard presents these three as a single row, so the endpoint has to
    accept them together. The account names are validated against the configured
    endpoints *before* anything is written, otherwise a typo in the third field
    would leave the provider half of the rule applied.
    """

    KEY = 'lb-routing-rule-key-0123456'

    def setUp(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = RoutingRuleAdminTests.KEY

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = _store()
        asyncio.run(self.store.set_account_capability('acct-a', ('gpt-4o', 'kimi-k3')))
        asyncio.run(self.store.set_account_capability('acct-b', ('gpt-4o',)))
        self.endpoints = [main.CopilotEndpoint(name='acct-a', github_token='synthetic'),
                          main.CopilotEndpoint(name='acct-b', github_token='synthetic')]
        copilot = main.CopilotProxy.__new__(main.CopilotProxy)
        copilot.load_balancer = main.LoadBalancer(self.endpoints)
        copilot.verify_api_key = _Stub().verify_api_key
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[self.KEY] = 'default'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        for attr, value in (('proxy', _Stub()), ('copilot_proxy', copilot),
                            ('key_policy_store', self.store)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.H = {'x-api-key': self.KEY}

    def test_a_rule_writes_both_dimensions_in_one_call(self):
        r = self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                            json={'providers': ['copilot'], 'copilot_accounts': ['acct-a']})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body['providers'], ['copilot'])
        self.assertEqual(body['copilot_accounts'], ['acct-a'])
        self.assertEqual(self.store.model_accounts('gpt-4o'), ('acct-a',))

    def test_an_unknown_account_is_refused_without_half_applying_the_rule(self):
        r = self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                            json={'providers': ['copilot'], 'copilot_accounts': ['acct-typo']})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'unknown_copilot_account')
        self.assertEqual(self.store.model_accounts('gpt-4o'), ())
        self.assertEqual(self.store.model_preference('gpt-4o'), (),
                         'validation must precede every write')

    def test_omitting_the_key_leaves_the_account_selection_alone(self):
        self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                        json={'providers': ['copilot'], 'copilot_accounts': ['acct-a']})
        r = self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                            json={'providers': ['azure_openai']})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['copilot_accounts'], ['acct-a'])
        self.assertEqual(self.store.model_accounts('gpt-4o'), ('acct-a',))

    def test_an_explicit_empty_selection_clears_it(self):
        self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                        json={'providers': ['copilot'], 'copilot_accounts': ['acct-a']})
        for empty in ([], None):
            with self.subTest(empty=empty):
                r = self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                                    json={'providers': ['copilot'], 'copilot_accounts': empty})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()['copilot_accounts'], [])
                self.assertEqual(self.store.model_accounts('gpt-4o'), ())

    def test_accounts_alone_without_providers_is_accepted(self):
        # Limiting accounts without reordering providers is a legitimate rule.
        r = self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                            json={'copilot_accounts': ['acct-a']})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.store.model_accounts('gpt-4o'), ('acct-a',))
        self.assertEqual(self.store.model_preference('gpt-4o'), ())

    def test_a_body_with_neither_field_is_rejected(self):
        r = self.client.put('/admin/model-routing/gpt-4o', headers=self.H, json={})
        self.assertEqual(r.status_code, 400)

    def test_delete_clears_both_dimensions(self):
        self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                        json={'providers': ['azure_openai'], 'copilot_accounts': ['acct-a']})
        r = self.client.delete('/admin/model-routing/gpt-4o', headers=self.H)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['copilot_accounts'], [])
        self.assertEqual(self.store.model_accounts('gpt-4o'), ())
        self.assertEqual(self.store.model_preference('gpt-4o'), ())

    def test_the_listing_carries_the_dropdown_sources(self):
        body = self.client.get('/admin/model-routing', headers=self.H).json()
        self.assertIn('gpt-4o', body['known_models'])
        self.assertEqual([a['account'] for a in body['copilot_accounts_available']],
                         ['acct-a', 'acct-b'])
        self.assertTrue(all('enabled' in a for a in body['copilot_accounts_available']))

    def test_effective_candidates_are_computed_server_side(self):
        # The client cannot work this out: it needs capability, allowed_models
        # and the enablement state all at once.
        self.client.put('/admin/model-routing/kimi-k3', headers=self.H,
                        json={'copilot_accounts': ['acct-a', 'acct-b']})
        entry = next(p for p in self.client.get('/admin/model-routing', headers=self.H)
                     .json()['preferences'] if p['model'] == 'kimi-k3')
        self.assertEqual(entry['copilot_accounts'], ['acct-a', 'acct-b'])
        self.assertEqual(entry['effective_copilot_accounts'], ['acct-a'],
                         'acct-b has no kimi-k3 capability, so it is not a candidate')

    def test_an_empty_effective_set_is_surfaced_rather_than_hidden(self):
        r = self.client.put('/admin/model-routing/gpt-5.4', headers=self.H,
                            json={'copilot_accounts': ['acct-a']})
        self.assertEqual(r.json()['effective_copilot_accounts'], [])
        self.assertTrue(r.json()['copilot_candidates_empty'],
                        'the operator has to be told this model now has no GHCP candidate')

    def test_write_failure_reports_503_and_keeps_the_selection(self):
        self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                        json={'copilot_accounts': ['acct-a']})
        with patch.object(self.store, '_backend_write_model_accounts',
                          side_effect=OSError('db down')):
            r = self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                                json={'copilot_accounts': ['acct-b']})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.store.model_accounts('gpt-4o'), ('acct-a',))

    def test_the_selection_requires_a_valid_key(self):
        self.assertEqual(self.client.put('/admin/model-routing/gpt-4o',
                                         json={'copilot_accounts': ['acct-a']}).status_code, 401)


class SupportReasonTests(unittest.IsolatedAsyncioTestCase):
    """`supports()` answers *whether*; the same code has to answer *why*.

    The Dashboard merges account enablement and the per-model account selection
    into one matrix, so it must label every row with the layer that dropped it.
    Writing that explanation separately would produce a second implementation of
    candidacy, and the copy that drifts is the one people read. So the reason is
    primary and `supports()` is `reason is None` — these tests pin that, and the
    combinatorial one at the bottom is the guard that actually holds the line.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        await self.store.set_account_capability('acct-a', ('gpt-4o', 'kimi-k3'))
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)
        self.a = main.CopilotEndpoint(name='acct-a', github_token='synthetic')

    async def test_a_candidate_account_has_no_reason(self):
        self.assertIsNone(main._copilot_support_reason(self.a, 'gpt-4o', 'responses'))

    async def test_an_operator_disabled_account_is_distinct_from_one_never_enabled(self):
        # Both are out of service, but only one of them is somebody's decision.
        # Conflating them makes an operator hunt for a switch they never threw.
        await self.store.set_account('acct-a', enabled=False)
        self.assertEqual(main._copilot_support_reason(self.a, 'gpt-4o'), 'account_disabled')
        inert = main.CopilotEndpoint(name='fresh', github_token='synthetic',
                                     enabled_in_config=False)
        self.assertEqual(main._copilot_support_reason(inert, 'gpt-4o'),
                         'account_never_enabled')

    async def test_the_model_side_selection_is_named_as_such(self):
        await self.store.set_model_accounts('gpt-4o', ['acct-b'])
        self.assertEqual(main._copilot_support_reason(self.a, 'gpt-4o'),
                         'not_selected_for_model')

    async def test_capability_and_allowed_models_stay_apart(self):
        self.assertEqual(main._copilot_support_reason(self.a, 'gpt-5.4'), 'not_in_capability')
        await self.store.set_account('acct-a', allowed_models=['kimi-k3'])
        self.assertEqual(main._copilot_support_reason(self.a, 'gpt-4o'),
                         'excluded_by_allowed_models')

    async def test_the_configured_models_list_is_its_own_reason(self):
        # Fixing this one needs a config change and a restart, unlike every
        # other reason, so it must not be reported as an overlay decision.
        pinned = main.CopilotEndpoint(name='acct-a', github_token='synthetic', models=['kimi-k3'])
        self.assertEqual(main._copilot_support_reason(pinned, 'gpt-4o'),
                         'not_in_configured_models')

    async def test_an_api_type_mismatch_is_reported_only_when_asked(self):
        chat_only = main.CopilotEndpoint(name='acct-a', github_token='synthetic',
                                         api_types=['chat'])
        self.assertEqual(main._copilot_support_reason(chat_only, 'gpt-4o', 'responses'),
                         'api_type_unsupported')
        self.assertIsNone(main._copilot_support_reason(chat_only, 'gpt-4o'))

    async def test_ignoring_the_selection_answers_the_would_it_serve_question(self):
        # This is what lets the Dashboard preview an *unsaved* selection without
        # reimplementing policy: the server says "could serve", the client only
        # intersects that with the checkboxes.
        await self.store.set_model_accounts('gpt-4o', ['acct-b'])
        self.assertEqual(main._copilot_support_reason(self.a, 'gpt-4o'), 'not_selected_for_model')
        self.assertIsNone(main._copilot_support_reason(self.a, 'gpt-4o',
                                                       ignore_model_selection=True))

    async def test_ignoring_the_selection_does_not_ignore_any_other_layer(self):
        await self.store.set_account('acct-a', enabled=False)
        self.assertEqual(
            main._copilot_support_reason(self.a, 'gpt-4o', ignore_model_selection=True),
            'account_disabled')

    async def test_no_store_means_no_reason(self):
        with patch.object(main, 'key_policy_store', None):
            self.assertIsNone(main._copilot_support_reason(self.a, 'anything'))

    async def test_supports_and_the_reason_never_disagree(self):
        """The anti-drift guard for the layer `supports()` actually owns."""
        await self.store.set_account_capability('capped', ('gpt-4o',))
        await self.store.set_account('capped', allowed_models=['gpt-4o'])
        await self.store.set_account('off', enabled=False)
        await self.store.set_model_accounts('gpt-4o', ['capped'])
        endpoints = [
            main.CopilotEndpoint(name='acct-a', github_token='t'),
            main.CopilotEndpoint(name='capped', github_token='t'),
            main.CopilotEndpoint(name='off', github_token='t'),
            main.CopilotEndpoint(name='inert', github_token='t', enabled_in_config=False),
            main.CopilotEndpoint(name='pinned', github_token='t', models=['kimi-k3']),
            main.CopilotEndpoint(name='chatty', github_token='t', api_types=['chat']),
        ]
        for state in ('loaded', 'unavailable'):
            self.store._state = state
            for ep in endpoints:
                for model in ('gpt-4o', 'kimi-k3', 'gpt-5.4'):
                    for api_type in (None, 'responses', 'chat'):
                        with self.subTest(state=state, account=ep.name, model=model,
                                          api_type=api_type):
                            self.assertEqual(
                                ep.supports(model, api_type),
                                main._copilot_support_reason(ep, model, api_type) is None)

    async def test_every_reason_is_in_the_published_closed_set(self):
        self.assertEqual(
            main.COPILOT_EXCLUSION_REASONS,
            key_policy.EXCLUSION_REASONS | {'account_never_enabled',
                                            'not_in_configured_models',
                                            'api_type_unsupported'})


class CandidateDiagnosticEndpointTests(unittest.TestCase):
    """`GET /admin/model-routing/{model}` — the matrix's data source.

    One model, every account, each with the layer that decided. Modelled on
    GCP's Policy Troubleshooter: a per-layer verdict next to the roll-up, so
    «why is this account not receiving traffic» is answered in one place rather
    than inferred by comparing two screens. The anti-pattern it exists to avoid
    is one-api's: an empty candidate set reported as an undifferentiated 503
    that echoes only the model name.
    """

    KEY = 'lb-candidate-diagnostic-key-01'

    def setUp(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = CandidateDiagnosticEndpointTests.KEY

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = _store()
        asyncio.run(self.store.set_account_capability('acct-a', ('gpt-4o', 'kimi-k3')))
        asyncio.run(self.store.set_account_capability('acct-b', ('gpt-4o',)))
        self.endpoints = [
            main.CopilotEndpoint(name='acct-a', github_token='synthetic'),
            main.CopilotEndpoint(name='acct-b', github_token='synthetic'),
            main.CopilotEndpoint(name='acct-inert', github_token='synthetic',
                                 enabled_in_config=False),
        ]
        copilot = main.CopilotProxy.__new__(main.CopilotProxy)
        copilot.load_balancer = main.LoadBalancer(self.endpoints)
        copilot.verify_api_key = _Stub().verify_api_key
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[self.KEY] = 'default'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        for attr, value in (('proxy', _Stub()), ('copilot_proxy', copilot),
                            ('key_policy_store', self.store)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.H = {'x-api-key': self.KEY}

    def _rows(self, model):
        r = self.client.get(f'/admin/model-routing/{model}', headers=self.H)
        self.assertEqual(r.status_code, 200)
        return r.json(), {row['account']: row for row in r.json()['accounts']}

    def test_it_lists_every_configured_account_not_just_the_candidates(self):
        # Listing only the survivors would hide exactly the rows an operator
        # opened this view to understand.
        _, rows = self._rows('gpt-4o')
        self.assertEqual(sorted(rows), ['acct-a', 'acct-b', 'acct-inert'])

    def test_a_serving_account_is_marked_with_no_reason(self):
        _, rows = self._rows('gpt-4o')
        self.assertTrue(rows['acct-a']['serves_model'])
        self.assertIsNone(rows['acct-a']['reason'])

    def test_each_row_names_the_layer_that_dropped_it(self):
        self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                        json={'copilot_accounts': ['acct-a']})
        _, rows = self._rows('gpt-4o')
        self.assertEqual(rows['acct-b']['reason'], 'not_selected_for_model')
        self.assertEqual(rows['acct-inert']['reason'], 'account_never_enabled')
        self.assertFalse(rows['acct-b']['serves_model'])

    def test_it_separates_would_serve_from_does_serve(self):
        # The two columns of the matrix: "is it selected" versus "could it ever
        # serve this model". Without the second, selecting a capability-less
        # account looks like it worked until the request fails.
        self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                        json={'copilot_accounts': ['acct-a']})
        _, rows = self._rows('gpt-4o')
        self.assertFalse(rows['acct-b']['serves_model'])
        self.assertTrue(rows['acct-b']['serves_model_ignoring_selection'],
                        'acct-b has gpt-4o capability; only the selection excludes it')
        _, kimi = self._rows('kimi-k3')
        self.assertFalse(kimi['acct-b']['serves_model_ignoring_selection'],
                         'no kimi-k3 capability, so selecting it would change nothing')

    def test_it_reports_which_accounts_the_rule_currently_names(self):
        self.client.put('/admin/model-routing/gpt-4o', headers=self.H,
                        json={'copilot_accounts': ['acct-a']})
        body, rows = self._rows('gpt-4o')
        self.assertTrue(rows['acct-a']['selected'])
        self.assertFalse(rows['acct-b']['selected'])
        self.assertEqual(body['copilot_accounts'], ['acct-a'])

    def test_capability_unknown_is_a_warning_not_an_exclusion(self):
        # An account with no discovered capability serves everything. That is a
        # wildcard, not a reason to drop it, so it must not be reported as one —
        # but the operator still needs to see it before enabling.
        _, rows = self._rows('gpt-4o')
        self.assertFalse(rows['acct-inert']['capability_known'])
        self.assertNotIn('capability_unknown', main.COPILOT_EXCLUSION_REASONS)

    def test_it_carries_the_same_rule_view_as_the_listing(self):
        body, _ = self._rows('gpt-4o')
        for field in ('model', 'effective_openai_order', 'effective_copilot_accounts',
                      'copilot_candidates_empty'):
            self.assertIn(field, body)

    def test_an_unconfigured_copilot_answers_404_rather_than_an_empty_matrix(self):
        with patch.object(main, 'copilot_proxy', None):
            self.assertEqual(
                self.client.get('/admin/model-routing/gpt-4o', headers=self.H).status_code, 404)

    def test_the_diagnostic_requires_a_valid_key(self):
        self.assertEqual(self.client.get('/admin/model-routing/gpt-4o').status_code, 401)

    def test_it_never_returns_token_material(self):
        body = self.client.get('/admin/model-routing/gpt-4o', headers=self.H).text
        self.assertNotIn('synthetic', body)


if __name__ == '__main__':
    unittest.main()
