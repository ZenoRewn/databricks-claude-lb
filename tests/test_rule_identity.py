"""One model, one rule — whichever of its three spellings names it.

Author: Zeno Ren.

The same Claude model is spelled three ways (`claude_model_identity`), and
per-model rules were keyed by the literal string. Two consequences, the second
of which is a silent correctness bug rather than an ergonomic one:

1. A rule configured on `databricks-claude-opus-5-5` does nothing for a client
   that sends `claude-opus-5-5`. Both dropdown entries exist in
   `known_models`, so the operator cannot tell which one the clients use.

2. **The GHCP account selection never applied to Claude on /v1/messages.**
   `_serve_messages_across_providers` rewrites the model to each provider's
   spelling *before* the pool runs, so `supports()` — and with it
   `_copilot_model_selection_reason` — sees `claude-opus-5.5`. A selection
   stored under the Databricks or Foundry spelling was therefore looked up
   under a key that cannot exist, and the limit silently did not subtract.

Identity matching fixes both: a Claude rule names the *model*, so any spelling
resolves to it. A non-Claude model has no canonical identity and keeps exact
matching — `gpt-5.4` and `gpt-5-4` are not known to be the same model and
guessing that they are would invent a routing rule nobody wrote.
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import claude_model_identity as claude_identity
import key_policy
import main


DASH = 'claude-opus-5-5'          # Foundry
DOTS = 'claude-opus-5.5'          # GitHub Copilot
ADB = 'databricks-claude-opus-5-5'  # Databricks
SPELLINGS = (DASH, DOTS, ADB)


class _StoreCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()


class PreferenceIdentityTests(_StoreCase):
    async def test_a_rule_named_in_one_spelling_applies_to_the_others(self):
        await self.store.set_model_preference(ADB, ['azure_anthropic'])
        for model in SPELLINGS:
            with self.subTest(model=model):
                self.assertEqual(self.store.model_preference(model), ('azure_anthropic',))

    async def test_writing_another_spelling_replaces_the_rule_rather_than_shadowing_it(self):
        await self.store.set_model_preference(ADB, ['azure_anthropic'])
        await self.store.set_model_preference(DOTS, ['copilot'])
        self.assertEqual(len(self.store.model_preferences_snapshot()), 1,
                         'one model must not hold two rules under two spellings')
        for model in SPELLINGS:
            with self.subTest(model=model):
                self.assertEqual(self.store.model_preference(model), ('copilot',))

    async def test_clearing_under_any_spelling_clears_the_rule(self):
        await self.store.set_model_preference(DOTS, ['copilot'])
        await self.store.clear_model_preference(ADB)
        self.assertEqual(self.store.model_preference(DOTS), ())
        self.assertEqual(self.store.model_preferences_snapshot(), {})

    async def test_a_legacy_collision_prefers_the_exact_spelling(self):
        # Rules written before identity matching can collide. The spelling the
        # caller asked for is the least surprising winner.
        self.store._preferences = {ADB: ('databricks',), DOTS: ('copilot',)}
        self.assertEqual(self.store.model_preference(ADB), ('databricks',))
        self.assertEqual(self.store.model_preference(DOTS), ('copilot',))
        # A third spelling matches neither exactly; the answer must at least be
        # stable across calls rather than depend on dict ordering.
        self.assertEqual(self.store.model_preference(DASH), self.store.model_preference(DASH))
        self.assertIn(self.store.model_preference(DASH), (('databricks',), ('copilot',)))

    async def test_a_non_claude_model_is_matched_exactly(self):
        await self.store.set_model_preference('gpt-5.4', ['copilot'])
        self.assertEqual(self.store.model_preference('gpt-5-4'), (),
                         'nothing says these are the same model')
        self.assertEqual(self.store.model_preference('gpt-5.4'), ('copilot',))

    async def test_a_different_version_is_a_different_model(self):
        await self.store.set_model_preference(ADB, ['azure_anthropic'])
        self.assertEqual(self.store.model_preference('claude-opus-5'), ())
        self.assertEqual(self.store.model_preference('claude-sonnet-5-5'), ())


class AccountSelectionIdentityTests(_StoreCase):
    async def test_a_selection_named_in_one_spelling_applies_to_the_others(self):
        await self.store.set_model_accounts(ADB, ['acct-a'])
        for model in SPELLINGS:
            with self.subTest(model=model):
                self.assertEqual(self.store.model_accounts(model), ('acct-a',))
                self.assertTrue(self.store.model_allows_account(model, 'acct-a'))
                self.assertFalse(self.store.model_allows_account(model, 'acct-b'))

    async def test_writing_another_spelling_replaces_the_selection(self):
        await self.store.set_model_accounts(ADB, ['acct-a'])
        await self.store.set_model_accounts(DASH, ['acct-b'])
        self.assertEqual(len(self.store.model_accounts_snapshot()), 1)
        self.assertEqual(self.store.model_accounts(DOTS), ('acct-b',))

    async def test_clearing_under_any_spelling_clears_the_selection(self):
        await self.store.set_model_accounts(DOTS, ['acct-a'])
        await self.store.clear_model_accounts(ADB)
        self.assertEqual(self.store.model_accounts(DASH), ())
        self.assertEqual(self.store.model_accounts_snapshot(), {})

    async def test_an_empty_selection_under_another_spelling_clears_the_limit(self):
        # Unchecking the last box is the UI's undo; it must find the rule even
        # when the operator is looking at a different spelling of the model.
        await self.store.set_model_accounts(ADB, ['acct-a'])
        await self.store.set_model_accounts(DOTS, [])
        self.assertEqual(self.store.model_accounts(ADB), ())
        self.assertEqual(self.store.model_accounts_snapshot(), {})

    async def test_a_non_claude_model_is_matched_exactly(self):
        await self.store.set_model_accounts('gpt-5.4', ['acct-a'])
        self.assertEqual(self.store.model_accounts('gpt-5-4'), ())

    async def test_the_json_backend_round_trips_through_identity(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            store = key_policy.KeyPolicyStore({'type': 'json', 'path': root})
            await store.start()
            await store.set_model_accounts(ADB, ['acct-a'])
            await store.set_model_preference(ADB, ['azure_anthropic'])
            reopened = key_policy.KeyPolicyStore({'type': 'json', 'path': root})
            await reopened.start()
            self.assertEqual(reopened.model_accounts(DOTS), ('acct-a',))
            self.assertEqual(reopened.model_preference(DOTS), ('azure_anthropic',))


class TheSelectionReachesTheCopilotSpellingTests(unittest.IsolatedAsyncioTestCase):
    """The silent bug: /v1/messages asks Copilot about Copilot's spelling.

    `_serve_messages_across_providers` rewrites the model before the pool runs,
    so the string that reaches `supports()` is `claude-opus-5.5` — never the
    Databricks or Foundry spelling the rule was stored under.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        for name in ('acct-a', 'acct-b'):
            await self.store.set_account_capability(name, (DOTS,))
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)
        self.a = main.CopilotEndpoint(name='acct-a', github_token='synthetic')
        self.b = main.CopilotEndpoint(name='acct-b', github_token='synthetic')

    async def test_a_selection_stored_in_the_databricks_spelling_still_subtracts(self):
        await self.store.set_model_accounts(ADB, ['acct-a'])
        forwarded = claude_identity.rewrite(ADB, 'copilot')
        self.assertEqual(forwarded, DOTS)
        self.assertTrue(self.a.supports(forwarded))
        self.assertFalse(self.b.supports(forwarded),
                         'the account limit must not be silently inert on /v1/messages')
        self.assertEqual(main._copilot_model_selection_reason(forwarded, 'acct-b'),
                         'not_selected_for_model')

    async def test_a_selection_stored_in_the_foundry_spelling_still_subtracts(self):
        await self.store.set_model_accounts(DASH, ['acct-a'])
        self.assertFalse(self.b.supports(DOTS))


class MessagesOrderFollowsIdentityTests(unittest.TestCase):
    def setUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        asyncio.run(self.store.start())
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def test_a_rule_on_one_spelling_orders_every_spelling(self):
        asyncio.run(self.store.set_model_preference(DOTS, ['azure_anthropic']))
        for model in SPELLINGS:
            with self.subTest(model=model):
                self.assertEqual(main._messages_provider_order(model)[0], 'azure_anthropic')

    def test_the_openai_entry_gate_follows_identity_too(self):
        # Opening claude on the OpenAI-style entries is per model, not per
        # spelling: a rule naming copilot opens the model it names.
        asyncio.run(self.store.set_model_preference(ADB, ['copilot']))
        for model in SPELLINGS:
            with self.subTest(model=model):
                self.assertTrue(main._model_pinned_to_openai_style_provider(model))
        self.assertFalse(main._model_pinned_to_openai_style_provider('claude-sonnet-5'))


class TheRoutingViewDoesNotContradictItselfTests(unittest.IsolatedAsyncioTestCase):
    """`effective_copilot_accounts` and `messages_paths` answered differently.

    Both describe the same hop. The path view already asks about the forwarded
    id (`ep.supports(forwarded)`), while the account view asked about the raw
    one — so for a Databricks-spelled Claude model the same response could say
    the Copilot path serves it *and* that no Copilot account does.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        await self.store.set_account_capability('acct-a', (DOTS,))
        ep = main.CopilotEndpoint(name='acct-a', github_token='synthetic')
        proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        proxy.load_balancer = main.LoadBalancer([ep])
        # `messages_pool` is a cached property that would build a real proxy;
        # pre-seed its backing field with a stand-in sharing the same load
        # balancer (as the real one does) so the view sees a configured copilot
        # pool without constructing an httpx client.
        proxy._messages_pool = SimpleNamespace(load_balancer=proxy.load_balancer)
        for target, value in (('key_policy_store', self.store), ('copilot_proxy', proxy),
                              ('proxy', None), ('azure_proxy', None)):
            p = patch.object(main, target, value)
            p.start()
            self.addCleanup(p.stop)

    def _copilot_path(self, view):
        return next(p for p in view['messages_paths'] if p['provider'] == 'copilot')

    async def test_the_account_view_agrees_with_the_path_view(self):
        for model in SPELLINGS:
            with self.subTest(model=model):
                view = main._model_routing_view(model)
                serving = self._copilot_path(view)['endpoints_serving_model']
                self.assertEqual(bool(view['effective_copilot_accounts']), bool(serving),
                                 'one response must not both claim and deny the hop')
                self.assertEqual(view['effective_copilot_accounts'], ['acct-a'])

    async def test_a_selection_in_another_spelling_shows_up_as_empty_candidates(self):
        await self.store.set_model_accounts(ADB, ['acct-b'])
        view = main._model_routing_view(DASH)
        self.assertEqual(view['effective_copilot_accounts'], [])
        self.assertTrue(view['copilot_candidates_empty'])


if __name__ == '__main__':
    unittest.main()
