"""The /v1/messages surface has three providers, not one. Author: Zeno Ren.

Verified 2026-10-17 (docs/reviews/2026-10-17-claude-three-paths/EVIDENCE.md):
Databricks, GitHub Copilot and Microsoft Foundry all serve Claude over the
Anthropic Messages protocol. Copilot declares
`supported_endpoints: ["/v1/messages", "/chat/completions"]` for every Claude
entry, and Foundry answers `POST /anthropic/v1/messages` with a standard
`type: "message"` body.

So provider order is per *protocol surface*, not global:

* the OpenAI-style surface (/v1/responses, /v1/chat/completions) orders
  copilot and azure_openai — `databricks` cannot serve it
* the Anthropic Messages surface (/v1/messages) orders databricks, copilot and
  azure_anthropic — `azure_openai` cannot serve it

One stored preference list feeds both, each filtering to what its surface can
actually reach. That keeps the existing "a preference reorders, it never
removes" rule intact on both surfaces.
"""
import unittest
from unittest.mock import patch

import key_policy
import main


class ProviderRegistryTests(unittest.TestCase):
    def test_foundry_anthropic_is_a_provider_of_its_own(self):
        # Not folded into azure_openai: different protocol, different auth
        # header (x-api-key, not api-key), different model ids, and CCU billing
        # instead of per-token list prices. A per-key allowlist that cannot
        # distinguish them cannot express "gpt on Foundry but not Claude".
        self.assertIn('azure_anthropic', key_policy.PROVIDER_IDS)
        self.assertIn('azure_anthropic', key_policy.ALL_PROVIDERS)
        self.assertEqual(key_policy.PROVIDER_LABELS['azure_anthropic'], 'Foundry Claude')

    def test_the_existing_three_provider_ids_keep_their_meaning(self):
        for provider, label in (('databricks', 'ADB'), ('copilot', 'GHCP')):
            with self.subTest(provider=provider):
                self.assertEqual(key_policy.PROVIDER_LABELS[provider], label)

    def test_every_provider_id_has_a_label(self):
        self.assertEqual(set(key_policy.PROVIDER_IDS), set(key_policy.PROVIDER_LABELS))

    def test_an_allowlist_may_name_the_new_provider(self):
        self.assertEqual(key_policy.normalise_providers(['azure_anthropic']), ('azure_anthropic',))

    def test_an_unknown_provider_is_still_rejected(self):
        with self.assertRaises(ValueError):
            key_policy.normalise_providers(['bedrock'])


class MessagesOrderTests(unittest.TestCase):
    def setUp(self):
        self.preference = ()
        store = type('Store', (), {
            'model_preference': lambda _self, model: self.preference,
            'model_accounts_exclusion_reason': lambda *a: None,
        })()
        patcher = patch.object(main, 'key_policy_store', store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_default_is_adb_then_ghcp_then_foundry(self):
        self.assertEqual(main._messages_provider_order('claude-opus-5-5'),
                         ('databricks', 'copilot', 'azure_anthropic'))

    def test_a_preference_reorders_without_removing(self):
        # Fallback is never lost: unlisted providers keep the default order
        # behind the listed ones.
        self.preference = ('copilot',)
        self.assertEqual(main._messages_provider_order('claude-opus-5-5'),
                         ('copilot', 'databricks', 'azure_anthropic'))

    def test_a_preference_may_put_foundry_first(self):
        self.preference = ('azure_anthropic', 'databricks')
        self.assertEqual(main._messages_provider_order('claude-opus-5-5'),
                         ('azure_anthropic', 'databricks', 'copilot'))

    def test_azure_openai_in_a_preference_is_ignored_on_this_surface(self):
        # It speaks the Azure OpenAI shape; it cannot serve /v1/messages.
        self.preference = ('azure_openai', 'copilot')
        self.assertEqual(main._messages_provider_order('claude-opus-5-5'),
                         ('copilot', 'databricks', 'azure_anthropic'))

    def test_the_openai_surface_order_is_unchanged_by_this_work(self):
        # Regression guard: adding a messages surface must not let databricks
        # or azure_anthropic leak into the OpenAI-style order.
        self.preference = ('databricks', 'azure_anthropic')
        self.assertEqual(main._openai_provider_order('gpt-6.1-sol'),
                         ('copilot', 'azure_openai'))

    def test_both_surfaces_read_the_same_stored_preference(self):
        self.preference = ('copilot',)
        self.assertEqual(main._messages_provider_order('claude-opus-5-5')[0], 'copilot')
        self.assertEqual(main._openai_provider_order('gpt-6.1-sol')[0], 'copilot')


class PricingSeparationTests(unittest.TestCase):
    """A Foundry model id must not silently inherit Anthropic list prices.

    `MODEL_PRICING` matches on a substring of the model name, so Foundry's
    `claude-opus-5-5` would otherwise hit the `opus-5-5` key and be costed at
    Anthropic's public per-token rate. Foundry bills in Claude Consumption
    Units through the Azure Marketplace, so that number would be confidently
    wrong — and CLAUDE.md already forbids mixing the two口径. Unknown is the
    correct answer until a CCU rate is configured.
    """

    def test_a_databricks_claude_model_still_uses_the_public_rate(self):
        self.assertIsNotNone(main._pricing_for_model('databricks-claude-opus-5-5',
                                                     provider='databricks'))

    def test_a_foundry_claude_model_has_no_public_rate(self):
        self.assertIsNone(main._pricing_for_model('claude-opus-5-5',
                                                  provider='azure_anthropic'))

    def test_the_same_canonical_model_differs_by_provider(self):
        # Identical model, different billing relationship. This is the whole
        # point of keeping the provider in the pricing lookup.
        adb = main._pricing_for_model('databricks-claude-opus-5-5', provider='databricks')
        foundry = main._pricing_for_model('claude-opus-5-5', provider='azure_anthropic')
        self.assertIsNotNone(adb)
        self.assertIsNone(foundry)

    def test_an_unpriced_provider_is_reported_as_unknown_not_zero(self):
        # Zero would roll into dashboard totals as "free"; unknown must stay
        # unknown, which the Dashboard already renders as such.
        self.assertIsNone(main._pricing_for_model('claude-sonnet-5-5',
                                                  provider='azure_anthropic'))

    def test_pricing_without_a_provider_keeps_the_legacy_behaviour(self):
        # Historical callers pass no provider; they must keep working.
        self.assertIsNotNone(main._pricing_for_model('databricks-claude-opus-5-5'))


class RoutingNoteTests(unittest.TestCase):
    def test_the_api_note_does_not_claim_databricks_opens_the_openai_entries(self):
        """The shipped note said something the code does not do.

        It read: "'databricks' in a preference only opens claude-* on the
        OpenAI-style entry points". But the gate is
        `_model_pinned_to_openai_style_provider`, which counts only copilot and
        azure_openai — a databricks-only preference opens nothing there. An
        operator following that sentence would configure a rule that silently
        does not work.
        """
        note = main._model_routing_note()
        self.assertNotIn("'databricks' in a preference only opens", note)
        # And it must say what actually happens on each surface.
        self.assertIn('/v1/messages', note)
        self.assertIn('azure_anthropic', note)


if __name__ == '__main__':
    unittest.main()
