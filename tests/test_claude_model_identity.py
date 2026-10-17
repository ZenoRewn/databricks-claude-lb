"""One Claude model, three provider naming conventions. Author: Zeno Ren.

Verified against the upstreams on 2026-10-17 (see
docs/reviews/2026-10-17-claude-three-paths/EVIDENCE.md). The same model is named
three different ways, and the differences are not cosmetic — asking Foundry for
`claude-opus-5.5` returns 404 `DeploymentNotFound`, because Foundry's deployment
names use dashes while GHCP's model ids use dots:

    Opus 5.5   ADB      databricks-claude-opus-5-5
               GHCP     claude-opus-5.5        <- dots
               Foundry  claude-opus-5-5        <- dashes

So a failover from ADB to GHCP to Foundry has to rewrite the model id at each
hop. This module owns that rewrite and nothing else.

What it deliberately does NOT own: whether a provider can actually serve the
model today. That is availability, not identity — ADB reads it from config,
GHCP from a discovered capability snapshot, Foundry from operator-declared
deployments (Foundry has no Models API at all: `GET /anthropic/v1/models`
returns 401). Baking the 2026-10-17 deployment inventory into code would turn a
point-in-time observation into a false permanent claim, which is exactly what
the capability catalog exists to avoid.
"""
import unittest

import claude_model_identity as identity


class CanonicalRecognitionTests(unittest.TestCase):
    def test_each_provider_spelling_maps_to_the_same_canonical_model(self):
        for spelling in ('databricks-claude-opus-5-5', 'claude-opus-5.5', 'claude-opus-5-5'):
            with self.subTest(spelling=spelling):
                self.assertEqual(identity.canonical(spelling), 'opus-5-5')

    def test_the_evidence_table_round_trips(self):
        # Every row verified against a live upstream on 2026-10-17.
        table = {
            'opus-5-5':   ('databricks-claude-opus-5-5',   'claude-opus-5.5',   'claude-opus-5-5'),
            'opus-5':     ('databricks-claude-opus-5',     'claude-opus-5',     'claude-opus-5'),
            'sonnet-5-5': ('databricks-claude-sonnet-5-5', 'claude-sonnet-5.5', 'claude-sonnet-5-5'),
            'sonnet-5':   ('databricks-claude-sonnet-5',   'claude-sonnet-5',   'claude-sonnet-5'),
            'haiku-5-5':  ('databricks-claude-haiku-5-5',  'claude-haiku-5.5',  'claude-haiku-5-5'),
            'haiku-4-5':  ('databricks-claude-haiku-4-5',  'claude-haiku-4.5',  'claude-haiku-4-5'),
            'opus-4-8':   ('databricks-claude-opus-4-8',   'claude-opus-4.8',   'claude-opus-4-8'),
        }
        for canonical, (adb, ghcp, foundry) in table.items():
            with self.subTest(canonical=canonical):
                self.assertEqual(identity.provider_model_id(canonical, 'databricks'), adb)
                self.assertEqual(identity.provider_model_id(canonical, 'copilot'), ghcp)
                self.assertEqual(identity.provider_model_id(canonical, 'azure_anthropic'), foundry)
                for spelling in (adb, ghcp, foundry):
                    self.assertEqual(identity.canonical(spelling), canonical)

    def test_a_single_segment_version_keeps_its_shape(self):
        # `opus-5` has no minor part, so dots and dashes agree. The rewrite must
        # not invent a `.0` or drop the segment.
        self.assertEqual(identity.provider_model_id('opus-5', 'copilot'), 'claude-opus-5')
        self.assertEqual(identity.provider_model_id('opus-5', 'azure_anthropic'), 'claude-opus-5')

    def test_fable_is_recognised_even_though_only_one_provider_has_it(self):
        # Identity is not availability: GHCP listed claude-fable-5.1 and Foundry
        # returned DeploymentNotFound for it. The name still has to parse.
        self.assertEqual(identity.canonical('claude-fable-5.1'), 'fable-5-1')
        self.assertEqual(identity.provider_model_id('fable-5-1', 'copilot'), 'claude-fable-5.1')

    def test_a_suffix_that_is_not_a_version_survives(self):
        # GHCP ships `claude-opus-4.8-fast`; the suffix is part of the identity.
        self.assertEqual(identity.canonical('claude-opus-4.8-fast'), 'opus-4-8-fast')
        self.assertEqual(identity.provider_model_id('opus-4-8-fast', 'copilot'), 'claude-opus-4.8-fast')

    def test_non_claude_models_are_not_claimed(self):
        for model in ('gpt-6.1-sol', 'gemini-3.8-flash', 'grok-4.7', 'kimi-k3',
                      'databricks-meta-llama-3', '', None):
            with self.subTest(model=model):
                self.assertIsNone(identity.canonical(model))

    def test_an_unknown_provider_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            identity.provider_model_id('opus-5-5', 'bedrock')

    def test_case_and_surrounding_space_do_not_change_identity(self):
        self.assertEqual(identity.canonical('  Claude-Opus-5.5 '), 'opus-5-5')

    def test_the_bare_family_alias_without_a_version_is_not_canonical(self):
        # `claude-opus` names no specific model. Resolving it would mean picking
        # a version on the user's behalf, which the repo's model resolution
        # deliberately never does.
        for model in ('claude-opus', 'claude', 'databricks-claude'):
            with self.subTest(model=model):
                self.assertIsNone(identity.canonical(model))


class DefaultOrderTests(unittest.TestCase):
    def test_claude_defaults_to_adb_then_ghcp_then_foundry(self):
        # Operator-set policy, 2026-10-17: ADB first, then GHCP, then Foundry.
        self.assertEqual(identity.DEFAULT_CLAUDE_ORDER,
                         ('databricks', 'copilot', 'azure_anthropic'))

    def test_the_order_names_only_anthropic_messages_providers(self):
        # azure_openai speaks the Azure OpenAI shape and cannot serve
        # /v1/messages, so it must not appear in this order.
        self.assertNotIn('azure_openai', identity.DEFAULT_CLAUDE_ORDER)
        for provider in identity.DEFAULT_CLAUDE_ORDER:
            with self.subTest(provider=provider):
                self.assertIn(provider, identity.MESSAGES_PROVIDERS)

    def test_every_messages_provider_has_a_naming_rule(self):
        for provider in identity.MESSAGES_PROVIDERS:
            with self.subTest(provider=provider):
                self.assertIsInstance(identity.provider_model_id('opus-5-5', provider), str)


class RewriteTests(unittest.TestCase):
    """The operation a failover actually performs."""

    def test_rewriting_for_the_next_hop_changes_only_the_spelling(self):
        self.assertEqual(identity.rewrite('databricks-claude-opus-5-5', 'copilot'), 'claude-opus-5.5')
        self.assertEqual(identity.rewrite('claude-opus-5.5', 'azure_anthropic'), 'claude-opus-5-5')
        self.assertEqual(identity.rewrite('claude-opus-5-5', 'databricks'), 'databricks-claude-opus-5-5')

    def test_rewriting_a_non_claude_model_returns_it_unchanged(self):
        # The caller should not have to special-case this: a gpt model simply
        # has no Claude identity to rewrite.
        self.assertEqual(identity.rewrite('gpt-6.1-sol', 'copilot'), 'gpt-6.1-sol')

    def test_rewriting_is_idempotent_within_one_provider(self):
        once = identity.rewrite('claude-opus-5.5', 'copilot')
        self.assertEqual(once, 'claude-opus-5.5')
        self.assertEqual(identity.rewrite(once, 'copilot'), once)


if __name__ == '__main__':
    unittest.main()
