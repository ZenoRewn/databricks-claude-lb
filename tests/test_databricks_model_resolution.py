"""Resolve Databricks Claude model names by normalising, never by guessing. Author: Zeno Ren.

The gateway used to sniff versions with a regex whose separator class also matched
the second dot in "5.5", so an explicit Opus 5.5 request silently ran on Opus 5 —
no error, only an INFO line. The same class of defect left Haiku 5.5 two tiers
down and turned Fable into Sonnet.

Upstream inventory verified on 2026-10-10 with minimal live requests:
  opus   4-5 4-6 4-7 4-8 5 5-5   (5-1, 6 and *-fast rejected)
  sonnet 4-6 5 5-5               (5-1 rejected)
  haiku  4-5 5-5                 (there is no haiku-5)
  fable  5 5-1
So "a minor version falls back to its base version" was itself wrong: 5-1 does
not exist and 5-5 is a separate model. An explicit version is therefore
normalised and passed through, leaving upstream as the only authority on
existence.
"""
import unittest

import main


class FiveFiveResolutionTests(unittest.TestCase):
    def test_five_five_is_not_swallowed_by_five(self):
        for family in ('opus', 'sonnet', 'haiku'):
            for sep in ('-', '.', '_'):
                name = f'claude-{family}-5{sep}5'
                with self.subTest(name=name):
                    self.assertEqual(main.get_databricks_model(name),
                                     f'databricks-claude-{family}-5-5')

    def test_fable_keeps_its_own_family(self):
        # Previously fell through to DEFAULT_MODEL, i.e. a cross-family swap.
        self.assertEqual(main.get_databricks_model('claude-fable-5-1'),
                         'databricks-claude-fable-5-1')
        self.assertEqual(main.get_databricks_model('claude-fable-5'),
                         'databricks-claude-fable-5')

    def test_five_five_survives_date_and_latest_suffixes(self):
        for suffix in ('-latest', '-20260101', '_20260101'):
            with self.subTest(suffix=suffix):
                self.assertEqual(main.get_databricks_model(f'claude-opus-5-5{suffix}'),
                                 'databricks-claude-opus-5-5')

    def test_case_insensitive(self):
        for name in ('CLAUDE-OPUS-5-5', 'Claude-Opus-5.5'):
            with self.subTest(name=name):
                self.assertEqual(main.get_databricks_model(name),
                                 'databricks-claude-opus-5-5')


class ExistingContractTests(unittest.TestCase):
    """Everything that already worked must keep working."""

    def test_opus_5_variants_unchanged(self):
        for name in ('databricks-claude-opus-5', 'opus-5', 'claude-opus-5',
                     'claude-opus-5-latest', 'claude-opus-5-20250514',
                     'claude-opus-5-20260101', 'claude-opus-5_20260101',
                     'CLAUDE-OPUS-5', 'Claude-Opus-5'):
            with self.subTest(name=name):
                self.assertEqual(main.get_databricks_model(name), 'databricks-claude-opus-5')

    def test_explicit_legacy_versions_unchanged(self):
        for name, expected in (('claude-opus-4-7', 'databricks-claude-opus-4-7'),
                               ('claude-opus-4-5', 'databricks-claude-opus-4-5'),
                               ('claude-opus-4.8', 'databricks-claude-opus-4-8'),
                               ('claude-sonnet-4-6', 'databricks-claude-sonnet-4-6'),
                               ('claude-sonnet-5', 'databricks-claude-sonnet-5'),
                               ('claude-haiku-4-5', 'databricks-claude-haiku-4-5')):
            with self.subTest(name=name):
                self.assertEqual(main.get_databricks_model(name), expected)

    def test_databricks_prefixed_names_pass_through_untouched(self):
        for name in ('databricks-claude-opus-5-5', 'databricks-claude-haiku-5-5',
                     'databricks-claude-fable-5-1'):
            with self.subTest(name=name):
                self.assertEqual(main.get_databricks_model(name), name)


class FamilyDefaultTests(unittest.TestCase):
    def test_bare_family_uses_the_current_default(self):
        for name, expected in (('claude-opus', 'databricks-claude-opus-5-5'),
                               ('claude-sonnet', 'databricks-claude-sonnet-5-5'),
                               ('claude-haiku', 'databricks-claude-haiku-5-5'),
                               ('claude-fable', 'databricks-claude-fable-5-1')):
            with self.subTest(name=name):
                self.assertEqual(main.get_databricks_model(name), expected)

    def test_dateonly_legacy_names_fall_back_to_the_family_default(self):
        # "claude-3-opus-20240229" carries no version after the family token.
        self.assertEqual(main.get_databricks_model('claude-3-opus-20240229'),
                         'databricks-claude-opus-5-5')
        self.assertEqual(main.get_databricks_model('claude-opus-latest'),
                         'databricks-claude-opus-5-5')

    def test_unrecognised_family_still_falls_back_to_default_model(self):
        self.assertEqual(main.get_databricks_model('gpt-5.4'), main.DEFAULT_MODEL)
        self.assertEqual(main.DEFAULT_MODEL, 'databricks-claude-sonnet-5-5')


class UnknownVersionTests(unittest.TestCase):
    """Upstream, not this table, decides whether a version exists."""

    def test_unknown_version_is_passed_through_not_downgraded(self):
        # Verified rejected upstream; a loud 400 beats silently serving opus-5.
        self.assertEqual(main.get_databricks_model('claude-opus-5.1'),
                         'databricks-claude-opus-5-1')
        self.assertEqual(main.get_databricks_model('claude-sonnet-5-1'),
                         'databricks-claude-sonnet-5-1')

    def test_future_major_version_needs_no_code_change(self):
        self.assertEqual(main.get_databricks_model('claude-opus-6'),
                         'databricks-claude-opus-6')

    def test_unknown_qualifier_is_preserved_not_swallowed(self):
        # Dropping "-fast" would serve the standard model at half the real rate.
        self.assertEqual(main.get_databricks_model('claude-opus-5-5-fast'),
                         'databricks-claude-opus-5-5-fast')


class NewModelPricingTests(unittest.TestCase):
    """A missing entry makes get_model_pricing return None and the cost vanish."""

    EXPECTED = {
        'databricks-claude-opus-5-5':   {'input': 4.00, 'output': 20.00, 'cache_write': 5.00,  'cache_read': 0.20},
        'databricks-claude-sonnet-5-5': {'input': 2.00, 'output': 10.00, 'cache_write': 2.50,  'cache_read': 0.10},
        'databricks-claude-haiku-5-5':  {'input': 0.10, 'output': 0.50,  'cache_write': 0.125, 'cache_read': 0.01},
        'databricks-claude-fable-5-1':  {'input': 10.00, 'output': 50.00, 'cache_write': 12.50, 'cache_read': 0.25},
        'databricks-claude-fable-5':    {'input': 10.00, 'output': 50.00, 'cache_write': 12.50, 'cache_read': 1.00},
    }

    def test_official_rates(self):
        # claude.com/pricing, fetched 2026-10-10.
        for model, rates in self.EXPECTED.items():
            with self.subTest(model=model):
                self.assertEqual(main.get_model_pricing(model), rates)

    def test_longer_keys_win_over_their_prefixes(self):
        # "opus-5-5" must not be priced by "opus-5", nor "fable-5-1" by "fable-5".
        self.assertNotEqual(main.get_model_pricing('databricks-claude-opus-5-5'),
                            main.get_model_pricing('databricks-claude-opus-5'))
        self.assertEqual(main.get_model_pricing('databricks-claude-opus-5')['input'], 5.00)
        self.assertEqual(main.get_model_pricing('databricks-claude-fable-5-1')['cache_read'], 0.25)
        self.assertEqual(main.get_model_pricing('databricks-claude-fable-5')['cache_read'], 1.00)

    def test_cost_is_computable_for_every_new_model(self):
        for model in self.EXPECTED:
            with self.subTest(model=model):
                self.assertIsNotNone(main.calculate_cost(model, 1000, 200, 100, 400))


class HistoricalAttributionTests(unittest.TestCase):
    def test_fable_is_priced_on_the_anthropic_basis(self):
        # Anthropic input excludes caches; without the 'fable-' prefix a bare
        # fable-* row was routed through the OpenAI-style inclusive branch.
        stats = {'requests': 1, 'input_tokens': 1000, 'output_tokens': 200,
                 'cache_creation_tokens': 100, 'cache_read_tokens': 400}
        quote = main.historical_model_cost('fable-5-1', stats)
        self.assertEqual(quote['pricing_status'], 'complete')
        self.assertEqual(quote['estimated_cost_usd'],
                         main.calculate_cost('fable-5-1', 1000, 200, 100, 400))


if __name__ == '__main__':
    unittest.main()
