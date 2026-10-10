"""The /models catalog must match what upstream actually serves. Author: Zeno Ren.

Verified on 2026-10-10 with minimal live requests against the deployed gateway.
Of the 25 ids the catalog used to return, 14 were dead (404 "rejected model", or
400 for an integrator restriction), 7 working models were missing, and no
Databricks Claude model was listed at all even though 12 of them answered on
/v1/messages.

The probes also showed the flat list cannot express a real distinction: several
models answer only on /v1/responses ("not accessible via the /chat/completions
endpoint"), gemini-3.8-flash answers only on chat ("does not support Responses
API"), and gpt-5.6-luna answers on both. Picking the wrong entry point is a 400,
so each verified entry carries supported_endpoints.

The catalog is an operator-verified point-in-time snapshot, not a provider
capability certification.
"""
import unittest
from unittest.mock import patch

import copilot_pricing
import main


RESPONSES = '/v1/responses'
CHAT = '/v1/chat/completions'
MESSAGES = '/v1/messages'

# 404 "rejected model" upstream, except gpt-5.6-cyber / gpt-5.4-nano which are
# integrator-scoped 400s. Either way they must not be advertised.
VERIFIED_UNAVAILABLE = (
    'gpt-5', 'gpt-5-nano', 'gpt-5-pro', 'gpt-5.1', 'gpt-5.2',
    'gpt-5-codex', 'gpt-5.1-codex',
    'o3', 'o3-mini', 'o3-pro', 'o4-mini',
    'gemini-2.5-pro', 'gemini-2.5-flash',
    'gpt-5.6-cyber', 'gpt-5.4-nano',
    'gpt-6-astra', 'kimi-k3',
)


class CatalogMatchesProbedRealityTests(unittest.TestCase):
    def test_dead_models_are_not_advertised(self):
        catalog = main.OPENAI_COMPAT_DEFAULT_MODELS
        for model in VERIFIED_UNAVAILABLE:
            with self.subTest(model=model):
                self.assertNotIn(model, catalog)

    def test_newly_verified_models_are_advertised(self):
        expected = {
            'gpt-6-luna': (RESPONSES,),
            'gpt-6-sol': (RESPONSES,),
            'gpt-6.1-sol': (RESPONSES,),
            'gpt-5.4-mini': (RESPONSES,),
            'grok-4.7': (RESPONSES,),
            'mai-code-1.1-flash': (RESPONSES,),
            'gemini-3.8-flash': (CHAT,),
        }
        for model, endpoints in expected.items():
            with self.subTest(model=model):
                self.assertEqual(main.OPENAI_COMPAT_DEFAULT_MODELS.get(model), endpoints)

    def test_previously_listed_models_that_still_work_are_kept(self):
        expected = {
            'gpt-5-mini': (RESPONSES,),
            'gpt-5.4': (RESPONSES,),
            'gpt-5.5': (RESPONSES,),
            'gpt-5.6-sol': (RESPONSES,),
            'gpt-5.6-terra': (RESPONSES,),
            'gpt-5.3-codex': (RESPONSES,),
            'gpt-4.1': (CHAT,),
            'gpt-4.1-2025-04-14': (CHAT,),
            'gpt-4o': (CHAT,),
            'gpt-4o-mini': (CHAT,),
        }
        for model, endpoints in expected.items():
            with self.subTest(model=model):
                self.assertEqual(main.OPENAI_COMPAT_DEFAULT_MODELS.get(model), endpoints)

    def test_dual_endpoint_model_lists_both(self):
        self.assertEqual(set(main.OPENAI_COMPAT_DEFAULT_MODELS['gpt-5.6-luna']), {RESPONSES, CHAT})

    def test_every_openai_entry_declares_a_known_endpoint(self):
        for model, endpoints in main.OPENAI_COMPAT_DEFAULT_MODELS.items():
            with self.subTest(model=model):
                self.assertTrue(endpoints, 'an entry with no endpoint is not discoverable')
                self.assertTrue(set(endpoints) <= {RESPONSES, CHAT}, endpoints)

    def test_legacy_id_tuple_still_exposed_for_callers(self):
        self.assertEqual(tuple(sorted(main.OPENAI_COMPAT_DEFAULT_MODEL_IDS)),
                         tuple(sorted(main.OPENAI_COMPAT_DEFAULT_MODELS)))


class DatabricksClaudeCatalogTests(unittest.TestCase):
    def test_all_probed_claude_models_are_listed_on_messages(self):
        expected = {
            'databricks-claude-opus-5-5', 'databricks-claude-opus-5',
            'databricks-claude-opus-4-8', 'databricks-claude-opus-4-7',
            'databricks-claude-opus-4-6', 'databricks-claude-opus-4-5',
            'databricks-claude-sonnet-5-5', 'databricks-claude-sonnet-5',
            'databricks-claude-sonnet-4-6', 'databricks-claude-sonnet-4-5',
            'databricks-claude-haiku-5-5', 'databricks-claude-haiku-4-5',
        }
        self.assertEqual(set(main.DATABRICKS_CLAUDE_MODELS), expected)
        for model, endpoints in main.DATABRICKS_CLAUDE_MODELS.items():
            with self.subTest(model=model):
                self.assertEqual(endpoints, (MESSAGES,))

    def test_five_five_is_discoverable(self):
        for model in ('databricks-claude-opus-5-5', 'databricks-claude-sonnet-5-5',
                      'databricks-claude-haiku-5-5'):
            with self.subTest(model=model):
                self.assertIn(model, main.DATABRICKS_CLAUDE_MODELS)

    def test_region_limited_fable_is_not_advertised(self):
        # 2 of 6 live samples succeeded; the rest returned NOT_FOUND "not
        # available in your region". Listing it would advertise a model that
        # fails on most endpoints. It stays reachable by explicit request.
        for model in main.DATABRICKS_CLAUDE_MODELS:
            with self.subTest(model=model):
                self.assertNotIn('fable', model)

    def test_claude_models_are_never_offered_on_openai_style_entry_points(self):
        for model, endpoints in main.DATABRICKS_CLAUDE_MODELS.items():
            with self.subTest(model=model):
                self.assertNotIn(RESPONSES, endpoints)
                self.assertNotIn(CHAT, endpoints)


class CatalogPricingTests(unittest.TestCase):
    """Cost must not vanish; which table prices a model depends on its provider.

    MODEL_PRICING carries public Anthropic/OpenAI/Google API list prices, so a
    Copilot-only model such as grok-4.7 or mai-code-1.1-flash has no entry there
    and inventing one would be fabrication. historical_model_cost asks
    copilot_pricing first for non-Anthropic models and only falls back to
    MODEL_PRICING, so either table is sufficient.
    """

    def test_every_catalog_model_is_priceable_by_some_table(self):
        for model in (*main.OPENAI_COMPAT_DEFAULT_MODELS, *main.DATABRICKS_CLAUDE_MODELS):
            with self.subTest(model=model):
                self.assertTrue(
                    main.get_model_pricing(model) or copilot_pricing.get_pricing(model),
                    f'{model}: neither MODEL_PRICING nor copilot_pricing can price it')

    def test_claude_models_are_priced_by_the_anthropic_table(self):
        for model in main.DATABRICKS_CLAUDE_MODELS:
            with self.subTest(model=model):
                self.assertIsNotNone(main.get_model_pricing(model))

    def test_copilot_only_models_need_no_generic_api_price(self):
        for model in ('grok-4.7', 'mai-code-1.1-flash', 'gpt-6-sol'):
            with self.subTest(model=model):
                self.assertIsNotNone(copilot_pricing.get_pricing(model))


class ModelEntryShapeTests(unittest.TestCase):
    def test_verified_entry_carries_supported_endpoints(self):
        entry = main._openai_model_entry('gpt-6-sol', (RESPONSES,))
        self.assertEqual(entry['id'], 'gpt-6-sol')
        self.assertEqual(entry['object'], 'model')
        self.assertEqual(entry['supported_endpoints'], [RESPONSES])

    def test_unverified_entry_omits_the_field_rather_than_guessing(self):
        # Azure deployments and explicit Copilot allow-lists come from config;
        # their entry-point support is unknown here, so claim nothing.
        self.assertNotIn('supported_endpoints', main._openai_model_entry('some-azure-deployment'))

    def test_openai_required_fields_are_preserved(self):
        for entry in (main._openai_model_entry('gpt-4o', (CHAT,)),
                      main._openai_model_entry('x')):
            with self.subTest(entry=entry['id']):
                self.assertEqual(set(entry) >= {'id', 'object', 'created', 'owned_by'}, True)


class CatalogAssemblyTests(unittest.TestCase):
    def test_catalog_includes_both_providers_when_wildcard(self):
        copilot = _wildcard_copilot()
        catalog = main._collect_model_catalog(azure=None, copilot=copilot, databricks=object())
        self.assertIn('gpt-6-sol', catalog)
        self.assertIn('databricks-claude-opus-5-5', catalog)
        self.assertEqual(catalog['databricks-claude-opus-5-5'], (MESSAGES,))

    def test_claude_models_absent_when_databricks_not_configured(self):
        # databricks=None falls back to the module global, so pin it rather than
        # depending on whatever this test session left configured.
        with patch.object(main, 'proxy', None):
            catalog = main._collect_model_catalog(azure=None, copilot=_wildcard_copilot(), databricks=None)
        self.assertNotIn('databricks-claude-opus-5-5', catalog)
        self.assertIn('gpt-6-sol', catalog)

    def test_explicit_copilot_allowlist_is_listed_without_endpoint_claims(self):
        copilot = _copilot_with(['some-private-model'])
        with patch.object(main, 'proxy', None):
            catalog = main._collect_model_catalog(azure=None, copilot=copilot, databricks=None)
        self.assertEqual(catalog.get('some-private-model'), ())
        # An explicit allow-list is not a wildcard, so the static catalog is not mixed in.
        self.assertNotIn('gpt-6-sol', catalog)

    def test_id_view_matches_catalog_keys(self):
        copilot = _wildcard_copilot()
        catalog = main._collect_model_catalog(azure=None, copilot=copilot, databricks=object())
        ids = main._collect_openai_model_ids(azure=None, copilot=copilot, databricks=object())
        self.assertEqual(ids, sorted(catalog))


class _Endpoint:
    def __init__(self, models=None, deployments=None):
        self.models = models or []
        self.deployments = deployments or []


class _Proxy:
    def __init__(self, endpoints):
        self.load_balancer = type('LB', (), {'endpoints': endpoints})()


def _wildcard_copilot():
    return _Proxy([_Endpoint(models=[])])


def _copilot_with(models):
    return _Proxy([_Endpoint(models=models)])


if __name__ == '__main__':
    unittest.main()
