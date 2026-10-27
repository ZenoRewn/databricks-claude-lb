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

Re-probed 2026-10-26 against the deployed gateway, and the snapshot had drifted
in both directions. Three advertised ids were gone from GHCP (`gpt-5-mini`,
`gpt-5.4-mini`, `gpt-5.5` → 404 «no provider available»), while two ids this
file had recorded as unavailable now answer: `gpt-6-astra` (also a release
business probe) and `gpt-5.4-nano`. The nano probe needed a real output budget —
at 16 tokens the whole allowance went to reasoning items and the text came back
empty, which reads exactly like a dead model if you do not look at `status`.

Four newly verified ids are added, and two deliberate omissions are recorded
below as tests rather than prose, because both will look like oversights later.

The catalog is an operator-verified point-in-time snapshot, not a provider
capability certification.
"""
import unittest
from unittest.mock import patch

import copilot_pricing
import key_policy
import main


class _FakeLoadBalancer:
    def __init__(self, endpoints):
        self.endpoints = endpoints


class _FakeProxy:
    def __init__(self, endpoints):
        self.load_balancer = _FakeLoadBalancer(endpoints)


RESPONSES = '/v1/responses'
CHAT = '/v1/chat/completions'
MESSAGES = '/v1/messages'

# 404 "rejected model" upstream, except gpt-5.6-cyber which is an
# integrator-scoped 400. Either way they must not be advertised.
#
# `gpt-5-mini` / `gpt-5.4-mini` / `gpt-5.5` joined this list on 2026-10-26: GHCP
# no longer lists them for either account and the gateway answers 404 «no
# provider available». Their *pricing* entries stay — those price past usage,
# and deleting them would turn historical cost into unknown.
VERIFIED_UNAVAILABLE = (
    'gpt-5', 'gpt-5-nano', 'gpt-5-pro', 'gpt-5.1', 'gpt-5.2',
    'gpt-5-codex', 'gpt-5.1-codex',
    'o3', 'o3-mini', 'o3-pro', 'o4-mini',
    'gemini-2.5-pro', 'gemini-2.5-flash',
    'gpt-5.6-cyber',
    'gpt-5-mini', 'gpt-5.4-mini', 'gpt-5.5',
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
            'grok-4.7': (RESPONSES,),
            'mai-code-1.1-flash': (RESPONSES,),
            'gemini-3.8-flash': (CHAT,),
            # 2026-10-26 probes. Upstream declares /responses for all four and
            # answers 400 `unsupported_api_for_model` on chat, so neither id
            # may carry CHAT.
            'gpt-6.1-terra': (RESPONSES,),
            'gpt-6-astra': (RESPONSES,),
            'gpt-5.4-nano': (RESPONSES,),
            'grok-4.6': (RESPONSES,),
            # The mirror image: 400 on /v1/responses, 200 on chat.
            'gemini-4-argon': (CHAT,),
        }
        for model, endpoints in expected.items():
            with self.subTest(model=model):
                self.assertEqual(main.OPENAI_COMPAT_DEFAULT_MODELS.get(model), endpoints)

    def test_previously_listed_models_that_still_work_are_kept(self):
        expected = {
            # gpt-5.4 is kept on different evidence from the rest: GHCP dropped
            # it, but it is the deployment every configured Azure OpenAI
            # endpoint declares, so the OpenAI-style entry points still route it
            # there. On 2026-10-26 that hop answered 401 «invalid subscription
            # key or wrong API endpoint» — a credential defect to fix, not proof
            # the model is gone. Removing the entry would hide the defect.
            'gpt-5.4': (RESPONSES,),
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

    def test_upstream_models_without_a_published_price_stay_out(self):
        """`gpt-5.6-sol-fast` and `mai-code-2-flash` answer, and still are not advertised.

        Both returned 200 on /v1/responses on 2026-10-26, so availability is not
        the reason. GitHub's pricing reference lists neither, and this module
        already guarantees every advertised model is priceable by some table —
        advertising them would either break that guard or invite an invented
        rate. They stay reachable by explicit request, the same treatment Fable
        gets for its region limit.

        `mai-code-2-flash` carries a second lesson: upstream declared
        `/chat/completions` for it, and that entry point answered 400 «no model
        endpoints available given user constraints». An upstream declaration is
        not a verification.
        """
        for model in ('gpt-5.6-sol-fast', 'mai-code-2-flash'):
            with self.subTest(model=model):
                self.assertNotIn(model, main.OPENAI_COMPAT_DEFAULT_MODELS)
                self.assertIsNone(copilot_pricing.get_pricing(model))

    def test_a_model_blocked_only_by_operator_policy_stays_out(self):
        """kimi-k3 is still omitted, but no longer for the reason recorded in 2026-10.

        Upstream *does* list it for zenoren-icloud now. The gateway cannot serve
        it because the enterprise account lacks it (`not_in_capability`) and the
        business account's `allowed_models` excludes it
        (`excluded_by_allowed_models`) — operator policy, not upstream absence.
        Advertising it would promise a 404. Whoever widens that allow-list
        should revisit this entry rather than read the omission as upstream fact.
        """
        self.assertNotIn('kimi-k3', main.OPENAI_COMPAT_DEFAULT_MODELS)

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


class DiscoveredCapabilityKeepsVerifiedEntryPointsTests(unittest.IsolatedAsyncioTestCase):
    """A discovered model must still carry the entry points we verified.

    Availability comes from discovery — the stored snapshot is more accurate
    than any hand list about *whether* an account serves a model. But the
    snapshot is a list of ids and nothing else, so taking it also meant
    discarding the verified entry points: on 2026-10-26 the live `/v1/models`
    returned `gpt-6.1-sol` with no `supported_endpoints` even though the table
    records it as Responses-only, because both accounts had discovered
    capability. A client reading that catalogue has no way to pick the right
    entry point and gets the 400 this field exists to prevent.

    The two sources answer different questions and both are kept: discovery says
    what is reachable, the verified table says where. A discovered id the table
    does not cover still carries no claim — that part was always right.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        await self.store.set_account_capability(
            'gh-biz', ('gpt-6.1-terra', 'gpt-6.1-sol', 'gemini-4-argon', 'kimi-k3'))
        patcher = patch.object(main, 'key_policy_store', self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.copilot = _FakeProxy([main.CopilotEndpoint(
            name='gh-biz', github_token='token', token_source={'type': 'literal'}, models=[])])

    def _catalog(self):
        return main._collect_model_catalog(azure=None, copilot=self.copilot, databricks=None)

    async def test_a_discovered_model_keeps_its_verified_entry_points(self):
        catalog = self._catalog()
        self.assertEqual(catalog['gpt-6.1-terra'], (RESPONSES,))
        self.assertEqual(catalog['gpt-6.1-sol'], (RESPONSES,))
        self.assertEqual(catalog['gemini-4-argon'], (CHAT,))

    async def test_a_discovered_model_outside_the_verified_table_claims_nothing(self):
        self.assertEqual(self._catalog()['kimi-k3'], ())

    async def test_the_payload_omits_the_field_only_for_unverified_entries(self):
        catalog = self._catalog()
        verified = main._openai_model_entry('gpt-6.1-terra', catalog['gpt-6.1-terra'])
        discovered = main._openai_model_entry('kimi-k3', catalog['kimi-k3'])
        self.assertEqual(verified['supported_endpoints'], [RESPONSES])
        self.assertNotIn('supported_endpoints', discovered)


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

    def test_the_offline_fallback_table_prices_the_newest_model(self):
        '''The static table must price gpt-6.1-terra, not only the weekly refresh.

        `get_pricing` prefers the refreshed catalogue and falls back to this
        module's constants, so a model present only in the refresh is priced
        until the first pod that cannot reach the pricing page — and then the
        cost silently becomes unknown. Rates taken from the 2026-10-26 refresh
        of GitHub's own reference, which is the same source the table documents.
        '''
        self.assertEqual(copilot_pricing.MODELS['gpt-6.1-terra'],
                         {'input': 1.0, 'output': 5.0, 'cache_read': 0.05,
                          'cache_write': 1.25})
        threshold, long_rates = copilot_pricing.LONG_CONTEXT['gpt-6.1-terra']
        self.assertEqual(threshold, 272000)
        self.assertEqual(long_rates, {'input': 2.0, 'output': 7.5,
                                      'cache_read': 0.1, 'cache_write': 2.5})
        priced = copilot_pricing.get_pricing('gpt-6.1-terra', threshold + 1)
        self.assertEqual(priced['tier'], 'long_context')


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
    """Minimal stand-in for a configured endpoint.

    `supports()` is part of the contract the catalogue relies on, not an extra:
    the catalogue routes every candidate decision through it so that account
    enablement and the per-model account selection cannot be bypassed. A stub
    without it would let the catalogue regress to asking one layer directly.
    """

    def __init__(self, models=None, deployments=None, name='stub'):
        self.models = models or []
        self.deployments = deployments or []
        self.name = name

    def supports(self, model, api_type=None):
        return not self.models or model in self.models


class _Proxy:
    def __init__(self, endpoints):
        self.load_balancer = type('LB', (), {'endpoints': endpoints})()


def _wildcard_copilot():
    return _Proxy([_Endpoint(models=[])])


def _copilot_with(models):
    return _Proxy([_Endpoint(models=models)])


if __name__ == '__main__':
    unittest.main()
