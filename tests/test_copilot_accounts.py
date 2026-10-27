"""Runtime enable/disable and per-account model allow-lists for Copilot. Author: Zeno Ren.

Measured in-cluster on 2026-10-12: the four business-seat accounts expose an
identical 38-model set, while the enterprise account (ZenoRewn) exposes 36 with
a different composition. Two differences drive this feature:

  gpt-5.4 / gpt-5.5 / gpt-5.6-luna  →  enterprise only
  gpt-6-astra / kimi-k3             →  business only

gpt-5.4 and gpt-5.6-luna are the release business probes, so adding the business
accounts as wildcards would route them to an account that cannot serve them.
Explicit per-account model limits are a precondition, not an optimisation.

Two layers per account: `capability` is discovered from upstream and read-only,
`allowed_models` is the operator's subset of it. A write that names a model
outside capability is refused, so an account can never be told to serve
something it does not have.

The policy only ever removes candidates. It does not reorder Copilot-first →
Azure-fallback and cannot make an account serve a model outside its capability.

A third layer sits under both: `enabled: false` in config. An account's token
lives in the mounted Secret and endpoints are built at startup only, so naming
an account in config is the only way to get an endpoint object the Dashboard can
act on at all — which also means naming it is otherwise identical to putting it
in service. The config flag is the initial state, and an explicit write through
the API is what lifts it; thereafter the stored record decides, in both
directions.
"""
import asyncio
import unittest
from unittest.mock import patch

import key_policy
import main

# Trimmed to the models that actually distinguish the two SKUs.
ENTERPRISE = ('gpt-5.4', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-4o')
BUSINESS = ('gpt-6-astra', 'kimi-k3', 'gpt-4o')


async def _make_store(**accounts):
    store = key_policy.KeyPolicyStore(None)
    await store.start()
    for name, capability in accounts.items():
        await store.set_account_capability(name, capability)
    return store


def _store(**accounts):
    """Sync wrapper; async test cases must await _make_store instead."""
    return asyncio.run(_make_store(**accounts))


def _endpoint(name, models=(), enabled=True):
    return main.CopilotEndpoint(name=name, github_token='synthetic', models=list(models),
                                enabled_in_config=enabled)


class AccountPolicyStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        await self.store.set_account_capability('biz', BUSINESS)

    async def test_unknown_account_is_allowed_so_existing_config_keeps_working(self):
        self.assertTrue(self.store.account_allows('never-configured', 'gpt-4o'))

    async def test_capability_is_recorded_with_a_timestamp(self):
        policy = self.store.account_policy('biz')
        self.assertEqual(set(policy['capability']), set(BUSINESS))
        self.assertIsNotNone(policy['capability_at'])

    async def test_disabling_an_account_blocks_every_model(self):
        await self.store.set_account('biz', enabled=False)
        for model in BUSINESS:
            with self.subTest(model=model):
                self.assertFalse(self.store.account_allows('biz', model))

    async def test_re_enabling_restores_the_previous_allow_list(self):
        await self.store.set_account('biz', allowed_models=['gpt-6-astra'])
        await self.store.set_account('biz', enabled=False)
        await self.store.set_account('biz', enabled=True)
        self.assertTrue(self.store.account_allows('biz', 'gpt-6-astra'))
        self.assertFalse(self.store.account_allows('biz', 'kimi-k3'))

    async def test_allow_list_restricts_to_the_chosen_subset(self):
        await self.store.set_account('biz', allowed_models=['gpt-6-astra'])
        self.assertTrue(self.store.account_allows('biz', 'gpt-6-astra'))
        self.assertFalse(self.store.account_allows('biz', 'kimi-k3'))

    async def test_a_model_outside_capability_cannot_be_allowed(self):
        with self.assertRaises(ValueError):
            await self.store.set_account('biz', allowed_models=['gpt-5.4'])
        self.assertFalse(self.store.account_allows('biz', 'gpt-5.4'))

    async def test_an_empty_allow_list_is_refused_rather_than_silently_disabling(self):
        with self.assertRaises(ValueError):
            await self.store.set_account('biz', allowed_models=[])

    async def test_clearing_the_allow_list_returns_to_unrestricted(self):
        await self.store.set_account('biz', allowed_models=['gpt-6-astra'])
        await self.store.set_account('biz', allowed_models=None)
        for model in BUSINESS:
            with self.subTest(model=model):
                self.assertTrue(self.store.account_allows('biz', model))

    async def test_never_loaded_store_applies_no_account_restriction(self):
        store = key_policy.KeyPolicyStore({'type': 'mysql'})
        with patch.object(store, '_backend_load', side_effect=OSError('db down')):
            await store.start()
        self.assertEqual(store.status()['state'], 'unavailable')
        self.assertTrue(store.account_allows('biz', 'anything'))

    async def test_write_failure_leaves_the_enforced_policy_unchanged(self):
        await self.store.set_account('biz', allowed_models=['gpt-6-astra'])
        with patch.object(self.store, '_backend_write_account', side_effect=OSError('db down')):
            with self.assertRaises(OSError):
                await self.store.set_account('biz', allowed_models=['kimi-k3'])
        self.assertTrue(self.store.account_allows('biz', 'gpt-6-astra'))
        self.assertFalse(self.store.account_allows('biz', 'kimi-k3'))

    async def test_snapshot_carries_no_credential_material(self):
        blob = repr(self.store.accounts_snapshot())
        for secret in ('ghu_', 'github_token', 'Bearer'):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, blob)


class EndpointSupportsHonoursPolicyTests(unittest.TestCase):
    """supports() is the single choke point every call site goes through."""

    def setUp(self):
        self.store = _store(ZenoRewn=ENTERPRISE, biz=BUSINESS)
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def test_config_model_limits_still_apply(self):
        ep = _endpoint('biz', models=['gpt-6-astra'])
        self.assertTrue(ep.supports('gpt-6-astra', 'responses'))
        self.assertFalse(ep.supports('kimi-k3', 'responses'))

    def test_disabled_account_supports_nothing(self):
        ep = _endpoint('biz')
        asyncio.run(self.store.set_account('biz', enabled=False))
        self.assertFalse(ep.supports('gpt-6-astra', 'responses'))
        self.assertFalse(ep.supports('gpt-4o', 'responses'))

    def test_unchecked_model_is_not_supported(self):
        ep = _endpoint('biz')
        asyncio.run(self.store.set_account('biz', allowed_models=['gpt-6-astra']))
        self.assertTrue(ep.supports('gpt-6-astra', 'responses'))
        self.assertFalse(ep.supports('kimi-k3', 'responses'))

    def test_enabling_cannot_grant_a_model_outside_capability(self):
        # The business account has no gpt-5.4 and no write can add it.
        ep = _endpoint('biz')
        self.assertFalse(ep.supports('gpt-5.4', 'responses'))

    def test_api_types_filter_is_unchanged(self):
        ep = _endpoint('biz')
        self.assertFalse(ep.supports('gpt-6-astra', 'nonexistent-api'))

    def test_policy_absent_means_unchanged_behaviour(self):
        with patch.object(main, 'key_policy_store', None):
            ep = _endpoint('anything')
            self.assertTrue(ep.supports('whatever', 'responses'))


class RoutingSubtractionOnlyTests(unittest.IsolatedAsyncioTestCase):
    """The probe models must never reach an account that cannot serve them."""

    async def asyncSetUp(self):
        self.store = await _make_store(ZenoRewn=ENTERPRISE, biz=BUSINESS)
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)
        self.enterprise = _endpoint('ZenoRewn', models=list(ENTERPRISE))
        self.business = _endpoint('biz', models=list(BUSINESS))
        self.proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        self.proxy.load_balancer = main.LoadBalancer([self.enterprise, self.business])

    async def test_probe_models_only_match_the_account_that_has_them(self):
        for model in ('gpt-5.4', 'gpt-5.6-luna'):
            with self.subTest(model=model):
                matched = [ep.name for ep in self.proxy.load_balancer.endpoints
                           if ep.supports(model, 'responses')]
                self.assertEqual(matched, ['ZenoRewn'])

    async def test_business_only_models_match_the_business_account(self):
        matched = [ep.name for ep in self.proxy.load_balancer.endpoints
                   if ep.supports('gpt-6-astra', 'responses')]
        self.assertEqual(matched, ['biz'])

    async def test_shared_model_matches_both_until_one_is_disabled(self):
        both = [ep.name for ep in self.proxy.load_balancer.endpoints
                if ep.supports('gpt-4o', 'responses')]
        self.assertEqual(sorted(both), ['ZenoRewn', 'biz'])
        await self.store.set_account('biz', enabled=False)
        left = [ep.name for ep in self.proxy.load_balancer.endpoints
                if ep.supports('gpt-4o', 'responses')]
        self.assertEqual(left, ['ZenoRewn'])

    async def test_disabling_every_account_leaves_no_candidate(self):
        await self.store.set_account('ZenoRewn', enabled=False)
        await self.store.set_account('biz', enabled=False)
        self.assertEqual([ep.name for ep in self.proxy.load_balancer.endpoints
                          if ep.supports('gpt-4o', 'responses')], [])
        self.assertFalse(self.proxy.can_handle('gpt-4o', 'responses'))


class CatalogFollowsAccountPolicyTests(unittest.TestCase):
    def setUp(self):
        self.store = _store(ZenoRewn=ENTERPRISE, biz=BUSINESS)
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def _catalog(self):
        copilot = main.CopilotProxy.__new__(main.CopilotProxy)
        copilot.load_balancer = main.LoadBalancer([
            _endpoint('ZenoRewn', models=list(ENTERPRISE)),
            _endpoint('biz', models=list(BUSINESS))])
        return main._collect_model_catalog(azure=None, copilot=copilot, databricks=None)

    def test_catalog_is_derived_from_the_configured_accounts(self):
        catalog = self._catalog()
        self.assertIn('gpt-5.4', catalog)
        self.assertIn('gpt-6-astra', catalog)

    def test_disabled_account_drops_its_exclusive_models(self):
        asyncio.run(self.store.set_account('biz', enabled=False))
        catalog = self._catalog()
        self.assertNotIn('gpt-6-astra', catalog)
        self.assertIn('gpt-5.4', catalog, 'the other account is unaffected')
        self.assertIn('gpt-4o', catalog, 'a shared model stays reachable')

    def test_unverified_entry_points_are_not_claimed(self):
        # Re-pinned on 2026-10-26: gpt-6-astra *has* been measured since
        # (200 on /v1/responses, and it is a release business probe), so it now
        # carries verified entry points. kimi-k3 is the honest stand-in — the
        # business account lists it, but the gateway cannot reach it and no
        # entry point was ever measured, so the catalogue claims nothing.
        catalog = self._catalog()
        self.assertEqual(catalog['gpt-6-astra'], ('/v1/responses',))
        self.assertEqual(catalog['kimi-k3'], ())
        self.assertNotIn('supported_endpoints',
                         main._openai_model_entry('kimi-k3', catalog['kimi-k3']))


class AdminEndpointTests(unittest.TestCase):
    KEY = 'lb-copilot-admin-key-0123'

    def setUp(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = AdminEndpointTests.KEY

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = _store(biz=BUSINESS)
        self.endpoint = _endpoint('biz', models=list(BUSINESS))
        copilot = main.CopilotProxy.__new__(main.CopilotProxy)
        copilot.load_balancer = main.LoadBalancer([self.endpoint])
        copilot.api_key = self.KEY
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

    def test_all_three_endpoints_require_a_valid_key(self):
        for method, path, body in (('get', '/admin/copilot/accounts', None),
                                   ('put', '/admin/copilot/accounts/biz', {'enabled': True}),
                                   ('post', '/admin/copilot/accounts/biz/discover', None)):
            with self.subTest(path=path):
                kwargs = {'json': body} if body is not None else {}
                r = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(r.status_code, 401)

    def test_listing_never_exposes_token_material(self):
        import json as _json
        body = _json.dumps(self.client.get('/admin/copilot/accounts', headers=self.H).json())
        for secret in ('synthetic', 'ghu_', 'Bearer'):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, body)

    def test_listing_reports_capability_and_effective_set(self):
        payload = self.client.get('/admin/copilot/accounts', headers=self.H).json()
        account = payload['accounts'][0]
        self.assertEqual(account['account'], 'biz')
        self.assertTrue(account['enabled'])
        self.assertEqual(set(account['capability']), set(BUSINESS))
        self.assertIsNone(account['allowed_models'])
        self.assertTrue(payload['counters_are_process_local'])

    def test_disabling_through_the_api_takes_effect_immediately(self):
        r = self.client.put('/admin/copilot/accounts/biz', headers=self.H, json={'enabled': False})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(self.endpoint.supports('gpt-4o', 'responses'))

    def test_allowed_models_outside_capability_is_rejected(self):
        r = self.client.put('/admin/copilot/accounts/biz', headers=self.H,
                            json={'allowed_models': ['gpt-5.4']})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'invalid_account_policy')
        self.assertTrue(self.endpoint.supports('gpt-4o', 'responses'))

    def test_empty_allowed_models_is_rejected(self):
        r = self.client.put('/admin/copilot/accounts/biz', headers=self.H,
                            json={'allowed_models': []})
        self.assertEqual(r.status_code, 400)

    def test_body_without_any_field_is_rejected(self):
        self.assertEqual(self.client.put('/admin/copilot/accounts/biz', headers=self.H,
                                         json={}).status_code, 400)

    def test_unknown_account_is_404(self):
        for method, path in (('put', '/admin/copilot/accounts/nobody'),
                             ('post', '/admin/copilot/accounts/nobody/discover')):
            with self.subTest(path=path):
                r = getattr(self.client, method)(path, headers=self.H, json={'enabled': True})
                self.assertEqual(r.status_code, 404)

    def test_write_failure_reports_503_and_leaves_policy_unchanged(self):
        with patch.object(self.store, '_backend_write_account', side_effect=OSError('db down')):
            r = self.client.put('/admin/copilot/accounts/biz', headers=self.H,
                                json={'enabled': False})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()['detail']['error']['code'], 'policy_write_failed')
        self.assertTrue(self.endpoint.supports('gpt-4o', 'responses'))

    def test_discovery_stores_the_upstream_snapshot(self):
        async def fake(endpoint):
            return ['gpt-4o', 'kimi-k3']
        with patch.object(main, '_discover_copilot_capability', fake):
            r = self.client.post('/admin/copilot/accounts/biz/discover', headers=self.H)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['discovered'], 2)
        self.assertEqual(set(self.store.account_policy('biz')['capability']), {'gpt-4o', 'kimi-k3'})
        # gpt-6-astra left the capability, so it must stop routing there.
        self.assertFalse(self.endpoint.supports('gpt-6-astra', 'responses'))

    def test_shrinking_capability_drops_stale_allow_list_entries(self):
        asyncio.run(self.store.set_account('biz', allowed_models=['gpt-6-astra', 'gpt-4o']))

        async def fake(endpoint):
            return ['gpt-4o']
        with patch.object(main, '_discover_copilot_capability', fake):
            self.client.post('/admin/copilot/accounts/biz/discover', headers=self.H)
        self.assertEqual(self.store.account_policy('biz')['allowed_models'], ('gpt-4o',))


class ConfigParsingTests(unittest.TestCase):
    """load_config is where `enabled` becomes enabled_in_config.

    Covered separately because every other test constructs the endpoint
    directly: a parsing typo would leave all of them green while the deployed
    config silently put an account into service.
    """

    def _load(self, endpoints):
        import tempfile

        import yaml
        config = {
            'databricks': {'endpoints': [{'name': 'db', 'api_base': 'https://x.invalid',
                                          'token': 'synthetic'}]},
            'auth': {'api_key': 'lb-synthetic-admin-key-1'},
            'github_copilot': {'endpoints': endpoints},
        }
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as handle:
            yaml.safe_dump(config, handle)
            path = handle.name
        self.addCleanup(lambda: __import__('os').unlink(path))
        _, _, copilot, _ = main.load_config(path)
        return copilot

    def test_enabled_false_reaches_the_endpoint(self):
        copilot = self._load([{'name': 'inert', 'github_token': 'synthetic-a', 'enabled': False},
                              {'name': 'live', 'github_token': 'synthetic-b'}])
        by_name = {ep.name: ep for ep in copilot.load_balancer.endpoints}
        self.assertFalse(by_name['inert'].enabled_in_config)
        self.assertTrue(by_name['live'].enabled_in_config, 'omitting the key must stay enabled')

    def test_a_non_boolean_enabled_fails_startup_rather_than_being_coerced(self):
        # 'false' and 0 are the shapes a hand-edited ConfigMap produces, and
        # truthiness would read 'false' as enabled.
        for bad in ('false', 0, 1, None, []):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self._load([{'name': 'x', 'github_token': 'synthetic', 'enabled': bad}])


class ConfigDeclaredInertAccountTests(unittest.TestCase):
    """`enabled: false` ships an account without putting it in service."""

    def setUp(self):
        self.store = _store(biz=BUSINESS)
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def test_an_inert_account_is_a_candidate_for_nothing(self):
        endpoint = _endpoint('biz', enabled=False)
        for model in BUSINESS:
            with self.subTest(model=model):
                self.assertFalse(endpoint.supports(model, 'responses'))

    def test_a_config_enabled_account_is_unaffected(self):
        self.assertTrue(_endpoint('biz').supports('gpt-6-astra', 'responses'))

    def test_an_explicit_enable_lifts_the_config_default(self):
        endpoint = _endpoint('biz', enabled=False)
        asyncio.run(self.store.set_account('biz', enabled=True))
        self.assertTrue(endpoint.supports('gpt-6-astra', 'responses'))

    def test_a_later_disable_takes_it_back_out_of_service(self):
        endpoint = _endpoint('biz', enabled=False)
        asyncio.run(self.store.set_account('biz', enabled=True))
        asyncio.run(self.store.set_account('biz', enabled=False))
        self.assertFalse(endpoint.supports('gpt-6-astra', 'responses'))

    def test_enabling_still_cannot_reach_outside_capability(self):
        endpoint = _endpoint('biz', enabled=False)
        asyncio.run(self.store.set_account('biz', enabled=True))
        self.assertFalse(endpoint.supports('gpt-5.4', 'responses'))

    def test_the_flag_holds_without_a_policy_store(self):
        # Otherwise `enabled: false` would be a lie precisely when no store is
        # configured: the account would serve traffic from first boot.
        with patch.object(main, 'key_policy_store', None):
            self.assertFalse(_endpoint('biz', enabled=False).supports('gpt-4o', 'responses'))

    def test_an_unavailable_store_leaves_an_inert_account_inert(self):
        # Failing open is right for a restriction layer, but this is not a
        # restriction being lost — nothing ever enabled the account.
        self.store._state = 'unavailable'
        self.assertFalse(_endpoint('biz', enabled=False).supports('gpt-4o', 'responses'))
        self.assertTrue(_endpoint('biz').supports('gpt-4o', 'responses'),
                        'a config-enabled account must still fail open')

    def test_an_inert_wildcard_account_does_not_open_the_whole_catalog(self):
        # The hazard this guards: a never-discovered wildcard endpoint makes
        # _collect_model_catalog fall back to the full static table, so an
        # account that serves nothing would advertise everything.
        copilot = main.CopilotProxy.__new__(main.CopilotProxy)
        copilot.load_balancer = main.LoadBalancer([_endpoint('fresh', enabled=False)])
        catalog = main._collect_model_catalog(azure=None, copilot=copilot, databricks=None)
        self.assertEqual(catalog, {})

    def test_an_inert_account_contributes_no_models_to_the_catalog(self):
        copilot = main.CopilotProxy.__new__(main.CopilotProxy)
        copilot.load_balancer = main.LoadBalancer([
            _endpoint('ZenoRewn', models=list(ENTERPRISE)),
            _endpoint('biz', models=list(BUSINESS), enabled=False)])
        catalog = main._collect_model_catalog(azure=None, copilot=copilot, databricks=None)
        self.assertNotIn('gpt-6-astra', catalog)
        self.assertIn('gpt-5.4', catalog, 'the enabled account is unaffected')


class InertAccountAdminTests(unittest.TestCase):
    KEY = 'lb-copilot-inert-key-0123'

    def setUp(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = InertAccountAdminTests.KEY

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = _store()
        self.endpoint = _endpoint('fresh', enabled=False)
        copilot = main.CopilotProxy.__new__(main.CopilotProxy)
        copilot.load_balancer = main.LoadBalancer([self.endpoint])
        copilot.api_key = self.KEY
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

    def _view(self):
        return self.client.get('/admin/copilot/accounts', headers=self.H).json()['accounts'][0]

    def test_the_listing_separates_the_config_default_from_the_effective_state(self):
        view = self._view()
        self.assertFalse(view['enabled'], 'effective state must not claim to be live')
        self.assertFalse(view['enabled_in_config'])
        self.assertEqual(view['enablement_source'], 'config',
                         'no operator decision has been recorded yet')

    def test_enabling_before_discovery_is_refused_rather_than_going_wildcard(self):
        # An inert account carries no configured models, so enabling it without
        # a capability snapshot would make it a candidate for every model —
        # including the ones its seat does not have.
        r = self.client.put('/admin/copilot/accounts/fresh', headers=self.H, json={'enabled': True})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'capability_unknown')
        self.assertFalse(self.endpoint.supports('gpt-4o', 'responses'))

    def test_enabling_after_discovery_puts_it_in_service_within_its_capability(self):
        async def fake(endpoint):
            return list(BUSINESS)
        with patch.object(main, '_discover_copilot_capability', fake):
            self.assertEqual(
                self.client.post('/admin/copilot/accounts/fresh/discover', headers=self.H).status_code, 200)
        r = self.client.put('/admin/copilot/accounts/fresh', headers=self.H, json={'enabled': True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()['account']['enabled'])
        self.assertEqual(r.json()['account']['enablement_source'], 'operator')
        self.assertTrue(self.endpoint.supports('gpt-6-astra', 'responses'))
        self.assertFalse(self.endpoint.supports('gpt-5.4', 'responses'),
                         'capability is still the outer bound')

    def test_discovery_alone_does_not_put_an_inert_account_in_service(self):
        async def fake(endpoint):
            return list(BUSINESS)
        with patch.object(main, '_discover_copilot_capability', fake):
            self.client.post('/admin/copilot/accounts/fresh/discover', headers=self.H)
        self.assertFalse(self.endpoint.supports('gpt-6-astra', 'responses'))
        self.assertFalse(self._view()['enabled'])


class EnabledButIdleIsItsOwnStateTests(unittest.TestCase):
    """«Enabled» and «actually receiving traffic» are different questions.

    An account can be switched on and still serve nothing, because a *different*
    layer — a per-model account selection written on the routing page — excluded
    it from every model. A panel that shows only the switch invites the reading
    "it is on, so it is working".

    AWS ALB has the precedent worth copying: a registered target that no
    listener rule routes to is reported as `unused`, a first-class status
    distinct from `unhealthy`, precisely because the cause lives in another
    layer. LiteLLM's console is the counter-example — it lists models the router
    has already dropped, and one such case went unnoticed for about 17 days.
    """

    KEY = 'lb-copilot-idle-key-0123'

    def setUp(self):
        from starlette.testclient import TestClient

        class _Stub:
            api_key = EnabledButIdleIsItsOwnStateTests.KEY

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = _store(biz=BUSINESS, other=BUSINESS)
        self.endpoints = [_endpoint('biz'), _endpoint('other')]
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

    def _views(self):
        payload = self.client.get('/admin/copilot/accounts', headers=self.H).json()
        return {a['account']: a for a in payload['accounts']}

    def test_an_enabled_account_with_candidates_is_reported_as_serving(self):
        views = self._views()
        self.assertTrue(views['biz']['enabled'])
        self.assertTrue(views['biz']['receiving_traffic'])
        self.assertEqual(views['biz']['serves_models'], len(BUSINESS))

    def test_an_account_excluded_from_every_model_is_enabled_but_idle(self):
        # The switch says yes; the routing layer says no to each model. Both are
        # true, and only showing the first is what confuses people.
        for model in BUSINESS:
            self.client.put(f'/admin/model-routing/{model}', headers=self.H,
                            json={'copilot_accounts': ['biz']})
        views = self._views()
        self.assertTrue(views['other']['enabled'], 'nobody turned this account off')
        self.assertFalse(views['other']['receiving_traffic'])
        self.assertEqual(views['other']['serves_models'], 0)

    def test_a_disabled_account_is_not_merely_idle(self):
        # Same observable traffic, different cause and different fix, so the two
        # must stay distinguishable.
        self.client.put('/admin/copilot/accounts/other', headers=self.H, json={'enabled': False})
        view = self._views()['other']
        self.assertFalse(view['enabled'])
        self.assertFalse(view['receiving_traffic'])

    def test_the_idle_count_is_scoped_to_the_known_catalogue(self):
        # Honesty about what was counted: the catalogue is a point-in-time view,
        # so this is "models we know about", not "all models that exist".
        payload = self.client.get('/admin/copilot/accounts', headers=self.H).json()
        self.assertIn('catalogue_models', payload)
        self.assertGreater(payload['catalogue_models'], 0)


if __name__ == '__main__':
    unittest.main()
