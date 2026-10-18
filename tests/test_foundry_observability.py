"""Foundry has to be visible once it can take traffic. Author: Zeno Ren.

The 2026-10-17 release put `azure_anthropic` into the `/v1/messages` candidate
set but left it out of `/stats` and `/metrics`: the stats payload carried
`endpoints` (ADB), `azure_openai` and `github_copilot`, and the 18 exported
`circuit_open` samples covered 9 + 4 + 5 endpoints — none of them Foundry's. So
the moment ADB and GHCP were both unavailable, the provider actually serving
traffic would be the one nobody could see.

Cost is the part that must stay *unknown* rather than become a number.
`MODEL_PRICING` matches by substring, so Foundry's `claude-opus-5-5` hits the
`opus-5-5` key and would be costed at Anthropic's public per-token rate.
Foundry bills in Claude Consumption Units through the Azure Marketplace, a
different unit entirely. A confident wrong number is worse than a blank, and a
zero would roll up in the dashboard as «free».
"""
import asyncio
import unittest
from unittest.mock import patch

import main


def _endpoint(name='eastus2-foundry'):
    return main.WorkspaceEndpoint(
        name=name, api_base='https://x.services.ai.azure.com', token='SENTINEL',
        models=['claude-opus-5-5'], provider='azure_anthropic',
        auth_style='x-api-key', anthropic_version='2023-06-01')


def _foundry_proxy(endpoint=None):
    proxy = object.__new__(main.ClaudeProxy)
    proxy.provider = 'azure_anthropic'
    proxy.load_balancer = main.LoadBalancer(endpoints=[endpoint or _endpoint()],
                                            strategy='least_requests')
    proxy.global_stats = main.GlobalStats()
    proxy.today_model_stats = {}
    proxy.today_date = '2026-10-17'
    return proxy


class StatsSectionTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = _endpoint()
        self.foundry = _foundry_proxy(self.endpoint)
        base = object.__new__(main.ClaudeProxy)
        base.provider = 'databricks'
        base.load_balancer = main.LoadBalancer(endpoints=[], strategy='least_requests')
        base.global_stats = main.GlobalStats()
        base.today_model_stats = {}
        base.today_date = '2026-10-17'
        base.foundry = self.foundry
        for name, value in (('proxy', base), ('azure_proxy', None), ('copilot_proxy', None)):
            patcher = patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def stats(self):
        return asyncio.run(main.stats())

    def test_foundry_has_a_section_of_its_own(self):
        # Not merged into azure_openai: same Azure resource, different face.
        self.assertIn('azure_anthropic', self.stats())

    def test_its_endpoints_are_reported(self):
        section = self.stats()['azure_anthropic']
        self.assertEqual([ep['name'] for ep in section['endpoints']], ['eastus2-foundry'])

    def test_the_declared_deployments_are_reported(self):
        # Foundry has no Models API, so the operator-declared deployment names
        # are the only statement of what it can serve — and they are also the
        # spelling most easily mis-configured (dashes, not dots). An empty
        # column would read as «serves nothing».
        section = self.stats()['azure_anthropic']
        self.assertEqual(section['endpoints'][0]['deployments'], ['claude-opus-5-5'])

    def test_the_circuit_state_is_reported(self):
        self.endpoint.circuit_open = True
        section = self.stats()['azure_anthropic']
        self.assertTrue(section['endpoints'][0]['circuit_open'])

    def test_cost_is_declared_unknown_rather_than_omitted_silently(self):
        # A missing key reads as «forgot to implement»; this says why.
        pricing = self.stats()['azure_anthropic']['pricing']
        self.assertEqual(pricing['status'], 'unknown')
        self.assertEqual(pricing['basis'], 'ccu_not_per_token')

    def test_no_per_token_cost_is_attached_to_foundry_usage(self):
        # The substring match would otherwise price claude-opus-5-5 at
        # Anthropic's public rate. Guard the whole section, not one field.
        self.endpoint.model_stats['claude-opus-5-5'] = {
            'requests': 1, 'input_tokens': 1000, 'output_tokens': 100,
            'cache_creation_tokens': 0, 'cache_read_tokens': 0}
        section = self.stats()['azure_anthropic']
        for ep in section['endpoints']:
            for model, mstats in (ep.get('model_stats') or {}).items():
                with self.subTest(model=model):
                    self.assertNotIn('estimated_cost_usd', mstats)

    def test_foundry_usage_does_not_enter_the_anthropic_cost_total(self):
        # The KPI total is the Databricks/Anthropic口径; CCU cannot join it.
        self.assertEqual(self.stats()['global']['estimated_total_cost_usd'], 0.0)

    def test_an_unconfigured_foundry_adds_no_section(self):
        del main.proxy.foundry
        self.assertNotIn('azure_anthropic', self.stats())


class UsageAttributionTests(unittest.TestCase):
    """用量事件必须说出它真正来自哪个上游。

    `ClaudeProxy` 的记录点曾硬编码 `provider='databricks'`，而三个
    Anthropic Messages 池**都是** `ClaudeProxy`（`CopilotMessagesProxy` 直接
    继承它，没有自己的记录逻辑）。于是 Foundry 与 GHCP-messages 的每一条用量
    都被标成 Databricks —— 计价口径从源头就错了，加列也救不回来。
    """

    def setUp(self):
        self.recorded = []
        store = type('Store', (), {
            'record': lambda _s, *a, **k: self.recorded.append(k.get('provider')),
            'persistence_stats': lambda _s: {'pending_events': 0},
        })()
        patcher = patch.object(main, 'usage_store', store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _record_through(self, provider):
        proxy = object.__new__(main.ClaudeProxy)
        proxy.provider = provider
        proxy.global_stats = main.GlobalStats()
        proxy.today_model_stats = {}
        proxy.today_date = '2026-10-18'
        endpoint = _endpoint()
        proxy.load_balancer = main.LoadBalancer(endpoints=[endpoint], strategy='least_requests')
        proxy._record_usage(endpoint, 'claude-opus-5-5', 10, 5, 0, 0,
                            generation_outcome='completed')
        return self.recorded

    def test_foundry_usage_is_attributed_to_foundry(self):
        self.assertEqual(self._record_through('azure_anthropic'), ['azure_anthropic'])

    def test_copilot_messages_usage_is_attributed_to_copilot(self):
        self.assertEqual(self._record_through('copilot'), ['copilot'])

    def test_databricks_remains_the_default_for_an_unset_provider(self):
        proxy = object.__new__(main.ClaudeProxy)
        proxy.global_stats = main.GlobalStats()
        proxy.today_model_stats = {}
        proxy.today_date = '2026-10-18'
        endpoint = _endpoint()
        proxy.load_balancer = main.LoadBalancer(endpoints=[endpoint], strategy='least_requests')
        # Harnesses build proxies via object.__new__, so the attribute can be
        # missing entirely; that must not crash the usage path.
        proxy._record_usage(endpoint, 'databricks-claude-opus-5', 1, 1, 0, 0,
                            generation_outcome='completed')
        self.assertEqual(self.recorded, ['databricks'])


class MetricsExportTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = _endpoint()
        self.foundry = _foundry_proxy(self.endpoint)
        base = object.__new__(main.ClaudeProxy)
        base.provider = 'databricks'
        base.load_balancer = main.LoadBalancer(endpoints=[], strategy='least_requests')
        base.global_stats = main.GlobalStats()
        base.foundry = self.foundry
        for name, value in (('proxy', base), ('azure_proxy', None), ('copilot_proxy', None)):
            patcher = patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def text(self):
        return asyncio.run(main.metrics()).body.decode()

    def test_the_circuit_gauge_is_exported_for_foundry(self):
        self.assertIn('azure_anthropic_endpoint_circuit_open{endpoint="eastus2-foundry"}',
                      self.text())

    def test_the_gauge_reflects_an_open_circuit(self):
        self.endpoint.circuit_open = True
        self.assertIn('azure_anthropic_endpoint_circuit_open{endpoint="eastus2-foundry"} 1',
                      self.text())

    def test_request_and_error_counters_are_exported(self):
        self.endpoint.total_requests = 7
        self.endpoint.total_errors = 2
        text = self.text()
        self.assertIn('azure_anthropic_endpoint_requests_total{endpoint="eastus2-foundry"} 7', text)
        self.assertIn('azure_anthropic_endpoint_errors_total{endpoint="eastus2-foundry"} 2', text)

    def test_every_foundry_family_is_declared_before_use(self):
        # A sample without its # TYPE line breaks strict Prometheus parsers.
        text = self.text()
        declared = {line.split()[2] for line in text.splitlines() if line.startswith('# TYPE ')}
        used = {line.split('{')[0].split()[0] for line in text.splitlines()
                if line.startswith('azure_anthropic_')}
        self.assertTrue(used, 'no foundry samples exported at all')
        self.assertFalse(used - declared)

    def test_the_credential_never_reaches_the_exposition(self):
        self.assertNotIn('SENTINEL', self.text())


if __name__ == '__main__':
    unittest.main()
