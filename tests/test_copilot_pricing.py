"""GitHub's published rates, request tiers, and unknown cost coverage."""
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main
import copilot_pricing as pricing


class CopilotPricingTests(unittest.TestCase):
    def test_all_observed_official_rows_match(self):
        snapshot=json.loads((Path(__file__).parent/'fixtures/copilot-pricing-20260921.json').read_text())
        for row in snapshot['rows']:
            name=row['Model'].replace(' (fast mode) (preview)','-fast').replace(' Flash1',' Flash')
            name=name.lower().replace(' ','-')
            threshold=row.get('Threshold (input tokens)','')
            tokens=int(threshold[2:-1])*1000+1 if row.get('Tier')=='Long context' else 0
            with self.subTest(model=name,tier=row.get('Tier')):
                result=pricing.get_pricing(name,tokens,at=date(2026,9,21))
                self.assertIsNotNone(result)
                for field,column in [('input','Input'),('output','Output'),('cache_read','Cached input'),('cache_write','Cache write')]:
                    raw=row.get(column,'Not applicable')
                    self.assertEqual(result[field],0 if raw=='Not applicable' else float(raw[1:]))

    def test_aliases_are_explicit_and_unknown_variants_do_not_fall_back(self):
        self.assertEqual(pricing.get_pricing('claude-fable-5.1'),pricing.get_pricing('CLAUDE-FABLE-5-1'))
        for name in ('gpt-5.6-future','gpt-5.6-luna-pro','claude-fable-5.2','unknown'):
            with self.subTest(name=name):self.assertIsNone(pricing.get_pricing(name))
        self.assertEqual(pricing.get_pricing('claude-fable-5.1')['cache_read'],0.25)
        self.assertEqual(pricing.get_pricing('claude-opus-4.8-fast')['input'],10)

    def test_tier_boundary_uses_total_input_of_one_request(self):
        for model,threshold in [('gpt-5.6-luna',200000),('gpt-5.6-sol',272000),('gpt-6-astra',272000),('grok-4.6',200000)]:
            with self.subTest(model=model):
                self.assertEqual(pricing.get_pricing(model,threshold)['tier'],'default')
                self.assertEqual(pricing.get_pricing(model,threshold+1)['tier'],'long_context')

    def test_cached_input_is_not_charged_twice(self):
        quote=pricing.estimate_request('gpt-5.6-luna',1000,200,100,400)
        self.assertAlmostEqual(quote['estimated_cost_usd'],(500*.2+200*1.2+100*.25+400*.02)/1000000)
        self.assertAlmostEqual(quote['estimated_ai_credits'],quote['estimated_cost_usd']*100)
        self.assertIsNone(pricing.estimate_request('gpt-5.6-luna',100,10,80,80))

    def test_expired_promotional_rates_are_unknown(self):
        self.assertIsNotNone(pricing.get_pricing('gemini-3.8-flash',at=date(2026,12,31)))
        self.assertIsNone(pricing.get_pricing('gemini-3.8-flash',at=date(2027,1,1)))

    def test_no_separate_cache_write_rate_keeps_normal_input_charge(self):
        quote=pricing.estimate_request('gpt-5.4',1000,0,100,400)
        self.assertAlmostEqual(quote['estimated_cost_usd'],(600*2.5+400*.25)/1000000)


class CopilotStatsPricingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.endpoint=main.CopilotEndpoint('fixture','synthetic')
        self.proxy=main.CopilotProxy(main.LoadBalancer([self.endpoint]),'synthetic')

    async def asyncTearDown(self):await self.proxy.close()

    def record(self,model='gpt-5.6-luna',**kwargs):
        with patch.object(main,'usage_store',None):
            self.proxy._record_usage(self.endpoint,model,150000,0,0,api_type='responses',**kwargs)

    async def test_stats_sum_per_request_quotes_instead_of_tiering_aggregate_input(self):
        self.record();self.record()
        base=SimpleNamespace(get_stats=lambda:{'global':{},'endpoints':[]},today_model_stats={},today_date='2026-09-21')
        with patch.object(main,'proxy',base),patch.object(main,'azure_proxy',None),patch.object(main,'copilot_proxy',self.proxy):
            result=(await main.stats())['github_copilot']
        model=result['endpoints'][0]['model_stats']['gpt-5.6-luna']
        self.assertAlmostEqual(model['estimated_cost_usd'],.06)
        self.assertEqual(model['pricing_tiers'],{'default':2})
        self.assertEqual(model['pricing_status'],'complete')
        self.assertAlmostEqual(result['global']['estimated_ai_credits'],6)
        self.assertEqual(result['pricing']['source_url'],pricing.SOURCE_URL)

    async def test_unknown_model_makes_total_partial_not_zero(self):
        self.record();self.record('gpt-5.6-future')
        stats=self.proxy.get_stats()
        unknown=stats['endpoints'][0]['model_stats']['gpt-5.6-future']
        self.assertIsNone(unknown['estimated_cost_usd'])
        self.assertEqual(unknown['pricing_status'],'unknown')
        self.assertIsNone(stats['global']['estimated_total_cost_usd'])
        self.assertAlmostEqual(stats['global']['known_cost_subtotal_usd'],.03)
        self.assertEqual(stats['global']['unpriced_requests'],1)

    async def test_incomplete_usage_cannot_be_a_complete_cost_estimate(self):
        self.record(generation_outcome='failed',usage_fields=['input_tokens'])
        model=self.proxy.get_stats()['endpoints'][0]['model_stats']['gpt-5.6-luna']
        self.assertIsNone(model['estimated_cost_usd'])
        self.assertEqual(model['unpriced_requests'],1)
