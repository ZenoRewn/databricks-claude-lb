"""Historical reference prices must not guess request tiers. Author: Zeno Ren."""
import copy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import copilot_pricing as pricing
import main


def usage(requests=2, inp=300000, out=200, write=0, read=0):
    return dict(requests=requests, input_tokens=inp, output_tokens=out,
                cache_creation_tokens=write, cache_read_tokens=read, errors=0)


class HistoricalReferenceTests(unittest.TestCase):
    def test_daily_input_is_not_a_single_long_request(self):
        quote = pricing.estimate_history_reference('gpt-6-astra', usage())
        self.assertEqual(quote['pricing_status'], 'complete')
        self.assertEqual(quote['estimate_kind'], 'range')
        self.assertIsNone(quote['estimated_cost_usd'])
        self.assertAlmostEqual(quote['estimated_cost_min_usd'], 3.01)
        self.assertAlmostEqual(quote['estimated_cost_max_usd'], 6.015)

    def test_single_request_and_small_daily_input_have_known_tiers(self):
        for stats in (usage(requests=1), usage(inp=200000)):
            quote = pricing.estimate_history_reference('gpt-6-astra', stats)
            expected = pricing.estimate_request('gpt-6-astra', stats['input_tokens'], stats['output_tokens'])
            self.assertEqual(quote['estimated_cost_usd'], expected['estimated_cost_usd'])
            self.assertEqual(quote['estimate_kind'], 'point')

    def test_input_cache_is_not_billed_twice(self):
        quote = pricing.estimate_history_reference('gpt-5.6-luna', usage(inp=1000, write=100, read=400))
        self.assertAlmostEqual(quote['estimated_cost_usd'], (500*.2 + 100*.25 + 400*.02 + 200*1.2)/1000000)

    def test_five_live_missing_models_all_have_reference_prices(self):
        for model in ('gpt-6-astra', 'gemini-3.8-flash', 'grok-4.6', 'gpt-6-luna', 'gpt-6-sol'):
            with self.subTest(model=model):
                q = pricing.estimate_history_reference(model, usage(), at=date(2026, 10, 1))
                self.assertEqual(q['pricing_status'], 'complete')
                self.assertGreater(q['estimated_cost_max_usd'], 0)
                self.assertEqual(q['pricing_basis'], 'current_copilot_reference')

    def test_expired_invalid_and_unknown_are_distinct(self):
        self.assertIsNone(pricing.estimate_history_reference('gpt-6-future', usage()))
        q = pricing.estimate_history_reference('gemini-3.8-flash', usage(), at=date(2027, 1, 1))
        self.assertEqual(q['pricing_status'], 'unknown')
        self.assertEqual(q['pricing_reason'], 'price_expired')
        for stats in (usage(inp=10, read=100), usage(inp=-1), usage(requests=0), usage(out=True)):
            self.assertEqual(pricing.estimate_history_reference('gpt-6-astra', stats)['pricing_status'], 'unknown')

    def test_current_refresh_is_used_by_history(self):
        html = (Path(__file__).parent/'fixtures/copilot-pricing-20261001.html').read_text()
        catalog = pricing.parse_pricing_html(html.replace('$0.125', '$0.150'), '2026-10-02')
        with patch.object(pricing, '_active', catalog):
            q = pricing.estimate_history_reference('gpt-6-luna', usage(inp=1000, write=100))
        self.assertEqual(q['pricing_checked_on'], '2026-10-02')
        self.assertAlmostEqual(q['estimated_cost_usd'], (900*.1+100*.15+200*.5)/1000000)

    def test_bounds_contain_mixed_request_tiers(self):
        requests = [(150000, 500, 1000, 20000), (300000, 100, 2000, 100000), (100, 5000, 0, 0)]
        stats = usage(len(requests), *(sum(r[i] for r in requests) for i in range(4)))
        for model in ('gpt-6-astra', 'gpt-6-luna', 'gpt-5.6-sol', 'grok-4.6'):
            actual = sum(pricing.estimate_request(model, *r)['estimated_cost_usd'] for r in requests)
            bounds = pricing.estimate_history_reference(model, stats)
            self.assertLessEqual(bounds['estimated_cost_min_usd'], actual)
            self.assertGreaterEqual(bounds['estimated_cost_max_usd'], actual)


class HistoryAPITests(unittest.IsolatedAsyncioTestCase):
    async def call(self, models):
        data = {'date': date.today().isoformat(), 'models': models, 'totals': {}}
        original = copy.deepcopy(data)
        store = SimpleNamespace(get_day_data=AsyncMock(return_value=data))
        with patch.object(main, 'usage_store', store):
            result = await main.stats_history(days=1)
        self.assertEqual(data, original, 'Pricing a view must not mutate stored/cache usage')
        return result

    async def test_history_prices_official_models_and_sums_ranges(self):
        result = await self.call({'gpt-6-astra': usage(), 'gemini-3.8-flash': usage()})
        day = result['history'][0]
        self.assertEqual(day['pricing_status'], 'complete')
        self.assertIsNone(day['estimated_total_cost_usd'])
        self.assertGreater(day['estimated_cost_max_usd'], day['estimated_cost_min_usd'])
        self.assertEqual(result['pricing']['basis'], 'current_reference_rates_not_historical_invoice')
        self.assertIn('copilot', result['pricing'])

    async def test_unknown_model_does_not_turn_missing_cost_into_zero(self):
        result = await self.call({'future-model': usage(), 'gpt-6-astra': usage()})
        day = result['history'][0]
        self.assertEqual(day['pricing_status'], 'partial')
        self.assertIsNone(day['estimated_total_cost_usd'])
        self.assertGreater(day['known_cost_subtotal_usd'], 0)
        self.assertEqual(day['models']['future-model']['pricing_status'], 'unknown')

    async def test_legacy_reference_and_databricks_counting_remain_separate(self):
        row = usage(inp=1000, read=400, write=100)
        result = await self.call({'databricks-claude-opus-5': row, 'gpt-4.1': row})
        day = result['history'][0]
        for model, stats in day['models'].items():
            self.assertEqual(stats['pricing_basis'], 'legacy_model_reference')
        self.assertEqual(day['models']['databricks-claude-opus-5']['estimated_cost_usd'],
                         main.calculate_cost('databricks-claude-opus-5', 1000, 200, 100, 400))
        self.assertEqual(day['models']['gpt-4.1']['estimated_cost_usd'], (600*2 + 200*8 + 400*.5)/1000000)

    async def test_official_family_variant_does_not_fall_back_to_an_older_model(self):
        day = (await self.call({'gpt-5.6-future': usage()}))['history'][0]
        self.assertEqual(day['pricing_status'], 'unknown')
        self.assertIsNone(day['models']['gpt-5.6-future']['estimated_cost_usd'])
