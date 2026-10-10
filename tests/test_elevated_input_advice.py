"""A second, louder size advisory above the existing large_input hint.

2026-10-09/10 incidents involved bodies of 1.58 MB, 2.20 MB and 3.21 MB, all of
which merely reported context_advice=large_input — the same value a 256 KiB
request gets. The threshold below is a heuristic for making that difference
visible, NOT an evidence-derived limit: large requests also succeed (103 of 179
over 1 MB completed in one window), so this never rejects, trims or summarises.
Author: Zeno Ren.
"""
import importlib
import os
import unittest
from unittest.mock import patch

import model_capabilities


class ElevatedInputAdviceTests(unittest.TestCase):
    def _estimate(self, text_bytes, tokens=10):
        return {'text_bytes': text_bytes, 'estimated_input_tokens': tokens,
                'estimate_complete': True, 'estimate_confidence': 'low',
                'unknown_components': []}

    def test_below_large_is_none(self):
        self.assertEqual(model_capabilities.context_advice(self._estimate(1024)), 'none')

    def test_between_large_and_elevated_is_large_input(self):
        advice = model_capabilities.context_advice(self._estimate(model_capabilities.LARGE_INPUT_BYTES))
        self.assertEqual(advice, 'large_input')

    def test_at_elevated_threshold_is_elevated_input(self):
        advice = model_capabilities.context_advice(self._estimate(model_capabilities.ELEVATED_INPUT_BYTES))
        self.assertEqual(advice, 'elevated_input')

    def test_well_above_elevated_is_elevated_input(self):
        # The 3.21 MB body observed on 2026-10-10.
        self.assertEqual(model_capabilities.context_advice(self._estimate(3_214_691)), 'elevated_input')

    def test_verified_limit_ratios_still_take_precedence(self):
        # An actual catalogued limit is stronger information than raw size.
        est = self._estimate(3_214_691, tokens=1_000_000)
        self.assertEqual(model_capabilities.context_advice(est, {'input_tokens': 100}),
                         'estimated_over_limit')
        est2 = self._estimate(3_214_691, tokens=85)
        self.assertEqual(model_capabilities.context_advice(est2, {'input_tokens': 100}),
                         'estimated_near_limit')

    def test_advice_passes_the_safe_field_filter(self):
        import safe_diagnostics
        self.assertEqual(safe_diagnostics.safe_fields({'context_advice': 'elevated_input'}),
                         {'context_advice': 'elevated_input'})

    def test_unknown_advice_values_are_still_rejected(self):
        import safe_diagnostics
        self.assertEqual(safe_diagnostics.safe_fields({'context_advice': 'make_it_up'}),
                         {'context_advice': 'unknown'})

    def test_default_is_above_the_large_threshold(self):
        self.assertGreater(model_capabilities.ELEVATED_INPUT_BYTES,
                           model_capabilities.LARGE_INPUT_BYTES)

    def test_threshold_is_configurable(self):
        with patch.dict(os.environ, {'LB_CONTEXT_ELEVATED_INPUT_BYTES': '2097152'}):
            mod = importlib.reload(model_capabilities)
            try:
                self.assertEqual(mod.ELEVATED_INPUT_BYTES, 2097152)
                self.assertEqual(mod.context_advice(self._estimate(2097152)), 'elevated_input')
                self.assertEqual(mod.context_advice(self._estimate(2097151)), 'large_input')
            finally:
                importlib.reload(model_capabilities)

    def test_threshold_must_exceed_the_large_threshold(self):
        # A value at or below large_input would make the louder tier unreachable.
        with patch.dict(os.environ, {'LB_CONTEXT_ELEVATED_INPUT_BYTES': '1024',
                                     'LB_CONTEXT_LARGE_INPUT_BYTES': '262144'}):
            with self.assertRaises(ValueError):
                importlib.reload(model_capabilities)
        importlib.reload(model_capabilities)

    def test_threshold_is_range_validated(self):
        for bad in ('0', str(65 * 1024 * 1024)):
            with self.subTest(bad=bad), patch.dict(os.environ,
                                                   {'LB_CONTEXT_ELEVATED_INPUT_BYTES': bad}):
                with self.assertRaises(ValueError):
                    importlib.reload(model_capabilities)
            importlib.reload(model_capabilities)

    def test_incomplete_estimate_still_reports_size_tier(self):
        # Size is known even when image/opaque token counts are not; the advisory
        # is about bytes on the wire, so it must not silently degrade to none.
        est = self._estimate(3_214_691)
        est['estimate_complete'] = False
        est['unknown_components'] = ['images']
        self.assertEqual(model_capabilities.context_advice(est), 'elevated_input')


if __name__ == '__main__':
    unittest.main()
