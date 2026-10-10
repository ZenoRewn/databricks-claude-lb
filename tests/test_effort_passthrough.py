"""Which Databricks models keep output_config.effort. Author: Zeno Ren.

The allow-list was a single exact string, so Opus 5.5 lost the field: a live
request returned `x-lb-dropped-parameters: output_config.effort` while Opus 5
returned `x-claude-effort-forwarded: low`.

Membership must stay exact rather than substring, because
"databricks-claude-opus-5-5" contains "databricks-claude-opus-5" and an
unverified qualifier like "-fast" must not inherit support.
"""
import unittest

import effort_compat


def _body(model, output_config):
    return {'model': model, 'messages': [], 'output_config': output_config}


SUPPORTED = ('databricks-claude-opus-5', 'databricks-claude-opus-5-5')
UNSUPPORTED = ('databricks-claude-opus-5-1', 'databricks-claude-opus-4-7',
               'databricks-claude-sonnet-5-5', 'databricks-claude-haiku-5-5',
               'databricks-claude-opus-5-5-fast', 'databricks-claude-fable-5-1')


class PreserveNativeEffortTests(unittest.TestCase):
    def test_supported_models_keep_effort(self):
        for model in SUPPORTED:
            with self.subTest(model=model):
                body = _body(model, {'effort': 'low'})
                effort_compat.preserve_native_effort(body)
                self.assertEqual(body.get('output_config'), {'effort': 'low'})

    def test_unsupported_models_drop_effort(self):
        for model in UNSUPPORTED:
            with self.subTest(model=model):
                body = _body(model, {'effort': 'low'})
                effort_compat.preserve_native_effort(body)
                self.assertNotIn('output_config', body)

    def test_match_is_case_insensitive(self):
        body = _body('Databricks-Claude-Opus-5-5', {'effort': 'high'})
        effort_compat.preserve_native_effort(body)
        self.assertEqual(body.get('output_config'), {'effort': 'high'})

    def test_only_effort_is_carried_over(self):
        # format/schema support is not claimed for any model.
        body = _body('databricks-claude-opus-5-5', {'effort': 'max', 'format': 'json'})
        effort_compat.preserve_native_effort(body)
        self.assertEqual(body.get('output_config'), {'effort': 'max'})

    def test_invalid_effort_values_are_not_defaulted(self):
        body = _body('databricks-claude-opus-5-5', {'effort': 'turbo'})
        effort_compat.preserve_native_effort(body)
        self.assertEqual(body.get('output_config'), {'effort': 'turbo'})


class ParameterDropReportTests(unittest.TestCase):
    def test_supported_models_do_not_report_effort_as_dropped(self):
        for model in SUPPORTED:
            with self.subTest(model=model):
                drops = effort_compat.databricks_parameter_drops(_body(model, {'effort': 'low'}))
                self.assertNotIn('output_config.effort', drops)

    def test_unsupported_models_report_effort_as_dropped(self):
        for model in UNSUPPORTED:
            with self.subTest(model=model):
                drops = effort_compat.databricks_parameter_drops(_body(model, {'effort': 'low'}))
                self.assertIn('output_config.effort', drops)

    def test_format_remains_dropped_even_on_supported_models(self):
        drops = effort_compat.databricks_parameter_drops(
            _body('databricks-claude-opus-5-5', {'effort': 'low', 'format': 'json'}))
        self.assertIn('output_config.format', drops)
        self.assertNotIn('output_config.effort', drops)


class ResponseHeaderTests(unittest.TestCase):
    def test_header_reports_the_forwarded_effort(self):
        body = _body('databricks-claude-opus-5-5', {'effort': 'low'})
        effort_compat.preserve_native_effort(body)
        self.assertEqual(effort_compat.effort_response_headers(body),
                         {'x-claude-effort-forwarded': 'low'})

    def test_no_header_when_the_field_was_dropped(self):
        body = _body('databricks-claude-sonnet-5-5', {'effort': 'low'})
        effort_compat.preserve_native_effort(body)
        self.assertEqual(effort_compat.effort_response_headers(body), {})


if __name__ == '__main__':
    unittest.main()
