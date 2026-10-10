"""A bare response.failed must not be recorded as an unknown failure.

2026-10-10: upstream answered 200 and then sent response.failed within 2.5-5.1s
carrying neither an error object nor usage. The stream was correctly classified
as failed, but failure_reason stayed at its 'unknown' default, which made
"upstream explicitly declared failure" indistinguishable from "we have no idea".
Author: Zeno Ren.
"""
import unittest
from unittest.mock import patch

import main
import request_telemetry


class BareUpstreamFailureTests(unittest.TestCase):
    def setUp(self):
        self.record = request_telemetry.RequestRecord('responses', request_telemetry.RequestTelemetry())
        self.token = request_telemetry.CURRENT.set(self.record)
        self.events = []
        self.patcher = patch.object(request_telemetry, 'log_event', self.events.append)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        request_telemetry.CURRENT.reset(self.token)

    def _observe(self, payload):
        obs = main._SSEObservation("responses")
        obs.observe(f'data: {main.json.dumps(payload)}\n\n'.encode())
        return obs

    def test_bare_failed_records_upstream_failure(self):
        obs = self._observe({'type': 'response.failed', 'response': {}})
        self.assertEqual(obs.terminal, 'failed')
        self.assertEqual(self.record.failure_reason, 'upstream_failure')

    def test_failed_with_null_error_records_upstream_failure(self):
        obs = self._observe({'type': 'response.failed', 'response': {'error': None}})
        self.assertEqual(obs.terminal, 'failed')
        self.assertEqual(self.record.failure_reason, 'upstream_failure')

    def test_failed_with_malformed_error_records_upstream_failure(self):
        # A string where an object belongs is not classifiable detail.
        obs = self._observe({'type': 'response.failed', 'response': {'error': 'boom'}})
        self.assertEqual(obs.terminal, 'failed')
        self.assertEqual(self.record.failure_reason, 'upstream_failure')

    def test_structured_error_still_wins_over_the_fallback(self):
        obs = self._observe({'type': 'response.failed',
                             'response': {'error': {'code': 'context_length_exceeded'}}})
        self.assertEqual(obs.terminal, 'failed')
        self.assertEqual(self.record.failure_reason, 'context_window_exceeded')

    def test_unrecognised_code_still_beats_unknown(self):
        # Pre-existing limitation, documented rather than changed here: the stream
        # path passes only {'error': ...} to failure_reason, so response.status is
        # not available and status-based classification (429/401/5xx) does not
        # apply to streamed failed events. An unrecognised code must therefore
        # still land on upstream_failure rather than falling back to unknown.
        self._observe({'type': 'response.failed',
                       'response': {'error': {'code': 'rate_limit_exceeded'}, 'status': 429}})
        self.assertEqual(self.record.failure_reason, 'upstream_failure')

    def test_incomplete_is_not_relabelled_as_a_failure(self):
        # response.incomplete is not necessarily a failure (max_output_tokens is
        # a normal truncation), so it must keep its existing semantics.
        obs = self._observe({'type': 'response.incomplete',
                             'response': {'incomplete_details': {'reason': 'max_output_tokens'}}})
        self.assertEqual(obs.terminal, 'incomplete')
        self.assertNotEqual(self.record.failure_reason, 'upstream_failure')
        self.assertTrue(obs.neutral)

    def test_bare_incomplete_keeps_unknown_rather_than_claiming_failure(self):
        obs = self._observe({'type': 'response.incomplete', 'response': {}})
        self.assertEqual(obs.terminal, 'incomplete')
        self.assertNotEqual(self.record.failure_reason, 'upstream_failure')

    def test_completed_records_no_failure(self):
        # Terminal validity for completed is covered elsewhere; here it only
        # matters that a completed event never takes the failure fallback.
        obs = self._observe({'type': 'response.completed',
                             'response': {'status': 'completed',
                                          'output': [{'type': 'message', 'content': [
                                              {'type': 'output_text', 'text': 'ok'}]}]}})
        self.assertNotEqual(obs.terminal, 'failed')
        self.assertNotEqual(self.record.failure_reason, 'upstream_failure')

    def test_fallback_does_not_mark_the_endpoint_neutral(self):
        # An unexplained upstream failure is not request-local evidence, so it
        # must keep counting against the shared endpoint.
        obs = self._observe({'type': 'response.failed', 'response': {}})
        self.assertFalse(obs.neutral)

    def test_reason_is_emitted_through_the_safe_filter(self):
        self._observe({'type': 'response.failed', 'response': {}})
        errors = [e for e in self.events if e.get('kind') == 'lb_upstream_error']
        self.assertTrue(errors, 'a bare failure must still emit lb_upstream_error')
        self.assertEqual(errors[-1]['reason'], 'upstream_failure')
        import safe_diagnostics
        self.assertEqual(safe_diagnostics.safe_fields({'failure_reason': 'upstream_failure'}),
                         {'failure_reason': 'upstream_failure'})


if __name__ == '__main__':
    unittest.main()
