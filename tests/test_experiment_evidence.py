"""Experiments require matched cohorts, coverage and actual improvement. Author: Zeno Ren."""
import copy
from datetime import datetime, timezone
import unittest


PLAN = {'schema_version': 1, 'timezone': 'Asia/Shanghai', 'hours': [9, 10, 11],
        'min_samples_per_arm': 100, 'min_completion_gain': .05, 'max_p95_ratio': 1.2,
        'max_disconnect_increase': .01, 'max_resource_ratio': 1.1, 'max_send_amplification_increase': .05}


def dataset(arm, completed=200, count=200):
    return {'schema_version': 1, 'reported_scope': 'synthetic', 'coverage': 'complete',
            'pending_requests': 0, 'cohort_started': count, 'diagnostic_dropped_events': 0,
            'identity': {'image_digest': 'sha256:' + ('a' if arm == 'baseline' else 'b') * 64},
            'window_start': '2026-09-30T01:00:00Z', 'window_end': '2026-09-30T02:00:00Z',
            'resource_peaks': {'rss_bytes': 128000000, 'admission_active': 8},
            'safety_events': {'replay_after_content': 0, 'duplicate_usage': 0, 'privacy_leak': 0},
            'requests': [{'lb_request_id': f'synthetic-{arm}-{i}', 'provider': 'copilot',
                          'requested_model': 'synthetic-model', 'forwarded_model': 'synthetic-model',
                          'api_type': 'responses', 'stream': True, 'body_size_bucket': 'small',
                          'source_tenant': 'synthetic', 'started_at_unix': datetime(2026, 9, 30, 1, tzinfo=timezone.utc).timestamp() + i,
                          'duration_seconds': 1 if i < completed else 180,
                          'outcome': 'completed' if i < completed else 'failed', 'upstream_sends': 1}
                         for i in range(count)]}


class ExperimentEvidenceTests(unittest.TestCase):
    def compare(self, baseline=None, candidate=None):
        from operations.experiment import compare
        return compare(baseline or dataset('baseline', 100), candidate or dataset('candidate'), PLAN)

    def test_good_synthetic_evidence_is_only_eligible_for_review_never_deployment(self):
        result = self.compare()
        self.assertEqual(result['status'], 'eligible_for_review')
        self.assertFalse(result['production_acceptance'])
        self.assertEqual(result['reported_scopes'], ['synthetic'])
        self.assertTrue(result['cohorts'])

    def test_missing_coverage_resources_samples_and_pending_are_inconclusive(self):
        for key, value in (('coverage', 'partial'), ('resource_peaks', {}),
                           ('pending_requests', 1), ('diagnostic_dropped_events', 1)):
            with self.subTest(key=key):
                candidate = dataset('candidate'); candidate[key] = value
                if key == 'pending_requests': candidate['cohort_started'] += 1
                self.assertEqual(self.compare(candidate=candidate)['status'], 'inconclusive')
        self.assertEqual(self.compare(candidate=dataset('candidate', 2, 2))['status'], 'inconclusive')

    def test_model_mix_and_input_bucket_changes_do_not_fake_matched_improvement(self):
        candidate = dataset('candidate')
        for row in candidate['requests']: row['body_size_bucket'] = 'large'
        self.assertEqual(self.compare(candidate=candidate)['status'], 'inconclusive')

    def test_extending_failure_delay_without_more_completion_does_not_pass(self):
        candidate = dataset('candidate', 100)
        for row in candidate['requests']:
            if row['outcome'] == 'failed': row['duration_seconds'] = 360
        self.assertEqual(self.compare(candidate=candidate)['status'], 'not_supported')

    def test_rejections_and_client_disconnects_remain_in_denominator(self):
        candidate = dataset('candidate', 100)
        for row in candidate['requests']:
            if row['outcome'] == 'failed': row['outcome'] = 'rejected'
        self.assertEqual(self.compare(candidate=candidate)['status'], 'not_supported')

    def test_duplicate_requests_and_missing_started_denominator_are_invalid(self):
        for mutation in ('duplicate', 'denominator'):
            candidate = dataset('candidate')
            if mutation == 'duplicate': candidate['requests'][1]['lb_request_id'] = candidate['requests'][0]['lb_request_id']
            else: candidate['cohort_started'] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.compare(candidate=candidate)

    def test_safety_failure_blocks_even_with_perfect_completion(self):
        candidate = dataset('candidate'); candidate['safety_events']['replay_after_content'] = 1
        self.assertEqual(self.compare(candidate=candidate)['status'], 'stop')

    def test_unknown_safety_evidence_is_not_zero(self):
        candidate = dataset('candidate'); candidate['safety_events'] = {}
        self.assertEqual(self.compare(candidate=candidate)['status'], 'inconclusive')
