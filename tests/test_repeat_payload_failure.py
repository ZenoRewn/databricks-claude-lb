"""2026-10-22: one poisoned conversation, resent five times by Codex, tripped
an account-wide circuit. Identical payload failures count once per streak."""
import json
import unittest
from unittest.mock import Mock

import main
import safe_diagnostics
import test_copilot_request_lifecycle as fixtures


class RepeatPayloadFailureTests(unittest.IsolatedAsyncioTestCase):
    def make(self, threshold=3):
        return fixtures.CopilotRequestLifecycleTests()._make_proxy(Mock(), threshold)

    async def fail(self, lb, ep, payload, payload_scoped=True):
        lease = await lb.on_request_start(ep, payload=payload)
        await lb.on_request_end(ep, False, lease=lease, payload_scoped=payload_scoped)

    async def test_identical_payload_failures_do_not_trip_endpoint(self):
        _, lb, ep = self.make()
        body = {'model': 'gpt-test', 'input': 'x' * 100}
        for _ in range(5):
            await self.fail(lb, ep, dict(body))
        self.assertFalse(ep.circuit_open)
        self.assertEqual(ep.consecutive_errors, 1)
        self.assertEqual(ep.total_errors, 5, 'telemetry keeps every failure')

    async def test_distinct_payload_failures_still_trip(self):
        _, lb, ep = self.make()
        for n in range(3):
            await self.fail(lb, ep, {'model': 'gpt-test', 'input': str(n)})
        self.assertTrue(ep.circuit_open)

    async def test_payload_free_failures_keep_old_counting(self):
        _, lb, ep = self.make()
        for _ in range(3):
            await self.fail(lb, ep, None)
        self.assertTrue(ep.circuit_open)

    async def test_non_payload_failures_of_one_body_still_trip(self):
        # Auth/transport/5xx with an identical body are account or link faults.
        _, lb, ep = self.make()
        for _ in range(3):
            await self.fail(lb, ep, {'input': 'same'}, payload_scoped=False)
        self.assertTrue(ep.circuit_open)

    async def test_success_resets_seen_payloads(self):
        _, lb, ep = self.make()
        body = {'model': 'gpt-test', 'input': 'same'}
        await self.fail(lb, ep, body)
        lease = await lb.on_request_start(ep, payload={'input': 'ok'})
        await lb.on_request_end(ep, True, lease=lease)
        await self.fail(lb, ep, body)
        self.assertEqual(ep.consecutive_errors, 1)


class UpstreamErrorDetailTests(unittest.TestCase):
    def observe(self, error):
        obs = main._SSEObservation('responses')
        event = {'type': 'response.failed', 'response': {'id': 'r', 'status': 'failed', 'error': error}}
        obs.observe(b'data: ' + json.dumps(event).encode() + b'\n\n')
        return obs

    def test_failed_event_code_and_type_are_kept_as_safe_identifiers(self):
        obs = self.observe({'code': 'server_error', 'type': 'api_error', 'message': 'secret prompt text'})
        self.assertEqual(obs.upstream_error_code, 'server_error')
        self.assertEqual(obs.upstream_error_type, 'api_error')

    def test_unsafe_code_is_not_recorded_verbatim(self):
        obs = self.observe({'code': 'has spaces and prompt text'})
        self.assertEqual(obs.upstream_error_code, 'unknown')

    def test_bare_failed_has_no_code(self):
        obs = self.observe(None)
        self.assertIsNone(obs.upstream_error_code)

    def test_fields_pass_the_diagnostics_allowlist(self):
        out = safe_diagnostics.safe_fields({'upstream_error_code': 'server_error',
                                            'upstream_error_type': 'api_error',
                                            'payload_fingerprint': 'abcdef012345',
                                            'repeat_payload_failure': True})
        self.assertEqual(out['upstream_error_code'], 'server_error')
        self.assertEqual(out['payload_fingerprint'], 'abcdef012345')
        self.assertIs(out['repeat_payload_failure'], True)


if __name__ == '__main__':
    unittest.main()


class ImageTrimRejectionEvidenceTests(unittest.TestCase):
    def test_rejection_records_image_count_and_limit(self):
        import request_telemetry as rt
        record = rt.RequestRecord('responses', rt.RequestTelemetry())
        token = rt.CURRENT.set(record)
        try:
            payload = {'input': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AA'}] * 3}
            with self.assertRaises(main.HTTPException) as ctx:
                main.trim_excess_images(payload, max_count=2)
            self.assertEqual(ctx.exception.status_code, 413)
        finally:
            rt.CURRENT.reset(token)
        self.assertEqual((record.image_count, record.image_limit), (3, 2))
        out = safe_diagnostics.safe_fields({'image_count': 3, 'image_limit': 2})
        self.assertEqual(out['image_limit'], 2)
