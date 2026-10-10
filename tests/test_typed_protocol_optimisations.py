"""Two typed-protocol facts the code was not yet acting on. Author: Zeno Ren.

Both rest on guarantees carried by an h2 event, never on an exception string,
elapsed time, request size or prompt content.

REFUSED_STREAM (RFC 7540 §8.1.4): "the stream is being closed prior to any
processing having occurred. Any request that was sent on the reset stream can be
safely retried." Execution state is therefore *known*, not unknown, so replaying
does not violate the no-replay-on-unknown-execution invariant.

GOAWAY with NO_ERROR (RFC 7540 §6.8) is a graceful connection shutdown. The
stream is still lost and the client must still see a failure, but the endpoint
is not unhealthy, so it must not accumulate circuit-breaker errors.
"""
import unittest

import httpx

import main


def _reset(code, *, remote=True):
    """Build the real httpx/httpcore cause chain carrying an h2 StreamReset.

    h2 4.x makes stream_id keyword-only, so construct through the real signature
    rather than setting attributes on a bare instance.
    """
    from h2.events import StreamReset
    event = StreamReset(stream_id=7, error_code=code, remote_reset=remote)
    exc = httpx.RemoteProtocolError('upstream reset')
    exc.__cause__ = Exception(event)
    return exc


def _goaway(code):
    from h2.events import ConnectionTerminated
    event = ConnectionTerminated()
    event.error_code = code
    event.last_stream_id = 2147483647
    exc = httpx.RemoteProtocolError('upstream goaway')
    exc.__cause__ = Exception(event)
    return exc


class RefusedStreamReplayTests(unittest.TestCase):
    def test_refused_stream_is_replay_safe(self):
        self.assertTrue(main._replay_safe_transport_failure(_reset(7)))

    def test_connect_failures_remain_replay_safe(self):
        for exc in (httpx.ConnectError('x'), httpx.ConnectTimeout('x'), httpx.PoolTimeout('x')):
            with self.subTest(exc=type(exc).__name__):
                self.assertTrue(main._replay_safe_transport_failure(exc))

    def test_other_reset_codes_are_not_replay_safe(self):
        # CANCEL/INTERNAL_ERROR/PROTOCOL_ERROR carry no no-processing guarantee,
        # and ENHANCE_YOUR_CALM must not be hammered.
        for code in (0, 1, 2, 8, 11):
            with self.subTest(code=code):
                self.assertFalse(main._replay_safe_transport_failure(_reset(code)))

    def test_locally_initiated_reset_is_not_replay_safe(self):
        # remote_reset=False means we cancelled it; upstream may have executed.
        self.assertFalse(main._replay_safe_transport_failure(_reset(7, remote=False)))

    def test_plain_protocol_error_is_not_replay_safe(self):
        self.assertFalse(main._replay_safe_transport_failure(httpx.RemoteProtocolError('bare')))

    def test_read_and_write_failures_remain_excluded(self):
        for exc in (httpx.ReadTimeout('x'), httpx.WriteTimeout('x'),
                    httpx.ReadError('x'), httpx.WriteError('x')):
            with self.subTest(exc=type(exc).__name__):
                self.assertFalse(main._replay_safe_transport_failure(exc))

    def test_goaway_refused_code_does_not_authorise_replay(self):
        # A connection-scoped GOAWAY carrying 7 is not a per-stream guarantee.
        self.assertFalse(main._replay_safe_transport_failure(_goaway(7)))


class GracefulGoawayTests(unittest.TestCase):
    def test_no_error_goaway_is_neutral(self):
        self.assertTrue(main._graceful_shutdown_failure(_goaway(0)))

    def test_goaway_with_real_error_is_not_neutral(self):
        for code in (1, 2, 7, 11):
            with self.subTest(code=code):
                self.assertFalse(main._graceful_shutdown_failure(_goaway(code)))

    def test_stream_reset_is_not_a_graceful_shutdown(self):
        for code in (0, 7, 8):
            with self.subTest(code=code):
                self.assertFalse(main._graceful_shutdown_failure(_reset(code)))

    def test_untyped_failures_are_not_neutral(self):
        for exc in (httpx.RemoteProtocolError('bare'), httpx.ConnectError('x'),
                    httpx.ReadTimeout('x'), None):
            with self.subTest(exc=type(exc).__name__):
                self.assertFalse(main._graceful_shutdown_failure(exc))

    def test_graceful_shutdown_keeps_endpoint_scope(self):
        # Neutrality is about not accumulating errors; it must not reclassify the
        # blast radius to a single model/API route.
        self.assertEqual(main._copilot_failure_scope(exception=_goaway(0)), 'endpoint')

    def test_graceful_shutdown_is_not_replay_safe(self):
        # The stream may have been processed; only the connection closed cleanly.
        self.assertFalse(main._replay_safe_transport_failure(_goaway(0)))


class CircuitAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ep = main.CopilotEndpoint('synthetic', '', models=['a'])
        self.lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=2)

    async def test_neutral_failure_does_not_trip_the_circuit(self):
        for _ in range(5):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, is_client_error=True, lease=lease)
        self.assertFalse(self.ep.circuit_open)
        self.assertEqual(self.ep.consecutive_errors, 0)
        self.assertEqual(self.ep.neutral_requests, 5)

    async def test_real_failure_still_trips_the_circuit(self):
        for _ in range(2):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, lease=lease)
        self.assertTrue(self.ep.circuit_open)


if __name__ == '__main__':
    unittest.main()
