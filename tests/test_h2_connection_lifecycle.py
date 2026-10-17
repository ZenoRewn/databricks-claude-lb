"""One long-lived HTTP/2 connection must not outlive its usefulness. Author: Zeno Ren.

2026-10-17, production. Every Copilot request multiplexes over a single pooled
h2 connection (`httpx_pool: {total: 1}`). After that connection had carried
roughly 340 streams, GitHub stopped accepting new ones:

    stream_id=683 code=8 headers=True  frames=469 idle=120.1s
    stream_id=685 code=7 headers=False frames=0
    stream_id=687 code=8 headers=True  frames=0   idle=120.0s
    stream_id=689 code=7 headers=False frames=0
    ... 691, 695, 697, 699, 701, 703

Odd ids, strictly +2, one connection. Two interleaved modes: REFUSED_STREAM
(code 7) for every *new* stream, and CANCEL (code 8) after an exact 120.0s idle
for streams already accepted. **GOAWAY appeared zero times**, so httpcore saw no
reason to retire the connection — `AsyncHTTP2Connection.is_available()` consults
`_used_all_stream_ids` and the h2 state machine, neither of which a silent
REFUSED_STREAM disturbs. The pool therefore handed the same dead connection to
every subsequent request, and every retry too:

    req 1605f0cc  attempt0 -> 685 REFUSED   attempt1 -> 689 REFUSED
    req b409791b  attempt0 -> 695 REFUSED   attempt1 -> 699 REFUSED
    req 6848db2f  attempt0 -> 697 REFUSED   attempt1 -> 701 REFUSED

REFUSED_STREAM is replay-safe per RFC 7540 §8.1.4, so retrying is correct — but
retrying onto the connection that just refused is structurally futile, and it is
what doubled the error count and filled the circuit threshold (5 errors from 3
business requests).

What this is NOT: a request-size problem. The same 0.9-3.0 MB bodies completed
112 times in the preceding thirteen minutes, `send_to_headers` p50 5.9s / max
17.4s, zero transport failures. Size correlates only because a large reasoning
request holds its stream open for minutes, so it is the one still alive when the
connection dies. Compressing the body would have changed nothing.

The fix is connection lifecycle, and it must be *graceful*: mark the connection
unavailable for NEW streams and let in-flight streams drain, which is exactly
what a GOAWAY would have done for us. Closing sockets outright would abort
other users' live responses.
"""
import types
import unittest
from unittest.mock import patch

import httpx

import main
import safe_diagnostics


def referenced_names(function):
    """Every name a function references, including inside comprehensions.

    Python 3.11 compiles a comprehension into its own code object while 3.12+
    inlines it (PEP 709), so a flat `co_names` check silently means different
    things per interpreter — CI caught exactly that on 3.11 while 3.12 and 3.13
    passed. Walking nested code objects makes the assertion a statement about
    the library instead of about the Python version.
    """
    names, stack = set(), [function.__code__]
    while stack:
        code = stack.pop()
        names.update(code.co_names)
        stack.extend(const for const in code.co_consts if isinstance(const, types.CodeType))
    return names


# --- stand-ins shaped like httpcore's pool -----------------------------------
# The real types need a socket to construct. These mirror only the attributes
# the retirement logic touches; PoolContractTests below pin that the real
# library still has that shape, so these cannot quietly drift into fiction.

class FakeH2State:
    def __init__(self, watermark):
        self.highest_outbound_stream_id = watermark


class FakeH2Connection:
    """Mirrors httpcore.AsyncHTTP2Connection's reuse-gating attributes."""

    def __init__(self, watermark=100, idle=False):
        self._used_all_stream_ids = False
        self._h2_state = FakeH2State(watermark)
        self._expire_at = None
        self._idle = idle
        self.closed = False

    def is_idle(self):
        return self._idle

    def is_closed(self):
        return self.closed

    async def aclose(self):
        self.closed = True


class FakeH11Connection:
    """No stream multiplexing, so nothing to retire and nothing to crash on."""

    def __init__(self):
        self.closed = False

    def is_idle(self):
        return True

    def is_closed(self):
        return False


class FakeWrapper:
    """httpcore.AsyncHTTPConnection: lazy wrapper around the real connection."""

    def __init__(self, inner):
        self._connection = inner


class FakePool:
    def __init__(self, connections):
        self._connections = list(connections)

    @property
    def connections(self):
        return list(self._connections)


class FakeTransport:
    def __init__(self, pool):
        self._pool = pool


class FakeClient:
    def __init__(self, *connections):
        self._transport = FakeTransport(FakePool(connections))


def client_with(*connections):
    return FakeClient(*[FakeWrapper(c) for c in connections])


class NameCollectionTests(unittest.TestCase):
    """The contract tests below are only as good as this helper."""

    def test_names_inside_a_nested_code_object_are_found(self):
        def probe(items):
            # A generator expression keeps its own code object even on 3.12+,
            # which is the same layout 3.11 gives a list comprehension — the
            # exact shape that made a flat `co_names` check pass on 3.12/3.13
            # and fail on 3.11.
            return any(item.is_available() for item in items)

        self.assertNotIn('is_available', probe.__code__.co_names,
                         'if this ever becomes flat, the helper is still correct but this test is moot')
        self.assertIn('is_available', referenced_names(probe))


class PoolContractTests(unittest.TestCase):
    """Pin the httpcore/h2 internals the retirement depends on.

    These are private attributes. That is a deliberate, narrow bet: there is no
    public API for "stop using this connection for new streams", and the
    alternative — rebuilding the whole client — aborts unrelated live streams.
    The bet is only acceptable if a library upgrade that invalidates it fails
    here, loudly, instead of silently turning retirement into a no-op in
    production.
    """

    def test_h2_reuse_is_gated_on_used_all_stream_ids(self):
        from httpcore._async.http2 import AsyncHTTP2Connection
        names = referenced_names(AsyncHTTP2Connection.is_available)
        self.assertIn('_used_all_stream_ids', names,
                      'httpcore no longer gates reuse on this flag; retirement needs a new lever')

    def test_h2_connection_initialises_the_attributes_retirement_writes(self):
        from httpcore._async.http2 import AsyncHTTP2Connection
        initialised = referenced_names(AsyncHTTP2Connection.__init__)
        for attribute in ('_used_all_stream_ids', '_h2_state', '_expire_at'):
            with self.subTest(attribute=attribute):
                self.assertIn(attribute, initialised)

    def test_h2_exposes_the_outbound_stream_watermark(self):
        import h2.connection
        self.assertEqual(h2.connection.H2Connection().highest_outbound_stream_id, 0)

    def test_the_lazy_wrapper_still_holds_the_real_connection(self):
        from httpcore._async.connection import AsyncHTTPConnection
        self.assertIn('_connection', referenced_names(AsyncHTTPConnection.__init__))

    def test_an_unavailable_connection_is_not_closed_by_the_pool(self):
        """The whole graceful-drain design rests on this.

        httpcore closes a pooled connection when it is closed, expired, or a
        surplus idle one — `is_available() == False` alone only stops NEW
        requests being assigned. If that ever changes, retiring an active
        connection would start aborting live streams.
        """
        from httpcore._async.connection_pool import AsyncConnectionPool
        names = referenced_names(AsyncConnectionPool._assign_requests_to_connections)
        self.assertIn('is_closed', names)
        self.assertIn('has_expired', names)
        self.assertIn('is_available', names)


class RetirementTests(unittest.TestCase):
    def test_retiring_stops_new_streams_on_an_h2_connection(self):
        conn = FakeH2Connection(watermark=683)
        result = main._retire_h2_connections(client_with(conn), reason='refused_stream')
        self.assertTrue(conn._used_all_stream_ids)
        self.assertEqual(result['retired'], 1)
        self.assertEqual(result['stream_id_watermark'], 683)
        self.assertIsNone(result['error'])

    def test_an_active_connection_keeps_its_in_flight_streams(self):
        conn = FakeH2Connection(idle=False)
        main._retire_h2_connections(client_with(conn), reason='refused_stream')
        self.assertFalse(conn.closed, 'retiring must never abort a live response')
        self.assertIsNone(conn._expire_at, 'an active connection must not be expired out from under its streams')

    def test_an_already_idle_connection_is_expired_immediately(self):
        # Nothing is streaming on it, so there is no reason to let it linger
        # until keepalive expiry just to be refused again.
        conn = FakeH2Connection(idle=True)
        with patch.object(main.time, 'monotonic', return_value=5000.0):
            main._retire_h2_connections(client_with(conn), reason='refused_stream')
        self.assertTrue(conn._used_all_stream_ids)
        self.assertIsNotNone(conn._expire_at)
        self.assertLessEqual(conn._expire_at, 5000.0)

    def test_http11_and_unconnected_wrappers_are_left_alone(self):
        h11, lazy = FakeH11Connection(), None
        result = main._retire_h2_connections(
            FakeClient(FakeWrapper(h11), FakeWrapper(lazy)), reason='refused_stream')
        self.assertEqual(result['retired'], 0)
        self.assertIsNone(result['error'])
        self.assertFalse(h11.closed)

    def test_retiring_is_idempotent(self):
        conn = FakeH2Connection()
        first = main._retire_h2_connections(client_with(conn), reason='refused_stream')
        second = main._retire_h2_connections(client_with(conn), reason='refused_stream')
        self.assertEqual(first['retired'], 1)
        self.assertEqual(second['retired'], 0, 'an already retired connection is not retired twice')

    def test_only_connections_past_the_watermark_retire_when_a_floor_is_given(self):
        young, old = FakeH2Connection(watermark=120), FakeH2Connection(watermark=640)
        result = main._retire_h2_connections(client_with(young, old),
                                             reason='stream_id_cap', min_stream_id=400)
        self.assertFalse(young._used_all_stream_ids)
        self.assertTrue(old._used_all_stream_ids)
        self.assertEqual(result['retired'], 1)

    def test_broken_internals_degrade_to_a_no_op_rather_than_raising(self):
        class Hostile:
            @property
            def _transport(self):
                raise RuntimeError('httpx internals moved')
        result = main._retire_h2_connections(Hostile(), reason='refused_stream')
        self.assertEqual(result['retired'], 0)
        self.assertIsNotNone(result['error'], 'the failure must be reported, not swallowed silently')

    def test_a_real_client_is_walked_without_error(self):
        # No connections yet, but this proves the attribute path matches the
        # library rather than only matching the fakes above.
        result = main._retire_h2_connections(httpx.AsyncClient(http2=False), reason='refused_stream')
        self.assertEqual(result['retired'], 0)
        self.assertIsNone(result['error'])

    def test_the_watermark_is_readable_without_retiring(self):
        self.assertEqual(main._h2_stream_id_watermark(client_with(FakeH2Connection(watermark=512))), 512)
        self.assertIsNone(main._h2_stream_id_watermark(client_with(FakeH11Connection())))


class TriggerTests(unittest.TestCase):
    """Which protocol evidence justifies retiring a shared connection."""

    @staticmethod
    def reset(code, *, remote=True):
        from h2.events import StreamReset
        exc = httpx.RemoteProtocolError('upstream reset')
        exc.__cause__ = Exception(StreamReset(stream_id=685, error_code=code, remote_reset=remote))
        return exc

    def test_refused_stream_retires_the_connection(self):
        # The server has said it will not accept new streams here. Retrying on
        # the same connection is what produced 685->689, 695->699, 697->701.
        self.assertTrue(main._protocol_evidence_retires_connection(self.reset(7)))

    def test_a_locally_initiated_refusal_proves_nothing_about_the_connection(self):
        self.assertFalse(main._protocol_evidence_retires_connection(self.reset(7, remote=False)))

    def test_other_reset_codes_do_not_retire(self):
        # CANCEL/INTERNAL_ERROR/PROTOCOL_ERROR are per-stream outcomes. The
        # 120s-idle CANCEL case is handled by the stall path, which has the
        # extra evidence that upstream accepted the stream and then went quiet.
        for code in (0, 1, 2, 3, 5, 8, 11):
            with self.subTest(code=code):
                self.assertFalse(main._protocol_evidence_retires_connection(self.reset(code)))

    def test_transport_failures_without_h2_evidence_do_not_retire(self):
        for exc in (httpx.ConnectError('x'), httpx.PoolTimeout('x'),
                    httpx.ReadTimeout('x'), httpx.RemoteProtocolError('bare')):
            with self.subTest(exc=type(exc).__name__):
                self.assertFalse(main._protocol_evidence_retires_connection(exc))


class EvidenceTests(unittest.TestCase):
    """The next occurrence has to be identifiable from the log alone."""

    def test_retirement_emits_its_reason_and_the_watermark(self):
        proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        proxy.client = client_with(FakeH2Connection(watermark=685))
        proxy.h2_connections_retired_total = {}
        with self.assertLogs('main', level='WARNING') as logs:
            proxy._retire_h2_connection('refused_stream')
        record = next(r for r in logs.records
                      if getattr(r, 'kind', '') == 'copilot_h2_connection_retired')
        self.assertEqual(record.retirement_reason, 'refused_stream')
        self.assertEqual(record.h2_stream_id_watermark, 685)
        self.assertEqual(record.retired_connections, 1)
        self.assertEqual(proxy.h2_connections_retired_total['refused_stream'], 1)

    def test_a_retirement_that_changed_nothing_is_not_reported_as_one(self):
        proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        proxy.client = client_with(FakeH11Connection())
        proxy.h2_connections_retired_total = {}
        with patch.object(main.logger, 'warning') as warn:
            self.assertEqual(proxy._retire_h2_connection('refused_stream'), 0)
        warn.assert_not_called()
        self.assertEqual(proxy.h2_connections_retired_total, {})

    def test_the_new_fields_survive_the_diagnostic_allowlist(self):
        # A field missing from the allowlist is dropped silently, which is how
        # an evidence field becomes useless without anyone noticing.
        payload = {'kind': 'copilot_h2_connection_retired',
                   'retirement_reason': 'refused_stream',
                   'h2_stream_id_watermark': 685,
                   'retired_connections': 1}
        safe = safe_diagnostics.safe_fields(payload)
        for key in payload:
            with self.subTest(key=key):
                self.assertIn(key, safe)


class RotationPolicyTests(unittest.TestCase):
    """What the periodic monitor retires, and what it leaves alone."""

    def proxy(self, *connections):
        proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        proxy.client = client_with(*connections)
        proxy.h2_connections_retired_total = {}
        return proxy

    def test_a_stalled_stream_retires_the_connection(self):
        # 200 headers then 90s of silence. The stream itself is left running —
        # the server will cancel it at 120s anyway — but no *new* request should
        # be put on a connection behaving like this.
        conn = FakeH2Connection(watermark=683)
        proxy = self.proxy(conn)
        self.assertEqual(proxy._rotate_h2_connections(stalled=1), 1)
        self.assertTrue(conn._used_all_stream_ids)
        self.assertEqual(proxy.h2_connections_retired_total, {'prefirst_content_stall': 1})

    def test_a_quiet_tick_retires_nothing(self):
        conn = FakeH2Connection(watermark=10)
        proxy = self.proxy(conn)
        self.assertEqual(proxy._rotate_h2_connections(stalled=0), 0)
        self.assertFalse(conn._used_all_stream_ids)

    def test_the_cap_rotates_before_the_server_starts_refusing(self):
        # Production refused new streams from 683 onward. The cap exists so we
        # rotate on our own schedule rather than discovering the ceiling.
        young, old = FakeH2Connection(watermark=120), FakeH2Connection(watermark=460)
        proxy = self.proxy(young, old)
        with patch.object(main.CopilotProxy, 'H2_RETIRE_STREAM_ID', 400):
            self.assertEqual(proxy._rotate_h2_connections(stalled=0), 1)
        self.assertFalse(young._used_all_stream_ids)
        self.assertTrue(old._used_all_stream_ids)
        self.assertEqual(proxy.h2_connections_retired_total, {'stream_id_cap': 1})

    def test_the_cap_can_be_switched_off(self):
        conn = FakeH2Connection(watermark=10_000)
        proxy = self.proxy(conn)
        with patch.object(main.CopilotProxy, 'H2_RETIRE_STREAM_ID', 0):
            self.assertEqual(proxy._rotate_h2_connections(stalled=0), 0)
        self.assertFalse(conn._used_all_stream_ids)

    def test_a_hand_built_proxy_without_the_counter_still_works(self):
        # Several test harnesses build CopilotProxy via object.__new__ and set
        # attributes by hand, so a new counter must not become a crash there.
        proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        proxy.client = client_with(FakeH2Connection())
        self.assertEqual(proxy._rotate_h2_connections(stalled=1), 1)
        self.assertEqual(proxy.h2_connections_retired_total, {'prefirst_content_stall': 1})


class StreamPathWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refused_stream_retires_the_connection(self):
        """The whole point: the retry must not land on the refusing connection.

        Production showed attempt 0 and attempt 1 of the same request taking
        consecutive stream ids on one connection and both being refused.
        """
        from tests.test_copilot_request_lifecycle import CopilotRequestLifecycleTests, _StreamClient

        class PooledStreamClient(_StreamClient):
            def __init__(self, outcomes, *connections):
                super().__init__(outcomes)
                self._transport = FakeTransport(FakePool([FakeWrapper(c) for c in connections]))

        conn = FakeH2Connection(watermark=685)
        refusals = [TriggerTests.reset(7) for _ in range(4)]
        proxy, load_balancer, endpoint = CopilotRequestLifecycleTests._make_proxy(
            self, PooledStreamClient(refusals, conn))
        proxy.h2_connections_retired_total = {}
        await load_balancer.on_request_start(endpoint)
        with self.assertLogs('main', level='WARNING') as logs:
            response = await proxy._stream_response(
                endpoint, "https://example.test/responses",
                {"model": "gpt-test", "input": "hello"}, {},
                "gpt-test", "responses", main.time.time(),
            )
            b"".join([chunk async for chunk in response.body_iterator])
        self.assertTrue(conn._used_all_stream_ids,
                        'a refusing connection must not be handed the retry')
        self.assertEqual(proxy.h2_connections_retired_total.get('refused_stream'), 1)
        retired = [r for r in logs.records
                   if getattr(r, 'kind', '') == 'copilot_h2_connection_retired']
        self.assertEqual(len(retired), 1, 'retirement is reported once, not once per attempt')
        self.assertEqual(retired[0].h2_stream_id_watermark, 685)


if __name__ == '__main__':
    unittest.main()
