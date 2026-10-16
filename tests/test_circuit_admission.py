import asyncio
import unittest
from unittest.mock import patch
import main
import request_timing


class CircuitAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch.object(main.time, 'monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.ep = main.CopilotEndpoint('test', 'synthetic', models=['a','b'])
        self.lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=2, circuit_breaker_timeout=30)

    async def trip(self):
        for _ in range(2):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, lease=lease)
        self.assertEqual(self.lb.circuit_state(self.ep), 'OPEN')

    async def test_concurrent_half_open_claim_is_bounded(self):
        await self.trip()
        self.now += 30
        async def claim():
            await asyncio.sleep(0)
            return await self.lb.on_request_start(self.ep)
        leases = await asyncio.gather(*(claim() for _ in range(100)), return_exceptions=True)
        admitted = [x for x in leases if isinstance(x, main.RequestAttempt)]
        self.assertEqual(len(admitted), 1)
        self.assertEqual(sum(isinstance(x, main.HTTPException) for x in leases), 99)
        self.assertEqual(self.ep.active_requests, 1)
        self.assertTrue(self.ep.circuit_open)
        self.assertEqual(self.ep.rejected_requests, 99)
        self.assertTrue(self.lb.is_available(self.ep, readiness=True))
        self.assertFalse(self.lb.is_available(self.ep))
        await self.lb.on_request_end(self.ep, True, lease=admitted[0])
        self.assertEqual(self.lb.circuit_state(self.ep), 'CLOSED')
        self.assertEqual(self.ep.total_errors, 2)
        self.assertEqual(self.ep.consecutive_errors, 0)

    async def test_late_completions_cannot_close_or_reopen_new_generation(self):
        old_ok = await self.lb.on_request_start(self.ep)
        old_bad = await self.lb.on_request_start(self.ep)
        await self.trip()
        await self.lb.on_request_end(self.ep, True, lease=old_ok)
        self.assertTrue(self.ep.circuit_open)
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, True, lease=trial)
        await self.lb.on_request_end(self.ep, False, lease=old_bad)
        self.assertFalse(self.ep.circuit_open)
        self.assertEqual(self.ep.total_errors, 3)
        self.assertEqual(self.ep.consecutive_errors, 0)
        self.assertEqual(self.ep.active_requests, 0)

    async def test_cancelled_trial_releases_once_without_health_failure(self):
        await self.trip()
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, False, lease=trial, cancelled=True)
        await self.lb.on_request_end(self.ep, False, lease=trial, cancelled=True)
        self.assertEqual(self.ep.cancelled_requests, 1)
        self.assertEqual(self.ep.total_errors, 2)
        self.assertEqual(self.ep.active_requests, 0)
        self.assertEqual(self.lb.circuit_state(self.ep), 'OPEN')
        self.assertEqual(self.lb.retry_after(), 30)
        self.now += 30
        next_trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, True, lease=next_trial)
        self.assertFalse(self.ep.circuit_open)

    async def test_trial_failure_reopens_for_fresh_cooldown(self):
        await self.trip()
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        self.now += 4
        await self.lb.on_request_end(self.ep, False, lease=trial)
        self.assertEqual(self.lb.retry_after(), 30)
        self.assertEqual(self.ep.total_errors, 3)
        self.assertFalse(self.ep.half_open_in_flight)

    async def test_neutral_result_does_not_reset_or_increment_failure_streak(self):
        lease = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, False, lease=lease)
        for kw in ({'is_client_error':True}, {'cancelled':True}):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, lease=lease, **kw)
        self.assertEqual(self.ep.total_errors, 1)
        self.assertEqual(self.ep.consecutive_errors, 1)
        self.assertEqual(self.ep.neutral_requests, 1)
        self.assertEqual(self.ep.cancelled_requests, 1)

    async def test_neutral_trial_does_not_claim_inference_recovered(self):
        await self.trip()
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, False, lease=trial, is_client_error=True)
        self.assertEqual(self.lb.circuit_state(self.ep), 'OPEN')
        self.assertEqual(self.ep.total_errors, 2)

    async def test_probe_not_abandoned_when_request_runs_longer_than_cooldown(self):
        await self.trip()
        self.now += 30
        lease = await self.lb.on_request_start(self.ep)
        self.now += 10000
        self.assertFalse(self.lb.get_available_endpoints())
        self.assertEqual(self.ep.active_requests, 1)
        await self.lb.on_request_end(self.ep, False, lease=lease, cancelled=True)
        self.assertEqual(self.ep.active_requests, 0)

    async def test_inspection_is_pure_and_does_not_advance_round_robin(self):
        await self.trip()
        self.now += 30
        proxy = object.__new__(main.CopilotProxy)
        proxy.load_balancer = self.lb
        for _ in range(10):
            self.assertTrue(proxy.can_handle('a'))
            self.assertIn(self.ep, self.lb.get_available_endpoints())
            self.assertEqual(self.lb.get_stats()['endpoints'][0]['circuit_state'], 'HALF_OPEN')
        self.assertEqual(self.ep.total_requests, 2)
        self.assertEqual(self.lb._rr_index, 0)
        self.assertFalse(self.ep.half_open_in_flight)
        self.assertTrue(self.ep.circuit_open)

    async def test_monotonic_cooldown_unaffected_by_wallclock(self):
        await self.trip()
        with patch.object(main.time, 'time', return_value=2**40):
            self.assertFalse(self.lb.is_available(self.ep))
        self.now += 29.2
        self.assertEqual(self.lb.retry_after(), 1)
        self.assertFalse(self.lb.is_available(self.ep))
        self.now += .8
        self.assertTrue(self.lb.is_available(self.ep))

    async def test_endpoint_state_is_independent(self):
        sibling = main.CopilotEndpoint('other', 'synthetic', models=['a'])
        self.lb.endpoints.append(sibling)
        await self.trip()
        self.assertEqual(self.lb.get_available_endpoints(), [sibling])
        lease = await self.lb.on_request_start(sibling)
        await self.lb.on_request_end(sibling, True, lease=lease)
        self.assertTrue(self.ep.circuit_open)
        self.assertFalse(sibling.circuit_open)

    async def test_legacy_completion_requires_unambiguous_identity(self):
        a = await self.lb.on_request_start(self.ep)
        b = await self.lb.on_request_start(self.ep)
        with self.assertRaises(RuntimeError):
            await self.lb.on_request_end(self.ep, True)
        self.assertEqual(self.ep.active_requests, 2)
        await self.lb.on_request_end(self.ep, True, lease=a)
        await self.lb.on_request_end(self.ep, True, lease=b)
        self.assertEqual(self.ep.active_requests, 0)


class RecoveryEvidenceTests(unittest.IsolatedAsyncioTestCase):
    """Why a trial was admitted, how it ended, and how long nobody came.

    The 2026-10-16 GHCP review could only say "需要补充日志判断" about the
    small-request preference, and had to infer the nine-minute recovery gap from
    log archaeology. Both are facts the gateway knows and did not report.
    """

    def setUp(self):
        self.now = 500.0
        clock = patch.object(main.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.ep = main.CopilotEndpoint('acct', 'synthetic')
        self.lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=2,
                                    circuit_breaker_timeout=60)

    async def trip(self):
        for _ in range(2):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, lease=lease)

    @staticmethod
    def _small():
        return {'messages': [{'role': 'user', 'content': 'say ok'}]}

    @staticmethod
    def _large():
        # Incomplete estimate: an image makes the size unknowable, which is
        # exactly why such a request is never classified as small.
        return {'messages': [{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': 'https://fixture.invalid/x.png'}]}]}

    def _transitions(self, logs):
        return [r for r in logs.records if getattr(r, 'kind', '') == 'lb_circuit_transition']

    async def test_trial_admission_records_why_it_was_admitted(self):
        await self.trip()
        self.now += 60
        with self.assertLogs('main', level='INFO') as logs:
            lease = await self.lb.on_request_start(self.ep, payload=self._large())
        event = self._transitions(logs)[-1]
        self.assertEqual(event.transition_reason, 'trial_admitted')
        # No small request was seen during this cooldown, so the preference
        # window never applied and a large request is legitimately admitted.
        self.assertIs(event.small_input_evidence, False)
        self.assertEqual(event.preference_window_remaining, 0)
        self.assertIs(event.admitted_payload_small, False)
        self.assertIs(event.admitted_estimate_complete, False)
        await self.lb.on_request_end(self.ep, True, lease=lease)

    async def test_admission_records_the_preference_window_when_evidence_exists(self):
        await self.trip()
        # A qualifying small request arriving during cooldown is the evidence
        # that makes deferring a large one worth anything.
        self.lb._note_small_input(self.ep, self._small())
        self.now += 60
        with self.assertLogs('main', level='INFO') as logs:
            lease = await self.lb.on_request_start(self.ep, payload=self._small())
        event = self._transitions(logs)[-1]
        self.assertIs(event.small_input_evidence, True)
        self.assertIs(event.admitted_payload_small, True)
        self.assertIs(event.admitted_estimate_complete, True)
        await self.lb.on_request_end(self.ep, True, lease=lease)

    async def _inconclusive(self, *, timeline=None, headers=False, **end):
        await self.trip()
        self.now += 60
        trial = await self.lb.on_request_start(self.ep)
        token = None
        if timeline:
            line = request_timing.Timeline(clock=lambda: self.now)
            line.begin_attempt(180)
            if headers:
                line.headers_received()
            token = request_timing.CURRENT.set(line)
        try:
            with self.assertLogs('main', level='INFO') as logs:
                await self.lb.on_request_end(self.ep, False, lease=trial, **end)
        finally:
            if token:
                request_timing.CURRENT.reset(token)
        return self._transitions(logs)[-1]

    async def test_an_inconclusive_trial_records_which_kind_and_whether_headers_arrived(self):
        # The discriminator the review lacked: a cancellation before upstream
        # headers says nothing about upstream, while one after headers at least
        # proves it accepted the stream. Recorded now, so the question of
        # shortening this cooldown can be decided on data later.
        event = await self._inconclusive(timeline=True, headers=False, cancelled=True)
        self.assertEqual(event.transition_reason, 'trial_inconclusive')
        self.assertEqual(event.inconclusive_kind, 'cancelled')
        self.assertIs(event.upstream_headers_received, False,
                      'this is the 11:21:55 case: cancelled before upstream answered')

    async def test_headers_before_the_cancellation_are_recorded_as_such(self):
        event = await self._inconclusive(timeline=True, headers=True, cancelled=True)
        self.assertIs(event.upstream_headers_received, True,
                      'upstream did accept the stream, which is different evidence')

    async def test_without_a_measured_request_the_answer_is_unknown_not_false(self):
        # Reporting False here would invent evidence. Absence stays absence.
        event = await self._inconclusive(cancelled=True)
        self.assertIsNone(event.upstream_headers_received)

    async def test_a_neutral_trial_is_distinguished_from_a_cancelled_one(self):
        event = await self._inconclusive(timeline=True, is_client_error=True)
        self.assertEqual(event.inconclusive_kind, 'neutral')

    async def test_the_threshold_event_says_how_many_requests_contributed(self):
        """attempt count and request count are different numbers.

        One replay-safe failure is retried up to three times, so two business
        requests can fill a threshold of five. Reported, not silently changed:
        counting per request would delay protection when an endpoint is truly
        dead, since each failing request burns its own retries anyway.
        """
        lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=5,
                               circuit_breaker_timeout=60)
        with patch.object(main, 'telemetry_request_id', side_effect=['req-small'] * 3
                          + ['req-large'] * 2):
            with self.assertLogs('main', level='INFO') as logs:
                for _ in range(5):
                    lease = await lb.on_request_start(self.ep)
                    await lb.on_request_end(self.ep, False, lease=lease)
        event = [r for r in self._transitions(logs)
                 if r.transition_reason == 'failure_threshold'][-1]
        self.assertEqual(event.contributing_attempts, 5)
        self.assertEqual(event.contributing_requests, 2,
                         'five failed attempts came from two business requests')

    async def test_recovery_stats_report_how_long_nobody_has_tried(self):
        await self.trip()
        stats = self.lb.resilience_stats(self.ep)
        self.assertEqual(stats['circuit_state'], 'OPEN')
        self.assertEqual(stats['cooldown_elapsed_seconds'], 0,
                         'still cooling down, so nothing is overdue yet')
        self.now += 60 + 540  # cooldown over, then nine more minutes of silence
        stats = self.lb.resilience_stats(self.ep)
        self.assertEqual(stats['circuit_state'], 'HALF_OPEN')
        self.assertEqual(stats['cooldown_elapsed_seconds'], 540,
                         'the gap the review had to reconstruct is now a number')
        self.assertIsNone(stats['last_trial_outcome'], 'no trial has run yet')
        self.assertIsNone(stats['trial_in_flight_age_seconds'])

    async def test_recovery_stats_report_the_last_trial_and_the_one_in_flight(self):
        await self.trip()
        self.now += 60
        trial = await self.lb.on_request_start(self.ep)
        self.now += 90  # the very_large trial held the slot this long
        stats = self.lb.resilience_stats(self.ep)
        self.assertEqual(stats['trial_in_flight_age_seconds'], 90)
        self.assertEqual(stats['seconds_since_last_trial'], 90)
        await self.lb.on_request_end(self.ep, False, lease=trial, cancelled=True)
        stats = self.lb.resilience_stats(self.ep)
        self.assertEqual(stats['last_trial_outcome'], 'inconclusive')
        self.assertEqual(stats['last_trial_age_seconds'], 0)
        self.assertIsNone(stats['trial_in_flight_age_seconds'],
                          'the slot was released, so nothing is in flight')
        self.now += 60
        next_trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, True, lease=next_trial)
        self.assertEqual(self.lb.resilience_stats(self.ep)['last_trial_outcome'], 'succeeded')

    async def test_a_closed_circuit_reports_no_pending_recovery(self):
        stats = self.lb.resilience_stats(self.ep)
        self.assertEqual(stats['circuit_state'], 'CLOSED')
        self.assertEqual(stats['cooldown_elapsed_seconds'], 0)
        self.assertIsNone(stats['trial_in_flight_age_seconds'])


class DeliberatelyUnchangedTests(unittest.IsolatedAsyncioTestCase):
    """What the 2026-10-16 review proposed and this round deliberately did not do.

    These are not aspirational: each pins current behaviour so that changing it
    later is a decision someone makes on purpose, with evidence, rather than a
    side effect. The reasoning lives in docs/RESILIENCE.md.
    """

    def setUp(self):
        self.now = 1000.0
        clock = patch.object(main.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.ep = main.CopilotEndpoint('acct', 'synthetic')
        self.lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=2,
                                    circuit_breaker_timeout=30)

    async def trip(self):
        for _ in range(2):
            lease = await self.lb.on_request_start(self.ep)
            await self.lb.on_request_end(self.ep, False, lease=lease)

    async def test_an_inconclusive_trial_still_rearms_the_full_cooldown(self):
        # Shortening this would help the 11:21:55 case (a client disconnected
        # before upstream answered, costing 60 more seconds for zero evidence)
        # but it also guards against a cancel/reprobe storm. The discriminator
        # is now logged; the policy waits for data.
        await self.trip()
        self.now += 30
        trial = await self.lb.on_request_start(self.ep)
        await self.lb.on_request_end(self.ep, False, lease=trial, cancelled=True)
        self.assertEqual(self.lb.retry_after(), 30,
                         'a cancelled trial still costs a full cooldown, on purpose')

    async def test_the_breaker_still_counts_attempts_rather_than_requests(self):
        # Counting per request would mean a genuinely dead endpoint needs five
        # failing requests to trip, each burning its own retries first. The
        # amplification is reported on the transition event instead.
        lb = main.LoadBalancer([self.ep], circuit_breaker_threshold=3,
                               circuit_breaker_timeout=30)
        with patch.object(main, 'telemetry_request_id', return_value='one-request'):
            for _ in range(3):
                lease = await lb.on_request_start(self.ep)
                await lb.on_request_end(self.ep, False, lease=lease)
        self.assertTrue(self.ep.circuit_open,
                        'three attempts of one request still trip a threshold of three')

    async def test_nothing_probes_the_upstream_on_its_own(self):
        # A background probe would send a real, billable inference request. The
        # wait is exposed as a number instead; intervening stays a human call.
        await self.trip()
        self.now += 30 + 600
        self.assertEqual(self.lb.circuit_state(self.ep), 'HALF_OPEN')
        self.assertGreater(self.lb.resilience_stats(self.ep)['cooldown_elapsed_seconds'], 0)
        self.assertIsNone(self.lb.resilience_stats(self.ep)['last_trial_outcome'],
                          'no trial has happened, so nothing synthetic ran')
        self.assertEqual(self.ep.total_requests, 2, 'only the two real failures')
