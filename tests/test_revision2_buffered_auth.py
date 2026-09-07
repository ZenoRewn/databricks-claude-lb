"""Buffered 401 ownership: real loopback HTTP, actual token lock, no auth traffic."""
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import main


class BufferedAuth(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def fixture(self, statuses=(401, 200), state="half_open", api="responses"):
        calls, handlers = [], set()
        second_received, release_second = asyncio.Event(), asyncio.Event()
        async def origin(reader, writer):
            handlers.add(asyncio.current_task())
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                length = next((int(x.split(b":", 1)[1]) for x in head.split(b"\r\n")
                               if x.lower().startswith(b"content-length:")), 0)
                body = await reader.readexactly(length) if length else b""
                self.assertNotIn(b"authorization:", head.lower())
                expected_path = b"/responses" if api == "responses" else b"/chat/completions"
                self.assertTrue(head.startswith(b"POST " + expected_path + b" HTTP/1.1"))
                calls.append((head, body))
                if len(calls) == 2:
                    second_received.set()
                    await release_second.wait()
                status = statuses[min(len(calls) - 1, len(statuses) - 1)]
                if status == "read_error":
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n{")
                else:
                    payload = b'{"id":"synthetic","output":[],"choices":[],"usage":{}}'
                    writer.write(b"HTTP/1.1 " + str(status).encode() + b" Synthetic\r\n"
                                 b"Content-Type: application/json\r\nConnection: close\r\nContent-Length: "
                                 + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(asyncio.current_task())
        server = await asyncio.start_server(origin, "127.0.0.1", 0)
        ep = main.CopilotEndpoint("synthetic-held", "", models=["gpt-5.6-sol", "sibling-model"], api_types=[api])
        ep.session_base_url = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1])
        other = main.CopilotEndpoint("synthetic-other", "", models=["unrelated-model"], api_types=[api])
        lb = main.LoadBalancer([ep, other], circuit_breaker_timeout=60, circuit_breaker_threshold=4)
        ep.total_errors, ep.consecutive_errors = 5, 3
        if state == "half_open":
            lb._open(ep)
            ep.circuit_retry_at = 0
        p = main.CopilotProxy(lb, "")
        await p.client.aclose()
        p.client = httpx.AsyncClient(trust_env=False, timeout=3)
        p._build_headers = AsyncMock(return_value={})
        p._exchange_token = AsyncMock(side_effect=AssertionError("forbidden token exchange"))
        lb.on_request_start = AsyncMock(wraps=lb.on_request_start)
        lb.on_request_end = AsyncMock(wraps=lb.on_request_end)
        try:
            yield p, ep, other, calls, second_received, release_second
        finally:
            release_second.set()
            await p.client.aclose()
            server.close()
            await server.wait_closed()
            await asyncio.gather(*list(handlers), return_exceptions=True)

    def settled(self, p, ep, other, *, cancelled=0, completed=0, errors=5):
        lb = p.load_balancer
        self.assertEqual(ep.active_requests, 0)
        self.assertEqual(len(lb._attempts.get(id(ep), {})), 0)
        self.assertEqual(ep.cancelled_requests, cancelled)
        self.assertEqual(ep.completed_requests, completed)
        self.assertEqual(ep.total_errors, errors)
        self.assertEqual(ep.neutral_requests, 0)
        self.assertFalse(ep.half_open_in_flight)
        self.assertEqual(len(p.client._transport._pool._requests), 0)
        self.assertEqual(lb.on_request_start.await_count, 1)
        self.assertEqual(lb.on_request_end.await_count, 1)
        call = lb.on_request_end.await_args
        self.assertIs(call.args[0], ep)
        self.assertIs(call.kwargs["lease"].endpoint, ep)
        self.assertTrue(call.kwargs["lease"].ended)
        self.assertEqual(other.total_requests, 0)
        self.assertEqual(other.total_errors, 0)
        self.assertTrue(lb.is_available(other))
        p._exchange_token.assert_not_awaited()

    async def test_actual_token_lock_ordinary_sol_disconnect_before_teardown(self):
        async with self.fixture() as (p, ep, other, calls, _, __):
            lb = p.load_balancer
            lock = p._get_token_lock(ep)
            await lock.acquire()
            generation = ep.circuit_generation
            async def disconnected():
                return bool(lock._waiters)
            try:
                with patch.object(main, "copilot_proxy", p), patch.object(main, "azure_proxy", None):
                    with self.assertRaises(main.HTTPException) as caught:
                        await asyncio.wait_for(main._route_openai_chat(
                            {"model": "gpt-5.6-sol", "messages": []}, True,
                            disconnect_checker=disconnected), 3)
                self.assertEqual(caught.exception.status_code, 499)
                self.assertEqual(len(calls), 1)
                self.assertFalse(json.loads(calls[0][1])["stream"])
                self.settled(p, ep, other, cancelled=1)
                self.assertTrue(ep.circuit_open)
                self.assertEqual(ep.consecutive_errors, 3)
                self.assertEqual(ep.circuit_generation, generation + 1)
                self.assertGreater(ep.circuit_retry_at, time.monotonic())
                self.assertIsNone(p._select_endpoint("sibling-model", "responses"))
                ep.circuit_retry_at = time.monotonic() - 1
                self.assertTrue(lb.is_available(ep))
                self.assertIs(p._select_endpoint("sibling-model", "responses"), ep)
                self.assertIsNone(p._select_endpoint("sibling-model", "chat"))
                print("REV2_REAL_LOCK_OK", json.dumps(dict(status=499, posts=len(calls), active=0,
                      owned_attempts=0, cancelled=1, trial_free=True, pool=0, cooldown_reeligible=True)), flush=True)
            finally:
                lock.release()

    async def repair_case(self, state, api, outcome):
        statuses = (401, 401 if outcome == "second401" else 200)
        async with self.fixture(statuses, state, api) as (p, ep, other, calls, second, release):
            lb = p.load_balancer
            initial_generation = ep.circuit_generation
            refresh_entered, allow_refresh = asyncio.Event(), asyncio.Event()
            leases = []
            async def refresh(endpoint, force=False):
                self.assertIs(endpoint, ep)
                self.assertTrue(force)
                leases.append(lb.current_attempt(ep))
                refresh_entered.set()
                await allow_refresh.wait()
                if outcome == "failed_refresh":
                    raise RuntimeError("synthetic refresh failure")
                return "synthetic-unused"
            p.get_session_token = AsyncMock(side_effect=refresh)
            method = p.proxy_responses if api == "responses" else p.proxy_chat_completions
            task = asyncio.create_task(method({"model": "gpt-5.6-sol", "input": "synthetic"}, stream=False))
            try:
                await asyncio.wait_for(refresh_entered.wait(), 2)
                self.assertEqual(ep.active_requests, 1)
                self.assertEqual(lb.on_request_end.await_count, 0)
                self.assertEqual(ep.consecutive_errors, 3)
                self.assertEqual(ep.circuit_open, state == "half_open")
                self.assertEqual(ep.half_open_in_flight, state == "half_open")
                allow_refresh.set()
                if outcome != "failed_refresh":
                    await asyncio.wait_for(second.wait(), 2)
                    self.assertIs(lb.current_attempt(ep), leases[0])
                    self.assertEqual(lb.on_request_start.await_count, 1)
                    self.assertEqual(lb.on_request_end.await_count, 0)
                    self.assertEqual(ep.circuit_generation, initial_generation)
                    self.assertEqual(ep.consecutive_errors, 3)
                    self.assertEqual(ep.circuit_open, state == "half_open")
                    self.assertEqual(ep.half_open_in_flight, state == "half_open")
                    self.assertEqual(calls[0][1], calls[1][1], "only explicit 401 permits identical body retry")
                    release.set()
                if outcome == "success":
                    response = await asyncio.wait_for(task, 2)
                    self.assertEqual(response.status_code, 200)
                    self.settled(p, ep, other, completed=1)
                    self.assertFalse(ep.circuit_open)
                    self.assertEqual(ep.consecutive_errors, 0)
                else:
                    with self.assertRaises(main.HTTPException) as caught:
                        await asyncio.wait_for(task, 2)
                    self.assertEqual(caught.exception.status_code, 401)
                    self.settled(p, ep, other, errors=6)
                    self.assertTrue(ep.circuit_open)
                    self.assertEqual(ep.consecutive_errors, 4)
                self.assertEqual(len(calls), 1 if outcome == "failed_refresh" else 2)
                p.get_session_token.assert_awaited_once_with(ep, force=True)
                self.assertIs(lb.on_request_end.await_args.kwargs["lease"], leases[0])
                print("REV2_AUTH_OK", state, api, outcome, "posts", len(calls), "starts=1 ends=1", flush=True)
            finally:
                allow_refresh.set()
                release.set()
                if not task.done():
                    task.cancel()
                with suppress(BaseException):
                    await task

    async def test_successful_repair_retains_single_admission_closed_and_half_open(self):
        for state in ("closed", "half_open"):
            for api in ("responses", "chat"):
                with self.subTest(state=state, api=api):
                    await self.repair_case(state, api, "success")

    async def test_failed_refresh_and_second_401_settle_real_failure_once(self):
        for state in ("closed", "half_open"):
            for outcome in ("failed_refresh", "second401"):
                for api in ("responses", "chat"):
                    with self.subTest(state=state, api=api, outcome=outcome):
                        await self.repair_case(state, api, outcome)

    async def test_no_ambiguous_503_or_read_replay_even_after_401(self):
        for outcome in (503, "read_error"):
            for auth_first in (False, True):
                with self.subTest(outcome=outcome, auth_first=auth_first):
                    statuses = (401, outcome) if auth_first else (outcome,)
                    async with self.fixture(statuses) as (p, ep, other, calls, _, release):
                        p.get_session_token = AsyncMock(return_value="synthetic-unused")
                        release.set()
                        with self.assertRaises(main.HTTPException):
                            await asyncio.wait_for(p.proxy_responses({"model": "gpt-5.6-sol"}), 3)
                        self.assertEqual(len(calls), 2 if auth_first else 1)
                        self.assertEqual(p.get_session_token.await_count, int(auth_first))
                        self.settled(p, ep, other, errors=6)

    async def test_auth_retry_does_not_expand_original_total_post_budget(self):
        # Explicit 429 remains the inherited nonexecution retry policy. The
        # in-lease 401 repair must consume, not add to, the three-POST budget.
        async with self.fixture((401, 429, 429, 200), "closed") as (p, ep, other, calls, _, release):
            p.load_balancer.circuit_breaker_threshold = 10
            p.get_session_token = AsyncMock(return_value="synthetic-unused")
            release.set()
            with self.assertRaises(main.HTTPException) as caught:
                await asyncio.wait_for(p.proxy_responses({"model": "gpt-5.6-sol"}), 5)
            self.assertEqual(caught.exception.status_code, 429)
            self.assertEqual(len(calls), 3)
            self.assertEqual(ep.total_requests, 2)
            self.assertEqual(p.load_balancer.on_request_start.await_count, 2)
            self.assertEqual(p.load_balancer.on_request_end.await_count, 2)
            self.assertEqual(ep.total_errors, 7)
            self.assertEqual(ep.completed_requests, 0)
            self.assertEqual(ep.active_requests, 0)
            self.assertEqual(len(p.load_balancer._attempts.get(id(ep), {})), 0)
            self.assertEqual(len(p.client._transport._pool._requests), 0)
            self.assertEqual(other.total_requests, 0)
            p.get_session_token.assert_awaited_once_with(ep, force=True)

    async def test_repeated_cancel_joins_settlement_without_double_end(self):
        async with self.fixture() as (p, ep, other, calls, _, __):
            lock = p._get_token_lock(ep)
            await lock.acquire()
            entered, release = asyncio.Event(), asyncio.Event()
            original_end = p.load_balancer.on_request_end
            async def delayed_end(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original_end(*args, **kwargs)
            p.load_balancer.on_request_end = delayed_end
            task = asyncio.create_task(p.proxy_responses({"model": "gpt-5.6-sol"}))
            try:
                async def wait_lock():
                    while not lock._waiters:
                        await asyncio.sleep(.001)
                await asyncio.wait_for(wait_lock(), 2)
                task.cancel()
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(task.done(), "cancellation must join owned cleanup")
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                p.load_balancer.on_request_end = original_end
                self.settled(p, ep, other, cancelled=1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(ep.consecutive_errors, 3)
            finally:
                release.set()
                lock.release()
                if not task.done():
                    task.cancel()
                with suppress(BaseException):
                    await task
