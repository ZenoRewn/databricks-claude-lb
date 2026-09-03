import asyncio
from contextlib import suppress
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from starlette.requests import ClientDisconnect

import main


class _StreamResponse:
    def __init__(self, chunks, block_after_chunks=False):
        self.status_code = 200
        self._chunks = chunks
        self._block_after_chunks = block_after_chunks
        self.closed = False
        self.blocked = asyncio.Event()
        self._release = asyncio.Event()

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk
        if self._block_after_chunks:
            self.blocked.set()
            await self._release.wait()

    async def aclose(self):
        self.closed = True
        self._release.set()


class _StreamClient:
    def __init__(self, outcomes):
        self._outcomes = iter(outcomes)

    def build_request(self, method, url, json, headers):
        return httpx.Request(method, url, json=json, headers=headers)

    async def send(self, request, stream=False):
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _BlockingClient:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def post(self, url, json, headers):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


class _ErrorStreamResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.closed = False

    async def aread(self):
        return self._body

    async def aclose(self):
        self.closed = True


class CopilotRequestLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def _make_proxy(self, client, threshold=1):
        endpoint = main.CopilotEndpoint(
            name="copilot-test",
            github_token="token",
            models=["gpt-test"],
            session_token="session-token",
            session_token_expires_at=2**31,
        )
        load_balancer = main.LoadBalancer(
            [endpoint], circuit_breaker_threshold=threshold
        )
        original_start = load_balancer.on_request_start
        original_end = load_balancer.on_request_end
        load_balancer.on_request_start = AsyncMock(wraps=original_start)
        load_balancer.on_request_end = AsyncMock(wraps=original_end)

        proxy = object.__new__(main.CopilotProxy)
        proxy.load_balancer = load_balancer
        proxy.api_key = "local-key"
        proxy.client = client
        proxy.global_stats = main.GlobalStats()
        proxy._token_locks = {}
        proxy._stream_connections = {}
        proxy._stream_overload_since = {}
        proxy.stream_disconnects_detected_total = 0
        proxy.stream_forced_releases_total = 0
        proxy.pool_timeout_total = 0
        proxy.pool_timeout_saturated_total = 0
        proxy.pool_timeout_upstream_stall_total = 0
        proxy.stream_truncated_no_completion_total = 0
        proxy.stream_truncated_no_completion_by_model = {}
        proxy.stream_read_timeout_total = 0
        proxy.stream_pump_queue_full_events_total = 0
        # Fields introduced when we added the DNS+TCP upstream probe on PoolTimeout.
        # We stub out the probe so tests never actually touch the network — otherwise
        # a sandboxed CI would hang on getaddrinfo for the placeholder host.
        proxy._probe_lock = asyncio.Lock()
        proxy._last_probe_at = 0.0
        proxy._last_probe_result = None
        proxy.last_negotiated_http_version = None
        proxy._probe_upstream_connect = AsyncMock(return_value={
            "host": "test", "port": 443, "dns_ms": 1, "tcp_ms": 1,
            "ok": True, "error": None, "resolved_ips": ["127.0.0.1"], "cached": False,
        })
        proxy._build_headers = AsyncMock(return_value={"Authorization": "Bearer test"})
        return proxy, load_balancer, endpoint

    async def test_non_stream_pool_timeout_is_local_and_retry_ends_once_per_attempt(self):
        # PoolTimeout with a second endpoint present exercises the retry path.
        # (With only one endpoint, the upstream_stall classification now fails
        # fast — see test_non_stream_pool_timeout_single_endpoint_upstream_stall_fails_fast.)
        request = httpx.Request("POST", "https://example.test/chat/completions")
        success = httpx.Response(
            200,
            request=request,
            json={"choices": [], "usage": {}},
        )
        client = unittest.mock.Mock()
        client.post = AsyncMock(
            side_effect=[httpx.PoolTimeout("pool exhausted", request=request), success]
        )
        proxy, load_balancer, endpoint = self._make_proxy(client)
        second_endpoint = main.CopilotEndpoint(
            name="copilot-test-2",
            github_token="token",
            models=[],
            session_token="session-token",
            session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(second_endpoint)

        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            response = await proxy.proxy_chat_completions(
                {"model": "gpt-test", "messages": []}, stream=False
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(second_endpoint.active_requests, 0)
        # Both attempts hit exactly one endpoint each (round-robin over 2).
        self.assertEqual(endpoint.total_requests + second_endpoint.total_requests, 2)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_start.await_count, 2)
        self.assertEqual(load_balancer.on_request_end.await_count, 2)

    async def test_non_stream_pool_timeout_single_endpoint_upstream_stall_fails_fast(self):
        """Single endpoint + upstream_connect_stalled must bail out immediately.

        Retrying against the only endpoint would just replay the same httpx pool
        wait — up to POOL_ACQUIRE_TIMEOUT * max_retries seconds of pain the user
        cannot escape. We fail after the first attempt so Codex sees the error
        and can decide whether to retry at its layer.
        """
        request = httpx.Request("POST", "https://example.test/chat/completions")
        client = unittest.mock.Mock()
        client.post = AsyncMock(
            side_effect=httpx.PoolTimeout("pool exhausted", request=request)
        )
        proxy, load_balancer, endpoint = self._make_proxy(client)

        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(HTTPException) as ctx:
                await proxy.proxy_chat_completions(
                    {"model": "gpt-test", "messages": []}, stream=False
                )

        self.assertEqual(ctx.exception.status_code, 503)
        # Exactly ONE attempt happened — no retry.
        self.assertEqual(endpoint.total_requests, 1)
        self.assertEqual(client.post.await_count, 1)
        # Accounting still balances.
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_non_stream_401_refresh_failure_does_not_double_decrement(self):
        request = httpx.Request("POST", "https://example.test/chat/completions")
        unauthorized = httpx.Response(
            401, request=request, json={"error": {"message": "expired"}}
        )
        client = unittest.mock.Mock()
        client.post = AsyncMock(return_value=unauthorized)
        proxy, load_balancer, endpoint = self._make_proxy(client, threshold=10)
        proxy.get_session_token = AsyncMock(side_effect=RuntimeError("refresh failed"))

        # Keep one unrelated request active. A double decrement would steal its slot.
        endpoint.active_requests = 1
        with self.assertRaises(HTTPException) as ctx:
            await proxy.proxy_chat_completions(
                {"model": "gpt-test", "messages": []}, stream=False
            )

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_non_stream_malformed_2xx_does_not_steal_another_request_slot(self):
        request = httpx.Request("POST", "https://example.test/chat/completions")
        malformed = httpx.Response(200, request=request, content=b"not-json")
        success = httpx.Response(
            200, request=request, json={"choices": [], "usage": {}}
        )
        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=[malformed, success])
        proxy, load_balancer, endpoint = self._make_proxy(client, threshold=10)
        endpoint.active_requests = 1

        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            response = await proxy.proxy_chat_completions(
                {"model": "gpt-test", "messages": []}, stream=False
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 2)
        self.assertEqual(load_balancer.on_request_end.await_count, 2)

    async def test_non_stream_usage_record_failure_does_not_double_decrement(self):
        request = httpx.Request("POST", "https://example.test/responses")
        responses = [
            httpx.Response(200, request=request, json={"usage": {}}),
            httpx.Response(200, request=request, json={"usage": {}}),
        ]
        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=responses)
        proxy, load_balancer, endpoint = self._make_proxy(client, threshold=10)
        endpoint.active_requests = 1
        proxy._record_usage = unittest.mock.Mock(
            side_effect=[RuntimeError("usage store failed"), None]
        )

        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            response = await proxy.proxy_responses(
                {"model": "gpt-test", "input": "hello"}, stream=False
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 2)
        self.assertEqual(load_balancer.on_request_end.await_count, 2)

    async def test_non_stream_task_cancellation_releases_slot_without_circuit_error(self):
        client = _BlockingClient()
        proxy, load_balancer, endpoint = self._make_proxy(client, threshold=1)
        task = asyncio.create_task(proxy.proxy_responses(
            {"model": "gpt-test", "input": "hello"}, stream=False
        ))
        await asyncio.wait_for(client.started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(client.cancelled.is_set())
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_header_build_cancellation_releases_slot_without_circuit_error(self):
        proxy, load_balancer, endpoint = self._make_proxy(unittest.mock.Mock(), threshold=1)
        started = asyncio.Event()

        async def blocking_headers(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        proxy._build_headers = blocking_headers
        task = asyncio.create_task(proxy.proxy_responses(
            {"model": "gpt-test", "input": "hello"}, stream=False
        ))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_buffered_wait_cancels_upstream_after_confirmed_disconnect(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def buffered_work():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with self.assertRaises(HTTPException) as ctx:
            await main._await_with_disconnect(
                buffered_work(), AsyncMock(return_value=True), interval=0.001
            )

        self.assertTrue(started.is_set())
        self.assertTrue(cancelled.is_set())
        self.assertEqual(ctx.exception.status_code, 499)

    async def test_chat_responses_adapter_passes_disconnect_cancellation_to_buffered_work(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_route(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(main, "_route_openai_responses", new=fake_route):
            with self.assertRaises(HTTPException) as ctx:
                await main._route_chat_via_responses(
                    {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hi"}]},
                    stream=True,
                    disconnect_checker=AsyncMock(return_value=True),
                )

        self.assertTrue(started.is_set())
        self.assertTrue(cancelled.is_set())
        self.assertEqual(ctx.exception.status_code, 499)

    async def test_stream_generator_aclose_returns_active_request_exactly_once(self):
        upstream = _StreamResponse([b"data: {\"id\":\"one\"}\n\n"], block_after_chunks=True)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        iterator = response.body_iterator

        self.assertEqual(await anext(iterator), b"data: {\"id\":\"one\"}\n\n")
        await iterator.aclose()

        self.assertTrue(upstream.closed)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_stream_consumer_cancellation_then_response_cleanup_releases_once(self):
        upstream = _StreamResponse([b"data: {\"id\":\"one\"}\n\n"], block_after_chunks=True)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        iterator = response.body_iterator
        self.assertEqual(await anext(iterator), b"data: {\"id\":\"one\"}\n\n")

        next_chunk = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(upstream.blocked.wait(), timeout=1)
        next_chunk.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await next_chunk
        await iterator.aclose()

        self.assertTrue(upstream.closed)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_stream_pool_timeout_does_not_trip_circuit_or_double_end_retry(self):
        # PoolTimeout must not trip the circuit or double-end the retry when
        # the retry does happen (2+ endpoints so the single-endpoint fail-fast
        # heuristic doesn't apply). See
        # test_stream_pool_timeout_single_endpoint_fails_fast for the 1-endpoint
        # behavior.
        request = httpx.Request("POST", "https://example.test/chat/completions")
        pool_timeout = httpx.PoolTimeout("pool exhausted", request=request)
        success = _StreamResponse([b"data: [DONE]\n\n"])
        proxy, load_balancer, endpoint = self._make_proxy(
            _StreamClient([pool_timeout, success]), threshold=1
        )
        second_endpoint = main.CopilotEndpoint(
            name="copilot-test-2",
            github_token="token",
            models=[],
            session_token="session-token",
            session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(second_endpoint)
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertEqual(chunks, [b"data: [DONE]\n\n"])
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(second_endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_start.await_count, 2)
        self.assertEqual(load_balancer.on_request_end.await_count, 2)

    async def test_stream_pool_timeout_single_endpoint_fails_fast(self):
        """Streaming with a single endpoint bails out on upstream_stall PoolTimeout.

        Symmetric to test_non_stream_pool_timeout_single_endpoint_upstream_stall_fails_fast:
        the retry loop would just replay the same shared httpx pool wait, so we
        surface an explicit SSE error immediately and let Codex decide.
        """
        request = httpx.Request("POST", "https://example.test/chat/completions")
        pool_timeout = httpx.PoolTimeout("pool exhausted", request=request)
        proxy, load_balancer, endpoint = self._make_proxy(
            _StreamClient([pool_timeout]), threshold=10
        )
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            chunks = [chunk async for chunk in response.body_iterator]

        # Exactly one SSE error frame followed by [DONE] for chat api_type.
        self.assertEqual(len(chunks), 1)
        joined = chunks[0]
        self.assertIn(b"upstream_connect_stalled", joined)
        self.assertIn(b"[DONE]", joined)
        # Only one attempt — no retry.
        self.assertEqual(endpoint.active_requests, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_stream_retry_updates_registry_to_current_endpoint(self):
        request = httpx.Request("POST", "https://example.test/chat/completions")
        first_error = httpx.ConnectError("connect failed", request=request)
        success = _StreamResponse([b"data: {\"id\":\"two\"}\n\n"], block_after_chunks=True)
        proxy, load_balancer, first_endpoint = self._make_proxy(
            _StreamClient([first_error, success]), threshold=10
        )
        second_endpoint = main.CopilotEndpoint(
            name="copilot-test-2",
            github_token="token",
            models=["gpt-test"],
            session_token="session-token",
            session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(second_endpoint)
        proxy._select_endpoint = unittest.mock.Mock(return_value=second_endpoint)
        await load_balancer.on_request_start(first_endpoint)

        response = await proxy._stream_response(
            first_endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        iterator = response.body_iterator
        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            self.assertEqual(await anext(iterator), b"data: {\"id\":\"two\"}\n\n")

        self.assertEqual(len(proxy._stream_connections), 1)
        registry_item = next(iter(proxy._stream_connections.values()))
        self.assertEqual(registry_item["endpoint"], second_endpoint.name)
        await iterator.aclose()
        self.assertEqual(first_endpoint.active_requests, 0)
        self.assertEqual(second_endpoint.active_requests, 0)

    async def test_asgi_send_disconnect_closes_upstream_and_releases_slot(self):
        upstream = _StreamResponse([b"data: {\"id\":\"one\"}\n\n"], block_after_chunks=True)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)
        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                raise OSError("downstream closed")

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        with self.assertRaises(ClientDisconnect):
            await response(scope, receive, send)

        self.assertTrue(upstream.closed)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)
        self.assertEqual(proxy._stream_connections, {})

    async def test_asgi_response_start_failure_releases_unstarted_stream_slot(self):
        upstream = _StreamResponse([b"unused"])
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)
        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(_message):
            raise OSError("downstream closed before response start")

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        with self.assertRaises(ClientDisconnect):
            await response(scope, receive, send)

        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)
        self.assertEqual(proxy._stream_connections, {})

    async def test_asgi_pre_24_disconnect_listener_releases_stream_slot(self):
        upstream = _StreamResponse([b"data: {\"id\":\"one\"}\n\n"], block_after_chunks=True)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)
        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        sent_body = asyncio.Event()

        async def receive():
            await sent_body.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                sent_body.set()
                await asyncio.sleep(0)

        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await response(scope, receive, send)

        self.assertTrue(upstream.closed)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)
        self.assertEqual(proxy._stream_connections, {})

    async def test_stream_end_log_carries_request_id_and_outcome(self):
        """Every Codex stream must emit exactly one `[Copilot stream_end]` line
        with a stable set of fields, so post-mortem doesn't require /metrics.

        Verifies:
          * INFO log for `outcome=completed` on a happy Responses stream
          * ERROR log for `outcome=truncated` on a stream that never emits
            `response.completed`
          * `req=<id>` field carries the request_id the caller passed in
          * Key fields (`endpoint`, `model`, `api_type`, `elapsed`, `chunks`,
            `first_event`, `last_event`, `saw_completion`) are all present
        """
        import logging
        # Case A: clean completion with event: header form (exercises new parser).
        clean_upstream = _StreamResponse([
            b"event: response.created\ndata: {\"id\":\"r_1\"}\n\n",
            b"event: response.completed\ndata: {\"response\":{\"usage\":"
            b"{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        ])
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([clean_upstream]))
        await load_balancer.on_request_start(endpoint)

        with self.assertLogs("main", level="INFO") as log_ctx_ok:
            response = await proxy._stream_response(
                endpoint,
                "https://example.test/responses",
                {"model": "gpt-test", "input": "hi"},
                {},
                "gpt-test",
                "responses",
                main.time.time(),
                request_id="req_smoketest",
            )
            _ = b"".join([c async for c in response.body_iterator])

        end_lines = [r for r in log_ctx_ok.records
                     if "[Copilot stream_end]" in r.getMessage()]
        self.assertEqual(len(end_lines), 1, "exactly one stream_end log per request")
        msg = end_lines[0].getMessage()
        self.assertIn("req=req_smoketest", msg)
        self.assertIn("outcome=completed", msg)
        self.assertIn("api_type=responses", msg)
        self.assertIn("model=gpt-test", msg)
        self.assertIn("endpoint=copilot-test", msg)
        self.assertIn("saw_completion=True", msg)
        self.assertIn("first_event=response.created", msg)
        self.assertIn("last_event=response.completed", msg)
        self.assertEqual(end_lines[0].levelno, logging.INFO)

        # Case B: truncation → ERROR level, outcome=truncated.
        trunc_upstream = _StreamResponse([b"data: {\"type\":\"response.output_text.delta\"}\n\n"])
        proxy2, lb2, ep2 = self._make_proxy(_StreamClient([trunc_upstream]))
        await lb2.on_request_start(ep2)
        with self.assertLogs("main", level="INFO") as log_ctx_bad:
            response = await proxy2._stream_response(
                ep2,
                "https://example.test/responses",
                {"model": "gpt-test", "input": "hi"},
                {},
                "gpt-test",
                "responses",
                main.time.time(),
                request_id="req_trunc",
            )
            body = b"".join([c async for c in response.body_iterator])

        # SSE body must carry the request_id in error.metadata so client-side
        # error correlates back to the log line.
        self.assertIn(b'"request_id": "req_trunc"', body)
        self.assertIn(b'"endpoint": "copilot-test"', body)

        end_lines = [r for r in log_ctx_bad.records
                     if "[Copilot stream_end]" in r.getMessage()]
        self.assertEqual(len(end_lines), 1)
        msg = end_lines[0].getMessage()
        self.assertIn("req=req_trunc", msg)
        self.assertIn("outcome=truncated", msg)
        self.assertIn("saw_completion=False", msg)
        self.assertIn("sent_any_chunk=True", msg)
        self.assertEqual(end_lines[0].levelno, logging.ERROR)

    async def test_stream_responses_missing_completion_event_emits_upstream_truncated(self):
        """P0 regression: `saw_completion` used to be uninitialized on the Copilot
        path. When upstream sent chunks but never emitted `response.completed`,
        the truncation check hit `UnboundLocalError` and the client saw a raw
        socket close ("stream closed before response.completed"). Now the LB must
        emit an explicit ``response.failed{code: upstream_truncated}`` event.
        """
        # An SSE stream carrying only mid-response events; upstream EOF closes it
        # without any `response.completed` marker.
        chunks = [
            b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            b"data: {\"type\":\"response.output_text.delta\",\"delta\":\" there\"}\n\n",
        ]
        upstream = _StreamResponse(chunks)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/responses",
            {"model": "gpt-test", "input": "hello"},
            {},
            "gpt-test",
            "responses",
            main.time.time(),
        )
        body = b"".join([c async for c in response.body_iterator])

        # Every mid-stream chunk must pass through untouched…
        for original in chunks:
            self.assertIn(original, body)
        # …and the LB must append a Responses-shaped failure event (no `[DONE]`
        # sentinel, which would break Codex/JS SDK JSON parsing).
        self.assertIn(b'"type": "response.failed"', body)
        self.assertIn(b'"code": "upstream_truncated"', body)
        self.assertNotIn(b"data: [DONE]\n\n", body)
        # Counters increment for both aggregate and (model, api_type) breakdown.
        self.assertEqual(proxy.stream_truncated_no_completion_total, 1)
        self.assertEqual(
            proxy.stream_truncated_no_completion_by_model.get(("gpt-test", "responses")), 1
        )
        self.assertEqual(endpoint.active_requests, 0)

    async def test_stream_responses_event_header_terminal_is_accepted(self):
        """SSE per WHATWG allows the terminal signal to be carried on the
        ``event:`` header line instead of inside the ``data:`` payload. Some
        GHCP model releases emit ``event: response.completed\\ndata: {"id":...}``
        with no ``type`` field in the payload. The LB must treat that as a
        legitimate completion and NOT fire silent-truncation.
        """
        chunks = [
            b"event: response.created\ndata: {\"id\":\"resp_1\"}\n\n",
            b"event: response.output_text.delta\ndata: {\"delta\":\"ok\"}\n\n",
            # Real Responses API `response.completed` payload keeps usage nested
            # under `response`; here we omit the payload `type` field on purpose
            # so the terminal signal must be recognised via the `event:` header.
            b"event: response.completed\ndata: {\"response\":{\"id\":\"resp_1\","
            b"\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n",
        ]
        upstream = _StreamResponse(chunks)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/responses",
            {"model": "gpt-test", "input": "hello"},
            {},
            "gpt-test",
            "responses",
            main.time.time(),
        )
        body = b"".join([c async for c in response.body_iterator])

        # Client sees the raw upstream verbatim, no synthetic terminal appended.
        for original in chunks:
            self.assertIn(original, body)
        self.assertNotIn(b'"code": "upstream_truncated"', body)
        self.assertNotIn(b'"type": "response.failed"', body)
        # Truncation counter must remain zero — this stream completed cleanly
        # through the header path.
        self.assertEqual(proxy.stream_truncated_no_completion_total, 0)
        self.assertEqual(proxy.stream_truncated_no_completion_by_model, {})
        # Usage from the terminal payload must be recorded.
        self.assertEqual(endpoint.total_input_tokens, 10)
        self.assertEqual(endpoint.total_output_tokens, 2)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)

    async def test_stream_chat_missing_done_emits_upstream_truncated(self):
        """Chat/completions companion of the Responses-truncation test."""
        chunks = [
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        ]
        upstream = _StreamResponse(chunks)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": []},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        body = b"".join([c async for c in response.body_iterator])

        self.assertIn(chunks[0], body)
        # Chat truncation path must append `[DONE]` after the synthetic error so
        # legacy clients still see a valid stream end.
        self.assertIn(b'"code": "upstream_truncated"', body)
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))
        self.assertEqual(proxy.stream_truncated_no_completion_total, 1)
        self.assertEqual(
            proxy.stream_truncated_no_completion_by_model.get(("gpt-test", "chat")), 1
        )

    async def test_stream_unsupported_model_returns_explicit_sse_error(self):
        upstream = _ErrorStreamResponse(
            404, b'{"error":{"code":"model_not_found","message":"unsupported"}}'
        )
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)
        response = await proxy._stream_response(
            endpoint,
            "https://example.test/responses",
            {"model": "gpt-test", "input": "hello"},
            {},
            "gpt-test",
            "responses",
            main.time.time(),
        )
        chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(chunks)

        # Responses API terminal MUST be `response.failed` shape (no `[DONE]`).
        # Sending `[DONE]` on a Responses stream breaks Codex Desktop and the
        # OpenAI JS SDK — they `serde_json` each `data:` line and `[DONE]` is
        # not valid JSON. See commit 6a0a6aa and `_sse_terminal_error`.
        self.assertIn(b'"type": "response.failed"', body)
        self.assertIn(b'"code": "unsupported_model"', body)
        self.assertNotIn(b"data: [DONE]", body)
        self.assertTrue(upstream.closed)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)

    async def test_monitor_releases_confirmed_disconnect_only_after_sustained_high_water(self):
        proxy, load_balancer, endpoint = self._make_proxy(unittest.mock.Mock())
        endpoint.active_requests = 1
        second_endpoint = main.CopilotEndpoint(
            name="copilot-test-2", github_token="token", models=["gpt-test"]
        )
        second_endpoint.active_requests = 1
        load_balancer.endpoints.append(second_endpoint)
        owner = asyncio.create_task(asyncio.Event().wait())
        released = asyncio.Event()

        async def release():
            proxy._stream_connections.pop("leaked", None)
            await load_balancer.on_request_end(
                endpoint, success=False, is_client_error=True
            )
            released.set()

        now = main.time.monotonic()
        proxy._stream_connections["leaked"] = {
            "endpoint": endpoint.name,
            "task": owner,
            "release": release,
            "disconnect_checker": AsyncMock(return_value=True),
            "started_at": now,
            "last_upstream_activity_at": now,
            "disconnected_since": None,
        }
        monitor = asyncio.create_task(proxy.connection_monitor_loop(
            interval=0.005,
            high_watermark=2,
            overload_grace=0,
            disconnect_grace=0,
        ))
        await asyncio.wait_for(released.wait(), timeout=1)
        await asyncio.sleep(0)
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor
        with suppress(asyncio.CancelledError):
            await owner

        self.assertTrue(owner.cancelled())
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(second_endpoint.active_requests, 1)
        self.assertEqual(proxy.stream_disconnects_detected_total, 1)
        self.assertEqual(proxy.stream_forced_releases_total, 1)

    async def test_monitor_never_kills_connected_long_thinking_stream(self):
        proxy, _load_balancer, endpoint = self._make_proxy(unittest.mock.Mock())
        endpoint.active_requests = 1
        owner = asyncio.create_task(asyncio.Event().wait())
        release = AsyncMock()
        now = main.time.monotonic()
        proxy._stream_connections["healthy"] = {
            "endpoint": endpoint.name,
            "task": owner,
            "release": release,
            "disconnect_checker": AsyncMock(return_value=False),
            "started_at": now - 3600,
            "last_upstream_activity_at": now - 3600,
            "disconnected_since": None,
        }
        monitor = asyncio.create_task(proxy.connection_monitor_loop(
            interval=0.005,
            high_watermark=1,
            overload_grace=0,
            disconnect_grace=0,
        ))
        await asyncio.sleep(0.03)
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor

        self.assertFalse(owner.done())
        release.assert_not_awaited()
        self.assertEqual(proxy.stream_forced_releases_total, 0)
        owner.cancel()
        with suppress(asyncio.CancelledError):
            await owner

    async def test_prometheus_exposes_connection_lifecycle_metrics(self):
        proxy, _load_balancer, _endpoint = self._make_proxy(unittest.mock.Mock())
        previous = main.copilot_proxy
        main.copilot_proxy = proxy
        try:
            response = await main.metrics()
        finally:
            main.copilot_proxy = previous

        body = response.body.decode("utf-8")
        self.assertIn("copilot_stream_connections_active 0", body)
        self.assertIn("copilot_stream_forced_releases_total 0", body)
        self.assertIn("copilot_pool_timeout_total 0", body)
        self.assertIn(
            f"copilot_stream_high_watermark {main.COPILOT_STREAM_HIGH_WATERMARK}", body
        )
        # Newly exposed counters must be present with 0 baseline.
        self.assertIn("copilot_stream_truncated_no_completion_total 0", body)
        self.assertIn("copilot_stream_read_timeout_total 0", body)
        self.assertIn("copilot_stream_pump_queue_full_events_total 0", body)
        # read_timeout gauge exports -1 sentinel for unlimited (default: None).
        self.assertRegex(body, r"copilot_pool_read_timeout_seconds -1|copilot_pool_read_timeout_seconds \d")


class AzureRequestLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def _make_proxy(self, client, threshold=10):
        endpoint = main.AzureOpenAIEndpoint(
            name="azure-test",
            endpoint="https://example.test",
            api_key="token",
            deployments=["gpt-test"],
        )
        load_balancer = main.LoadBalancer(
            [endpoint], circuit_breaker_threshold=threshold
        )
        original_start = load_balancer.on_request_start
        original_end = load_balancer.on_request_end
        load_balancer.on_request_start = AsyncMock(wraps=original_start)
        load_balancer.on_request_end = AsyncMock(wraps=original_end)
        proxy = object.__new__(main.AzureOpenAIProxy)
        proxy.load_balancer = load_balancer
        proxy.api_key = "local-key"
        proxy.client = client
        proxy.global_stats = main.GlobalStats()
        return proxy, load_balancer, endpoint

    async def test_malformed_2xx_uses_exactly_once_accounting(self):
        request = httpx.Request("POST", "https://example.test/responses")
        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=[
            httpx.Response(200, request=request, content=b"not-json"),
            httpx.Response(200, request=request, json={"usage": {}}),
        ])
        proxy, load_balancer, endpoint = self._make_proxy(client)
        endpoint.active_requests = 1

        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            response = await proxy.proxy_responses(
                {"model": "gpt-test", "input": "hello"}, stream=False
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 2)
        self.assertEqual(load_balancer.on_request_end.await_count, 2)

    async def test_usage_record_failure_uses_exactly_once_accounting(self):
        request = httpx.Request("POST", "https://example.test/responses")
        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=[
            httpx.Response(200, request=request, json={"usage": {}}),
            httpx.Response(200, request=request, json={"usage": {}}),
        ])
        proxy, load_balancer, endpoint = self._make_proxy(client)
        endpoint.active_requests = 1
        proxy._record_usage = unittest.mock.Mock(
            side_effect=[RuntimeError("usage store failed"), None]
        )

        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            response = await proxy.proxy_responses(
                {"model": "gpt-test", "input": "hello"}, stream=False
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 2)
        self.assertEqual(load_balancer.on_request_end.await_count, 2)

    async def test_buffered_cancellation_releases_slot_without_circuit_error(self):
        client = _BlockingClient()
        proxy, load_balancer, endpoint = self._make_proxy(client, threshold=1)
        task = asyncio.create_task(proxy.proxy_responses(
            {"model": "gpt-test", "input": "hello"}, stream=False
        ))
        await asyncio.wait_for(client.started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(client.cancelled.is_set())
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertFalse(endpoint.circuit_open)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

    async def test_stream_send_disconnect_closes_upstream_and_releases_slot(self):
        upstream = _StreamResponse([b"data: {\"id\":\"one\"}\n\n"], block_after_chunks=True)
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        await load_balancer.on_request_start(endpoint)
        response = await proxy._stream_response(
            endpoint,
            "https://example.test/responses",
            {"model": "gpt-test", "input": "hello"},
            {},
            "gpt-test",
            "responses",
            main.time.time(),
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                raise OSError("downstream closed")

        with self.assertRaises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        self.assertTrue(upstream.closed)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)


if __name__ == "__main__":
    unittest.main()


class RequestIdSurfaceTests(unittest.IsolatedAsyncioTestCase):
    """Codex-rs (openai/codex codex-api/src/sse/responses.rs) reads
    ``REQUEST_ID_HEADER = "x-request-id"`` and does NOT parse
    ``error.metadata`` from an SSE ``response.failed`` payload. These tests
    guard the three surfacing channels so no future refactor can silently
    strand the ID from the client's view.
    """

    def test_apply_request_id_prefix_prepends_when_metadata_has_id(self):
        out = main._apply_request_id_prefix("upstream stalled", {"request_id": "req_x"})
        self.assertTrue(out.startswith("[req=req_x] "))
        self.assertIn("upstream stalled", out)

    def test_apply_request_id_prefix_noop_when_no_metadata(self):
        self.assertEqual(main._apply_request_id_prefix("boom", None), "boom")
        self.assertEqual(main._apply_request_id_prefix("boom", {}), "boom")
        self.assertEqual(main._apply_request_id_prefix("boom", {"endpoint": "x"}), "boom")

    def test_apply_request_id_prefix_idempotent(self):
        once = main._apply_request_id_prefix("hi", {"request_id": "req_x"})
        twice = main._apply_request_id_prefix(once, {"request_id": "req_x"})
        self.assertEqual(once, twice)  # already-prefixed message is left alone

    def test_sse_terminal_error_prefixes_message_and_embeds_metadata(self):
        payload = main._sse_terminal_error(
            "responses", "upstream_truncated", "stream died",
            metadata={"request_id": "req_xyz", "endpoint": "gh-1"},
        )
        text = payload.decode("utf-8")
        import json as _json
        data = _json.loads(text.split("data: ", 1)[1])
        err = data["response"]["error"]
        self.assertTrue(err["message"].startswith("[req=req_xyz] "))
        self.assertEqual(err["metadata"]["request_id"], "req_xyz")
        self.assertEqual(err["metadata"]["endpoint"], "gh-1")

    def test_request_id_response_headers_returns_pair(self):
        h = main._request_id_response_headers("req_abc")
        self.assertEqual(h.get("X-Request-Id"), "req_abc")
        self.assertEqual(h.get("OpenAI-Request-Id"), "req_abc")

    def test_request_id_response_headers_empty_when_none(self):
        self.assertEqual(main._request_id_response_headers(None), {})
        self.assertEqual(main._request_id_response_headers(""), {})
