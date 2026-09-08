import asyncio
from contextlib import suppress
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from starlette.requests import ClientDisconnect

import main


class _StreamResponse:
    def __init__(self, chunks, block_after_chunks=False, status_code=200,
                 headers=None, http_version="HTTP/1.1"):
        self.status_code = status_code
        # httpx.Response.headers 是 Headers (Mapping)；测试里用 dict 足够。
        self.headers = headers if headers is not None else {}
        self.http_version = http_version
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

    async def aread(self):
        # 兼容 _stream_response 内 `error_body = await response.aread()` 的调用
        return b"".join(self._chunks) if self._chunks else b""

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
    def __init__(self, status_code, body, headers=None, http_version="HTTP/1.1"):
        self.status_code = status_code
        self._body = body
        self.headers = headers if headers is not None else {}
        self.http_version = http_version
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
        # HTML 上游诊断 —— 与 CopilotProxy.__init__ 保持一致
        proxy.upstream_html_events_total = 0
        proxy.upstream_html_events_by_status = {}
        # 状态亲和 pinning 计数（handoff §7.2 防跨账户 opaque state 401）
        proxy.stateful_pinned_events = {}
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
        # (With only one endpoint, the pool timeout now fails
        # fast — see test_non_stream_pool_timeout_single_endpoint_fails_fast.)
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

    async def test_non_stream_pool_timeout_single_endpoint_fails_fast(self):
        """Single endpoint + pool_acquire_timeout must bail out immediately.

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
            with self.assertRaises(HTTPException) as caught:
                response = await proxy.proxy_chat_completions(
                    {"model": "gpt-test", "messages": []}, stream=False
                )

        self.assertEqual(caught.exception.status_code, 502)
        self.assertEqual(client.post.await_count, 1)  # Never replay a successful/ambiguous POST.
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

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
        self.assertEqual(endpoint.usage_record_errors, 1)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertEqual(client.post.await_count, 1)  # Never replay a successful/ambiguous POST.
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

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
        """Streaming with a single endpoint bails out on PoolTimeout.

        Symmetric to test_non_stream_pool_timeout_single_endpoint_fails_fast:
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
        self.assertIn(b"pool_acquire_timeout", joined)
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
            b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r_1","usage":'
            b'{"input_tokens":3,"output_tokens":1,"total_tokens":4}}}\n\n',
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
        """A named event with a client-valid JSON discriminator is accepted.

        Header-only success is intentionally rejected by separate protocol tests;
        WHATWG framing does not define the Responses payload schema.
        """
        chunks = [
            b"event: response.created\ndata: {\"id\":\"resp_1\"}\n\n",
            b"event: response.output_text.delta\ndata: {\"delta\":\"ok\"}\n\n",
            # Real Responses API `response.completed` payload keeps usage nested
            # under `response`; the pinned client requires a payload `type`
            # even when the event header also names the terminal.
            b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\","
            b"\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12}}}\n\n",
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

    async def test_stream_200_html_content_type_treated_as_upstream_html_error(self):
        # 上游 status=200 但 content-type=text/html —— Cloudflare "Just a moment"
        # 挑战页的典型形态。旧版会把 HTML 原字节 pump 给客户端；新版必须识别为
        # upstream_html_error 并触发软熔断。
        html_body = (
            b"<!DOCTYPE html><html><head><title>Just a moment...</title></head>"
            b"<body>Please enable JavaScript.</body></html>"
        )
        upstream = _ErrorStreamResponse(
            200,
            html_body,
            headers={"content-type": "text/html; charset=UTF-8", "cf-ray": "abc123-SIN"},
        )
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
        body = b"".join([chunk async for chunk in response.body_iterator])

        # 客户端拿到规范化 error，不含原始 HTML 字节。Chat 分支带 [DONE]。
        self.assertNotIn(b"<!DOCTYPE html", body)
        self.assertNotIn(b"<html", body)
        self.assertIn(b'"code": "upstream_html_error"', body)
        self.assertIn(b"cf-ray=abc123-SIN", body)
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))
        # 计数器 + 软熔断都要生效
        self.assertEqual(proxy.upstream_html_events_total, 1)
        self.assertEqual(endpoint.upstream_html_events_total, 1)
        self.assertGreater(endpoint.html_soft_cooldown_until, main.time.time())

    async def test_stream_403_html_triggers_soft_cooldown_and_endpoint_skip(self):
        # 上游 4xx HTML 也走软熔断路径。同时验证 _select_endpoint 会跳过冷却中的
        # endpoint —— 加一个 "healthy" endpoint，冷却状态下 _select_endpoint 应优先选它。
        html_body = b"<html><body>Access denied</body></html>"
        upstream = _ErrorStreamResponse(
            403, html_body, headers={"content-type": "text/html", "cf-ray": "xyz-999"},
        )
        proxy, load_balancer, endpoint = self._make_proxy(_StreamClient([upstream]))
        healthy = main.CopilotEndpoint(
            name="copilot-healthy", github_token="t2", models=["gpt-test"],
            session_token="s", session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(healthy)
        await load_balancer.on_request_start(endpoint)

        response = await proxy._stream_response(
            endpoint,
            "https://example.test/responses",
            {"model": "gpt-test", "input": "hi"},
            {},
            "gpt-test",
            "responses",
            main.time.time(),
        )
        body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertIn(b'"code": "upstream_html_error"', body)
        self.assertNotIn(b"<html", body)
        # Responses API 终止必须是 response.failed，不带 [DONE]
        self.assertIn(b'"type": "response.failed"', body)
        self.assertNotIn(b"data: [DONE]", body)
        self.assertGreater(endpoint.html_soft_cooldown_until, main.time.time())
        # _select_endpoint 应跳过 cooldown 中的原 endpoint，选 healthy
        selected = proxy._select_endpoint("gpt-test")
        self.assertIs(selected, healthy)

    async def test_upstream_ids_captured_and_propagated_to_sse_error(self):
        # 上游返回 502 HTML + cf-ray + x-github-request-id + server 三种诊断头。
        # 断言：SSE error message 尾部 + error.upstream_ids 都能拿到这些字段。
        html_body = b"<html><body>Bad Gateway</body></html>"
        upstream = _ErrorStreamResponse(
            502,
            html_body,
            headers={
                "content-type": "text/html; charset=UTF-8",
                "cf-ray": "abc123-SIN",
                "x-github-request-id": "F1E2:1234",
                "server": "cloudflare",
            },
        )
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
        body = b"".join([chunk async for chunk in response.body_iterator])

        # SSE error.message 尾部有 `cf-ray=abc123-SIN x-github-request-id=... server=cloudflare`
        self.assertIn(b"cf-ray=abc123-SIN", body)
        self.assertIn(b"x-github-request-id=F1E2:1234", body)
        # error.upstream_ids 结构化字段也要有
        self.assertIn(b'"upstream_ids"', body)
        # /stats 里 endpoint 的 HTML 计数 +1
        self.assertEqual(endpoint.upstream_html_events_total, 1)

    # -- 状态亲和 pinning ---------------------------------------------------

    def test_request_has_opaque_state_detects_previous_response_id(self):
        # Responses API + previous_response_id 非空 → True
        self.assertTrue(main.CopilotProxy._request_has_opaque_state(
            {"model": "gpt-test", "previous_response_id": "resp_abc"}, "responses"
        ))
        # 空串 / None / 不存在 → False
        self.assertFalse(main.CopilotProxy._request_has_opaque_state(
            {"previous_response_id": ""}, "responses"
        ))
        self.assertFalse(main.CopilotProxy._request_has_opaque_state(
            {"previous_response_id": None}, "responses"
        ))
        self.assertFalse(main.CopilotProxy._request_has_opaque_state({}, "responses"))

    def test_request_has_opaque_state_detects_encrypted_content(self):
        # 顶层 item.encrypted_content
        self.assertTrue(main.CopilotProxy._request_has_opaque_state(
            {"input": [{"encrypted_content": "cE=="}]}, "responses"
        ))
        # 嵌套在 content 数组里
        self.assertTrue(main.CopilotProxy._request_has_opaque_state(
            {"input": [{"content": [{"encrypted_content": "cE=="}]}]}, "responses"
        ))
        # 空 encrypted_content 不算
        self.assertFalse(main.CopilotProxy._request_has_opaque_state(
            {"input": [{"encrypted_content": ""}]}, "responses"
        ))

    def test_request_has_opaque_state_ignores_chat_completions(self):
        # Chat 协议不适用 opaque state 语义，即便同名字段存在也返 False
        self.assertFalse(main.CopilotProxy._request_has_opaque_state(
            {"previous_response_id": "resp_abc"}, "chat"
        ))
        self.assertFalse(main.CopilotProxy._request_has_opaque_state(
            {"messages": [{"role": "user", "content": "hi"}]}, "chat"
        ))

    async def test_select_endpoint_pinned_returns_pin_when_available(self):
        # 两个 endpoint 都健康时，pinned=A 应始终返回 A，无论 lb 策略偏向谁
        proxy, load_balancer, endpoint_a = self._make_proxy(unittest.mock.Mock())
        endpoint_b = main.CopilotEndpoint(
            name="copilot-b", github_token="t2", models=["gpt-test"],
            session_token="s", session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(endpoint_b)
        # 让 A 的活跃请求数远高于 B，least_requests 策略下 A 本来会输
        endpoint_a.active_requests = 10
        endpoint_b.active_requests = 0
        # 不 pinned → 应该选 B（least_requests）
        chosen_free = proxy._select_endpoint("gpt-test")
        self.assertIs(chosen_free, endpoint_b)
        # pinned=A → 无视统计，返回 A
        chosen_pinned = proxy._select_endpoint("gpt-test", pinned=endpoint_a)
        self.assertIs(chosen_pinned, endpoint_a)

    async def test_select_endpoint_pinned_returns_none_when_pin_unavailable(self):
        # pinned endpoint 被熔断（circuit OPEN）→ 返回 None，让调用方 fail-fast。
        # caa9b3e 后 circuit 状态由 monotonic 冷却窗口决定，需要显式把
        # circuit_retry_at 设到未来才是 OPEN；仅设 circuit_open=True 会被
        # is_available() 判为 HALF_OPEN 允许试探。
        proxy, load_balancer, endpoint_a = self._make_proxy(unittest.mock.Mock())
        endpoint_b = main.CopilotEndpoint(
            name="copilot-b", github_token="t2", models=["gpt-test"],
            session_token="s", session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(endpoint_b)
        # 把 A 从可用池摘掉：circuit OPEN + 冷却窗口未过
        endpoint_a.circuit_open = True
        endpoint_a.circuit_retry_at = main.time.monotonic() + 60
        chosen = proxy._select_endpoint("gpt-test", pinned=endpoint_a)
        self.assertIsNone(chosen)

    async def test_select_endpoint_pinned_bypasses_html_cooldown(self):
        # pinned endpoint 处于 HTML 软熔断窗口内 → 仍然返回它（宁可再踩一次挑战页也
        # 不换账户导致 401）。handoff §7.2 结论。
        proxy, load_balancer, endpoint_a = self._make_proxy(unittest.mock.Mock())
        endpoint_b = main.CopilotEndpoint(
            name="copilot-b", github_token="t2", models=["gpt-test"],
            session_token="s", session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(endpoint_b)
        endpoint_a.html_soft_cooldown_until = main.time.time() + 30
        # 无 pinned → 因为 A 在 cooldown，选 B
        chosen_free = proxy._select_endpoint("gpt-test")
        self.assertIs(chosen_free, endpoint_b)
        # pinned=A → 无视 cooldown，仍返回 A
        chosen_pinned = proxy._select_endpoint("gpt-test", pinned=endpoint_a)
        self.assertIs(chosen_pinned, endpoint_a)

    def test_apply_html_cooldown_tracks_status_bucket(self):
        # 200 / 4xx / 5xx / other 分桶正确
        proxy, load_balancer, endpoint = self._make_proxy(unittest.mock.Mock())
        proxy._apply_html_cooldown(endpoint, "responses", 200)
        proxy._apply_html_cooldown(endpoint, "responses", 403)
        proxy._apply_html_cooldown(endpoint, "responses", 502)
        proxy._apply_html_cooldown(endpoint, "responses", 502)
        proxy._apply_html_cooldown(endpoint, "responses", 999)
        self.assertEqual(proxy.upstream_html_events_by_status, {
            "200": 1, "4xx": 1, "5xx": 2, "other": 1,
        })

    def test_note_stateful_pin_accumulates_by_reason(self):
        proxy, _, _ = self._make_proxy(unittest.mock.Mock())
        proxy._note_stateful_pin("http_5xx")
        proxy._note_stateful_pin("http_5xx")
        proxy._note_stateful_pin("pool_timeout")
        self.assertEqual(proxy.stateful_pinned_events, {
            "http_5xx": 2, "pool_timeout": 1,
        })

    async def test_stream_upstream_html_includes_lb_request_id(self):
        # 502 HTML + request_id → error.upstream_ids["lb_request_id"] 存在
        html_body = b"<html><body>Bad Gateway</body></html>"
        upstream = _ErrorStreamResponse(
            502, html_body,
            headers={"content-type": "text/html", "cf-ray": "abc-999"},
        )
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
            request_id="lb-req-abcdef123456",
        )
        body = b"".join([chunk async for chunk in response.body_iterator])
        # 结构化字段
        self.assertIn(b'"lb_request_id"', body)
        self.assertIn(b"lb-req-abcdef123456", body)

    async def test_stream_5xx_never_replays_regardless_of_stateful(self):
        # caa9b3e 引入的 POST replay 禁令覆盖 pinning：无论 stateful 与否，5xx
        # 一律不重试（RESILIENCE.md 契约）——保护点从 pinning-fail-fast 变成
        # 更严格的"any 5xx = no replay"。这个 test 验证 stateful 5xx 到 SSE
        # 错误终止且**不换到 B**（B 上 active_requests 保持 0）。
        error_upstream = _ErrorStreamResponse(
            502, b'{"error":{"message":"upstream busted"}}',
            headers={"content-type": "application/json"},
        )
        proxy, load_balancer, endpoint_a = self._make_proxy(_StreamClient([error_upstream]))
        endpoint_b = main.CopilotEndpoint(
            name="copilot-b", github_token="t2", models=["gpt-test"],
            session_token="s", session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(endpoint_b)
        endpoint_a.active_requests = 0
        stateful_body = {
            "model": "gpt-test",
            "previous_response_id": "resp_prev_abc",
            "input": [{"content": [{"type": "input_text", "text": "hi"}]}],
        }
        await load_balancer.on_request_start(endpoint_a)
        response = await proxy._stream_response(
            endpoint_a,
            "https://example.test/responses",
            stateful_body,
            {},
            "gpt-test",
            "responses",
            main.time.time(),
        )
        body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertIn(b"upstream busted", body)
        # 没换到 B —— POST replay 禁令即已挡下
        self.assertEqual(endpoint_b.active_requests, 0)

    async def test_stream_stateless_5xx_also_never_replays(self):
        # 无状态请求，A 返 502。caa9b3e 之后无论 stateful 与否 5xx 都不 replay。
        # 保证 pinning 计数没触发（对照 test_stream_5xx_never_replays_regardless_of_stateful）。
        error_upstream = _ErrorStreamResponse(
            502, b'{"error":{"message":"upstream busted"}}',
            headers={"content-type": "application/json"},
        )
        proxy, load_balancer, endpoint_a = self._make_proxy(
            _StreamClient([error_upstream]), threshold=10,
        )
        endpoint_b = main.CopilotEndpoint(
            name="copilot-b", github_token="t2", models=["gpt-test"],
            session_token="s", session_token_expires_at=2**31,
        )
        load_balancer.endpoints.append(endpoint_b)
        await load_balancer.on_request_start(endpoint_a)
        response = await proxy._stream_response(
            endpoint_a,
            "https://example.test/chat/completions",
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            {},
            "gpt-test",
            "chat",
            main.time.time(),
        )
        body = b"".join([chunk async for chunk in response.body_iterator])
        # SSE error terminal 会补 data: [DONE]（chat 侧协议要求），
        # 但那不是失败切换后的成功 [DONE]。
        self.assertIn(b"upstream busted", body)
        self.assertEqual(endpoint_b.active_requests, 0)
        self.assertNotIn("http_5xx", proxy.stateful_pinned_events)

    async def test_monitor_releases_confirmed_disconnect_only_after_sustained_high_water(self):
        proxy, load_balancer, endpoint = self._make_proxy(unittest.mock.Mock())
        await load_balancer.on_request_start(endpoint)
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
            with self.assertRaises(HTTPException) as caught:
                response = await proxy.proxy_responses(
                    {"model": "gpt-test", "input": "hello"}, stream=False
                )

        self.assertEqual(caught.exception.status_code, 502)
        self.assertEqual(client.post.await_count, 1)  # Never replay a successful/ambiguous POST.
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

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
        self.assertEqual(endpoint.usage_record_errors, 1)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertEqual(client.post.await_count, 1)  # Never replay a successful/ambiguous POST.
        self.assertEqual(endpoint.active_requests, 1)
        self.assertEqual(load_balancer.on_request_start.await_count, 1)
        self.assertEqual(load_balancer.on_request_end.await_count, 1)

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


class AnthropicRequestIdSurfaceTests(unittest.IsolatedAsyncioTestCase):
    """P1.4: ClaudeProxy._stream_request 现在带 request_id，SSE error 与
    _build_upstream_error_detail 都能把 X-Request-Id 透传给下游。"""

    def test_proxy_request_signature_accepts_request_id(self):
        import inspect
        sig = inspect.signature(main.ClaudeProxy.proxy_request)
        self.assertIn("request_id", sig.parameters)

    def test_stream_request_signature_accepts_request_id(self):
        import inspect
        sig = inspect.signature(main.ClaudeProxy._stream_request)
        self.assertIn("request_id", sig.parameters)

    def test_normal_request_signature_accepts_request_id(self):
        import inspect
        sig = inspect.signature(main.ClaudeProxy._normal_request)
        self.assertIn("request_id", sig.parameters)

    def test_build_upstream_error_detail_lb_request_id_lands_in_output(self):
        # 生产分支已经在 error.upstream_ids 里加了 lb_request_id；P1.4 只是让
        # Anthropic path 也把参数传下来。这里确认 upstream_ids 结构不变。
        detail = main._build_upstream_error_detail(
            502, "some html body", "Databricks", "adb-1",
            content_type="text/html", upstream_headers={},
            lb_request_id="req_anth123",
        )
        self.assertEqual(detail["error"].get("upstream_ids", {}).get("lb_request_id"), "req_anth123")
