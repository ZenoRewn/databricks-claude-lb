"""Token control-plane tests; only synthetic tokens and in-memory transports."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
from fastapi import HTTPException

import main


# Keep the real constructor when patching main.httpx.AsyncClient below.
AsyncClient = httpx.AsyncClient


class CopilotTokenExchangeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.endpoint = main.CopilotEndpoint(name="token-test", github_token="synthetic-oauth")
        self.shared = Mock()
        self.shared.aclose = AsyncMock()
        with patch.object(main.httpx, "AsyncClient", return_value=self.shared):
            self.proxy = main.CopilotProxy(
                main.LoadBalancer([self.endpoint], circuit_breaker_threshold=1), "synthetic-key"
            )
        self.clients = []
        self.client_options = []
        self.requests = []
        self.responses = []

    def success(self):
        return httpx.Response(200, json={
            "token": "synthetic-session",
            "expires_at": int(main.time.time()) + 1800,
            "endpoints": {"api": "https://copilot.example.test/"},
        })

    def client_factory(self, outcomes):
        outcomes = iter(outcomes)

        async def handle(request):
            self.requests.append(request)
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            self.responses.append(outcome)
            return outcome

        def factory(**kwargs):
            self.client_options.append(kwargs)
            client = AsyncClient(transport=httpx.MockTransport(handle), **kwargs)
            self.clients.append(client)
            return client

        return factory

    async def test_refresh_ignores_exhausted_or_broken_shared_pool(self):
        """Regression: shared GET raised an empty error and tripped the circuit."""
        for error_type in (httpx.PoolTimeout, httpx.ConnectError):
            with self.subTest(error_type=error_type.__name__):
                self.shared.get = AsyncMock(side_effect=error_type(""))
                self.shared.post = AsyncMock(return_value=httpx.Response(
                    200, json={"choices": [], "usage": {}}
                ))
                self.endpoint.session_token = None
                self.endpoint.circuit_open = False
                with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([self.success()])), \
                     patch.object(main.asyncio, "sleep", new=AsyncMock()):
                    response = await self.proxy.proxy_chat_completions(
                        {"model": "gpt-test", "messages": []}, stream=False
                    )
                self.assertEqual(response.status_code, 200)
                self.shared.get.assert_not_awaited()
                self.assertFalse(self.endpoint.circuit_open)
                self.assertEqual(self.endpoint.total_errors, 0)
                self.assertEqual(self.endpoint.active_requests, 0)
                self.assertEqual(self.endpoint.token_refresh_failed_total, 0)
                self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_transient_failure_retries_on_fresh_client(self):
        self.shared.get = AsyncMock(side_effect=httpx.PoolTimeout(""))
        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
            httpx.PoolTimeout(""), self.success(),
        ])), patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep:
            token = await self.proxy.get_session_token(self.endpoint)
        self.assertEqual(token, "synthetic-session")
        self.assertEqual(len(self.clients), 2)
        self.assertTrue(all(client.is_closed for client in self.clients))
        sleep.assert_awaited_once_with(0.2)
        self.assertEqual(self.endpoint.token_refresh_total, 1)
        self.assertEqual(self.endpoint.token_refresh_failed_total, 1)

    async def test_empty_network_error_log_includes_exception_type(self):
        error = RuntimeError("")
        self.shared.get = AsyncMock(side_effect=error)  # Baseline uses this path.
        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([error])), \
             self.assertLogs("main", level="ERROR") as logs:
            with self.assertRaises(RuntimeError):
                await self.proxy._exchange_token(self.endpoint)
        self.assertIn("RuntimeError", "\n".join(logs.output))

    async def test_retry_allowlist_and_exhaustion_are_bounded(self):
        for error_type in (
            httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
            httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError,
        ):
            with self.subTest(error_type=error_type.__name__):
                errors = [error_type("") for _ in range(3)]
                before = self.endpoint.token_refresh_failed_total
                with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory(errors)) as factory, \
                     patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep, \
                     patch.object(main, "reload_github_token") as reload_token, \
                     self.assertLogs("main", level="WARNING") as logs:
                    with self.assertRaises(error_type) as caught:
                        await self.proxy.get_session_token(self.endpoint)
                self.assertIs(caught.exception, errors[-1])
                self.assertEqual(factory.call_count, 3)
                self.assertEqual(sleep.await_args_list, [call(0.2), call(0.4)])
                self.assertEqual(self.endpoint.token_refresh_failed_total - before, 3)
                self.assertEqual(self.endpoint.token_refresh_total, 0)
                self.assertEqual(self.proxy.pool_timeout_total, 0)
                self.assertFalse(self.endpoint.circuit_open)
                reload_token.assert_not_called()
                self.assertTrue(all(client.is_closed for client in self.clients))
                self.assertEqual([r.attempt for r in logs.records], [1, 2, 3])
                self.assertEqual([r.retry for r in logs.records], [True, True, False])
                self.assertTrue(all(r.error_type == error_type.__name__ for r in logs.records))

    async def test_non_transient_exceptions_are_not_retried_or_logged_verbatim(self):
        for error_type in (httpx.LocalProtocolError, httpx.UnsupportedProtocol, ValueError, RuntimeError):
            with self.subTest(error_type=error_type.__name__):
                # Exception messages may contain request data; never echo them here.
                error = error_type("synthetic-sensitive-marker")
                with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([error])) as factory, \
                     patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep, \
                     self.assertLogs("main", level="ERROR") as logs:
                    with self.assertRaises(error_type) as caught:
                        await self.proxy._exchange_token(self.endpoint)
                self.assertIs(caught.exception, error)
                self.assertEqual(factory.call_count, 1)
                sleep.assert_not_awaited()
                self.assertTrue(self.clients[-1].is_closed)
                text = main._JsonLogFormatter().format(logs.records[-1])
                self.assertIn(error_type.__name__, text)
                self.assertIn("copilot_token_exchange_error", text)
                self.assertNotIn("synthetic-sensitive-marker", text)
                self.assertNotIn(self.endpoint.github_token, text)

    async def test_http_errors_including_401_are_not_network_retries(self):
        for status in (401, 403, 429, 500, 502, 503):
            with self.subTest(status=status):
                error_type = main._LongLivedTokenInvalidError if status == 401 else httpx.HTTPStatusError
                before = self.endpoint.token_refresh_failed_total
                with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
                    httpx.Response(status, json={"error": "synthetic"}),
                ])) as factory, patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep:
                    with self.assertRaises(error_type):
                        await self.proxy._exchange_token(self.endpoint)
                self.assertEqual(factory.call_count, 1)
                sleep.assert_not_awaited()
                self.assertEqual(self.endpoint.token_refresh_failed_total - before, 1)
                self.assertTrue(self.clients[-1].is_closed)
                self.assertTrue(self.responses[-1].is_closed)

    async def test_401_without_new_oauth_token_trips_circuit_once(self):
        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
            httpx.Response(401),
        ])) as factory, patch.object(main, "reload_github_token", return_value=False) as reload_token, \
             patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(HTTPException) as caught:
                await self.proxy.get_session_token(self.endpoint)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(factory.call_count, 1)
        reload_token.assert_called_once_with(self.endpoint)
        sleep.assert_not_awaited()
        self.assertTrue(self.endpoint.circuit_open)
        self.assertEqual(self.endpoint.total_errors, 1)
        self.assertEqual(self.endpoint.token_refresh_failed_total, 1)

    async def test_401_reloads_changed_source_once_and_recovers(self):
        self.endpoint.token_source = {"type": "env", "key": "TEST_SYNTHETIC_OAUTH"}
        self.endpoint.circuit_open = True
        self.endpoint.total_errors = 4
        with patch.dict(main.os.environ, {"TEST_SYNTHETIC_OAUTH": "synthetic-rotated"}), \
             patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
                 httpx.Response(401), self.success(),
             ])) as factory, patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep:
            token = await self.proxy.get_session_token(self.endpoint)
        self.assertEqual(token, "synthetic-session")
        self.assertEqual(factory.call_count, 2)  # Changed credential, not a 401 retry loop.
        self.assertEqual(self.requests[0].headers["Authorization"], "Bearer synthetic-oauth")
        self.assertEqual(self.requests[1].headers["Authorization"], "Bearer synthetic-rotated")
        sleep.assert_not_awaited()
        self.assertEqual(self.endpoint.token_reload_total, 1)
        self.assertEqual(self.endpoint.token_refresh_failed_total, 1)
        self.assertEqual(self.endpoint.token_refresh_total, 1)
        self.assertFalse(self.endpoint.circuit_open)
        self.assertEqual(self.endpoint.total_errors, 0)
        self.assertTrue(self.proxy.is_any_endpoint_healthy())

    async def test_reloaded_token_still_401_does_not_loop(self):
        self.endpoint.token_source = {"type": "env", "key": "TEST_SYNTHETIC_OAUTH"}
        with patch.dict(main.os.environ, {"TEST_SYNTHETIC_OAUTH": "synthetic-rotated"}), \
             patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
                 httpx.Response(401), httpx.Response(401),
             ])) as factory, patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(HTTPException) as caught:
                await self.proxy.get_session_token(self.endpoint)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(factory.call_count, 2)
        sleep.assert_not_awaited()
        self.assertEqual(self.endpoint.token_reload_total, 1)
        self.assertEqual(self.endpoint.token_refresh_failed_total, 2)
        self.assertTrue(self.endpoint.circuit_open)
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_cache_lock_expiry_force_and_connection_options(self):
        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
            self.success(), self.success(), self.success(),
        ])) as factory:
            tokens = await asyncio.gather(*[
                self.proxy.get_session_token(self.endpoint) for _ in range(10)
            ])
            self.assertEqual(tokens, ["synthetic-session"] * 10)
            self.assertEqual(factory.call_count, 1)
            await self.proxy.get_session_token(self.endpoint, force=True)
            self.assertEqual(factory.call_count, 2)
            self.endpoint.session_token_expires_at = int(main.time.time()) + 59
            await self.proxy.get_session_token(self.endpoint)
            self.assertEqual(factory.call_count, 3)
        self.assertEqual(self.endpoint.session_base_url, "https://copilot.example.test")
        self.assertGreater(self.endpoint.last_token_refresh_at, 0)
        self.assertEqual(self.endpoint.token_refresh_total, 3)
        for options in self.client_options:
            self.assertEqual(options["timeout"].as_dict(), dict(connect=10.0, read=10.0, write=10.0, pool=10.0))
            self.assertEqual(options["limits"].max_connections, 1)
            self.assertEqual(options["limits"].max_keepalive_connections, 0)
        for request in self.requests:
            self.assertEqual(str(request.url), main.COPILOT_TOKEN_URL)
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.headers["Accept"], "application/json")
            for header in ("User-Agent", "Editor-Version", "Editor-Plugin-Version"):
                self.assertEqual(request.headers[header], main.COPILOT_HEADERS[header])
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertTrue(all(response.is_closed for response in self.responses))
        self.shared.aclose.assert_not_awaited()
        await self.proxy.close()
        self.shared.aclose.assert_awaited_once()

    async def test_bad_json_closes_client_without_retry(self):
        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
            httpx.Response(200, content=b"not-json"),
        ])) as factory, patch.object(main.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(ValueError):
                await self.proxy._exchange_token(self.endpoint)
        self.assertEqual(factory.call_count, 1)
        sleep.assert_not_awaited()
        self.assertTrue(self.clients[0].is_closed)
        self.assertTrue(self.responses[0].is_closed)
        self.assertEqual(self.endpoint.token_refresh_total, 0)

    async def test_cancellation_closes_client_unlocks_and_does_not_trip_circuit(self):
        started = asyncio.Event()

        async def handle(_request):
            started.set()
            await asyncio.Event().wait()

        client = AsyncClient(transport=httpx.MockTransport(handle))
        with patch.object(main.httpx, "AsyncClient", return_value=client) as factory:
            task = asyncio.create_task(self.proxy.proxy_responses(
                {"model": "gpt-test", "input": "hello"}, stream=False
            ))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertTrue(client.is_closed)
        self.assertEqual(factory.call_count, 1)
        self.assertFalse(self.proxy._get_token_lock(self.endpoint).locked())
        self.assertFalse(self.endpoint.circuit_open)
        self.assertEqual(self.endpoint.active_requests, 0)
        self.assertEqual(self.endpoint.total_errors, 0)
        self.assertEqual(self.endpoint.token_refresh_failed_total, 0)
        self.shared.aclose.assert_not_awaited()

    async def test_cancellation_during_backoff_does_not_start_another_client(self):
        async def cancel_during_backoff(_delay):
            self.assertTrue(self.clients[-1].is_closed)
            raise asyncio.CancelledError()

        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
            httpx.ReadTimeout(""),
        ])) as factory, patch.object(main.asyncio, "sleep", side_effect=cancel_during_backoff):
            with self.assertRaises(asyncio.CancelledError):
                await self.proxy.get_session_token(self.endpoint)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(self.endpoint.token_refresh_failed_total, 1)
        self.assertFalse(self.proxy._get_token_lock(self.endpoint).locked())
        self.assertFalse(self.endpoint.circuit_open)

    async def test_token_metrics_remain_separate_from_stream_pool_metrics(self):
        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([
            httpx.ConnectError(""), self.success(),
        ])), patch.object(main.asyncio, "sleep", new=AsyncMock()):
            await self.proxy.get_session_token(self.endpoint)
        with patch.object(main, "copilot_proxy", self.proxy):
            response = await main.metrics()
        text = response.body.decode()
        for sample in (
            'copilot_token_refresh_total{endpoint="token-test"} 1',
            'copilot_token_refresh_failed_total{endpoint="token-test"} 1',
            'copilot_token_reload_total{endpoint="token-test"} 0',
            'copilot_endpoint_circuit_open{endpoint="token-test"} 0',
            'copilot_endpoint_active_requests{endpoint="token-test"} 0',
            'copilot_pool_timeout_total 0',
            'copilot_stream_read_timeout_total 0',
        ):
            self.assertIn(sample, text)

    async def test_admin_reload_forces_isolated_exchange(self):
        self.endpoint.session_token = "synthetic-cached"
        self.endpoint.session_token_expires_at = int(main.time.time()) + 1800
        self.endpoint.token_source = {"type": "env", "key": "TEST_SYNTHETIC_OAUTH"}
        request = main.Request({"type": "http", "headers": []})
        with patch.object(main, "copilot_proxy", self.proxy), \
             patch.dict(main.os.environ, {"TEST_SYNTHETIC_OAUTH": "synthetic-rotated"}), \
             patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([self.success()])):
            result = await main.admin_copilot_reload(request, x_api_key="synthetic-key")
        self.assertTrue(result["endpoints"][0]["reloaded"])
        self.assertTrue(result["endpoints"][0]["session_refreshed"])
        self.assertEqual(self.endpoint.token_reload_total, 1)
        self.assertTrue(self.clients[0].is_closed)
        self.shared.get.assert_not_called()

    async def test_background_refresh_uses_isolated_client_and_closes_on_cancel(self):
        refreshed = asyncio.Event()
        original_exchange = self.proxy._exchange_token

        async def exchange(endpoint):
            result = await original_exchange(endpoint)
            refreshed.set()
            return result

        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([self.success()])), \
             patch.object(self.proxy, "_exchange_token", side_effect=exchange):
            task = asyncio.create_task(self.proxy.background_refresh_loop(interval=0.001))
            try:
                await asyncio.wait_for(refreshed.wait(), timeout=1)
            finally:
                task.cancel()
                await task
        self.assertEqual(self.endpoint.token_refresh_total, 1)
        self.assertTrue(self.clients[0].is_closed)
        self.shared.get.assert_not_called()

    async def test_upstream_401_forced_refresh_works_for_buffered_and_streaming(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.endpoint.session_token = "synthetic-cached"
                self.endpoint.session_token_expires_at = int(main.time.time()) + 1800
                self.endpoint.session_base_url = "https://copilot.example.test"
                requests = []
                responses = []

                async def inference(request):
                    requests.append(request)
                    if len(requests) == 1:
                        response = httpx.Response(401, json={"error": {"message": "expired"}})
                    elif stream:
                        response = httpx.Response(200, content=(
                            b'data: {"type":"response.completed","response":{"usage":{}}}\n\n'
                        ))
                    else:
                        response = httpx.Response(200, json={"usage": {}})
                    responses.append(response)
                    return response

                async with AsyncClient(transport=httpx.MockTransport(inference)) as shared:
                    self.proxy.client = shared
                    with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([self.success()])) as factory:
                        response = await self.proxy.proxy_responses(
                            {"model": "gpt-test", "input": "hello"}, stream=stream
                        )
                        if stream:
                            body = b"".join([chunk async for chunk in response.body_iterator])
                            self.assertIn(b'response.completed', body)
                            self.assertNotIn(b'response.failed', body)
                        self.assertEqual(response.status_code, 200)
                    self.assertEqual(factory.call_count, 1)
                    self.assertFalse(shared.is_closed)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1].headers["Authorization"], "Bearer synthetic-session")
                self.assertTrue(all(response.is_closed for response in responses))
                self.assertTrue(all(client.is_closed for client in self.clients))
                self.assertEqual(self.endpoint.active_requests, 0)
                self.assertFalse(self.endpoint.circuit_open)

    async def test_real_saturated_httpx_pool_does_not_block_token_exchange(self):
        """One unfinished local stream fills an actual httpcore pool, no Internet."""
        release = asyncio.Event()
        handlers = set()
        token_body = self.success().content

        async def serve(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            try:
                request = await reader.readuntil(b"\r\n\r\n")
                if request.startswith(b"GET /stream "):
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nx")
                    await writer.drain()
                    await release.wait()
                else:
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        b"Connection: close\r\nContent-Length: " + str(len(token_body)).encode()
                        + b"\r\n\r\n" + token_body
                    )
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            async with AsyncClient(
                limits=httpx.Limits(max_connections=1),
                timeout=httpx.Timeout(1, pool=0.02), trust_env=False,
            ) as shared:
                self.proxy.client = shared
                held = await shared.send(shared.build_request("GET", base + "/stream"), stream=True)
                try:
                    self.assertFalse(held.is_closed)
                    self.assertEqual(self.endpoint.active_requests, 0)
                    with self.assertRaises(httpx.PoolTimeout):
                        await shared.get(base + "/token")
                    with patch.object(main, "COPILOT_TOKEN_URL", base + "/token"), \
                         patch.dict(main.os.environ, {}, clear=True):
                        token = await asyncio.wait_for(self.proxy.get_session_token(self.endpoint), timeout=2)
                    self.assertEqual(token, "synthetic-session")
                    self.assertFalse(held.is_closed)
                    with self.assertRaises(httpx.PoolTimeout):
                        await shared.get(base + "/token")
                    self.assertEqual(self.endpoint.token_refresh_failed_total, 0)
                    self.assertFalse(self.endpoint.circuit_open)
                finally:
                    await held.aclose()
        finally:
            release.set()
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.gather(*list(handlers))

    async def test_warmup_uses_isolated_token_get_and_shared_head(self):
        self.shared.get = AsyncMock(side_effect=httpx.PoolTimeout(""))
        self.shared.head = AsyncMock(return_value=httpx.Response(200))
        with patch.object(main.httpx, "AsyncClient", side_effect=self.client_factory([self.success()])):
            await self.proxy.warmup()
        self.assertEqual(self.endpoint.token_refresh_total, 1)
        self.shared.get.assert_not_awaited()
        self.shared.head.assert_awaited_once()
        self.assertTrue(self.clients[0].is_closed)


if __name__ == "__main__":
    unittest.main()
