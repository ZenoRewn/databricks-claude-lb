"""Real HTTP/1.1 lifecycle tests: no credentials, external upstream or app lifespan.

Events control header completion, not GC or timing guesses. Private httpcore
inspection is confined to assertions; server-observed EOF verifies socket close.
"""
import asyncio
from contextlib import suppress
import time
import unittest
from unittest.mock import AsyncMock, patch

import anyio
import httpx
from starlette.requests import ClientDisconnect

import main


class Peer:
    def __init__(self, writer):
        self.writer = writer
        self.allow_headers = asyncio.Event()
        self.closed = asyncio.Event()

    async def chunk(self, data):
        self.writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        await self.writer.drain()

    async def finish(self):
        self.writer.write(b"0\r\n\r\n")
        await self.writer.drain()


class TrackingClient(httpx.AsyncClient):
    def __init__(self):
        super().__init__(trust_env=False, timeout=httpx.Timeout(
            connect=1, read=None, write=1, pool=.2),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
        self.completed_headers = asyncio.Queue()

    async def send(self, *args, **kwargs):
        response = await super().send(*args, **kwargs)
        self.completed_headers.put_nowait(response)
        return response


class OwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.peers = asyncio.Queue()
        self.handlers = set()
        self.servers = []
        self.clients = []
        original = main._await_with_heartbeat
        # Change only the default, preserving the actual iterator and ownership.
        self.defaults = original.__defaults__
        original.__defaults__ = (.01,)
        self.addCleanup(setattr, original, '__defaults__', self.defaults)
        self.heartbeat_patch = patch.object(main, 'STREAM_HEARTBEAT_INTERVAL', .01)
        self.heartbeat_patch.start()
        self.addCleanup(self.heartbeat_patch.stop)
        self.url = await self.origin()

    async def origin(self):
        async def handler(reader, writer):
            task = asyncio.current_task()
            self.handlers.add(task)
            peer = Peer(writer)
            try:
                raw = await reader.readuntil(b'\r\n\r\n')
                length = next((int(line.split(b':', 1)[1]) for line in raw.split(b'\r\n')
                               if line.lower().startswith(b'content-length:')), 0)
                if length:
                    await reader.readexactly(length)
                self.peers.put_nowait(peer)
                # Detect EOF even before headers, for the pre-header cancel case.
                header_wait = asyncio.create_task(peer.allow_headers.wait())
                eof_wait = asyncio.create_task(reader.read())
                try:
                    done, _ = await asyncio.wait({header_wait, eof_wait},
                                                return_when=asyncio.FIRST_COMPLETED)
                    if eof_wait not in done:
                        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n'
                                     b'Transfer-Encoding: chunked\r\n\r\n')
                        await writer.drain()
                        await eof_wait
                finally:
                    for child in (header_wait, eof_wait):
                        child.cancel()
                    await asyncio.gather(header_wait, eof_wait, return_exceptions=True)
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()
                peer.closed.set()
                self.handlers.discard(task)
        server = await asyncio.start_server(handler, '127.0.0.1', 0)
        self.servers.append(server)
        return f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/responses'

    async def asyncTearDown(self):
        for client in self.clients:
            await client.aclose()
        for server in self.servers:
            server.close()
            await server.wait_closed()
        tasks = list(self.handlers)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def wait(self, awaitable):
        return await asyncio.wait_for(awaitable, 2)

    async def setup_proxy(self, provider='copilot'):
        if provider == 'copilot':
            endpoint = main.CopilotEndpoint(name='loopback', github_token='')
            cls = main.CopilotProxy
        elif provider == 'azure':
            endpoint = main.AzureOpenAIEndpoint(name='loopback', endpoint=self.url, api_key='')
            cls = main.AzureOpenAIProxy
        else:
            endpoint = main.WorkspaceEndpoint(name='loopback', api_base=self.url, token='')
            cls = main.ClaudeProxy
        proxy = cls(main.LoadBalancer([endpoint]), '')
        await proxy.client.aclose()
        proxy.client = TrackingClient()
        self.clients.append(proxy.client)
        proxy._record_usage = lambda *a, **kw: None
        proxy._probe_upstream_connect = AsyncMock(return_value={'ok': True})
        proxy.load_balancer.on_request_end = AsyncMock(wraps=proxy.load_balancer.on_request_end)
        return proxy, endpoint

    async def response(self, proxy, endpoint, url=None):
        await proxy.load_balancer.on_request_start(endpoint)
        if isinstance(proxy, main.ClaudeProxy):
            return await proxy._stream_request(endpoint, url or self.url, {}, {},
                                               model='synthetic', start_time=time.time())
        return await proxy._stream_response(endpoint, url or self.url, {}, {},
                                            'synthetic', 'responses', time.time())

    async def paused(self, proxy, endpoint, url=None, headers=True):
        response = await self.response(proxy, endpoint, url)
        self.assertTrue((await self.wait(anext(response.body_iterator))).startswith(b':'))
        peer = await self.wait(self.peers.get())
        if headers:
            peer.allow_headers.set()
            upstream = await self.wait(proxy.client.completed_headers.get())
            self.assertFalse(upstream.is_closed)
        return response, peer

    async def released(self, proxy, endpoint, peer, ends=1):
        # Check before client teardown: client.aclose() must not mask a leak.
        pool = proxy.client._transport._pool
        self.assertEqual(len(pool._requests), 0)
        self.assertFalse(any(not c.is_idle() and not c.is_closed() for c in pool.connections))
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(proxy.load_balancer.on_request_end.await_count, ends)
        self.assertEqual(getattr(proxy, '_stream_connections', {}), {})
        self.assertEqual(endpoint.total_errors, 0)
        await self.wait(peer.closed.wait())

    async def test_headers_finish_while_heartbeat_paused_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint)
                await response.body_iterator.aclose()
                await self.released(proxy, endpoint, peer)

    async def test_preheader_close_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint, headers=False)
                await response.body_iterator.aclose()
                await self.released(proxy, endpoint, peer)

    async def test_body_close_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint)
                await peer.chunk(b'data: {"type":"response.output_text.delta"}\n\n')
                while (await self.wait(anext(response.body_iterator))).startswith(b':'):
                    pass
                await response.body_iterator.aclose()
                await self.released(proxy, endpoint, peer)

    async def test_repeated_disconnects_multiple_origins_then_success(self):
        proxy, endpoint = await self.setup_proxy()
        urls = [self.url, await self.origin()]
        for i in range(12):
            response, peer = await self.paused(proxy, endpoint, urls[i % 2])
            await response.body_iterator.aclose()
            await self.released(proxy, endpoint, peer, ends=i + 1)
        await self.long_stream(proxy, endpoint, urls[1], ends=13)

    async def long_stream(self, proxy, endpoint, url=None, ends=1):
        response, peer = await self.paused(proxy, endpoint, url)
        # Headers transfer to caller; consuming silence must not close the result.
        for _ in range(5):
            self.assertTrue((await self.wait(anext(response.body_iterator))).startswith(b':'))
        self.assertFalse(peer.closed.is_set())
        self.assertEqual(len(proxy.client._transport._pool._requests), 1)
        data = b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
        terminal = b'data: {"type":"response.completed"}\n\n'
        await peer.chunk(data)
        await peer.chunk(terminal)
        await peer.finish()
        body = b''.join([chunk async for chunk in response.body_iterator])
        self.assertIn(data, body)
        self.assertIn(terminal, body)
        self.assertNotIn(b'response.failed', body)
        self.assertEqual(len(proxy.client._transport._pool._requests), 0)
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(proxy.load_balancer.on_request_end.await_count, ends)
        self.assertEqual(getattr(proxy, '_stream_connections', {}), {})
        # A fully consumed response may keep an idle reusable socket; not a leak.
        self.assertTrue(all(c.is_idle() or c.is_closed() for c in proxy.client._transport._pool.connections))

    async def test_normal_silent_stream_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                await self.long_stream(proxy, endpoint)

    async def asgi_disconnect(self, provider, spec, mode, body=False):
        proxy, endpoint = await self.setup_proxy(provider)
        response = await self.response(proxy, endpoint)
        paused = asyncio.Event()
        disconnect = asyncio.Event()
        holder = {}
        async def receive():
            await disconnect.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            if message['type'] != 'http.response.body':
                return
            if 'peer' not in holder:
                peer = await self.wait(self.peers.get())
                holder['peer'] = peer
                peer.allow_headers.set()
                await self.wait(proxy.client.completed_headers.get())
                if body:
                    await peer.chunk(b'data: {"type":"response.output_text.delta"}\n\n')
            if body and message.get('body', b'').startswith(b':'):
                return
            paused.set()
            if mode == 'send_error':
                raise OSError('synthetic downstream closed')
            if mode == 'disconnect':
                disconnect.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(response({'type': 'http', 'asgi': {'spec_version': spec}}, receive, send))
        await self.wait(paused.wait())
        if mode == 'cancel':
            task.cancel()
        try:
            await self.wait(task)
        except (ClientDisconnect, asyncio.CancelledError):
            if mode == 'disconnect':
                raise
        await self.released(proxy, endpoint, holder['peer'])

    async def test_asgi_send_failure_headers_and_body_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            for body in (False, True):
                with self.subTest(provider=provider, body=body):
                    await self.asgi_disconnect(provider, '2.4', 'send_error', body)

    async def test_asgi_disconnect_cancels_paused_send_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            for body in (False, True):
                with self.subTest(provider=provider, body=body):
                    await self.asgi_disconnect(provider, '2.3', 'disconnect', body)

    async def test_asgi_task_cancellation_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                await self.asgi_disconnect(provider, '2.4', 'cancel')

    async def test_finished_on_cancellation_response_is_reclaimed(self):
        client = TrackingClient()
        self.clients.append(client)
        async def cancellation_race():
            response = await client.send(client.build_request('GET', self.url), stream=True)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return response  # transport finishes at the cancellation boundary
        helper = main._await_with_heartbeat(cancellation_race(), b': heartbeat\n\n', .01)
        await self.wait(anext(helper))
        peer = await self.wait(self.peers.get())
        peer.allow_headers.set()
        response = await self.wait(client.completed_headers.get())
        await helper.aclose()
        self.assertTrue(response.is_closed)
        self.assertEqual(len(client._transport._pool._requests), 0)
        await self.wait(peer.closed.wait())

    async def test_transferred_response_not_closed_by_helper(self):
        client = TrackingClient()
        self.clients.append(client)
        helper = main._await_with_heartbeat(client.send(client.build_request('GET', self.url), stream=True), b':', .01)
        await self.wait(anext(helper))
        peer = await self.wait(self.peers.get())
        peer.allow_headers.set()
        response = await self.wait(client.completed_headers.get())
        self.assertEqual(await self.wait(anext(helper)), ('result', response))
        await helper.aclose()
        self.assertFalse(response.is_closed)
        await response.aclose()
        await self.wait(peer.closed.wait())

    async def test_actual_queued_count_excludes_assigned_response(self):
        proxy, endpoint = await self.setup_proxy()
        response, peer = await self.paused(proxy, endpoint)
        pool = proxy.client._transport._pool
        self.assertEqual(proxy._httpx_pool_stats()['requests_waiting'], 0)
        queued = asyncio.create_task(proxy.client.get(await self.origin()))
        for _ in range(100):
            if any(r.is_queued() for r in pool._requests):
                break
            await asyncio.sleep(.001)
        self.assertEqual(len(pool._requests), 2)
        self.assertEqual(proxy._httpx_pool_stats()['requests_waiting'], 1)
        queued.cancel()
        with suppress(asyncio.CancelledError):
            await queued
        await response.body_iterator.aclose()
        await self.released(proxy, endpoint, peer)

    async def test_repeated_cancellation_during_slow_close_is_joined(self):
        proxy, endpoint = await self.setup_proxy()
        response, peer = await self.paused(proxy, endpoint)
        # Delay closing the real unclaimed response while its socket is owned.
        pool = proxy.client._transport._pool
        entered = asyncio.Event()
        finish = asyncio.Event()
        original_close = httpx.Response.aclose
        async def slow_close(upstream):
            entered.set()
            await finish.wait()
            await original_close(upstream)
        with patch.object(httpx.Response, 'aclose', slow_close):
            task = asyncio.create_task(response.body_iterator.aclose())
            await self.wait(entered.wait())
            task.cancel()
            await asyncio.sleep(.01)
            task.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(task.done(), 'cleanup must stay owned until joined')
            self.assertEqual(len(pool._requests), 1)
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await self.wait(task)
        await self.released(proxy, endpoint, peer)

    async def test_repeated_cancel_during_body_unwind_joins_response_close(self):
        proxy, endpoint = await self.setup_proxy()
        response, peer = await self.paused(proxy, endpoint)
        await anext(response.body_iterator)  # transfer headers, wait in body pump
        entered = asyncio.Event()
        finish = asyncio.Event()
        closed = asyncio.Event()
        original_close = httpx.Response.aclose
        async def slow_close(upstream):
            entered.set()
            await finish.wait()
            await original_close(upstream)
            closed.set()
        with patch.object(httpx.Response, 'aclose', slow_close), patch.object(main, 'STREAM_HEARTBEAT_INTERVAL', 10):
            task = asyncio.create_task(anext(response.body_iterator))
            await asyncio.sleep(.01)
            task.cancel()
            await self.wait(entered.wait())
            task.cancel()
            await asyncio.sleep(.01)
            premature = task.done()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await self.wait(task)
        self.assertFalse(premature, 'response close was interrupted by repeated cancellation')
        self.assertTrue(closed.is_set())
        await self.released(proxy, endpoint, peer)

    async def test_close_inside_cancelled_anyio_scope(self):
        proxy, endpoint = await self.setup_proxy()
        response, peer = await self.paused(proxy, endpoint)
        with anyio.CancelScope() as scope:
            scope.cancel()
            await response.body_iterator.aclose()
        await self.released(proxy, endpoint, peer)

    async def test_cancel_awaiting_headers_before_first_heartbeat_all_providers(self):
        original = main._await_with_heartbeat
        original.__defaults__ = (10,)
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response = await self.response(proxy, endpoint)
                task = asyncio.create_task(anext(response.body_iterator))
                peer = await self.wait(self.peers.get())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await self.wait(task)
                await self.released(proxy, endpoint, peer)

    async def test_cancel_awaiting_body_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint)
                # Transfer headers and enter the body pump before cancelling.
                self.assertTrue((await self.wait(anext(response.body_iterator))).startswith(b':'))
                with patch.object(main, 'STREAM_HEARTBEAT_INTERVAL', 10):
                    task = asyncio.create_task(anext(response.body_iterator))
                    await asyncio.sleep(.01)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await self.wait(task)
                await self.released(proxy, endpoint, peer)

    async def test_upstream_timeout_error_is_not_a_heartbeat(self):
        async def fail():
            raise TimeoutError('synthetic upstream failure')
        helper = main._await_with_heartbeat(fail(), b':', .01)
        with self.assertRaisesRegex(TimeoutError, 'synthetic upstream failure'):
            await anext(helper)

    async def test_pool_timeout_neutral_despite_business_count_or_probe(self):
        proxy, endpoint = await self.setup_proxy()
        proxy.POOL_MAX_CONNECTIONS = 1
        response, peer = await self.paused(proxy, endpoint)
        # A real second-origin request cannot acquire the one occupied pool slot.
        with self.assertRaises(httpx.PoolTimeout) as caught:
            await proxy.client.get(await self.origin())
        _, message, fields = await proxy._describe_pool_timeout(endpoint.name, caught.exception)
        self.assertEqual(fields['classification'], 'pool_acquire_timeout')
        self.assertTrue(fields['httpx_pool_observed_full'])
        self.assertEqual(proxy.pool_timeout_saturated_total, 1)
        self.assertEqual(proxy.pool_timeout_upstream_stall_total, 0)
        self.assertNotIn('upstream connect stalled', message)
        self.assertNotIn('local pool still has slots', message)
        await response.body_iterator.aclose()
        await self.released(proxy, endpoint, peer)
        # Many business leases with no HTTPX connections cannot prove saturation.
        endpoint.active_requests = 500
        _, _, fields = await proxy._describe_pool_timeout(endpoint.name, caught.exception)
        self.assertEqual(fields['classification'], 'pool_acquire_timeout')
        self.assertFalse(fields['httpx_pool_observed_full'])
        self.assertEqual(proxy.pool_timeout_saturated_total, 1)
        endpoint.active_requests = 0

    async def test_disconnect_log_reports_offered_body_chunk(self):
        proxy, endpoint = await self.setup_proxy()
        response, peer = await self.paused(proxy, endpoint)
        await peer.chunk(b'data: {"type":"response.output_text.delta"}\n\n')
        while (await self.wait(anext(response.body_iterator))).startswith(b':'):
            pass
        with self.assertLogs('main', level='INFO') as logs:
            await response.body_iterator.aclose()
        record = next(r for r in logs.records if getattr(r, 'kind', '') == 'copilot_stream_end')
        self.assertEqual(record.chunks, 1)
        self.assertEqual(record.first_event, 'response.output_text.delta')
        self.assertTrue(record.sent_any_chunk)
        await self.released(proxy, endpoint, peer)

    async def test_header_heartbeats_do_not_fake_upstream_activity(self):
        proxy, endpoint = await self.setup_proxy()
        response, peer = await self.paused(proxy, endpoint, headers=False)
        before = proxy.get_stream_connection_stats()['max_upstream_idle_seconds']
        for _ in range(3):
            self.assertTrue((await self.wait(anext(response.body_iterator))).startswith(b':'))
        after = proxy.get_stream_connection_stats()['max_upstream_idle_seconds']
        self.assertGreater(after, before + .02)
        await response.body_iterator.aclose()
        await self.released(proxy, endpoint, peer)


if __name__ == '__main__':
    unittest.main()
