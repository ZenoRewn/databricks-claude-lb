"""Response-owned close boundaries, using production HTTPX/httpcore and real TCP.

Private state locks are test-only scheduling controls, never repaired/mutated.
Every pool assertion precedes client teardown; recovery must complete a real GET.
"""
import asyncio
from contextlib import asynccontextmanager
import unittest
from unittest.mock import patch

import anyio
import httpx
from httpcore._async.http11 import AsyncHTTP11Connection
from httpx._client import BoundAsyncStream

import main
from test_stream_response_ownership import OwnershipTests


class TransportCloseTests(OwnershipTests):
    @asynccontextmanager
    async def count_close(self, upstream):
        calls = []
        original = BoundAsyncStream.aclose

        async def counted(stream):
            if stream._response is upstream:
                calls.append(asyncio.current_task())
            await original(stream)

        with patch.object(BoundAsyncStream, 'aclose', counted):
            yield calls

    async def entered_close(self, upstream):
        async with asyncio.timeout(2):
            while not upstream.is_closed:
                await asyncio.sleep(.001)

    async def recovered(self, proxy, endpoint, peer):
        pool = proxy.client._transport._pool
        self.assertEqual(len(pool._requests), 0, 'transport close must finish before release')
        self.assertTrue(all(c.is_idle() or c.is_closed() for c in pool.connections))
        self.assertEqual(endpoint.active_requests, 0)
        self.assertEqual(proxy.load_balancer.on_request_end.await_count, 1)
        self.assertEqual(endpoint.total_errors, 0)
        self.assertEqual(getattr(proxy, '_stream_connections', {}), {})
        # A different origin forces eviction of the now-idle old connection.
        recovery = asyncio.create_task(proxy.client.get(await self.origin()))
        next_peer = await self.wait(self.peers.get())
        next_peer.allow_headers.set()
        await self.wait(next_peer.headers_sent.wait())
        await next_peer.chunk(b'ok')
        await next_peer.finish()
        result = await self.wait(recovery)
        self.assertEqual(result.content, b'ok')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(pool._requests), 0)
        self.assertTrue(all(c.is_idle() or c.is_closed() for c in pool.connections))
        await self.wait(peer.closed.wait())
        # Buffered TrackingClient.send queues its response only after aread().
        self.assertIs(await self.wait(proxy.client.completed_headers.get()), result)

    async def test_eof_close_repeated_cancel_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint)
                await self.wait(anext(response.body_iterator))
                frame = response.body_iterator._iterator.ag_frame
                upstream, pump = frame.f_locals['response'], frame.f_locals['pump_task']
                connection = proxy.client._transport._pool.connections[0]._connection
                async with self.count_close(upstream) as calls:
                    async with connection._state_lock:
                        await peer.finish()
                        await self.entered_close(upstream)
                        closing = asyncio.create_task(response.body_iterator.aclose())
                        await asyncio.sleep(.01)
                        # Cancel the waiter repeatedly AND the pump which is waiting
                        # for implicit EOF close. Never cancel a private close worker.
                        pump.cancel()
                        closing.cancel()
                        await asyncio.sleep(.01)
                        pump.cancel()
                        closing.cancel()
                        await asyncio.sleep(.01)
                        premature = closing.done() or pump.done()
                    # Lock release is independent of finalizer completion.
                    with self.assertRaises(asyncio.CancelledError):
                        await self.wait(closing)
                    self.assertTrue(pump.cancelled())
                    self.assertFalse(premature, 'first close must remain owned until joined')
                    await response.body_iterator.aclose()
                    self.assertEqual(len(calls), 1)
                await self.recovered(proxy, endpoint, peer)

    async def test_error_aread_auto_close_repeated_cancel_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint, headers=False)
                peer.status = b'400 Bad Request'
                peer.allow_headers.set()
                upstream = await self.wait(proxy.client.completed_headers.get())
                connection = proxy.client._transport._pool.connections[0]._connection
                async with self.count_close(upstream) as calls:
                    async with connection._state_lock:
                        reading = asyncio.create_task(anext(response.body_iterator))
                        await peer.chunk(b'{}')
                        await peer.finish()
                        await self.entered_close(upstream)
                        reading.cancel()
                        await asyncio.sleep(.01)
                        reading.cancel()
                        await asyncio.sleep(.01)
                        premature = reading.done()
                    with self.assertRaises(asyncio.CancelledError):
                        await self.wait(reading)
                    self.assertFalse(premature, 'aread EOF close must finish before cancellation escapes')
                    await response.body_iterator.aclose()
                    self.assertEqual(len(calls), 1)
                await self.recovered(proxy, endpoint, peer)

    async def test_eof_uncontended_checkpoint_cancels_original_pump_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint)
                await self.wait(anext(response.body_iterator))
                frame = response.body_iterator._iterator.ag_frame
                pump, upstream = frame.f_locals['pump_task'], frame.f_locals['response']
                original = AsyncHTTP11Connection._response_closed

                async def cancel_pump_at_checkpoint(connection):
                    # Same cancellation source as the finalizer, even when the
                    # protected transport close runs in a response-owned task.
                    asyncio.get_running_loop().call_soon(pump.cancel)
                    return await original(connection)

                with patch.object(AsyncHTTP11Connection, '_response_closed', cancel_pump_at_checkpoint):
                    async with self.count_close(upstream) as calls:
                        await peer.finish()
                        with self.assertRaises(asyncio.CancelledError):
                            await self.wait(pump)
                        await self.wait(response.body_iterator.aclose())
                        self.assertEqual(len(calls), 1)
                await self.recovered(proxy, endpoint, peer)

    async def test_internal_read_close_survives_finalizer_cancel_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            for error_body in (False, True):
                with self.subTest(provider=provider, error_body=error_body):
                    proxy, endpoint = await self.setup_proxy(provider)
                    response, peer = await self.paused(proxy, endpoint, headers=False)
                    if error_body:
                        peer.status = b'400 Bad Request'
                    peer.allow_headers.set()
                    await self.wait(proxy.client.completed_headers.get())
                    reading = None
                    if error_body:
                        reading = asyncio.create_task(anext(response.body_iterator))
                    else:
                        await self.wait(anext(response.body_iterator))
                    connection = proxy.client._transport._pool.connections[0]._connection
                    entered = asyncio.Event()
                    original = AsyncHTTP11Connection._response_closed
                    calls = []

                    async def internal_close(conn):
                        if conn is connection:
                            calls.append(asyncio.current_task())
                            entered.set()
                        return await original(conn)

                    with patch.object(AsyncHTTP11Connection, '_response_closed', internal_close):
                        async with connection._state_lock:
                            # A truncated real response triggers httpcore's internal
                            # error close BEFORE the public response close adapter.
                            peer.writer.close()
                            await self.wait(entered.wait())
                            if error_body:
                                closing = reading
                                closing.cancel()
                            else:
                                closing = asyncio.create_task(response.body_iterator.aclose())
                            await asyncio.sleep(.01)
                            closing.cancel()
                            await asyncio.sleep(.01)
                            premature = closing.done()
                        with self.assertRaises(asyncio.CancelledError):
                            await self.wait(closing)
                    self.assertFalse(premature)
                    self.assertEqual(len(calls), 1)
                    await response.body_iterator.aclose()
                    await self.recovered(proxy, endpoint, peer)

    async def test_silent_read_cancel_joins_internal_close_all_providers(self):
        for provider in ('copilot', 'azure', 'databricks'):
            with self.subTest(provider=provider):
                proxy, endpoint = await self.setup_proxy(provider)
                response, peer = await self.paused(proxy, endpoint)
                await self.wait(anext(response.body_iterator))
                frame = response.body_iterator._iterator.ag_frame
                pump = frame.f_locals['pump_task']
                connection = proxy.client._transport._pool.connections[0]._connection
                entered = asyncio.Event()
                original = AsyncHTTP11Connection._response_closed

                async def internal_close(conn):
                    entered.set()
                    return await original(conn)

                with patch.object(AsyncHTTP11Connection, '_response_closed', internal_close):
                    async with connection._state_lock:
                        closing = asyncio.create_task(response.body_iterator.aclose())
                        await self.wait(entered.wait())
                        pump.cancel()
                        closing.cancel()
                        await asyncio.sleep(.01)
                        pump.cancel()
                        closing.cancel()
                        await asyncio.sleep(.01)
                        premature = pump.done() or closing.done()
                    with self.assertRaises(asyncio.CancelledError):
                        await self.wait(closing)
                self.assertFalse(premature)
                self.assertTrue(pump.cancelled())
                await self.recovered(proxy, endpoint, peer)

    async def test_close_exception_is_retained_not_retried(self):
        class BrokenClose(httpx.AsyncByteStream):
            calls = 0
            async def __aiter__(self):
                yield b'ok'
            async def aclose(self):
                self.calls += 1
                raise RuntimeError('synthetic close failure')

        stream = BrokenClose()
        upstream = httpx.Response(200, stream=stream)
        async def send():
            return upstream
        helper = main._await_with_heartbeat(send(), b':', .01)
        await anext(helper)
        await helper.aclose()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'synthetic close failure'):
                await main._close_owned_response(upstream)
        self.assertEqual(stream.calls, 1)

    async def test_send_finish_cancel_race_close_is_owned_once(self):
        proxy, endpoint = await self.setup_proxy()
        entered, finish = asyncio.Event(), asyncio.Event()

        async def racing_send():
            upstream = await proxy.client.send(proxy.client.build_request('GET', self.url), stream=True)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return upstream

        helper = main._await_with_heartbeat(racing_send(), b':', .01)
        await self.wait(anext(helper))
        peer = await self.wait(self.peers.get())
        peer.allow_headers.set()
        upstream = await self.wait(proxy.client.completed_headers.get())
        original = BoundAsyncStream.aclose
        calls = []

        async def slow_close(stream):
            if stream._response is upstream:
                calls.append(asyncio.current_task())
                entered.set()
                await finish.wait()
            await original(stream)

        with patch.object(BoundAsyncStream, 'aclose', slow_close):
            closing = asyncio.create_task(helper.aclose())
            await self.wait(entered.wait())
            closing.cancel()
            await asyncio.sleep(.01)
            closing.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(closing.done())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await self.wait(closing)
            # The returned response must already have an owned close adapter,
            # including when send handled cancellation while returning headers.
            await upstream.stream.aclose()
            self.assertEqual(len(calls), 1)
        self.assertEqual(len(proxy.client._transport._pool._requests), 0)
        await self.wait(peer.closed.wait())

    async def test_adapter_concurrent_close_join_and_cancelled_scope(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        class SlowStream(httpx.AsyncByteStream):
            calls = 0
            async def __aiter__(self):
                yield b'ok'
            async def aclose(self):
                self.calls += 1
                entered.set()
                await finish.wait()

        stream = SlowStream()
        upstream = httpx.Response(200, stream=stream)
        async def send():
            return upstream
        helper = main._await_with_heartbeat(send(), b':', .01)
        self.assertEqual(await anext(helper), ('result', upstream))
        await helper.aclose()
        first = asyncio.create_task(upstream.aclose())
        await self.wait(entered.wait())
        second = asyncio.create_task(main._close_owned_response(upstream))
        await asyncio.sleep(.01)  # second must reach the early-is_closed join path
        first.cancel()
        second.cancel()
        await asyncio.sleep(.01)
        first.cancel()
        await asyncio.sleep(.01)
        premature = first.done()
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await self.wait(first)
        with self.assertRaises(asyncio.CancelledError):
            await self.wait(second)
        self.assertFalse(premature)
        with anyio.CancelScope() as scope:
            scope.cancel()
            await upstream.stream.aclose()
        self.assertEqual(stream.calls, 1)


def load_tests(loader, tests, pattern):
    # Inherit fixtures only; the original 20 methods run in their own module.
    return unittest.TestSuite(TransportCloseTests(name) for name in TransportCloseTests.__dict__
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
