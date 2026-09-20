"""Hostile upstream errors must not consume unbounded time/memory or replay POST."""
import asyncio
import gzip
import json
import unittest
from functools import partial
from unittest.mock import patch

import httpx

from upstream_body import read_error_body, error_body_hook


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks=(), stall=False):
        self.chunks, self.stall, self.closed = chunks, stall, False
    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.stall:
            await asyncio.Event().wait()
    async def aclose(self):
        self.closed = True


class ErrorBodyTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_small_error_and_retry_after(self):
        body=Body([b'{"error_code":"TEMPORARILY_UNAVAILABLE"}'])
        r=httpx.Response(503,stream=body,headers={'Retry-After':'120'})
        result=await read_error_body(r,max_bytes=128,timeout=.1)
        self.assertEqual(json.loads(result)['error_code'],'TEMPORARILY_UNAVAILABLE')
        self.assertEqual(r.status_code,503)
        self.assertEqual(r.headers['Retry-After'],'120')
        self.assertTrue(body.closed)
        self.assertEqual(await r.aread(),result)

    async def test_never_ending_body_has_bounded_wait(self):
        body=Body([b'{'],stall=True)
        r=httpx.Response(503,stream=body)
        result=await read_error_body(r,max_bytes=128,timeout=.01)
        self.assertEqual(json.loads(result)['error']['code'],'upstream_error_body_timeout')
        self.assertTrue(body.closed)

    async def test_oversized_body_does_not_expose_partial_provider_error(self):
        body=Body([b'{"message":"input item does not belong to this connection",',b'x'*1000])
        result=await read_error_body(httpx.Response(401,stream=body),max_bytes=128,timeout=.1)
        self.assertNotIn(b'input item',result)
        self.assertEqual(json.loads(result)['error']['code'],'upstream_error_body_too_large')
        self.assertTrue(body.closed)

    async def test_gzip_bomb_and_concatenated_members_are_bounded(self):
        for raw in (gzip.compress(b'x'*4_000_000),gzip.compress(b'a'*80)+gzip.compress(b'b'*80)):
            with self.subTest(size=len(raw)):
                body=Body([raw])
                result=await read_error_body(httpx.Response(503,stream=body,headers={'content-encoding':'gzip'}),max_bytes=128,timeout=.1)
                self.assertEqual(json.loads(result)['error']['code'],'upstream_error_body_too_large')
                self.assertTrue(body.closed)

    async def test_httpx_hook_bounds_non_streaming_post_before_automatic_aread(self):
        body=Body([b'x'],stall=True)
        calls=0
        async def upstream(request):
            nonlocal calls
            calls+=1
            return httpx.Response(503,stream=body)
        async def hook(response):
            await error_body_hook(response,max_bytes=128,timeout=.01)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream),event_hooks={'response':[hook]}) as client:
            r=await client.post('https://fixture.invalid/messages',json={})
        self.assertEqual(r.status_code,503)
        self.assertEqual(r.json()['error']['code'],'upstream_error_body_timeout')
        self.assertTrue(body.closed)
        self.assertEqual(calls,1)

    async def test_successful_stream_is_not_consumed_by_error_hook(self):
        body=Body(stall=True)
        r=httpx.Response(200,stream=body,headers={'content-type':'text/event-stream'})
        await error_body_hook(r,max_bytes=128,timeout=.01)
        self.assertFalse(r.is_stream_consumed)
        self.assertFalse(body.closed)
        await r.aclose()

    async def test_html_200_limit_cannot_turn_into_successful_json(self):
        r=httpx.Response(200,stream=Body([b'x'*1000]),headers={'content-type':'text/html'})
        await error_body_hook(r,max_bytes=128,timeout=.1)
        with self.assertRaises(ValueError):
            r.json()

    async def test_cancelled_reader_closes_response_and_preserves_cancellation(self):
        body=Body(stall=True)
        r=httpx.Response(503,stream=body)
        task=asyncio.create_task(read_error_body(r,max_bytes=128,timeout=10))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(body.closed)

    async def test_real_socket_non_stream_error_releases_connection_without_replay(self):
        import main
        disconnected=asyncio.Event()
        calls=0
        async def peer(reader,writer):
            nonlocal calls
            try:
                head=await reader.readuntil(b'\r\n\r\n')
                size=next((int(line.split(b':',1)[1]) for line in head.split(b'\r\n') if line.lower().startswith(b'content-length:')),0)
                if size:
                    await reader.readexactly(size)
                calls+=1
                writer.write(b'HTTP/1.1 503 Service Unavailable\r\nRetry-After: 30\r\nTransfer-Encoding: chunked\r\n\r\n1\r\n{\r\n')
                await writer.drain()
                await reader.read()
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()
        server=await asyncio.start_server(peer,'127.0.0.1',0)
        port=server.sockets[0].getsockname()[1]
        try:
            with patch.object(main,'read_error_body',partial(read_error_body,max_bytes=128,timeout=.05)):
                async with httpx.AsyncClient(trust_env=False,event_hooks={'response':[main._guard_upstream_error_response]},timeout=1) as client:
                    r=await client.post(f'http://127.0.0.1:{port}',json={})
                await asyncio.wait_for(disconnected.wait(),1)
                self.assertEqual(r.status_code,503)
                self.assertEqual(r.headers['Retry-After'],'30')
                self.assertEqual(r.json()['error']['code'],'upstream_error_body_timeout')
                self.assertEqual(calls,1)
        finally:
            server.close()
            await server.wait_closed()

    async def test_repeated_cancellation_keeps_the_existing_cleanup_owner(self):
        import main
        reading,closing,release=asyncio.Event(),asyncio.Event(),asyncio.Event()
        class SlowClose(httpx.AsyncByteStream):
            close_count=0
            async def __aiter__(self):
                reading.set()
                await asyncio.Event().wait()
                yield b''
            async def aclose(self):
                self.close_count+=1
                closing.set()
                await release.wait()
        stream=SlowClose()
        response=httpx.Response(503,stream=stream)
        task=asyncio.create_task(main._bounded_error_body(response))
        await reading.wait()
        task.cancel()
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(stream.close_count,1)
