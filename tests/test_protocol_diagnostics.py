"""Safe protocol forensics; synthetic transport only. Author: Zeno Ren."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpcore
import httpx
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import DataReceived, StreamEnded
from h2.events import ConnectionTerminated, StreamReset

import main
import safe_diagnostics
import request_telemetry as telemetry


def reset_error(code=8):
    event = StreamReset(stream_id=3, error_code=code, remote_reset=True)
    cause = httpcore.RemoteProtocolError(event)
    error = httpx.RemoteProtocolError(str(event))
    error.__cause__ = cause
    return error


def test_structured_reset_and_goaway_keep_only_numeric_protocol_fields():
    reset = safe_diagnostics.transport_details(reset_error(), http_version='HTTP/2')
    assert reset['protocol_error_kind'] == 'http2_stream_reset'
    assert reset['protocol_scope'] == 'stream'
    assert reset['http2_error_code'] == 8
    assert reset['http2_stream_id'] == 3
    event = ConnectionTerminated()
    event.error_code, event.last_stream_id, event.additional_data = 11, 9, b'PRIVATE_DEBUG_DATA'
    error = httpx.RemoteProtocolError('PRIVATE_EXCEPTION')
    error.__cause__ = httpcore.RemoteProtocolError(event)
    result = safe_diagnostics.transport_details(error, http_version='HTTP/2')
    assert result['protocol_error_kind'] == 'http2_goaway'
    assert result['protocol_scope'] == 'connection'
    assert result['http2_last_stream_id'] == 9
    rendered = str(safe_diagnostics.safe_fields(result))
    assert 'PRIVATE' not in rendered
    assert 'http2_goaway' in rendered


def test_arbitrary_exception_text_does_not_establish_stream_scope():
    for text in ('PRIVATE_TOKEN', '<StreamReset stream_id:3, error_code:8, remote_reset:True>'):
        result = safe_diagnostics.transport_details(httpx.RemoteProtocolError(text))
        assert result['protocol_scope'] == 'unknown'
        assert text not in str(result)
    error = httpx.RemoteProtocolError('PRIVATE')
    error.__cause__ = error
    assert safe_diagnostics.transport_details(error)['protocol_scope'] == 'unknown'


def test_log_allowlist_validates_protocol_enums_and_numbers():
    result = safe_diagnostics.safe_fields({'protocol_scope': 'PRIVATE', 'http_version': 'PRIVATE',
        'protocol_error_kind': 'PRIVATE', 'http2_error_code': 'PRIVATE', 'upstream_idle_seconds': 120.1})
    assert result['protocol_scope'] == 'unknown'
    assert result['http_version'] == 'unknown'
    assert result['protocol_error_kind'] == 'unknown'
    assert result['http2_error_code'] is None
    assert result['upstream_idle_seconds'] == 120.1


class ProtocolStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_failure_event_retains_typed_protocol_diagnostics(self):
        async def app(scope,receive,send):
            call=AsyncMock(side_effect=reset_error())
            try:await telemetry.inference_call(call(),'copilot','responses',model='synthetic-model')
            except httpx.RemoteProtocolError:pass
            await send({'type':'http.response.start','status':502,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        with self.assertLogs('main',level='INFO') as logs:
            await telemetry.RequestTelemetryMiddleware(app,telemetry.RequestTelemetry())(
                {'type':'http','method':'POST','path':'/v1/responses','headers':[]},AsyncMock(),AsyncMock())
        event=next(r for r in logs.records if getattr(r,'kind','')=='lb_upstream_error')
        self.assertEqual(event.protocol_error_kind,'http2_stream_reset')
        self.assertEqual(event.http2_error_code,8)

    async def test_loopback_http2_reset_and_goaway_survive_real_httpcore_mapping(self):
        for goaway in (False, True):
            with self.subTest(goaway=goaway):
                finished = asyncio.Event()
                async def serve(reader, writer):
                    h2 = H2Connection(config=H2Configuration(client_side=False))
                    h2.initiate_connection()
                    writer.write(h2.data_to_send())
                    try:
                        while data := await reader.read(65536):
                            for event in h2.receive_data(data):
                                if isinstance(event, DataReceived):
                                    h2.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                                if isinstance(event, StreamEnded):
                                    h2.send_headers(event.stream_id, [(':status','200'),('content-type','text/event-stream')])
                                    writer.write(h2.data_to_send());await writer.drain()
                                    await asyncio.sleep(.01)
                                    if goaway:
                                        h2.close_connection(error_code=11, additional_data=b'PRIVATE_DEBUG_DATA')
                                    else:
                                        h2.reset_stream(event.stream_id,error_code=8)
                            writer.write(h2.data_to_send());await writer.drain()
                    finally:
                        writer.close();await writer.wait_closed();finished.set()
                server=await asyncio.start_server(serve,'127.0.0.1',0)
                port=server.sockets[0].getsockname()[1]
                try:
                    transport=httpx.AsyncHTTPTransport(http1=False,http2=True)
                    async with httpx.AsyncClient(transport=transport,timeout=2,trust_env=False) as client:
                        async with client.stream('POST',f'http://127.0.0.1:{port}/responses',json={}) as response:
                            with self.assertRaises(httpx.RemoteProtocolError) as caught:
                                async for _ in response.aiter_bytes():pass
                            details=safe_diagnostics.transport_details(caught.exception,http_version=response.http_version)
                            self.assertEqual(details['http_version'],'HTTP/2')
                            self.assertEqual(details['protocol_error_kind'],'http2_goaway' if goaway else 'http2_stream_reset')
                            self.assertEqual(details['http2_error_code'],11 if goaway else 8)
                            self.assertNotIn('PRIVATE',str(details))
                    await asyncio.wait_for(finished.wait(),2)
                finally:
                    server.close();await server.wait_closed()

    async def test_real_httpx_stream_failure_records_version_and_idle_without_replay(self):
        class ResetStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"type":"response.created"}\n\n'
                await asyncio.sleep(.01)
                raise reset_error()
            async def aclose(self):
                pass

        ep = main.CopilotEndpoint('synthetic', '', models=['synthetic-model'])
        proxy = main.CopilotProxy(main.LoadBalancer([ep]), '')
        await proxy.client.aclose()
        calls = []
        def upstream(req):
            calls.append(req)
            return httpx.Response(200, stream=ResetStream(), extensions={'http_version': b'HTTP/2'})
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False)
        proxy._build_headers = AsyncMock(return_value={})
        proxy._probe_upstream_connect = AsyncMock(return_value={'ok': True})
        try:
            with self.assertLogs('main', level='INFO') as logs, patch.object(main, 'usage_store', None):
                response = await proxy.proxy_responses({'model':'synthetic-model'}, stream=True)
                wire = b''.join([part async for part in response.body_iterator])
            event = next(r for r in logs.records if getattr(r,'kind','') == 'copilot_stream_network_error')
            self.assertEqual(event.http_version, 'HTTP/2')
            self.assertEqual(event.protocol_error_kind, 'http2_stream_reset')
            self.assertGreaterEqual(event.upstream_idle_seconds, .005)
            self.assertTrue(event.upstream_headers_received)
            self.assertEqual(len(calls), 1)
            self.assertIn(b'response.failed', wire)
        finally:
            await proxy.close()
