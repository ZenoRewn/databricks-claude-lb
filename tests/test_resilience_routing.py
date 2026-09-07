import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch
import httpx
import main
import test_copilot_request_lifecycle as fixtures


class RetryRoutingTests(unittest.IsolatedAsyncioTestCase):
    def make(self, client=None):
        return fixtures.CopilotRequestLifecycleTests()._make_proxy(client or Mock(), 20)

    async def test_buffered_ambiguous_transport_matrix_is_never_replayed(self):
        for kind in (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.WriteError,
                     httpx.RemoteProtocolError, RuntimeError):
            with self.subTest(kind=kind.__name__):
                p, lb, ep = self.make()
                p.client.post = AsyncMock(side_effect=kind('synthetic'))
                with patch.object(main.asyncio, 'sleep', new=AsyncMock()) as sleep:
                    with self.assertRaises(main.HTTPException) as caught:
                        await p.proxy_responses({'model':'gpt-test', 'input':[]})
                self.assertEqual(caught.exception.status_code, 502)
                self.assertEqual(p.client.post.await_count, 1)
                sleep.assert_not_awaited()
                self.assertEqual(ep.active_requests, 0)
                self.assertEqual(ep.total_errors, 1)

    async def test_connect_failure_is_bounded_and_payload_identity_preserved(self):
        p, lb, ep = self.make()
        seen = []
        body = {'model':'gpt-test','input':[{'role':'user','content':'synthetic'}],
                'tools':[{'type':'function','name':'test_tool','parameters':{'type':'object'}}]}
        async def post(url, json, headers):
            seen.append((json.copy(), headers.copy()))
            raise httpx.ConnectError('synthetic')
        p.client.post = AsyncMock(side_effect=post)
        with patch.object(main.asyncio, 'sleep', new=AsyncMock()):
            with self.assertRaises(main.HTTPException):
                await p.proxy_responses(body)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(payload == body and headers == seen[0][1] for payload, headers in seen))
        self.assertEqual(ep.active_requests, 0)
        self.assertEqual(ep.total_errors, 3)

    async def test_5xx_response_not_replayed_or_cross_provider_fallback(self):
        for status in (500, 502, 503, 504):
            with self.subTest(status=status):
                p, lb, ep = self.make()
                req = httpx.Request('POST', 'https://test.invalid/responses')
                p.client.post = AsyncMock(return_value=httpx.Response(status, request=req, json={'error':{'message':'synthetic'}}))
                azure = Mock()
                azure.load_balancer.endpoints = [main.AzureOpenAIEndpoint('az','https://test.invalid','synthetic', deployments=['gpt-test'])]
                azure.proxy_responses = AsyncMock()
                with patch.object(main, 'copilot_proxy', p), patch.object(main, 'azure_proxy', azure), \
                     patch.object(main.asyncio, 'sleep', new=AsyncMock()) as sleep:
                    with self.assertRaises(main.HTTPException) as caught:
                        await main._route_openai({'model':'gpt-test'}, False, 'responses')
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(p.client.post.await_count, 1)
                azure.proxy_responses.assert_not_awaited()
                sleep.assert_not_awaited()

    async def test_retry_after_rejection_returned_not_shortened(self):
        for delay in ('120','Mon, 07 Sep 2026 09:00:00 GMT'):
            p, lb, ep = self.make()
            req = httpx.Request('POST','https://test.invalid/responses')
            p.client.post = AsyncMock(return_value=httpx.Response(429, request=req, headers={'Retry-After':delay},json={'error':{'message':'limited'}}))
            with patch.object(main.asyncio, 'sleep', new=AsyncMock()) as sleep:
                with self.assertRaises(main.HTTPException) as caught:
                    await p.proxy_responses({'model':'gpt-test'})
            self.assertEqual(caught.exception.status_code, 429)
            self.assertEqual(caught.exception.headers['Retry-After'], delay)
            self.assertEqual(p.client.post.await_count, 1)
            sleep.assert_not_awaited()

    async def test_half_open_real_buffered_request_resolves_trial(self):
        p, lb, ep = self.make()
        now = 100.
        with patch.object(main.time, 'monotonic', side_effect=lambda: now):
            lb._open(ep)
            now += lb.circuit_breaker_timeout
            req = httpx.Request('POST','https://test.invalid/responses')
            p.client.post = AsyncMock(return_value=httpx.Response(200,request=req,json={'usage':{}}))
            result = await p.proxy_responses({'model':'gpt-test'})
        self.assertEqual(result.status_code, 200)
        self.assertFalse(ep.circuit_open)
        self.assertEqual(ep.active_requests, 0)
        self.assertEqual(ep.completed_requests, 1)

    async def test_half_open_concurrent_stream_ownership_cancel_and_recovery(self):
        upstream = fixtures._StreamResponse([], block_after_chunks=True)
        client = fixtures._StreamClient([upstream])
        p, lb, ep = self.make(client)
        now = 100.
        with patch.object(main.time, 'monotonic', side_effect=lambda: now):
            lb._open(ep)
            now += lb.circuit_breaker_timeout
            response = await p.proxy_responses({'model':'gpt-test'}, stream=True)
            with self.assertRaises(main.HTTPException) as caught:
                await p.proxy_responses({'model':'gpt-test'}, stream=True)
            self.assertEqual(caught.exception.status_code, 503)
            await response.body_iterator.aclose()  # never-started stream still owns a trial
            self.assertEqual(ep.cancelled_requests, 1)
            self.assertEqual(ep.active_requests, 0)
            self.assertEqual(lb.circuit_state(ep), 'OPEN')
            now += lb.circuit_breaker_timeout
            p.client = fixtures._StreamClient([fixtures._StreamResponse([b'data: {"type":"response.completed","response":{"id":"synthetic"}}\n\n'])])
            response = await p.proxy_responses({'model':'gpt-test'}, stream=True)
            body = b''.join([chunk async for chunk in response.body_iterator])
        self.assertIn(b'response.completed', body)
        self.assertFalse(ep.circuit_open)
        self.assertEqual(ep.active_requests, 0)

    async def test_stream_precontent_read_failure_not_retried(self):
        client = fixtures._StreamClient([httpx.ReadTimeout('synthetic')])
        client.send = AsyncMock(wraps=client.send)
        p, lb, ep = self.make(client)
        with patch.object(main.asyncio, 'sleep', new=AsyncMock()) as sleep:
            response = await p.proxy_responses({'model':'gpt-test'}, stream=True)
            body = b''.join([chunk async for chunk in response.body_iterator])
        self.assertIn(b'upstream_network_error', body)
        self.assertNotIn(b'"type": "response.completed"', body)
        self.assertEqual(client.send.await_count, 1)
        sleep.assert_not_awaited()
        self.assertEqual(ep.active_requests, 0)

    async def test_stream_partial_content_never_replayed_even_connecterror(self):
        class Partial(fixtures._StreamResponse):
            async def aiter_bytes(self):
                yield b'data: {"type":"response.output_text.delta","delta":"synthetic"}\n\n'
                raise httpx.ConnectError('synthetic late failure')
        upstream = Partial([])
        client = fixtures._StreamClient([upstream])
        client.send = AsyncMock(wraps=client.send)
        p, lb, ep = self.make(client)
        with patch.object(main.asyncio, 'sleep', new=AsyncMock()) as sleep:
            response = await p.proxy_responses({'model':'gpt-test'}, stream=True)
            body = b''.join([chunk async for chunk in response.body_iterator])
        self.assertIn(b'output_text.delta', body)
        self.assertIn(b'upstream_network_error', body)
        self.assertEqual(client.send.await_count, 1)
        self.assertTrue(upstream.closed)
        sleep.assert_not_awaited()

    async def test_metrics_expose_counts_not_request_content(self):
        p, lb, ep = self.make()
        lease = await lb.on_request_start(ep)
        await lb.on_request_end(ep, False, lease=lease, cancelled=True)
        with patch.object(main, 'copilot_proxy', p), patch.object(main, 'azure_proxy', None), patch.object(main,'proxy',None):
            response = await main.metrics()
        body = response.body.decode()
        self.assertIn('copilot_endpoint_cancelled_requests_total{endpoint="copilot-test"} 1', body)
        self.assertIn('copilot_endpoint_consecutive_errors{endpoint="copilot-test"} 0', body)
        self.assertNotIn('session-token', body)
        self.assertNotIn('Authorization', body)

    async def test_process_health_and_ready_are_separate(self):
        p, lb, ep = self.make()
        lb._open(ep)
        with patch.object(main, 'copilot_proxy', p), patch.object(main, 'azure_proxy', None), patch.object(main,'proxy',None):
            self.assertEqual(await main.health(), {'status':'healthy'})
            self.assertEqual(await main.health_live(), {'status':'alive'})
            with self.assertRaises(main.HTTPException) as caught:
                await main.health_ready()
        self.assertEqual(caught.exception.status_code,503)

    async def test_azure_configured_model_circuit_unavailable_is_503(self):
        p, lb, ep = fixtures.AzureRequestLifecycleTests()._make_proxy(Mock())
        lb._open(ep)
        with self.assertRaises(main.HTTPException) as caught:
            await p.proxy_responses({'model':'gpt-test'})
        self.assertEqual(caught.exception.status_code,503)
        self.assertEqual(caught.exception.headers['Retry-After'],'60')

    async def test_usage_failure_is_not_inference_failure_or_replay(self):
        p, lb, ep = self.make()
        req = httpx.Request('POST','https://test.invalid/responses')
        p.client.post = AsyncMock(return_value=httpx.Response(200,request=req,json={'usage':{}}))
        p._record_usage = Mock(side_effect=RuntimeError('synthetic telemetry failure'))
        response = await p.proxy_responses({'model':'gpt-test'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(p.client.post.await_count,1)
        self.assertEqual(ep.usage_record_errors,1)
        self.assertEqual(ep.total_errors,0)
        self.assertEqual(ep.completed_requests,1)

    def make_databricks(self, client):
        p = object.__new__(main.ClaudeProxy)
        ep = main.WorkspaceEndpoint('db', 'https://test.invalid', 'synthetic')
        p.load_balancer = main.LoadBalancer([ep], circuit_breaker_threshold=20)
        p.global_stats = main.GlobalStats()
        p.client = client
        p._record_usage = Mock()
        return p, p.load_balancer, ep

    async def test_databricks_buffered_cancellation_settles_identity(self):
        client = fixtures._BlockingClient()
        p, lb, ep = self.make_databricks(client)
        task = asyncio.create_task(p.proxy_request({'model':'claude-test','messages':[]}))
        await client.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(ep.active_requests,0)
        self.assertEqual(ep.cancelled_requests,1)
        self.assertEqual(ep.total_errors,0)
        self.assertFalse(ep.circuit_open)

    async def test_pool_contention_not_upstream_failure_all_buffered_providers(self):
        for kind in ('databricks','azure'):
            with self.subTest(provider=kind):
                client = Mock()
                client.post = AsyncMock(side_effect=httpx.PoolTimeout('synthetic local contention'))
                p, lb, ep = (self.make_databricks(client) if kind == 'databricks' else
                             fixtures.AzureRequestLifecycleTests()._make_proxy(client, threshold=20))
                call = p.proxy_request if kind == 'databricks' else p.proxy_responses
                with patch.object(main.asyncio,'sleep',new=AsyncMock()):
                    with self.assertRaises(main.HTTPException):
                        await call({'model':'gpt-test','input':[]})
                self.assertEqual(ep.total_errors,0)
                self.assertEqual(ep.active_requests,0)
                self.assertFalse(ep.circuit_open)
                self.assertEqual(ep.neutral_requests,3)
