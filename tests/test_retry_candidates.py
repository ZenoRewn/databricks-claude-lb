"""Prefer compatible untried endpoints without relaxing replay or account pinning."""
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import main


class CandidateTests(unittest.TestCase):
    def test_tried_endpoint_does_not_win_weighted_selection_again(self):
        a=main.WorkspaceEndpoint('a','https://a.invalid','synthetic',weight=1000)
        b=main.WorkspaceEndpoint('b','https://b.invalid','synthetic')
        b.total_requests=100
        lb=main.LoadBalancer([a,b])
        self.assertIs(lb.select_endpoint(tried={'a'}),b)

    def test_model_constraints_apply_before_untried_preference(self):
        a=main.WorkspaceEndpoint('a','https://a.invalid','synthetic',models=['databricks-claude-opus-5'])
        b=main.WorkspaceEndpoint('b','https://b.invalid','synthetic',models=['databricks-claude-sonnet-5'])
        lb=main.LoadBalancer([a,b])
        self.assertIs(lb.select_endpoint(model='databricks-claude-opus-5',tried={'a'}),a)
        self.assertIsNone(lb.select_endpoint(model='databricks-claude-haiku-4-5'))

    def test_azure_never_retries_to_a_different_deployment(self):
        a=main.AzureOpenAIEndpoint('a','https://a.invalid','synthetic',deployments=['gpt-test'])
        b=main.AzureOpenAIEndpoint('b','https://b.invalid','synthetic',deployments=['other'])
        c=main.AzureOpenAIEndpoint('c','https://c.invalid','synthetic',deployments=['gpt-test'])
        lb=main.LoadBalancer([a,b,c])
        self.assertIs(lb.select_endpoint_for_model('gpt-test',tried={'a'}),c)

    def test_single_candidate_retains_bounded_same_endpoint_retry_policy(self):
        ep=main.WorkspaceEndpoint('only','https://only.invalid','synthetic')
        self.assertIs(main.LoadBalancer([ep]).select_endpoint(tried={'only'}),ep)

    def test_invalid_model_allowlist_is_rejected_at_configuration_boundary(self):
        for value in (None,'model',[''],[1]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                main.WorkspaceEndpoint('a','https://a.invalid','synthetic',models=value)


class RetryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_absent_pool_remains_unavailable_not_unsupported_model(self):
        proxy=main.ClaudeProxy(main.LoadBalancer([]),'synthetic')
        try:
            with self.assertRaises(main.HTTPException) as captured:
                await proxy.proxy_request({'model':'claude-opus-5','messages':[]})
            self.assertEqual(captured.exception.status_code,503)
        finally:await proxy.close()

    async def test_databricks_429_uses_a_different_compatible_candidate(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                a=main.WorkspaceEndpoint('a','https://a.invalid','synthetic',weight=1000)
                b=main.WorkspaceEndpoint('b','https://b.invalid','synthetic')
                b.total_requests=100
                proxy=main.ClaudeProxy(main.LoadBalancer([a,b]),'synthetic')
                await proxy.client.aclose()
                visited=[]
                async def upstream(request):
                    visited.append(request.url.host)
                    if len(visited)==1:return httpx.Response(429,json={'message':'quota'})
                    if stream:return httpx.Response(200,content=b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
                    return httpx.Response(200,json={'type':'message','stop_reason':'end_turn','usage':{}})
                proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
                try:
                    with patch.object(main.asyncio,'sleep',AsyncMock()),patch.object(main,'usage_store',None):
                        response=await proxy.proxy_request({'model':'claude-opus-5','messages':[]},stream=stream)
                        if stream:
                            async for _ in response.body_iterator:pass
                    self.assertEqual(visited,['a.invalid','b.invalid'])
                    self.assertEqual(a.active_requests+b.active_requests,0)
                finally:await proxy.close()

    async def test_copilot_pin_and_affinity_still_take_priority(self):
        a=main.CopilotEndpoint('a','synthetic-a');b=main.CopilotEndpoint('b','synthetic-b')
        proxy=main.CopilotProxy(main.LoadBalancer([a,b]),'synthetic')
        try:
            self.assertIs(proxy._select_endpoint('gpt-test','responses',tried={'a'}),b)
            self.assertIs(proxy._select_endpoint('gpt-test','responses',pinned=a,tried={'a'}),a)
            key='stable-key';original=proxy._select_endpoint('gpt-test','responses',session_key=key)
            self.assertIs(proxy._select_endpoint('gpt-test','responses',session_key=key,tried={original.name}),original)
        finally:await proxy.close()

    async def test_azure_stream_tracks_every_prior_candidate(self):
        endpoints=[main.AzureOpenAIEndpoint(n,f'https://{n}.invalid','synthetic',deployments=['gpt-test'],weight=w)
                   for n,w in (('a',1000),('b',100),('c',1))]
        endpoints[1].total_requests=100;endpoints[2].total_requests=1000
        proxy=main.AzureOpenAIProxy(main.LoadBalancer(endpoints),'synthetic')
        await proxy.client.aclose();visited=[]
        async def upstream(request):
            visited.append(request.url.host)
            if len(visited)<3:return httpx.Response(429,json={'message':'quota'})
            return httpx.Response(200,content=b'data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}\n\n')
        proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            with patch.object(main.asyncio,'sleep',AsyncMock()),patch.object(main,'usage_store',None):
                response=await proxy.proxy_responses({'model':'gpt-test','input':[]},stream=True)
                async for _ in response.body_iterator:pass
            self.assertEqual(visited,['a.invalid','b.invalid','c.invalid'])
            self.assertTrue(all(ep.active_requests==0 for ep in endpoints))
        finally:await proxy.close()

    async def test_copilot_runtime_uses_untried_account_only_for_unpinned_requests(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                a=main.CopilotEndpoint('a','synthetic-a',weight=1000)
                b=main.CopilotEndpoint('b','synthetic-b');b.total_requests=100
                a.session_base_url='https://a.invalid';b.session_base_url='https://b.invalid'
                proxy=main.CopilotProxy(main.LoadBalancer([a,b]),'synthetic')
                proxy._build_headers=AsyncMock(return_value={})
                await proxy.client.aclose();visited=[]
                async def upstream(request):
                    visited.append(request.url.host)
                    if len(visited)==1:return httpx.Response(429,json={'message':'quota'})
                    payload={'id':'r','status':'completed','output':[],'usage':{}}
                    if stream:return httpx.Response(200,content=b'data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}\n\n')
                    return httpx.Response(200,json=payload)
                proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
                try:
                    with patch.object(main.asyncio,'sleep',AsyncMock()),patch.object(main,'usage_store',None):
                        response=await proxy.proxy_responses({'model':'gpt-test','input':[]},stream=stream)
                        if stream:
                            async for _ in response.body_iterator:pass
                    self.assertEqual(visited,['a.invalid','b.invalid'])
                    self.assertEqual(a.active_requests+b.active_requests,0)
                finally:await proxy.close()
