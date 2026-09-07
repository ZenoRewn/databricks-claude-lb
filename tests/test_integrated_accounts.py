"""Synthetic capability/selection tests: no enrollment, files or real accounts."""
import asyncio
import io
import json
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
import main

NAMES=('zenoren-icloud','zenoren-msft','zenoren-outlook','zenoren-qq')
MODELS=['gpt-6-astra','gpt-5.6-sol']


class CapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def make(self):
        endpoints=[main.CopilotEndpoint(name,'',models=MODELS,api_types=['responses']) for name in NAMES]
        proxy=main.CopilotProxy(main.LoadBalancer(endpoints,strategy='least_requests'),'')
        await proxy.client.aclose()
        requests=[]
        async def handler(req):
            requests.append(req)
            return httpx.Response(200,json={'id':'synthetic','status':'completed','output':[],'usage':{}})
        proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(handler),trust_env=False)
        proxy._build_headers=AsyncMock(return_value={})
        self.addAsyncCleanup(proxy.client.aclose)
        return proxy,endpoints,requests

    async def test_missing_api_types_legacy_both_explicit_validation(self):
        ep=main.CopilotEndpoint('legacy','',1,[],{'type':'literal'})
        self.assertEqual(ep.token_source,{'type':'literal'})
        self.assertTrue(ep.supports('anything','responses'));self.assertTrue(ep.supports('anything','chat'))
        for bad in (None,[],(),'', 'responses',['messages'],['responses','responses'],[1],{}):
            with self.subTest(value=bad),self.assertRaises(ValueError):
                main.CopilotEndpoint('bad','',api_types=bad)

    async def test_model_and_api_before_selection_availability_trial(self):
        p,eps,requests=await self.make()
        for ep in eps:
            p.load_balancer._open(ep);ep.circuit_retry_at=0
        for _ in range(100):
            self.assertIsNone(p._select_endpoint('gpt-6-astra','chat'))
            self.assertFalse(p.can_handle('gpt-6-astra','chat'))
            self.assertFalse(p.supports_model('gpt-6-astra','chat'))
            self.assertIsNone(p._select_endpoint('other','responses'))
        self.assertTrue(p.can_handle('gpt-6-astra','responses'))
        self.assertTrue(all(not e.half_open_in_flight and e.total_requests==0 for e in eps))
        self.assertEqual(requests,[])
        with self.assertRaises(main.HTTPException) as caught:
            await p.proxy_chat_completions({'model':'gpt-6-astra'},stream=True)
        self.assertEqual(caught.exception.status_code,404)
        p._build_headers.assert_not_awaited()

    async def test_four_equal_weight_least_requests_sequential_and_concurrent(self):
        p,eps,requests=await self.make()
        # Sequential tied active counts must use total_requests for a fair split.
        for i in range(40):
            await p.proxy_responses({'model':MODELS[i%2]})
        self.assertEqual([e.total_requests for e in eps],[10]*4)
        self.assertEqual(len(requests),40)
        held=[]
        for _ in range(12):
            ep=p._select_endpoint(MODELS[0],'responses')
            held.append((ep,await p.load_balancer.on_request_start(ep)))
        self.assertEqual([e.active_requests for e in eps],[3]*4)
        for ep,lease in held:
            await p.load_balancer.on_request_end(ep,success=True,lease=lease)
        self.assertTrue(all(e.active_requests==0 for e in eps))

    async def test_mixed_legacy_chat_and_responses_only_no_wrong_retry(self):
        p,eps,requests=await self.make()
        legacy=main.CopilotEndpoint('legacy','',models=MODELS)
        p.load_balancer.endpoints.append(legacy)
        for _ in range(10):
            ep=p._select_endpoint('gpt-6-astra','chat')
            self.assertIs(ep,legacy)
        p.load_balancer._open(legacy)
        with patch.object(main,'copilot_proxy',p),patch.object(main,'azure_proxy',None):
            with self.assertRaises(main.HTTPException) as caught:
                await main._route_openai({'model':'gpt-6-astra'},False,'chat')
        self.assertEqual(caught.exception.status_code,503)
        self.assertEqual(requests,[])
        self.assertTrue(all(e.total_requests==0 for e in eps))

    async def test_router_uses_actual_adapter_api_not_incoming_chat(self):
        p,eps,requests=await self.make()
        with patch.object(main,'copilot_proxy',p),patch.object(main,'azure_proxy',None):
            result=await main._route_openai_chat({'model':'gpt-5.6-sol','messages':[{'role':'user','content':'synthetic'}]},False)
            self.assertEqual(result.status_code,200)
            self.assertEqual(len(requests),1)
            self.assertTrue(requests[0].url.path.endswith('/responses'))
            self.assertNotIn('gpt-6-astra',main.OPENAI_CHAT_TO_RESPONSES_MODELS_LOWER)
            with self.assertRaises(main.HTTPException) as caught:
                await main._route_openai_chat({'model':'gpt-6-astra','messages':[]},False)
            self.assertEqual(caught.exception.status_code,404)
            self.assertEqual(len(requests),1)

    async def test_retry_filters_actual_api_and_never_replays_partial(self):
        p,eps,requests=await self.make()
        chat=main.CopilotEndpoint('chat-only','',models=MODELS,api_types=['chat'])
        p.load_balancer.endpoints=[chat,eps[0]]
        attempts=[]
        async def handler(req):
            attempts.append(req)
            if len(attempts)==1: raise httpx.ConnectError('synthetic pre-execution connect')
            return httpx.Response(200,json={'usage':{}})
        await p.client.aclose()
        p.client=httpx.AsyncClient(transport=httpx.MockTransport(handler),trust_env=False)
        self.addAsyncCleanup(p.client.aclose)
        with patch.object(main.asyncio,'sleep',new=AsyncMock()):
            await p.proxy_responses({'model':MODELS[0]})
        self.assertEqual(len(attempts),2)
        self.assertEqual(chat.total_requests,0)
        self.assertEqual(eps[0].total_requests,2)

    async def test_configuration_invalid_allowlist_before_auth_resolution(self):
        for value in ('null','[]','responses','[chat, bad]'):
            config='github_copilot:\n  endpoints:\n    - name: synthetic\n      api_types: '+value+'\n'
            with patch('builtins.open',return_value=io.StringIO(config)),patch.object(main,'resolve_github_token') as resolve:
                with self.assertRaises(ValueError): main.load_config('synthetic.yaml')
                resolve.assert_not_called()

    async def test_configuration_no_repeated_legacy_credential_fallback(self):
        config='github_copilot:\n  endpoints:\n'+''.join(f'    - name: {name}\n      models: [gpt-6-astra, gpt-5.6-sol]\n      api_types: [responses]\n' for name in NAMES)
        with patch('builtins.open',return_value=io.StringIO(config)),patch.object(main,'ClaudeProxy'),patch.object(main,'CopilotProxy') as construct,patch.object(main,'resolve_github_token',return_value=('synthetic-shared',{'type':'file','path':'synthetic-legacy'})):
            with self.assertRaisesRegex(ValueError,'Duplicate Copilot credential'):
                main.load_config('synthetic.yaml')
            construct.assert_not_called()
        with patch('builtins.open',return_value=io.StringIO(config)),patch.object(main,'ClaudeProxy'),patch.object(main,'CopilotProxy') as construct,patch.object(main,'resolve_github_token',side_effect=[(f'synthetic-{i}',{'type':'file','path':f'synthetic-{i}'}) for i in range(4)]):
            main.load_config('synthetic.yaml')
            eps=construct.call_args.args[0].endpoints
            self.assertEqual([e.name for e in eps],list(NAMES))
            self.assertTrue(all(e.models==MODELS and e.api_types==('responses',) and e.weight==1 for e in eps))

    async def test_auth_429_5xx_stream_failures_not_blanket_neutral(self):
        for provider in ('copilot','azure','databricks'):
            for status in (401,403,429,500,503):
                with self.subTest(provider=provider,status=status):
                    if provider=='copilot': ep=main.CopilotEndpoint('synthetic','');cls=main.CopilotProxy
                    elif provider=='azure': ep=main.AzureOpenAIEndpoint('synthetic','http://synthetic','');cls=main.AzureOpenAIProxy
                    else: ep=main.WorkspaceEndpoint('synthetic','http://synthetic','');cls=main.ClaudeProxy
                    p=cls(main.LoadBalancer([ep],circuit_breaker_threshold=1),'')
                    await p.client.aclose()
                    calls=[]
                    async def handler(req):
                        calls.append(req)
                        return httpx.Response(status,json={'error':{'message':'synthetic'}},headers={'Retry-After':'120'})
                    p.client=httpx.AsyncClient(transport=httpx.MockTransport(handler),trust_env=False)
                    p.get_session_token=AsyncMock(return_value='synthetic')
                    p._build_headers=AsyncMock(return_value={})
                    try:
                        await p.load_balancer.on_request_start(ep)
                        if provider=='databricks': response=await p._stream_request(ep,'http://synthetic',{}, {},model='synthetic',start_time=time.time())
                        else: response=await p._stream_response(ep,'http://synthetic',{}, {},'synthetic','responses',time.time())
                        body=b''.join([b async for b in response.body_iterator])
                        self.assertEqual(ep.total_errors,1);self.assertEqual(ep.neutral_requests,0)
                        self.assertEqual(ep.active_requests,0);self.assertEqual(ep.successful_requests,0)
                        self.assertEqual(len(calls),2 if provider=='copilot' and status==401 else 1)
                    finally: await p.client.aclose()

    async def test_responses_image_shape_and_actual_built_utf8_bytes(self):
        p,eps,requests=await self.make()
        body={'model':MODELS[0],'input':[{'role':'user','content':[{'type':'input_image','image_url':'https://example.invalid/synthetic.png'},{'type':'input_text','text':'汉😀'}]}]}
        self.assertTrue(p._has_image(body))
        self.assertTrue(p._has_image({'messages':[{'content':[{'type':'image_url','image_url':{}}]}]}))
        self.assertFalse(p._has_image({'input':'synthetic'}))
        self.assertFalse(p._has_image({'input':[None,{'content':'synthetic'}]}))
        async def handler(req):
            requests.append(req)
            return httpx.Response(200,content=b'data: {"type":"response.completed","response":{"id":"synthetic"}}\n\n')
        await p.client.aclose()
        p.client=httpx.AsyncClient(transport=httpx.MockTransport(handler),trust_env=False)
        self.addAsyncCleanup(p.client.aclose)
        with self.assertLogs('main',level='INFO') as logs:
            result=await p.proxy_responses(body,stream=True)
            b''.join([b async for b in result.body_iterator])
        ended=[r for r in logs.records if getattr(r,'kind',None)=='copilot_stream_end']
        self.assertEqual(len(ended),1)
        self.assertEqual(ended[0].input_bytes,len(requests[0].content))
        self.assertTrue(ended[0].has_image)
        self.assertNotEqual(ended[0].input_bytes,len(json.dumps(body,ensure_ascii=False)))


if __name__=='__main__': unittest.main()
