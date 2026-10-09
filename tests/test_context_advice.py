"""Advisory context visibility, never content trimming. Author: Zeno Ren."""
import copy
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import main
import model_capabilities as capabilities
from tests.test_context_capabilities import catalog_data


def test_size_advice_is_available_without_fabricated_model_limits():
    payload={'input':'x'*300000}
    original=copy.deepcopy(payload)
    result=capabilities.evaluate_budget(capabilities.CapabilityCatalog(),'copilot','responses','synthetic-model',payload)
    assert result['context_advice']=='large_input'
    assert result['input_limit'] is None
    assert result['estimate_confidence']=='low'
    assert result['enforcement_allowed'] is False
    assert original==payload


def test_verified_limits_support_near_and_over_advice_without_input_rejection():
    catalog=capabilities.CapabilityCatalog(catalog_data(limits={
        'input_tokens':1000,'context_tokens':2000,'output_tokens':100}),
        now=lambda:datetime(2026,9,30,tzinfo=timezone.utc))
    for text,expected in [('x'*3400,'estimated_near_limit'),('x'*5000,'estimated_over_limit')]:
        result=capabilities.evaluate_budget(catalog,'copilot','responses','synthetic-model',{'input':text},mode='enforce')
        assert result['context_advice']==expected


class ContextAdviceHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_and_sse_expose_estimate_confidence_and_keep_input(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                ep=main.CopilotEndpoint('synthetic','',models=['synthetic-model'])
                proxy=main.CopilotProxy(main.LoadBalancer([ep]),'synthetic')
                await proxy.client.aclose()
                sent=[]
                def upstream(request):
                    sent.append(json.loads(request.content))
                    if stream:
                        return httpx.Response(200,content=b'data: {"type":"response.completed","response":{"id":"synthetic","status":"completed","output":[]}}\n\n',headers={'content-type':'text/event-stream'})
                    return httpx.Response(200,json={'id':'synthetic','status':'completed','output':[]})
                proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream),trust_env=False)
                proxy._build_headers=AsyncMock(return_value={})
                payload={'model':'synthetic-model','stream':stream,'input':[
                    {'role':'user','content':[{'type':'input_text','text':'PRIVATE_CONTENT'*30000},
                        {'type':'input_image','image_url':'https://fixture.invalid/image'}]}]}
                try:
                    with patch.object(main,'copilot_proxy',proxy),patch.object(main,'azure_proxy',None), \
                         patch.object(capabilities,'CATALOG',capabilities.CapabilityCatalog()), \
                         patch.object(capabilities,'MODE','observe'),patch.object(main,'usage_store',None):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app),base_url='http://local',trust_env=False) as client:
                            result=await client.post('/v1/responses',json=payload,headers={'Authorization':'Bearer synthetic'})
                    self.assertEqual(result.status_code,200)
                    self.assertEqual(result.headers['x-lb-context-advice'],'large_input')
                    self.assertEqual(result.headers['x-lb-context-estimate-confidence'],'low')
                    self.assertEqual(result.headers['x-lb-context-estimate-complete'],'false')
                    self.assertIn('images',result.headers['x-lb-context-unknown-components'])
                    self.assertNotIn('PRIVATE_CONTENT',str(result.headers))
                    self.assertEqual(sent[0]['input'],payload['input'])
                    self.assertEqual(len(sent),1)
                finally:await proxy.close()

    async def test_budget_off_emits_no_advice_headers(self):
        from request_telemetry import RequestTelemetryMiddleware, RequestTelemetry
        messages=[]
        async def app(scope,receive,send):
            capabilities.observe_route_budget('copilot','responses','synthetic-model',{'input':'x'*300000})
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        async def send(message):messages.append(message)
        with patch.object(capabilities,'MODE','off'):
            await RequestTelemetryMiddleware(app,RequestTelemetry())(
                {'type':'http','method':'POST','path':'/v1/responses','headers':[]},AsyncMock(),send)
        self.assertFalse(any(k.startswith(b'x-lb-context-') for k,v in messages[0]['headers']))
