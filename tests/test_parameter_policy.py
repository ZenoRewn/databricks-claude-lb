"""Expose compatibility drops; opt-in strict mode rejects before any model call."""
import json
import unittest
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace

import httpx
import main


class ParameterPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self,*,stream=False,strict=None,output=None,extras=None):
        endpoint=main.WorkspaceEndpoint('fixture','https://fixture.invalid','synthetic')
        proxy=main.ClaudeProxy(main.LoadBalancer([endpoint]),'synthetic')
        await proxy.client.aclose();sent=[]
        async def upstream(request):
            sent.append(json.loads(request.content))
            if stream:return httpx.Response(200,content=b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
            return httpx.Response(200,json={'type':'message','stop_reason':'end_turn','usage':{}})
        proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        body={'model':'claude-opus-5','messages':[],'stream':stream,
              'output_config':output if output is not None else {'effort':'high','format':{'type':'json_schema'}}}
        body.update(extras or {})
        headers={'Authorization':'Bearer synthetic'}
        if strict is not None:headers['X-LB-Strict-Parameters']=strict
        try:
            with patch.object(main,'proxy',proxy),patch.object(main,'usage_store',None):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app),base_url='http://local') as client:
                    response=await client.post('/v1/messages',json=body,headers=headers)
            return response,sent
        finally:await proxy.close()

    async def test_compat_reports_schema_drop_on_both_response_types(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                response,sent=await self.run_case(stream=stream)
                self.assertEqual(response.status_code,200)
                self.assertIn('output_config.format',response.headers.get('x-lb-dropped-parameters',''))
                self.assertEqual(response.headers.get('x-lb-parameter-policy'),'compat')
                self.assertEqual(sent[0]['output_config'],{'effort':'high'})

    async def test_strict_rejects_unforwarded_schema_before_post(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                response,sent=await self.run_case(stream=stream,strict='true')
                self.assertEqual(response.status_code,400)
                self.assertEqual(response.json()['detail']['error']['code'],'parameter_not_forwarded')
                self.assertEqual(sent,[])

    async def test_strict_preserves_the_existing_native_effort_contract(self):
        response,sent=await self.run_case(strict='true',output={'effort':'high'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(sent[0]['output_config'],{'effort':'high'})
        self.assertEqual(response.headers.get('x-lb-parameter-policy'),'strict')
        self.assertNotIn('x-lb-dropped-parameters',response.headers)

    async def test_known_tool_and_context_drops_are_visible_without_values(self):
        response,_=await self.run_case(output={'effort':'high'},extras={
            'context_management':{'sensitive':'must-not-appear'},
            'tools':[{'name':'fixture','defer_loading':True,'input_examples':['must-not-appear']}]})
        names=response.headers.get('x-lb-dropped-parameters','')
        for field in ('context_management','tools.defer_loading','tools.input_examples'):
            self.assertIn(field,names)
        self.assertNotIn('must-not-appear',names)

    async def test_invalid_policy_value_is_not_silently_treated_as_compat(self):
        response,sent=await self.run_case(strict='typo')
        self.assertEqual(response.status_code,400)
        self.assertEqual(sent,[])

    async def test_responses_sampling_drop_obeys_the_same_strict_policy(self):
        proxy=SimpleNamespace(verify_api_key=lambda key:key=='synthetic')
        for strict in ('false','true'):
            with self.subTest(strict=strict):
                route=AsyncMock(return_value=main.JSONResponse({'id':'r','status':'completed','output':[]}))
                with patch.object(main,'proxy',None),patch.object(main,'azure_proxy',None),patch.object(main,'copilot_proxy',proxy),\
                        patch.object(main,'_route_openai',route):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app),base_url='http://local') as client:
                        response=await client.post('/v1/responses',json={'model':'gpt-5.6-sol','input':[],'temperature':.5},
                                                   headers={'Authorization':'Bearer synthetic','X-LB-Strict-Parameters':strict})
                if strict=='true':
                    self.assertEqual(response.status_code,400);route.assert_not_awaited()
                else:
                    self.assertEqual(response.status_code,200)
                    self.assertIn('temperature',response.headers['x-lb-dropped-parameters'])
                    self.assertNotIn('temperature',route.call_args.args[0])
