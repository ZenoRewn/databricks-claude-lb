"""Microsoft Decision models on `/v1/decision`. Author: Zeno Ren.

Decision-1.1 is not a generative model: the deployment declares
`chatCompletion: false`, and on 2026-11-26 the live eastus2 resource answered
`/openai/v1/chat/completions` with 404 and `/openai/v1/responses` with
«The requested operation is unsupported». Its only surface is
`POST {resource}/providers/microsoft/v1/systemone` taking `state` + typed
`questions` and returning `answers` + `usage` (verified live the same day, 200
in ~1.7 s, `usage: {input_tokens: 65, output_tokens: 2}`).

So it gets its own pool (`decision_deployments`) and its own entry point, and
the chat/responses candidate sets must never see it.
"""
import asyncio
import os
import sys
import tempfile
import textwrap
import unittest
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException

warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette.testclient")
from starlette.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from response_semantics import assess_json

KEY = 'test-key-decision'
PUBLIC = 'MAI-Decision-1.1'
DEPLOYMENT = 'Decision-1.1'
ANSWER = {'model': 'microsoft-decision-1.1',
          'answers': {'team': {'type': 'choice', 'choice': 'engineering', 'confidence': 0.99,
                               'probabilities': {'billing': 0.01, 'engineering': 0.99}}},
          'usage': {'input_tokens': 65, 'output_tokens': 2}}
BODY = {'model': PUBLIC, 'state': 'The API returns 500 on every call.',
        'questions': {'team': {'type': 'choice', 'instructions': 'Which team?',
                               'criteria': {'billing': 'Charges', 'engineering': 'Bugs'}}}}


def _endpoint(name='e', decision=None, deployments=('gpt-5.4',)):
    return main.AzureOpenAIEndpoint(name, f'https://{name}.invalid', 'synthetic',
                                    deployments=list(deployments),
                                    decision_deployments=dict(decision or {PUBLIC: DEPLOYMENT}))


class ConfigTests(unittest.TestCase):
    def _load(self, extra):
        cfg = textwrap.dedent('''\
            endpoints: []
            azure_openai:
              endpoints:
              - name: e
                endpoint: https://e.invalid
                api_key: k
                deployments: [gpt-5.4]
            ''') + textwrap.indent(textwrap.dedent(extra), '    ')
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.write(cfg)
        try:
            loaded = main.load_config(f.name)
        finally:
            os.unlink(f.name)
        for item in loaded:
            lb = getattr(item, 'load_balancer', None)
            if lb is not None and lb.endpoints and isinstance(lb.endpoints[0], main.AzureOpenAIEndpoint):
                return lb.endpoints[0]
        self.fail('azure proxy not built')

    def test_public_name_maps_to_the_deployment(self):
        ep = self._load('decision_deployments: {MAI-Decision-1.1: Decision-1.1}\n')
        self.assertEqual(ep.decision_deployments, {PUBLIC: DEPLOYMENT})

    def test_a_bare_list_serves_each_deployment_under_its_own_name(self):
        ep = self._load('decision_deployments: [Decision-1.1]\n')
        self.assertEqual(ep.decision_deployments, {DEPLOYMENT: DEPLOYMENT})

    def test_absent_means_none(self):
        self.assertEqual(self._load('weight: 1\n').decision_deployments, {})

    def test_malformed_entries_refuse_to_start(self):
        for bad in ('decision_deployments: Decision-1.1\n',
                    'decision_deployments: {MAI-Decision-1.1: ""}\n',
                    'decision_deployments: [1]\n'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._load(bad)

    def test_a_name_cannot_be_both_generative_and_decision(self):
        with self.assertRaises(ValueError):
            self._load('decision_deployments: [gpt-5.4]\n')


class CandidateIsolationTests(unittest.TestCase):
    def test_chat_and_responses_never_select_a_decision_deployment(self):
        lb = main.LoadBalancer([_endpoint()])
        self.assertIsNone(lb.select_endpoint_for_model(PUBLIC))
        self.assertIsNotNone(lb.select_endpoint_for_model(PUBLIC, api_type='decision'))
        self.assertIsNone(lb.select_endpoint_for_model('gpt-5.4', api_type='decision'))

    def test_url(self):
        self.assertEqual(main.azure_request_url(_endpoint(), 'decision'),
                         'https://e.invalid/providers/microsoft/v1/systemone')


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def _proxy(self, handler, endpoints=None):
        proxy = main.AzureOpenAIProxy(main.LoadBalancer(endpoints or [_endpoint()]), 'synthetic')
        await proxy.client.aclose()
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(proxy.close)
        return proxy

    async def test_forwards_to_systemone_with_the_deployment_name_and_records_usage(self):
        seen = []

        async def upstream(request):
            seen.append((request.url.path, __import__('json').loads(request.content)))
            return httpx.Response(200, json=ANSWER)
        proxy = await self._proxy(upstream)
        store = MagicMock()
        with patch.object(main, 'usage_store', store):
            result = await proxy.proxy_decision(dict(BODY))
        self.assertEqual(result.status_code, 200)
        path, sent = seen[0]
        self.assertEqual(path, '/providers/microsoft/v1/systemone')
        self.assertEqual(sent['model'], DEPLOYMENT)
        self.assertEqual(sent['questions'], BODY['questions'])
        args, kwargs = store.record.call_args
        self.assertEqual(args[:3], (PUBLIC, 65, 2))
        self.assertEqual(kwargs['provider'], 'azure_openai')
        self.assertEqual(kwargs['generation_outcome'], 'completed')

    async def test_upstream_client_errors_pass_through_without_replay(self):
        calls = []

        async def upstream(request):
            calls.append(1)
            return httpx.Response(422, json={'error': {'message': 'bad question'}})
        proxy = await self._proxy(upstream)
        with patch.object(main, 'usage_store', None), self.assertRaises(HTTPException) as ctx:
            await proxy.proxy_decision(dict(BODY))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(len(calls), 1)

    async def test_a_200_without_answers_is_not_success(self):
        async def upstream(request):
            return httpx.Response(200, json={'model': 'x', 'usage': {'input_tokens': 1, 'output_tokens': 0}})
        proxy = await self._proxy(upstream)
        with patch.object(main, 'usage_store', None), self.assertRaises(HTTPException):
            await proxy.proxy_decision(dict(BODY))

    async def test_unknown_model_is_404(self):
        proxy = await self._proxy(lambda r: httpx.Response(500))
        with self.assertRaises(HTTPException) as ctx:
            await proxy.proxy_decision({**BODY, 'model': 'nope'})
        self.assertEqual(ctx.exception.status_code, 404)


class AssessmentTests(unittest.TestCase):
    def test_answers_with_usage_is_completed(self):
        result = assess_json(ANSWER, 'decision')
        self.assertEqual(result.outcome, 'completed')
        self.assertEqual(result.usage, {'input_tokens': 65, 'output_tokens': 2})

    def test_missing_answers_is_invalid_protocol(self):
        self.assertEqual(assess_json({'model': 'x'}, 'decision').reason, 'invalid_protocol')


class RequestValidationTests(unittest.TestCase):
    def test_accepts_the_documented_shapes(self):
        for body in (BODY,
                     {**BODY, 'state': {'message': 'x', 'n': 3}},
                     {**BODY, 'questions': {'s': {'type': 'score', 'criteria': ['a', 'b']},
                                            'n': {'type': 'noul', 'instructions': 'x'}}}):
            with self.subTest(body=body):
                self.assertIsNone(main.decision_request_error(body))

    def test_rejects_what_upstream_would_reject_or_we_cannot_serve(self):
        for body in ({**BODY, 'questions': {}},
                     {**BODY, 'questions': {'q': {'type': 'freeform'}}},
                     {**BODY, 'questions': ['x']},
                     {k: v for k, v in BODY.items() if k != 'state'},
                     {**BODY, 'stream': True},
                     {**BODY, 'assets': 'x'}):
            with self.subTest(body=body):
                self.assertIsInstance(main.decision_request_error(body), str)


class _Stub:
    api_key = KEY

    def verify_api_key(self, key):
        return key == KEY


class HttpTests(unittest.TestCase):
    def setUp(self):
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[KEY] = 'default'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        self.azure = main.AzureOpenAIProxy(main.LoadBalancer([_endpoint()]), KEY)
        self.azure.proxy_decision = AsyncMock(return_value=main.JSONResponse(ANSWER))
        for attr, value in (('azure_proxy', self.azure), ('copilot_proxy', None), ('proxy', None)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def post(self, path, body, key=KEY):
        return self.client.post(path, json=body, headers={'Authorization': f'Bearer {key}'})

    def test_success_is_passed_through(self):
        r = self.post('/v1/decision', BODY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['answers']['team']['choice'], 'engineering')

    def test_wrong_key_is_401(self):
        self.assertEqual(self.post('/v1/decision', BODY, key='wrong').status_code, 401)

    def test_invalid_body_is_400_before_upstream(self):
        r = self.post('/v1/decision', {**BODY, 'questions': {}})
        self.assertEqual(r.status_code, 400)
        self.azure.proxy_decision.assert_not_called()

    def test_provider_allowlist_applies(self):
        store = MagicMock()
        store.allowed.return_value = frozenset({'copilot'})
        with patch.object(main, 'key_policy_store', store):
            r = self.post('/v1/decision', BODY)
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()['detail']['error']['code'], 'provider_not_allowed')

    def test_chat_entry_point_points_callers_at_decision(self):
        for path, body in (('/v1/chat/completions', {'model': PUBLIC, 'messages': [{'role': 'user', 'content': 'x'}]}),
                           ('/v1/responses', {'model': PUBLIC, 'input': 'x'})):
            with self.subTest(path=path):
                r = self.post(path, body)
                self.assertEqual(r.status_code, 400, r.text)
                err = r.json()['detail']['error']
                self.assertEqual(err['code'], 'unsupported_api_for_model')
                self.assertIn('/v1/decision', err['message'])

    def test_model_listing_finds_it(self):
        listing = self.client.get('/v1/models').json()['data']
        entry = next(e for e in listing if e['id'] == PUBLIC)
        self.assertEqual(entry['supported_endpoints'], ['/v1/decision'])
        for path in ('/models', '/v1/models'):
            with self.subTest(path=path):
                self.assertIn(PUBLIC, [e['id'] for e in self.client.get(path).json()['data']])
                one = self.client.get(f'{path}/{PUBLIC}')
                self.assertEqual(one.status_code, 200)
                self.assertEqual(one.json()['supported_endpoints'], ['/v1/decision'])


class PricingTests(unittest.TestCase):
    def test_global_standard_list_price(self):
        # Azure Retail Prices API, eastus2, «Decision 1.1 Gl 1M Tokens», checked 2026-11-26.
        pricing = main.get_model_pricing(PUBLIC)
        self.assertEqual((pricing['input'], pricing['output']), (0.042, 0.042))
        self.assertEqual(main.calculate_cost(PUBLIC, 1_000_000, 1_000_000), 0.084)


if __name__ == '__main__':
    unittest.main()
