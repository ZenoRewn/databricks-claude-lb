"""Actual installed SDKs obey the explicit unsafe-replay response header."""
import unittest


class SDKRetryTests(unittest.TestCase):
    def test_openai_does_not_retry_ambiguous_gateway_502(self):
        import httpx
        import openai
        calls=[]
        def upstream(request):
            calls.append(request)
            return httpx.Response(502,headers={'X-Should-Retry':'false'},json={'error':{'code':'invalid_upstream_response','retryable':False,'execution_certainty':'unknown'}})
        with openai.OpenAI(api_key='synthetic',base_url='https://fixture.invalid/v1',max_retries=3,
                           http_client=httpx.Client(transport=httpx.MockTransport(upstream))) as client:
            with self.assertRaises(openai.APIStatusError):client.responses.create(model='synthetic',input='synthetic')
        self.assertEqual(len(calls),1)

    def test_anthropic_does_not_retry_ambiguous_gateway_502(self):
        import anthropic
        import httpx2
        calls=[]
        def upstream(request):
            calls.append(request)
            return httpx2.Response(502,headers={'X-Should-Retry':'false'},json={'type':'error','error':{'type':'api_error','message':'synthetic','retryable':False}})
        with anthropic.Anthropic(api_key='synthetic',base_url='https://fixture.invalid',max_retries=3,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(upstream))) as client:
            with self.assertRaises(anthropic.APIStatusError):client.messages.create(model='synthetic',messages=[{'role':'user','content':'synthetic'}],max_tokens=1)
        self.assertEqual(len(calls),1)
