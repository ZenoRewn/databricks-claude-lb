"""End-to-end ASGI lifecycle accounting, independent of HTTP success codes."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from request_telemetry import (RequestTelemetry, RequestTelemetryMiddleware,
                               inference_call, note_admission, note_generation,
                               note_json_result)
from request_telemetry import TELEMETRY


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, app, send_failure=False, path='/v1/responses'):
        metrics = RequestTelemetry()
        messages = []
        async def send(message):
            if send_failure and message['type'] == 'http.response.body':
                raise OSError('synthetic disconnect')
            messages.append(message)
        scope = {'type':'http', 'method':'POST', 'path':path, 'headers':[]}
        try:
            await RequestTelemetryMiddleware(app, metrics)(scope, AsyncMock(), send)
        except OSError:
            pass
        return metrics, messages

    async def test_http_200_error_is_failed_once_and_not_completed(self):
        async def app(scope, receive, send):
            note_generation('error')
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'data: error\n\n'})
        metrics, _ = await self.drive(app)
        self.assertEqual(metrics.outcomes, {('responses','failed'):1})
        self.assertEqual(metrics.active['responses'], 0)

    async def test_valid_upstream_terminal_is_not_delivery_after_disconnect(self):
        async def app(scope, receive, send):
            note_generation('completed')
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'data: completed\n\n'})
        metrics, _ = await self.drive(app, send_failure=True)
        self.assertEqual(metrics.outcomes, {('responses','client_disconnected'):1})

    async def test_200_without_generation_evidence_is_unknown(self):
        async def app(scope, receive, send):
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        metrics, _ = await self.drive(app)
        self.assertEqual(metrics.outcomes, {('responses','unknown'):1})

    async def test_incomplete_json_is_separate_from_completed(self):
        async def app(scope, receive, send):
            note_json_result({'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'}}, 'responses')
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        metrics, _ = await self.drive(app)
        self.assertEqual(metrics.outcomes, {('responses','incomplete'):1})

    async def test_two_send_invocations_in_one_admission_are_visible(self):
        async def app(scope, receive, send):
            note_admission()
            for status in (401, 200):
                await inference_call(AsyncMock(return_value=httpx.Response(status))(), 'copilot', 'responses')
            note_generation('completed')
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        metrics, messages = await self.drive(app)
        self.assertEqual(metrics.sends[('copilot','responses')], 2)
        self.assertEqual(metrics.admissions['responses'], 1)
        self.assertEqual(metrics.outcomes, {('responses','completed'):1})
        self.assertIn(b'x-lb-request-id', dict(messages[0]['headers']))
        exposition = metrics.render()
        self.assertIn('lb_upstream_send_started_total{provider="copilot",api_type="responses"} 2', exposition)
        self.assertNotIn('request_id=', exposition)

    async def test_parallel_requests_do_not_share_outcomes(self):
        metrics = RequestTelemetry()
        async def app(scope, receive, send):
            note_generation('completed' if scope['headers'] else 'incomplete')
            await asyncio.sleep(0)
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        mw=RequestTelemetryMiddleware(app,metrics)
        await asyncio.gather(*(mw({'type':'http','method':'POST','path':'/v1/responses','headers':h},AsyncMock(),AsyncMock())
                               for h in ([],[(b'x-test',b'1')])) )
        self.assertEqual(metrics.outcomes, {('responses','completed'):1,('responses','incomplete'):1})

    async def test_cancellation_has_one_outcome_and_context_is_reset(self):
        async def app(scope, receive, send):
            raise asyncio.CancelledError
        metrics=RequestTelemetry()
        with self.assertRaises(asyncio.CancelledError):
            await RequestTelemetryMiddleware(app,metrics)({'type':'http','method':'POST','path':'/v1/messages','headers':[]},AsyncMock(),AsyncMock())
        note_admission()
        self.assertEqual(metrics.outcomes,{('messages','cancelled'):1})
        self.assertFalse(metrics.admissions)
        self.assertEqual(metrics.active['messages'],0)

    async def test_probes_are_not_inference_requests(self):
        metrics,_=await self.drive(AsyncMock(),path='/health/live')
        self.assertFalse(metrics.started)


class ProxyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_entrypoint_reports_json_success_and_sse_failure(self):
        import main
        ep=main.WorkspaceEndpoint('fixture','https://fixture.invalid','synthetic')
        proxy=main.ClaudeProxy(main.LoadBalancer([ep]),'synthetic')
        await proxy.client.aclose()
        async def upstream(request):
            body=json.loads(request.content)
            if body.get('stream'):
                return httpx.Response(503,json={'error_code':'TEMPORARILY_UNAVAILABLE','message':'capacity'})
            return httpx.Response(200,json={'type':'message','stop_reason':'end_turn','usage':{}})
        proxy.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            with patch.object(main,'proxy',proxy), patch.object(main,'usage_store',None):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app),base_url='http://local') as client:
                    for stream,outcome in ((False,'completed'),(True,'failed')):
                        before=TELEMETRY.outcomes[('messages',outcome)]
                        sends=TELEMETRY.sends[('databricks','messages')]
                        response=await client.post('/v1/messages',json={'model':'claude-opus-5','messages':[], 'stream':stream},
                                                   headers={'Authorization':'Bearer synthetic'})
                        self.assertEqual(response.status_code,200)
                        self.assertIn('x-lb-request-id',response.headers)
                        self.assertEqual(TELEMETRY.outcomes[('messages',outcome)],before+1)
                        self.assertEqual(TELEMETRY.sends[('databricks','messages')],sends+1)
                    exposition=(await client.get('/metrics')).text
                    self.assertIn('lb_requests_finished_total',exposition)
        finally:
            await proxy.close()
