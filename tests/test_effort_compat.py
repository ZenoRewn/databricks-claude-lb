"""Preserve the deployed effort contract without enabling unverified schema passthrough."""
import json
import unittest

import httpx
import main


class DeployedEffortTests(unittest.IsolatedAsyncioTestCase):
    async def test_deployed_payload_contract_on_json_and_stream_paths(self):
        for stream in (False, True):
            for model, output, expected in (
                ("claude-opus-5", {"effort": "high", "format": {"type": "json_schema"}}, {"effort": "high"}),
                ("claude-opus-5", {"format": {"type": "json_schema"}}, None),
                ("claude-sonnet-5", {"effort": "high"}, None),
                ("claude-opus-5", {"effort": "invalid"}, {"effort": "invalid"}),
            ):
                with self.subTest(stream=stream, model=model, output=output):
                    captured = []
                    async def upstream(request):
                        captured.append(json.loads(request.content))
                        if stream:
                            return httpx.Response(200, content=b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
                                                  headers={"content-type": "text/event-stream"})
                        return httpx.Response(200, json={"type": "message", "usage": {}})
                    ep = main.WorkspaceEndpoint("fixture", "https://fixture.invalid", "synthetic")
                    proxy = main.ClaudeProxy(main.LoadBalancer([ep]), "synthetic")
                    await proxy.client.aclose()
                    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
                    try:
                        response = await proxy.proxy_request({"model": model, "messages": [],
                                                              "output_config": output}, stream=stream)
                        if stream:
                            async for _ in response.body_iterator:
                                pass
                        self.assertEqual(captured[0].get("output_config"), expected)
                        if not stream:
                            self.assertEqual(response.headers.get("x-claude-effort-forwarded"),
                                             "high" if expected == {"effort": "high"} else None)
                        self.assertEqual(ep.active_requests, 0)
                    finally:
                        await proxy.close()
