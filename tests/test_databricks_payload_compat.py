import unittest

import main


class DatabricksPayloadCompatTests(unittest.TestCase):
    def test_promotes_system_role_messages_to_top_level_system(self):
        body = {
            "system": "base instructions",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "new instructions",
                            "cache_control": {"type": "ephemeral", "scope": "turn"},
                        }
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        }

        promoted = main.promote_system_messages(body)
        stripped = main.strip_cache_control_extras(body)

        self.assertEqual(promoted, 1)
        self.assertEqual(stripped, 1)
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(
            body["system"],
            [
                {"type": "text", "text": "base instructions"},
                {
                    "type": "text",
                    "text": "new instructions",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        )

    def test_leaves_payload_without_system_messages_unchanged(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}

        promoted = main.promote_system_messages(body)

        self.assertEqual(promoted, 0)
        self.assertEqual(body, {"messages": [{"role": "user", "content": "hello"}]})


class StreamingHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_yields_heartbeat_while_waiting_for_response_headers(self):
        async def delayed_result():
            import asyncio

            await asyncio.sleep(0.03)
            return "response"

        events = []
        async for kind, payload in main._await_with_heartbeat(
            delayed_result(), b": keep-alive\n\n", interval=0.01
        ):
            events.append((kind, payload))

        self.assertIn(("heartbeat", b": keep-alive\n\n"), events)
        self.assertEqual(events[-1], ("result", "response"))


class OpenAICompatTests(unittest.TestCase):
    def test_collects_model_catalog_from_azure_copilot_and_wildcard_defaults(self):
        class FakeLoadBalancer:
            def __init__(self, endpoints):
                self.endpoints = endpoints

        class FakeProxy:
            def __init__(self, endpoints):
                self.load_balancer = FakeLoadBalancer(endpoints)

        azure = FakeProxy([
            main.AzureOpenAIEndpoint(
                name="az-east",
                endpoint="https://example.openai.azure.com",
                api_key="key",
                deployments=["gpt-4o", "gpt-4.1"],
            )
        ])
        copilot = FakeProxy([
            main.CopilotEndpoint(
                name="gh-all",
                github_token="token",
                token_source={"type": "literal"},
                models=[],
            ),
            main.CopilotEndpoint(
                name="gh-explicit",
                github_token="token",
                token_source={"type": "literal"},
                models=["gemini-2.5-pro"],
            ),
        ])

        ids = main._collect_openai_model_ids(azure, copilot)

        self.assertEqual(ids, sorted(set(ids)))
        self.assertIn("gpt-4.1", ids)
        self.assertIn("gpt-4o", ids)
        self.assertIn("gpt-5.5", ids)
        self.assertIn("gpt-5-codex", ids)
        self.assertIn("gemini-2.5-pro", ids)

    def test_drops_only_nonpositive_chat_token_limits(self):
        body = {
            "model": "gpt-4.1",
            "max_tokens": 0,
            "max_completion_tokens": "0",
            "temperature": 0,
        }

        removed = main._drop_nonpositive_token_limits(body)

        self.assertEqual(removed, ["max_tokens", "max_completion_tokens"])
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["temperature"], 0)

    def test_builds_responses_payload_from_chat_body(self):
        body = {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "stay in character"},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 0,
            "max_completion_tokens": 512,
            "temperature": 0.3,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look something up",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }

        payload = main._build_responses_payload_from_chat(body)

        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertEqual(
            payload["input"],
            [
                {"role": "system", "content": "stay in character"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(payload["max_output_tokens"], 512)
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["name"], "lookup")
        self.assertEqual(payload["tool_choice"], "auto")

    def test_wraps_responses_text_as_chat_completion(self):
        response = {
            "id": "resp_123",
            "output_text": "hello",
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }

        chat = main._responses_json_to_chat_completion(response, "gpt-5.5")

        self.assertEqual(chat["object"], "chat.completion")
        self.assertEqual(chat["model"], "gpt-5.5")
        self.assertEqual(chat["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(chat["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            chat["usage"],
            {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        )

    def test_wraps_responses_function_call_as_chat_tool_call(self):
        response = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "lookup",
                    "arguments": "{\"q\":\"hi\"}",
                }
            ]
        }

        chat = main._responses_json_to_chat_completion(response, "gpt-5.5")

        message = chat["choices"][0]["message"]
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["id"], "call_123")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(chat["choices"][0]["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
