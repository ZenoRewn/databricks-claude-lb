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


if __name__ == "__main__":
    unittest.main()
