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


if __name__ == "__main__":
    unittest.main()
