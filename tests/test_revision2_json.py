"""Client-parser edge regressions, preserving real upstream/downstream wire bytes."""
import unittest

import main
from test_revision2_review import _load  # Installs the unchanged socket helper only.
from protocol_socket_all import run_case


ROOT_FIELDS = ("type", "headers", "metadata", "response", "item", "item_id", "call_id",
               "delta", "text", "summary_index", "content_index", "safety_buffering")
INVALID = [
    ('overflow', '{"type":"response.completed","response":{"id":"s","metadata":{"n":1e400}}}'),
    ('negative_overflow', '{"type":"response.completed","response":{"id":"s","metadata":{"n":-1e400}}}'),
    ('surrogate_id', '{"type":"response.completed","response":{"id":"\\ud800"}}'),
    ('surrogate_nested_array', '{"type":"response.completed","response":{"id":"s","metadata":{"a":[["\\udfff"]]}}}'),
    ('surrogate_key', '{"type":"response.completed","response":{"id":"s","metadata":{"\\ud800":1}}}'),
    ('surrogate_overwritten', '{"type":"response.completed","response":{"id":"s","metadata":{"x":"\\ud800","x":"valid"}}}'),
    ('overflow_overwritten', '{"type":"response.completed","response":{"id":"s","metadata":{"n":1e400,"n":1}}}'),
    ('failed_then_completed', '{"type":"response.failed","type":"response.completed","response":{"id":"s"}}'),
]
VALID = [
    ('finite', '{"type":"response.completed","response":{"id":"s","metadata":{"n":1e308,"m":-1e308,"z":1e-400}}}'),
    ('unicode_pair', '{"type":"response.completed","response":{"id":"\\ud83d\\ude00","metadata":{"s":"汉😀"}}}'),
    ('nested_duplicates', '{"type":"response.completed","response":{"id":"old","id":"s","metadata":{"type":"x","type":"y","x":1,"x":2}}}'),
    ('unknown_root_duplicates', '{"type":"response.completed","unknown":1,"unknown":2,"response":{"id":"s"}}'),
    ('typed_usage_duplicates', '{"type":"response.completed","response":{"id":"s","usage":{"input_tokens":9,"input_tokens":1,"output_tokens":2,"total_tokens":3,"input_tokens_details":{"cached_tokens":7,"cached_tokens":1},"codex_rollout_budget_units":1.5}}}'),
]


class JsonCompatibility(unittest.IsolatedAsyncioTestCase):
    async def socket_case(self, provider, label, payload, valid):
        wire = b"data: " + payload.encode("utf-8") + b"\n\n"
        r = await run_case("revision2_" + label, wire, provider=provider)
        self.assertTrue(r["prefix_byte_identical"], "never rewrite or drop complete upstream frames")
        self.assertEqual(r["endpoint_successes"], int(valid))
        self.assertEqual(r["endpoint_errors"], int(not valid))
        self.assertEqual(r["source_requests"], 1)
        self.assertEqual(r["settlement_calls"], 1)
        for key in ("pool_requests_before_teardown", "business_active_before_teardown", "registry_before_teardown"):
            self.assertEqual(r[key], 0)
        self.assertEqual(main._STREAM_BUFFER_BUDGET.retained, 0)
        if valid:
            self.assertTrue(r["exact_wire"])
            self.assertEqual(r["downstream_error_events"], 0)
        else:
            # Original bad completed bytes are deliberately still on the wire.
            # The LB may append its independent error, never another completion.
            self.assertEqual(r["downstream_completed_events"], 1)
            self.assertGreaterEqual(r["downstream_error_events"], 1)
        if label == "typed_usage_duplicates":
            self.assertEqual(r["recorded_input_tokens"], 1)
            self.assertEqual(r["recorded_output_tokens"], 2)

    async def test_parser_rejections_both_responses_providers_preserve_wire(self):
        for provider in ("copilot", "azure"):
            for label, payload in INVALID:
                with self.subTest(provider=provider, label=label):
                    await self.socket_case(provider, label, payload, False)

    async def test_finite_unicode_nested_value_and_usage_controls_both_providers(self):
        for provider in ("copilot", "azure"):
            for label, payload in VALID:
                with self.subTest(provider=provider, label=label):
                    await self.socket_case(provider, label, payload, True)

    def test_duplicate_handling_is_root_typed_only(self):
        for key in ROOT_FIELDS:
            if key == "type":
                payload = '{"type":"response.completed","type":"response.completed","response":{"id":"s"}}'
            elif key == "response":
                payload = '{"type":"response.completed","response":{"id":"s"},"response":{"id":"s"}}'
            else:
                payload = '{"type":"response.completed","response":{"id":"s"},"' + key + '":null,"' + key + '":null}'
            with self.subTest(key=key):
                observation = main._SSEObservation("responses")
                observation.observe(b"data: " + payload.encode() + b"\n\n")
                self.assertIsNone(observation.terminal)
        for label, payload in VALID:
            with self.subTest(label=label):
                observation = main._SSEObservation("responses")
                observation.observe(b"data: " + payload.encode() + b"\n\n")
                self.assertEqual(observation.terminal, "completed")
        # No new typed Responses duplicate policy imposed on the other APIs.
        for api, terminal in (("chat", "unused"), ("messages", "message_stop")):
            block = 'data: {"type":"x","type":"' + terminal + '"}\n\n'
            _, parsed, _ = main._parse_sse_event_block(block)
            self.assertEqual(parsed["type"], terminal)
