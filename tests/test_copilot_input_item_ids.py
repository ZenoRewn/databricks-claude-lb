"""GHCP connection-bound `input[*].id` 剥离与兜底恢复.

背景（2026-09-09 直连 GHCP Enterprise 上游实测，见
CopilotProxy._strip_input_item_ids 的 docstring）：GHCP 铸造的 Responses item
`id` 是 424~428 字符的签名不透明 blob，绑在服务端一个 "connection" 上。该
connection 消亡后，客户端仍在回放的旧 id 会让整个会话永久被拒：

    {"code":"bad_request","type":"websocket_error",
     "message":"input item ID does not belong to this connection"}

LB 的每一次中途断流都会制造这种 orphaned id，所以我们在转发前一律剥掉。
实测代价为零：上下文（encrypted_content）、工具配对（call_id）、prompt cache
全不受影响。
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import CopilotProxy

# 真实形态：GHCP 的 item id 是 424~428 字符 base64 blob，不是 msg_xxx
GHCP_ITEM_ID = "S/dziB8nkJ6YzEkxlvfUAFEXh5NRPlYMT8H66snz" + "A" * 384


class StripInputItemIdsTests(unittest.TestCase):

    def test_strips_top_level_item_ids_only(self):
        body = {"input": [
            {"type": "reasoning", "id": GHCP_ITEM_ID, "encrypted_content": "E" * 5080,
             "summary": [{"text": "..."}]},
            {"type": "function_call", "id": GHCP_ITEM_ID, "call_id": "call_IyFGBuGe6CAosMWjjzaCCP8X",
             "name": "get_weather", "arguments": '{"city":"Tokyo"}'},
            {"type": "message", "role": "assistant", "id": GHCP_ITEM_ID,
             "content": [{"type": "output_text", "text": "hi"}]},
        ]}
        n = CopilotProxy._strip_input_item_ids(body, "responses")
        self.assertEqual(n, 3)
        for item in body["input"]:
            self.assertNotIn("id", item)

    def test_preserves_encrypted_content_and_call_id(self):
        """实测：id 与 encrypted_content 互相密码学绑定，错配会 400；
        call_id 是工具配对的唯一依据。两者都必须原样保留。"""
        body = {"input": [
            {"type": "reasoning", "id": GHCP_ITEM_ID, "encrypted_content": "E" * 100},
            {"type": "function_call", "id": GHCP_ITEM_ID, "call_id": "call_abc123"},
            {"type": "function_call_output", "call_id": "call_abc123", "output": "{}"},
        ]}
        CopilotProxy._strip_input_item_ids(body, "responses")
        self.assertEqual(body["input"][0]["encrypted_content"], "E" * 100)
        self.assertEqual(body["input"][1]["call_id"], "call_abc123")
        self.assertEqual(body["input"][2]["call_id"], "call_abc123")

    def test_idempotent_second_call_strips_nothing(self):
        """兜底重试依赖这个性质来天然限制为「最多重试一次」。"""
        body = {"input": [{"type": "message", "id": GHCP_ITEM_ID}]}
        self.assertEqual(CopilotProxy._strip_input_item_ids(body, "responses"), 1)
        self.assertEqual(CopilotProxy._strip_input_item_ids(body, "responses"), 0)

    def test_chat_completions_untouched(self):
        """Chat 协议没有 input[*].id，一律返回 0 且不改 body。"""
        body = {"messages": [{"role": "user", "content": "hi"}], "input": [{"id": "x"}]}
        self.assertEqual(CopilotProxy._strip_input_item_ids(body, "chat"), 0)
        self.assertEqual(body["input"][0]["id"], "x")

    def test_tolerates_malformed_input(self):
        for body in ({}, {"input": None}, {"input": "not-a-list"},
                     {"input": [None, 42, "str", {"no_id": 1}]}):
            self.assertEqual(CopilotProxy._strip_input_item_ids(body, "responses"), 0)
        self.assertEqual(CopilotProxy._strip_input_item_ids(None, "responses"), 0)

    def test_does_not_strip_falsy_but_absent_distinction(self):
        """id 为空字符串也算存在过 —— pop 返回 "" 是 falsy，不能因此漏计数。"""
        body = {"input": [{"type": "message", "id": ""}]}
        n = CopilotProxy._strip_input_item_ids(body, "responses")
        # 空 id 本来就不该发给上游，剥掉即可；计数与否不影响正确性，
        # 但 body 里必须不再有 id。
        self.assertNotIn("id", body["input"][0])
        self.assertIn(n, (0, 1))


class OrphanedItemIdErrorDetectionTests(unittest.TestCase):

    def test_matches_upstream_wording_variants(self):
        """上游大小写不稳定：copilot-cli 报 "input item ID"，
        我们生产日志里是 "input item does not belong"。两种都要认。"""
        cases = [
            (400, '{"error":{"code":"bad_request","type":"websocket_error",'
                  '"message":"input item ID does not belong to this connection"}}'),
            (400, "input item does not belong to this connection"),
            (401, "Input Item ID Does Not Belong To This Connection"),
        ]
        for status, text in cases:
            with self.subTest(status=status, text=text[:40]):
                self.assertTrue(CopilotProxy._is_orphaned_item_id_error(status, text))

    def test_ignores_other_400s(self):
        others = [
            "previous_response_id is not supported",
            "Invalid 'input[0].id': string too long. Expected a string with maximum length 64",
            '{"error":{"message":"","code":"invalid_request_body"}}',
        ]
        for text in others:
            with self.subTest(text=text[:40]):
                self.assertFalse(CopilotProxy._is_orphaned_item_id_error(400, text))

    def test_only_400_and_401(self):
        text = "input item ID does not belong to this connection"
        for status in (200, 403, 404, 429, 500, 502, 503):
            with self.subTest(status=status):
                self.assertFalse(CopilotProxy._is_orphaned_item_id_error(status, text))
        self.assertTrue(CopilotProxy._is_orphaned_item_id_error(400, text))
        self.assertTrue(CopilotProxy._is_orphaned_item_id_error(401, text))

    def test_handles_empty_body(self):
        self.assertFalse(CopilotProxy._is_orphaned_item_id_error(400, ""))
        self.assertFalse(CopilotProxy._is_orphaned_item_id_error(400, None))


class StripDoesNotAffectStatefulPinningTests(unittest.TestCase):
    """statefulness 只看 previous_response_id / encrypted_content，不看 id，
    所以剥离 id 绝不能让一个 stateful 请求被误判成无状态而跨账户重放。"""

    def test_encrypted_content_still_marks_request_stateful_after_strip(self):
        body = {"input": [{"type": "reasoning", "id": GHCP_ITEM_ID,
                           "content": [{"encrypted_content": "E" * 50}]}]}
        self.assertTrue(CopilotProxy._request_has_opaque_state(body, "responses"))
        CopilotProxy._strip_input_item_ids(body, "responses")
        self.assertTrue(CopilotProxy._request_has_opaque_state(body, "responses"),
                        "剥 id 后仍必须判定为 stateful")

    def test_item_level_encrypted_content_survives(self):
        body = {"input": [{"type": "reasoning", "id": GHCP_ITEM_ID,
                           "encrypted_content": "E" * 50}]}
        self.assertTrue(CopilotProxy._request_has_opaque_state(body, "responses"))
        CopilotProxy._strip_input_item_ids(body, "responses")
        self.assertTrue(CopilotProxy._request_has_opaque_state(body, "responses"))

    def test_id_alone_never_marked_stateful(self):
        """只有 id、没有 opaque state 的请求不该被钉住 endpoint。"""
        body = {"input": [{"type": "message", "id": GHCP_ITEM_ID}]}
        self.assertFalse(CopilotProxy._request_has_opaque_state(body, "responses"))


class SettingsTests(unittest.TestCase):

    def test_env_switch_defaults_true_and_can_be_disabled(self):
        prev = os.environ.pop("COPILOT_STRIP_INPUT_ITEM_IDS", None)
        try:
            self.assertTrue(main.LBSettings.load().copilot_strip_input_item_ids)
            os.environ["COPILOT_STRIP_INPUT_ITEM_IDS"] = "false"
            self.assertFalse(main.LBSettings.load().copilot_strip_input_item_ids)
            os.environ["COPILOT_STRIP_INPUT_ITEM_IDS"] = "true"
            self.assertTrue(main.LBSettings.load().copilot_strip_input_item_ids)
        finally:
            os.environ.pop("COPILOT_STRIP_INPUT_ITEM_IDS", None)
            if prev is not None:
                os.environ["COPILOT_STRIP_INPUT_ITEM_IDS"] = prev


if __name__ == "__main__":
    unittest.main()


class ProxyForwardPathTests(unittest.IsolatedAsyncioTestCase):
    """转发路径集成：_proxy 必须在真正发出之前剥掉 id，并在上游仍拒绝时兜底重试。

    复用 test_copilot_request_lifecycle 的 _make_proxy 夹具，保证与生产 proxy
    的字段集合一致（新增计数器已同步进那个夹具）。
    """

    def _proxy(self, client):
        import test_copilot_request_lifecycle as fixtures
        return fixtures.CopilotRequestLifecycleTests()._make_proxy(client, threshold=20)

    async def test_non_stream_responses_request_has_ids_stripped_before_send(self):
        import httpx
        from unittest.mock import AsyncMock

        sent = []

        async def capture_post(url, json, headers):
            sent.append(json)
            return httpx.Response(
                200, request=httpx.Request("POST", url),
                json={"output": [], "usage": {}})

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=capture_post)
        proxy, _lb, _ep = self._proxy(client)

        body = {"model": "gpt-test", "input": [
            {"type": "reasoning", "id": GHCP_ITEM_ID, "encrypted_content": "E" * 64},
            {"type": "function_call", "id": GHCP_ITEM_ID, "call_id": "call_keepme"},
        ]}
        await proxy.proxy_responses(body, stream=False)

        self.assertEqual(len(sent), 1)
        forwarded = sent[0]
        for item in forwarded["input"]:
            self.assertNotIn("id", item, "id 必须在发出前就已剥掉")
        self.assertEqual(forwarded["input"][0]["encrypted_content"], "E" * 64)
        self.assertEqual(forwarded["input"][1]["call_id"], "call_keepme")
        self.assertEqual(proxy.input_item_ids_stripped_total, 2)
        self.assertEqual(proxy.input_item_ids_stripped_requests_total, 1)

    async def test_chat_request_is_not_touched(self):
        import httpx
        from unittest.mock import AsyncMock

        sent = []

        async def capture_post(url, json, headers):
            sent.append(json)
            return httpx.Response(200, request=httpx.Request("POST", url),
                                  json={"choices": [], "usage": {}})

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=capture_post)
        proxy, _lb, _ep = self._proxy(client)
        await proxy.proxy_chat_completions(
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            stream=False)
        self.assertEqual(proxy.input_item_ids_stripped_total, 0)
        self.assertEqual(proxy.input_item_ids_stripped_requests_total, 0)

    async def test_disabling_the_switch_forwards_ids_untouched(self):
        import httpx
        from unittest.mock import AsyncMock, patch

        sent = []

        async def capture_post(url, json, headers):
            sent.append(json)
            return httpx.Response(200, request=httpx.Request("POST", url),
                                  json={"output": [], "usage": {}})

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=capture_post)
        proxy, _lb, _ep = self._proxy(client)

        disabled = main.LBSettings.load()
        object.__setattr__(disabled, "copilot_strip_input_item_ids", False)
        with patch.object(main, "LB_SETTINGS", disabled):
            await proxy.proxy_responses(
                {"model": "gpt-test", "input": [{"type": "message", "id": GHCP_ITEM_ID}]},
                stream=False)
        self.assertEqual(sent[0]["input"][0]["id"], GHCP_ITEM_ID,
                         "COPILOT_STRIP_INPUT_ITEM_IDS=false 必须恢复原样透传")
        self.assertEqual(proxy.input_item_ids_stripped_total, 0)

    async def test_non_stream_orphaned_id_rejection_is_recovered_by_retry(self):
        """兜底路径：开关关掉让 id 透传出去，上游返回该 400，
        LB 必须就地剥 id 重发一次并成功。"""
        import httpx
        from unittest.mock import AsyncMock, patch

        sent = []
        ORPHAN = ('{"error":{"code":"bad_request","type":"websocket_error",'
                  '"message":"input item ID does not belong to this connection"}}')

        async def post(url, json, headers):
            # 深拷贝：body 会被就地剥离，不拷会看不出两次发送的差异
            sent.append(copy.deepcopy(json))
            req = httpx.Request("POST", url)
            if len(sent) == 1:
                return httpx.Response(400, request=req, text=ORPHAN,
                                      headers={"content-type": "application/json"})
            return httpx.Response(200, request=req, json={"output": [], "usage": {}})

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=post)
        proxy, _lb, _ep = self._proxy(client)

        disabled = main.LBSettings.load()
        object.__setattr__(disabled, "copilot_strip_input_item_ids", False)
        with patch.object(main, "LB_SETTINGS", disabled):
            resp = await proxy.proxy_responses(
                {"model": "gpt-test",
                 "input": [{"type": "message", "id": GHCP_ITEM_ID}]},
                stream=False)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(sent), 2, "必须重发一次")
        self.assertEqual(sent[0]["input"][0].get("id"), GHCP_ITEM_ID)
        self.assertNotIn("id", sent[1]["input"][0], "重发时 id 必须已被剥掉")
        self.assertEqual(proxy.orphaned_item_id_events.get("detected"), 1)
        self.assertEqual(proxy.orphaned_item_id_events.get("recovered"), 1)
        self.assertNotIn("unrecoverable", proxy.orphaned_item_id_events)

    async def test_orphaned_rejection_without_ids_is_not_masked(self):
        """没有 id 可剥时不能无限重试、也不能把错误吞掉。"""
        import httpx
        from unittest.mock import AsyncMock

        calls = []
        ORPHAN = "input item ID does not belong to this connection"

        async def post(url, json, headers):
            calls.append(json)
            return httpx.Response(400, request=httpx.Request("POST", url),
                                  text=ORPHAN,
                                  headers={"content-type": "application/json"})

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=post)
        proxy, _lb, _ep = self._proxy(client)

        with self.assertRaises(Exception):
            await proxy.proxy_responses(
                {"model": "gpt-test", "input": [{"type": "message"}]}, stream=False)
        self.assertGreaterEqual(proxy.orphaned_item_id_events.get("unrecoverable", 0), 1)
        self.assertEqual(proxy.orphaned_item_id_events.get("recovered", 0), 0)
