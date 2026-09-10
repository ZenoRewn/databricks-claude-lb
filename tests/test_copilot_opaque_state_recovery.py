"""上游拒绝 opaque state 时的有界恢复阶梯。

背景（2026-09-10 生产现场 + 直连 GHCP Enterprise 实测）：

生产 `/metrics` 显示 `orphaned_item_id_events{detected}=12, recovered=0,
unrecoverable=12`，而 `upstream_401_total{scope="request"}=6`。恒 2:1 不是巧合：

  1. `_proxy` 已在发送前主动剥掉 `input[*].id`，所以事后再调那个幂等 helper 必然
     返 0 → `recovered` 结构性不可达，永远走 `unrecoverable`
  2. 记完 `unrecoverable` 之后**继续掉进 401 auth repair** —— 强制刷一次 session
     token（注定无用：实测 token 轮换不使 opaque state 失效，见 TROUBLESHOOTING §16）
     再原样重发，于是同一个客户端请求把每个 orphan 计数器都记了两遍

`token_refresh_total=7 = 6 次浪费的强制刷新 + 1 次 warmup` 独立佐证唯一请求数是 6。

恢复为什么可行（直连 GHCP，`gpt-5.4-mini`，`store:false` + function tool 回路）：

  | 组 | 处理 | 结果 |
  |---|---|---|
  | 剥 id + 保留 encrypted_content（线上现行） | 200，答出暗号 |
  | 剥 id + 整条删 reasoning item            | 200，答出暗号 |
  | 剥 id + 只删 encrypted_content           | 200，答出暗号 |
  | 剥 id + 篡改 encrypted_content           | **400，另一个错误** |

  - `call_id` 回路在删除后仍正确配对
  - token / prompt cache 代价为零：`input_tokens=1637 / cached_tokens=1280` 三组一致
  - 篡改组的报错是 `invalid_request_body` "could not be verified"，**不是**
    "does not belong to this connection" → 上游是两级校验（先解密解析、再校验归属），
    而第二类我们此前完全没兜底

诚实边界：`does not belong to this connection` 需要另一个账户/连接铸造的 blob，
单账号复现不出。所以「删 blob 能救那类拒绝」是推断；已实测的是删后请求形态合法、
上下文保真，且另一类 opaque-state 拒绝确定被它救回。故 rung 2 带 kill switch
`COPILOT_OPAQUE_STATE_RECOVERY`。
"""
import asyncio
import copy
import os
import pathlib
import sys
import unittest
import unittest.mock
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import CopilotProxy

import test_copilot_request_lifecycle as fixtures

JSON_CT = {"content-type": "application/json"}
TEXT_CT = {"content-type": "text/plain"}

# 2026-09-10 用两个真实 GHCP 账户（enterprise + business 两套部署）跨账户回放实测到的
# 两种原文，**双向 × 3 次重复 = 24/24 确定性**：
#   带 item id 回放      → 401 "input item ID does not belong to this connection"
#   只剥 id（LB 现行）   → 401 "input item does not belong to this connection"
# 后者与生产日志逐字一致 —— 因为线上默认就会主动剥 id。两种大小写都要匹配到，
# 所以 marker 只取不含 ID/id 的稳定子串。
ORPHAN_401 = ('{"error":{"message":"input item does not belong to this connection",'
              '"code":""}}')
ORPHAN_401_WITH_ID = ('{"error":{"message":"input item ID does not belong to this '
                      'connection","code":""}}')
# copilot-cli 侧报的是 400（github/copilot-cli#2147）
ORPHAN_400 = ('{"error":{"code":"bad_request","type":"websocket_error",'
              '"message":"input item ID does not belong to this connection"}}')
# 2026-09-10 篡改 encrypted_content 实测拿到的原文
UNVERIFIABLE_400 = ('{"error":{"message":"The encrypted content Zb+H...LQ== could not be '
                    'verified. Reason: Encrypted content could not be decrypted or parsed.",'
                    '"code":"invalid_request_body"}}')
PLAIN_401 = '{"error":{"message":"Bad credentials","code":"unauthorized"}}'

GHCP_ITEM_ID = "X" * 424  # 实测长度 420~428 的签名不透明 blob


def stateful_body(*, with_id=False):
    """携带 opaque state（reasoning encrypted_content）的 Responses 请求。"""
    item = {"type": "reasoning", "encrypted_content": "E" * 256,
            "summary": [{"type": "summary_text", "text": "thinking"}]}
    if with_id:
        item["id"] = GHCP_ITEM_ID
    return {"model": "gpt-test", "input": [item]}


def stateless_body():
    return {"model": "gpt-test",
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "hi"}]}]}


COMPLEX_BODY = {"model": "gpt-test", "input": [
    {"type": "message", "role": "user",
     "content": [{"type": "input_text", "text": "check Oslo weather"}]},
    {"type": "reasoning", "encrypted_content": "E" * 64,
     "summary": [{"type": "summary_text", "text": "plan"}]},
    {"type": "function_call", "call_id": "call_1", "name": "get_weather",
     "arguments": '{"city":"Oslo"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": '{"temp_c":-3}'},
]}


class _PinnedSettings:
    """把用例依赖到的开关钉成显式值。

    测试结论不能随「跑测试时 shell 里恰好有什么环境变量」变化 —— 否则
    `COPILOT_OPAQUE_STATE_RECOVERY=false pytest` 会让一堆与开关无关的断言一起红，
    分不清是真回归还是环境噪音。要验开关本身的用例在方法内再局部覆盖即可（
    `patch.object` 可以嵌套）。
    """

    SETTINGS_OVERRIDES = {"copilot_opaque_state_recovery": True}

    def setUp(self):
        super().setUp()
        snapshot = main.LBSettings.load()
        for key, value in self.SETTINGS_OVERRIDES.items():
            object.__setattr__(snapshot, key, value)
        patcher = patch.object(main, "LB_SETTINGS", snapshot)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = snapshot


def _settings(**overrides):
    """基于当前 LB_SETTINGS 派生一份覆盖过的快照，供局部 patch 使用。"""
    snapshot = main.LBSettings.load()
    for key, value in overrides.items():
        object.__setattr__(snapshot, key, value)
    return snapshot


def _make_proxy(threshold=5):
    """复用生产字段集合一致的夹具；token 交换与 header 构建都 mock，绝不打网络。"""
    proxy, lb, ep = fixtures.CopilotRequestLifecycleTests()._make_proxy(
        unittest.mock.Mock(), threshold=threshold)
    forced = []

    async def fake_token(endpoint, force=False):
        forced.append(bool(force))
        return "fresh-session-token"

    proxy.get_session_token = fake_token
    return proxy, lb, ep, forced


class _RecordingStreamClient(fixtures._StreamClient):
    """记录每次 send 实际带出去的 body —— 恢复阶梯改了什么必须看得见。"""

    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.sent = []

    def build_request(self, method, url, json, headers):
        self.sent.append(copy.deepcopy(json))
        return super().build_request(method, url, json=json, headers=headers)


def _err_stream(status, text, headers=None):
    return fixtures._ErrorStreamResponse(status, text.encode("utf-8"),
                                         headers=dict(headers or TEXT_CT))


def _ok_stream():
    return fixtures._StreamResponse([
        b'event: response.created\ndata: {"id":"r_1"}\n\n',
        b'event: response.completed\ndata: {"type":"response.completed","response":'
        b'{"id":"r_1","usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4}}}\n\n',
    ])


# ---------------------------------------------------------------- 纯函数判据


class DropReasoningEncryptedContentTests(unittest.TestCase):
    """rung 2 的载荷改写：只删该删的，其余一律不动。"""

    def test_drops_only_reasoning_encrypted_content(self):
        body = copy.deepcopy(COMPLEX_BODY)
        self.assertEqual(CopilotProxy._drop_reasoning_encrypted_content(body, "responses"), 1)
        items = body["input"]
        self.assertNotIn("encrypted_content", items[1], "reasoning 的 blob 必须删掉")
        self.assertEqual(items[1]["type"], "reasoning", "item 本体保留（实测保留即可 200）")
        self.assertEqual(items[1]["summary"][0]["text"], "plan", "summary 是明文，不能删")
        self.assertEqual(items[0]["content"][0]["text"], "check Oslo weather",
                         "message 历史必须原样保留")
        self.assertEqual(items[2]["call_id"], "call_1")
        self.assertEqual(items[3]["call_id"], "call_1")
        self.assertEqual(items[3]["output"], '{"temp_c":-3}', "工具结果不能删")

    def test_is_idempotent(self):
        body = copy.deepcopy(COMPLEX_BODY)
        self.assertEqual(CopilotProxy._drop_reasoning_encrypted_content(body, "responses"), 1)
        self.assertEqual(CopilotProxy._drop_reasoning_encrypted_content(body, "responses"), 0,
                         "第二次必须返 0，否则预算判据会失真")

    def test_chat_protocol_and_malformed_bodies_return_zero(self):
        self.assertEqual(CopilotProxy._drop_reasoning_encrypted_content(
            {"messages": [{"role": "user", "content": "hi"}]}, "chat"), 0)
        self.assertEqual(CopilotProxy._drop_reasoning_encrypted_content(
            {"input": "not-a-list"}, "responses"), 0)
        self.assertEqual(CopilotProxy._drop_reasoning_encrypted_content(
            "not-a-dict", "responses"), 0)

    def test_does_not_touch_non_reasoning_encrypted_content(self):
        """只有 reasoning item 的 blob 在 rung 2 的射程内。"""
        body = {"model": "gpt-test", "input": [
            {"type": "message", "encrypted_content": "keep-me"}]}
        self.assertEqual(CopilotProxy._drop_reasoning_encrypted_content(body, "responses"), 0)
        self.assertEqual(body["input"][0]["encrypted_content"], "keep-me")


class ClassifyOpaqueStateRejectionTests(unittest.TestCase):
    """两级校验 → 两种 kind；其余一律 None。"""

    def test_orphaned_ownership_rejection(self):
        for status in (400, 401):
            for text in (ORPHAN_400, ORPHAN_401, ORPHAN_401_WITH_ID,
                         "input item ID does not belong to this connection"):
                with self.subTest(status=status, text=text[:40]):
                    self.assertEqual(
                        CopilotProxy._classify_opaque_state_rejection(status, text),
                        "orphaned_id")

    def test_unverifiable_content_rejection(self):
        self.assertEqual(
            CopilotProxy._classify_opaque_state_rejection(400, UNVERIFIABLE_400),
            "unverifiable_content")

    def test_unrelated_verification_failures_are_not_opaque_state(self):
        """判据必须同时命中「主题是 encrypted content」与「校验失败」两半。

        只匹配 "could not be verified" 会误吞别的 400，代价很实：无理由删掉用户的
        推理 blob，还把真实原因（如凭证问题）改写成 orphaned_conversation_state 藏起来。
        """
        for text in (
            '{"error":{"message":"The provided API key could not be verified.","code":"invalid_api_key"}}',
            '{"error":{"message":"Signature could not be verified","code":"bad_request"}}',
            '{"error":{"message":"Webhook payload could not be decrypted or parsed","code":"bad_request"}}',
            # 只有主题、没有校验失败措辞 —— 也不该命中
            '{"error":{"message":"encrypted content is too long","code":"bad_request"}}',
        ):
            with self.subTest(text=text[:60]):
                self.assertIsNone(
                    CopilotProxy._classify_opaque_state_rejection(400, text))

    def test_snake_case_subject_also_matches(self):
        self.assertEqual(
            CopilotProxy._classify_opaque_state_rejection(
                400, '{"error":{"message":"input[0].encrypted_content could not be verified"}}'),
            "unverifiable_content")

    def test_unrelated_rejections_are_none(self):
        for status, text in ((401, PLAIN_401), (400, '{"error":{"message":"bad schema"}}'),
                             (403, ORPHAN_401), (429, ORPHAN_401), (500, ORPHAN_401),
                             (200, ORPHAN_401)):
            with self.subTest(status=status):
                self.assertIsNone(
                    CopilotProxy._classify_opaque_state_rejection(status, text))

    def test_legacy_orphan_predicate_is_unchanged(self):
        """`_is_orphaned_item_id_error` 是既有测试与文档引用的入口，不得改语义。"""
        self.assertTrue(CopilotProxy._is_orphaned_item_id_error(400, ORPHAN_400))
        self.assertTrue(CopilotProxy._is_orphaned_item_id_error(401, ORPHAN_401))
        self.assertFalse(CopilotProxy._is_orphaned_item_id_error(400, UNVERIFIABLE_400))
        self.assertFalse(CopilotProxy._is_orphaned_item_id_error(500, ORPHAN_401))


class CrossAccountReplayShapeTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):
    """把跨账户实测的**真实形态**固化成回归。

    2026-09-10 实测（两个真实 GHCP 账户，enterprise + business，双向 × 3 次 = 24/24）：

    | 处理 | 上游 |
    |---|---|
    | 原样回放（id + blob） | 401 `input item ID does not belong to this connection` |
    | 只剥 id（LB 生产现行） | 401 `input item does not belong to this connection` |
    | 剥 id + 删 blob（rung 2） | **200，且答出上一轮种下的暗号** |
    | 剥 id + 整条删 reasoning | 200，同上 |

    所以「rung 2 能救回归属类拒绝」不再是推断。这里用两种真实原文各跑一遍完整路径，
    确认 LB 的分类 → 阶梯 → 恢复链条对**实测到的字节**成立，而不只是对我构造的样例。
    """

    def _client(self, first_error_text):
        sent = []

        async def post(url, json, headers):
            sent.append(copy.deepcopy(json))
            req = httpx.Request("POST", url)
            # 第一次按实测原文拒绝；blob 被删掉之后才放行（与实测一致）
            has_blob = any(it.get("encrypted_content")
                           for it in (json.get("input") or [])
                           if isinstance(it, dict))
            if has_blob:
                return httpx.Response(401, request=req, text=first_error_text,
                                      headers=dict(JSON_CT))
            return httpx.Response(200, request=req, json={"output": [], "usage": {}})

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=post)
        return client, sent

    async def test_both_measured_error_texts_recover_via_rung2(self):
        for label, text in (("剥 id 后的原文（生产形态）", ORPHAN_401),
                            ("带 id 的原文", ORPHAN_401_WITH_ID)):
            with self.subTest(variant=label):
                proxy, lb, ep, forced = _make_proxy(threshold=99)
                proxy.client, sent = self._client(text)
                resp = await proxy.proxy_responses(stateful_body(), stream=False)

                self.assertEqual(resp.status_code, 200, "rung 2 必须把它救回来")
                self.assertEqual(len(sent), 2)
                self.assertIn("encrypted_content", sent[0]["input"][0])
                self.assertNotIn("encrypted_content", sent[1]["input"][0])
                self.assertEqual(proxy.opaque_state_rejections.get("orphaned_id"), 1)
                self.assertEqual(proxy.opaque_state_recovery.get("succeeded"), 1)
                self.assertEqual(forced.count(True), 0,
                                 "跨账户拒绝与凭证无关，不该刷 token")
                self.assertEqual(ep.consecutive_errors, 0, "账户不得被熔断")
                self.assertEqual(ep.active_requests, 0)


# ---------------------------------------------------------------- 非流式


class NonStreamRecoveryTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):

    def _client(self, responses):
        """responses: [(status, text)] 或 ("ok", None)；按序返回。"""
        sent = []

        async def post(url, json, headers):
            sent.append(copy.deepcopy(json))
            idx = min(len(sent) - 1, len(responses) - 1)
            status, text = responses[idx]
            req = httpx.Request("POST", url)
            if status == 200:
                return httpx.Response(200, request=req, json={"output": [], "usage": {}})
            return httpx.Response(status, request=req, text=text, headers=dict(JSON_CT))

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=post)
        return client, sent

    async def test_orphan_401_no_longer_wastes_a_forced_token_exchange(self):
        """核心回归：这是把生产 12/12-vs-6 与 token_refresh=7 一起解释掉的那条。

        改动前：记完 unrecoverable 继续掉进 auth repair → 1 次强制刷新 + 原样重发，
        每个计数器 ×2。改动后：那一次上游调用改成真正的恢复尝试，强制刷新为 0。
        """
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(401, ORPHAN_401)])
        with self.assertRaises(Exception):
            await proxy.proxy_responses(stateful_body(), stream=False)

        self.assertEqual(forced.count(True), 0,
                         "opaque-state 拒绝不得触发强制 token 交换（注定无用）")
        self.assertEqual(len(sent), 2, "1 次原始 + 1 次 rung 2 恢复尝试")
        self.assertIn("encrypted_content", sent[0]["input"][0])
        self.assertNotIn("encrypted_content", sent[1]["input"][0],
                         "重发时 blob 必须已删掉")
        self.assertEqual(proxy.orphaned_item_id_events.get("detected"), 2,
                         "两次上游拒绝就是两次 detected 事件")
        self.assertEqual(proxy.opaque_state_requests_total, 1,
                         "唯一受影响请求数才是运维该读的口径")
        self.assertEqual(proxy.opaque_state_rejections.get("orphaned_id"), 2)
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("succeeded", 0), 0)
        self.assertEqual(ep.consecutive_errors, 0, "stateful 401 仍中性，账户不熔断")
        self.assertEqual(ep.active_requests, 0, "lease 必须归零")

    async def test_rung2_recovers_and_counts_success_only_after_a_real_response(self):
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(401, ORPHAN_401), (200, None)])
        resp = await proxy.proxy_responses(stateful_body(), stream=False)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(sent), 2)
        self.assertNotIn("encrypted_content", sent[1]["input"][0])
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("succeeded"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted", 0), 0)
        self.assertNotIn("unrecoverable", proxy.orphaned_item_id_events,
                         "恢复成功就不该记 unrecoverable")
        self.assertEqual(forced.count(True), 0)
        self.assertEqual(ep.active_requests, 0)

    async def test_unverifiable_content_rejection_is_also_recovered(self):
        """篡改/截断的 blob 是实测确定能被 rung 2 救回的那一类。"""
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(400, UNVERIFIABLE_400), (200, None)])
        resp = await proxy.proxy_responses(stateful_body(), stream=False)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(proxy.opaque_state_rejections.get("unverifiable_content"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("succeeded"), 1)
        self.assertEqual(proxy.opaque_state_requests_total, 1)

    async def test_ladder_walks_id_then_blob_within_budget(self):
        """两级都有料时依次走完，总上游调用受显式预算约束。"""
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(401, ORPHAN_401)])
        with patch.object(main, "LB_SETTINGS",
                          _settings(copilot_strip_input_item_ids=False,
                                    copilot_opaque_state_recovery=True)):
            with self.assertRaises(Exception):
                await proxy.proxy_responses(stateful_body(with_id=True), stream=False)

        self.assertEqual(len(sent), 3, "原始 + rung1(剥 id) + rung2(删 blob)，不得更多")
        self.assertEqual(sent[0]["input"][0].get("id"), GHCP_ITEM_ID)
        self.assertNotIn("id", sent[1]["input"][0])
        self.assertIn("encrypted_content", sent[1]["input"][0], "rung1 只碰 id")
        self.assertNotIn("encrypted_content", sent[2]["input"][0])
        self.assertEqual(proxy.orphaned_item_id_events.get("recovered"), 1,
                         "真剥到 id 才是「还有别的 id 通道」的金丝雀")
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 2)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)
        self.assertEqual(proxy.orphaned_item_id_events.get("unrecoverable"), 1,
                         "unrecoverable 只在阶梯彻底走完后记一次")

    async def test_nothing_to_recover_exhausts_immediately(self):
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(400, ORPHAN_400)])
        with self.assertRaises(Exception):
            await proxy.proxy_responses(stateless_body(), stream=False)

        self.assertEqual(len(sent), 1, "无料可恢复时不得凭空重发")
        self.assertEqual(proxy.opaque_state_recovery.get("attempted", 0), 0)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)
        self.assertEqual(proxy.orphaned_item_id_events.get("unrecoverable"), 1)

    async def test_exhausted_error_tells_the_client_to_rebuild(self):
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(401, ORPHAN_401)])
        with self.assertRaises(main.HTTPException) as cm:
            await proxy.proxy_responses(stateful_body(), stream=False)

        detail = cm.exception.detail
        self.assertEqual(detail["error"]["code"], "orphaned_conversation_state")
        self.assertIn("conversation state", detail["error"]["message"].lower())
        self.assertIn("orphaned_id", detail["error"]["message"])
        self.assertEqual(detail["error"]["opaque_state_kind"], "orphaned_id")
        # 2026-09-10 实测 codex-cli 0.145.0：LB 返 401/403/409/422 客户端都重试 6 次
        # （每次重试都跑一整条阶梯 = 2 次 GHCP 调用），返 400 只打 1 次就放弃，而且是
        # 唯一会把错误体逐字显示给用户的状态码。这条拒绝确定不可重试，所以改 400。
        self.assertEqual(cm.exception.status_code, 400,
                         "阶梯耗尽必须让客户端立刻停手，而不是重试 6 次")

    async def test_exhausted_400_must_not_trigger_azure_fallback(self):
        """状态码改成 400 之后不能被路由层当成「Copilot 不支持该模型」而 fallback。

        `_route_openai_*` 只在 `_UnsupportedModelError` 或 HTTPException 404/503 时
        fallback。400 不在其中 —— 这是有意的：会话状态坏掉换 provider 也救不回，
        Azure 同样解不开 GHCP 的 blob。
        """
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(401, ORPHAN_401)])
        with self.assertRaises(main.HTTPException) as cm:
            await proxy.proxy_responses(stateful_body(), stream=False)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertNotIn(cm.exception.status_code, (404, 503),
                         "404/503 会触发 Azure fallback")
        self.assertNotIsInstance(cm.exception, main._UnsupportedModelError)

    async def test_kill_switch_disables_rung2_only(self):
        proxy, lb, ep, forced = _make_proxy()
        proxy.client, sent = self._client([(401, ORPHAN_401)])
        with patch.object(main, "LB_SETTINGS",
                          _settings(copilot_opaque_state_recovery=False)):
            with self.assertRaises(Exception):
                await proxy.proxy_responses(stateful_body(), stream=False)

        self.assertEqual(len(sent), 1, "rung 2 关掉后不得重发")
        self.assertEqual(proxy.opaque_state_recovery.get("attempted", 0), 0)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)
        self.assertEqual(forced.count(True), 0,
                         "即便关掉恢复，也不该退回去浪费 token 交换")


# ---------------------------------------------------------------- 流式


class StreamRecoveryTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):

    async def _drain(self, proxy, body):
        res = await proxy.proxy_responses(body, stream=True)
        chunks = []
        async for chunk in res.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        return b"".join(chunks)

    async def test_orphan_401_no_longer_wastes_a_forced_token_exchange(self):
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([_err_stream(401, ORPHAN_401) for _ in range(4)])
        proxy.client = client
        out = await self._drain(proxy, stateful_body())

        self.assertEqual(forced.count(True), 0)
        self.assertEqual(len(client.sent), 2, "1 次原始 + 1 次 rung 2 恢复尝试")
        self.assertNotIn("encrypted_content", client.sent[1]["input"][0])
        self.assertEqual(proxy.opaque_state_requests_total, 1)
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)
        self.assertEqual(ep.consecutive_errors, 0)
        self.assertEqual(ep.active_requests, 0)
        self.assertIn(b"orphaned_conversation_state", out)
        self.assertNotIn(b"[DONE]", out,
                         "Responses 流永远不能出现 [DONE]（Codex serde_json 会报错）")

    async def test_rung2_recovers_and_success_needs_a_terminal_event(self):
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([_err_stream(401, ORPHAN_401), _ok_stream()])
        proxy.client = client
        out = await self._drain(proxy, stateful_body())

        self.assertIn(b"response.completed", out)
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("succeeded"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted", 0), 0)
        self.assertEqual(ep.active_requests, 0)

    async def test_upstream_accepting_the_retry_is_not_success_without_completion(self):
        """上游 200 但流被截断 → 不能记 succeeded（文档 §9 的硬要求）。"""
        proxy, lb, ep, forced = _make_proxy()
        truncated = fixtures._StreamResponse([
            b'event: response.created\ndata: {"id":"r_1"}\n\n',
        ])
        proxy.client = _RecordingStreamClient([_err_stream(401, ORPHAN_401), truncated])
        await self._drain(proxy, stateful_body())

        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("succeeded", 0), 0,
                         "上游只是接受了请求，不等于最终有效完成")
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted", 0), 0,
                         "上游收下了改写后的请求，阶梯并没有走完")
        self.assertEqual(proxy.stream_truncated_no_completion_total, 1)
        # 这就是 attempted - succeeded - exhausted 的那一桶：恢复被接受但流断了。
        # 运维读 succeeded/attempted 时要知道分母里含这一类，具体量看 truncation 指标。

    async def test_unverifiable_content_rejection_is_also_recovered(self):
        proxy, lb, ep, forced = _make_proxy()
        proxy.client = _RecordingStreamClient([
            _err_stream(400, UNVERIFIABLE_400, JSON_CT), _ok_stream()])
        await self._drain(proxy, stateful_body())

        self.assertEqual(proxy.opaque_state_rejections.get("unverifiable_content"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("succeeded"), 1)

    async def test_no_recovery_after_any_chunk_was_committed(self):
        """已提交 SSE 帧之后一律不重放（RESILIENCE.md 的 POST replay 约束）。"""
        proxy, lb, ep, forced = _make_proxy()
        partial = fixtures._StreamResponse([
            b'event: response.created\ndata: {"id":"r_1"}\n\n',
        ])
        proxy.client = _RecordingStreamClient([partial, _ok_stream()])
        await self._drain(proxy, stateful_body())

        self.assertEqual(proxy.opaque_state_requests_total, 0)
        self.assertEqual(proxy.opaque_state_recovery.get("attempted", 0), 0)
        self.assertEqual(ep.active_requests, 0)

    async def test_kill_switch_disables_rung2_only(self):
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([_err_stream(401, ORPHAN_401) for _ in range(3)])
        proxy.client = client
        with patch.object(main, "LB_SETTINGS",
                          _settings(copilot_opaque_state_recovery=False)):
            out = await self._drain(proxy, stateful_body())

        self.assertEqual(len(client.sent), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)
        self.assertEqual(forced.count(True), 0)
        self.assertIn(b"orphaned_conversation_state", out)


class RetryBudgetFloorTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):
    """attempt 预算耗尽时，客户端必须拿到明确终端，绝不能收到空流。

    流式 `for attempt in range(max_retries)` 里有两类 `continue`：

      - **端点切换类**（5xx / PoolTimeout / 网络错误）都自带 `attempt < max_retries - 1`
        守卫，所以第 3 轮一定落到终端分支
      - **原地修复类**（恢复 rung、401 auth repair）改动前上限是 2（剥离幂等 + auth
        限死在 `attempt == 0`），刚好留一轮收尾

`_OpaqueStateRecovery` 把 rung 提到 2 级、auth repair 与 attempt 解耦之后，原地修复
    最多能吃 3 轮，**生成器会在 for 循环后直接结束**（循环体就是生成器体的最后一段），
    下游拿到 HTTP 200 + 0 字节。那正是第 14 节里会毒化会话的「无终端事件断流」。

    不变量：**没有下一轮可以消费修好的请求时，就不要做原地修复**（与端点切换类同一条
    约定），外加一层循环后兜底终端。
    """

    async def _drain(self, proxy, body):
        res = await proxy.proxy_responses(body, stream=True)
        return b"".join([c if isinstance(c, bytes) else c.encode()
                         async for c in res.body_iterator])

    async def test_three_in_place_repairs_still_emit_a_terminal(self):
        """rung1 + rung2 + auth repair 想吃掉全部 3 轮 —— 必须仍有终端事件。"""
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([
            _err_stream(401, ORPHAN_401),          # 想触发 rung 1（剥 id）
            _err_stream(401, ORPHAN_401),          # 想触发 rung 2（删 blob）
            _err_stream(401, PLAIN_401, JSON_CT),  # 想触发 auth repair
            _err_stream(401, PLAIN_401, JSON_CT),
        ])
        proxy.client = client
        with patch.object(main, "LB_SETTINGS",
                          _settings(copilot_strip_input_item_ids=False,
                                    copilot_opaque_state_recovery=True)):
            out = await self._drain(proxy, stateful_body(with_id=True))

        self.assertGreater(len(out), 0, "下游绝不能收到空流（HTTP 200 + 0 字节）")
        self.assertIn(b"response.failed", out, "Responses 流必须以合法终端事件收尾")
        self.assertNotIn(b"[DONE]", out)
        self.assertEqual(ep.active_requests, 0)
        self.assertLessEqual(len(client.sent), 3, "上游调用不得超过 attempt 预算")
        self.assertEqual(proxy.stream_retry_budget_exhausted_total, 0,
                         "不变量成立时金丝雀必须恒 0（非 0 = 客户端本会收到空流）")

    async def test_endpoint_retry_then_recovery_still_emits_a_terminal(self):
        """混合场景：一次端点重试 + 两级恢复想吃满 3 轮 —— 仍必须有终端。

        429 无 Retry-After → `_retry_rejected_response` 允许换端点重试（单端点也会
        重新选中自己），吃掉 attempt 0；随后带 id 的 orphan 让 rung1、rung2 各想吃一轮。
        """
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([
            _err_stream(429, '{"error":{"message":"slow down"}}', JSON_CT),
            _err_stream(401, ORPHAN_401),
            _err_stream(401, ORPHAN_401),
            _err_stream(401, ORPHAN_401),
        ])
        proxy.client = client
        with patch.object(main.asyncio, "sleep", new=AsyncMock()), \
             patch.object(main, "LB_SETTINGS",
                          _settings(copilot_strip_input_item_ids=False,
                                    copilot_opaque_state_recovery=True)):
            out = await self._drain(proxy, stateful_body(with_id=True))

        self.assertGreater(len(out), 0, "下游绝不能收到空流")
        self.assertIn(b"response.failed", out)
        self.assertEqual(ep.active_requests, 0)
        self.assertLessEqual(len(client.sent), 3)

    async def test_invariant_survives_a_larger_rung_budget(self):
        """把 MAX_RUNGS 调高（最可能的未来改动）也不能让循环静默耗尽。

        守的是不变量本身而不是「2 这个数字」：原地修复只在还剩下一轮时才做。
        """
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([_err_stream(401, ORPHAN_401) for _ in range(8)])
        proxy.client = client
        with patch.object(main._OpaqueStateRecovery, "MAX_RUNGS", 5), \
             patch.object(main, "LB_SETTINGS",
                          _settings(copilot_strip_input_item_ids=False,
                                    copilot_opaque_state_recovery=True)):
            out = await self._drain(proxy, stateful_body(with_id=True))

        self.assertGreater(len(out), 0)
        self.assertIn(b"response.failed", out)
        self.assertLessEqual(len(client.sent), 3, "attempt 预算仍是硬上限")
        self.assertEqual(ep.active_requests, 0)
        self.assertEqual(proxy.stream_retry_budget_exhausted_total, 0)

    async def test_pure_recovery_path_is_unaffected(self):
        """最常见场景（只有恢复、没有别的重试）行为不变：2 次上游 + 结构化错误。"""
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([_err_stream(401, ORPHAN_401) for _ in range(3)])
        proxy.client = client
        out = await self._drain(proxy, stateful_body())

        self.assertEqual(len(client.sent), 2)
        self.assertIn(b"orphaned_conversation_state", out)
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)


class TerminalShapeTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):
    """两种协议的终端帧形状不同，恢复阶梯与兜底都不能弄错。

    Responses 流出现 `data: [DONE]` 会让 Codex 的 serde_json 报
    `error decoding response body`；Chat 流缺 `[DONE]` 则客户端一直等。
    """

    async def _drain(self, proxy, coro):
        res = await coro
        return b"".join([c if isinstance(c, bytes) else c.encode()
                         async for c in res.body_iterator])

    async def test_chat_stream_exhausted_terminal_has_done(self):
        proxy, lb, ep, forced = _make_proxy()
        proxy.client = _RecordingStreamClient([_err_stream(401, ORPHAN_401) for _ in range(3)])
        out = await self._drain(proxy, proxy.proxy_chat_completions(
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            stream=True))
        self.assertIn(b"orphaned_conversation_state", out)
        self.assertIn(b"data: [DONE]", out, "Chat 流必须以 [DONE] 收尾")

    async def test_responses_stream_exhausted_terminal_has_no_done(self):
        proxy, lb, ep, forced = _make_proxy()
        proxy.client = _RecordingStreamClient([_err_stream(401, ORPHAN_401) for _ in range(3)])
        out = await self._drain(proxy, proxy.proxy_responses(stateful_body(), stream=True))
        self.assertIn(b"orphaned_conversation_state", out)
        self.assertIn(b"response.failed", out)
        self.assertNotIn(b"[DONE]", out)

    def test_retry_budget_floor_frame_is_wellformed_for_both_protocols(self):
        """循环后兜底终端按不变量不可达，但它产出的帧形状必须现在就对。"""
        responses = main._sse_terminal_error(
            "responses", "retry_budget_exhausted", "budget spent").decode()
        self.assertIn('"type": "response.failed"', responses)
        self.assertIn("retry_budget_exhausted", responses)
        self.assertNotIn("[DONE]", responses)

        chat = main._sse_terminal_error(
            "chat", "retry_budget_exhausted", "budget spent").decode()
        self.assertIn("retry_budget_exhausted", chat)
        self.assertIn("data: [DONE]", chat)


# ---------------------------------------------------------------- auth repair 隔离


class AuthRepairIsolationTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):
    """把 opaque-state 拒绝挪出 auth repair，绝不能顺手把 401 自愈关掉。"""

    async def test_genuine_401_still_gets_its_auth_repair_buffered(self):
        proxy, lb, ep, forced = _make_proxy()
        sent = []

        async def post(url, json, headers):
            sent.append(1)
            return httpx.Response(401, request=httpx.Request("POST", url),
                                  text=PLAIN_401, headers=dict(JSON_CT))

        proxy.client.post = AsyncMock(side_effect=post)
        with self.assertRaises(Exception):
            await proxy.proxy_responses(stateless_body(), stream=False)

        self.assertEqual(forced.count(True), 1, "真凭证 401 仍要刷一次 token")
        self.assertEqual(len(sent), 2)
        self.assertEqual(ep.consecutive_errors, 1, "无状态 401 仍计入 endpoint 健康度")
        self.assertEqual(proxy.opaque_state_requests_total, 0)

    async def test_genuine_401_still_gets_its_auth_repair_streaming(self):
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([_err_stream(401, PLAIN_401, JSON_CT)
                                         for _ in range(3)])
        proxy.client = client
        res = await proxy.proxy_responses(stateless_body(), stream=True)
        async for _chunk in res.body_iterator:
            pass

        self.assertEqual(forced.count(True), 1)
        self.assertEqual(len(client.sent), 2)
        self.assertEqual(ep.consecutive_errors, 1)

    async def test_auth_repair_budget_survives_a_recovery_rung(self):
        """上一轮 TROUBLESHOOTING §15 记为「已知次要交互，未修」的那条。

        恢复分支 `continue` 会推进 attempt，旧守卫 `attempt == 0` 于是让紧随其后的
        真 token 过期拿不到那次免费修复。改成一次性 flag 后必须拿得到。
        """
        proxy, lb, ep, forced = _make_proxy()
        client = _RecordingStreamClient([
            _err_stream(401, ORPHAN_401),        # rung 2 触发，attempt -> 1
            _err_stream(401, PLAIN_401, JSON_CT),  # 真 token 过期，必须仍有 auth repair
            _ok_stream(),
        ])
        proxy.client = client
        res = await proxy.proxy_responses(stateful_body(), stream=True)
        out = b"".join([c if isinstance(c, bytes) else c.encode()
                        async for c in res.body_iterator])

        self.assertEqual(forced.count(True), 1,
                         "opaque-state 恢复用掉一次 attempt，不该吃掉 auth repair 预算")
        self.assertEqual(len(client.sent), 3)
        self.assertIn(b"response.completed", out)
        self.assertEqual(ep.active_requests, 0)


# ---------------------------------------------------------------- 指标 / 配置


class UpstreamCallBudgetTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):
    """把上游调用次数上界钉死。共享预算的意义就在这：外层 attempt 重试不能让恢复
    阶梯重新装弹，否则一个中毒会话会成倍放大上游负载。

    实测上界（2026-09-10）：
      流式    ≤3 —— `for attempt in range(3)` 是硬顶，每个 continue 都要求还剩一轮
      非流式  ≤4 —— 1 次原始 + 1 次 401 auth repair + 2 级恢复；外层 attempt 循环
                    的重试不叠加，因为 recovery 是每请求一份
      生产形态 2 —— 主动剥离已开 ⇒ rung1 无料，只有 rung2。与改动前的调用次数相同，
                    区别是那一次从「注定无用的强制 token 刷新」换成真正的恢复尝试
    """

    def _client(self, seq):
        sent = []

        async def post(url, json, headers):
            sent.append(copy.deepcopy(json))
            status, text = seq[min(len(sent) - 1, len(seq) - 1)]
            return httpx.Response(status, request=httpx.Request("POST", url),
                                  text=text, headers=dict(JSON_CT))

        client = unittest.mock.Mock()
        client.post = AsyncMock(side_effect=post)
        return client, sent

    async def test_buffered_worst_case_is_four_calls(self):
        proxy, lb, ep, forced = _make_proxy(threshold=99)
        proxy.client, sent = self._client([(401, PLAIN_401)] + [(401, ORPHAN_401)] * 8)
        with patch.object(main, "LB_SETTINGS",
                          _settings(copilot_strip_input_item_ids=False,
                                    copilot_opaque_state_recovery=True)), \
             patch.object(main.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(main.HTTPException):
                await proxy.proxy_responses(stateful_body(with_id=True), stream=False)

        self.assertEqual(len(sent), 4, "1 原始 + 1 auth repair + 2 级恢复")
        self.assertEqual(forced.count(True), 1)
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 2)
        self.assertEqual(ep.active_requests, 0)

    async def test_outer_attempt_retry_does_not_reload_the_ladder(self):
        """429 吃掉一轮外层 attempt 后，恢复预算不得重置。"""
        proxy, lb, ep, forced = _make_proxy(threshold=99)
        proxy.client, sent = self._client(
            [(429, '{"error":{"message":"slow"}}')] + [(401, ORPHAN_401)] * 8)
        with patch.object(main, "LB_SETTINGS",
                          _settings(copilot_strip_input_item_ids=False,
                                    copilot_opaque_state_recovery=True)), \
             patch.object(main.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(main.HTTPException):
                await proxy.proxy_responses(stateful_body(with_id=True), stream=False)

        self.assertLessEqual(len(sent), 4, "共享预算必须挡住倍增")
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 2)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)

    async def test_production_shape_is_two_calls(self):
        """主动剥离开启（线上默认）+ 纯 orphan：与改动前调用次数相同。"""
        proxy, lb, ep, forced = _make_proxy(threshold=99)
        proxy.client, sent = self._client([(401, ORPHAN_401)] * 8)
        await self._expect_failure(proxy)
        self.assertEqual(len(sent), 2)
        self.assertEqual(forced.count(True), 0, "改动前这一次是浪费的强制 token 刷新")

    async def _expect_failure(self, proxy):
        with self.assertRaises(main.HTTPException):
            await proxy.proxy_responses(stateful_body(), stream=False)

    async def test_chat_protocol_never_fires_a_rung(self):
        """Chat Completions 没有 input[]，两级改写都必须是空操作。"""
        proxy, lb, ep, forced = _make_proxy(threshold=99)
        proxy.client, sent = self._client([(401, ORPHAN_401)] * 4)
        with self.assertRaises(main.HTTPException):
            await proxy.proxy_chat_completions(
                {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
                stream=False)
        self.assertEqual(len(sent), 1, "chat 上没有可改写的载荷，不得重发")
        self.assertEqual(proxy.opaque_state_recovery.get("attempted", 0), 0)
        self.assertEqual(proxy.opaque_state_recovery.get("exhausted"), 1)


class OtherProvidersMustNotBePatchedTests(unittest.TestCase):
    """结构守卫：恢复阶梯只属于 Copilot。

    ADB 的 401 = `dapi` token 坏，Azure 的 401 = `api-key` 坏，两者都没有 opaque
    state 这个概念。防止后人把这套「顺手」推广过去。
    """

    def setUp(self):
        self.src = pathlib.Path(main.__file__).read_text(encoding="utf-8")

    def test_recovery_helpers_have_only_copilot_call_sites(self):
        # 1 处定义 + 流式/非流式各 1 处调用
        self.assertEqual(self.src.count("_classify_opaque_state_rejection("), 3)
        # 1 处定义 + _OpaqueStateRecovery.try_next_rung 1 处调用
        self.assertEqual(self.src.count("_drop_reasoning_encrypted_content("), 2)
        # 1 处定义 + 流式/非流式各 1 处调用
        self.assertEqual(self.src.count("_mark_opaque_state_exhausted("), 3)
        # 恢复预算对象只在 Copilot 三处构造（_proxy / _stream_response / _normal_request）
        self.assertEqual(self.src.count("_OpaqueStateRecovery(self"), 3)

    def test_databricks_and_azure_predicates_are_untouched(self):
        self.assertEqual(self.src.count("is_client_error = 400 <= "), 4,
                         "ADB×2 + Azure×2 共 4 处必须保持原样")


class RecoveryResourceOwnershipTests(_PinnedSettings, unittest.IsolatedAsyncioTestCase):
    """恢复用的原地重试**不** end/start lease（loop 头不会重新 start），所以取消与
    断连路径的收尾必须逐条验，不能靠「测试都过了」推断。
    """

    async def test_cancel_during_the_recovery_retry_releases_the_lease_once(self):
        proxy, lb, ep, forced = _make_proxy()
        blocking = fixtures._StreamResponse(
            [b'event: response.created\ndata: {"id":"r_1"}\n\n'],
            block_after_chunks=True)
        proxy.client = _RecordingStreamClient([_err_stream(401, ORPHAN_401), blocking])

        res = await proxy.proxy_responses(stateful_body(), stream=True)
        iterator = res.body_iterator
        first = await anext(iterator)
        self.assertIn(b"response.created", first)
        self.assertEqual(proxy.opaque_state_recovery.get("attempted"), 1,
                         "第一帧之前应已走过一次恢复重试")

        pending = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(blocking.blocked.wait(), timeout=2)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await iterator.aclose()

        self.assertTrue(blocking.closed, "上游响应必须被关闭")
        self.assertEqual(ep.active_requests, 0)
        self.assertEqual(lb.on_request_start.await_count, 1,
                         "恢复重试不得重新 start lease")
        self.assertEqual(lb.on_request_end.await_count, 1,
                         "lease 只能被结算一次")
        self.assertFalse(ep.circuit_open)

    async def test_recovery_retry_does_not_double_start_the_lease(self):
        """成功收尾时也确认 start/end 各一次 —— 原地重试必须复用同一 lease。"""
        proxy, lb, ep, forced = _make_proxy()
        proxy.client = _RecordingStreamClient([_err_stream(401, ORPHAN_401), _ok_stream()])
        res = await proxy.proxy_responses(stateful_body(), stream=True)
        async for _c in res.body_iterator:
            pass
        self.assertEqual(lb.on_request_start.await_count, 1)
        self.assertEqual(lb.on_request_end.await_count, 1)
        self.assertEqual(ep.active_requests, 0)
        self.assertEqual(proxy.opaque_state_recovery.get("succeeded"), 1)

    async def test_never_iterated_recovery_response_still_releases(self):
        """客户端拿到 StreamingResponse 却一帧都不读就关掉 —— 不能泄漏 lease。"""
        proxy, lb, ep, forced = _make_proxy()
        proxy.client = _RecordingStreamClient([_err_stream(401, ORPHAN_401), _ok_stream()])
        res = await proxy.proxy_responses(stateful_body(), stream=True)
        await res.body_iterator.aclose()
        self.assertEqual(ep.active_requests, 0)


class MetricsBaselineTests(unittest.IsolatedAsyncioTestCase):
    """dict 驱动的新指标必须自带零样本，告警才能写 `> 0` 而不是 `absent()`。"""

    async def _scrape(self, proxy):
        previous = main.copilot_proxy
        main.copilot_proxy = proxy
        try:
            response = await main.metrics()
        finally:
            main.copilot_proxy = previous
        return response.body.decode("utf-8")

    async def test_fresh_proxy_exposes_zero_samples(self):
        proxy, _lb, _ep, _f = _make_proxy()
        body = await self._scrape(proxy)
        for expected in (
            "copilot_opaque_state_requests_total 0",
            # 循环不变量的金丝雀：必须有零样本，告警才能写 `> 0` 而不是 grep 日志
            "copilot_stream_retry_budget_exhausted_total 0",
            'copilot_opaque_state_rejections_total{kind="orphaned_id"} 0',
            'copilot_opaque_state_rejections_total{kind="unverifiable_content"} 0',
            'copilot_opaque_state_recovery_total{outcome="attempted"} 0',
            'copilot_opaque_state_recovery_total{outcome="exhausted"} 0',
            'copilot_opaque_state_recovery_total{outcome="succeeded"} 0',
        ):
            with self.subTest(sample=expected):
                self.assertIn(expected, body)

    async def test_real_counts_and_unknown_labels_both_show_up(self):
        proxy, _lb, _ep, _f = _make_proxy()
        proxy.opaque_state_requests_total = 4
        proxy.opaque_state_rejections["orphaned_id"] = 9
        proxy.opaque_state_recovery["some_future_outcome"] = 2
        body = await self._scrape(proxy)
        self.assertIn("copilot_opaque_state_requests_total 4", body)
        self.assertIn('copilot_opaque_state_rejections_total{kind="orphaned_id"} 9', body)
        self.assertIn('copilot_opaque_state_rejections_total{kind="unverifiable_content"} 0', body)
        self.assertIn('copilot_opaque_state_recovery_total{outcome="some_future_outcome"} 2', body)
        self.assertIn('copilot_opaque_state_recovery_total{outcome="attempted"} 0', body)


class SettingsTests(unittest.TestCase):

    def test_env_switch_defaults_true_and_can_be_disabled(self):
        prev = os.environ.pop("COPILOT_OPAQUE_STATE_RECOVERY", None)
        try:
            self.assertTrue(main.LBSettings.load().copilot_opaque_state_recovery)
            os.environ["COPILOT_OPAQUE_STATE_RECOVERY"] = "false"
            self.assertFalse(main.LBSettings.load().copilot_opaque_state_recovery)
            os.environ["COPILOT_OPAQUE_STATE_RECOVERY"] = "true"
            self.assertTrue(main.LBSettings.load().copilot_opaque_state_recovery)
        finally:
            os.environ.pop("COPILOT_OPAQUE_STATE_RECOVERY", None)
            if prev is not None:
                os.environ["COPILOT_OPAQUE_STATE_RECOVERY"] = prev


if __name__ == "__main__":
    unittest.main()
