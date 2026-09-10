"""上游 401 按来源分类，别让一个坏会话把整个 Copilot 账户熔断掉。

背景（2026-09-09 实测，见 CopilotProxy._classify_upstream_failure 的 docstring）：
401 被显式排除在 `is_client_error` 之外 → `failed=True` → `consecutive_errors += 1`
→ 连续 5 次打开 **endpoint 级** 熔断。而一个 Copilot endpoint 承载该账户的全部
模型，于是「一个会话持续 401」会升级成「该账户所有模型下线」。

实测的计数关系（本文件用真实代码路径复现，不是读代码推断）：
  - 一个持续 401 的请求贡献 **1** 次 consecutive_errors —— 首次 401 走
    auth repair 的 `continue`（在 end_current_request 之前），不计数
  - 阈值 5 ⇒ 第 5 个这样的请求打开熔断
  - 熔断后 `_select_endpoint` 对**任意**模型、**任意** api_type 都返 None

分类判据是 `_request_has_opaque_state`：stateful 请求的 401 是 request-scoped
（中性），无状态请求的 401 才是 endpoint-scoped（计数）。为什么这样才对 ——
`_select_endpoint` 对 stateful 请求返 `pinned if pinned in matched else None`，
而 matched 已排除熔断端点，所以对 stateful 请求熔断**只有害无益**：它杀掉那个
会话唯一可能服务的 endpoint，并顺带带走该账户其他全部流量，而 failover 本来
就被 pinning 禁止，熔断换不来任何可用性。
"""
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

UNAUTH_BODY = '{"error":{"message":"Bad credentials","code":"unauthorized"}}'
JSON_CT = {"content-type": "application/json"}

# 携带 opaque state 的请求（reasoning 加密上下文）—— stateful
STATEFUL_BODY = {
    "model": "gpt-test",
    "input": [{"type": "reasoning", "encrypted_content": "E" * 128}],
}
# 纯输入，无 previous_response_id / encrypted_content —— stateless
STATELESS_BODY = {"model": "gpt-test", "input": [{"type": "message", "role": "user"}]}


def _make_copilot_proxy(threshold=5):
    """复用生产字段集合一致的夹具；token 交换 mock 掉，绝不打网络。"""
    proxy, lb, ep = fixtures.CopilotRequestLifecycleTests()._make_proxy(
        unittest.mock.Mock(), threshold=threshold)
    proxy.get_session_token = AsyncMock(return_value="fresh-session-token")
    return proxy, lb, ep


def _resp401(url="https://example.test/responses"):
    return httpx.Response(401, request=httpx.Request("POST", url),
                          text=UNAUTH_BODY, headers=dict(JSON_CT))


class ClassifyUpstreamFailureTests(unittest.TestCase):
    """纯判据表。True = is_client_error = 中性 = 不计入 endpoint 健康度。"""

    def test_401_is_neutral_only_when_request_carries_opaque_state(self):
        self.assertTrue(CopilotProxy._classify_upstream_failure(401, request_is_stateful=True))
        self.assertFalse(CopilotProxy._classify_upstream_failure(401, request_is_stateful=False))

    def test_every_other_status_ignores_statefulness(self):
        """只有 401 的语义是歧义的；其余状态码行为必须与改动前逐位相同。"""
        for status in (400, 402, 403, 404, 409, 422, 429, 500, 502, 503, 200):
            legacy = 400 <= status < 500 and status not in (401, 403, 429)
            for stateful in (True, False):
                with self.subTest(status=status, stateful=stateful):
                    self.assertEqual(
                        CopilotProxy._classify_upstream_failure(
                            status, request_is_stateful=stateful),
                        legacy,
                        f"status={status} 的分类不得因本次改动而变化",
                    )

    def test_403_and_429_still_count_toward_circuit(self):
        """403（Cloudflare bot management）与 429（服务端过载）确是 endpoint 级信号。"""
        for status in (403, 429):
            self.assertFalse(CopilotProxy._classify_upstream_failure(
                status, request_is_stateful=True))

    def test_kill_switch_restores_legacy_counting(self):
        disabled = main.LBSettings.load()
        object.__setattr__(disabled, "copilot_stateful_401_neutral", False)
        with patch.object(main, "LB_SETTINGS", disabled):
            self.assertFalse(CopilotProxy._classify_upstream_failure(
                401, request_is_stateful=True))


class NonStreamCircuitScopeTests(unittest.IsolatedAsyncioTestCase):

    async def _drive(self, body, threshold=5, requests=1):
        proxy, lb, ep = _make_copilot_proxy(threshold=threshold)
        calls = []

        async def post(url, json, headers):
            calls.append(url)
            return _resp401(url)

        proxy.client.post = AsyncMock(side_effect=post)
        for _ in range(requests):
            with self.assertRaises(Exception):
                await proxy.proxy_responses(dict(body), stream=False)
        return proxy, lb, ep, calls

    async def test_stateful_401_does_not_pollute_endpoint_health(self):
        proxy, lb, ep, calls = await self._drive(STATEFUL_BODY)
        self.assertEqual(len(calls), 2, "首次 401 后仍走一次 auth repair 重发")
        self.assertEqual(ep.consecutive_errors, 0, "stateful 401 不得计入熔断")
        self.assertEqual(ep.neutral_requests, 1)
        self.assertEqual(ep.total_errors, 0)
        self.assertEqual(lb.circuit_state(ep), "CLOSED")
        self.assertEqual(proxy.upstream_401_events.get("request"), 1)

    async def test_stateless_401_still_counts(self):
        """闭合缺口：seat/policy 被撤时 token 交换仍成功，
        `_mark_endpoint_unhealthy` 永不触发，只有计数能熔断它。"""
        proxy, lb, ep, _calls = await self._drive(STATELESS_BODY)
        self.assertEqual(ep.consecutive_errors, 1)
        self.assertEqual(ep.total_errors, 1)
        self.assertEqual(proxy.upstream_401_events.get("endpoint"), 1)

    async def test_five_stateless_401s_still_open_the_circuit(self):
        proxy, lb, ep, _ = await self._drive(STATELESS_BODY, threshold=5, requests=5)
        self.assertEqual(ep.consecutive_errors, 5)
        self.assertTrue(ep.circuit_open)
        self.assertEqual(lb.circuit_state(ep), "OPEN")

    async def test_many_stateful_401s_never_open_the_circuit(self):
        proxy, lb, ep, _ = await self._drive(STATEFUL_BODY, threshold=5, requests=12)
        self.assertEqual(ep.consecutive_errors, 0)
        self.assertFalse(ep.circuit_open)
        self.assertEqual(proxy.upstream_401_events.get("request"), 12)
        # 爆炸半径回归：无关模型仍可选中
        self.assertIsNotNone(proxy._select_endpoint("gpt-test", "chat"))

    async def test_kill_switch_false_reproduces_the_defect(self):
        """开关关掉必须原样重现旧行为 —— 证明我们测的是真缺陷而非夹具假象。"""
        proxy, lb, ep = _make_copilot_proxy(threshold=5)
        proxy.client.post = AsyncMock(side_effect=lambda url, json, headers: _resp401(url))
        disabled = main.LBSettings.load()
        object.__setattr__(disabled, "copilot_stateful_401_neutral", False)
        with patch.object(main, "LB_SETTINGS", disabled):
            for _ in range(5):
                with self.assertRaises(Exception):
                    await proxy.proxy_responses(dict(STATEFUL_BODY), stream=False)
        self.assertEqual(ep.consecutive_errors, 5)
        self.assertTrue(ep.circuit_open, "旧行为下 5 个 stateful 401 就会熔断整个账户")


class StreamCircuitScopeTests(unittest.IsolatedAsyncioTestCase):

    async def _drive(self, body, threshold=5, requests=1):
        proxy, lb, ep = _make_copilot_proxy(threshold=threshold)
        for _ in range(requests):
            # 每个请求两次上游 401（原始 + auth repair 重发）
            proxy.client = fixtures._StreamClient([
                fixtures._ErrorStreamResponse(401, UNAUTH_BODY.encode(), headers=dict(JSON_CT))
                for _ in range(2)
            ])
            res = await proxy.proxy_responses(dict(body), stream=True)
            async for _chunk in res.body_iterator:   # 必须真消费才会跑到 401 分支
                pass
        return proxy, lb, ep

    async def test_stateful_401_does_not_pollute_endpoint_health(self):
        proxy, lb, ep = await self._drive(STATEFUL_BODY)
        self.assertEqual(ep.consecutive_errors, 0)
        self.assertEqual(ep.neutral_requests, 1)
        self.assertEqual(lb.circuit_state(ep), "CLOSED")
        self.assertEqual(proxy.upstream_401_events.get("request"), 1)

    async def test_stateless_401_still_counts(self):
        proxy, lb, ep = await self._drive(STATELESS_BODY)
        self.assertEqual(ep.consecutive_errors, 1)
        self.assertEqual(proxy.upstream_401_events.get("endpoint"), 1)

    async def test_five_stateless_401s_still_open_the_circuit(self):
        proxy, lb, ep = await self._drive(STATELESS_BODY, threshold=5, requests=5)
        self.assertTrue(ep.circuit_open)

    async def test_many_stateful_401s_never_open_the_circuit(self):
        proxy, lb, ep = await self._drive(STATEFUL_BODY, threshold=5, requests=12)
        self.assertFalse(ep.circuit_open)
        self.assertEqual(ep.consecutive_errors, 0)


class OtherProvidersMustNotBePatchedTests(unittest.IsolatedAsyncioTestCase):
    """ADB 的 401 = dapi token 坏，Azure 的 401 = api-key 坏 —— 两者**真的**
    等于 endpoint 不健康，计数是对的。本次分类只适用于 Copilot。
    """

    async def test_azure_401_still_counts_toward_circuit(self):
        request = httpx.Request("POST", "https://example.test/openai/v1/responses")
        client = unittest.mock.Mock()
        client.post = AsyncMock(return_value=httpx.Response(
            401, request=request, text=UNAUTH_BODY, headers=dict(JSON_CT)))
        proxy, lb, ep = fixtures.AzureRequestLifecycleTests()._make_proxy(
            client, threshold=5)
        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(Exception):
                await proxy.proxy_responses(
                    {"model": "gpt-test", "input": [
                        {"type": "reasoning", "encrypted_content": "E" * 128}]},
                    stream=False)
        self.assertGreaterEqual(ep.consecutive_errors, 1,
                                "Azure 的 401 必须仍计入熔断")

    def test_classification_helper_has_exactly_two_call_sites(self):
        """结构守卫：防止后人把这个补丁「顺手」推广到 ADB / Azure 那 4 处。"""
        src = pathlib.Path(main.__file__).read_text(encoding="utf-8")
        # 1 处定义 + 2 处调用（CopilotProxy._proxy / _stream_response）。注意不能
        # 分别数 "self._classify..." 与 "proxy_self._classify..." —— 后者的子串里
        # 就含前者，会重复计数。
        self.assertEqual(src.count("_classify_upstream_failure("), 3,
                         "只有 CopilotProxy._proxy 与 _stream_response 可以调它")
        # ADB×2 + Azure×2 必须仍是原地字面量赋值。数赋值形式而不是数
        # "not in (401, 403, 429)" —— 后者在 _classify_upstream_failure 的
        # fallback 分支里也有一份（那是 return 而非赋值）。
        self.assertEqual(src.count("is_client_error = 400 <= "), 4,
                         "ADB×2 + Azure×2 共 4 处必须保持原样")


class CanaryMetricsHaveZeroBaselineTests(unittest.IsolatedAsyncioTestCase):
    """dict 驱动的 label 指标在健康态下必须有 0 序列，而不是整条不见。

    此前 `copilot_orphaned_item_id_events_total` 只有 HELP/TYPE 没有 sample
    （空 dict → 空样本列表 → emit 不写行），于是被 67e91cd 指定为「必须恒 0」
    的金丝雀在健康态下压根没有可观测的 0，运维只能用 absent() 猜「零事件」还是
    「LB 没部署」。与 test_prometheus_exposes_connection_lifecycle_metrics 的
    "Newly exposed counters must be present with 0 baseline." 是同一条要求。
    """

    async def _scrape(self, proxy):
        previous = main.copilot_proxy
        main.copilot_proxy = proxy
        try:
            response = await main.metrics()
        finally:
            main.copilot_proxy = previous
        return response.body.decode("utf-8")

    async def test_fresh_proxy_exposes_zero_samples_for_every_known_label(self):
        proxy, _lb, _ep = _make_copilot_proxy()
        body = await self._scrape(proxy)
        for expected in (
            'copilot_orphaned_item_id_events_total{stage="detected"} 0',
            'copilot_orphaned_item_id_events_total{stage="recovered"} 0',
            'copilot_orphaned_item_id_events_total{stage="unrecoverable"} 0',
            'copilot_stateful_request_pinned_total{reason="http_5xx"} 0',
            'copilot_stateful_request_pinned_total{reason="network_error"} 0',
            'copilot_stateful_request_pinned_total{reason="pinned_unavailable"} 0',
            # 实际 reason 是 pool_acquire_timeout（_note_stateful_pin 的调用点），
            # 不是旧注释与文档里写的 pool_timeout
            'copilot_stateful_request_pinned_total{reason="pool_acquire_timeout"} 0',
            'copilot_upstream_html_events_by_status_total{status_bucket="200"} 0',
            'copilot_upstream_html_events_by_status_total{status_bucket="4xx"} 0',
            'copilot_upstream_html_events_by_status_total{status_bucket="5xx"} 0',
            'copilot_upstream_html_events_by_status_total{status_bucket="other"} 0',
            'copilot_upstream_401_total{scope="endpoint"} 0',
            'copilot_upstream_401_total{scope="request"} 0',
        ):
            with self.subTest(sample=expected):
                self.assertIn(expected, body)

    async def test_real_counts_override_the_zero_baseline(self):
        proxy, _lb, _ep = _make_copilot_proxy()
        proxy.orphaned_item_id_events["detected"] = 3
        proxy.upstream_401_events["request"] = 7
        body = await self._scrape(proxy)
        self.assertIn('copilot_orphaned_item_id_events_total{stage="detected"} 3', body)
        self.assertIn('copilot_orphaned_item_id_events_total{stage="recovered"} 0', body)
        self.assertIn('copilot_upstream_401_total{scope="request"} 7', body)
        self.assertIn('copilot_upstream_401_total{scope="endpoint"} 0', body)

    async def test_unknown_labels_are_still_exposed(self):
        """预置零样本不能把未预置的新 label 吞掉。"""
        proxy, _lb, _ep = _make_copilot_proxy()
        proxy.stateful_pinned_events["some_future_reason"] = 2
        body = await self._scrape(proxy)
        self.assertIn('copilot_stateful_request_pinned_total{reason="some_future_reason"} 2', body)
        self.assertIn('copilot_stateful_request_pinned_total{reason="http_5xx"} 0', body)

    async def test_open_ended_model_labels_are_not_preseeded(self):
        """按模型拆分的 truncation 指标 label 是开放集合，不能凭空造零样本。"""
        proxy, _lb, _ep = _make_copilot_proxy()
        body = await self._scrape(proxy)
        self.assertNotIn("copilot_stream_truncated_no_completion_by_model_total{", body)


class SettingsTests(unittest.TestCase):

    def test_env_switch_defaults_true_and_can_be_disabled(self):
        prev = os.environ.pop("COPILOT_STATEFUL_401_NEUTRAL", None)
        try:
            self.assertTrue(main.LBSettings.load().copilot_stateful_401_neutral)
            os.environ["COPILOT_STATEFUL_401_NEUTRAL"] = "false"
            self.assertFalse(main.LBSettings.load().copilot_stateful_401_neutral)
            os.environ["COPILOT_STATEFUL_401_NEUTRAL"] = "true"
            self.assertTrue(main.LBSettings.load().copilot_stateful_401_neutral)
        finally:
            os.environ.pop("COPILOT_STATEFUL_401_NEUTRAL", None)
            if prev is not None:
                os.environ["COPILOT_STATEFUL_401_NEUTRAL"] = prev


if __name__ == "__main__":
    unittest.main()
