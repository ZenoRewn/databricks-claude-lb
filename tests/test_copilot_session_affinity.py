"""会话亲和：同一会话的每一轮必须落到同一个 Copilot 账户。

为什么这是硬需求（2026-09-10 用两个真实 GHCP 账户实测，双向 × 3 次重复 = 24/24）：
A 账户铸造的 reasoning `encrypted_content` 拿到 B 账户回放**必然** 401
`input item does not belong to this connection`。而 `least_requests` 会把同一会话的
连续轮次分到不同账户，所以**只要配了第二个账户，这个 401 就是必然事件而非偶发**。
恢复阶梯能救回来（也是实测的），但每次都要丢掉推理链 —— 亲和是从根上不产生它。

键取 `prompt_cache_key`。实测 codex-cli 0.145.0：
  - 跨 6 次客户端重试完全一致
  - 跨同一会话的多个轮次一致（turn1 7 个 input item / turn2 9 个，同一个键）
  - 它等于 Codex 的 `session_id` / `thread_id`

**单账户下整条机制是 no-op**（只有一个合格端点时压根不走亲和分支），所以默认开对
当前生产零影响。
"""
import os
import sys
import unittest
import unittest.mock
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import CopilotProxy

import test_copilot_request_lifecycle as fixtures

# 实测抓到的真实 Codex 会话键（UUIDv7 形态；等于 session_id / thread_id）
REAL_KEY_1 = "01a089f5-b677-72f3-b7cc-524244171675"
REAL_KEY_2 = "01a089f7-f96c-7c33-8c04-47cb92803a75"


class _PinnedAffinity:
    """把亲和开关钉成显式开启。

    测试结论不能随「跑测试时 shell 里恰好有什么环境变量」变化 —— 否则
    `COPILOT_SESSION_AFFINITY=false pytest` 会让一堆与开关无关的断言一起红，
    分不清是真回归还是环境噪音。验开关本身的用例在方法内再局部覆盖（patch 可嵌套）。
    """

    def setUp(self):
        super().setUp()
        snapshot = main.LBSettings.load()
        object.__setattr__(snapshot, "copilot_session_affinity", True)
        patcher = patch.object(main, "LB_SETTINGS", snapshot)
        patcher.start()
        self.addCleanup(patcher.stop)


def _multi_proxy(names=("acct-a", "acct-b", "acct-c"), threshold=99):
    """多账户夹具：复用单账户夹具，再把 endpoint 列表换成 N 个。"""
    proxy, lb, _ep = fixtures.CopilotRequestLifecycleTests()._make_proxy(
        unittest.mock.Mock(), threshold=threshold)
    endpoints = [main.CopilotEndpoint(name=n, github_token="t", models=["gpt-test"],
                                      session_token="s", session_token_expires_at=2**31)
                 for n in names]
    lb.endpoints = endpoints
    return proxy, lb, endpoints


class AffinityKeyTests(unittest.TestCase):

    def test_prompt_cache_key_is_the_key(self):
        self.assertEqual(
            CopilotProxy._session_affinity_key(
                {"prompt_cache_key": REAL_KEY_1, "input": []}, "responses"),
            REAL_KEY_1)

    def test_absent_or_blank_key_disables_affinity(self):
        for body in ({"input": []}, {"prompt_cache_key": ""},
                     {"prompt_cache_key": "   "}, {"prompt_cache_key": None},
                     {"prompt_cache_key": 12345}):
            with self.subTest(body=body):
                self.assertIsNone(
                    CopilotProxy._session_affinity_key(body, "responses"))

    def test_chat_protocol_has_no_affinity(self):
        """Chat 没有 prompt_cache_key 也没有 opaque state，不做亲和。"""
        self.assertIsNone(CopilotProxy._session_affinity_key(
            {"prompt_cache_key": REAL_KEY_1, "messages": []}, "chat"))

    def test_malformed_body_is_tolerated(self):
        self.assertIsNone(CopilotProxy._session_affinity_key("not-a-dict", "responses"))


class StableHashTests(unittest.TestCase):
    """哈希必须跨进程/跨副本确定 —— 否则这个机制的唯一目的就没了。"""

    def test_index_is_deterministic_and_in_range(self):
        for n in (2, 3, 5, 9):
            idx = {CopilotProxy._affinity_index(REAL_KEY_1, n) for _ in range(50)}
            self.assertEqual(len(idx), 1, "同一键同一 n 必须恒等")
            self.assertTrue(0 <= idx.pop() < n)

    def test_does_not_use_pythons_salted_hash(self):
        """内置 `hash()` 对 str 加 per-process 随机盐（PYTHONHASHSEED），多副本
        之间结果不同。用子进程验证 blake2b 路径不受它影响。"""
        import subprocess
        code = ("import sys; sys.path.insert(0, '.');"
                "import main;"
                f"print(main.CopilotProxy._affinity_index({REAL_KEY_1!r}, 7))")
        outs = set()
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, env=env,
                               cwd=os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__))))

            self.assertEqual(r.returncode, 0, r.stderr[-400:])
            outs.add(r.stdout.strip())
        self.assertEqual(len(outs), 1,
                         f"不同 PYTHONHASHSEED 下结果必须一致，实际 {outs}")

    def test_distributes_across_accounts(self):
        """不同会话要散开，否则亲和就退化成「全都钉在一个账户」。"""
        keys = [f"01a089f5-b677-72f3-b7cc-{i:012d}" for i in range(200)]
        buckets = {CopilotProxy._affinity_index(k, 3) for k in keys}
        self.assertEqual(buckets, {0, 1, 2}, "3 个账户都应被用到")

    def test_zero_accounts_does_not_divide_by_zero(self):
        self.assertEqual(CopilotProxy._affinity_index(REAL_KEY_1, 0), 0)


class SelectionTests(_PinnedAffinity, unittest.TestCase):

    def test_same_session_always_lands_on_the_same_account(self):
        proxy, lb, eps = _multi_proxy()
        chosen = {proxy._select_endpoint("gpt-test", "responses",
                                        session_key=REAL_KEY_1).name
                  for _ in range(40)}
        self.assertEqual(len(chosen), 1, f"同一会话必须恒定落到同一账户，实际 {chosen}")
        self.assertEqual(proxy.session_affinity_events.get("hit"), 40)

    def test_different_sessions_can_land_on_different_accounts(self):
        proxy, lb, eps = _multi_proxy()
        a = proxy._select_endpoint("gpt-test", "responses", session_key=REAL_KEY_1).name
        b = proxy._select_endpoint("gpt-test", "responses", session_key=REAL_KEY_2).name
        # 不强求这两个特定键落在不同账户（哈希说了算），但整体必须散开
        keys = [f"sess-{i}" for i in range(60)]
        names = {proxy._select_endpoint("gpt-test", "responses", session_key=k).name
                 for k in keys}
        self.assertGreater(len(names), 1, "不同会话不该全钉在一个账户上")
        self.assertIn(a, [e.name for e in eps])
        self.assertIn(b, [e.name for e in eps])

    def test_mapping_is_stable_when_an_unrelated_account_goes_down(self):
        """关键性质：哈希打在**配置态**集合上，所以某个端点掉线只影响映射到它的会话。

        若打在「可用集合」上，任何一次熔断都会把所有会话重新洗牌 —— 那等于每次熔断
        都给所有活跃会话制造一次跨账户回放。
        """
        proxy, lb, eps = _multi_proxy()
        keys = [f"sess-{i}" for i in range(60)]
        before = {k: proxy._select_endpoint("gpt-test", "responses", session_key=k).name
                  for k in keys}
        # 让一个**没被这批会话用到最多**的端点熔断
        victim = eps[0]
        victim.circuit_open = True
        victim.circuit_retry_at = main.time.time() + 300
        after = {k: (proxy._select_endpoint("gpt-test", "responses", session_key=k) or victim).name
                 for k in keys}
        moved = [k for k in keys if before[k] != after[k]]
        expected_moved = [k for k in keys if before[k] == victim.name]
        self.assertEqual(sorted(moved), sorted(expected_moved),
                         "只有原本映射到熔断端点的会话可以改变归属")

    def test_falls_back_when_the_affine_account_is_unavailable(self):
        proxy, lb, eps = _multi_proxy()
        target = eps[CopilotProxy._affinity_index(REAL_KEY_1, len(eps))]
        target.circuit_open = True
        target.circuit_retry_at = main.time.time() + 300
        got = proxy._select_endpoint("gpt-test", "responses", session_key=REAL_KEY_1)
        self.assertIsNotNone(got, "亲和目标不可用时不能拒绝服务")
        self.assertNotEqual(got.name, target.name)
        self.assertEqual(proxy.session_affinity_events.get("unavailable"), 1)
        self.assertEqual(proxy.session_affinity_events.get("hit", 0), 0)

    def test_single_account_is_a_complete_noop(self):
        """当前生产就是单账户：亲和分支压根不该被走到。"""
        proxy, lb, eps = _multi_proxy(names=("only",))
        got = proxy._select_endpoint("gpt-test", "responses", session_key=REAL_KEY_1)
        self.assertEqual(got.name, "only")
        self.assertEqual(proxy.session_affinity_events, {},
                         "单账户下不得产生任何亲和事件")

    def test_pinned_takes_precedence_over_affinity(self):
        """pinning 管「同一请求内不许换」，亲和管「跨请求落回同一个」。前者更强 ——
        请求已经开始承载 opaque state 时，换到亲和目标同样会 401。
        """
        proxy, lb, eps = _multi_proxy()
        affine = eps[CopilotProxy._affinity_index(REAL_KEY_1, len(eps))]
        other = next(e for e in eps if e is not affine)
        got = proxy._select_endpoint("gpt-test", "responses",
                                     pinned=other, session_key=REAL_KEY_1)
        self.assertIs(got, other, "pinned 必须压过亲和")

    def test_kill_switch_restores_plain_load_balancing(self):
        proxy, lb, eps = _multi_proxy()
        disabled = main.LBSettings.load()
        object.__setattr__(disabled, "copilot_session_affinity", False)
        with patch.object(main, "LB_SETTINGS", disabled):
            names = {proxy._select_endpoint("gpt-test", "responses",
                                            session_key=REAL_KEY_1).name
                     for _ in range(30)}
        self.assertEqual(proxy.session_affinity_events, {},
                         "关掉开关后不得记亲和事件")
        # least_requests 在全 0 活跃时会随机选，所以这里只验「不再恒定」不可靠；
        # 关键是没有走亲和分支（上面那条断言）。
        self.assertTrue(names.issubset({e.name for e in eps}))


class EndToEndTests(_PinnedAffinity, unittest.IsolatedAsyncioTestCase):
    """走完整 _proxy 路径，确认 session_key 真的被穿到选点处。"""

    async def test_consecutive_turns_of_one_session_hit_the_same_account(self):
        import copy
        import httpx
        proxy, lb, eps = _multi_proxy()
        used = []

        async def post(url, json, headers):
            used.append(url)
            return httpx.Response(200, request=httpx.Request("POST", url),
                                  json={"output": [], "usage": {}})

        proxy.client.post = unittest.mock.AsyncMock(side_effect=post)
        body = {"model": "gpt-test", "prompt_cache_key": REAL_KEY_1,
                "input": [{"type": "message", "role": "user"}]}
        for _ in range(6):
            await proxy.proxy_responses(copy.deepcopy(body), stream=False)

        hosts = {u.split("/responses")[0] for u in used}
        self.assertEqual(len(hosts), 1,
                         f"同一会话的 6 轮必须打到同一账户，实际 {hosts}")
        self.assertEqual(proxy.session_affinity_events.get("hit"), 6)

    async def test_request_without_a_session_key_is_counted_as_absent(self):
        import httpx
        proxy, lb, eps = _multi_proxy()
        proxy.client.post = unittest.mock.AsyncMock(
            side_effect=lambda url, json, headers: httpx.Response(
                200, request=httpx.Request("POST", url), json={"output": [], "usage": {}}))
        await proxy.proxy_responses(
            {"model": "gpt-test", "input": [{"type": "message", "role": "user"}]},
            stream=False)
        self.assertEqual(proxy.session_affinity_events.get("absent"), 1)
        self.assertEqual(proxy.session_affinity_events.get("hit", 0), 0)


class MetricsTests(_PinnedAffinity, unittest.IsolatedAsyncioTestCase):

    async def test_zero_baseline_is_exposed(self):
        proxy, lb, eps = _multi_proxy()
        previous = main.copilot_proxy
        main.copilot_proxy = proxy
        try:
            body = (await main.metrics()).body.decode()
        finally:
            main.copilot_proxy = previous
        for outcome in main._SESSION_AFFINITY_OUTCOMES:
            with self.subTest(outcome=outcome):
                self.assertIn(
                    f'copilot_session_affinity_total{{outcome="{outcome}"}} 0', body)


class SettingsTests(unittest.TestCase):

    def test_env_switch_defaults_true(self):
        prev = os.environ.pop("COPILOT_SESSION_AFFINITY", None)
        try:
            self.assertTrue(main.LBSettings.load().copilot_session_affinity)
            os.environ["COPILOT_SESSION_AFFINITY"] = "false"
            self.assertFalse(main.LBSettings.load().copilot_session_affinity)
        finally:
            os.environ.pop("COPILOT_SESSION_AFFINITY", None)
            if prev is not None:
                os.environ["COPILOT_SESSION_AFFINITY"] = prev


if __name__ == "__main__":
    unittest.main()
