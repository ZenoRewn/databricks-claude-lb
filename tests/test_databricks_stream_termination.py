"""Databricks 路径的终端契约与失败归因。

这个文件锁两条**实测**结论，都是 pyflakes 的「赋值但从未使用」告警带出来的真缺陷：

**① `last_error` 被丢掉。** `ClaudeProxy.proxy_request` 从前在重试都失败后无条件抛一个
固定文案的 502，把实际异常类型扔了。于是 PoolTimeout / ConnectError / RemoteProtocolError
在客户端看来完全一样，而 `docs/RESILIENCE.md` 里 **503（连接获取/建立失败，未执行，
可重放）vs 502（执行不明，禁止重放）** 的语义区分在这条路径上整个丢失。Copilot 的
`_proxy` 一直是做对的，这里与它对齐。

**② `sent_message_start` 是死变量，而它要门控的行为本身没必要。** `CLAUDE.md` 曾声称
「若已发出 `message_start` 后 upstream 失败，先补发 `message_stop` 再发 `error`」——
实测 anthropic SDK 1.3.0：

| 收尾 | SDK 反应 |
|---|---|
| 只发 `event: error` | 抛 `APIStatusError`，带我们的错误消息 —— 干净 |
| 先 `message_stop` 再 `error` | **与上面完全相同**，补发毫无作用 |
| 直接断流、无任何终端 | **不抛异常**，静默返回部分文本 ← 真正危险的是这个 |

所以正确的不变量不是「补 message_stop」，而是**每条错误路径都必须发出 `event: error`**。
下面按这条不变量测。
"""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


def _make_proxy(threshold=99, n=1):
    endpoints = [main.WorkspaceEndpoint(name=f"ws-{i}", api_base=f"https://ws{i}.test/serving-endpoints",
                                       token="dapi-x")
                 for i in range(n)]
    lb = main.LoadBalancer(endpoints, circuit_breaker_threshold=threshold)
    proxy = object.__new__(main.ClaudeProxy)
    proxy.load_balancer = lb
    proxy.api_key = "k"
    proxy.global_stats = main.GlobalStats()
    proxy.today_model_stats = {}
    proxy.client = unittest.mock.Mock()
    return proxy, lb, endpoints


class FailureAttributionTests(unittest.IsolatedAsyncioTestCase):
    """重试都失败之后，客户端必须拿到「是什么失败了」以及正确的 503/502 语义。"""

    async def _drive(self, exc):
        proxy, lb, eps = _make_proxy()
        proxy.client.post = AsyncMock(side_effect=exc)
        with patch.object(main.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(main.HTTPException) as cm:
                await proxy.proxy_request({"model": "claude-opus-5", "messages": []},
                                          stream=False)
        return cm.exception, eps[0]

    async def test_connect_failure_is_503_and_names_the_failure(self):
        """连接获取/建立失败 = 未执行 = 可重放语义 → 503，且带 failure_type。"""
        for exc in (httpx.PoolTimeout("pool"), httpx.ConnectTimeout("ct"),
                    httpx.ConnectError("ce")):
            with self.subTest(exc=type(exc).__name__):
                e, _ep = await self._drive(exc)
                self.assertEqual(e.status_code, 503,
                                 "pre-execution 传输失败不能报成 502")
                err = e.detail["error"]
                self.assertEqual(err["failure_type"], type(exc).__name__,
                                 "从前这里被丢掉了，客户端拿不到任何线索")
                self.assertEqual(err["code"], "upstream_connect_failed")

    async def test_ambiguous_failure_stays_502_but_still_names_it(self):
        """读写失败/协议错误 = 执行不明 = 禁止重放 → 502，但仍要说清是什么。"""
        for exc in (httpx.ReadTimeout("rt"), httpx.RemoteProtocolError("rp"),
                    RuntimeError("boom")):
            with self.subTest(exc=type(exc).__name__):
                e, _ep = await self._drive(exc)
                self.assertEqual(e.status_code, 502)
                self.assertEqual(e.detail["error"]["failure_type"], type(exc).__name__)
                self.assertEqual(e.detail["error"]["code"], "upstream_request_failed")

    async def test_all_three_providers_share_the_failure_shape(self):
        """三个 provider 的收尾必须同形 —— 运维语义不该按 provider 分叉。

        Copilot 1 处（它一直是做对的）+ Databricks 2 处 + Azure 2 处（503/502 各一）。
        """
        src = __import__("pathlib").Path(main.__file__).read_text(encoding="utf-8")
        self.assertEqual(src.count('"failure_type": type(last_error).__name__'), 5)
        self.assertEqual(src.count('"code": "upstream_connect_failed"'), 2,
                         "Databricks + Azure 各一处 503 分支")


class StreamTerminalContractTests(unittest.IsolatedAsyncioTestCase):
    """每条错误路径都必须发出 `event: error`。

    实测（anthropic SDK 1.3.0）：只发 `error` 就足够让 SDK 抛 `APIStatusError`；
    补 `message_stop` 毫无作用；而**没有任何终端**会让 SDK 静默返回部分内容 ——
    那是唯一真正危险的收尾方式，所以这里守的是「一定有 error 帧」。
    """

    async def _collect(self, outcomes, body=None):
        proxy, lb, eps = _make_proxy()
        proxy.client = _StreamClient(outcomes)
        await lb.on_request_start(eps[0])
        res = await proxy._stream_request(
            eps[0], f"{eps[0].api_base}/anthropic/v1/messages",
            body or {"model": "claude-opus-5", "messages": []}, {},
            model="claude-opus-5", start_time=main.time.time())
        out = b"".join([c if isinstance(c, bytes) else c.encode()
                        async for c in res.body_iterator])
        return out, eps[0]

    async def test_http_error_before_any_frame_emits_error_event(self):
        out, ep = await self._collect([_ErrStream(500, b'{"error":{"message":"boom"}}')])
        self.assertIn(b"event: error", out)
        self.assertEqual(ep.active_requests, 0)

    async def test_truncated_stream_emits_error_event_not_silence(self):
        """上游发了 message_start 就断 —— 必须补 error 帧。

        若这里什么都不发，SDK 会**静默返回部分文本**（实测），客户端以为拿到了
        一个短答案。这是本仓库最不想要的失败形态。
        """
        out, ep = await self._collect([_OkStream([
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1"}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"partial"}}\n\n',
        ])])
        self.assertIn(b"event: error", out)
        self.assertIn(b"upstream_truncated", out)
        self.assertIn(b"partial", out, "已发出的内容不能被吞掉")
        self.assertEqual(ep.active_requests, 0)

    async def test_network_error_mid_stream_emits_error_event(self):
        out, ep = await self._collect([httpx.ReadError("dropped")])
        self.assertIn(b"event: error", out)
        self.assertEqual(ep.active_requests, 0)

    async def test_clean_completion_needs_no_synthetic_terminal(self):
        out, ep = await self._collect([_OkStream([
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1"}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ])])
        self.assertNotIn(b"event: error", out)
        self.assertIn(b"message_stop", out)
        self.assertEqual(ep.active_requests, 0)

    def test_no_dead_message_stop_gate_remains(self):
        """`sent_message_start` 已删除：它门控的行为实测无效，留着只会误导后人
        以为「补 message_stop」是必要的。"""
        src = __import__("pathlib").Path(main.__file__).read_text(encoding="utf-8")
        self.assertNotIn("sent_message_start", src)


class _OkStream:
    def __init__(self, chunks):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self.http_version = "HTTP/1.1"
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    async def aread(self):
        return b"".join(self._chunks)

    async def aclose(self):
        self.closed = True


class _ErrStream:
    def __init__(self, status, body):
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self.http_version = "HTTP/1.1"
        self._body = body
        self.closed = False

    async def aread(self):
        return self._body

    async def aclose(self):
        self.closed = True


class _StreamClient:
    def __init__(self, outcomes):
        self._it = iter(outcomes)

    def build_request(self, method, url, json=None, headers=None):
        return httpx.Request(method, url, json=json, headers=headers)

    async def send(self, request, stream=False):
        o = next(self._it)
        if isinstance(o, BaseException):
            raise o
        return o


if __name__ == "__main__":
    unittest.main()
