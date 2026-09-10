"""三个业务入口的 HTTP 层契约（鉴权、尺寸门、错误形状、request-id）。

为什么补这一层：`proxy_request` / `_proxy` 的内部逻辑覆盖很厚，但**客户端真正看到的
东西**在 handler 里 —— 鉴权、`MAX_RAW_REQUEST_SIZE`、JSON 解析、准入 413、错误体形状、
`X-Request-Id`。这些此前零覆盖。它们出问题的表现很难归因：客户端只看到一个状态码。

关键契约（TROUBLESHOOTING §3 的判断口诀依赖它）：
  **LB 拒绝一律是 JSON**，HTML 413 只可能来自 ingress-nginx。所以这里逐个断言
  `content-type: application/json` 与 `error.type`。
"""
import json
import os
import sys
import unittest
import warnings
from unittest.mock import patch

# starlette 对 httpx 1.x 有弃用警告；这里只用它做 ASGI in-process 调用，
# 不值得为此引入 httpx2 依赖。过滤掉噪音，别污染每次测试输出。
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="starlette.testclient")
warnings.filterwarnings("ignore", message=".*httpx.*starlette.testclient.*")

from starlette.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main

KEY = "test-key-http-entrypoints"


class _StubProxy:
    """只提供 handler 鉴权/路由前置检查需要的最小面。

    这些用例断言的都是**上游之前**就该返回的结果（鉴权、尺寸门、JSON 解析、
    模型拒绝、准入 413），所以 stub 上的任何方法都不该被真正调用到。
    """

    def __init__(self):
        self.api_key = KEY

    def verify_api_key(self, key):
        return bool(key) and key == self.api_key


class _Client(unittest.TestCase):
    """把全局 proxy 换成 stub，只验 handler 层，不碰上游。

    注意三个 handler 都在鉴权**之前**先检查「provider 是否配置」并可能返 404，
    所以不装 stub 的话所有用例都会拿到 404 而不是预期的 401/413。
    """

    def setUp(self):
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[KEY] = "default"
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        for attr, value in (("proxy", _StubProxy()),
                            ("copilot_proxy", _StubProxy()),
                            ("azure_proxy", None)):
            patcher = patch.object(main, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def post(self, path, body, key=KEY, **kw):
        headers = kw.pop("headers", {})
        if key is not None:
            headers.setdefault("Authorization", f"Bearer {key}")
        return self.client.post(path, json=body, headers=headers, **kw)


class AuthTests(_Client):

    def test_all_three_entrypoints_reject_a_wrong_key(self):
        cases = [("/v1/messages", {"model": "claude-opus-5", "messages": []}),
                 ("/v1/responses", {"model": "gpt-test", "input": []}),
                 ("/v1/chat/completions", {"model": "gpt-test", "messages": []})]
        for path, body in cases:
            with self.subTest(path=path):
                r = self.post(path, body, key="wrong-key")
                self.assertEqual(r.status_code, 401, f"{path} 必须拒绝错误 key")

    def test_missing_key_is_rejected_not_treated_as_anonymous(self):
        for path, body in [("/v1/responses", {"model": "gpt-test", "input": []}),
                           ("/v1/chat/completions", {"model": "gpt-test", "messages": []})]:
            with self.subTest(path=path):
                r = self.post(path, body, key=None)
                self.assertEqual(r.status_code, 401)

    def test_empty_key_never_matches(self):
        """空 key 不能匹配任何 tenant —— 否则配置里出现空值就是 auth bypass。"""
        r = self.post("/v1/responses", {"model": "gpt-test", "input": []}, key="")
        self.assertEqual(r.status_code, 401)

    def test_admin_and_config_endpoints_are_gated(self):
        for path in ("/admin/copilot/reload", "/admin/copilot/reset-pool"):
            with self.subTest(path=path):
                self.assertEqual(self.post(path, {}, key="wrong").status_code, 401)
        self.assertEqual(
            self.client.get("/config/effective",
                            headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(
            self.client.request("DELETE", "/stats/history",
                                headers={"Authorization": "Bearer wrong"}).status_code, 401)



class OpenProbeTests(unittest.TestCase):
    """探针与 /metrics 必须免认证（K8s probe / Prometheus 抓取依赖）。

    单独一个类、不装 stub proxy —— `/metrics` 要读真实 proxy 上的计数器字段，
    薄 stub 会让它 500，那会掩盖「端点是否开放」这个真正要验的事。
    """

    def test_probes_and_metrics_stay_open(self):
        client = TestClient(main.app, raise_server_exceptions=False)
        for path in ("/health", "/health/live", "/metrics"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200)


class RejectionShapeTests(_Client):
    """LB 的拒绝一律是 JSON —— HTML 413 只可能来自 ingress（TROUBLESHOOTING §3）。"""

    def test_raw_size_limit_returns_json_413(self):
        body = {"model": "gpt-test", "input": [{"type": "message", "text": "x" * 200}]}
        with patch.object(main, "MAX_RAW_REQUEST_SIZE", 64):
            r = self.post("/v1/responses", body)
        self.assertEqual(r.status_code, 413)
        self.assertTrue(r.headers["content-type"].startswith("application/json"))
        self.assertEqual(r.json()["detail"]["error"]["type"], "request_too_large")

    def test_invalid_json_returns_400_json(self):
        r = self.client.post("/v1/responses", content=b"{not json",
                             headers={"Authorization": f"Bearer {KEY}",
                                      "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Invalid JSON", json.dumps(r.json()))

    def test_anthropic_model_on_openai_endpoint_is_rejected_clearly(self):
        for path in ("/v1/responses", "/v1/chat/completions"):
            with self.subTest(path=path):
                body = {"model": "claude-opus-5"}
                body["input" if path.endswith("responses") else "messages"] = []
                r = self.post(path, body)
                self.assertEqual(r.status_code, 400)
                self.assertIn("/v1/messages", json.dumps(r.json()),
                              "要告诉客户端该走哪个端点")

    def test_get_on_responses_is_501_with_allow_post(self):
        """Codex 的 background/polling 模式未实现。必须是 501 而不是 404/405 ——
        后两者会触发客户端指数重试风暴。"""
        r = self.client.get("/v1/responses")
        self.assertEqual(r.status_code, 501)
        self.assertEqual(r.headers.get("Allow"), "POST")
        self.assertEqual(r.json()["error"]["code"], "responses_get_not_supported")


class RequestIdTests(_Client):

    def test_response_carries_a_request_id_header(self):
        """客户端错误面板里的 id 要能直接跳到 LB 结构化日志。"""
        r = self.post("/v1/responses", {"model": "claude-opus-5", "input": []})
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.headers.get("X-Request-Id"), "拒绝响应也要带 X-Request-Id")

    def test_client_supplied_request_id_is_preserved(self):
        r = self.post("/v1/responses", {"model": "claude-opus-5", "input": []},
                      headers={"X-Request-Id": "client-supplied-id"})
        self.assertEqual(r.headers.get("X-Request-Id"), "client-supplied-id")


class ImageAdmissionOverHttpTests(_Client):
    """准入门在 HTTP 层的可见结果：JSON 413 + request_too_large。"""

    def test_too_many_images_yields_json_413_after_trim_cannot_help(self):
        if not main._PIL_AVAILABLE:
            self.skipTest("Pillow 不可用")
        import base64, io
        from PIL import Image as I
        buf = io.BytesIO(); I.new("RGB", (4000, 3000)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        block = {"type": "image", "source": {"type": "base64",
                                            "media_type": "image/png", "data": b64}}
        body = {"model": "gpt-test",
                "input": [{"role": "user", "content": [dict(block) for _ in range(2)]}]}
        # 门槛压到很低让准入必然运行；像素预算压到 1 张都过不了，trim 也救不回
        with patch.object(main, "_IMG_COMPRESS_THRESHOLD", 1), \
             patch.object(main, "_IMG_MAX_TOTAL_PIXELS", 1000), \
             patch.object(main, "_IMG_MAX_COUNT", 1):
            r = self.post("/v1/responses", body)
        self.assertEqual(r.status_code, 413)
        self.assertTrue(r.headers["content-type"].startswith("application/json"))
        self.assertEqual(r.json()["detail"]["error"]["type"], "request_too_large")


if __name__ == "__main__":
    unittest.main()
