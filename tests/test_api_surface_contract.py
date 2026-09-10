"""`CLAUDE.md` 的 API 端点表 ↔ 真实路由，双向机械核验 + 鉴权行为核验。

为什么值得单独一层：2026-09-10 审计发现那张表**漏了 7 条真实路由** —— 其中
`POST /api/event_logging/batch` 完全无鉴权、`/v1/models` 家族是表格「需要/不需要」
二元格式表达不了的**第三种模式**（带 key 就必须有效、不带则放行）。表自称是端点清单，
于是任何照它做安全复核或客户端接入的人，都会漏掉这些。

这类缺陷行为测试抓不到（每条路由自己都是对的），只有把「文档 = 事实」变成断言才抓得到。
它同时是**面向未来**的：以后加了新路由却忘了写进表，这里就会红。

鉴权一律**行为核验**（发一个错 key 看是不是 401），不靠 grep handler 源码 —— 审计时
我第一版探测器就漏了真正的 helper 名 `_verify_lb_api_key`，把 5 个已鉴权端点误报成裸奔。
"""
import os
import pathlib
import re
import sys
import unittest
import warnings
from unittest.mock import patch

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="starlette.testclient")

from starlette.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main

KEY = "test-key-api-surface"
# FastAPI 的内建路由不属于业务契约
IGNORED_PREFIXES = ("/openapi", "/docs", "/redoc")

REQUIRED, OPEN, OPTIONAL = "需要", "不需要", "可选"


def _norm(path: str) -> str:
    """把路径参数名归一化：文档写 `{tail}` / `{id}`，代码是 `{tail:path}` /
    `{model_id}`。契约是「这条路由存在」，不是「参数叫什么」。"""
    return re.sub(r"\{[^}]*\}", "{}", path)


def _documented():
    """解析 CLAUDE.md 的端点表 → {(归一化路径, 方法): 认证要求}。"""
    text = (pathlib.Path(main.__file__).parent / "CLAUDE.md").read_text(encoding="utf-8")
    table = {}
    for line in text.splitlines():
        m = re.match(r'^\|\s*(`[^|]+`)\s*\|\s*([A-Z]+)\s*\|\s*([^|]+?)\s*\|', line)
        if not m:
            continue
        paths = re.findall(r'`([^`]+)`', m.group(1))
        method = m.group(2)
        need = m.group(3).replace("*", "").strip()
        if need not in (REQUIRED, OPEN, OPTIONAL):
            continue
        for p in paths:
            table[(_norm(p), method)] = need
    return table


def _actual():
    routes = {}
    for r in main.app.routes:
        path, methods = getattr(r, "path", None), getattr(r, "methods", None)
        if not path or not methods or path.startswith(IGNORED_PREFIXES):
            continue
        for method in sorted(methods - {"HEAD", "OPTIONS"}):
            routes[(_norm(path), method)] = r
    return routes


class _StubProxy:
    """三个业务入口在鉴权**之前**先检查 provider 是否配置并可能返 404，
    所以不装 stub 的话所有用例都会拿到 404 而不是预期的 401。"""

    def __init__(self):
        self.api_key = KEY

    def verify_api_key(self, key):
        return bool(key) and key == self.api_key


class TableCompletenessTests(unittest.TestCase):
    """双向：表里的每一行必须存在；每条真实路由必须在表里。"""

    def test_every_documented_route_exists(self):
        actual = _actual()
        missing = sorted(k for k in _documented() if k not in actual)
        self.assertEqual(missing, [], "文档表列了不存在的路由")

    def test_every_real_route_is_documented(self):
        documented = _documented()
        undocumented = sorted(k for k in _actual() if k not in documented)
        self.assertEqual(
            undocumented, [],
            "这些真实路由不在 CLAUDE.md 的 API 端点表里 —— 表自称完整，"
            "漏了就会让人在做接入或安全复核时看不见它们")

    def test_the_table_is_not_trivially_empty(self):
        """防解析器悄悄失效后两个方向都「通过」。"""
        self.assertGreaterEqual(len(_documented()), 15)


class AuthBehaviourMatchesTheTableTests(unittest.TestCase):
    """用真实请求核验「认证」列，不 grep 源码。"""

    def setUp(self):
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[KEY] = "default"
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        for attr in ("proxy", "copilot_proxy"):
            p = patch.object(main, attr, _StubProxy())
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(main, "azure_proxy", None)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def _call(self, path, method, headers):
        # 路径参数填一个无害的占位值
        url = re.sub(r"\{\}", "placeholder", path)
        return self.client.request(method, url, headers=headers,
                                   json={} if method in ("POST", "PUT", "PATCH") else None)

    def test_auth_column_is_truthful_for_every_row(self):
        for (path, method), need in sorted(_documented().items()):
            with self.subTest(route=f"{method} {path}", need=need):
                wrong = self._call(path, method, {"Authorization": "Bearer definitely-wrong"})
                if need == REQUIRED:
                    self.assertEqual(wrong.status_code, 401,
                                     "标了「需要」却没拒绝错误 key")
                elif need == OPEN:
                    self.assertNotEqual(wrong.status_code, 401,
                                        "标了「不需要」却拒绝了请求")
                else:  # 可选
                    self.assertEqual(wrong.status_code, 401,
                                     "「可选」的语义是：带了 key 就必须有效")
                    anon = self._call(path, method, {})
                    self.assertNotEqual(anon.status_code, 401,
                                        "「可选」的语义是：完全不带 key 要放行")


class UndocumentedBehaviourNowPinnedTests(unittest.TestCase):
    """审计新补进表的三条路由，把它们的实际语义钉住。"""

    def setUp(self):
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[KEY] = "default"
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        for attr, value in (("proxy", _StubProxy()), ("copilot_proxy", _StubProxy()),
                            ("azure_proxy", None)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def test_event_logging_sink_never_reads_the_body(self):
        """它是给 Codex 遥测的空 sink。handler 不接 `Request` —— 这不是疏漏而是
        设计：不读 body 才谈得上「无鉴权也不构成暴露面」。"""
        import inspect
        sig = inspect.signature(main.event_logging)
        self.assertEqual(list(sig.parameters), [],
                         "一旦开始接受 Request / body，无鉴权就需要重新评估")
        src = inspect.getsource(main.event_logging)
        for forbidden in ("await request", "request.body", "request.json", "logger"):
            self.assertNotIn(forbidden, src, "空 sink 不应落盘/转发/记录任何东西")
        r = self.client.post("/api/event_logging/batch", json={"events": [1, 2, 3]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_responses_get_is_501_with_allow_post(self):
        """501 而非 404/405 —— 后两者会触发客户端指数重试风暴。"""
        for url in ("/v1/responses", "/v1/responses/resp_abc123"):
            with self.subTest(url=url):
                r = self.client.get(url)
                self.assertEqual(r.status_code, 501)
                self.assertEqual(r.headers.get("Allow"), "POST")

    def test_models_listing_rejects_a_wrong_key_but_allows_none(self):
        for url in ("/v1/models", "/models"):
            with self.subTest(url=url):
                self.assertEqual(
                    self.client.get(url, headers={"Authorization": "Bearer nope"}).status_code,
                    401)
                self.assertNotEqual(self.client.get(url).status_code, 401)

    def test_provider_configuration_is_checked_before_auth(self):
        """本仓库所有 handler 一致的既有顺序：provider 未配置先返 404，再谈鉴权。

        钉住它是因为这直接决定上面那批鉴权断言必须装 stub proxy 才有意义 ——
        不装的话每条都拿到 404，测试「通过」但什么都没验到（我第一版就踩了）。
        """
        with patch.object(main, "copilot_proxy", None), \
             patch.object(main, "azure_proxy", None):
            r = self.client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        self.assertEqual(r.status_code, 404, "未配置 provider 时 404 优先于 401")


if __name__ == "__main__":
    unittest.main()
