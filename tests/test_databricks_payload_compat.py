import asyncio
import importlib
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

    # -- 模型名映射 ---------------------------------------------------------

    def test_claude_opus_5_maps_to_databricks_opus_5(self):
        # Databricks 端已 GA opus-5，客户端发 claude-opus-5-* 应直通，不再降级到 4-7
        self.assertEqual(
            main.get_databricks_model("claude-opus-5"),
            "databricks-claude-opus-5",
        )
        self.assertEqual(
            main.get_databricks_model("claude-opus-5-20260101"),
            "databricks-claude-opus-5",
        )
        # 大小写不敏感
        self.assertEqual(
            main.get_databricks_model("Claude-Opus-5"),
            "databricks-claude-opus-5",
        )

    def test_opus_5_all_variants_bypass_downgrade(self):
        """回归防护：任何 opus-5 变体都必须映射到 databricks-claude-opus-5，
        绝不能被通用 opus 分支吞掉降级到 4-7。列出常见客户端可能发出的形态。"""
        variants = [
            "databricks-claude-opus-5",   # 已经是 Databricks 名，原样返回
            "opus-5",                     # 裸模型名
            "claude-opus-5",              # 无日期后缀
            "claude-opus-5-latest",       # latest 别名
            "claude-opus-5-20250514",     # 20250514 date suffix
            "claude-opus-5-20260101",     # 未来 date suffix
            "claude-opus-5.1",            # 小版本号（次要修订）
            "CLAUDE-OPUS-5",              # 全大写
            "Claude-Opus-5",              # 混合大小写
            "claude-opus-5_20260101",     # 下划线分隔（非常见但要 robust）
        ]
        for v in variants:
            self.assertEqual(
                main.get_databricks_model(v),
                "databricks-claude-opus-5",
                msg=f"variant {v!r} was NOT mapped to opus-5 (regression!)",
            )

    def test_claude_opus_4_x_still_maps_correctly(self):
        # 加 opus-5 分支后，旧版本映射必须保持不变
        self.assertEqual(
            main.get_databricks_model("claude-opus-4-7"),
            "databricks-claude-opus-4-7",
        )
        self.assertEqual(
            main.get_databricks_model("claude-opus-4-5"),
            "databricks-claude-opus-4-5",
        )
        # 无版本号的 "claude-opus" 走默认（4-7），不能被误识别为 5
        self.assertEqual(
            main.get_databricks_model("claude-opus"),
            "databricks-claude-opus-4-7",
        )

    def test_new_2026_models_have_pricing(self):
        """回归：2026-09 新增的模型必须都有定价，避免 get_model_pricing 静默返 None。"""
        required = [
            # Anthropic 新增
            "databricks-claude-sonnet-5",
            "databricks-claude-opus-4-8",
            # OpenAI 新增 / 修正
            "o4-mini", "o3-pro",
            "gpt-5-pro",
            "gpt-5.2-pro", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro",
            "gpt-5.5-pro",
            "gpt-5.6-cyber",
            "gpt-5.3-codex",
        ]
        for m in required:
            p = main.get_model_pricing(m)
            self.assertIsNotNone(p, msg=f"{m}: no pricing")
            self.assertGreater(p["input"], 0, msg=f"{m}: zero input price")
            self.assertGreater(p["output"], 0, msg=f"{m}: zero output price")

    def test_openai_default_catalog_all_priced(self):
        """/models 端点默认 catalog 里每个 model 都必须能计费——否则 /stats 里
        estimated_total_cost_usd 会因 pricing=None 少统计一部分请求。"""
        for m in main.OPENAI_COMPAT_DEFAULT_MODEL_IDS:
            self.assertIsNotNone(main.get_model_pricing(m), msg=f"{m}: missing pricing")

    def test_sonnet_5_maps_to_databricks_sonnet_5(self):
        """Sonnet 5 与 Opus 5 使用相同的 regex 分支模式（数字 5 boundary）。"""
        for m in ["sonnet-5", "claude-sonnet-5", "claude-sonnet-5-latest",
                  "claude-sonnet-5-20260101", "Claude-Sonnet-5"]:
            self.assertEqual(
                main.get_databricks_model(m),
                "databricks-claude-sonnet-5",
                msg=f"{m}: not mapped to sonnet-5",
            )
        # 未指定 5 时仍走默认（4-6）
        self.assertEqual(main.get_databricks_model("claude-sonnet"), "databricks-claude-sonnet-4-6")
        # 显式 4-6 保持不变
        self.assertEqual(main.get_databricks_model("claude-sonnet-4-6"), "databricks-claude-sonnet-4-6")

    def test_opus_5_pricing_matches_opus_4_7_placeholder(self):
        # 子串匹配：opus-5 的定价 key 长度大于 opus，应优先命中
        p_opus_5 = main.get_model_pricing("databricks-claude-opus-5")
        p_opus_4_7 = main.get_model_pricing("databricks-claude-opus-4-7")
        self.assertIsNotNone(p_opus_5)
        self.assertEqual(p_opus_5, p_opus_4_7)

    def test_supports_adaptive_thinking_default_true_for_new_models(self):
        # 黑名单策略：不在旧模型标记里的都默认 adaptive
        # 用户线索：Opus 5 未被原白名单覆盖 → 修好后必须返 True
        self.assertTrue(main.supports_adaptive_thinking("databricks-claude-opus-5"))
        self.assertTrue(main.supports_adaptive_thinking("databricks-claude-opus-4-7"))
        self.assertTrue(main.supports_adaptive_thinking("databricks-claude-opus-4-6"))
        self.assertTrue(main.supports_adaptive_thinking("databricks-claude-sonnet-4-6"))
        # 未来假想版本，代码零改动应默认支持
        self.assertTrue(main.supports_adaptive_thinking("databricks-claude-opus-6"))
        self.assertTrue(main.supports_adaptive_thinking("databricks-claude-sonnet-5"))

    def test_supports_adaptive_thinking_false_for_legacy(self):
        # 旧模型（4-5 系列）仍需 enabled + budget_tokens
        self.assertFalse(main.supports_adaptive_thinking("databricks-claude-opus-4-5"))
        self.assertFalse(main.supports_adaptive_thinking("databricks-claude-sonnet-4-5"))

    def test_supports_adaptive_thinking_handles_bad_input(self):
        # 非字符串输入不该崩：None 直接返 False；空串按黑名单默认 True（走 adaptive，
        # 让上游对空 model 报清晰错误，比在 LB 里静默转 enabled+budget_tokens 更好）
        self.assertFalse(main.supports_adaptive_thinking(None))
        self.assertFalse(main.supports_adaptive_thinking(123))  # 非 str
        self.assertTrue(main.supports_adaptive_thinking(""))


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


class OpenAICompatTests(unittest.TestCase):
    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers

    def test_extracts_api_key_from_common_client_headers(self):
        self.assertEqual(
            main._extract_api_key(self.FakeRequest({"api-key": " game-key "})),
            "game-key",
        )
        self.assertEqual(
            main._extract_api_key(self.FakeRequest({"authorization": "bearer game-key"})),
            "game-key",
        )
        self.assertEqual(
            main._extract_api_key(
                self.FakeRequest({"api-key": "game-key", "authorization": "Bearer local"})
            ),
            "game-key",
        )
        self.assertEqual(
            main._extract_api_key(self.FakeRequest({"authorization": "game-key"})),
            "game-key",
        )
        self.assertEqual(
            main._extract_api_key(self.FakeRequest({}), x_api_key=" game-key "),
            "game-key",
        )

    def test_collects_model_catalog_from_azure_copilot_and_wildcard_defaults(self):
        class FakeLoadBalancer:
            def __init__(self, endpoints):
                self.endpoints = endpoints

        class FakeProxy:
            def __init__(self, endpoints):
                self.load_balancer = FakeLoadBalancer(endpoints)

        azure = FakeProxy([
            main.AzureOpenAIEndpoint(
                name="az-east",
                endpoint="https://example.openai.azure.com",
                api_key="key",
                deployments=["gpt-4o", "gpt-4.1"],
            )
        ])
        copilot = FakeProxy([
            main.CopilotEndpoint(
                name="gh-all",
                github_token="token",
                token_source={"type": "literal"},
                models=[],
            ),
            main.CopilotEndpoint(
                name="gh-explicit",
                github_token="token",
                token_source={"type": "literal"},
                models=["gemini-2.5-pro"],
            ),
        ])

        ids = main._collect_openai_model_ids(azure, copilot)

        self.assertEqual(ids, sorted(set(ids)))
        self.assertIn("gpt-4.1", ids)
        self.assertIn("gpt-4o", ids)
        self.assertIn("gpt-5.5", ids)
        self.assertIn("gpt-5.6-sol", ids)
        self.assertIn("gpt-5.6-luna", ids)
        self.assertIn("gpt-5.6-terra", ids)
        self.assertIn("gpt-5-codex", ids)
        self.assertIn("gemini-2.5-pro", ids)

    def test_openai_models_payload_has_data_plus_empty_codex_models_key(self):
        # `data` 是 OpenAI 官方规范字段（Python/JS SDK 都读它）—— 包含完整 catalog。
        # `models` 是 codex-cli 0.145.0 私有必需字段：必须存在，且每个 entry 要求
        # slug/display_name/default_reasoning_level/... 十几个必填字段。填不全
        # 会连续报 "missing field `slug`" → "missing field `display_name`" ...；
        # 而我们对每个模型的 reasoning tier / service_tiers 元数据没有真实来源，
        # 所以返 `models: []` 让 codex-cli 认为"provider 未声明 catalog"，
        # 回落到 config.toml 的 `model = "..."` 或 `-m` 指定，功能不受影响。
        class FakeLB:
            def __init__(self, eps): self.endpoints = eps
        class FakeCopilot:
            def __init__(self, eps): self.load_balancer = FakeLB(eps)
        fake_copilot = FakeCopilot([
            main.CopilotEndpoint(name="unit-fake", github_token="tok",
                                  token_source={"type":"literal"}, models=["gpt-5.6-sol"]),
        ])
        original = main.copilot_proxy
        main.copilot_proxy = fake_copilot
        try:
            payload = main._build_openai_models_payload()
        finally:
            main.copilot_proxy = original
        self.assertEqual(payload["object"], "list")
        # data 里携带完整 catalog（供 OpenAI 标准客户端消费）
        self.assertGreaterEqual(len(payload["data"]), 1)
        for entry in payload["data"]:
            self.assertIn("id", entry)
            self.assertEqual(entry["object"], "model")
        # models 存在但为空（让 codex-cli 满足字段存在检查，跳过 entry schema 校验）
        self.assertIn("models", payload)
        self.assertEqual(payload["models"], [])

    def test_routes_gpt_5_6_chat_models_through_responses_adapter(self):
        self.assertTrue(main._should_adapt_chat_to_responses("gpt-5.6-sol"))
        self.assertTrue(main._should_adapt_chat_to_responses("GPT-5.6-LUNA"))
        self.assertTrue(main._should_adapt_chat_to_responses(" gpt-5.6-terra "))

    def test_uses_specific_gpt_5_6_pricing(self):
        # 2026-09-08 与 OpenAI 官方 dev docs 对齐后的价：
        # sol 4/20, terra 2/12, luna 0.20/1.20 (short-context tier)
        self.assertEqual(main.get_model_pricing("gpt-5.6-sol")["input"], 4.00)
        self.assertEqual(main.get_model_pricing("gpt-5.6-terra")["output"], 12.00)
        # OpenAI 无独立 cache write 收费，全部归 0
        self.assertEqual(main.get_model_pricing("gpt-5.6-terra")["cache_write"], 0.0)
        # luna cached input 折扣价 = $0.02/M
        self.assertEqual(main.get_model_pricing("gpt-5.6-luna")["cache_read"], 0.02)
        # 新加入的 cyber 层级验证一下
        self.assertEqual(main.get_model_pricing("gpt-5.6-cyber")["input"], 12.50)
        self.assertEqual(main.get_model_pricing("gpt-5.6-cyber")["output"], 75.00)

    def test_drops_only_nonpositive_chat_token_limits(self):
        body = {
            "model": "gpt-4.1",
            "max_tokens": 0,
            "max_completion_tokens": "0",
            "temperature": 0,
        }

        removed = main._drop_nonpositive_token_limits(body)

        self.assertEqual(removed, ["max_tokens", "max_completion_tokens"])
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["temperature"], 0)

    def test_builds_responses_payload_from_chat_body(self):
        body = {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "stay in character"},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 0,
            "max_completion_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look something up",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }

        payload = main._build_responses_payload_from_chat(body)

        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertEqual(
            payload["input"],
            [
                {"role": "system", "content": "stay in character"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(payload["max_output_tokens"], 512)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["name"], "lookup")
        self.assertEqual(payload["tool_choice"], "auto")

    def test_reports_removed_chat_to_responses_sampling_fields(self):
        body = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.3,
            "top_p": 0.9,
        }

        removed = main._responses_adapter_removed_sampling_fields(body)

        self.assertEqual(removed, ["temperature", "top_p"])

    def test_strips_unsupported_sampling_fields_from_direct_responses_body(self):
        body = {
            "model": "gpt-5.5",
            "input": "hello",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 100,
        }

        removed = main._strip_unsupported_responses_sampling_fields(body)

        self.assertEqual(removed, ["temperature", "top_p"])
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertEqual(body["max_output_tokens"], 100)

    def test_does_not_classify_unsupported_parameter_as_unsupported_model(self):
        body = '{"error":{"message":"Unsupported parameter: \\"temperature\\" is not supported with this model.","code":"invalid_request_body"}}'

        self.assertFalse(main.CopilotProxy._is_unsupported_model_error(400, body))

    def test_classifies_true_copilot_model_errors_as_unsupported_model(self):
        self.assertTrue(main.CopilotProxy._is_unsupported_model_error(404, '{"error":{"code":"model_not_found"}}'))
        self.assertTrue(main.CopilotProxy._is_unsupported_model_error(400, '{"error":{"message":"unknown model"}}'))
        self.assertTrue(main.CopilotProxy._is_unsupported_model_error(400, '{"error":{"code":"unsupported_model"}}'))
        self.assertTrue(main.CopilotProxy._is_unsupported_model_error(400, '{"error":{"message":"unknown model","code":"invalid_request_body"}}'))

    def test_wraps_responses_text_as_chat_completion(self):
        response = {
            "id": "resp_123",
            "output_text": "hello",
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }

        chat = main._responses_json_to_chat_completion(response, "gpt-5.5")

        self.assertEqual(chat["object"], "chat.completion")
        self.assertEqual(chat["model"], "gpt-5.5")
        self.assertEqual(chat["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(chat["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            chat["usage"],
            {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        )

    def test_wraps_responses_function_call_as_chat_tool_call(self):
        response = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "lookup",
                    "arguments": "{\"q\":\"hi\"}",
                }
            ]
        }

        chat = main._responses_json_to_chat_completion(response, "gpt-5.5")

        message = chat["choices"][0]["message"]
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["id"], "call_123")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(chat["choices"][0]["finish_reason"], "tool_calls")

    def test_copilot_status_summary_includes_endpoint_state(self):
        class FakeLoadBalancer:
            def __init__(self, endpoints):
                self.endpoints = endpoints

        class FakeCopilot:
            def __init__(self, endpoints):
                self.load_balancer = main.LoadBalancer(endpoints)

        old_copilot = main.copilot_proxy
        try:
            main.copilot_proxy = FakeCopilot([
                main.CopilotEndpoint(
                    name="gh-all",
                    github_token="token",
                    token_source={"type": "literal"},
                    models=[],
                )
            ])
            message = main._build_no_provider_message("gpt-5.5")
        finally:
            main.copilot_proxy = old_copilot

        self.assertIn("Copilot has selectable endpoint(s) for model 'gpt-5.5': gh-all", message)
        self.assertNotIn("configured but failing", message)


class OpenAICompatAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_copilot_unsupported_model_error_without_azure(self):
        class RejectingCopilot:
            def can_handle(self, model, api_type=None):
                return True

            async def proxy_responses(self, body, stream=False, disconnect_checker=None,
                                      request_id=None):
                raise main._UnsupportedModelError("404: unsupported")

        old_copilot, old_azure = main.copilot_proxy, main.azure_proxy
        try:
            main.copilot_proxy = RejectingCopilot()
            main.azure_proxy = None
            with self.assertRaises(main.HTTPException) as ctx:
                await main._route_openai({"model": "gpt-5.5"}, stream=False, api_type="responses")
        finally:
            main.copilot_proxy, main.azure_proxy = old_copilot, old_azure

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail["error"]["code"], "unsupported_model")
        self.assertIn("Copilot upstream rejected model 'gpt-5.5'", ctx.exception.detail["error"]["message"])
        self.assertIn("copilot_status", ctx.exception.detail["error"])


class LatencyHistogramTests(unittest.TestCase):
    def test_observe_populates_buckets_and_totals(self):
        h = main.LatencyHistogram(buckets=(0.1, 1.0, 10.0))
        h.observe("copilot", "responses", 0.05)   # <=0.1, tenant="default"
        h.observe("copilot", "responses", 0.5)
        h.observe("copilot", "responses", 5.0)
        h.observe("copilot", "responses", 120.0)  # +Inf only

        # observe() stores cumulative counts directly (Prom-shape).
        # P3.2: key now is (provider, api_type, tenant); default tenant="default".
        row = h.counts[("copilot", "responses", "default")]
        self.assertEqual(row[0], 1)  # <=0.1
        self.assertEqual(row[1], 2)  # <=1.0
        self.assertEqual(row[2], 3)  # <=10.0
        self.assertEqual(row[3], 4)  # +Inf

        count, total = h.totals[("copilot", "responses", "default")]
        self.assertEqual(count, 4)
        self.assertAlmostEqual(total, 0.05 + 0.5 + 5.0 + 120.0, places=6)

    def test_render_prom_emits_cumulative_buckets_and_sum_count(self):
        h = main.LatencyHistogram(buckets=(0.1, 1.0))
        h.observe("azure", "chat", 0.05)
        h.observe("azure", "chat", 0.5)
        h.observe("azure", "chat", 5.0)  # only +Inf

        text = h.render_prom("proxy_request_latency")
        self.assertIn('proxy_request_latency_seconds_bucket{provider="azure",api_type="chat",tenant="default",le="0.1"} 1', text)
        self.assertIn('proxy_request_latency_seconds_bucket{provider="azure",api_type="chat",tenant="default",le="1.0"} 2', text)
        self.assertIn('proxy_request_latency_seconds_bucket{provider="azure",api_type="chat",tenant="default",le="+Inf"} 3', text)
        self.assertIn('proxy_request_latency_seconds_count{provider="azure",api_type="chat",tenant="default"} 3', text)
        self.assertIn('proxy_request_latency_seconds_sum{provider="azure",api_type="chat",tenant="default"} 5.550000', text)

    def test_observe_per_tenant_separates_series(self):
        # P3.2: 同 provider/api_type 不同租户产出独立系列
        h = main.LatencyHistogram(buckets=(0.1, 1.0))
        h.observe("copilot", "chat", 0.05, tenant="tenant-a")
        h.observe("copilot", "chat", 0.05, tenant="tenant-b")
        h.observe("copilot", "chat", 0.5, tenant="tenant-a")

        row_a = h.counts[("copilot", "chat", "tenant-a")]
        row_b = h.counts[("copilot", "chat", "tenant-b")]
        self.assertEqual(row_a[-1], 2)  # +Inf = 2 for tenant-a
        self.assertEqual(row_b[-1], 1)

        text = h.render_prom("proxy_request_latency")
        self.assertIn('tenant="tenant-a"', text)
        self.assertIn('tenant="tenant-b"', text)

    def test_render_empty_yields_only_help_type_headers(self):
        h = main.LatencyHistogram()
        text = h.render_prom("proxy_request_latency")
        self.assertIn("# HELP proxy_request_latency_seconds", text)
        self.assertIn("# TYPE proxy_request_latency_seconds histogram", text)
        # No sample rows — the two header lines only
        self.assertEqual(len([l for l in text.splitlines() if not l.startswith("#")]), 0)


class OpenTelemetrySetupTests(unittest.TestCase):
    """P3.1: OTel setup 是否 opt-in、缺 packages 时是否优雅降级."""

    def test_setup_returns_false_when_disabled(self):
        import os
        from otel_setup import setup_tracing
        os.environ.pop("OTEL_ENABLED", None)
        # 假 app —— setup 不会碰到 FastAPI instrumentor
        self.assertFalse(setup_tracing(object()))

    def test_setup_returns_true_when_enabled_and_packages_present(self):
        import os
        # `import opentelemetry` 是不够的 —— 它是 namespace package，装了任意一个
        # otel 发行包（例如 azure-monitor-opentelemetry 传递带入 opentelemetry-api）
        # 就 import 成功，而 setup_tracing() 真正需要下面这几个子模块。用 namespace
        # 做 guard 会让「装了一部分 otel 包」的机器跳不过去、直接断言失败。
        for mod in ("opentelemetry.trace",
                    "opentelemetry.sdk.resources",
                    "opentelemetry.sdk.trace",
                    "opentelemetry.sdk.trace.export",
                    "opentelemetry.instrumentation.fastapi",
                    "opentelemetry.instrumentation.httpx"):
            try:
                importlib.import_module(mod)
            except ImportError:
                self.skipTest(f"opentelemetry tracing deps incomplete: missing {mod}")

        os.environ["OTEL_ENABLED"] = "true"
        app = None
        try:
            from otel_setup import setup_tracing
            # 用真正的 FastAPI app（instrumentor 会 attach middleware）
            from fastapi import FastAPI
            app = FastAPI()
            ok = setup_tracing(app)
            self.assertTrue(ok)
        finally:
            os.environ.pop("OTEL_ENABLED", None)
            # setup_tracing 会全局打补丁（HTTPXClientInstrumentor().instrument()
            # patch 掉整个 httpx，且自己从不撤销），不还原会把 instrumentation
            # 状态和一个仍在后台导出 span 的 BatchSpanProcessor 泄漏给后续测试。
            try:
                from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
                HTTPXClientInstrumentor().uninstrument()
            except Exception:
                pass
            if app is not None:
                try:
                    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
                    FastAPIInstrumentor.uninstrument_app(app)
                except Exception:
                    pass
            # BatchSpanProcessor 的后台线程若不 shutdown，进程退出时会拿已关闭的
            # exporter 报 "Exception while exporting Span"。
            try:
                from opentelemetry import trace
                provider = trace.get_tracer_provider()
                shutdown = getattr(provider, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            except Exception:
                pass

    def test_get_tracer_returns_noop_when_packages_missing(self):
        # 即使真装了包，get_tracer 也能 gracefully 用；这里主要看它不 raise。
        from otel_setup import get_tracer
        t = get_tracer()
        with t.start_as_current_span("test-span"):
            pass  # no-op or real span both fine


class LBSettingsTests(unittest.TestCase):
    """P2.3: LBSettings dataclass 集中所有 env vars."""

    def test_default_values_match_documented(self):
        # 显式在环境里清空这些 var，让 defaults 生效
        import os
        keys = [
            "STREAM_HEARTBEAT_INTERVAL", "COPILOT_HTML_SOFT_COOLDOWN",
            "COPILOT_POOL_ACQUIRE_TIMEOUT", "COPILOT_POOL_MAX_CONNECTIONS",
            "COPILOT_STREAM_HIGH_WATERMARK", "COPILOT_HTTP2",
            "IMG_COMPRESS_CONCURRENCY", "IMG_MAX_COUNT",
        ]
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            s = main.LBSettings.load()
            self.assertEqual(s.stream_heartbeat_interval, 15.0)
            self.assertEqual(s.copilot_html_soft_cooldown, 30.0)
            self.assertEqual(s.copilot_pool_acquire_timeout, 20.0)
            self.assertEqual(s.copilot_pool_max_connections, 500)
            self.assertEqual(s.copilot_stream_high_watermark, 400)
            self.assertTrue(s.copilot_http2)  # P2.6 opt-out default
            self.assertEqual(s.img_compress_concurrency, 2)
            self.assertEqual(s.img_max_count, 50)
            self.assertEqual(s.img_max_total_pixels, 100_000_000)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_no_dual_sourced_env_default_drift(self):
        """同一个 env 被「模块级常量」与「LBSettings」分别读取时，默认值必须一致。

        LBSettings 是 introspection 层（`GET /config/effective`），散点的
        `os.environ.get` 才是实际执行时读的 backing store。两边默认值一旦不同，
        运维不设该 env 时 introspection 就在说谎 —— 而这个端点存在的唯一目的就是
        防止这种情况。曾真的漂移过：`IMG_MAX_TOTAL_PIXELS` 报 200M、执行 100M，
        查 413 的人照 introspection 会得出「没超预算」的错误结论。

        这里比的是**运行期实际值**而不是源码字面量，所以 `"30"` vs `30.0` 这类
        形式差异不会误报。
        """
        import os
        pairs = [
            ("IMG_ADMISSION_ENABLED", "_IMG_ADMISSION_ENABLED", "img_admission_enabled"),
            ("IMG_COMPRESS_CONCURRENCY", "_IMG_COMPRESS_CONCURRENCY", "img_compress_concurrency"),
            ("IMG_MAX_COUNT", "_IMG_MAX_COUNT", "img_max_count"),
            ("IMG_MAX_TOTAL_PIXELS", "_IMG_MAX_TOTAL_PIXELS", "img_max_total_pixels"),
        ]
        saved = {env: os.environ.pop(env, None) for env, _, _ in pairs}
        try:
            settings = main.LBSettings.load()
            for env, const_name, field in pairs:
                with self.subTest(env=env):
                    self.assertEqual(
                        getattr(main, const_name), getattr(settings, field),
                        f"{env}: 执行侧 {const_name} 与 introspection 侧 "
                        f"LBSettings.{field} 默认值不一致",
                    )
        finally:
            for env, v in saved.items():
                if v is not None:
                    os.environ[env] = v

    def test_env_overrides_apply(self):
        import os
        os.environ["STREAM_HEARTBEAT_INTERVAL"] = "5.5"
        try:
            s = main.LBSettings.load()
            self.assertEqual(s.stream_heartbeat_interval, 5.5)
        finally:
            os.environ.pop("STREAM_HEARTBEAT_INTERVAL", None)

    def test_as_dict_is_json_serializable(self):
        import json
        s = main.LBSettings.load()
        d = s.as_dict()
        json.dumps(d)  # 不 raise 即证明所有字段都是 scalar
        # 必须包含关键字段
        self.assertIn("copilot_pool_acquire_timeout", d)
        self.assertIn("stream_heartbeat_interval", d)


class MultiTenantApiKeysTests(unittest.TestCase):
    """P3.2: auth.api_key (str) 与 auth.api_keys (dict) 都要注册到全局 map."""

    def setUp(self):
        # 每个测试都清空 map，防污染
        main.API_KEY_TO_TENANT.clear()

    def tearDown(self):
        main.API_KEY_TO_TENANT.clear()

    def test_register_single_key_becomes_default_tenant(self):
        main._register_api_keys("single-key-xyz", None)
        self.assertEqual(main.API_KEY_TO_TENANT, {"single-key-xyz": "default"})
        self.assertEqual(main._lookup_tenant("single-key-xyz"), "default")
        self.assertIsNone(main._lookup_tenant("wrong-key"))

    def test_register_multi_tenant_keys(self):
        main._register_api_keys("", {
            "team-a": "key-a",
            "team-b": "key-b",
        })
        self.assertEqual(main._lookup_tenant("key-a"), "team-a")
        self.assertEqual(main._lookup_tenant("key-b"), "team-b")
        self.assertIsNone(main._lookup_tenant("unknown"))

    def test_register_both_single_and_multi_coexist(self):
        main._register_api_keys("legacy-key", {"team-x": "key-x"})
        self.assertEqual(main._lookup_tenant("legacy-key"), "default")
        self.assertEqual(main._lookup_tenant("key-x"), "team-x")

    def test_register_rejects_empty_tenant_name(self):
        with self.assertRaises(ValueError):
            main._register_api_keys("", {"": "some-key"})

    def test_register_rejects_empty_key_value(self):
        with self.assertRaises(ValueError):
            main._register_api_keys("", {"team-x": ""})

    def test_lookup_tenant_empty_key_returns_none(self):
        main._register_api_keys("legacy-key", None)
        self.assertIsNone(main._lookup_tenant(""))
        self.assertIsNone(main._lookup_tenant(None))


class BackgroundLoopSelfHealTests(unittest.IsolatedAsyncioTestCase):
    """P2.5: 后台 loop 不再因单轮迭代抛错而静默退出."""

    async def test_background_refresh_loop_survives_outer_scope_error(self):
        # 让 load_balancer.endpoints 属性访问 raise —— 模拟真正的
        # "outer-scope" 意外错误（不是 per-endpoint 分支已经 catch 住的场景）
        proxy = object.__new__(main.CopilotProxy)
        calls = {"n": 0}
        class ExplodingLB:
            @property
            def endpoints(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("simulated outer failure")
                return []  # 第 2 次正常返回
        proxy.load_balancer = ExplodingLB()

        # interval=0 让循环紧凑；self-heal 后 sleep 5s，我们及时 cancel
        task = asyncio.create_task(proxy.background_refresh_loop(interval=0, threshold=1))
        # 等第一轮抛错被 self-heal 捕获（sleep 5s）
        for _ in range(20):
            await asyncio.sleep(0.01)
            if calls["n"] >= 1:
                break
        task.cancel()
        # loop 内部 break 后正常返回；不管是 CancelledError 还是 clean return，
        # 都不应该 raise 其他类型的异常
        try:
            await task
        except asyncio.CancelledError:
            pass  # normal cancellation propagation
        # 第一次抛错后 loop 没有退出——self-heal 有效
        self.assertGreaterEqual(calls["n"], 1)


class ProviderAgnosticHtmlCooldownTests(unittest.TestCase):
    """P1.3: _note_upstream_html + LoadBalancer._prefer_html_fresh 三条 proxy 通用."""

    def _fake_proxy(self):
        class Proxy:
            upstream_html_events_total = 0
            upstream_html_events_by_status: dict = {}
        return Proxy()

    def test_note_upstream_html_sets_cooldown_and_counters(self):
        # Databricks endpoint 也有 html_soft_cooldown_until 字段（P1.3 加）
        ep = main.WorkspaceEndpoint(name="db-1", api_base="https://x", token="t")
        proxy = self._fake_proxy()
        proxy.upstream_html_events_by_status = {}
        main._note_upstream_html(proxy, ep, "Databricks", "messages", 502,
                                 upstream_ids={"cf-ray": "abc-SIN"})
        self.assertEqual(ep.upstream_html_events_total, 1)
        self.assertEqual(proxy.upstream_html_events_total, 1)
        self.assertEqual(proxy.upstream_html_events_by_status, {"5xx": 1})
        self.assertGreater(ep.html_soft_cooldown_until, main.time.time())

    def test_note_upstream_html_status_buckets(self):
        proxy = self._fake_proxy()
        proxy.upstream_html_events_by_status = {}
        ep = main.AzureOpenAIEndpoint(name="az-1", endpoint="https://x", api_key="k")
        for status in (200, 403, 502, 999):
            main._note_upstream_html(proxy, ep, "Azure", "chat", status)
        self.assertEqual(proxy.upstream_html_events_by_status,
                         {"200": 1, "4xx": 1, "5xx": 1, "other": 1})

    def test_prefer_html_fresh_skips_cooldown_endpoints(self):
        ep_a = main.WorkspaceEndpoint(name="a", api_base="https://x", token="t")
        ep_b = main.WorkspaceEndpoint(name="b", api_base="https://y", token="t")
        ep_a.html_soft_cooldown_until = main.time.time() + 30
        chosen = main.LoadBalancer._prefer_html_fresh([ep_a, ep_b])
        # 只有 ep_b 不在冷却窗口 → 返回 [ep_b]
        self.assertEqual([e.name for e in chosen], ["b"])

    def test_prefer_html_fresh_degrades_when_all_in_cooldown(self):
        # 全部处于 cooldown 时，仍然返回原候选（保可用性；下次踩 HTML 会刷新窗口）
        ep_a = main.WorkspaceEndpoint(name="a", api_base="https://x", token="t")
        ep_b = main.WorkspaceEndpoint(name="b", api_base="https://y", token="t")
        now = main.time.time()
        ep_a.html_soft_cooldown_until = now + 30
        ep_b.html_soft_cooldown_until = now + 30
        chosen = main.LoadBalancer._prefer_html_fresh([ep_a, ep_b])
        self.assertEqual([e.name for e in chosen], ["a", "b"])

    def test_prefer_html_fresh_ignores_legacy_endpoints_without_field(self):
        # 老代码里没 html_soft_cooldown_until 字段的对象也不 crash
        class LegacyEndpoint:
            name = "legacy"
        chosen = main.LoadBalancer._prefer_html_fresh([LegacyEndpoint()])
        self.assertEqual(chosen[0].name, "legacy")


if __name__ == "__main__":
    unittest.main()
