# 开发与接口约定

Author: Zeno Ren

本文件保留编码协作入口及完整 API 表。产品说明见 [README](README.md)，开发环境与模块边界见 [开发指南](docs/DEVELOPMENT.md)，全部文档见 [文档导航](docs/README.md)。仓库适用的 AGENTS.md 与用户当次授权决定操作边界。

## 不变量

- 推理 POST 的执行状态不明、已输出内容或可能产生工具副作用时，不自动重放；不能通过重新推理补账。见 [可靠性契约](docs/RESILIENCE.md)。
- 保留 provider/model 兼容性、账户 pinning 和会话亲和。Copilot 优先；仅明确未准入或不支持模型等既有分支允许 Azure fallback，不能泛化成任意 503 后重试。
- HTTP 200 或首个 SSE chunk 不代表生成完成。保留成功终态、失败、incomplete、refusal 和工具语义，见 [流协议](docs/STREAM_PROTOCOL.md)。
- MySQL 账本幂等防止已持久批次重复累计；尚未 flush 的内存事件仍可能丢失，不能承诺零 RPO。JSON 后端只支持单写者。
- Dashboard 的未知费用保持未知，部分计价保留已知小计。Copilot 实时估算采用 [GHCP 价格口径](docs/COPILOT_PRICING.md)，历史参考价不能混称实际渠道账单。
- Dashboard 保持独立 HTML 资产，不引入前端框架构建依赖。UI 与发布文件清单一起核验；运行版本来自构建元数据，不从当前 Git 分支或硬编码推断。

## 验证入口

```bash
python -m pytest tests/ -q --tb=short
node tests/test_copilot_pricing_ui.cjs
node tests/test_dashboard_ui.cjs
```

行为变更新增失败测试后实现；文案与布局采用链接、结构及渲染验证。发布工具与应用分开验收，独立 QA 不以自测代替。AKS 操作遵守 [部署指南](docs/AKS.md) 和 [发布工具](docs/RELEASE_TOOL.md)，不得整份 apply 示例 manifest 覆盖既有环境。

## 上游 401 按来源分类

Copilot 的 opaque-state 拒绝是请求级问题，不应污染账户熔断；无 opaque state 的凭据/席位/策略失败仍按端点级处理。既有修复阶梯有明确预算，并保持同账户、同 lease。细节与回归见 [重试和熔断契约](docs/RESILIENCE.md)。

连接池排障使用 `httpx_pool_observed_full` 这一观测字段；池等待超时本身不能证明 TCP/TLS 故障。不要恢复已撤回的二分类诊断。症状入口见 [排障指南](docs/TROUBLESHOOTING.md)。

## API 端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `/v1/messages` | POST | 需要 | Databricks Claude 消息 API（仅 `claude-*` 模型） |
| `/v1/messages/count_tokens` | POST | 需要 | 本地输入 Token 估算，包含 system/tools；响应头声明低置信度及未知图片/状态开销，不能作为精确硬阈值 |
| `/v1/responses` | POST | 需要 | OpenAI Responses API（按模型分流：Copilot 优先 → Azure fallback；`claude-*` 拒绝） |
| `/v1/responses`、`/v1/responses/{tail}` | GET | 不需要 | **501** + `Allow: POST`：Codex 的 background/polling 模式未实现。必须是 501 而不是 404/405 —— 后两者会触发客户端指数重试风暴 |
| `/v1/chat/completions` | POST | 需要 | OpenAI Chat Completions API（按模型分流：Copilot 优先 → Azure fallback；`claude-*` 拒绝） |
| `/v1/models`、`/v1/models/{id}`、`/models`、`/models/{id}` | GET | **可选** | 模型清单（客户端发现用）。第三种鉴权模式：**带了 key 就必须有效（否则 401），完全不带则放行** —— 见 `_verify_optional_models_auth` |
| `/health`、`/health/live` | GET | 不需要 | Liveness probe（仅检查进程） |
| `/health/ready` | GET | 不需要 | Readiness probe（检查依赖就绪；故障返回 503 + issues 数组） |
| `/health/accepting` | GET | 不需要 | 本地接流量就绪：初始化完成、路由已配置且未 draining；不因共享上游故障摘除整个网关 |
| `/metrics` | GET | 不需要 | Prometheus 文本格式 metrics（K8s / Azure Monitor 抓取） |
| `/admin/copilot/reload` | POST | 需要 | 运维端点：从源重读所有 Copilot endpoint 的 long-lived token + 强制刷新 session（K8s Secret rotation 后立刻生效） |
| `/admin/copilot/reset-pool` | POST | 需要 | 运维端点：重建共享 httpx.AsyncClient，逐出所有 keepalive/半开连接 |
| `/admin/model-capabilities` | GET | 需要 | 渠道/模型/API 能力目录、来源与有效期；未验证或过期能力不自动变成硬限额 |
| `/config/effective` | GET | 需要 | 返回实际生效设置与构建身份，不用于公开版本展示 |
| `/version` | GET | 不需要 | 有界公开构建标识：源码提交、GitHub 链接、运行文件匹配状态；无配置/凭据，不宣称 GitHub 最新状态 |
| `/stats` | GET | 不需要 | 端点统计（含成本估算、Azure OpenAI、GitHub Copilot） |
| `/stats/history` | GET | 不需要 | 历史用量数据（`?days=7`，含每日成本） |
| `/stats/history` | DELETE | **需要** | 清理历史数据（`?keep_days=30`）—— P0.3 加 auth |
| `/stats/dashboard` | GET | 不需要 | 可视化工作台（Databricks / Azure / Copilot / 历史），含运行版本与深浅主题；response 头带 CSP + X-Content-Type-Options: nosniff |
| `/reset` | POST | **需要** | 重置内存统计（持久化数据保留）—— P0.3 加 auth |
| `/api/event_logging/batch` | POST | 不需要 | **故意的空 sink**：handler 不接 `Request`、**从不读 body**、无条件返 `{"status":"ok"}`。存在的唯一目的是让 Codex 的遥测 POST 不拿到 404（否则客户端刷错误日志）。**不要给它加鉴权**，也不要让它去解析 body —— 它不落盘、不转发、不计数，所以「无鉴权」在这里不构成暴露面 |

> 这张表是**完整**的路由清单（`/openapi.json`、`/docs`、`/redoc` 除外），由
> `tests/test_api_surface_contract.py` 双向机械核验：表里的每一行必须真实存在，
> 每条真实路由必须在表里，且「认证」列与实际行为一致。加了新路由忘了写文档会红。
