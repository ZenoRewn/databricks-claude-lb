# 开发与接口约定

Author: Zeno Ren

本文件保留编码协作入口及完整 API 表。产品说明见 [README](README.md)，开发环境与模块边界见 [开发指南](docs/DEVELOPMENT.md)，全部文档见 [文档导航](docs/README.md)。仓库适用的 AGENTS.md 与用户当次授权决定操作边界。

## 不变量

- 推理 POST 的执行状态不明、已输出内容或可能产生工具副作用时，不自动重放；不能通过重新推理补账。见 [可靠性契约](docs/RESILIENCE.md)。
- 保留 provider/model 兼容性、账户 pinning 和会话亲和。默认 Copilot 优先；仅明确未准入或不支持模型等既有分支允许 Azure fallback，不能泛化成任意 503 后重试。per-model 优先序可以**重排**这两个 provider（`/admin/model-routing`），但只重排不筛除，fallback 必须保留；筛除是 per-key allowlist 的职责。
- `claude-*` 默认只走 Databricks 的 `/v1/messages`。**例外是逐模型 opt-in**：给某个 claude 模型显式配置指向 Copilot/Azure 的优先序后，该模型在 OpenAI 风格入口放行（Copilot 原生服务其 Responses/Chat 形状）。没配就保持 400，既有行为不变。`/v1/messages` 仍然只走 Databricks —— 从 Copilot 提供 Anthropic Messages 协议需要双向适配器，本仓库只有解析没有合成。
- **上游凭据可以由运维录入并持久化，但只以封装形式存储。** 这是对早先「网关从不存储凭据」的有意变更：AKS 上 Secret 挂载与 config 都是只读，应用 SA 无 secret 写权限，用量库是唯一可写的持久存储。AES-256-GCM，密钥取自 `LB_CREDENTIAL_KEY`（留在 Secret、不入库），每条独立 nonce，`provider|account` 作为 AAD 绑定行身份。**未配密钥时录入功能整体不可用（503），绝不退化为存明文。** 任何端点都不返回明文或密文。加密没有消除风险：运行进程内存里有明文，密钥丢失则凭据不可恢复。
- 凭据**录入时即向上游校验**，不通过不入库；端点仍在**启动时**统一构建（配置 + 已存凭据两个来源），所以新录入需一次滚动重启才生效。不做运行时增删端点 —— 那会重洗会话亲和哈希并打乱账户 pinning。
- 所有策略层共用一条降级规则：**配置是基线，策略存储是 overlay；overlay 缺失或未加载时回落配置基线。** provider allowlist 回落到「config 配了哪些 provider」、模型偏好回落到默认顺序、`allowed_models` 回落到 config 的 `models`。这些看着像 fail-open 只是因为那些基线恰好宽松 —— 规则本身是「回落基线」，不是「放行」。
- Copilot 账户启用态是该规则下的三态：`enabled: false` 声明「部署但不投入服务」，端点照常构建、在 Dashboard 可操作，但在被**显式启用**前不进任何候选集；基线是限制性的，所以同一条规则给出「保持 inert」。一次显式写入后由存储接管，启用与停用两个方向都归它裁决。判据是「运维决定过」而不是「有记录」：`discover` 是对上游的只读查询，它建记录但**不写** `enabled`（该列可空，`NULL` = 尚无决定），否则一次刷新能力就会把 inert 账户推上线。这沿用既有的「capability 是上游事实且只读，allowed_models 是运维意图」分离。
- 计价按**记录的模型名**区分渠道：`databricks-claude-*`（及裸 `opus-/sonnet-/haiku-/fable-` 旧别名）按 Anthropic 公开 API 价与「input 不含 cache」口径；裸 `claude-*` 来自 Copilot，按 copilot_pricing 分档价与 inclusive-input 口径。两者不可混用。
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
| `/v1/models`、`/v1/models/{id}`、`/models`、`/models/{id}` | GET | **可选** | 模型清单（客户端发现用），含 Databricks Claude 与 OpenAI 风格两侧。核验过的条目带非标准 `supported_endpoints` 字段标明可用入口（`/v1/messages`、`/v1/responses`、`/v1/chat/completions`），配置发现来的条目不带该字段而不是替上游断言。清单是**运维人工核验的时点快照，不是供应商能力认证**。第三种鉴权模式：**带了 key 就必须有效（否则 401），完全不带则放行** —— 见 `_verify_optional_models_auth` |
| `/health`、`/health/live` | GET | 不需要 | Liveness probe（仅检查进程） |
| `/health/ready` | GET | 不需要 | Readiness probe（检查依赖就绪；故障返回 503 + issues 数组） |
| `/health/accepting` | GET | 不需要 | 本地接流量就绪：初始化完成、路由已配置且未 draining；不因共享上游故障摘除整个网关 |
| `/metrics` | GET | 不需要 | Prometheus 文本格式 metrics（K8s / Azure Monitor 抓取） |
| `/admin/copilot/reload` | POST | 需要 | 运维端点：从源重读所有 Copilot endpoint 的 long-lived token + 强制刷新 session（K8s Secret rotation 后立刻生效） |
| `/admin/copilot/reset-pool` | POST | 需要 | 运维端点：重建共享 httpx.AsyncClient，逐出所有 keepalive/半开连接 |
| `/admin/model-capabilities` | GET | 需要 | 渠道/模型/API 能力目录、来源与有效期；未验证或过期能力不自动变成硬限额 |
| `/admin/credentials` | GET | 需要 | 已录入上游凭据的**身份**与校验结果；不返回明文也不返回密文，并标明哪些尚未生效 |
| `/admin/credentials/{provider}/{account}` | PUT | 需要 | 录入凭据：**先向上游校验，通过才封装入库**。与配置同名的 account 返回 409，不允许静默遮蔽；未配 `LB_CREDENTIAL_KEY` 时 503 |
| `/admin/credentials/{provider}/{account}` | DELETE | 需要 | 删除凭据（已生效端点到下次重启才消失） |
| `/admin/credentials/{provider}/{account}/validate` | POST | 需要 | 对已存凭据重新校验，会调一次上游 |
| `/admin/credentials/copilot/device/start` | POST | 需要 | 启动 GitHub device flow，返回验证码与授权链接；`device_code` 只留进程内，不落库不进日志不进响应 |
| `/admin/credentials/copilot/device/poll` | POST | 需要 | 轮询一次；`authorization_pending` 是正常中间态，拿到 token 后先校验再入库 |
| `/admin/model-routing` | GET | 需要 | per-model 的 provider 优先序与生效顺序 |
| `/admin/model-routing/{model}` | PUT | 需要 | 设置优先序（**顺序即偏好**，不去重不重排）。偏好只重排不筛除，fallback 保留；给 `claude-*` 配指向 Copilot/Azure 的偏好会在 OpenAI 风格入口放行该模型 |
| `/admin/model-routing/{model}` | DELETE | 需要 | 清除偏好回到默认顺序；`claude-*` 随之重新被 OpenAI 风格入口拒绝 |
| `/admin/copilot/accounts` | GET | 需要 | Copilot 账户的启用态、已发现能力（capability）与允许模型；只返回账户名与模型名，不返回 token。启用态是**三态**：`enabled` 是生效值，`enabled_in_config` 是配置声明的初始态，`policy_configured` 表明运维是否真的决定过。请求数为进程内累计 |
| `/admin/copilot/accounts/{account}` | PUT | 需要 | 切换账户启用态与允许模型。**只能在已发现 capability 内勾选**，空列表被拒（停用请用 `enabled: false`）；启用一个配置声明 inert、无 `models`、也从未发现过 capability 的账户返回 400 `capability_unknown`（启用即通配）；写失败时执行中的策略不变 |
| `/admin/copilot/accounts/{account}/discover` | POST | 需要 | 向上游重新发现该账户的模型能力并存快照。时点发现，不是能力认证；需带 editor 头部，否则 GitHub 返 403 |
| `/admin/api-keys` | GET | 需要 | 已激活密钥的**身份**与渠道限制：掩码 + sha256 前缀，**绝不返回明文**，短密钥连掩码都不给。请求数与最后使用时间是进程内计数，重启归零 |
| `/admin/api-keys/{tenant}/providers` | PUT | 需要 | 设置该租户允许的后端 provider（allowlist，只做减法）。写失败时执行中的策略保持不变，不假装成功 |
| `/admin/api-keys/reload` | POST | 需要 | 从 `CONFIG_PATH` 重读客户端密钥（同 `/admin/copilot/reload` 的模型：**源头轮换 + 网关重读**，网关不生成密钥）。校验在替换之前，坏配置不会把调用方锁在门外；`${ENV}` 值仍需滚动重启 |
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
