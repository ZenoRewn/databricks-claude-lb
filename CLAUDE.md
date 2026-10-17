# 开发与接口约定

Author: Zeno Ren

本文件保留编码协作入口及完整 API 表。产品说明见 [README](README.md)，开发环境与模块边界见 [开发指南](docs/DEVELOPMENT.md)，全部文档见 [文档导航](docs/README.md)。仓库适用的 AGENTS.md 与用户当次授权决定操作边界。

## 不变量

- 推理 POST 的执行状态不明、已输出内容或可能产生工具副作用时，不自动重放；不能通过重新推理补账。见 [可靠性契约](docs/RESILIENCE.md)。
- 保留 provider/model 兼容性、账户 pinning 和会话亲和。默认 Copilot 优先；仅明确未准入或不支持模型等既有分支允许 Azure fallback，不能泛化成任意 503 后重试。per-model 优先序可以**重排**这两个 provider（`/admin/model-routing`），但只重排不筛除，fallback 必须保留；provider 维度的筛除是 per-key allowlist 的职责。
- per-model 的 **GHCP 账户选择**（`/admin/model-routing/{model}` 的 `copilot_accounts`）筛的是**同一 provider 内的哪个账户**，不是哪个 provider —— 所以它是筛除，却不与上一条冲突，provider 级 fallback 完全不受影响。它与 per-account 的 `allowed_models` 是**交集**：前者把账户从模型上摘掉，后者把模型从账户上摘掉，任一侧成立即移出候选。排查某账户为何不接流量时两处都要看。选中的账户恰好都服务不了该模型时 Copilot 候选为空，请求落到 Azure —— 与那些账户全部熔断同效，不是错误。筛选只能加在 `CopilotEndpoint.supports()`：那是所有候选判断的唯一收口，含会话亲和用的「配置态合格集合」，漏掉亲和那一处会在回落时把所有会话重新洗牌。条件本体住在 `_copilot_support_reason()` 里，`supports()` 退化成「没有排除原因」—— 收口没变，新条件仍然只能加在那个函数里。
- **「为什么不接流量」的答案与候选判断是同一份代码。** Dashboard 要说出是哪一层摘掉了某个账户，而那个解释一旦与判断分成两份实现必然漂移，漂移的那份恰好是被人当真的诊断 —— 比没有诊断更糟。所以 reason 是主实现、布尔是派生：`supports()` ≡ `_copilot_support_reason(...) is None`，`key_policy.account_allows` / `model_allows_account` 同样委托给各自的 reason 读者。原因是**闭集**（`main.COPILOT_EXCLUSION_REASONS`），不是自由文本，否则 UI 无从分支、测试无从断言。两处分别必须保留：`not_in_capability` 是上游事实（修法是重新发现），`excluded_by_allowed_models` 是运维意图（修法是改勾选）；`account_disabled`（有人关过）与 `account_never_enabled`（从没开过）同理。**`capability_unknown` 不在这个集合里**：没发现过能力的账户服务所有模型，那是通配警告而不是排除原因，由 `capability_known` 单独表达。reason 与 capability 一样是**时点**判断，不是能力认证。
- **「已启用」与「实际接流量」是两个状态。** 账户可以开着却一个模型都服务不了 —— 因为另一层（per-model 账户选择）把它从每个已知模型上摘掉了。`/admin/copilot/accounts` 的 `receiving_traffic` / `serves_models` 单独报这件事，计数口径是「时点目录 ∪ 已发现能力」而不是「世上所有模型」。只显示启用开关会让人读成「开着就是在用」。
- `claude-*` 默认只走 Databricks 的 `/v1/messages`。**例外是逐模型 opt-in**：给某个 claude 模型显式配置指向 Copilot/Azure 的优先序后，该模型在 OpenAI 风格入口放行（Copilot 原生服务其 Responses/Chat 形状）。没配就保持 400，既有行为不变。`/v1/messages` 仍然只走 Databricks —— 从 Copilot 提供 Anthropic Messages 协议需要双向适配器，本仓库只有解析没有合成。
- **上游凭据可以由运维录入并持久化，但只以封装形式存储。** 这是对早先「网关从不存储凭据」的有意变更：AKS 上 Secret 挂载与 config 都是只读，应用 SA 无 secret 写权限，用量库是唯一可写的持久存储。AES-256-GCM，密钥取自 `LB_CREDENTIAL_KEY`（留在 Secret、不入库），每条独立 nonce，`provider|account` 作为 AAD 绑定行身份。**未配密钥时录入功能整体不可用（503），绝不退化为存明文。** 任何端点都不返回明文或密文。加密没有消除风险：运行进程内存里有明文，密钥丢失则凭据不可恢复。
- 凭据**录入时即向上游校验**，不通过不入库；端点仍在**启动时**统一构建（配置 + 已存凭据两个来源），所以新录入需一次滚动重启才生效。不做运行时增删端点 —— 那会重洗会话亲和哈希并打乱账户 pinning。
- 所有策略层共用一条降级规则：**配置是基线，策略存储是 overlay；overlay 缺失或未加载时回落配置基线。** provider allowlist 回落到「config 配了哪些 provider」、模型偏好回落到默认顺序、`allowed_models` 回落到 config 的 `models`、per-model 账户选择回落到「不限定」。这些看着像 fail-open 只是因为那些基线恰好宽松 —— 规则本身是「回落基线」，不是「放行」。
- Copilot 账户启用态是该规则下的三态：`enabled: false` 声明「部署但不投入服务」，端点照常构建、在 Dashboard 可操作，但在被**显式启用**前不进任何候选集；基线是限制性的，所以同一条规则给出「保持 inert」。一次显式写入后由存储接管，启用与停用两个方向都归它裁决。判据是「运维决定过」而不是「有记录」：`discover` 是对上游的只读查询，它建记录但**不写** `enabled`（该列可空，`NULL` = 尚无决定），否则一次刷新能力就会把 inert 账户推上线。这沿用既有的「capability 是上游事实且只读，allowed_models 是运维意图」分离。
- 计价按**记录的模型名**区分渠道：`databricks-claude-*`（及裸 `opus-/sonnet-/haiku-/fable-` 旧别名）按 Anthropic 公开 API 价与「input 不含 cache」口径；裸 `claude-*` 来自 Copilot，按 copilot_pricing 分档价与 inclusive-input 口径。两者不可混用。
- HTTP 200 或首个 SSE chunk 不代表生成完成。保留成功终态、失败、incomplete、refusal 和工具语义，见 [流协议](docs/STREAM_PROTOCOL.md)。
- MySQL 账本幂等防止已持久批次重复累计；尚未 flush 的内存事件仍可能丢失，不能承诺零 RPO。JSON 后端只支持单写者。
- Dashboard 的未知费用保持未知，部分计价保留已知小计。Copilot 实时估算采用 [GHCP 价格口径](docs/COPILOT_PRICING.md)，历史参考价不能混称实际渠道账单。
- Dashboard 保持独立 HTML 资产，不引入前端框架构建依赖。UI 与发布文件清单一起核验；运行版本来自构建元数据，不从当前 Git 分支或硬编码推断。
- **「密钥与渠道」页的页面脚本不持有任何凭据。** 解锁是把面板密码（`LB_DASHBOARD_PASSWORD`）或一个有效 API Key 换成 **HttpOnly 会话 cookie**，此后由浏览器携带，脚本读不到也就偷不走 —— 这比早先把 key 放在闭包变量里更强，不是只为省事。会话是服务端进程内的 30 分钟滑动窗口，重启全失效，锁定时**服务端一并失效**（只清浏览器那份会留下一个仍可用的会话）。刷新页面保留会话，这正是目的：过去刷新就等于重输。两条入口都换成同一种会话，所以「脚本不持凭据」在回落到 API Key 时同样成立。未配密码或密码短于 12 字符时面板登录整体不可用，**绝不退化为无密码放行** —— `/admin/*` 由没有 OAuth 的那个 ingress 提供，对公网可达。密码按来源指数退避（5 次后 30s 起翻倍、有上限），退避期内**正确密码也拒**且不区分原因，否则它就成了一个旁路的密码预言机；退避只计密码路径，让它挡住高熵的 API Key 等于把猜测攻击变成自伤。**上游凭据（`credSecret`）仍是单次请求后立即清除** —— 会话是便利，那个是秘密。
- **锁定态是「内容不渲染」，不是「内容模糊」。** 被门禁的那一整块用 `hidden`（`display:none`），因此同时退出 tab 序与无障碍树。纯 CSS 模糊**不是门禁** —— 模糊后的内容仍可 Tab 到、仍被屏幕阅读器完整朗读；而最常见的补救「给外层加 `aria-hidden`」本身是 WCAG 4.1.2 Level A 失败（axe `aria-hidden-focus`，容器含可聚焦控件）。毛玻璃只是解锁卡片自身的观感，不拿它冒充门禁；这一块也不标 `aria-modal`，它是页面区域而不是对话框。
- **面板密码存的是单向验证器，不是密码。** 上游凭据必须可重放给上游所以用 AES 封装；面板密码只需要**验证**，所以存 PBKDF2-SHA256（算法与迭代数随哈希一起存，调成本不作废旧记录），**即使 `LB_CREDENTIAL_KEY` 泄露也推不出密码**，且这条路径不依赖那把密钥。代价写明：**忘记即不可恢复**，回头路是用一把有效 LB API Key 解锁后重设。优先级沿用那条降级规则：存储是 overlay、`LB_DASHBOARD_PASSWORD` 是基线，overlay 缺失或未加载回落 env，**两者都没有则登录整体不可用而不是放行**。PBKDF2 是同步 CPU 操作，必须 `asyncio.to_thread`，否则阻塞事件循环。**策略存储只在启动时加载**，所以多副本下页内改密码只对当前副本即时生效，其余副本重启前仍认 env 里的旧密码 —— 改完要同步 Secret 再滚动重启。

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

## 事件时间与重试预算

- **每条结构化事件自带 `event_time`（和 `event_monotonic`）。** `started_at_unix` 是**请求**的起点，同一请求的所有事件共用它 —— 拿它当事件时间，一个重试了六分钟的请求就会被压成同一瞬间。2026-10-16 的 GHCP 复盘正是这样把熔断跳闸、试探准入、试探结果三件事都标到了请求起点，**撤回了四条表述**。**线上没有设置 `LOG_FORMAT`，走默认 text formatter，一个时间字段都没有** —— 所以当时只能退回 `kubectl logs --timestamps`（容器运行时的时间，不是事件的）。注意 `deploy/k8s/` 是示例形态而非线上形态，判断生产事实只能读 live 对象；json 模式下的 `ts` 也只是 formatter 加在外层，导出与二次解析可能只带走载荷。新增事件字段必须进 `safe_diagnostics` 白名单，否则会被静默过滤。
- **重试继承剩余预算，不另拿一份完整预算。** 每次 attempt 的启动预算 = `min(固定预算, 剩余 deadline)`，由 `effective_startup_seconds()` 收口，日志里 `startup_budget_source` 说明是否被 deadline 绑住。剩余预算低于 `UPSTREAM_MIN_RETRY_SECONDS` 时**不开始**重试：注定完不成的重试只换来拖尾延迟、上游负担与重放暴露。这不改「执行状态不明不重放」的边界 —— 安全性由 `_replay_safe_transport_failure` 判定，预算由 `retry_within_budget()` 判定，两个问题两个函数。
- **「这个账户为什么不接流量」与候选判断同源，熔断恢复状态必须可见而不是可推。** 试探准入事件带 `small_input_evidence` / `preference_window_remaining` / `admitted_payload_small` / `admitted_estimate_complete`；inconclusive 结算带 `inconclusive_kind` 与 `upstream_headers_received`（未收到响应头的取消对上游零证据，收到了的至少证明上游接了流）。`resilience_stats` 暴露 `cooldown_elapsed_seconds` / `seconds_since_last_trial` / `last_trial_outcome` / `trial_in_flight_age_seconds`。**attempt 数与业务请求数是两个口径**：一次可重放失败最多重试 3 次，所以阈值 5 可以由 2 个业务请求打满；转换事件用 `contributing_attempts` / `contributing_requests` 把这件事摆出来，而**不**改计数语义 —— 改成按请求计数会让端点真死时保护晚得多。
- **Claude 的三条路径都说 Anthropic Messages，优先序按协议面划分。** 2026-10-17 实测（[证据](docs/reviews/2026-10-17-claude-three-paths/EVIDENCE.md)）：Databricks、GitHub Copilot、Microsoft Foundry 都能在 Anthropic Messages 协议上服务 Claude —— GHCP 自己的 `/models` 对每个 claude 条目声明 `supported_endpoints: ["/v1/messages","/chat/completions"]`，Foundry 的 `POST /anthropic/v1/messages` 返回标准 `type: "message"`。**所以不需要协议合成器。** 由此有四条约束：① 优先序是**每个协议面各一份**：`/v1/messages` 面排 `databricks / copilot / azure_anthropic`（默认 ADB → GHCP → Foundry），OpenAI 风格面排 `copilot / azure_openai`；某个 provider 出现在它服务不了的面上会被忽略（`azure_openai` 之于 messages 面、`databricks` 之于 OpenAI 面）。两面读**同一份**存储的偏好，各自过滤。② **同一模型三种拼法**，差异会出错不是观感：ADB `databricks-claude-opus-5-5`、GHCP `claude-opus-5.5`（点）、Foundry `claude-opus-5-5`（杠），拿点号问 Foundry 会 404 `DeploymentNotFound`。改写收口在 `claude_model_identity.py`，该模块只管**身份**不管**可用性** —— 可用性仍归 config / 已发现 capability / 运维声明的 deployment。③ **`azure_anthropic` 与 `azure_openai` 是同一个 Foundry 资源的两个面，必须分开**：协议、鉴权头（实测只有 `x-api-key` 与 Bearer 可用，官方也列的 `api-key` 回 401）、模型 id、计费单位（CCU vs 按 token 列价）全不同；合并会让 per-key allowlist 无法表达「Foundry 上允许 gpt 但不允许 Claude」。④ **计价口径不可混用**：`MODEL_PRICING` 是按模型名子串匹配的，Foundry 的 `claude-opus-5-5` 会命中 `opus-5-5` 从而被套上 Anthropic 公开价 —— 必须经 `_pricing_for_model(model, provider=...)` 判为未知，`None` 比一个自信的错数字好，0 会在面板里汇总成「免费」。GHCP 的模型条目还自带 `warning_message` 说已改 usage-based、分档乘数不再适用。⑤ **GHCP 有没有 Claude 取决于账户 SKU**：实测 enterprise seat 0 个、business seat 各 10 个，所以这条路径可用性必须按账户算，不能按 provider 假设。
- **一条 h2 连接不能用到服务端开始拒绝它。** Copilot 开着 HTTP/2，于是**所有**流量复用池里同一条连接（实测 `httpx_pool.total=1`）。服务端对单连接的流数有一个没有文档的上限，而且撞上去时**不发 GOAWAY**：它只对每条新流回 REFUSED_STREAM（code 7），对已接受的流在精确 120.0 秒空闲后回 CANCEL（code 8）。httpcore 的 `is_available()` 只看 `_used_all_stream_ids`、`_connection_error` 和 h2 状态机，这三样静默拒绝一个都不碰，所以池子会把那条死连接交给每一个新请求**包括重试** —— 2026-10-17 因此出现 685→689 / 695→699 / 697→701 三对「重试撞回同一条死连接」，把错误数翻倍并填满熔断阈值。修法是**优雅退休**：置 `_used_all_stream_ids` 只让连接退出新流分配，在途流照常跑完（httpcore 关连接的条件里没有「不可用」），即补上服务端没发的那个 GOAWAY。触发条件三个：REFUSED_STREAM（**在重试之前**退休）、收到响应头后静默超阈值、stream id 达到 `COPILOT_H2_RETIRE_STREAM_ID`。**不要用重建 client 代替** —— 那会打断在途响应，只适合手动自救（`/admin/copilot/reset-pool`）。这条路径依赖 httpx/httpcore 私有属性，因为没有公开 API；赌注由 `tests/test_h2_connection_lifecycle.py::PoolContractTests` 钉住，库升级会在那里红，而不是在生产静默失效。**这不是请求体积问题**：同样 0.9–3.0 MB 的 body 在断崖前完成了 112 个，压缩请求体不解决任何东西；体积只是占用时长的代理。
- **「这个请求带不带图」只有一个实现**：`model_capabilities.payload_has_image()`，与 `estimate_payload` 共用遍历。它不只是日志字段 —— `_has_image` 门控 `Copilot-Vision-Request` 头，所以漏判等于把带图请求裸发给上游。旧实现只看顶层 `messages`/`input` 且要求 `content` 是 list，于是 tool 输出里的图、mapping 形 `content`、Anthropic 的 `type: image`、裸 `data:` URL 全漏，并与同一个 body 的 `image_count` 对不上（2026-10-16 实测 16 vs False）。假阳性那侧由遍历本身兜住：`tools`/`response_format`/`text`/`output_config` 是 definition 子树，描述图片的 schema 不算图片。

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
| `/admin/session` | GET | **不需要** | 面板会话状态：`authenticated` / `expires_in` / `password_configured` / `auth_methods`。**故意不鉴权** —— 这是 Dashboard 加载后的第一个请求，用来回答「要不要登录、怎么登录」，要求凭据才能问这个问题是循环的。只回布尔与秒数，不回显密码或 token |
| `/admin/session` | POST | **密码** | 第四种鉴权模式：凭据是 `LB_DASHBOARD_PASSWORD` 而**不是** API key，所以带错 key 不会 401，而没有正确密码绝不放行。成功则下发 HttpOnly 会话 cookie（`SameSite=Strict`、`Path=/admin`、30 分钟滑动窗口，`Secure` 仅 https）。未配密码或密码短于 12 字符时 503；连续失败 5 次后按 IP 指数退避，退避期内**正确密码也拒**且不区分原因（否则它就成了一个旁路的密码预言机）。响应体不含 token |
| `/admin/session` | DELETE | 不需要 | 登出。标「不需要」是准确的：它只清调用方自己的 cookie 与对应的服务端条目，无凭据时是 no-op 并返回 200（幂等的「确保已登出」），既不泄露信息也不影响他人会话 |
| `/admin/panel-password` | PUT | 需要 | 设置或更改面板密码。存的是 **PBKDF2-SHA256 单向验证器**，不是密码也不是可解开的密文 —— 所以**忘记即不可恢复**（回头路是用一把有效 LB API Key 重设），且不依赖 `LB_CREDENTIAL_KEY`。除会话外还要**出示当前密码**（会话可能是 API Key 换来的，不证明知道密码）或一把有效 key；错误的当前密码走登录那同一套按来源退避，否则这里就是绕过退避的猜测通道。新密码下限 12 个**字符**（非字节）。成功后**失效其他会话**、保留调用方自己的。响应与日志都不回显密码或验证器 |
| `/admin/panel-password` | DELETE | 需要 | 清除存储的密码，回落 `LB_DASHBOARD_PASSWORD` 基线。与「重置渠道限制」同形：overlay 只能被显式删除。env 也没配时面板登录随之整体不可用 —— 那是正确结果，不是故障 |
| `/admin/model-capabilities` | GET | 需要 | 渠道/模型/API 能力目录、来源与有效期；未验证或过期能力不自动变成硬限额 |
| `/admin/credentials` | GET | 需要 | 已录入上游凭据的**身份**与校验结果；不返回明文也不返回密文，并标明哪些尚未生效 |
| `/admin/credentials/{provider}/{account}` | PUT | 需要 | 录入凭据：**先向上游校验，通过才封装入库**。与配置同名的 account 返回 409，不允许静默遮蔽；未配 `LB_CREDENTIAL_KEY` 时 503 |
| `/admin/credentials/{provider}/{account}` | DELETE | 需要 | 删除凭据（已生效端点到下次重启才消失） |
| `/admin/credentials/{provider}/{account}/validate` | POST | 需要 | 对已存凭据重新校验，会调一次上游 |
| `/admin/credentials/copilot/device/start` | POST | 需要 | 启动 GitHub device flow，返回验证码与授权链接；`device_code` 只留进程内，不落库不进日志不进响应 |
| `/admin/credentials/copilot/device/poll` | POST | 需要 | 轮询一次；`authorization_pending` 是正常中间态，拿到 token 后先校验再入库 |
| `/admin/model-routing` | GET | 需要 | per-model 规则的三段（provider 优先序、生效顺序、GHCP 账户选择），外加两个下拉的数据源（`known_models`、`copilot_accounts_available`） |
| `/admin/model-routing/{model}` | GET | 需要 | **候选诊断**：一个模型、每个账户、各自被哪一层摘掉。每行给 `reason`（闭集，`null` = 在候选里）、`selected`、`serves_model`，以及 `serves_model_ignoring_selection`（「若选上它能服务吗」，让 Dashboard 在**保存前**算出生效候选）。列出全部已配置账户而不只是候选 —— 被摘掉的那几行正是要看的。reason 取自 `CopilotEndpoint.supports()` 背后同一个函数，不是另写一份判断；与 capability 一样是**时点**判断，不是能力认证。`capability_known: false` 是通配警告而**不是**排除原因 |
| `/admin/model-routing/{model}` | PUT | 需要 | 设置规则。`providers`（**顺序即偏好**，不去重不重排）只重排不筛除，fallback 保留；给 `claude-*` 配指向 Copilot/Azure 的偏好会在 OpenAI 风格入口放行该模型。`copilot_accounts` 把该模型限定到指定 GHCP 账户（空/`null` = 不限定），名字不是已配置端点返回 400 `unknown_copilot_account`。**两段都在任何写入之前校验**，不会半应用；响应带 `effective_copilot_accounts` 与 `copilot_candidates_empty` |
| `/admin/model-routing/{model}` | DELETE | 需要 | 清除整条规则（两段一起），回到默认顺序与不限定账户；`claude-*` 随之重新被 OpenAI 风格入口拒绝 |
| `/admin/copilot/accounts` | GET | 需要 | Copilot 账户的启用态、已发现能力（capability）与允许模型；只返回账户名与模型名，不返回 token。启用态是**三态**：`enabled` 是生效值，`enabled_in_config` 是配置声明的初始态，`policy_configured` 表明运维是否真的决定过。请求数为进程内累计 |
| `/admin/copilot/accounts/{account}` | PUT | 需要 | 切换账户启用态与允许模型。**只能在已发现 capability 内勾选**，空列表被拒（停用请用 `enabled: false`）；启用一个配置声明 inert、无 `models`、也从未发现过 capability 的账户返回 400 `capability_unknown`（启用即通配）；写失败时执行中的策略不变 |
| `/admin/copilot/accounts/{account}/discover` | POST | 需要 | 向上游重新发现该账户的模型能力并存快照。时点发现，不是能力认证；需带 editor 头部，否则 GitHub 返 403 |
| `/admin/api-keys` | GET | 需要 | 已激活密钥的**身份**与渠道限制：掩码 + sha256 前缀，**绝不返回明文**，短密钥连掩码都不给。请求数与最后使用时间是进程内计数，重启归零 |
| `/admin/api-keys/{tenant}/providers` | PUT | 需要 | 设置该租户允许的后端 provider（allowlist，只做减法）。写失败时执行中的策略保持不变，不假装成功 |
| `/admin/api-keys/{tenant}/providers` | DELETE | 需要 | 清除该租户的渠道限制，回到配置基线。PUT 不接受空集（那等于把租户锁在门外），所以这是唯一的反向操作；幂等，已无限制时也返回 200 |
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
