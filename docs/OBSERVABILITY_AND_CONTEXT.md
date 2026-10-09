# 请求诊断、参数与上下文契约

Author: Zeno Ren

2026-10-09 补充下述协议诊断与上下文提示，见 [本轮验证](reviews/2026-10-09-stream-reliability/REPORT.md)。已随源码 `fea7d60` 发布到 AKS，见 [部署回执](reviews/2026-10-09-stream-aks/REPORT.md)；后续是否仍运行此版本应读回现场，只更新本仓库不会修改线上服务。

日期：2026-09-30。本页描述仓库实现；生产是否启用以实际镜像、文件 hash、配置及验收回执为准。[本轮验证](reviews/2026-09-30-lb-contracts/VALIDATION.md) 单列本地、镜像、数据库、独立复核和 GitHub 状态。

本轮补齐安全诊断、分阶段时间、Chat→Responses 语义和渠道能力观测。推理 POST 重放白名单、账户 pinning/会话亲和、默认 startup/total timeout、副本数与生产入口均未放宽。

## 1. 请求、发送与终态

`X-LB-Request-Id` 是服务端生成的主键；错误体的 `lb_request_id` 与它一致。`X-Request-Id` / `OpenAI-Request-Id` 保留为兼容辅助 ID，仅接受 1～128 位字母、数字、点、冒号、下划线和短横线；无效输入替换为服务端 ID，不原样反射任意长值。`X-LB-Operation-Id` 只用于关联调用方的重试，不提供幂等推理保证。

| 事件 | 含义 |
|---|---|
| `lb_request_received` | 已鉴权、读取并解析的请求结构；不包含正文 |
| `lb_upstream_send_start` / `lb_upstream_send_end` | 一次 HTTP client 调用；每次调用有独立 `upstream_attempt_id`，401/opaque 修复会产生新 ID |
| `lb_upstream_error` | 有限错误原因、来源、HTTP 状态及同一次发送关联；不含原始错误 body |
| `lb_stream_end` | 三个 provider 的共同流观察结果，区分上游 terminal 是否出现；不是客户端接收确认 |
| `lb_request_end` | 每次入口恰好一次最终结果，包含实际 phase、start 时间与 request/send 计数 |
| `lb_parameter_policy` / `lb_context_budget` | 字段决策、估算方法与能力可信度；不记录字段值或 prompt |
| `lb_circuit_transition` | OPEN、HALF_OPEN 试探准入、成功/失败/中性试探、管理重置的事件证据 |

`lb_request_end` 的 `api_type` 是入口协议；send 事件/指标使用实际上游协议。缓冲 Chat 适配因此可能出现入口 chat、上游 responses，不能把不同分母强行相等。

`requested_model` 来自请求，`forwarded_model` 来自路由，`resolved_model` 只来自上游实际报告；缺失时是 unknown。适配器在 Chat 响应中回显的模型名不能成为上游模型证据。关联 ID、模型和 endpoint alias 不加入新的 Prometheus 高基数标签。

协议异常、startup/read/write timeout、pool、连接、缺 terminal、本地压力、取消和总 deadline 有独立有限原因。401 修复成功后最终 reason 为 none，但原 attempt 的 401 事件保留。明确的本地准入拒绝不归因为上游不可用。失败且原因无法分类时为 unknown，不以 none 伪装。

HTTP 200、发送返回、有效生成终态、用量落盘和客户端业务成功仍是不同事实。`downstream_content_started` 表示生成器已提供可识别的文本、工具参数或 refusal 内容，不证明用户收到。旧 Copilot `saw_completion` 保留；新增 `terminal_seen` 明确表示看见终态，failed/incomplete 也可以为 true。

## 2. 日志出口与隐私

协议诊断保留实际 `http_version`、`protocol_error_kind`、`protocol_scope` 及数值 `http2_error_code` / `http2_stream_id` / `http2_last_stream_id`。仅从有限深度的真实 h2 异常原因对象提取；看似 StreamReset 的异常字符串不能成为隔离证据。GOAWAY debug data、异常原文和请求正文均不进入日志。

`copilot_stream_prefirst_content_stall` 在已收到响应头、但超过 `COPILOT_STREAM_PREFIRST_CONTENT_WARN_SECONDS`（默认 90，范围 0～3600，0 关闭）仍无任何正文时发出，每条流最多一次，对应计数器 `copilot_stream_prefirst_content_stall_total`。它纯观测：不中止流、不触发重试、不影响熔断，也不改变重放白名单。默认值低于 2026-10-09 观测到的约 120 秒上游取消，目的是在该定时器触发前就能看到停滞，而不是事后从 `CANCEL(8)` 倒推。阈值不是 SLA，也不代表上游真实超时值。

Copilot network/stream summary 新增 `upstream_headers_received` 与 `upstream_idle_seconds`。后者是从最近响应头或解码正文 delivery 到异常捕获的间隔，不是抓包得到的 TCP 空闲时间，也不包含错误后的 DNS/TCP 探测耗时；只发本地 heartbeat 不刷新它。分层熔断日志的 `circuit_scope=endpoint|model_api` 与 `/stats` 中的 `model_api_circuits` 解释实际阻断范围。

诊断只保留允许字段，外部标识限制字符和长度；原始 UA 仅归类，不能当可信租户。异常仅保留类型，错误 body、正文、工具参数/结果、完整 schema、图片和 opaque 内容不进入请求诊断。旧非结构化推理日志仅保留来源位置与级别；排障应查询结构化事件，而非依赖原始错误片段。

默认 stdout/stderr 日志 sink 通过有界队列运行，容量由 `LB_DIAGNOSTIC_QUEUE_CAPACITY` 控制，默认 4096，范围 1～65536。队列满、sink 写入失败、事件编码失败和关闭后的丢弃可在 v3 指标中观察。sink 卡住不会让请求等待磁盘/网络 I/O；关闭只做有界等待，不承诺日志无丢失。第三方主动重配 logging handler 的行为不在默认出口保证内。

`LOG_FORMAT=text` 的通用结构化事件在 `LEVEL:logger:` 后保留安全 JSON 字段，既有 Copilot request/stream summary 保留文本标记及安全键值；`LOG_FORMAT=json` 继续输出字段在外层的 JSON。两种模式都保留关联主键与结果。默认 stderr 使用直接文件描述符写入，避免真实日志管道背压让解释器退出等待 Python stdio 缓冲锁；短写和写入异常也有回归。

JSON `ts` 使用 LogRecord 创建时间，避免把异步排队时间写成业务发生时间。`started_at_unix` 是入口时间，供按开始时段建立 cohort；持续时间使用 monotonic clock。

服务不新增诊断文件或无限历史库。外部日志平台仍需配置访问权限、保留期和容量；队列不是持久审计账本。回退可以关闭额外观测或降低日志级别，不恢复原始正文日志。

## 3. 阶段时间与预算

| 字段 | 计时口径 |
|---|---|
| `body_read` | 读取请求体的实际调用区间 |
| `admission_wait` | 本地准入调用到获准/拒绝/取消，包含快速准入路径 |
| `auth_prepare` | 凭据/请求头准备，包含相关锁等待；嵌套计时不重复相加 |
| `send_to_headers` | HTTP 调用开始到真实 response hook；多次发送的已知区间累计 |
| `send_call` | HTTP client 调用全过程；非流式可能包含响应体缓冲 |
| `first_event_from_ingress` | 入口到首个可识别 JSON 协议事件；不表示成功或首用户内容 |
| `first_content_from_ingress` | 入口到首次提供可识别文本、工具参数或 refusal 内容；heartbeat 不计 |
| `stream` | SSE 观察开始到流退出，含背压；不包含之后全部清理 |
| `cleanup` | 本请求清理等待区间的并集，嵌套和重叠 owner 不重复计时 |

DNS/TCP/TLS/pool 没有可靠独立 hook，保持 null。阶段可能嵌套，不可直接相加当总时间；未进入/无法观测的阶段不伪填 0。`phase_offsets_seconds` 记录有对应 hook 的开始/结束偏移。deadline 到达时记录可见阶段，无法确定时为 null。

全局默认仍是 `UPSTREAM_STARTUP_TIMEOUT_SECONDS=180`、`INFERENCE_TOTAL_TIMEOUT_SECONDS=1800`。startup 等待 headers，绝不是容器冷启动或 TTFT。`LB_STARTUP_BUDGET_OVERRIDES` 接受最多 32 个精确 provider/API/model 规则，例如以下仅为配置结构示例，不是生产建议：

```json
[{"provider":"copilot","api_type":"responses","model":"synthetic-model","seconds":240}]
```

默认规则为空。规则必须是有限正值，不支持通配或重复键；仍受 total deadline、取消与准入约束。规则不允许 timeout 后重新 POST。如何形成实验与停止条件见 [实验与耐久性说明](OPERATIONS_EXPERIMENTS.md)。

## 4. 指标兼容性

- `/metrics` 默认继续输出 `lb-metrics-v2`，响应头 `X-LB-Metrics-Schema` 声明版本。
- `/metrics?schema=lb-metrics-v3` 显式增加阶段 histogram、诊断丢弃，以及用量 accepted/persisted/oldest-pending/volatile-buffer 指标。
- 未知版本返回 400；v2/v3 定义分别在 `operations/metrics-contract.json` 与 `operations/metrics-contract-v3.json`。解析 v3 时显式传 `parse_exposition(text, schema_version='lb-metrics-v3')`。
- send result 的有限 provider/API/result 组合在初始抓取就有零值；不预创建任意模型或用户 series。新 reason/parameter 枚举仍需消费者认识 unknown 与扩展值。
- 计数按 Pod UID/容器启动时间分段；缺失、重启、采集失败和 sparse 首值不能当 0 增量。`operations.reporting` 的类型化窗口仍可复用。

升级外部采集器前确认真实运行版本和响应头，再切换 v3 解析器；本轮没有修改 OpenClaw 自动化、baseline 或历史日报。

## 5. 参数和 Chat→Responses 适配

`X-LB-Strict-Parameters: true` 继续在已知不等价丢弃前拒绝；等价转换允许通过，并在 `X-LB-Transformed-Parameters` 中报告有限字段名。compat 的已知丢弃保留 `X-LB-Dropped-Parameters`。不支持的关键 adapter 语义直接返回可操作的 400，不以普通文字伪装工具或 schema。

| Chat 输入/输出 | 转换 |
|---|---|
| system/developer/user/assistant | 保留角色与文本；不补造默认用户指令 |
| function tool call/result | 保留 `call_id`、name、arguments 和 output；检查孤立、重复、缺结果的配对，工具定义不因历史出现结果而删除 |
| `response_format` | 等价映射为 `text.format`，保留 schema/name/strict |
| `reasoning_effort` | 映射为 `reasoning.effort`，不自行改变 effort 值 |
| 函数工具未指定 strict | 显式发送 strict=false，保持 Chat 默认语义，避免 Responses 默认收紧 |
| `tool_choice` | 映射函数选择对象；保留 auto/none/required |
| token limit | 映射 `max_output_tokens`；相互冲突的正上限与非整数被拒绝 |
| text/image/file 内容块 | 转换成对应 Responses 输入类型，保留内容；不支持的类型明确拒绝 |
| store/cache/metadata 等共同参数 | 保留请求值，不自行选择存储保留期、缓存 TTL 或计费策略；实际支持仍由渠道决定 |
| Responses incomplete | 已知输出上限映射为 Chat length；content_filter 保留；不把未知 incomplete 转成 stop |
| refusal、混合文本/工具、usage | 保留在 JSON/缓冲 SSE；tool delta 包含 index，include_usage 请求可收到 usage chunk |

该 adapter 仍调用一次**非流式** Responses，再包装 Chat SSE。`X-LB-Stream-Mode: buffered-adapter` 明示这一点；不是原生上游流式或断点续传。建议真实支持 Responses 的客户端直接使用原生接口。

`OPENAI_CHAT_TO_RESPONSES_MODELS` 继续控制哪些模型进入适配器。`LB_CHAT_ADAPTER_CONTRACT=preserve` 默认启用上述契约；`text-only` 可退回只接受可保留的文本请求，对需要工具/schema 等能力的请求明确拒绝。`LB_CHAT_ADAPTER_PRESERVE_MODELS` 可将 preserve 限于指定模型；未设置表示全部已配置的适配模型，空值表示全部仅文本。不会通过开关恢复静默工具扁平化。

模型/provider 能力、输出是否真正遵守任意 JSON schema 和真实客户端端到端仍需独立验证。unsupported 的 n>1、stop 等没有可靠等价映射时明确拒绝；本实现不宣称完整覆盖全部 Chat 参数。

映射依据：[OpenAI 官方迁移说明](https://developers.openai.com/api/docs/guides/migrate-to-responses)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)。这些是 API 形态依据，不是 Copilot 渠道能力认证。

## 6. 图片与 token 估算

推理 JSON/SSE 响应头现在提供 `X-LB-Context-Estimated-Input-Tokens`、`X-LB-Context-Text-Bytes`、`X-LB-Context-Estimate-Confidence`、`X-LB-Context-Estimate-Complete`、`X-LB-Context-Unknown-Components` 和 `X-LB-Context-Advice`。未知限额不返回数值；有当前已核验目录时才返回 `X-LB-Context-Input-Limit` / `X-LB-Context-Context-Limit`。所有字段为估算或目录元数据，不含正文。流式响应头只反映提交头部时已观察到的路由，不能宣称后续重试目标的限额。

建议值为 `none`、`large_input`、`estimated_near_limit` 或 `estimated_over_limit`。语义输入默认达到 256 KiB 时提示 large_input；已核验限额的低可信度估算达到 80% / 超过 100% 时给出 near / over 提示，始终不据此硬拒绝输入、不摘要或删除历史。调用方可主动整理会话、工具输出或图片；Codex 是否显示这些自定义响应头尚未验收。`LB_CONTEXT_BUDGET_MODE=off` 关闭推理上下文提示，受保护的本地 count_tokens 接口仍提供其估算头。

`LB_CONTEXT_LARGE_INPUT_BYTES=262144` 可配置规模提示阈值，范围 1 B～64 MiB；这不是模型上下文容量或服务拒绝阈值。

图片数量超过预算时，`LB_IMAGE_TRIM_POLICY=reject` 默认保留原输入并返回 413。调用方可用 `X-LB-Image-Trim: allow` 明确允许旧图替换，或由管理员设置全局 allow；响应头显示实际策略和 `images.trimmed`。strict 模式仍在任何裁剪前返回 400。允许裁剪后必须重跑图片准入，不绕过像素/内存限制。

既有图片压缩仍运行；实际压缩会报告 `images.compressed`。这里的保留指不擅自删除图片，不承诺无损重编码。Databricks 4 MiB 出站限制和 LB 64 MiB 原始字节限制不等于模型 token 窗口，不能把某次 2 MB 失败样本变成全局模型阈值。

`POST /v1/messages/count_tokens` 现在需要 LB 鉴权，复用有界读取和本地准入。响应保留 `{"input_tokens": N}`，新增：

- `X-LB-Token-Count-Method: utf8-json-estimate-v1`
- `X-LB-Token-Count-Confidence: low`
- `X-LB-Token-Count-Complete`：是否完整遍历可见的已提供输入，不代表精确 tokenizer 或包含厂商隐藏前缀
- `X-LB-Unknown-Components`：图片、文件、opaque/prior state、遍历上限等未知开销

估算覆盖 system/instructions、tools、messages/input 及相关 schema；不从密文或图片字节猜精确 token。计数接口不调用付费模型。它占本地内存准入资源，但不计入三个推理 POST 的 request/send 分母。

## 7. 渠道能力

`LB_MODEL_CAPABILITIES_PATH` 可指向有版本的本地 JSON。参见 [未验证示例](../model-capabilities.example.json)。默认目录为空，示例全部限额为 null，没有编造厂商窗口。

每条记录精确匹配 provider/API/model，可进一步限定 endpoint alias；包含模型版本、输入/总窗口/输出限额、特性、来源、核验时间和失效时间。`operator_verified` 是管理员对渠道证据的声明，不是 LB 自动认证。过期、未知或未核验记录不会自动成为硬限制；不能拿 OpenAI 直连参数冒充 Copilot 渠道事实。

`LB_CONTEXT_BUDGET_MODE=observe` 默认只输出安全观测。`off` 关闭附加预算观测。`enforce` 也**永远不按低置信度输入估算硬拒绝**，只会在有效已核验记录下检查明确输出上限和已知不支持的特性。首次检查发生在端点 lease、发送和流 headers 之前；既有重试切换端点时，在新 lease 和 POST 之前重新检查。若此前已返回流 headers，拒绝会成为对应协议的不可重试错误终态，不向不兼容端点发第二次请求，也不把本地拒绝记作该端点故障。不会摘要、删除历史、换账号或换模型。

特性检查包含 Responses `function_call_output.output` 数组中的图片，同时继续排除工具 schema/example 内的示意数据。Chat 入口不再通过整数强转把布尔值或小数输出上限当零删除；适配器会明确拒绝这些非法类型，保留原有合法整数及整数文本的兼容行为。

`GET /admin/model-capabilities` 沿用管理接口的 LB 鉴权，显示目录来源和状态；标准 `/v1/models` 的 data/models 返回结构保持兼容。`/config/effective` 展示实际开关与目录 hash，并返回构建身份、文件匹配和完整 manifest 覆盖状态。
