# 服务可靠性：本地实现与验收边界

Author: Zeno Ren  
日期：2026-09-20

本批覆盖线上 effort 基线、请求/准入/发送的独立观测、上游错误体双限额，以及可独立运行的确定性日报与归档组件。应用变更没有开启新的 503 重试，没有修改生产调度、探针、副本或数据库。

## 线上基线

基线提交 `3a807af` 的 `main.py` 和新增 `effort_compat.py` 分别与 2026-09-20 重新读取的现网文件 hash 一致：

- main：`f9ad2781e492fd15e444522b69bd76352db88518ad02f61292e321e83c9f49f0`
- effort：`75eefaf3d111c8d04e9c5c667acd0d4029a09db10fbc72105696123801c27436`

该行为只为 Databricks Opus 5 保留 `output_config.effort`，不宣称 schema/format 已生效。非流式响应继续保留线上已有的 effort 提示头。Dockerfile 和 CI smoke 文件清单包含所有新增运行模块。

## 请求观测契约

每次推理入口返回新的 `X-LB-Request-Id`；现有 `X-Request-Id`/`OpenAI-Request-Id` 保留。日志关联两个 ID；调用方可传 `X-LB-Operation-Id` 关联其自己的重试。外部 ID 只接受长度不超过 128 的字母、数字、下划线、点、冒号、短横线，不作为 Prometheus label，也不提供自动去重语义。

| 指标 | 单位/分母 |
|---|---|
| `lb_requests_started_total` | 进入三个 POST 推理路由的请求，含本地拒绝 |
| `lb_requests_active` | 尚未结束的入口请求 |
| `lb_requests_finished_total{outcome}` | 每次入口结束时恰好一次本地结果 |
| `lb_endpoint_admissions_total` | 准入 lease，与原端点计数口径对应 |
| `lb_upstream_send_started_total` | 推理 HTTP client 调用次数，含同 lease 内 auth/opaque 修复 |
| `lb_upstream_send_finished_total{result}` | HTTP 结果分类、传输错误或取消；不等于生成完成 |
| `lb_request_duration_seconds` | 从入口到 ASGI 完整处理结束；按结果拆分，包含失败/取消 |

send started 不证明字节已经上网或上游已执行；PoolTimeout 也属于一次发送尝试。所有 ID 仅在日志/响应头中，指标维度固定为 provider、API 和有限结果枚举。

请求结果分为 completed、failed、incomplete、unknown、http_error、rejected、overloaded、cancelled、client_disconnected、internal_error。HTTP 200 不自动视为 completed；有效生成终态与 ASGI body 完成同时成立才会得到相应完成结果。JSON failed/incomplete 与流式错误有独立分类，缺少结果证据为 unknown。日志 `downstream_body_completed` 表示 ASGI send 返回，不证明客户端收到或工具任务完成。

第一批复用现有 SSE 协议观察器；没有增加第二份无限流缓存。非流式的显式结果语义只用于观测，尚未改成更严格的 serving 拒绝策略。完整客户端 schema/业务验收、首语义事件延迟、下游逐帧 terminal 确认属于后续批次。

其他修正：Databricks 入口设置与其他入口一致的 tenant ContextVar；只配置 Databricks 时 `/metrics` 不再因 Copilot 分支未初始化时间变量而报错。原 `/metrics` 内部免鉴权契约保留，公网访问控制应由后续部署变更单独实施。

## 错误读取的资源预算

| 配置 | 本批默认 | 含义 |
|---|---:|---|
| `UPSTREAM_ERROR_BODY_MAX_BYTES` | 65536 | 错误正文解码后的最多捕获字节；有效范围 256 B～16 MiB |
| `UPSTREAM_ERROR_BODY_TIMEOUT_SECONDS` | 2 | 从开始读取错误体到读取结束的时限，必须为有限正数 |

两个值在进程启动时加载，并出现在 `/config/effective`。适用于 DB/Azure/Copilot 的流式与非流式 HTTP 错误及 HTML 错误页。HTTPX response hook 在非流式 `post()` 自动 `aread()` 之前保护正文；Copilot 重建连接池时也保留 hook。

支持 identity、gzip（包括连续成员）和 deflate 的有界读取；其他编码作为诊断正文不可读处理。gzip/deflate 在解压时限制输出，并有独立的压缩输入字节上限。超过限额、读取超时或解码失败后，返回安全摘要，丢弃局部上游文本，保留原状态和 Retry-After。不会因为读不到完整错误体而推断 opaque-state 拒绝或重发执行不明的 POST。

`lb_upstream_error_body_reads_total{result}` 区分 complete、timeout、too_large、unsupported_encoding、invalid_encoding、read_error。正常成功流不会被该 hook 读取，因此本批不会给长 thinking 添加短 read timeout。

复用原响应 ownership 和 shielded cleanup。正文读取时限不取消清理 owner；清理耗时的独立监控仍待补充。本批默认值经过本地故障注入，仍需用目标镜像、真实客户端和运行配置完成发布验收。

## 请求预算

第二批新增 `INFERENCE_TOTAL_TIMEOUT_SECONDS=1800`（整次推理 30 分钟）与 `UPSTREAM_STARTUP_TIMEOUT_SECONDS=180`（每次上游发送等待响应头 3 分钟），均要求有限正数并在 `/config/effective` 可见。这是待生产验收的保守本地默认值，真实工作负载需要时可调整。

总预算从推理入口开始，覆盖请求读取、排队/鉴权、重试退避、上游调用与流式输出；heartbeat 不延长它。响应头到达后立即解除该次启动计时，非流式请求的长正文读取也不会继续受启动时限约束。HTTPX 成功正文同样使用已有 ownership 保护，使取消期间的首次 close 保持可等待。

总预算超时：headers 前返回 HTTP 504 和 `request_deadline_exceeded`；已进入 SSE 时按 Messages/Responses/Chat 对应协议发错误终态。不会拼接另一次推理或伪造完成。已结束的响应不会在 cleanup 阶段被补发错误；已提交的非 SSE 部分响应只能结束传输，不能再改状态或混入 SSE。

启动超时属于执行不明的读超时，非流式保留 HTTP 502 与 `failure_type=UpstreamStartupTimeout`，流式使用明确错误路径；它不进入安全 POST 重放白名单。`lb_requests_finished_total{outcome="deadline_exceeded"}` 单列总预算超时，发送结果中的 startup_timeout 单列启动超时。

预算到期会取消业务等待并等待原 owner 清理，不能给清理时间“硬上限”后丢弃它。错误终态通知另有 1 秒写出余量；因此总预算是业务取消时点，不是对所有 cleanup 都已完成的严格墙钟保证。已有长流部分输出不能因为本地 deadline 再次生成。

## 日报与归档组件

第三批还增加了请求内候选记录：DB/Azure 优先选择尚未尝试且模型兼容、满足熔断/软冷却条件的候选；候选都尝试过后仍沿用原有有界同端重试。Copilot 的 pinned account 和已命中的 session affinity 优先于此策略，避免为了轮换破坏会话状态。没有自动更换模型或 provider，也没有把所有 workspace 视为独立配额。

Databricks 可选 `endpoints[].models` 白名单使用转换后的原生模型名；缺省或空列表维持兼容的通配行为，非空列表不匹配返回 unsupported_model。配置本身不是能力验证证据。`lb_retry_decisions_total` 和结构化候选/重试日志解释状态白名单、Retry-After、剩余循环次数、候选/已尝试数量及总预算；仍只有原先允许的 429 条件可触发状态码重试。

`operations.reporting` 是纯 Python 标准库工具，不调用模型、不发送消息、不修改 scheduler。现有 OpenClaw watcher 应将其快照适配成 [example-snapshots.json](../operations/example-snapshots.json) 的结构，再使用计算结果生成解释。示例数据完全为合成数据。

每条快照需要带时区的 `timestamp`、`pod_uid`、`container_start_time`、`collection_status` 和 `metrics`。指标单位由 `metric_units` 明确声明。字段剥离数的单位是 fields，不是 token。失败采集仍需保留一条快照记录，不能删掉失败样本后跨过去相减。

```bash
python3 -m operations.reporting \
  --input operations/example-snapshots.json \
  --start 2026-09-19T01:30:12Z \
  --end 2026-09-19T02:30:12Z \
  --anchor-policy in_window \
  --output-dir /tmp/lb-report-example \
  --run-id synthetic-20260920-v2
```

`counter-window-v1` 默认只取窗内样本。`include_previous` 显式纳入起点之前最近样本，记录窗外扩展秒数，不插值、不把边界段伪装为精确窗口。报告保留名义窗口、真实样本首末、缺口、有效区间、配置的最大间隔和排除原因；CLI 同时记录输入文件 SHA-256。生命周期变化、counter reset、缺指标、失败采集、超过 75 分钟的间隔均不能变成零错误。每个指标返回 complete/partial/unknown；未知增量使用 null。

输入代表一个目标的时间序列。多副本需逐 Pod 序列计算再显式聚合；不能通过 Service 随机抽一个 Pod 混在同一序列里。此批未实现新旧副本重叠的集群聚合算法。

每次运行使用独立目录，保存 `summary.json`、`report.md`、`receipt.json`。临时文件写入/fsync → 原子替换 → 读取并校验 hash 和字节数 → artifact verified。执行、文件、投递状态独立记录；相同 run ID/相同内容可幂等回读，不同内容或不完整产物不会覆盖历史。

现有发送器只有在 `verify_archive()` 为 verified 后才应发送，并用 `record_delivery()` 保存真实消息 ID。发送器失败用 failed，ACK 丢失用 unknown；不得自动假设已投递或用户已读。工具本身不触发任何发送。缺文件或 hash 不符时不能写 delivered；已有 delivered 回执不能改成另一条消息。

本地归档算法与测试已提供，作为可选交接组件。按用户确认的范围，本轮只优化本地 LB，不接入或修改 OpenClaw。其 watcher 权威定义在 Gateway automation 中，探针为 `/openclaw/tmp/lb-monitor-probe.sh`，采集脚本为 `/openclaw/scripts/lb-monitor-records.py`；这些位置不能据此被改写或清理。生产日报尚未使用本组件。

## 验证与下一批

新增测试覆盖线上 effort 请求形态、ASGI 完整请求生命周期、同准入多次 POST、并发上下文隔离、SSE error/JSON incomplete、真实 TCP 错误流、压缩放大、重复取消的清理所有权，以及窗口与归档故障。既有重试/opaque-state/资源回收回归保持通过。

发布前继续遵守 [RESILIENCE.md 的独立复核门禁](RESILIENCE.md#tests-and-review-gate)。不能全量 apply 当前仓库的 K8s 模板覆盖现场。

后续顺序：有界启动/总 deadline 与客户端契约 → 能力候选/tried-set/有界排队 → 幂等 usage、readiness/drain 与双副本。精确容量 503 是独立开关实验，接受重复计费风险的决策与新生产参数应随具体发布方案审阅；本批没有提前开启。
