# LB 持续监控反馈与优化方案评估

Author: Zeno Ren

日期：2026-09-30（北京时间）  
评估基线：`main@16de51d29d1fbb3e81c65434ce55fadc3573da61`  
输入：用户提供的《LB项目优化开发清单-20260930.md》  
范围：文档、当前源码、现有本地回归及新增合成诊断。未修改应用代码、生产配置、OpenClaw 任务或监控历史，未提交或推送。

**建议采纳清单的总体方向，并调整实施顺序：先修复已经复现的错误归因、日志安全和参数语义缺口；随后补分阶段耗时和能力观测；获得可比较证据后再做单路由实验。当前证据不足以支持全局延长 timeout、扩大 retry 或增加副本。**

已有预算、准入、流终态、清理 ownership、用量批次幂等和监控契约值得保留。收益首先是减少错误归因和静默语义变化，使后续优化有可信依据；不能据此承诺消除上游超时。

## 1. 证据可信度与版本边界

| 材料 | 本轮确认了什么 | 仍不能确认什么 |
|---|---|---|
| 附件 E1–E5 | 阅读了其历史窗口、归因限制与优化建议 | 未取得并重算 9/24、9/25、9/30 的原始账本；其中故障数是附件陈述 |
| 当前仓库 | `main` 提交固定；应用已跟踪文件无改动；关键文件 hash 已保存 | 与 AKS 实际运行文件是否一致 |
| 现有回归 | 本轮执行结果为 651 passed、6 skipped、627 subtests passed | 不能覆盖现有测试未断言的字段和语义 |
| 本轮合成诊断 | 对当前代码复现下述具体行为；使用 MockTransport/ASGI，禁止网络 connect | 不证明生产故障的唯一原因或出现频率 |
| AKS/OpenClaw 现场 | 未连接、未变更 | 当下 digest、Pod、配置、负载、监控采集覆盖及真实客户端结果 |

附件的 `generation=70`、历史 digest 和 `revision=892e397...` 只能定位历史证据。当前 Git HEAD 不同，不等于证明线上运行的是旧代码；revision 注解可能没有随 out-of-tree 构建更新。进入任何生产实验前，应核对 registry/platform digest、运行 imageID、Pod UID、启动时间、关键文件 hash、依赖和脱敏配置指纹。

附件引用的三个 `/openclaw/...` 原始报告/账本路径在本机不存在。本轮没有用今日聚合计数推断每次失败的根因，也没有把附件中的开发指令当成实施授权。

## 2. 当前代码的直接发现

以下结果见 [合成诊断输出](assessment-probes.json)，可用 [诊断脚本](assessment_probes.py) 重现。脚本观察现状，不属于优化已经通过验收的证据。

| 发现 | 本轮结果 | 对优化的意义 |
|---|---|---|
| F1：半途协议异常的原因丢失 | 合成 Copilot 流输出后抛 `RemoteProtocolError`，最终 `outcome=failed`、`failure_reason=none`。发送 1 次，响应已关闭，endpoint 活动数回到 0 | 先修诊断传播；本例已有禁止重放与资源释放保护，不需要为了修原因而改重试策略 |
| F2：成功流残留此前失败原因 | 401 → 有界鉴权修复 → completed，发送 2 次，最终仍为 `failure_reason=authentication` | 区分 attempt 历史与 request 最终原因，否则只读 reason counter 会把修复成功误当最终认证失败 |
| F3：错误日志包含正文，且 sink 可同步阻塞 | 合成错误正文中的 canary 出现在日志；注入 30ms handler 延迟时，`log_event()` 耗时约 31.6ms | 截断不是脱敏；捕获日志异常不是非阻塞。需要字段白名单、有界队列、丢弃观测和旧日志路径收敛 |
| F4：strict 没有覆盖适配器的关键参数 | Chat→Responses 路径带 `response_format=json_schema` 与 strict=true，仍返回 200/strict；上游参数没有 schema，丢弃提示头缺失 | 这是可实施的语义修复，应提前，不能等全量能力表完成后再处理 |
| F5：现有 token 计数忽略重要输入 | 相同 messages 加 10,000 字符 system 和 10,000 字符工具 description，返回估计仍为 10 → 10 | 不能把当前 `/v1/messages/count_tokens` 当可信计数器接到硬拒绝逻辑 |
| F6：已有图片裁剪与新目标存在边界冲突 | `trim_excess_images()` 会把较旧图片替换为文本占位；三条入口在大体积图片准入失败后均有调用路径 | “不静默删内容”需要审阅已有行为，新增能力表本身不能保证这一目标；不能未经兼容评估直接移除现有保护 |
| F7：关键 counter 仍有稀疏首值 | 新 `RequestTelemetry` 导出的 `lb_upstream_send_finished_total` 样本数为 0；其他许多有限枚举已预初始化 | OPT-07 应做定点补齐，不能把所有指标说成缺少零值；也不能据此认定线上丢请求 |

F1/F2 还确认：Copilot stream summary 没有 `lb_request_id` / `upstream_attempt_id`，而 send summary 的 attempt ID 只在 `inference_call()` 局部生成。已有两个 request ID 可以通过结束日志关联，但不能视为统一的完整 attempt 生命周期。

源码定位（基于本次固定提交）：

| 位置 | 已核对职责 |
|---|---|
| [request_telemetry.py](../../../request_telemetry.py)，L37、L103、L181、L209、L289 | 同步日志；请求记录；generation/reason 分开更新；send ID 生命周期；最终计数 |
| [main.py](../../../main.py)，L248、L2841、L5685、L5936、L7230 | formatter 接收任意 extra；错误正文片段；网络异常；原始 model/UA 日志 |
| [main.py](../../../main.py)，L1109、L3628、L5756 | 已有 SSE 观察器；HTTP 错误设置 reason；401 修复分支 |
| [main.py](../../../main.py)，L6688、L6975、L7134 | 字符估算；adapter 字段构造；buffered Responses 调用 |
| [main.py](../../../main.py)，L1634、L6636、L7543、L7614 | 图片替换及三条入口的现有裁剪路径 |
| [request_budget.py](../../../request_budget.py)，L31、L37、L80 | 响应头解除 startup 计时、每次发送预算、整次请求 deadline |

## 3. 对 OPT-01～08 的评估

| 项目 | 已有实现 | 应补的增量 | 建议 |
|---|---|---|---|
| OPT-01 错误事件与关联 | 服务端 LB ID、send ID、结果/原因 counter、凭据映射 tenant | attempt 上下文跨 response/stream/cleanup 传播；模型及路由安全字段；最终原因一致性；日志白名单与有界异步出口 | 第一批；已具备直接复现依据 |
| OPT-02 阶段时间 | monotonic 总时长、startup/total 预算、准入等待汇总、cleanup 汇总 | 请求级 body/admission/auth/headers/首事件/首内容/stream/cleanup 时间点，实际生效预算和失败阶段 | 第一批单独小 PR，避免与日志重构混成大改动 |
| OPT-03 上下文能力 | 模型路由、模型白名单、字节和图片预算；上游超限粗分类已存在 | 有来源和时效的渠道能力表、计数可靠性声明、调用端预算、observe 模式 | 先观测；能力未知时不设置硬 token 阈值 |
| OPT-04 流终态与清理 | 终态观察器、failed/incomplete 区分、缺 terminal 错误、保守重放、owned cleanup 及大量回归 | 修原因传播；明确 terminal_seen 与 completed；补真实 SDK 与慢消费者的验收证据 | 原清单方向正确，但无需另建平行状态机或整体重写 |
| OPT-05 参数与 adapter | compat/strict、丢弃响应头、有限字段计数 | schema/reasoning/tool 等关键字段覆盖表；已复现的 strict 漏检；图片变更可见性 | 与流诊断分开；明确语义修复优先于全面 token 强制校验 |
| OPT-06 retry/fallback/breaker | 429 窄策略、Connect/Pool 白名单、一次鉴权修复、opaque 修复、pinning/affinity、熔断试探 | 使用分阶段证据复核边界、attempt 放大量和 provider 契约 | 暂不放宽；不得用 HTTP 外壳判断 POST 未执行 |
| OPT-07 指标与构建身份 | `lb-metrics-v2`、类型化窗口算法、build-info 文件 hash/依赖、发布校验工具 | 稀疏 send result 零值；新指标与 parser 同步；dirty/out-of-tree/发布回执关联 | 将身份核实移到前置步骤；其余小范围补齐 |
| OPT-08 用量与副本 | 内存缓冲、固定 batch ID、事务账本、提交 ACK 丢失幂等、尾批排空 | 明确可接受 RPO/RTO，再做崩溃/多 writer/节点终止验证；必要时 durable outbox | 独立架构设计；当前材料没有证明已发生丢账事故 |

需要修正清单中的几个实现细节：

- **现成组件复用。** `cleanup_observability.py`、`operations/metrics_contract.py`、`operations/reporting.py` 和 Dockerfile 的 `build-info.json` 都已经存在。扩大覆盖时优先扩展这些组件。
- **指标增量也可能破坏采集。** 当前 `parse_exposition()` 会拒绝未知 `lb_` family；增加 diagnostic drop 或阶段耗时指标时，需同步更新版本化契约、解析器与 fixture，再核对外部消费者，不能只往 `/metrics` 加字段。
- **镜像 digest 分阶段记录。** build-info 可在构建时记录 commit、文件 hash 和 dirty/out-of-tree；最终 registry/platform digest 在推送后绑定到外部发布回执。不要把镜像最终 digest 塞回同一镜像造成自引用。
- **不要机械要求每个修复都有恢复旧行为开关。** 观测可降采样，能力强制检查可退回 observe；日志安全回退仍应使用安全最小记录，不能重新开启原始正文日志。
- **默认 OTel 不可假设已启用。** 项目有可选 bootstrap，默认镜像未安装 OTel 依赖；自动 HTTP span 也不自动提供首语义内容或完整 DNS/TCP/TLS 分解。首批不必因接入 tracing 平台而阻塞。

## 4. 三个主要现象应怎样优化

### 上下文超限

附件 E1 足以支持优先调查该渠道的输入预算，不足以确定精确窗口，也不支持“超过 2MB 拒绝”。LB 中的字节/内存预算与模型 token 窗口是两种约束。

能力记录建议包含 provider/渠道、模型版本、输入/总窗口/输出上限、工具/图片/状态/结构化输出、来源、核验时间与 confidence。区分 requested model、LB forwarded model 与上游实际报告 model；最后一项未返回时保持 unknown。

现有 token 接口只计算 `json.dumps(messages)` 的长度除以 4；它既遗漏 system/tools，也没有可信的多模态、状态和渠道 tokenizer 规则。先让管理侧与调用端知道“这是估计”，保持现有标准接口兼容；新增 observe 记录估计方法及缺失项，再决定是否替换计数实现。禁止把估计当精确值硬拒绝。

调用端负责摘要、历史分段与输出预留；LB 负责可信能力检查和可操作错误。当前已识别的 context error 可复用。对敏感 system、工具配对和 opaque 状态的变更，应有明确调用端策略；模型/账号切换不属于默认恢复。

### 晨间 startup_timeout

本地实现确认 startup 计时从一次 HTTP 调用开始，到收到 headers 后解除；它可能包含 pool/连接/上传/等响应头。非流式 `inference_call()` 的完整时长还可能包含 body 读取。该指标不能解释为容器冷启动或 TTFT。

先记录 monotonic 事件点：入口、body 完成、admission 获得、鉴权准备完成、send 开始、headers、首有效事件、首用户内容、stream 结束、cleanup 完成。没有可靠 hook 的阶段为 null。admission、body、auth 的真实先后以代码路径为准；嵌套时间不相加。等待中的 heartbeat 或任意 frame 不能充当首用户内容。

对同 provider/模型/API/stream/尺寸桶/可信调用方/时段比较成功与超时，额外观察队列、CPU throttling、内存、连接、断开和 cleanup。若证据支持 headers 预算不足，再对单一路由修改一个变量；是否支持分路由配置也须先实现和验证。验收看增加了多少明确完成，以及尾时延/占用是否恶化，不能只看报错变晚。具体新 timeout、样本量、费用和改善阈值尚未确定。

### 半流失败与客户端断开

本轮 F1 中既没有自动重放，也没有遗留 endpoint 活动计数；这说明具体缺口首先在诊断传播。保留 response/transport 清理 owner 和 exactly-once 结算，集中修正错误原因与 ID 关联。

区分上游 headers、下游 headers、任意帧、用户内容、有效 terminal、生成结果、ASGI 发送结束和 cleanup。`saw_completion` 在现有代码中实际表示 terminal 已出现，失败也可为 true；宜增加或迁移到无歧义字段，保留旧消费者兼容。ASGI send 完成仍不等于用户收到。

附件 E4 的 40 次 client_disconnected 需要按请求时间、客户端 deadline、active/queue 和 ingress 断开证据解释；不能全部记为用户主动取消，也不能全部归罪 LB。客户端 SDK 与 OpenClaw 的重试次数需独立计入整条链路，避免 LB 不重放、外层却重新提交产生副作用。

## 5. 持续监控应该怎样使用这些改进

保持三方责任清晰：LB 提供可信事实与安全错误；OpenClaw 负责采集、窗口计算与解释；调用端负责会话预算和业务恢复。本轮只提出接口及验收建议，没有修改后两者。

建议每个监控窗口保留以下并列视图：

| 视图 | 核心内容 | 解读限制 |
|---|---|---|
| 数据质量 | 采样覆盖、缺样、重启、Pod/容器身份、契约版本 | 缺失/unsupported 是 unknown；新 series 首值不能直接当窗口增量 |
| 请求结果 | started、finished 各 outcome、active，显式呈现拒绝和断开 | 请求数不等于 attempts，更不等于 OpenClaw 任务数 |
| 失败与修复 | 最终 reason、error origin、attempt 历史、repair 结局 | 一次修复成功请求不计两次用户失败；保留它实际发过两次的事实 |
| 阶段与资源 | headers/首内容/stream/cleanup 时延，admission 与资源压力 | 不用平均时长掩盖失败和长尾；非可靠阶段不估填 |
| 用量与持久化 | pending、rejected、flush failure、最近成功、事件/账本关联 | 推理成功与用量落盘独立；不能重新推理补账 |
| 版本与维护 | 镜像/源码身份、有效路由、accepting、维护所有权 | Pod Ready 不替代公网业务与完整终态验收 |

在同一个连续、无 reset、完整采集的实例序列中，可校验 `Δstarted − Δfinished = active_end − active_start`。跨窗口未结束请求、Pod 重启和采集失败需要先解释；差值不是直接的“丢请求数”。若展示 completed/finished 比例，同时展示全部 outcome 分布、样本量与覆盖率，不能通过排除断开/超限流量制造改善。

模型与请求关联优先放在有访问控制的结构化事件中；Prometheus 仅使用有限枚举。不要把任意用户 model、request ID、operation ID 或原始 UA 扩成标签。时间窗口与展示统一注明北京时间，事件仍保留带时区时间戳。

## 6. 推荐实施顺序与验收

P0 在这里表示开发依赖优先级，不代表当前生产事故等级。

| 阶段 | 具体交付 | 验收与停止条件 |
|---|---|---|
| 0：版本与基线 | 核对当前运行身份与 Git 差异、现有消费者、真实配置、代表性窗口；保存私有发布/恢复材料 | 未确认运行兼容性不进入生产实验；不需要先关闭流量 |
| PR-A：安全诊断闭环 | F1/F2 原因修复、attempt 关联、安全模型/路由字段、日志白名单/有界出口；同步 metrics 契约 | known exception 有原因；修复成功的最终原因清零、历史保留；隐私 canary 不落盘；sink 慢/满/异常不阻塞或改变推理；部分流仍一次 send |
| PR-B：阶段观测 | 在既有预算/准入/流 owner 上加事件点、有效预算与有限阶段指标；补 send result 零值 | 缺值为 null；heartbeat 非首内容；只计正确时段；压测测量 CPU/RSS/延迟与诊断丢弃率，无未经验证的低开销承诺 |
| PR-C：关键参数契约 | 修 strict adapter 的 schema 漏检；审阅 tools/结果、reasoning、token-limit 和图片裁剪；逐模型/API 定义等价转换或明确拒绝 | 合法请求不误拒；必需 schema/tool 语义不丢；compat 变更可见；buffered 行为诚实呈现；实际 SDK 回归 |
| PR-D：能力与上下文观测 | 有来源的能力表、未知/过期策略、计数方法、调用端预算协作；默认 observe | 短/长/临界/中文/工具/图片/状态覆盖；未知渠道不伪造阈值；获得可信计数与能力后再讨论 enforce |
| 单路由实验 | 仅验证一个已获证据支持的假设；固定 cohort、变量、窗口、费用、通过/停止条件 | 重复晨间窗口的有效样本；增加明确完成，并同时检查长尾、断开、准入和资源；样本不足记观察不足 |
| 后续独立设计 | 定向 retry/breaker；必要的 durable outbox；多副本 RPO/RTO 与全局限额 | 故障注入与真实隔离存储验证；禁止以 batch 幂等推导硬崩溃零丢失，禁止直接双副本翻倍进程额度 |

相对附件的调整：把原 PR-1 拆成安全诊断和计时两部分；将原 PR-3 中已复现的原因/关键参数问题提前；能力强制校验后置。流清理回归贯穿每次行为修改，不单独启动一次没有缺陷依据的大重构。

实施时按 TDD 先把本轮诊断转成有明确期望的失败测试，再改最小逻辑；结构整理与行为修复分开。首批诊断不需要改数据库、timeout、重试、账号选择或部署拓扑。指标/日志契约的兼容变更应提供消费端迁移说明，不要求清空监控历史或修改 baseline。

## 7. 发布、回退与架构边界

仓库已有 `operations/release`、维护暂停/恢复和镜像/业务校验工具，其本地存在不证明已经安装到 AKS，也不证明本轮候选已获发布验收。沿用项目规定的目标镜像、独立评审、现场最小补丁、互斥与恢复责任门禁；应用测试不能替代发布工具故障注入。

`gateway_lifecycle.py` 同时存在不可逆 drain marker 与带 release/revision 的可恢复 maintenance pause。不能把删除 marker 当恢复，也不能忽略完整 Service selector、EndpointSlice、正在运行的长流和单 writer 退役。维护前完成准备，不在排空后等待下一次对话接力。

用量 buffer 与 pending batch 目前在进程内存中，正常约 30 秒 flush 周期不是“最多丢 30 秒”的保证；数据库故障、积压和进程丢失都改变边界。先确定业务能接受的记账缺口与恢复时间，再决定 outbox 是否必要。多副本还需校准全局准入/配额、亲和、凭据刷新、容量和排空，不能仅凭 MySQL 增量 upsert 就宣布可直接扩容。

回退保留追加账本和监控证据，不以旧数据库快照覆盖新用量。隐私泄漏、重复执行/记账、工具/schema 语义损坏、合法请求误拒或资源越界均应触发对应变更停止条件。

## 8. 本轮验证与下一步决策

执行命令：`python3 -m pytest tests/ -q --tb=short`。结果：**651 passed、6 skipped、627 subtests passed，40.98 秒**。实际 Python 为 **3.14.5**；这是当前主机回归，不是 Dockerfile 的 Python 3.12 目标镜像验证，也不是 CI 的 3.11/3.12/3.13 矩阵重跑。6 个隔离 MySQL 测试按原有 opt-in 规则跳过，未连接真实 MySQL。历史取证 fixtures 按仓库既有 collection 规则处理。

新增合成诊断确认了 F1–F7 的上述观察，并保存当前结果。关键源文件/依赖和附件 hash 见 [验证记录](validation.json)。没有本轮真实 provider、真实客户端、AKS、公网链路、压力或长时间稳定性结论；没有对实现 PR 做独立 QA，因为本轮尚未实施应用变更。

后续不影响开始本地 PR-A 的决定包括：诊断访问/保留/容量预算、模型能力权威来源与维护人、调用端预算维护人、允许的摘要/模型切换策略，以及实验路由/费用/通过阈值。这些在进入对应生产步骤前确定，不能用猜测填生产默认值。

**可进入实施的第一步是 PR-A：把已经复现的原因与日志问题变成可靠契约。PR-C 的 strict/schema 修复也已有明确需求；不应把研究完整上下文能力表作为它的前置条件。**
