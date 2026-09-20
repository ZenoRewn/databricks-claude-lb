# 协议、记账与监控契约加固

Author: Zeno Ren

本轮为本地实现与隔离验证，不是 AKS 发布回执。此前 892e397 的生产状态来自用户提供的独立检查报告；本轮没有再次连接生产。

## 响应与成功判定

三个 provider 共享 `response_semantics.py`：传输状态、协议有效性、生成终态和可信用量分别判断。HTTP 200 的错误对象、显式 HTML、无法编码的 JSON 或重复关键判别字段不能计为 completed。

非流式异常 200 返回 502，带结构化错误及 `X-Should-Retry: false`。已开始 SSE 的请求保持 HTTP 200，按 Messages / Responses / Chat 的格式返回错误终态；不会重发响应头或混入另一轮生成。结构化字段包括上游状态、受支持的上游错误码、原始 Retry-After、LB 请求 ID、原因类别、执行确定性与重试提示。

SSE 的响应头在上游结果已知前提交，因此所有推理 SSE 预先携带 `X-Should-Retry: false`；随后由错误事件说明实际原因。无法确认执行结果的本地流错误同时携带 `retryable=false` 和 `execution_certainty=unknown`。上游原生 SSE 终态帧保留原始字节和已有字段，网关不改写其中的供应商载荷；所有流仍有统一的禁止重试响应头。这不改变 LB 内部已有的明确 429 处理，也不保证客户端应用层不会自行重发。

执行不明错误不自动重放。显式 429 保持既有的有界重试/等待契约，Retry-After 不被缩短；重试提示不替代调用方自身的总预算。OpenAI/Anthropic SDK 的离线用例验证了禁止重试 header，但不能保证其他客户端采用同一策略。

上下文超限统一归为 `context_window_exceeded`；永久输入拒绝不使健康端点熔断。不按 MB 猜测模型 token 上限，不静默裁剪请求，不自动换模型。strict 参数模式仍只保证网关不会静默丢弃已知字段，不代表供应商原生支持 schema。

## 用量与结果

无可信 usage 的错误不再创建零 token 的成功事件；有可信计数的失败、incomplete 或中断保留已观察用量。每条流最多结算一次，后续清理不会重复入账。生成成功与客户端实际收到结果仍然不同。

新账本事件增加 `generation_outcome` 与 `usage_fields`，仍使用现有 JSON payload 和批次事务，无新增 DDL。旧载荷缺少结果字段时保持 unknown，不回填猜测。历史 `usage_daily.requests` 是已记录事件数，不应作生成成功率分母；usage errors 也不是端点熔断计数。

失败恢复、稳定 batch ID、payload hash 和同事务累加保持原有语义。待写队列仍在内存，Pod 硬丢失可能丢失尚未提交事件；本轮不实现 durable outbox，也不把模型估算费用当作实际账单。

## 观测和报告

- `lb_request_reasons_total` 使用有界原因标签，区分相同错误在 HTTP/SSE 中的外层状态差异。
- `lb_cleanup_*` 记录回收任务的活动数、结束结果、耗时与最老未结束年龄；不把任务数量当连接数量，不取消原有 shield owner 来伪造硬超时。
- [机器可读指标契约](../operations/metrics-contract.json) 声明指标族、类型、单位和标签；`python -m operations.metrics_contract` 输出同一契约。
- v2 报告保存 labels 和 Pod/容器生命周期。Counter 分段差分；Gauge 展示实际观测值及范围；Histogram 的 bucket/count/sum 使用一致覆盖区间；Summary 不跨实例拼接分位数。
- 缺样、重启、错误采集和缺失指标保持 partial/unknown。不同 Pod 独立呈现，不把混合 Service 采样伪装成实例连续序列。v1 counter 输入继续可用。

OpenClaw 的采集器和日报没有改动；本仓库提供的契约与工具尚不等于外部系统已经采用。

## 构建与验证

`requirements.in` 表达依赖来源，运行、测试和协调器分别有完整精确版本锁。基础镜像固定 digest；`build-info.json` 记录运行文件 hash、依赖版本及 OS 包清单。OS 仓库安装仍有时点差异，不宣称整个镜像字节可复现。

标准 Dockerfile 同时提供在线和显式离线依赖阶段。离线方式使用 `operations.build_wheels` 核对 PyPI SHA-256，再通过 `--build-arg DEPENDENCY_STAGE=dependencies-offline` 构建；没有关闭 TLS 验证。`.wheelhouse` 和私有发布目录均不纳入 Git；私有配置和密钥不进入 Docker build context。

部分旧测试使用 `{usage:{}}` 或空 choices 代表成功。本轮把这些成功夹具补成真实终态响应，原有重试次数、lease 结算、取消与资源释放断言保留。新增负例则要求这些不完整响应被拒绝，不能靠改弱断言获得绿色结果。

最新测试范围和未验证项以 [验证记录](reviews/2026-09-20-post-upgrade-hardening/IMPLEMENTATION_PROGRESS.md) 为准。安全发布工具详见 [发布与恢复说明](RELEASE_TOOL.md)。
