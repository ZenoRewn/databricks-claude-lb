# LB 协议与发布工具：本地交付验收

Author: Zeno Ren

日期：2026-09-21（验证开始于 2026-09-20）。本报告对应本地分支 `codex/protocol-release-hardening-20260920`，最终运行代码提交 `a24d48cc8de038a54e8246c87b71a2f4c89647d5`。

三个工作包已实现，并完成下述本地及隔离验证。没有推送 GitHub、连接或修改 AKS、修改 OpenClaw，也没有调用真实模型供应商。独立发布复核及生产环境验收仍是后续发布门禁；本报告不冒充独立评审或生产发布回执。

## 交付内容

| 工作包 | 已交付行为 | 入口 |
|---|---|---|
| 协议与记账 | 三 provider 的结果判定、异常 200/HTML 拒绝、SSE 错误提示、统一上下文归因、失败可信用量、一次结算 | [兼容说明](../../POST_UPGRADE_HARDENING.md) |
| 观测与构建 | 类型/标签/单位契约、v1/v2 报告、cleanup 指标、精确依赖锁、基础镜像 digest、文件及 OS 包清单 | [指标契约](../../../operations/metrics-contract.json) |
| 发布与恢复 | CLI、双副本协调器、Lease、持久状态、受限暂停/恢复、旧 writer 证据、回滚与人工接管 | [发布指南](../../RELEASE_TOOL.md)、[隔离安装清单](kind-installation.json) |

保守 POST 重放、账户亲和、默认 compat 参数模式和原有预算保持。未加入容量 503 自动重试、durable outbox、应用双副本或蓝绿发布。

## 最终代码回归

| 环境 | 实际结果 | 原始输出 |
|---|---|---|
| 本机 Python 3.14 | 618 passed；6 skipped；538 subtests passed | [日志](test-logs/host-release-final.txt) |
| Linux amd64 Python 3.11 | 617 passed；7 skipped；538 subtests passed | [日志](test-logs/py311-release-final.txt) |
| Linux amd64 Python 3.12 候选应用镜像 | 617 passed；7 skipped；538 subtests passed | [日志](test-logs/image312-release-final.txt) |
| Linux amd64 Python 3.13 | 617 passed；7 skipped；538 subtests passed | [日志](test-logs/py313-release-final.txt) |
| 候选协调器镜像 | 18 passed；20 subtests passed | [日志](test-logs/controller-release-final.txt) |
| 隔离 MySQL 8.0 | 6/6 事务测试通过 | [日志](test-logs/mysql80-tests.txt) |
| 隔离 MySQL 8.4 | 6/6 事务测试通过 | [日志](test-logs/mysql84-tests.txt) |

六个 skip 是另行执行的真实 MySQL 集成测试；容器中额外一个 skip 是未安装可选 OpenTelemetry SDK 的 tracing 用例。本机该用例通过。MySQL 测试覆盖提交 ACK 丢失、并发相同批次、部分写入回滚、提交后取消、共享聚合和保留期；它们使用 network=none 容器内的临时 loopback 数据库。Kind 另使用开启证书及主机名验证的 MySQL TLS。

应用容器测试没有挂载应用 Python 文件或 Dashboard：运行文件来自镜像。挂载的是测试、文档、独立 operations 工具及离线测试依赖；发布器另在其候选镜像内测试，不挂载 operations 源码。测试安装使用已校验 PyPI SHA-256 的本地 wheel，不访问供应商。

回归包含三种 API、三个 provider 的有效组合、流式/非流式、断连、重试次数、opaque-state 修复、endpoint lease、用量及响应回收。OpenAI 2.44.0、Anthropic 1.3.0 的离线 SDK 检查验证禁止重试 header；不能外推至所有 SDK 或应用层重试。

## 镜像身份

- 应用候选：`claude-lb:hardening-final-v3`（linux/amd64）。
- 协调器候选：`lb-release-controller:hardening-final-v3`（linux/amd64）。
- Kind 使用对应源码的 linux/arm64 镜像，只上传到本机 loopback registry。
- 四个应用/协调器镜像及两个 Python 回归镜像的运行文件均与最终 Git 源码逐项核对；[镜像索引](release-final-images.json) 记录实际 image ID、平台与本地 registry digest。
- [应用清单](application-amd64-release-final-integrity.json) 与 [协调器清单](coordinator-amd64-release-final-integrity.json) 包含 source revision、文件 SHA-256、Python 包及 OS 包版本。基础 Python 3.12 镜像固定 digest；apt 仓库存在时点差异，不宣称字节可复现构建。

本地 image ID 不等于可在 AKS 拉取的 registry digest。`localhost:5008` 镜像引用仅属于已隔离的试验环境；清理 registry 后不能作为发布地址。此前带 `frozen`、`r1` 等名字的证据是阶段记录，以 `release-final` 文件为最终镜像身份。

## 真实 Kind 故障演练

环境是一次性三节点 Kind（一个 control-plane、两个 worker），显式私有 kubeconfig，目标 `lb-lab/claude-lb`。协调器在不同节点运行，两个 Service 都纳入计划。所有业务请求使用合成上游和测试数据。

| 场景 | 观测结果 | 回执 |
|---|---|---|
| 892e397 首次兼容升级 | succeeded；旧入口摘流路径、旧 writer 终止、三 API 与三条账本关联；维护约 15.6 秒 | [bootstrap](kind-bootstrap-r1.json) |
| 最终源码的新版本发布 | succeeded；完整路由、业务与账本验收；恰好 3 次供应商模拟发送 | [最终发布](release-final-05.json) |
| Service 实际写入后丢 ACK | 读回后完成，3 次发送 | [ACK 丢失](ack-gate-01.json) |
| 业务 exec 已完成后丢 ACK | 读取缓存回执，3 次发送，无重复推理 | [exec ACK](exec-ack-loss-01.json) |
| 回执存储失败 | 旧路由恢复，最终 rolled_back，0 次发送 | [日志失败](frozen-journal-04.json) |
| 新后端验证失败 | 回滚恢复，0 次发送 | [回滚](rollback-01.json) |
| 长请求超过排空预算 | rolled_back；旧 Pod 保留，请求正常结束，只有原请求 1 次发送 | [长请求](long-stream-01.json) |
| 人工接管 selector | needs_attention，保留人工变更；显式解除冲突后恢复 | [拒绝覆盖](manual-takeover-01-attention.json)、[恢复](manual-takeover-01.json) |
| 无健康回滚目标 | needs_attention，不盲开流量；故障解除后受控恢复 | [关闭保持](unavailable-rollback-01-attention.json)、[恢复](unavailable-rollback-01.json) |
| 新 Pod 无法调度 | 确认从未调度的 Pod 删除后恢复旧版；rolled_back，0 次发送，维护约 57.9 秒 | [调度失败重测](scheduling-04.json) |
| 双执行者竞争 | 真实 API 上 10 轮同时争抢均只有一个持有者；新代次拒绝旧执行者 | [竞争](kind-lease-contention.json) |

协调器在 **gating、pausing、draining、stopping、starting、verifying_backend、restoring_routes、verifying_business、finalizing** 九个阶段逐一被终止，均由新的 Lease 持有者接管并完成；每轮 3 次模拟发送，没有留下无主维护 selector。对应 `crash-*.json` 分别保存实际结果。测试终止的是协调器容器，没有强删应用 Pod 制造 writer 退出证明。

首次升级 fixture 使用已核对的 892e397 应用文件，运行在本地固定基础依赖中，不能等同于原 AKS 镜像。九阶段故障矩阵在此前候选执行；随后修复了条件补丁冲突、恢复错误信号及 SSE 提示，分别补测了调度失败、日志失败、发布器回归和最终镜像完整发布，没有声称所有场景都在最后一个 image ID 上重跑。

## 发现的问题与保留的失败证据

1. 回执写入失败后，旧路由虽恢复，但下一轮把 recovery-only 标记误判为外部漂移。修复恢复模式识别后自动回滚通过。保留 [首次失败](kind-journal-initial-failure.json) 与 [人工恢复](kind-journal-recovered-r6.json)。
2. 调度失败暴露未调度 Pod 删除状态竞态和 Kubernetes JSON Patch 422 条件冲突。现在先读回 UID/resourceVersion，再重新协调；不放宽条件写入。保留 [首次失败](kind-scheduling-initial-failure.json)、[第二次失败](kind-scheduling-second-failure.json) 和各自恢复回执。首次演练中短至 5 秒的 drain fixture 也曾提前安全回滚，不能当成调度场景通过。
3. 补充负例发现 SSE 响应头和 Databricks 本地错误缺少双重禁止重试信号，以及原生 context_window_exceeded 没有统一 neutral 结算。修复后五个 provider/API 组合均通过，正常/原生 SSE 帧仍保留字节契约。[负例](test-logs/sse-retry-red.txt)、[上下文负例](test-logs/native-context-red.txt)、[针对性回归](test-logs/sse-and-context-green.txt)。
4. 最终镜像夹具曾漏挂说明文档；失败属于测试环境文件缺失，已修正挂载后重跑完整套件。另一次 journal fixture 的 `phase=null` 导致没有注入故障，普通发布虽成功但演练被判失败；[回执](kind-journal-fixture-not-injected.json) 保留，修正后还要求读回 `consumed=true`。
5. 回执存储与紧急恢复同时失败时，增加独立的脱敏错误信号；不会吞掉恢复失败或标记成功。

历史独立测试的一处 `{usage:{}}` 成功夹具已改成有效 completed 对象，原断言未削弱；[原始字节](../../../tests/fixtures/revision2/test_adversarial.original.py.txt) 另存。它属于适配后的回归，不能描述为本轮独立审查。

## 使用边界与后续门禁

- 发布器 v1 仅支持单容器、单 writer、MySQL 账本维护模式。Kubernetes RBAC 对动态 Pod exec 不能按标签收窄，目标应使用专用命名空间；只读 Node 权限用于 writer 证据。
- Kubernetes 资源发现不能证明不存在所有网络绕行，生产入口覆盖仍需明确核实。旧节点失联且无终止证据时必须 needs_attention；TTL 不自动开放路由。
- 300 秒是有恢复目标时的预算，不是控制平面、节点或恢复后端失效时的可用性保证。回滚保留新增账本，不回放结果不明的业务探针、不恢复旧数据库快照。
- usage 待写队列仍在内存，未提供硬故障零丢账；没有改变 schema，也未回填历史结果。
- 公网 TLS/真实 Ingress、真实供应商、真实客户端、长期负载及生产安装未验收。Kind 的集群内测试地址不能证明真实公网链路可用。
- 独立发布复核仍待执行；发布工具的隔离验收已单独进行，未用应用单测替代。任何后续 GitHub 发布、AKS 安装或真实供应商探针均需按当次任务范围执行。

一次性 Kind 集群、loopback registry、预览服务和私有临时目录已清理；本地最终候选镜像保留，[清理回执](cleanup.json) 已核对。结构化结果见 [validation.json](validation.json)，重现入口见 [REPRODUCE.md](REPRODUCE.md)。原有未提交的 `AGENTS.md` 和事故复盘目录保留。
