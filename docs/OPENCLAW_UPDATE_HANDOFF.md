# LB 本地候选更新：给 OpenClaw 的交接说明

Author: Zeno Ren

日期：2026-09-20

> 历史候选交接，以下“尚未部署”描述的是本文编写时点。用户提供的后续独立检查报告记录了 892e397 已在生产运行。本轮新的本地优化及监控契约见 [升级后加固说明](POST_UPGRADE_HARDENING.md)；未修改或接入 OpenClaw，不应把这里的旧版本或本地新候选当作当前实时生产状态。

**本地代码已优化；本文不是 AKS 发布回执。尚未更新生产镜像、Deployment、OpenClaw 自动化或监控脚本。** 实际部署后，先核对运行版本，再按以下契约调整现有监控。实现与默认参数详见 [SERVICE_RELIABILITY.md](SERVICE_RELIABILITY.md)。

## 先确认真实运行版本

- 本地分支：`codex/service-reliability-20260920`。
- 运行代码候选提交：`e4dde8d`（此前各批次提交见 `git log`）。
- 最终本地验证镜像标签：`claude-lb:reliability-review-20260920-r2`；镜像 ID、应用文件 hash、测试结果见 [验证记录](reviews/2026-09-20-service-optimization/implementation-r2/validation.json)。本地 image ID 不是已推送 ACR 的可拉取 digest。
- 部署后重新记录实际 registry digest、Pod UID、容器启动时间、generation、关键模块 hash 与配置。未确认新版本时，不把缺失的新指标当作零。

## 新版应该如何判断成功与失败

| 信号 | 正确含义 |
|---|---|
| `lb_requests_started_total` / `lb_requests_finished_total{outcome}` | LB 一次入口调用及其唯一结束结果；不等于 OpenClaw 整个任务 |
| `lb_endpoint_admissions_total` | 上游端点 lease 准入次数 |
| `lb_upstream_send_started_total` | 实际发起 HTTP client 调用的次数，含同 lease 内 auth/opaque 修复；不证明上游已执行 |
| `lb_request_duration_seconds{outcome}` | 成功、失败、取消等分别统计的完整请求时间 |
| `lb_admission_*` | 本地在途、排队、原始输入预留与本地拒绝原因 |
| `lb_usage_*` | 待写、in-flight、写入失败/成功、拒绝用量事件、最近成功时间和最近持久化状态 |
| `lb_parameter_policy_requests_total{action,parameter}` | 受本地字段决策影响的请求数，不是字段总数或 token 数 |

HTTP 200 + SSE error 不再按请求完成计算；incomplete、unknown、deadline_exceeded、overloaded、client_disconnected 等各自呈现。旧 endpoint/completed/global stats 保留兼容，但不得替代新的请求结果分母或最终客户端业务验收。

关联请求使用 `X-LB-Request-Id`，原 `X-Request-Id` 保留；调用方可用 `X-LB-Operation-Id` 跨自身重试关联。operation ID 不提供自动去重或 exactly-once 推理保证。日志 source_tenant 由凭据映射决定，未配置独立来源时就是 default，不能根据内容猜成 OpenClaw。

建议在现有异常分析中增加新结果/准入/记账异常的候选信号，并保留最低样本量、窗口和持续性判断。不要只看 endpoint errors：本地排队拒绝、总 deadline 取消或记账失败可能不增加上游端点错误。

## 调用方需要了解的行为

- 本地默认总预算 1800 秒，上游响应头预算 180 秒；响应头到达后解除启动计时，heartbeat 不续期总预算。超时不会由 LB 自动补发执行不明的 POST。
- 仍需协调 SDK/OpenClaw 的自身重试预算。LB 没重放不代表 SDK 没重试；部分输出后禁止拼接第二次生成或重复执行有副作用的工具。
- 输入上传预算 120 秒，原始单请求 64 MiB；进程默认活动 128、排队 64、排队等待 10 秒、原始输入预留 128 MiB。属于本地保护参数，尚未按实际负载校准，也不是供应商配额。
- 默认兼容模式通过 `X-LB-Dropped-Parameters` 显示已知字段移除。需要这些字段必须保留时，可用 `X-LB-Strict-Parameters: true` 在推理前明确失败；这不代表上游支持已获认证。schema/JSON/工具结果仍需业务验证。
- 503 重试策略没有放宽；账户 pinning/亲和、危险 POST 重放禁令和响应回收设计保留。

## 发布前的实际前置条件

1. 新 MySQL `usage_batch_ledger` 是追加表；需检查建表权限和迁移。它与全部模型增量同事务提交，可处理 ACK 丢失与部分写入。payload 按 usage retention 清理，最小去重回执保留。
2. 待写事件仍在进程内存，不能承诺 Pod 硬丢失零丢账。出现 pending 增长、flush failure、rejected events，应作为记账缺口处理，不能重新生成来“补账”。
3. `/health/accepting` 是本地接流量就绪；`/health/ready` 保留严格 provider 诊断；共享上游故障不应自动解释为需要重启整个 Pod。
4. drain 需要配合足够的 preStop/grace 与另一 Ready 实例或维护窗口。不要直接覆盖当前 K8s 模板；此轮没有增加副本或节点。单副本加等待会影响新请求，不能称为零中断发布。
5. 真客户端、供应商契约、压力/长期稳定性和节点维护演练还需发布流程验收；本地合成测试与真实本地 MySQL 测试不是生产认证。

## 保留现有 OpenClaw 管理边界

根据用户提供的信息，watcher 的权威逻辑在 Gateway automation 的 `trigger.script`/`payload.message`，不是备份仓库中的某个 JS 文件。三个任务 ID 保留：

- watcher：`29abb4b6-a517-4ab5-99be-bed87049a6cf`
- 小时归档：`d33f2cb3-09bc-407e-a5f1-318b8cfc9abb`
- 每日汇总：`b268ddf5-5c72-4dd9-92c5-51b9ec1ab00f`

保留 `/openclaw/tmp/lb-monitor-probe.sh`、`/openclaw/scripts/lb-monitor-records.py` 和 `/openclaw/data/lb-monitor/`。不存在的旧 `lb-watcher-trigger-v2.js` 不能当来源；只 clone 配置备份仓库不能恢复有效任务和全部历史数据。

本地 `operations.reporting` 只是可选组件，尚未接入。若后续采用，先适配实际快照格式，再验证窗口边界、Pod/reset、缺指标与归档回执；不要创建重复任务，也不要覆盖历史报告冒充原件。此次没有清理任何 OpenClaw agent、文件或自动化。
