# 文档导航

Author: Zeno Ren

按要完成的任务查阅。实现说明、历史设计和生产发布回执分别维护，历史文档中的“未部署”或旧默认值只适用于其写作时点。

## 开始使用

| 任务 | 文档 |
|---|---|
| 项目介绍与本地启动 | [README](../README.md) |
| 渠道、认证、存储与客户端 | [配置指南](CONFIGURATION.md) |
| 开发、测试、Dashboard 预览与镜像 | [开发指南](DEVELOPMENT.md) |
| API 与鉴权完整清单 | [接口表](../CLAUDE.md#api-端点) |

## 运维面（Dashboard「密钥与渠道」页）

| 要做的事 | 文档 |
|---|---|
| 解锁这一页（面板密码、会话 cookie、退避与回落） | [解锁契约](CONFIGURATION.md#dashboard-unlock-contract) |
| 在页面上改面板密码、以及多副本边界 | [改面板密码](CONFIGURATION.md#changing-the-panel-password) |
| 五层策略怎么叠加、谁做减法、在哪收口 | [策略层总览](CONFIGURATION.md#policy-layers-overview) |
| 配「模型 → Provider → GHCP 账户」规则 | [路由规则](CONFIGURATION.md#routing-rules) |
| **某账户为何不接某模型的流量** | [排除原因闭集](CONFIGURATION.md#exclusion-reasons)、[候选矩阵](CONFIGURATION.md#candidate-matrix) |
| 账户开着却不接流量 | [已启用 vs 实际接流量](CONFIGURATION.md#enabled-versus-receiving-traffic) |
| 按密钥限渠道、以及怎么重置回基线 | [渠道限制](CONFIGURATION.md#per-key-provider-allowlist) |
| GHCP 账户先落地、确认后再启用 | [三态启用](CONFIGURATION.md#ship-inert-then-enable) |
| 录入上游凭据与 GHCP device flow | [凭据录入](CONFIGURATION.md#entering-upstream-credentials) |
| 解锁不了 / 规则不生效 | [排障 19 与 20](TROUBLESHOOTING.md) |

## 当前行为与运行

| 主题 | 文档 |
|---|---|
| 诊断、计时、参数、图片、上下文与版本化指标 | [当前契约](OBSERVABILITY_AND_CONTEXT.md) |
| 重试、熔断、账户亲和与多实例边界 | [可靠性约束](RESILIENCE.md) |
| SSE framing 与资源边界 | [流协议](STREAM_PROTOCOL.md)、[响应所有权](STREAM_OWNERSHIP.md) |
| Copilot 费用与价格覆盖 | [计费口径](COPILOT_PRICING.md) |
| Opaque state 的机制与历史实测 | [状态说明](OPAQUE_STATE.md) |
| 用量易失边界、单路由实验与证据工具 | [运维实验](OPERATIONS_EXPERIMENTS.md) |
| AKS 现场配置与受控发布 | [部署指南](AKS.md)、[发布工具](RELEASE_TOOL.md) |
| 故障症状与排查 | [排障指南](TROUBLESHOOTING.md) |

## 历史与证据

[验证与发布索引](reviews/README.md) 汇总每次复核、镜像和部署回执。最新代码不自动意味着最新生产版本；Dashboard 的 Release 显示实际构建提交。

以下保留作历史设计依据，不应单独当作当前默认配置或生产状态：

- [2026-09-20 可靠性实现](SERVICE_RELIABILITY.md) 与 [后续加固](POST_UPGRADE_HARDENING.md)。
- [2026-09-20 候选进度](OPTIMIZATION_PROGRESS.md)。
- [OpenClaw 交接事项](OPENCLAW_UPDATE_HANDOFF.md)：采用时核对真实 LB 版本，不代表外部 watcher 已改动。

旧根目录 `ANALYSIS.md` 已从当前工作树移除；原文可在 [6037755 时点](https://github.com/ZenoRewn/databricks-claude-lb/blob/6037755c0d5686de09fb51275e4057ca1711a5c8/ANALYSIS.md) 查看。代码回归仍使用的历史 fixtures、故障证据、发布指纹和私有恢复材料保留。
