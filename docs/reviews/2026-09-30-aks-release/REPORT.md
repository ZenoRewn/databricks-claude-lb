# AKS 发布回执：2026-09-30

Author: Zeno Ren

本轮优化已成功部署到 AKS。集群持久记录为 `succeeded`，北京时间 **21:03:04–21:04:20** 进入维护并恢复路由，记录时长 **75.165 秒**；21:04:41 完成业务、记账和清理验收。对话中断期间，集群双协调器继续执行并完成发布，没有依赖本机交互接力。未执行回滚，清理错误为空。

结构化结果见 [receipt.json](receipt.json)。私有备份、配置、完整日志、请求 ID、Pod/Lease 身份及完整状态机回执保存在私有发布目录，未进入 GitHub。

## 运行版本与变更范围

| 对象 | 已核验身份 |
|---|---|
| 应用源码 | `1d75685655f672177374cf3327cd16c4bcffe2db`，已随 [PR #8](https://github.com/ZenoRewn/databricks-claude-lb/pull/8) 合并 |
| 应用 registry manifest | `sha256:3389f5ea25e03c4170d5b8243b38d98fddbb7057b455868b521823a8e926f42c`，linux/amd64 |
| 应用 runtime manifest | `87b7a17d60a24d1aa78648d343bbbbb6462cb8f41aed4138be144d119fdb7ffd`；21 个运行文件及依赖锁与 Git 匹配 |
| 协调器源码 | `11746b3d1aa8c24bd0e7fb313e52d39f5fdadf23`，已随 [PR #9](https://github.com/ZenoRewn/databricks-claude-lb/pull/9) 合并 |
| 协调器 registry manifest | `sha256:b481e1f291ac1eef1c0a3bd746632f0f3de4f129930e3879444d8c009beee3e4`；两个跨节点实例各核验 14 个源文件 |

应用只更换不可变镜像并修正源码注解；单副本、现有滚动策略、环境变量、资源额度、配置/Secret、Ingress、探针、preStop 和 90 秒 Pod grace 保持现场值。协调器仅修改镜像，保留两副本、调度、权限和资源配置。没有整份 apply 仓库示例 manifest。

镜像、Pod imageID、实际文件 hash、源码注解和公网 `/config/effective` 的构建身份相互匹配。新 Pod 核验时 Ready、零重启；发布器留存了旧 writer 终止证据，再启动新 writer。

## 发布前门禁

- 应用目标镜像完成既有回归，源码身份修复另通过完整主机回归、协调器镜像测试和只读独立 QA，见 [应用验证](../2026-09-30-lb-contracts/VALIDATION.md) 与 [发布工具修复](../2026-09-30-release-source-identity/REPORT.md)。PR #9 与合并提交 `80da865` 各六项 CI 全部通过，含 Python 3.11/3.12/3.13、Docker smoke、MySQL 8.0/8.4。
- 用户确认业务均经过 Service/Ingress；现场发现的两个 Service 和全部相关 Ingress 已纳入计划，配置和凭据指纹在提交前再次核对。API Server dry-run 校验完整目标模板。
- 两张 InnoDB 用量表完成只读一致性备份。较早快照已在隔离 MySQL 8.4.11 中实际恢复，行数及规范化内容 hash 一致；最终快照另于 UTC 12:54:55–12:54:56 获取，包含 331 条日用量记录、5,747 个账本批次，并通过逐行完整性校验。最终快照未再次做恢复演练，不混淆两类证据。
- 最后一次备份的首个新建连接在 5 秒时超时，未输出备份数据；一次 15 秒连接预算的只读重试成功。发布器仍使用原有的真实镜像、导入和 5 秒数据库连接预检，预检通过后才进入维护；未放宽生产准入或推理预算。
- 公网 TLS 健康基线和持续采样均就绪。维护总预算 600 秒、前向 360 秒、排空 180 秒，预留 240 秒恢复。协调器使用集群 Lease、持久计划/回执、条件写入及旧 writer 终止证据。

## 线上验收

| 公网检查 | 实际 provider | 结果 |
|---|---|---|
| Messages JSON / `databricks-claude-opus-5` | Databricks | 非空 `LB_OK`、成功终态、唯一 completed 账本事件 |
| Responses SSE / `gpt-5.4` | Copilot | 重组后的非空 `LB_OK`、成功终态、唯一 completed 账本事件 |
| Chat SSE / `gpt-5.6-luna` | Copilot，经缓冲 Responses 适配 | 非空 `LB_OK`、成功终态、唯一 completed 账本事件 |

provider/endpoint 归属通过既有 request ID 的账本和发送诊断核对，没有为补证据重新推理。现网采用 Copilot 优先，因此 `gpt-5.4` 公网成功不能当成 Azure 成功。

Azure 另在候选镜像的独立进程中，对一个已配置的 East US 2 端点执行 `gpt-5.4` Responses SSE 检查，最多 512 输出 token，获得非空 `LB_OK` 和成功终态。这证明该镜像的 Azure 路径及该端点可用；它没有经过公网 provider 选择，也不构成生产应用账本或所有 Azure 端点验收。现场路由策略保持原样。

两个 Service 已恢复完整原 selector，各有一个 Ready、非 terminating 的新 Pod 后端。公网 `/health/live`、`/health/ready`、`/health/accepting` 均返回 200；暂停和 drain 标记均已清除。无凭据访问三个推理入口、token-count 和 `/config/effective` 均返回预期 401；带有效凭据的构建身份读取通过。这些主动鉴权检查产生的 401 不是供应商推理故障。

默认 `/metrics` 仍为 `lb-metrics-v2`，显式 v3 请求返回 `lb-metrics-v3`。新 Pod 核验时 usage backend ready，未观察到该进程的用量 flush/rejected 错误或诊断丢弃；业务流量产生的短期待写 buffer 仍按正常周期落盘，不宣称内存队列永久为零。

最终快照中的 **5,747 个历史账本 batch ID 与 payload hash 全部保留**，三条发布探针分别且仅一次记账。没有恢复生产数据库快照、清空账本或执行 schema 迁移。预检 Pod 和一次性 Azure 验证 Pod 已清理；恢复所需镜像及私有备份保留。

## 验证边界

这次是受限生产冒烟与发布验收，未覆盖所有 provider/endpoint 组合、真实客户端长会话、长时间负载、节点/控制平面故障注入，不能据此宣称容量收益、长期稳定性或零 RPO。现有单副本发布仍有短暂维护窗口；本轮没有实施扩容或 durable outbox。

OpenClaw 自动化、脚本、调度、基线和历史数据未作修改。监控接口保持默认 v2；这不等于已部署外部 watcher 的新采集逻辑或验证全部外部告警链路。
