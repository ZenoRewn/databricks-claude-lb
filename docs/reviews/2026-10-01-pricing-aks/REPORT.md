# 每周价格刷新与 Logo AKS 发布回执

Author: Zeno Ren

2026-10-01 已将 [PR #14](https://github.com/ZenoRewn/databricks-claude-lb/pull/14) 合并到 GitHub `main`，并将应用源码 `7b71cdb7a222cddaeb144086564c9216f86b54a3` 发布到 AKS。集群持久回执为 **succeeded**；北京时间 **14:27:13–14:28:10** 维护并恢复路由，时长 **57.117 秒**。业务、账本与清理验收通过，未回滚，清理错误为空。结构化证据见 [receipt.json](receipt.json)。

## 生效内容

- Copilot 内置基线补齐至 35 个模型、47 行档位；应用内每 7 天后台刷新官方价格，失败保留有效旧表并在 1 小时后重试。历史已累计估算不重算。
- AKS 内已真实成功获取并安装价格快照，状态 `ok`、`stale=false`。最近成功为北京时间 **2026-10-01 14:27:59**，当前下次计划为 **2026-10-08 14:27:59**。这证明首次刷新和计划设置，不代表已经观察了一周。
- Dashboard 已接入刷新日期、最近成功、下次检查及异常提示；用户选择的 **01 汇流**透明 Logo 已随 HTML 生效。运行 HTML hash 与 Git 源码一致。
- 价格缓存位于当前 Pod 可写文件系统；本轮没有新增 PVC。Pod 重建后会使用内置基线并重新获取价格。

## 身份与发布前验证

应用单平台 registry manifest 为 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:6b66730ed8149b505ad6d198a1a26806d585849301127e5f000616c2a86f2679`，平台 linux/amd64；registry manifest 原始内容 hash、不可变拉取结果和运行 Pod imageID 一致。21 个运行文件及依赖清单与 Git 源码一致；公网 [`/version`](https://lb.zeno.ink/version) 返回 `7b71cdb`、完整提交与 `verification=matched`，新 Pod 零重启。

PR 与合并后 `main` 的六项 CI 分别全部通过，覆盖 Python 3.11/3.12/3.13、Docker、MySQL 8.0/8.4；[main CI](https://github.com/ZenoRewn/databricks-claude-lb/actions/runs/36824375194)。最终应用镜像内 52 项测试和 55 个子测试通过，未挂载应用源码。原本地与视觉证据见 [实现验证](../2026-10-01-pricing-refresh/VALIDATION.md)。

发布仅更新应用 image 与源码注解；两个 Service 的完整 selector、Ingress、配置/凭据、资源、副本策略、探针、preStop、90 秒 Pod grace 和既有双协调器均保留。维护前完成新鲜现场与配置指纹核对、最小补丁比较、server dry-run、健康基线、恢复与验证脚本；维护由跨节点协调器持有互斥 Lease 和持久回执，不依赖本机持续在线。预算 600 秒，前向 360 秒、排空 180 秒、预留恢复 240 秒；固定合成业务探针总输出上限 1152 tokens。

数据库于 UTC **2026-10-01T06:23:03.887992–2026-10-01T06:23:05.386023** 取得两张 InnoDB 表的只读一致性快照：**334 条日汇总、6175 个账本批次**。gzip 内容逐行校验通过，并在隔离 MySQL 8.4.11 中真实恢复，行数与规范化内容 hash 一致；TLS/主机名验证保持开启，network=none，无宿主端口，恢复容器已移除。私有备份保留，未覆盖生产数据库；备份后的 **6175 个历史 batch ID/payload hash 全部保留**。最初一次读取连接 EOF 已记录，未将失败尝试当作备份成功。

## 发布后验收

| 业务探针 | 实际 provider | 结果 |
|---|---|---|
| Messages JSON / databricks-claude-opus-5 | Databricks | 非空 LB_OK、成功终态、唯一 completed 用量事件 |
| Responses SSE / gpt-5.4 | Copilot | 重组后的非空 LB_OK、成功终态、唯一 completed 用量事件 |
| Chat SSE / gpt-5.6-luna | Copilot | 非空 LB_OK、成功终态、唯一 completed 用量事件 |

provider 来自同一请求的诊断与账本回读；没有为补证据重新推理。两个 Service 均恢复原 selector，各有一个 Ready、非 terminating 的新 Pod 后端。Deployment/Service 所有权标记、Pod finalizer、应用 pause/drain 均已清理；公网三个健康入口均为 200，无凭据调用受保护 API 仍为 401，Dashboard 保留原 OAuth 302 跳转。

发布完成后连续观测 **114 轮、约 252 秒**，每轮检查三个公网健康入口，全部返回 200，无采集错误；维护窗口内观测到预期 503。此结果是短时验收，不代表长期 SLA。

## 验证边界

本轮由主 agent 完成代码/计划复核、CI 和实际发布验收，不宣称新增独立 QA。未改动的准入、重放和发布器实现保留之前已完成的独立评审。未重跑全部 provider/endpoint、Azure 独立调用、真实客户端长会话、生产负载或节点故障演练。生产 Dashboard 的 OAuth 边界、运行 HTML 和源码已核对，视觉证据来自本地合成明暗主题预览。

本回执提交仅更新文档，不再次滚动已发布的 `7b71cdb` 应用。完整计划、配置/Secret、Pod/Lease 身份、请求 ID、账本回读及观测原始数据保留在本机私有发布目录，不进入 GitHub。
