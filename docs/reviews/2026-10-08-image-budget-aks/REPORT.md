# 图片像素准入修复 AKS 发布回执

Author: Zeno Ren

2026-10-08 已将 [PR #17](https://github.com/ZenoRewn/databricks-claude-lb/pull/17) 合并到 GitHub `main`，并将应用源码 `284ca53ec94b0dbb0d31325c09684bf1aa12fc02` 发布到 AKS。集群发布回执为 **succeeded**；北京时间 **16:15:48–16:17:26** 维护并恢复流量，时长 **98.211 秒**。三个常规业务探针、账本与清理通过，未回滚，清理错误为空。

**额外大图探针通过了 LB 缩图与准入，但端到端生成失败：Copilot 返回 503。** 该结果没有被 HTTP 200 或常规探针成功掩盖，也没有重发推理。结构化结果见 [receipt.json](receipt.json)。

## 生效内容与范围

- 内嵌图片先检查单张 40M / 源累计 400M，再逐张缩至最长边 1280px，最后检查原有 100M 输出像素预算。编码不足 200KB 的大尺寸图同样处理；默认保留图片与顺序，缩图有损。
- 三个 API 共用图片处理流程；Pillow bomb 显式拒绝，缓冲显式释放，取消请求后仍等待真实线程结束再释放压缩名额。
- 实际 Deployment spec 仅改变应用 image 和 source-revision 注解。现有 500m/2Gi、单副本、压缩并发默认 2、完整 Service selector、Ingress、配置/凭据、探针、preStop、90 秒 grace 和双协调器均保留；不改数据库 schema 或推理 POST 重放策略。

源码及本地 TDD/合成资源样本见 [发布前报告](../2026-10-08-image-budget/REPORT.md)。单图/累计/张数仍有保护上限；本轮没有重放原始用户请求，不能保证未知图片构成的原请求必然满足所有限制。

## 固定身份与维护前验收

应用单平台 registry manifest 为 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:2b0e273e26f181ad7258c9ead871a86242d91151122f85c4630eb38a0161b9c6`，平台 linux/amd64。上传结果、registry 原始 manifest hash、不可变拉取、运行 Pod imageID 相符；manifest 指向实际测试过的镜像配置。21 个运行文件及依赖清单 hash 与合并源码一致，公网 `/version` 返回 `284ca53` 与 `verification=matched`，新 Pod 零重启。

PR 和合并后 `main` 的六项 CI 均通过：Python 3.11/3.12/3.13、Docker、隔离 MySQL 8.0/8.4；[main CI](https://github.com/ZenoRewn/databricks-claude-lb/actions/runs/37747733634)。正式镜像内 68 项测试通过，应用源码没有通过挂载替换。独立 QA 核对源码、发布器历史故障证据、两协调器运行 hash、当前现场、备份、恢复、计划及一次性验收脚本。未重跑未修改发布器的全部 Kind 故障矩阵。

维护前完成新鲜发现、完整 spec 最小差异比较、server dry-run、三个连续公网健康样本、恢复材料及验证脚本。计划 hash 为 `504508547464deaf8d772130c10cdd276cadf7d690bc4342c6e7e405bae5e808`。既有跨节点协调器以互斥 Lease 和持久回执持有维护责任；总预算 600 秒、前向 360 秒、排空 180 秒、恢复预留 240 秒。没有强杀长请求。当前资源和既有约定指向两个 Service；未知外部直连 Pod 配置不能仅由 Kubernetes API 证明不存在，此边界保留。

数据库于 UTC **08:07:28.640806–08:07:30.869145** 取得两张 InnoDB 表的一致性只读快照：`usage_daily` **368 行**、`usage_batch_ledger` **8,102 行**。源 MySQL 8.4.9-azure，隔离恢复至 MySQL 8.4.11；字段、索引、引擎及规范化内容 hash 一致。TLS/hostname 验证保持开启，恢复容器 network=none、无宿主端口、临时数据已清理。私有备份保留，未覆盖生产；上线后全部 8,102 个历史账本 batch ID/payload hash 保留。

## 公网业务与收尾

| 验收 | 模型 / provider | 结果 |
|---|---|---|
| Messages JSON | databricks-claude-opus-5 / Databricks | 非空 LB_OK、成功终态、唯一 completed 账本事件 |
| Responses SSE | gpt-5.4 / Copilot | 重组后的非空 LB_OK、成功终态、唯一 completed 账本事件 |
| Chat SSE | gpt-5.6-luna / Copilot | 非空 LB_OK、成功终态、唯一 completed 账本事件 |
| 额外大图 Responses SSE | gpt-6-astra / Copilot | LB 准入通过、9 图保留；上游 503，生成未完成 |

两个 Service 各恢复一个 Ready、非 terminating 的新 Pod 后端，完整 spec/Ingress/配置指纹核对通过。Deployment/Service 维护标识、Pod finalizer 和应用 pause/drain 均清理。公网三个健康入口为 200；无凭据调用受保护 API 仍为 401；Dashboard 保持 OAuth 302。现有 v2/v3 监控指标、usage backend ready 与接流状态正常。

恢复后观测 **114 轮、约 240 秒**，每轮三个公网健康入口均为 200，无采集错误。维护内观测到预期 503。该短时结果不是长期 SLA。

## 大图验收的具体边界

额外探针是一次新的合成请求，固定 `gpt-6-astra`、9 张 4000×3000 PNG，共 **108,000,000 原始像素**，JSON 507,818 字节，最大输出 512 tokens；与三个常规探针合计最大输出额度 1664 tokens。没有使用用户图片或重放用户会话。

响应包含 `images.compressed`，上下文日志确认仍有 9 图，未丢弃参数/图片。只发生一次上游 POST。随后收到 Copilot HTTP 503；下游已开始 SSE 所以 HTTP 状态为 200，但没有成功终态、没有答案文本，额外端到端验收明确未通过。只读账本查询未找到该探针事件，不据此推断成功持久化或零计费；上游执行状态仍未知，未自动重试。

证据确认 LB 的原始 100M 像素阻断已解除，不能进一步断言上游 503 是容量、模型、图片限制或与图片无关的问题。近期另有同模型正常请求完成，不证明这个大图请求能够完成。未测全部 provider/endpoint、用户原请求、真实客户端图片细节效果、生产压力或故障演练。

完整计划、配置/Secret、备份、Pod/Lease 身份和请求诊断保留于本机 0700/0600 私有发布目录，不进入公开 GitHub。本回执仅更新文档，不再次滚动应用。
