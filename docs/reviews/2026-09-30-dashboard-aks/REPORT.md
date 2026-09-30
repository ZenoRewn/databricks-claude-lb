# Dashboard AKS 发布回执

Author: Zeno Ren

2026-09-30 已将 GitHub `main` 的 `ccb25cd734687090eb76cd26c2f65c173d504ec1` 部署到 AKS。集群持久回执为 **succeeded**，北京时间 **23:40:54–23:41:57** 维护并恢复路由，时长 **62.611 秒**；23:42:19 完成业务、持久化与清理验收。未回滚，清理错误为空。结构化证据见 [receipt.json](receipt.json)。

## 版本与范围

- 应用 registry manifest：`sha256:1b246e211450d440af42eba51eec5a2da78d397233b6fa96eceaee80ad5c6e2a`，平台 linux/amd64。原始 manifest 字节的 SHA-256、不可变拉取结果和 Pod imageID 一致。
- 21 个运行文件及依赖清单与 Git 源码一致；公网 [`/version`](https://lb.zeno.ink/version) 返回 `ccb25cd`、完整提交及 `verification=matched`，响应 `Cache-Control: no-store`。
- 新 Dashboard 包含 Release 面板、明亮 Token/耗时配色及刷新/错误状态优化；实际容器返回的 HTML hash 与发布源码一致，HTML 禁止缓存。
- 仅更新应用 image 和源码注解。副本数、滚动策略、资源、环境变量、配置/凭据、探针、preStop、90 秒 Pod grace、全部 Service/Ingress 和既有双协调器均保留。
- 本次没有实现前一轮 [usage 告警建议](../2026-09-30-chart-usage-review/ASSESSMENT.md)，也没有改变记账、推理重放或 OpenClaw 行为。

## 发布前门禁

PR [#11](https://github.com/ZenoRewn/databricks-claude-lb/pull/11)、[#12](https://github.com/ZenoRewn/databricks-claude-lb/pull/12) 已合并。目标 `main` 的 [六项 CI](https://github.com/ZenoRewn/databricks-claude-lb/actions/runs/36734340804) 全部通过，包含 Python 3.11/3.12/3.13、Docker 和 MySQL 8.0/8.4。目标镜像内 5 项版本/API 测试通过，实际版本接口和运行文件完成核对。

使用本次新建计划与现场快照，显式绑定 context、namespace、Deployment 和 container。两个 Service 与全部相关 Ingress 已发现；用户此前对相同入口作出的“无 Pod 直连”确认继续适用。server dry-run 和独立比较确认只改允许字段；提交前重新核对 Deployment、旧 Pod、入口及配置/Secret 指纹无漂移。

两个既有协调器跨节点 Ready，各自 14 个文件 hash 与已审阅发布工具匹配。独立准备 QA 通过；维护预算 600 秒，前向 360 秒、排空 180 秒，预留 240 秒恢复。固定三协议探针合计最多 1152 输出 token。验证、观测和恢复材料均在提交前准备完成，集群动态预检通过后才摘流。旧 writer 终止证据已写入持久回执，没有 force delete 或手动覆盖路由。

数据库于 UTC **15:28:03–15:28:05** 取得两张 InnoDB 表的只读一致性快照：331 条日汇总、5,941 个账本批次。gzip 流逐行校验通过；同一快照在本地隔离 MySQL 8.4.11 中真实恢复，行数与规范化内容 hash 一致，TLS/主机名验证开启，network=none，无宿主端口。一次性恢复容器已移除；私有备份与恢复材料保留，目录 0700、敏感文件 0600。

## 发布后验收

| 业务探针 | 实际 provider | 结果 |
|---|---|---|
| Messages JSON / databricks-claude-opus-5 | Databricks | 非空 LB_OK、成功终态、唯一 completed 用量事件 |
| Responses SSE / gpt-5.4 | Copilot | 重组后的非空 LB_OK、成功终态、唯一 completed 用量事件 |
| Chat SSE / gpt-5.6-luna | Copilot | 非空 LB_OK、成功终态、唯一 completed 用量事件 |

provider/endpoint 来自同一请求的日志与既有账本回读，没有为了补证据重新推理。备份中的 **5,941 个 batch ID 和 payload hash 全部保留**；没有把旧快照恢复到生产，也没有执行 schema 迁移。

两个 Service 均恢复完整原 selector，各有一个 Ready、非 terminating 的新 Pod 后端；新 Pod 零重启。Deployment/Service 的发布所有权和维护标记、Pod finalizer、应用 pause/drain 均已清理。

公网三个健康入口返回 200；无凭据访问 Messages、Responses、Chat、token-count 和 `/config/effective` 均返回预期 401。Dashboard 未认证访问仍为 302，转向原 OAuth 登录入口；没有取消登录保护。鉴权后的公网构建身份、公开 `/version`、容器 HTML 与运行文件核验相符。默认指标仍为 v2，显式 v3 可用。

发布完成后的 61 个公网采样覆盖约 133 秒，三个健康入口全部 200。观测器在摘流前 UTC 15:40:20/26 各有一次 ReadTimeout/ConnectTimeout，15:40:31 已恢复；这些是本地采集失败，不能当作零错误，也不能据此认定服务端故障。协调器随后的公网预检通过。维护期间观测到预期 503，路由恢复后重新为 200。

短时 usage 快照为已接收 10、已确认 8、待写 2（最旧约 5.5 秒），backend_ready=1，flush failure/rejected 均为 0，诊断丢弃为 0。计数随新 Pod 重置，短期待写处于正常周期；**不能把它解释为既有间歇 flush 失败已被修复，或未落盘事件已有耐久保障**。

## 边界与材料

最终独立只读验收复核通过：发布终态、业务与唯一账本、21 个运行文件、5,941 个历史批次、路由/标记清理与公开回执逐项一致。42 个相对文件链接有效；报告经 GFM 渲染检查，表格可读且无横向溢出，公开材料未发现原始请求 ID、Pod UID 或凭据泄露。

本次是限定发布验收，没有重跑全部 provider/endpoint、Azure 独立链路、真实客户端长会话、生产负载或节点故障演练。生产 Dashboard 的 OAuth 重定向与运行 HTML 已核对，没有冒充已完成浏览器登录后的视觉验收；界面渲染证据为此前的合成预览。

完整计划、私有快照、配置/Secret 备份、Pod/Lease 身份、请求 ID、账本回读及原始观测保留在本机本次 `.release-private/` 目录，不上传 GitHub。此回执提交只更新文档，不改变已发布的 `ccb25cd` 应用运行文件。
