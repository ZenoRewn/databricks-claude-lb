# 2026-10-17 AKS 发布回执：Dashboard 工作台 + h2 连接退休

Author: Zeno Ren

一次发布覆盖两批已合并内容。源码 `34acc39`，线上此前 `0e42828`。

| | |
|---|---|
| 源码 | `34acc39e9c26541d56d830e1bcfeb7d36b0e9754`（PR #46 Dashboard，含 PR #45 h2 连接退休） |
| 镜像 | `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:b332e2e8…`（本地 image ID `59d1ce83…`，两者是不同证据） |
| source manifest | `b20bb0800ce4f418c3b48faf5f986a53fb01f455da71e58c39f3aa7f3b695bcb`（23 个运行文件） |
| 线上此前 | `0e42828`，generation 132，镜像 `sha256:c71661e4…` |
| 线上现在 | `34acc39` / `matched`，generation 136 |
| 维护窗口 | **66.7 秒**，11 个阶段全过，一次通过（未回滚、未重试） |
| 业务探针 | messages / responses / chat 三协议均 `completed` + `marker_found`，`persistence_verified: true`，3 条用量事件入账 |
| 清理 | `verified: true`，`cleanup_errors: []` |

阶段序列：`preflight → gating → pausing → draining → stopping → starting → verifying_backend → restoring_routes → verifying_business → finalizing → succeeded`

## 内容

- **h2 连接退休**（PR #45，见 [事件回执](../2026-10-17-ghcp-h2-connection/REPORT.md)）：REFUSED_STREAM / 首内容停滞 / stream id 上限三个触发条件，优雅退休（只拦新流，在途流照常跑完），外加 `copilot_h2_stream_id_watermark` 与 `copilot_h2_connections_retired_total{reason}` 两个指标
- **Dashboard 密钥与路由工作台**（PR #46，见 [实施与验收](../2026-10-17-dashboard-keys-channels/REVIEW.md)）

## 前置：本批没有前置

- 新增 env `COPILOT_H2_RETIRE_STREAM_ID` **有默认值 400** → 不需要改 Secret/ConfigMap，控制器 Role 不用动
- 无新表、无 DDL → 只读 schema 门控照常跑
- 无新运行时文件（仍 23 个，改动的 `dashboard.html` / `main.py` / `safe_diagnostics.py` /
  `request_telemetry.py` / `release_probe.py` / `cleanup_observability.py` 都已在清单内）
  → 三处清单不用同步
- `requirements.lock` / `Dockerfile` 自 `0e42828` 起未变 → `.wheelhouse` 复用（构建实测命中缓存）
- `operations/release/*` 确实变了，但**只是删未用导入加一行注释、零行为变更**，且不在 source-hash
  清单里 → **不需要重建发布控制器镜像**。运行中的控制器继续跑功能等价的旧代码，本地 CLI 跑新代码

所以这一次只有一次滚动重启（发布本身）。

## 发布前验证

镜像内自检 **21 项全过**：构建身份（`34acc39` / manifest `b20bb080…` / 23 文件 / tree 不脏）、
h2 退休的五个符号与默认阈值 400、退休判据三分支（code 7 触发、code 8 不触发、本地 reset 不触发）、
退休是优雅的（在途连接既未关闭也未被置过期）、内省失败降级不抛、新字段过 `safe_diagnostics`
白名单、Dashboard 非占位且以 `</html>` 收尾且 >200KB。

推送前确认镜像是**单平台 manifest**（`application/vnd.docker.distribution.manifest.v2+json`，
不是多架构 index）—— 发布工具拒绝无法直接核对的 index digest。

`plan` 阶段 `production_changed: false`、`server_dry_run_verified: true`；两个入口
（`claude-lb` / `claude-lb-internal`）、preStop、三个探针与 90 秒 grace 全部保留。

## 线上实测

- `/version` → `34acc39` / `verification: matched`；`/health/{live,ready,accepting}` 全 200
- Pod `claude-lb-858f58fb6f-6xfcm`，1/1 Ready，**0 重启**
- `copilot_h2_stream_id_watermark` → **7**。这正是事件当晚缺失的那个信号（当时已经到 683）：
  全新连接、离 400 的轮换阈值很远
- `copilot_h2_connections_retired_total` 只有 HELP/TYPE、**没有样本** —— 重启后还没有发生过退休，
  符合预期
- 五个 Copilot 端点与全部 Databricks 端点 `circuit_open=0`
- `/admin/session` 未认证只回布尔：`password_configured: true`、`password_source: "env"`
  （面板密码 overlay 未设置，回落 env 基线，符合设计）

**Dashboard 完整性不靠抓页面证明。** `/stats/dashboard` 对未认证调用返回 **302**（入口层 OAuth，
与 `/stats` 同理），所以抓到的是重定向页而不是面板 —— 我第一次比对哈希就是比错了对象。真实证据是
`/version` 的 `verification: matched`（运行文件与 source manifest 一致，`dashboard.html` 在那 23 个
里面）加上 preflight 在镜像内逐个核对的 23 个文件哈希。

## 边界

- **进程内累计计数器随本次重启归零**（请求数、错误数、token 累计等），这是预期的
- h2 连接退休**本次没有在生产触发过**：它是针对 2026-10-17 事件的修复，验证来自镜像内自检与
  27 项回归测试，**不是生产复现**。下一次 REFUSED_STREAM 才会产生第一条
  `copilot_h2_connection_retired` 事件
- GitHub 那个拒绝上限的单位（流数 / 连接寿命 / 字节数）仍未确认；`copilot_h2_stream_id_watermark`
  的历史会随时间给出答案
- 单账户故障域未动：`gpt-6.1-sol` 仍只有一个启用账户，Azure 不服务该模型
