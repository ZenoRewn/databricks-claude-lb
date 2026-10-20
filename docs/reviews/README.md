# 验证与发布索引

Author: Zeno Ren

回执按时点保存，保留失败、修复、镜像身份和未验证边界。源码验证、GitHub 合并、生产部署与长期运行是不同证据，不能互相替代。

| 日期 | 范围 | 入口 |
|---|---|---|
| 2026-10-20 | GHCP 原生 Responses 的 message ID 稳定化：默认关闭、按模型启用，保留协议终态/工具字段/用量；本地与独立 QA 已通过，真实桌面与生产另行验收 | [实现与验证](2026-10-20-ghcp-response-identity/REPORT.md) |
| 2026-10-19 | **AKS 发布** `53615db`：Messages 面的候选判断**绕过了四层账户策略** —— 生产实测一次强制走 copilot 的 `/v1/messages` 落在运维**从未启用**的账户上，而诊断同时报它被排除。`select_endpoint` 只看 config 的 `models` 列表，而 Copilot 账户声明 `models: []`，谓词恒真。同一行代码还把 Foundry/GHCP-messages 的遥测归因写死成 databricks。维护 78.5 秒（首次提交在 preflight 403 中止，原因是迁移漏了一步 RBAC，生产未被触碰）| [发布回执](2026-10-19-messages-account-policy/REPORT.md) |
| 2026-10-18 | **AKS 发布 + 运维变更** `1d00c58`：上游 17 条凭据从 ConfigMap 明文迁入 Secret。先发一道**行为无变化**的闸门 —— `${ENV}` 引用解析不出来就拒绝启动，因为此前缺失变量被静默替换成空串且没有任何一层拒绝空 token；再用校验 Pod 确认 `load_config` 退出码 0 才动生产。发布维护 55.4 秒；四条真实上游请求与 MySQL 账本读写均通过 | [迁移回执](2026-10-18-credential-secret-migration/REPORT.md) |
| 2026-10-18 | **AKS 发布** `3464a78`：per-model 规则改为按**模型身份**匹配（三种拼法共用一条）—— 这修掉的不只是「配了没用」，`/v1/messages` 的失败转移先改写模型名，所以按 ADB/Foundry 拼法保存的 GHCP 账户限定**静默地不再做减法**；同时 Messages 面可指定第二跳。维护 62.6 秒、11 阶段一次通过 | [发布回执](2026-10-18-rule-identity-release/REPORT.md) |
| 2026-10-18 | **AKS 发布** `3afb3ac`：`/v1/messages` 优先序可逐模型设定（此前工作台唯一的「Foundry」是 gpt 那一面，对 Claude 无效），并补上 Foundry 的 `/stats`、`/metrics` 与看板入口；维护 58.0 秒、11 阶段一次通过。三个缺陷由真实浏览器实测抓到，fixture 全未覆盖 | [发布回执](2026-10-18-messages-order-release/REPORT.md) |
| 2026-10-17 | **AKS 发布** `0a9f830`：Claude 三路径上线，并首次把 Foundry 凭据配进生产；维护 48.0 秒、11 阶段一次通过。本次控制器必须重建（`APP_FILES` 多一个文件，旧控制器会拒 plan），且 Foundry 不在 `/stats` 与 `/metrics` 里是已知缺口 | [发布回执](2026-10-17-three-paths-release/REPORT.md) |
| 2026-10-17 | Claude 三路径（ADB / GHCP / Foundry）**接入完成**：三家都说 Anthropic Messages，合成器不需要；看板按协议面显示路径；实测抓到凭据进 repr、共享规整污染另外两家等四个真问题 | [接入回执](2026-10-17-claude-three-paths/REPORT.md)、[实测证据](2026-10-17-claude-three-paths/EVIDENCE.md) |
| 2026-10-17 | Claude 三路径实测证据：三家都说 Anthropic Messages，合成器不需要；推翻我自己两条文档推断，并发现已启用的 enterprise 账户 0 个 claude | [实测证据](2026-10-17-claude-three-paths/EVIDENCE.md) |
| 2026-10-17 | **AKS 发布** `34acc39`：Dashboard 工作台与 h2 连接退休一次上线；维护 66.7 秒、11 阶段一次通过，三协议探针与账本验收通过 | [发布回执](2026-10-17-aks-release/REPORT.md) |
| 2026-10-17 | 基于 a2007ad 的 Dashboard「密钥与渠道」五 TAB 与路由工作台优化；包含本地/独立 QA，GitHub 交付与生产发布分别验收 | [实施与验收](2026-10-17-dashboard-keys-channels/REVIEW.md) |
| 2026-10-17 | GHCP 再次 HALF_OPEN：一条 h2 连接用到服务端不再接受新流（无 GOAWAY，池子照旧复用），连接优雅退休与 stream id 观测；排除了体积与多路复用两个假设 | [事件回执](2026-10-17-ghcp-h2-connection/REPORT.md) |
| 2026-10-16 | GHCP HALF_OPEN 事件的证据口径（事件时间、重试预算、恢复可见性、图片判定收敛）AKS 发布；三项优化候选有意未改并记明理由，首轮因发布工具把已通过的镜像预检读成失败而回滚，发布后实测纠正了我自己的一条判断 | [事件与发布回执](2026-10-16-ghcp-half-open/REPORT.md) |
| 2026-10-15 | 面板独立密码与 HttpOnly 会话 AKS 发布；两步发布，并记下一次差点造成镜像回滚的误判 | [部署回执](2026-10-15-panel-password-aks/REPORT.md) |
| 2026-10-14 | 路由规则三段式（模型 → Provider → GHCP 账户）与面板解锁 AKS 发布；密钥驻留契约变更，三处缺陷在验证中发现并修复 | [部署回执](2026-10-14-routing-rules-aks/REPORT.md) |
| 2026-10-14 | 账户先落地后启用、凭据录入与 Foundry v1 AKS 发布；首轮因控制器缺一个 Secret 读权限而在 preflight 中止 | [部署回执](2026-10-14-inert-accounts-aks/REPORT.md) |
| 2026-10-10 | Claude 5.5 模型解析修复 AKS 发布；首轮因探针选用容量受限模型而回滚 | [部署回执](2026-10-10-model-resolution-aks/REPORT.md) |
| 2026-10-10 | 协议保证落地与重放白名单首次扩展 AKS 发布 | [部署回执](2026-10-10-replay-aks/REPORT.md) |
| 2026-10-10 | 停滞信号与证据门控 AKS 发布；preflight 拒绝与账本恢复 | [部署回执](2026-10-10-stall-aks/REPORT.md) |
| 2026-10-09 | 120 秒上游取消的生产取证；两项保护的实测局限 | [流取证](2026-10-09-stream-forensics/REPORT.md) |
| 2026-10-09 | 流协议取证与分层保护 AKS 发布；原故障未复现、未归因 | [部署回执](2026-10-09-stream-aks/REPORT.md)、[本地验证](2026-10-09-stream-reliability/REPORT.md) |
| 2026-10-08 | 图片像素修复 AKS 发布；额外大图上游 503 边界 | [部署回执](2026-10-08-image-budget-aks/REPORT.md)、[本地验证](2026-10-08-image-budget/REPORT.md) |
| 2026-10-01 | 每周价格刷新与 Logo AKS 发布；首次刷新成功不等于已观察一周 | [部署回执](2026-10-01-pricing-aks/REPORT.md)、[本地验证](2026-10-01-pricing-refresh/VALIDATION.md) |
| 2026-10-01 | 历史页参考费用修复：档位不可判时给上下界，无价仍显示未知 | [只读验证](2026-10-01-history-pricing/VALIDATION.md) |
| 2026-09-30 | Dashboard、明亮配色与版本标识 AKS 发布 | [部署回执](2026-09-30-dashboard-aks/REPORT.md) |
| 2026-09-30 | 图表明亮配色与 usage 告警只读评估 | [现场证据与建议](2026-09-30-chart-usage-review/ASSESSMENT.md) |
| 2026-09-30 | Dashboard、运行版本与仓库整理 | [本轮记录](2026-09-30-dashboard/REPORT.md) |
| 2026-09-30 | AKS 生产发布 | [部署回执](2026-09-30-aks-release/REPORT.md) |
| 2026-09-30 | 发布镜像与源码注解一致性 | [工具修复](2026-09-30-release-source-identity/REPORT.md) |
| 2026-09-30 | 诊断、阶段时间、适配与能力目录 | [实现](2026-09-30-lb-contracts/IMPLEMENTATION.md)、[验证](2026-09-30-lb-contracts/VALIDATION.md)、[独立 QA](2026-09-30-lb-contracts/QA.md) |
| 2026-09-30 | OpenClaw 附件与系统性评估 | [评估](2026-09-30-openclaw-assessment/ASSESSMENT.md) |
| 2026-09-21 | SSE 分段与缓存验收回执 | [修复记录](2026-09-21-probe-sse/REPORT.md) |
| 2026-09-21 | 合并前复核 | [Review](2026-09-21-main-merge/REVIEW.md) |
| 2026-09-21 | Copilot 价格与缓存写入 | [验证数据](2026-09-21-copilot-pricing/validation.json) |
| 2026-09-20 | 发布器故障、接管和恢复 | [验证](2026-09-20-post-upgrade-hardening/VALIDATION.md)、[复现入口](2026-09-20-post-upgrade-hardening/REPRODUCE.md) |
| 2026-09-20 | 服务可靠性第一批实现 | [历史记录](2026-09-20-service-optimization/README.md) |

原始日志和 JSON 由相应报告或证据 manifest 引用，保留供回溯，不混入日常快速开始。历史模型/端点样本不证明当前供应商能力或当前生产状态。未跟踪的私有事故目录不属于可公开发布材料。
