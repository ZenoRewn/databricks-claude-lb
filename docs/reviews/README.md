# 验证与发布索引

Author: Zeno Ren

回执按时点保存，保留失败、修复、镜像身份和未验证边界。源码验证、GitHub 合并、生产部署与长期运行是不同证据，不能互相替代。

| 日期 | 范围 | 入口 |
|---|---|---|
| 2026-10-09 | 流协议取证与分层保护 AKS 发布；原故障未复现、未归因 | [部署回执](2026-10-09-stream-aks/REPORT.md)、[本地验证](2026-10-09-stream-reliability/REPORT.md) |
| 2026-10-08 | 图片像素修复 AKS 发布；额外大图上游 503 边界 | [部署回执](2026-10-08-image-budget-aks/REPORT.md)、[本地验证](2026-10-08-image-budget/REPORT.md) |
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
