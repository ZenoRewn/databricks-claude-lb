# Dashboard、运行版本与仓库整理

Author: Zeno Ren

## 范围与结果

本轮整理说明文档、替换 Dashboard 界面，并新增公开只读 `/version`。保留原生 HTML/JavaScript 与 FastAPI；没有新增前端构建依赖，也没有更改推理重放、账户亲和或用量存储协议。

README 从逐次追加记录收敛为项目入口、快速开始、Dashboard、版本与发布边界。配置、开发及历史证据分别进入独立指南与索引；CLAUDE 保留模块入口和完整 API 表。根目录过时 `ANALYSIS.md` 从当前树移除，但 [文档索引](../../README.md) 保留不可变历史链接。旧截图被本轮合成预览替换。

Dashboard 使用语义颜色、稳定布局、深浅主题、四个渠道/历史视图、手动与暂停刷新、失败提示及明确统计口径。未知费用不显示为完整零费用，缺失熔断状态不显示 Closed，业务连接租约不冒充底层 socket 利用率。图表无法加载时仍可读取表格。

Release 面板显示实际构建 SHA 和 GitHub commit/compare 链接，区分完整匹配、文件变化和来源未知。`/version` 只返回五个公开字段，Dashboard 与接口都禁止缓存。匹配状态只证明完整运行文件与构建记录一致，不能证明 GitHub main 没有后续提交或所有副本已升级。

历史清理使用一次性的 `x-api-key`，保留入口代理 Authorization；验证天数、显式确认、拒绝重定向、检查响应并清空密钥。未知结果不自动重发。历史竞态响应不会覆盖新查询，重新建立 canvas 时释放旧 Chart。

## 验证

| 层级 | 证据与结论 |
|---|---|
| 完整本地回归 | [host-full.txt](host-full.txt)：735 passed、6 skipped、722 subtests；跳过的外部环境测试不计为通过 |
| API 与构建身份 | [targeted-python.txt](targeted-python.txt)：17 passed、40 subtests，覆盖版本字段白名单、非法 SHA、未知/修改来源、无缓存与接口契约 |
| 前端行为 | [node-ui.txt](node-ui.txt)、[node-pricing.txt](node-pricing.txt)：鉴权清理、无重放、特殊模型键、未知价格与错误状态 |
| 桌面浏览器 | [browser-validation.json](browser-validation.json)：四视图、深浅主题、历史范围、初次失败、版本状态、空状态及图表降级；均为合成数据 |
| 独立 QA | [封板通过](QA.md)：独立重跑 Node、核对截图/链接/镜像指纹及实际缓存清理，无剩余阻断 |
| linux/amd64 镜像 | [image-tests.txt](image-tests.txt)：5 tests OK；[image-validation.json](image-validation.json)：21 个运行文件 hash 全部与提交源码一致，实际 `/version` 为 matched，HTML/API 均 no-store |

`*-red.txt` 保存新增行为和 QA 发现的失败证据。最终首次失败状态另经浏览器复现并修复；旧数据保留与首次无数据分别测试。本地路径在发布证据中统一替换为 `<repository>`。

预览：[浅色 Dashboard](../../../pictures/dashboard.jpg)、[深色 Release 面板](../../../pictures/dashboard-release.jpg)。图内标明“本地预览 · 合成数据”；示例 SHA 不代表真实发布。

## 清理与保留

保留回归实际引用的 `tests/fixtures/revision2`、bootstrap 所用 `operations/release/legacy-892e397.json`、历史故障/回滚/身份材料及离线 wheelhouse。它们仍有依赖或恢复用途，不按年龄删除。

只清理可再生 Python/pytest 缓存；具体目录、文件数和字节数见 [清理回执](cleanup.json)。Git 历史保留已移除文档与旧图，可从基线 `6037755c0d5686de09fb51275e4057ca1711a5c8` 恢复。用户未跟踪文件、私有恢复目录和外部 OpenClaw 资产不纳入本轮清理或公开提交。

目标镜像绑定源码 `9a61e264513ddc84db61a2ee4aeeda1fea648288`，构建时运行源码无本地修改；后续回执提交不改变运行文件。记录中的 local image ID 不是 registry digest；该镜像仅在本地使用，未推送 registry。

## 交付边界

本报告描述本地开发和验证；GitHub 合并状态以对应 PR 与提交检查为准。本轮未修改 AKS，目标镜像未推送 registry，也未调用生产数据库或真实上游；没有进行真实历史删除或生产负载测试。

上一轮 [AKS 发布](../2026-09-30-aks-release/REPORT.md) 的应用源码为 `1d75685`。新增 Dashboard 与 `/version` 只有在后续独立发布并验收后才会出现在生产，不能把本轮 GitHub 提交当作已部署版本。
