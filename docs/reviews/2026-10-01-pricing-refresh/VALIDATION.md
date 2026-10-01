# 价格自动更新与 Logo 验证回执

Author: Zeno Ren

日期：2026-10-01。范围：本地代码、公开官方价格源、隔离测试和应用镜像；未推送 GitHub，未部署 AKS/生产。

## 已完成

- [实施计划](PLAN.md)已落实。价格基线 35 模型 / 47 档位，补齐 6 模型；应用内每 7 天刷新、失败 1 小时后重试、持久缓存和安全回退已实现。
- 后台任务接入启动与关闭；Dashboard 显示核对日期、覆盖模型数、最近成功、下次检查、刷新失败与过期状态。旧累计估算不因价格更新重算。
- Azure OpenAI `gpt-image-2.5-sunburst` 实际成功生成 6 张 PNG。用户选择 01 汇流后，透明处理并内嵌 Dashboard；完整提示词与原图见 [生成记录](../../../output/imagegen/lb-logo-20261001/README.md)。

## 自动化验证

| 验证 | 结果 | 边界 |
|---|---|---|
| 先失败回归 | 27 个价格刷新用例在实现前失败 | 缺少功能的行为复现；后续补充调度、注释漂移和生命周期用例 |
| `python3 -m pytest tests/ -q --tb=short` | 766 passed、6 skipped、722 subtests passed，43.47 秒 | Python 3.14；6 个隔离 MySQL 用例未启用 |
| `node tests/test_copilot_pricing_ui.cjs` | 通过 | 部分/未知估算、价格更新状态和脚本语法 |
| `node tests/test_dashboard_ui.cjs` | 通过 | 现有 Dashboard 交互与版本边界 |
| Logo 接入后 Dashboard/构建身份目标测试 | 9 passed、10 subtests passed | 未重复无关完整回归 |
| 真实 GitHub Docs 抓取及新缓存重载 | 通过，35 个模型、12 个长上下文档位 | 2026-10-01 04:03:11 UTC；临时缓存目录，与生产数据隔离 |
| Linux/amd64 应用镜像内价格/刷新/生命周期测试 | 47 passed、45 subtests passed | Python 3.12，挂载只读 tests，未挂载应用源码 |
| 镜像内运行文件 hash | `runtime_files_match=true`、`manifest_covers_current_runtime=true` | 本地未提交代码构建，dirty/out_of_tree 均为 true |

调度测试用模拟时钟验证首次抓取、7 天间隔、失败后 1 小时重试与恢复，不宣称已经观察了真实运行一周。回归涵盖坏 HTML、列/脚注漂移、非数字价格、缺表、长上下文不成对、促销到期、重定向、超大响应、写盘失败、损坏/未来/篡改缓存、取消、关闭开关与生命周期清理。

## 镜像与环境记录

- 本地标签：`lb-pricing-weekly-local:20261001`；平台 `linux/amd64`。
- 本地 image ID：`sha256:1fccaaa2e8204ec685f513edb5e76f58b465bf985e4efb35667c77d76407c519`，不是 registry manifest digest。
- 源码基准提交：`299927519f26804618d372e10ac0577347fd2032`，包含本轮未提交改动。
- 运行源清单 SHA-256：`e848a297840907ca51e5c095767de253ac8894fddef5919dd514dc185c64a58f`。
- 初次在线构建在 PyPI 下载遇到 TLS EOF；随后使用仓库已有锁定 wheelhouse 和 `dependencies-offline` 成功构建，未关闭 TLS 校验或改依赖版本。
- 镜像测试产生一个 pytest 缓存目录只读警告；测试用例全部通过。

## 视觉检查

使用 Playwright 在 1440px 桌面视口检查合成数据页面：浅色、深色、错误/过期提示。Logo 使用 34 × 34 显示，透明背景，无图片加载失败；价格更新信息可读并正常换行。截图为合成预览，不是生产观测。

- [浅色](../../../output/playwright/pricing-refresh/light.png)
- [深色](../../../output/playwright/pricing-refresh/dark.png)
- [刷新失败与过期](../../../output/playwright/pricing-refresh/error.png)
- [6 款 Logo 与 34px 候选预览](../../../output/imagegen/lb-logo-20261001/contact-sheet.png)

预览服务器唯一观察到的 console 错误是未提供 favicon 的 404，与本轮图片及脚本无关。

## 未验证与生效条件

未进行生产部署、生产公网业务测试、真实 Copilot 推理调用、实际账单核对或跨实例刷新一致性测试。没有改动 MySQL schema、追加用量账本、推理重试策略或生产配置。服务采用更新后的代码/镜像并启用 Copilot 后，才会运行本轮新增任务；缓存路径需要可写，跨容器重建保存需位于持久卷。
