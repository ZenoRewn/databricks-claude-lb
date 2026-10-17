# Dashboard「密钥与渠道」UI/UX 实施与验收

Author: Zeno Ren  
日期：2026-10-17  
GitHub 集成基线：`a2007adef81c14efeb8f2bf6c6b7e5b0424f4c87`  
设计与初次本地验收基线：`eaa44b8d0d4e13c91b48c29f5f18ad55cd85edac`

用户已批准设计，并明确选择推荐的「路由工作台」：左侧模型列表，右侧单模型编辑与诊断。本次交付范围为基于 `a2007ad` 整合 Dashboard、测试和文档，并通过 GitHub PR 与 CI 交付。没有调用真实管理写入或上游推理，也不包含生产发布。

`a2007ad` 中的 `main.py`、`safe_diagnostics.py`、`tests/test_h2_connection_lifecycle.py` 及 H2 可靠性文档保持原样。本次只改变 Dashboard、对应的前端回归与验证入口；旧的私有事故目录和本地 `AGENTS.md` 不纳入提交。

## 实施结果

| TAB | 结果 |
|---|---|
| 客户端密钥 | 只读列表先呈现租户、掩码与指纹、渠道和本进程使用信息；按需打开详情编辑渠道；恢复配置基线与源头轮换分层。 |
| 路由规则 | 可搜索的模型列表与详情工作台；目录模型保留默认行为，支持手输模型名；明确保持当前顺序或显式调整；指定/不限定账户使用独立模式；全局启用只读；逐账户说明首个排除原因；保存前核对差异、入口与作用范围。 |
| GHCP 账户 | 启用、候选覆盖和能力时点列表；详情编辑全局启用和允许模型；指定模型空集被拒绝，不静默扩大范围；能力快照与配置候选均不当成实际流量或推理认证。 |
| 上游凭据 | 新增入口清楚可见；按 Provider 显示相关字段，切换时清空秘密，隐藏字段不跨 Provider 提交；校验结果与运行加载状态分列。 |
| 面板密码 | 来源与表单聚焦；技术说明折叠；危险回退独立；会话失效与当前密码错误分别处理；秘密在完成、拒绝和取消时清空。 |

页面沿用现有 Dashboard 主题、管理 API 和 Cookie 会话，不引入前端依赖或新的运行资产。

## 关键行为与安全边界

- `providers` 省略表示保持当前值；预览沿用已读取顺序，不伪装成恢复默认。模型账户 `[]` 仍表示不限定，但“仅指定账户”模式下空选择无效，不能悄悄扩大范围。
- 模型页不写账户全局启用。整条恢复明确同时清除渠道偏好与账户限定，并撤销 Claude 模型在 OpenAI 风格入口的显式放行；Messages 始终保持原 ADB 路径。
- 旧的 `databricks` 单独偏好不被误判为已开放 Claude；首次显式 GHCP/Foundry 放行及空候选、整条恢复都需确认具体影响。
- 读取失败、模型不符、旧请求晚返回不能授权保存或覆盖新模型诊断。草稿与已保存配置分开，未保存离开需要明确决定；账户矩阵重绘保留键盘焦点。
- 管理写入只发送一次。ACK、读回失败和部分应用分别呈现；未知结果只提供读取核对，不自动重试写入，不声称原配置一定没变。候选为空不承诺 Foundry 一定可用。
- 会话撤销推进 epoch，清除身份缓存、详情、设备授权句柄和秘密输入；迟到读取或写入回调不能回填。锁定立即隐藏内容；DELETE 未确认时保持本地锁定并提供可见重试；锁定请求未结束前不允许并发解锁。

本次未修改后端推理重放、Provider 执行、schema、存储事务或部署。设计评估中发现的 Foundry 优先 5xx fallback 与可靠性文档冲突仍是独立后端议题；本 UI 不承诺任意失败自动切换安全或成功。

## 验证结果

| 验证 | 结果 |
|---|---|
| Python 全量回归 | 基于 `a2007ad` 的 `python3 -m pytest tests/ -q --tb=short`：1326 passed，1305 subtests passed，6 skipped。 |
| 路由 UI | `node tests/test_dashboard_routing_ui.cjs`：27/27 passed；包括保持顺序、指定空集、诊断错配/失败/乱序、读回/部分应用、风险确认、ADB-only 旧偏好、转义、锁定与键盘焦点。 |
| 会话 UI | `node tests/test_dashboard_session_ui.cjs`：18/18 passed；包括清除身份、迟到响应、401、秘密清理、锁定 ACK 丢失、并发解锁隔离和重试入口。 |
| 既有与四 TAB UI | `test_dashboard_ui.cjs`、`test_dashboard_panels_ui.cjs`、`test_copilot_pricing_ui.cjs` 均通过。原有安全断言保留，旧路由状态覆盖迁入更完整的独立套件。 |
| 静态检查 | JavaScript 语法、重复 ID、label 关联、`git diff --check` 通过。新增套件已接入既有 CI 和开发文档入口。 |
| 桌面浏览器 | 正式页面使用本地合成 HTTP fixture：检查五 TAB、路由保存后读回、空集防护、键盘勾选、详情表单、按渠道字段；深/浅色与 1024px 桌面布局已检查，1024px 页面宽度无横向溢出，控制台 error 为空。 |
| 独立 QA | 独立代理完成源码/契约/测试复核，所报问题修复后重新检查；无剩余阻塞发现。 |

6 项跳过均来自 `tests/integration/test_mysql_usage.py`，原因是未配置隔离的 loopback MySQL。本次没有 schema 或持久化逻辑修改；这些跳过不能视为真实 MySQL 验收通过。

行为修改先取得失败测试，再实现；视觉低影响调整采用结构和渲染检查。已完成的 Python 全量证据没有因后续纯前端小修而无意义重复，后续新增/修改的前端行为均补测相应 Node 套件。

本地截图覆盖五 TAB，文件名为 `implemented-routing-dark.png`、`implemented-routing-light.png`、`implemented-routing-1024.png`、`implemented-clientkeys.png`、`implemented-accounts.png`、`implemented-credentials.png`、`implemented-password.png`。所有截图和浏览器操作均为合成数据，不能替代真实 API、模型生成、数据库持久化或生产验收。
