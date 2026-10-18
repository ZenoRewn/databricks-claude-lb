# 2026-10-18 AKS 发布回执：Messages 面优先序可设 + Foundry 进统计口径

Author: Zeno Ren

源码 `3afb3ac`，线上此前 `0a9f830`。本次让 `/v1/messages` 的优先序**可以逐模型设定**，
并补上前一次发布留下的那个缺口：Foundry 进了候选集却不在 `/stats` 与 `/metrics` 里。

| | |
|---|---|
| 源码 | `3afb3aceda8967740d4992253eb63c7136cf059f`（PR #52） |
| 镜像 | `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:5286859d…`（本地 image ID `493d4d02…`，两者是不同证据） |
| source manifest | `fbb4a4ccb320cf07cbd8639e1a61a31b19f8856efbda8bbbebe6c8e7ba69e625`（24 个运行文件，与上次相同） |
| 线上此前 | `0a9f830`，generation 140，镜像 `sha256:173cba6b…` |
| 线上现在 | `3afb3ac` / `matched`，generation 144 |
| 维护窗口 | **58.0 秒**，11 个阶段全过，一次通过（未回滚、未重试） |
| 业务探针 | messages / responses / chat 三协议均 `completed` + `marker_found`，`persistence_verified: true`，3 条用量事件入账 |
| 清理 | `verified: true`，`cleanup_errors: []` |

**本次没有前置**：运行文件清单未变（仍 24 个），所以
[上一次](../2026-10-17-three-paths-release/REPORT.md) 必须做的控制器重建这次不需要
—— 运行中的两个控制器副本自检均为 `22 files`，与本地 CLI 口径一致。无新表、无 DDL、
无配置变更。只有一次滚动重启。

## 修的是什么

`claude-opus-5-5` 选了「Foundry 优先」，下方 Anthropic Messages 面却仍显示
ADB → GHCP → Foundry Claude。**运维的判断是对的：当时确实无法设定 messages 面的顺序。**

根因不在后端 —— `ordered_providers()` 一直接受全部四个 provider 并保序。问题全在工作台：

| 现象 | 根因 |
|---|---|
| 「Foundry 优先」对 Claude 毫无作用 | 顺序下拉的选项只来自 `default_openai_order`，所以其中唯一的「Foundry」是 **`azure_openai`**（服务 gpt 的那一面），而 `_messages_provider_order()` 会把它过滤掉 |
| 想选的 `azure_anthropic` 不在下拉里 | 后端没有暴露 `default_messages_order` |
| 即使构造出来也存不进去 | `routeCanSave()` 把两个面的选择都拿 `defaultOrder` 校验 |
| 选完看不出变化 | `renderMessagesPaths()` 读已保存的 baseline，不读草稿 |
| 文案是错的 | 旧文案称这一段「只调整 GHCP / Foundry」，而同一份偏好也排 messages 面 |

一个没有选项来源的协议面，就是配不了的协议面。

## 实测抓到三个缺陷，fixture 都没覆盖

用 `tests/dashboard_preview.py` 起本地实例、在真实浏览器里走完整路径
（解锁 → 选模型 → 改顺序 → 预览 → 保存 → 回读）时发现的。它们都不是本次新写的代码才有
的问题类型，而是**只有真实 DOM 与真实往返才会暴露**的那一类：

1. **只编辑一个面会把另一个面的已保存选择整份擦掉。** `providers` 是整份替换，而最初
   实现只发被编辑的那一段。原 fixture 里那个模型的已保存值恰好等于该面默认首位
   （`['copilot']`），所以单元测试一路绿。修法：编辑一个面时带上另一面**已保存**的选择。
2. **两个 tab 的 KPI 共用 `kpi-azure` 前缀。** 从 Azure tab 复制渲染代码时沿用了它，于是
   `getElementById` 命中 Azure 面板的节点 —— Foundry 页永远显示「—」，同时**悄悄覆盖**
   Azure 的数字。已加机械核验：所有 KPI 网格的 id 前缀必须互不相同。
3. **`Deployments` 列为空。** `LoadBalancer.get_stats()` 只在端点有 `deployments` 字段时
   输出它，而 Foundry 的 `WorkspaceEndpoint` 用的是 `models`。Foundry 没有 Models API，
   这份运维声明的列表是唯一的「它能服务什么」，空列会被读成「什么都不服务」。

另外修了一个自己造的回归：既有契约是
**「keep means omit providers, never reset them」**（两面都「保持当前顺序」就省略
`providers`），一度被破坏，由既有测试抓回。

## 两段如何合成一份偏好

两个面读**同一份**存储的偏好，`copilot` 是唯一共享的 provider，所以它是锚点：把 OpenAI
段按它与 `copilot` 的相对位置插进 messages 段，得到一份两面各自保留首位的全序。

镜像内自检逐一钉住（31 项全过），包括锚点换位的两种方向：

| 存储的偏好 | messages 面首位 | OpenAI 面首位 |
|---|---|---|
| `[azure_anthropic, databricks, copilot, azure_openai]` | `azure_anthropic` | `copilot` |
| `[azure_openai, copilot, databricks, azure_anthropic]` | `copilot` | `azure_openai` |

## 上一轮的缺口已填

| | 上一轮 | 现在 |
|---|---|---|
| `/stats` 顶层 | 无 `azure_anthropic` | **有**，带 `pricing.status = "unknown"` / `basis = "ccu_not_per_token"` |
| `/metrics` `circuit_open` 样本数 | **18**（9 ADB + 4 Azure + 5 Copilot） | **19**（多出 `eastus2-foundry`） |
| Foundry 指标族 | 0 | 15 个样本，含恢复可见性（`half_open_in_flight` / `retry_after_seconds` / `trial_in_flight_age_seconds` 等） |
| 看板 | 无入口 | 第六个 tab「Foundry Claude」 |

**刻意不计价。** `MODEL_PRICING` 按模型名子串匹配，Foundry 的 `claude-opus-5-5` 会命中
`opus-5-5` 从而被套上 Anthropic 公开 per-token 价。线上实测该段
`estimated_cost_usd` **不存在**、不进 Anthropic 费用合计，页面也不渲染费用 KPI —— 但写明
了 CCU 口径，否则「没有费用」会被读成「忘了实现」。

## 线上实测

- `/version` → `3afb3ac` / `verification: matched`；`/health/{live,ready,accepting}` 全 200
- Pod `claude-lb-54d4cb55df-jskw2`，1/1 Ready，**0 重启**，generation 144
- `/admin/model-routing` 两份选项都在：`default_openai_order = [copilot, azure_openai]`、
  `default_messages_order = [databricks, copilot, azure_anthropic]`，`provider_labels` 含
  `azure_anthropic: Foundry Claude`
- `/stats` 的 `azure_anthropic` 段：`pricing.status = unknown`，`eastus2-foundry` 带 4 个
  deployment（`claude-opus-5-5` / `claude-opus-5` / `claude-sonnet-5-5` / `claude-haiku-5-5`）
- Foundry 端点照常加载；`copilot_h2_stream_id_watermark` → **7**（全新连接，离 400 很远）；
  h2 退休仍无样本；19 个 `circuit_open` 样本**全部为 0**

## 验证

发布前：`1421 passed, 6 skipped, 1366 subtests`；四个前端测试全过
（`test_dashboard_routing_ui` **39 passed**）。镜像内自检 **31 项全过**。
`plan` 阶段 `production_changed: false`、`server_dry_run_verified: true`；两个入口、
preStop 45 秒、三个探针与 90 秒 grace 全部保留。

## 边界

- **每个下拉只指定首位**，其余按该面默认序补齐。`Foundry Claude → GHCP → ADB`
  （第二位也换）表达不了 —— 要它需要换成完整排序控件，是另一轮
- **规则按模型名精确匹配**：给 `claude-opus-5-5` 配的偏好**不会**应用到
  `databricks-claude-opus-5-5` 的请求，而后者才是 `/v1/models` 列出、客户端实际发送的
  名字。这是既有语义（per-key allowlist、GHCP 账户选择同样按模型名），本轮未改。
  表现恰好和最初报告的「设了没用」一样，排查时要先确认模型名对得上
- **Foundry 端到端仍未在生产验证过。** 本轮让它**可以**被排到第一跳；真打过去的证据要等
  在 Dashboard 上配完并发一次真实请求
- **GHCP 那一跳仍是通配乐观**：`ZenoRewn` 的 `capability_known: false`，实测 0 个 claude。
  把 Foundry 排第一恰好绕开它，但它本身要靠 `discover` 或启用 business 账户解决
- **进程内累计计数器随本次重启归零**（请求数、错误数、token 累计等），这是预期的
- 前端那三个缺陷的验证来自本地预览与回归测试，**不是生产复现**
