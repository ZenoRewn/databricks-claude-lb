# 2026-10-18 AKS 发布回执：规则按模型身份匹配 + Messages 面第二跳

Author: Zeno Ren

源码 `3464a78`，线上此前 `20b89c6`。本次修掉的是
[上一次发布回执](../2026-10-18-messages-order-release/REPORT.md) 边界里列明的两条
——「每个下拉只指定首位」和「规则按模型名精确匹配」。后者在排查时才看清不只是体验问题：
它让 **GHCP 账户限定在 `/v1/messages` 上静默失效**。

| | |
|---|---|
| 源码 | `3464a78da624c7ebcbd9f4f841c44f63383e6355`（PR #56） |
| 镜像 | `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:a7a45d6e…`（本地 image ID `c4aedf53…`，两者是不同证据） |
| source manifest | `3879d94e23bbc02090c04cc7832d5eeddc1ae1c2a012dea4ffbd7529690a16d3`（24 个运行文件，清单未变） |
| 线上此前 | `20b89c6`，镜像 `sha256:d7b31e57…` |
| 线上现在 | `3464a78` / `matched`，Pod `claude-lb-75c7f556b5-5tbc6`，1/1、**0 重启** |
| 维护窗口 | **62.6 秒**，11 个阶段一次通过（未回滚、未重试） |
| 业务探针 | messages / responses / chat 三协议均 `completed` + `marker_found`，`persistence_verified: true`，3 条用量事件入账 |
| 清理 | `verified: true`；`restore_routes.verified: true` |

**无前置**：运行文件清单未变（仍 24 个、同一组文件），控制器不需要重建；无新表、无 DDL、
无配置变更。只有一次滚动重启。

镜像构建走 `dependencies-offline` + `.wheelhouse`：`dependencies-online` 阶段的
`pip install` 仍在本机以 `SSL: UNEXPECTED_EOF_WHILE_READING` 失败（同机 curl 正常），
与前几次相同，已有的离线路径就是为此存在。

## 一、per-model 规则按模型身份匹配，不按拼法

同一个 Claude 模型有三种拼法，`known_models` 里是三个条目，但规则命名的是**模型**。

这修掉的不只是「配了没用」。`/v1/messages` 的失败转移在调用池**之前**就把模型改写成各
provider 的拼法（`claude_model_identity.rewrite`），所以 `supports()` 看到的是 GHCP 拼法
`claude-opus-5.5`。按 ADB 或 Foundry 拼法保存的 `copilot_accounts` 因此永远查不到那个 key
—— **这一层静默地不再做减法**。运维会看到一个「已限定」的规则，实际上所有账户照常参与候选。

收口在 `key_policy.rule_key()`：

- **精确拼法优先** —— 最不意外，也让身份匹配之前写的规则继续按原样工作；
- 否则按 `claude_model_identity.canonical()` 归并；
- **非 Claude 模型不归并**。没有任何依据说 `gpt-5.4` 与 `gpt-5-4` 是同一个模型，凭空归并
  等于施加一条没人写过的路由规则；
- 写入会**收敛**其他拼法下的同一条规则，所以「一个模型一条规则」是**存储不变量**，而不只是
  读时启发 —— 否则 Dashboard 会把同一个模型列两次、给两个互相覆盖的答案。

顺序是**先写新值、再删兄弟键**：失败只会多留一个可确定性解析（排序取首）的陈旧拼法，下一次
写入收敛；反过来先删后写，一旦写失败就把运维的规则悄悄打回默认顺序。

### 诊断侧同一处病根

capability 快照存的是上游返回的 id。`_model_routing_view` 与
`admin_model_routing_candidates` 过去用调用方的原始拼法去问 `supports()`，而
`_messages_path_views` 早就按 `forwarded` 去问 —— 于是**同一个响应会既说 Copilot 这一跳能
服务、又说没有账户能服务它**。现在两处都经 `_copilot_model_id()`。

该改写**只用于诊断**，不下推进 `_copilot_support_reason`：OpenAI 风格面原样转发调用方的
模型 id（只有 `_serve_messages_across_providers` 改写），在候选收口里改写会让 `supports()`
承诺一个请求体并不会使用的拼法 —— 把一次此处的诚实拒绝换成上游的 404。

## 二、Messages 面可指定第二跳

三个 provider 里定下前两位即确定全序，所以两个下拉就是全部表达力，不需要排序控件。
`Foundry Claude → GHCP → ADB` 从此可表达。第二跳的选项按第一跳**动态**生成并排除它本身
（重复项会被 `ordered_providers` 拒 400，那是一个本可以不发生的失败）；不选第二跳就回到
「只写首位、其余按默认序补齐」，写入的仍然只有一个 provider。

OpenAI 风格面只有两个 provider，指定首位即完整排序，不需要第二个控件。

## 真实浏览器又抓到两个 fixture 没覆盖的问题

用 `tests/dashboard_preview.py` 走完整路径时发现的，都不是单元测试能看见的那一类：

1. **重建 `<option>` 会把 select 重置到第一项。** 浏览器丢弃 option 节点而**不按值重匹配**
   —— 已在 Chrome 实测：对一个 `value='copilot'` 的 select 重写 innerHTML（新列表里仍有
   `copilot`），`value` 仍变成 `''`。第二跳的选项按第一跳动态重建，所以每次渲染都会清掉它，
   而 `renderRulePreview` 下一次会把清空后的 DOM 读回草稿 —— **预览对、提交错**：
   确认表显示 `Foundry Claude → ADB → GHCP`，而草稿本应是 `→ GHCP → ADB`。
   DOM 存根原先用普通字符串属性模拟 `innerHTML`，看不到这件事；现在照实模拟，该 bug 会在
   存根里红（已验证：修复前 3 条失败）。
2. **保存前的确认表只列 OpenAI 面的顺序。** Messages 面的改动在运维唯一会逐行核对的地方
   完全不可见 —— 而那正是第二个下拉的全部意义。

另：工作台两处「Messages 仍只走 ADB」自三路径上线起即为错，而且 `test_dashboard_ui.cjs`
有一条断言**在钉这个错说法**。一个有测试守护的错误陈述比没有测试更糟，已改为反向断言。

## 线上实测

`/version` → `3464a78` / `verification: matched`；`/health/{live,ready,accepting}` 全 200。
`/admin/model-routing` 的 `note` 含身份匹配说明，`default_messages_order` 仍是
`[databricks, copilot, azure_anthropic]`。

按 **ADB 拼法**写入 `providers: [azure_anthropic, copilot]` 后，三种拼法读回一致：

| 查询的拼法 | `effective_messages_order` | `effective_openai_order` |
|---|---|---|
| `databricks-claude-opus-5-5` | `azure_anthropic → copilot → databricks` | `copilot → azure_openai` |
| `claude-opus-5.5` | 同上 | 同上 |
| `claude-opus-5-5` | 同上 | 同上 |

第二跳生效（GHCP 在第二位，不是默认的 ADB），另一个协议面停在默认序（**无串扰**）。
接着按**另一种拼法**重写，规则列表仍只有一条（`['claude-opus-5.5']`，不是两条）；再按
**第三种拼法** DELETE，整条清除、顺序回到 `databricks → copilot → azure_anthropic`。
验证后未留任何临时配置。

`image_preflight` 回执为 `{"new": "verified", "old": "verified"}` —— 前一轮
[那个修复](../../RELEASE_TOOL.md) 的 receipt 短路在生产可见地生效了。

## 验证

发布前：`python3 -m pytest tests/ -q` → **1471 passed, 6 skipped, 1387 subtests**；
四个前端套件全过（`test_dashboard_routing_ui` **48 passed**）。
新增 `tests/test_rule_identity.py`（18 条 + 18 subtests），**有效性已验证**：改前 19 条红。
`plan` 阶段 `production_changed: false`、`server_dry_run_verified: true`。

## 边界

- 偏好仍**只重排不筛除**，三条路径都保留为回落；provider 维度的筛除仍是 per-key allowlist
  的职责
- 身份归并**不改可用性判断**：capability 仍是上游只读事实，`allowed_models` 仍是运维意图，
  Foundry 的 deployment 仍是运维声明
- **迁移前写入的两种拼法冲突**按排序确定性解析（精确优先），下一次写入收敛。生产当时无任何
  已存规则，所以本次没有需要收敛的历史数据
- 第二跳只加在 Messages 面。OpenAI 面两个 provider 不需要，但这也意味着**将来**若该面增加
  第三个 provider，同一处要补
- 进程内累计计数器随本次重启归零（请求数、错误数、token 累计等），这是预期的
- 那两个前端缺陷的验证来自本地预览与回归测试，**不是生产复现**
