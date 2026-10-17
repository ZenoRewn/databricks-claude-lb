# Claude 三路径接入回执（ADB / GHCP / Foundry）

Author: Zeno Ren

> 范围：本地源码与文档，**未发布 AKS**，未改生产配置，未启用任何账户。
> 实测证据见同目录 [EVIDENCE.md](EVIDENCE.md)。

## 结果

`/v1/messages` 现在由三条都说 Anthropic Messages 协议的路径服务，默认顺序
**ADB → GHCP → Foundry**（运维指定），Dashboard 可按模型覆盖。

| 路径 | 端点 | 鉴权 | 端到端实测 |
|---|---|---|---|
| ADB | `{api_base}/anthropic/v1/messages` | Bearer | 生产在跑 |
| **GHCP** | `{session_base_url}/v1/messages` | 会话 token + editor 头部 | ✅ 走完整网关代码 |
| **Foundry** | `{api_base}/anthropic/v1/messages` | **`x-api-key`** | ✅ 走完整网关代码 |

两次端到端都不是裸 curl，而是构造真实 endpoint/LoadBalancer/Proxy 后调
`proxy_request()`，验到 `type=message`、`stop_reason=end_turn`、用量入账与端点计数。

## 不需要协议合成器 —— 原计划删掉了最高风险那一项

最初我判断「GHCP 没有 Anthropic 端点，`/v1/messages` 的三路径需要双向适配器」。
GHCP 自己的 `/models` 对**每个** claude 条目声明
`supported_endpoints: ["/v1/messages", "/chat/completions"]`，Foundry 则有原生
`/anthropic/v1/*` 面。两条推断都是我从文档缺失反推出来的，被上游的能力自述推翻。

## 实现：差异收在一个钩子里

`ClaudeProxy` 本来就是「Anthropic Messages 上游池」，所以三家复用它的全套机制
（熔断、用量记账、流语义）。新增 `async _upstream_target(endpoint)` 把差异收口：

- Databricks / Foundry：路径带 `/anthropic` 前缀，凭据静态
- Copilot：路径**不带**前缀，凭据是需异步交换且会过期的会话 token，且必须带 editor
  头部（缺了 GitHub 回 403 "approved clients"）—— 这就是钩子必须 async 的原因

`CopilotMessagesProxy` 与 `CopilotProxy` **共用同一个 load balancer 与 httpx client**：
熔断状态、池上限、h2 连接退休是同一批对象，是一个上游的两个视图而非两个上游。

跨 provider 的模型改写归 `claude_model_identity`（同一模型三种拼法），编排器
`_serve_messages_across_providers` 逐跳改写并给每个 provider **自己的 body 副本**。

## 回落判据刻意保守

只在「该 provider 根本没有候选」（503 `lb_unavailable` / 404 `unsupported_model`）时
换下一家：

- 上游产生的 **4xx 是答案不是故障** —— 换一家只会把一个诚实错误变成三个
- **5xx 意味上游已被触达、执行状态不明** —— 正是不重放不变量要护的
- 流一旦交付下游，编排器**根本看不到**后续失败。有意如此：字节可能已发出时没有
  安全的换家方式

跨 provider 回落在这个面上成立，是因为 Anthropic Messages 把整段对话放在
`messages` 里，没有 `previous_response_id` 这类会被搁死的 opaque state（那是 OpenAI
风格面的问题）。prompt cache 断点不跨跳存活，损失的是钱不是正确性。

## 实测抓到的四个真问题

| # | 问题 | 怎么发现的 |
|---|---|---|
| 1 | **凭据出现在 `repr()` 里** —— `CopilotEndpoint.github_token` 与 `WorkspaceEndpoint.token` 都没设 `repr=False`，任何打印 endpoint 的日志或异常回溯都会泄漏明文 | 一次只读诊断打印 endpoint 时实测泄漏 |
| 2 | **共享规整污染另外两家** —— `_normalize_for_databricks` 第一步是 `get_databricks_model()`，把 Copilot 的 `claude-opus-5.5` 改写成 Databricks 拼法，随后匹配不到端点，表现为 404 `unsupported_model`，而那个账户明明服务该模型 | 真实调用 GHCP 时 404 |
| 3 | **Foundry 与 GHCP 都拒绝 `context_management`** —— 我原以为 Anthropic 原生面不需要 Databricks 那套删减 | 两家各回一次 `400 … Extra inputs are not permitted` |
| 4 | **Foundry 的 `api-key` 头不可用** —— 官方文档列了它，实测 401；只有 `x-api-key` 与 Bearer 可用 | 组合探测 |

第 1 条的泄漏发生在我的诊断输出里（**相关 GHCP 账户凭据应轮换**），不是代码在生产中
泄漏过。两处已改为 `field(repr=False)` 并加三个 provider 各一条的回归测试。

## 看板：按协议面显示路径

工作台此前只显示 OpenAI 风格那一份顺序，于是 **ADB 根本不出现** —— 运维无法区分
「有意不在此面」和「漏了」。**缺席本身就是那个问题。** 现在：

- OpenAI 风格入口那一段明确标注是哪个面
- 新增 `/v1/messages` 面，**三条路径全部列出**，服务不了的标记 `✕` 并说明原因
  （闭集：`provider_not_configured` / `no_endpoint_permits_model`）
- 每条芯片下方显示**该渠道实际收到的模型名** —— 配 Foundry 的 deployments 必须用
  Foundry 的短横拼法，这是最容易配错的一处

## 计价：第三套口径，宁可标未知

`MODEL_PRICING` 按模型名子串匹配，Foundry 的 `claude-opus-5-5` 会命中 `opus-5-5` 被
套上 Anthropic 公开价。Foundry 按 Azure Marketplace 的 CCU 计量，那个数字会是
**自信的错数字**，而 0 会在面板汇总成「免费」。`_pricing_for_model(model, provider=)`
对 `azure_anthropic` 返回 `None`。GHCP 的模型条目还自带
`warning_message: "… usage-based billing and model multipliers no longer apply"`，
说明现有分档口径可能已失效 —— 本轮未动 copilot 计价，只记下这条。

## 仍未完成 / 需要你的决定

1. **启用一个 business GHCP 账户** —— 已启用的 `ZenoRewn` 是 enterprise seat、**0 个
   claude**；四个闲置的 business seat 各 10 个。GHCP 这条路径在启用前不会接流量。
   启用是写生产策略，本轮未动。
2. **Foundry 的生产配置** —— `config.yaml.example` 已给出 `azure_anthropic` 段的写法
   与四个实测存在的 deployment；要在 AKS 生效需要加配置并滚动重启。
3. **GHCP 计价口径复核** —— 那条 usage-based 警告意味着分档乘数可能已不适用。
4. **未发布 AKS。**

## 验证

```
python3 -m pytest tests/ -q      → 1400 passed, 6 skipped, 1362 subtests
五个 Node UI 套件                 → 全过（routing 31 项）
pyflakes（一方代码）              → 零告警
```

新增 `tests/test_claude_model_identity.py`（15）、`test_messages_surface_routing.py`（17）、
`test_messages_provider_failover.py`（32）、`test_messages_path_views.py`（10），
以及 routing UI 的 4 条看板断言。
