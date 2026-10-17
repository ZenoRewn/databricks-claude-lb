# Claude 三路径（ADB / GHCP / Foundry）实测证据

Author: Zeno Ren

日期：2026-10-17。范围：**只读探测**，未改代码、未改生产配置、未启用任何账户。
每条结论都来自对上游的实际调用，不是文档推断 —— 这份文件存在的原因就是我先前的两条
文档推断都错了，见下面「推翻的两条」。

## 结论：三条路径都说 Anthropic Messages，不需要协议合成器

| 路径 | Anthropic Messages 端点 | 鉴权 | 实测 |
|---|---|---|---|
| **ADB** | Databricks `/v1/messages` | 既有 | 生产在跑 |
| **GHCP** | `api.{enterprise,business}.githubcopilot.com/v1/messages` | 会话 token | 上游 `/models` 自报 |
| **Foundry** | `{resource}.services.ai.azure.com/anthropic/v1/messages` | **`x-api-key`** | 200，返回标准 message |

## 推翻的两条（我先前判错，记下来免得重犯）

1. **「Copilot 没有 Anthropic 端点」—— 错。** 我由「Claude Code 官方 provider 列表里没有
   GitHub Copilot」推断出这条。但 GHCP 自己的 `/models` 对**每个** claude 条目声明
   `supported_endpoints: ["/v1/messages", "/chat/completions"]`。上游的能力自述比从文档
   缺失推断强得多。
2. **「Foundry 只需运维建部署，不是代码问题」—— 错，反了。** Microsoft Foundry 的 Claude
   走的是 `/anthropic/v1/*` 这条**原生 Anthropic 面**，而本仓库的 `azure_openai` provider
   说的是 Azure OpenAI 形状（deployments / gpt-*）。那是同一资源的两个不同面，网关没有
   实现 Anthropic 那一面。

## GHCP：账户 SKU 决定有没有 Claude

对五个已配置账户逐个做 token 交换 + `GET /models`（在生产 Pod 内执行，token 未外泄）：

| 账户 | base | 清单条目 | claude 条目 |
|---|---|---|---|
| `ZenoRewn`（**唯一已启用**） | `api.enterprise.githubcopilot.com` | 38 | **0** |
| `zenoren-icloud` | `api.business.githubcopilot.com` | 41 | 10 |
| `zenoren-msft` | 同上 | 41 | 10 |
| `zenoren-outlook` | 同上 | 41 | 10 |
| `zenoren-qq` | 同上 | 41 | 10 |

**现在启用的 enterprise 账户一个 Claude 都不服务；四个闲置的 business 账户各有 10 个。**
这印证了 `config.yaml.example` 里那条「不同 SKU 的账户模型集合会互缺」，只是这次缺的是
整个 Claude 家族。**所以 GHCP 这条路径要真的可用，必须启用一个 business 账户。**

claude 条目的入口声明一律是 `["/v1/messages", "/chat/completions"]` —— 注意**没有**
`/responses`，而 Codex 用的 `gpt-6.1-sol` 等恰好只有 `["/responses", "ws:/responses"]`。

GHCP 每个模型还自带可用能力（取 `claude-opus-5.5`）：

```
supports: adaptive_thinking, parallel_tool_calls, streaming, structured_outputs,
          tool_calls, vision, reasoning_effort[low|medium|high|xhigh|max]
vision mime: image/jpeg, image/png, image/webp, image/gif, application/pdf
tokenizer: o200k_base      vendor: Anthropic      policy.state: enabled
```

**计价警告（影响口径）**：条目里带
`warning_message: "Your billing plan has changed to usage-based billing and model
multipliers no longer apply."` —— 仓库的 copilot 分档乘数口径可能已不适用这些账户。

## Foundry：实测 4 个 deployment，鉴权与文档不一致

`https://eastus2-foundry-service.services.ai.azure.com`（凭据由运维提供，未入库未入日志）

| header | 结果 |
|---|---|
| **`x-api-key`** | **200** |
| `Authorization: Bearer <key>` | 200 |
| `api-key`（**官方文档说可用**） | **401** |

**所以实现必须用 `x-api-key`（或 Bearer），不能照文档用 `api-key`。**

`GET /anthropic/v1/models` → 401；与文档「Models API 不支持」一致，**能力只能由运维声明**。

deployment 盘点（`model` 参数传的是 deployment 名）：

| deployment | 结果 |
|---|---|
| `claude-opus-5-5` | ✅ |
| `claude-opus-5` | ✅ |
| `claude-sonnet-5-5` | ✅ |
| `claude-haiku-5-5` | ✅ |
| `claude-opus-4-8` / `claude-sonnet-5` / `claude-haiku-4-5` / `claude-fable-5-1` | 404 `DeploymentNotFound` |
| `claude-opus-5.5`（点号写法） | 404 —— **id 约定不同，必须映射** |

响应是标准 Anthropic Messages，外加三个 Azure 字段（实测均为 `null`）：
`container` / `stop_details` / `diagnostics`。usage 比 Anthropic 公开 API 更丰富：

```
input_tokens, cache_creation_input_tokens, cache_read_input_tokens,
cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens},
output_tokens, output_tokens_details.thinking_tokens,
service_tier, inference_geo, speed
```

追踪头：`request-id`、`apim-request-id`、`x-ratelimit-remaining-requests`（实测 39）。
**没有** Anthropic 标准的 `anthropic-ratelimit-*` 头，所以限流处理不能假设它们存在。

## 模型 id 三种约定，必须映射

| 模型 | ADB | GHCP | Foundry |
|---|---|---|---|
| Opus 5.5 | `databricks-claude-opus-5-5` | `claude-opus-5.5` | `claude-opus-5-5` |
| Opus 5 | `databricks-claude-opus-5` | `claude-opus-5` | `claude-opus-5` |
| Sonnet 5.5 | `databricks-claude-sonnet-5-5` | `claude-sonnet-5.5` | `claude-sonnet-5-5` |
| Haiku 5.5 | `databricks-claude-haiku-5-5` | `claude-haiku-5.5` | `claude-haiku-5-5` |
| Sonnet 5 | `databricks-claude-sonnet-5` | `claude-sonnet-5` | ✗ |
| Haiku 4.5 / Opus 4.8 | ✓ | ✓ | ✗ |
| Fable 5 / 5.1 | ✗ | ✓ | ✗ |

**三条路径同时可用的交集是 4 个**：Opus 5.5、Opus 5、Sonnet 5.5、Haiku 5.5。
其余模型只能做两路或单路回落 —— 候选集必须按模型逐个算，不能按 provider 整体假设。

## 由此确定的默认策略

运维指定：Claude 默认 **ADB → GHCP → Foundry**，除非 Dashboard 为某模型设定了顺序。

## 实现必须解决的问题清单

1. 新增 Anthropic-Messages provider 面（Foundry），与既有 `azure_openai`（OpenAI 面）**并存而非替换**
2. GHCP 在 `/v1/messages` 上成为候选 —— 目前 copilot 只用于 OpenAI 风格入口
3. 规范模型名 ↔ 三套 provider id 的映射，按模型算候选集
4. per-model 优先序扩展到覆盖 messages 面（含 `databricks`），默认 ADB → GHCP → Foundry
5. **第三套计价口径**（Foundry CCU）+ GHCP usage-based 警告；按既有不变量不可混用，宁可标未知
6. 能力缺口要能表达：Foundry 不支持 Models API / Agent Skills / programmatic tool calling /
   Files API / computer-browser use；hosted-on-Azure 还少 code execution 与新版 web search
7. Foundry 无限流标准头；追踪用 `request-id` + `apim-request-id`
8. 会话亲和与 opaque state 跨 provider **不可迁移** —— 回落只能发生在新会话，这一点必须在
   候选选择里保留，不能为了三路径把它洗掉
