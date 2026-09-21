# GitHub Copilot 价格映射

Author: Zeno Ren

核对日期：2026-09-21。来源：[GitHub Models and pricing](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing)。官方表目前按 token 定价，1 AI credit = 0.01 USD；之前“GHCP 无 per-token 计费”的页面说明已移除。

`copilot_pricing.py` 收录当日官方表的 29 个模型、37 行档位，包含 GPT-6 Astra、GPT-5.6 Sol/Terra/Luna、Claude Fable 5/5.1、Gemini 3.5～3.8 Flash、MAI-Code、Grok 和 Kimi。官方原始表格快照位于 `tests/fixtures/copilot-pricing-20260921.json`，回归逐行核对价格，不在推理路径请求外部价格站点。

| 模型 | 输入 | 输出 | 缓存读取 | 缓存写入 |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | $0.20 | $1.20 | $0.02 | $0.25 |
| GPT-5.6 Sol | $4.00 | $20.00 | $0.40 | $5.00 |
| GPT-5.6 Terra | $2.00 | $12.00 | $0.20 | $2.50 |
| GPT-6 Astra | $10.00 | $50.00 | $1.00 | $12.50 |
| Claude Fable 5.1 | $10.00 | $50.00 | $0.25 | $12.50 |

表中单位均为 USD / 100 万 tokens，列出默认档；模块包含官方长上下文档。Luna 和 Grok 的阈值为 200,000 输入 tokens；GPT-5.4/5.5、Sol/Terra、Astra 为 272,000。严格超过阈值才使用长上下文档，按每条上游已报告 usage 分别计算，再累加费用，不能把多次短请求的累计输入当成长上下文请求。

Copilot 的 Chat/Responses 输入统计是包含缓存细分的总输入，因此普通输入计费量为 `input_tokens - cache_read_tokens - cache_creation_tokens`。缓存读取与写入分别应用官方价格；未设置独立缓存写入费的模型，其写入 tokens 仍按普通输入价估算，不视为免费。计费估算不改变原始 token 统计或追加账本。负数、不完整的必需 usage 或不一致的缓存总量不生成完整费用估计。

模型标识采用明确匹配，并兼容数字版本中的点号/连字符，例如 `claude-fable-5.1` 与 `claude-fable-5-1`。没有收录的版本或变体保持未知，不退回 GPT-5 或邻近模型价格。Gemini 3.6/3.7/3.8 Flash 的当前促销价仅使用至 2026-12-31，之后在更新官方价格前保持未知。

## 展示与口径

- Copilot 实时页增加美元估算、AI credits 及已定价/未定价覆盖数。金额是本次进程运行期间有 usage 的记录之估算；没有 usage 的调用不能据此推断免费。
- `/stats.github_copilot` 提供价格来源和日期。模型记录带 `pricing_status`、`priced_requests`、`unpriced_requests`、`pricing_tiers` 与 `known_cost_subtotal_usd`。
- 完全未知时 `estimated_cost_usd` 为 null；部分可估算时保留已知小计，页面标注“+ 未计价”，不把缺价变成 $0。整体存在缺价时不伪造完整总费用。
- 实际账单还取决于套餐额度、折扣、计费对象及 GitHub 的规则。旧年付 Pro/Pro+ 仍可能采用请求制，此 token 估算不代表其实际扣费；不把模型价表当成套餐发票。
- Databricks/Azure 与历史按天累计的通用参考价格保持原口径。历史汇总没有足够的逐次上下文边界和 provider 信息，本轮不将其重算为 GHCP 官方分档费用，也不回填猜测的历史账单。

本轮只更新代码与本地验证。服务环境只有在后续部署新镜像后才会显示这些变化。
