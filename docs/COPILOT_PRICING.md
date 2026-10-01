# GitHub Copilot 价格映射

Author: Zeno Ren

随代码提供的基线核对日期：2026-10-01。来源：[GitHub Models and pricing](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing)。官方表按 token 定价，1 AI credit = 0.01 USD。启用 Copilot 时，项目服务默认每 7 天后台获取、验证并更新官方价格。

`copilot_pricing.py` 基线收录当日官方表的 35 个模型、47 行档位，在上一版基础上补齐 GPT-6 Luna、GPT-6 Sol、GPT-6.1 Sol、Claude Opus 5.5、Claude Sonnet 5.5、Grok 4.7。官方 HTML 表格与脚注摘录位于 `tests/fixtures/copilot-pricing-20261001.html`；9 月 21 日 JSON 快照保留用于回归。后台获取与推理请求相互独立，不向 GitHub Docs 发送模型凭据、用户输入或用量记录。

## 每周自动更新

- 随 Copilot 的应用生命周期启动；没有可用缓存时先用内置基线，后台立即尝试获取。有效缓存从上次成功抓取时间起满 7 天刷新，重启不重置此时间。服务停机期间不执行任务，重启后补做已到期检查。
- 只读取上述固定 HTTPS 官方地址，验证 TLS，不跟随重定向。下载总预算 45 秒，HTTP 阶段超时 20 秒，解码后正文上限 2 MiB；不在推理热路径下载价格。
- 验证完整 provider 表、列名、价格数字、唯一模型映射、成对长上下文阈值、AI credit 换算和促销脚注。当前明确识别官方 Gemini 促销脚注格式；新的条件价、无法识别的脚注、缺表或结构变化会拒绝整个新快照，等待适配解析器。
- 成功后先用临时文件、fsync 和原子替换保存缓存，再一次性切换内存价格。缓存包含来源 URL、UTC 获取时间、正文及 SHA-256；重启重新校验和解析，不执行网页脚本。
- 失败时保留上次有效价格，1 小时后重试；后台失败不影响推理。缓存损坏、来源不符、时间来自未来或早于内置基线时不加载。价格核对超过 7 天显示过期提示；旧表中已到期的促销仍返回未知。
- 新价格用于更新之后记录的 Copilot 实时用量估算，不重新推理、不改写用量账本、不重算实时页已经累计的估算。历史页另按查询时的参考价生成只读视图，缺少单次档位时显示范围，见下文。
- 关闭应用时取消网络任务并等待缓存写入结束。机制是应用内后台任务，无需 Codex 自动化或 GitHub Actions；不自动提交价格到 Git、不重新构建或部署应用。

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `COPILOT_PRICING_AUTO_REFRESH` | `true` | `false`、`0`、`no` 关闭网络刷新；仍可加载有效缓存 |
| `COPILOT_PRICING_CACHE_PATH` | JSON 用量目录下的 `copilot-pricing.json`；MySQL 后端为 `./usage_data/copilot-pricing.json` | 可写缓存位置；建议放在已有持久卷中，每个实例使用自己的文件 |

多副本各自每周检查，允许短暂的价格版本差异；本机制不承诺跨实例原子刷新。无持久卷时进程内刷新仍可工作，但容器重建后会丢失缓存并重新获取。只读/不可写缓存目录会使刷新失败并保留现有价格，Dashboard 可见失败状态。

`/stats.github_copilot.pricing` 提供 `checked_on`、`model_count`、`source_sha256`、`catalog_origin` 与 `refresh`（启用状态、上次尝试、最近成功、下次检查、过期状态和安全错误类型）。Dashboard 的 Copilot 页同步显示核对日期和更新状态。界面的秒级自动刷新仍仅指统计刷新，与每 7 天价格更新分开。

| 模型 | 输入 | 输出 | 缓存读取 | 缓存写入 |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | $0.20 | $1.20 | $0.02 | $0.25 |
| GPT-5.6 Sol | $4.00 | $20.00 | $0.40 | $5.00 |
| GPT-5.6 Terra | $2.00 | $12.00 | $0.20 | $2.50 |
| GPT-6 Astra | $10.00 | $50.00 | $1.00 | $12.50 |
| GPT-6 Luna | $0.10 | $0.50 | $0.01 | $0.125 |
| GPT-6 Sol | $2.00 | $10.00 | $0.20 | $2.50 |
| GPT-6.1 Sol | $2.00 | $10.00 | $0.10 | $2.50 |
| Claude Opus 5.5 | $4.00 | $20.00 | $0.20 | $5.00 |
| Claude Sonnet 5.5 | $2.00 | $10.00 | $0.20 | $2.50 |
| Grok 4.7 | $2.00 | $6.00 | $0.50 | 不适用 |
| Claude Fable 5.1 | $10.00 | $50.00 | $0.25 | $12.50 |

表中单位均为 USD / 100 万 tokens，列出默认档；模块包含官方长上下文档。GPT-5.6 Luna 和 Grok 的阈值为 200,000 输入 tokens；GPT-5.4/5.5、GPT-5.6 Sol/Terra、GPT-6 Astra/Luna/Sol 和 GPT-6.1 Sol 为 272,000。严格超过阈值才使用长上下文档，按每条上游已报告 usage 分别计算，再累加费用，不能把多次短请求的累计输入当成长上下文请求。

Copilot 的 Chat/Responses 输入统计是包含缓存细分的总输入，因此普通输入计费量为 `input_tokens - cache_read_tokens - cache_creation_tokens`。缓存读取与写入分别应用官方价格；未设置独立缓存写入费的模型，其写入 tokens 仍按普通输入价估算，不视为免费。计费估算不改变原始 token 统计或追加账本。负数、不完整的必需 usage 或不一致的缓存总量不生成完整费用估计。

模型标识采用明确匹配，并兼容数字版本中的点号/连字符，例如 `claude-fable-5.1` 与 `claude-fable-5-1`。没有收录的版本或变体保持未知，不退回 GPT-5 或邻近模型价格。Gemini 3.6/3.7/3.8 Flash 的当前促销价仅使用至 2026-12-31，之后在更新官方价格前保持未知。

## 展示与口径

- Copilot 实时页增加美元估算、AI credits 及已定价/未定价覆盖数。金额是本次进程运行期间有 usage 的记录之估算；没有 usage 的调用不能据此推断免费。
- `/stats.github_copilot` 提供价格来源和日期。模型记录带 `pricing_status`、`priced_requests`、`unpriced_requests`、`pricing_tiers` 与 `known_cost_subtotal_usd`。
- 完全未知时 `estimated_cost_usd` 为 null；部分可估算时保留已知小计，页面标注“+ 未计价”，不把缺价变成 $0。整体存在缺价时不伪造完整总费用。
- 实际账单还取决于套餐额度、折扣、计费对象及 GitHub 的规则。旧年付 Pro/Pro+ 仍可能采用请求制，此 token 估算不代表其实际扣费；不把模型价表当成套餐发票。
- Databricks/Azure 实时页保持原有参考价格口径。历史汇总没有完整的逐次上下文边界和 provider 信息，不将历史参考估算宣称为某个渠道的实际账单。

## 历史页参考价格

`/stats/history` 的 OpenAI 风格模型使用同一份自动更新的 Copilot 公开价格表。由此补齐 GPT-6 Astra/Luna/Sol、Gemini 3.8 Flash、Grok 4.6 等旧历史表未收录的型号；历史模型匹配采用明确 ID 和小数点/连字符版本别名，不把未知变体套用到邻近型号。

- 这是**按查询时当前参考价估算已记录 tokens**，不是恢复发生当日的单价，也不是供应商发票。仅在返回视图里添加费用，不修改每日用量、缓存或追加账本。
- 多次请求的日累计输入不能直接套长上下文档位。如果日输入未超过阈值，可确定均在默认档；只有一条记录时可按该条输入判断；其他情况按各 token 分项的默认/长上下文单价给出保守上下界。
- 档位不明时 `estimated_cost_usd` 为 null，但 `estimated_cost_min_usd` / `estimated_cost_max_usd` 有值，`pricing_status=complete` 表示价格覆盖完整。UI 显示美元区间；模型汇总、日均费用及图表均保留上下界，不再把“有参考范围”显示成“未知”。
- 真正无价、促销已过期或用量不合法时仍为 unknown；有部分已知模型时保留已知范围并标注“+ 未计价”，不把未覆盖部分当作零。
- OpenAI 风格输入包含缓存细分，估算前扣除缓存读/写，再分别按对应单价收费；没有独立 cache-write 单价时按普通输入计。Anthropic/Databricks 的输入不包含缓存分项，历史视图保留其独立加总方式。
- 未被 Copilot 表覆盖的旧型号使用已有的明确模型参考价；Anthropic/Databricks 使用该参考表。历史日表没有 provider 字段，这些来源不表示已确认历史调用的供应商。

`/stats/history.pricing` 标注当前参考价口径和 Copilot 价格元数据。模型行提供 `pricing_basis`、`pricing_reason` 与参考金额范围；日汇总提供完整或部分范围。实时 Copilot 页仍按每次请求计价，不受历史范围展示影响。

## 2026-10-01 更新验证

本轮本地完整回归、应用镜像定向测试、真实官方价格抓取及 Dashboard 明暗主题渲染结果见 [验证回执](reviews/2026-10-01-pricing-refresh/VALIDATION.md)。

## 2026-09-21 历史验证

完整本地回归 634 passed、598 subtests passed；6 个隔离 MySQL 用例未在本轮重复执行。候选应用镜像无源码挂载验证 59 passed、49 subtests passed，1 个可选 OTel 用例跳过。价格表逐行核对、独立只读 QA、前端未知/部分费用断言和桌面浏览器合成数据渲染检查均通过，详见 [验证回执](reviews/2026-09-21-copilot-pricing/validation.json)。未调用真实模型或核对实际 GitHub 账单。
