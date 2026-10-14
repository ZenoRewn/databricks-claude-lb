# 渠道、存储与客户端配置

Author: Zeno Ren

从 [完整配置示例](../config.yaml.example) 复制本地 `config.yaml`，只启用实际使用的渠道。`CONFIG_PATH` 可以指定其他路径；`${VARIABLE}` 从服务进程环境展开，容器不会自动继承宿主机全部环境变量。

## 认证与凭据

`auth.api_key` 用于客户端访问 LB，与上游 Databricks、Azure、Copilot 凭据分开。多租户 `auth.api_keys` 的映射语义见配置示例及 [诊断契约](OBSERVABILITY_AND_CONTEXT.md)。不要把真实配置、`.env`、token 缓存或 Kubernetes Secret 放进 Git。

上游凭据可使用环境变量或已有 Secret 挂载。在现有 AKS 中改凭据来源、挂载路径或 `envFrom` 必须核对现场配置，不能只复制示例的一半。见 [AKS 指南](AKS.md)。

<a id="policy-degradation-config-is-the-baseline"></a>

### 策略降级：配置是基线，存储是 overlay

运行时策略（按密钥限渠道、模型→provider 优先序、模型→GHCP 账户、GHCP 账户启用与勾选）全部存在用量存储里，而它们**都是叠在配置之上的 overlay**。只有一条降级规则：

> **overlay 缺失或未加载时，回落到配置基线。**

| 层 | overlay 不可用时的结果 | 回落到的基线 |
|---|---|---|
| 按密钥限渠道 | 允许全部渠道 | config 里配置了哪些 provider |
| 模型 → provider 优先序 | 默认顺序 | Copilot 优先 → Azure fallback |
| 模型 → GHCP 账户 | 本层不限定账户 | 全部启用账户 |
| GHCP `allowed_models` | 本层不施加约束 | 端点的 `models` 列表 |
| GHCP 账户启用态 | 保持 inert 或启用 | 端点的 `enabled` 声明 |

前四行看起来像 "fail-open"，那只是因为它们的基线恰好宽松。规则本身是**回落基线**而非**放行** —— 最后一行的基线可以是 `enabled: false`，于是同一条规则给出的是「保持不投入服务」。把它读成例外会得出错误结论。

理由是限制层不应把数据库故障放大成网关整体不可用，但同样不该把故障变成一次意料之外的扩权。两种状态都显式暴露：从未成功加载为 `unavailable`，加载过但刷新失败为 `stale`（继续执行最后一次成功读取的策略），在 `/stats` 与 Dashboard 可见，不静默。

跨层的回归断言在 `tests/test_key_policy.py::DegradationFallsBackToTheConfigBaselineTests` —— 五层放在一个类里断言，任何一层改成真正的 fail-open 都会红。

<a id="dashboard-unlock-contract"></a>

### Dashboard「密钥与渠道」页的解锁契约

这一页的四块（客户端密钥、路由规则、GHCP 账户、上游凭据）都需要凭据。点**解锁**一次并发读出四块，之后每次保存自动复用同一个会话；某一块失败不影响其余三块，未配置的渠道显示为「该渠道未配置」而不是报错。

**页面脚本不持有任何凭据。** 解锁是把密码换成一个 **HttpOnly 会话 cookie**，此后由浏览器携带：

```
POST /admin/session  {"password": "…"}        → Set-Cookie: lb_panel=…; HttpOnly; SameSite=Strict; Path=/admin
GET  /admin/session                           → {authenticated, expires_in, password_configured}
DELETE /admin/session                         → 服务端失效 + 清浏览器那一份
```

- 密码来自 `LB_DASHBOARD_PASSWORD`（env，来自 Secret）。**未配置或短于 12 字符 → 面板登录整体不可用**，绝不退化为无密码放行 —— `/admin/*` 由没有 OAuth 的那个 ingress 提供，对公网可达
- 会话是服务端进程内的 **30 分钟滑动窗口**（任何操作续期），**重启全部失效**，多副本之间不共享（换 Pod 要重新解锁，不是故障）
- **刷新页面保留会话** —— 这正是目的：过去刷新就等于重输。页面加载时先问 `GET /admin/session`，已认证就直接恢复四块，全程零交互
- 锁定会**在服务端一并失效**并清空已读出的内容（掩码、指纹、账户名本身就是会话授权才能看到的）。只清浏览器那一份会留下一个仍可用的会话
- 连续 5 次失败按来源**指数退避**（30s 起翻倍，有上限）。退避期内**正确密码也拒**且不区分原因，否则它就成了一个旁路的密码预言机。退避**只计密码路径**：让它挡住高熵的 API Key 等于把一次猜测攻击变成自伤
- `Secure` 只在 https 下设置，否则本地 http 调试根本拿不到 cookie
- `SameSite=Strict` + `Path=/admin` + 同源 XHR 已覆盖 CSRF，不再加 token

**没配密码时回落到 LB API Key**，而它**也换成同一种会话**而不是留在页面里 —— 否则「脚本不持凭据」只在一半情况下成立。安全上等价：能出示有效 key 的人本来就能直接调用每个 admin 端点。

这一页在入口层已由 oauth2-proxy 保护（`lb.zeno.ink/stats*`，限定单个 Google 账号），但 `/admin/*` 走的是另一个没有 OAuth 的 ingress —— 所以凭据仍然必要，不能把「页面已登录」当作授权。

**上游凭据不在此列。** `credSecret` 保持提交后立即清除的单次请求契约 —— 会话是便利，上游凭据是秘密。

### 按密钥限制后端渠道

渠道 id 为 `databricks`(ADB) / `copilot`(GHCP) / `azure_openai`(Foundry)，与用量记录里的 `provider` 取值一致。设置入口是 Dashboard「密钥与渠道」页或 `PUT /admin/api-keys/{tenant}/providers`，存在用量存储中，立即生效并跨重启保留。**未设置的租户默认允许全部渠道**，所以既有密钥行为不变。

allowlist **只做减法**：它从候选里移除渠道，不新增候选、不改「Copilot 优先 → Azure fallback」的既有顺序，也不能让某渠道服务它本来不支持的模型。策略拒绝返回 403 `provider_not_allowed`，与「无可用端点」的 404 区分，便于分清是策略还是容量。

设置接口**不接受空集**（允许空集等于把该租户锁在门外，且无从与手误区分），所以回到「不限制」要用 `DELETE /admin/api-keys/{tenant}/providers`（Dashboard 该行的「重置」按钮）—— 它把该租户清回配置基线，幂等，已无限制时也返回 200。

存储层保存的是**租户名与渠道 id**，不含任何密钥材料。

降级按[统一规则](#policy-degradation-config-is-the-baseline)回落到配置基线，对这一层即「允许全部渠道」；状态在 `/stats` 的 `provider_policy` 与 Dashboard 上显式暴露，不静默。发布预检会校验该表的列宽与主键，但表不存在时不报错 —— 它由应用首次使用时惰性创建，引入它的那次发布之前本就不存在。

### 密钥身份与轮换

`GET /admin/api-keys` 返回密钥**身份**：租户名、掩码（首尾各 4 位；长度不足 12 位时不给掩码，因为露出 8/10 个字符等于交出大半个秘密）、sha256 前 12 位指纹，以及**进程内**请求数与最后使用时间（重启归零，不代表密钥未被使用）。**不返回明文**。

轮换沿用 `/admin/copilot/reload` 的模型：**在源头（Secret / env / 文件）替换，再调 `POST /admin/api-keys/reload` 让网关重读**。网关不生成也不存储客户端密钥。重读只处理 auth 段，不重建 provider 与连接池；校验在替换之前完成，所以配置写坏时原有密钥仍然有效。`${ENV}` 形式的值取自进程环境，改文件不会改变它们，端点会在响应里如实标注仍需滚动重启。

## Databricks

每个 `endpoints[]` 配置独立 `name`、`api_base`、`token` 和权重。可用 `models` 限定转换后的原生模型名；没有白名单表示保留通配路由，不是上游能力认证。

Claude 模型使用 `/v1/messages`。在 OpenAI 风格入口传入 Claude 模型会被拒绝，不会隐式换协议或供应商。模型映射和支持范围以代码配置及上游实际可用性为准。

### 模型名解析

`get_databricks_model` 规范化而不猜测版本：定家族（opus/sonnet/haiku/fable）→ 剥离日期戳与 `latest` → 统一 `-_.` 分隔符 → 拼成 `databricks-claude-<家族>-<版本>`。

- **写了版本号就按写的转发**，由上游裁决是否存在。新版本上线当天无需改码即可用；不存在的版本得到上游明确的 `passthrough is not supported for model ...`，不会被静默降级成另一个模型。
- **只有完全没写版本号**才用 `DATABRICKS_FAMILY_DEFAULTS` 里的家族默认值，该默认值是代码里唯一的版本假设。
- 日期戳与 `latest` 之外的尾缀一律保留进模型名（如 `-fast`），避免被吞掉后按另一档价格计费。
- `databricks-` 前缀的名字原样透传，不做任何解析。

历史上这里曾用正则嗅探版本，其分隔符字符类同时匹配 `5.5` 里的第二个点，导致显式请求 Opus 5.5 静默跑在 Opus 5 上；因此不要恢复按版本分支的 if/elif 写法。

解析成功不代表每个端点都有该模型。`claude-fable-5-1` 仅部分 workspace 可用，其余返回 `NOT_FOUND ... not available in your region`，按端点轮询会间歇失败；区域受限模型需要用 `endpoints[].models` 限定到确有该模型的端点。Opus 5.5 另有间歇性 `TEMPORARILY_UNAVAILABLE` 容量拒绝，属上游容量，网关侧无开关。

### `/models` 清单语义

`/v1/models` 同时列出 Databricks Claude 与 OpenAI 风格两侧的模型。核验过的条目带非标准 `supported_endpoints` 字段，值是完整入口路径；OpenAI SDK 会忽略未知字段，不影响兼容性。

入口差异是实测的，不是推断：部分模型只在 `/v1/responses` 可用（在 `/v1/chat/completions` 上报 `not accessible via the /chat/completions endpoint`），`gemini-3.8-flash` 只在 Chat 可用（在 Responses 上报 `does not support Responses API`），`gpt-5.6-luna` 两者皆可。`claude-*` 只标 `/v1/messages` —— 它们在 OpenAI 风格入口按不变量一律拒绝。

两类条目要分清：

- **静态核验目录**（Copilot 配 `models: []` 通配时使用）是运维人工核验的**时点快照，不是供应商能力认证**。2026-10-10 核验时移除了 14 个实际不可用的条目（`gpt-5`、`gpt-5.1`、`o3` 系列、`gemini-2.5-*` 等均 404 `rejected model`；`gpt-5.6-cyber` 为 integrator 限制），补入 7 个实测可用的条目。发现条目失效时请重新核验后更新，不要凭价目表推断 —— 价目表是 GitHub 公布的清单，不等于本账户可用。
- **配置发现条目**（Azure `deployments`、Copilot 显式 `models` 白名单）照常收录，但**不带** `supported_endpoints`：其入口支持情况无从核验，宁可不声明。

目录里的模型必须至少能被 `MODEL_PRICING` 或 `copilot_pricing` 之一计价，否则成本会静默漏统计。两张表分工不同：前者是公开 API 价（历史/通用口径），后者是 Copilot 官方分档价并用于实时估算，所以 Copilot 专属模型（如 grok、mai-code）只在后者有价，不应为其臆造公开 API 价。

### 录入上游凭据（ADB / Foundry / GHCP）

三个 provider 都可以由运维录入认证信息；GHCP 另外支持由网关驱动 GitHub device flow。入口是 `/admin/credentials/*` 或 Dashboard「密钥与渠道」页。

**这是对早先「网关从不存储凭据」的有意变更。** 原因是现场约束：AKS 上 Secret 挂载是只读 tmpfs、`config.yaml` 只读、应用 ServiceAccount 没有 secret 写权限 —— 用量库是网关唯一可写的持久存储。

因此封装是强制的，不是可选项：

- **AES-256-GCM**，密钥来自 `LB_CREDENTIAL_KEY`（base64 32 字节）。密钥留在 Kubernetes Secret 里、**不进数据库**，所以单独拖走数据库或备份不足以解开
- 每次写入独立随机 nonce，相同凭据不会产生相同密文
- `provider|account` 作为 AAD 参与认证，密文**无法被挪到另一行**解开
- **未配密钥时录入功能整体返回 503**，绝不退化为存明文
- 任何端点都不返回明文或密文；只给指纹、校验结果与非敏感元数据

封装**没有**消除的风险（不要据此认为凭据绝对安全）：运行中的进程内存里有明文，能在 Pod 内执行命令的人仍可取得；没有托管 KMS、没有自动轮换；**密钥丢失则已存凭据不可恢复**，只能重新录入。密钥换了之后旧记录会被标为 `sealed_with_other_key` 并跳过建端点，而不是抛一个裸的解密错误。

**录入时即校验**，不通过不入库。各 provider 用最便宜的只读调用，不发推理请求：

| Provider | 校验 |
|---|---|
| `databricks` | `GET {api_base}` 列 serving endpoints |
| `azure_openai` | `GET {api_base}/openai/v1/models` |
| `copilot` | token 交换（带 editor 头部）→ SKU；再取模型清单，顺带写入该账户的 capability |

**生效需一次滚动重启。** 端点仍在启动时统一构建，来源是「配置 + 已存凭据」两处；运行时增删端点会重洗会话亲和哈希并打乱账户 pinning，是有意避开的。Dashboard 会标出哪些凭据尚未生效。与配置中已声明的端点同名会被拒（409），不允许静默遮蔽。

device flow 的 `device_code` 只存在进程内并带 TTL，不落库、不进日志、不出现在响应里；`authorization_pending` 是正常中间态而非错误。发布预检会校验 `provider_credential` 的列与主键，表不存在时放行。

### 路由规则：模型 → Provider → GHCP 账户

`PUT /admin/model-routing/{model}` 设置单个模型的规则（Dashboard「密钥与渠道」页是三段式的下拉表单）。两段都可选：省略 `providers` 只改账户，省略 `copilot_accounts` 不动账户。**没有规则的模型维持原本行为。**

两段都在**任何写入之前**校验完。第三段打错一个账户名却让 provider 那半边已经生效，是最难排查的一种半应用。

#### 第一段：provider 优先序

**顺序即偏好**，不去重也不重排，重复项会被拒。

**偏好只重排，不筛除。** 列出的 provider 提到前面，未列出的按默认顺序（`copilot` → `azure_openai`）接在后面，所以 fallback 能力完整保留。provider 维度的筛除是 per-key allowlist 的职责；两者叠加后的候选是：

```
(能服务该模型的 provider) ∩ (该 key 允许的 provider)，先按偏好排序，再按默认顺序
```

偏好里的 `databricks` 不参与 OpenAI 风格入口的顺序 —— 它服务 `/v1/messages`。Dashboard 的下拉因此只给 GHCP 与 Foundry 两项：列 ADB 会给出一个什么都不做的选项。

#### 第二段：限定 GHCP 账户

`copilot_accounts` 把该模型限定到指定的 GHCP 账户，**空列表或 `null` = 不限定**（这是基线，也是撤销限定的唯一表达方式）。账户名必须是已配置的 Copilot 端点，否则 400 `unknown_copilot_account` —— 写错的名字静默生效就等于把该模型的 GHCP 候选清空，而界面上看着像配好了。

这一段**是筛除**，但筛的是**同一渠道内的哪个账户**，不是哪个 provider，所以上面那条「只重排不筛除」不受影响，provider 级 fallback 完整保留。

它与按账户的 `allowed_models` 是**交集**：前者把账户从模型上摘掉，后者把模型从账户上摘掉，任一侧成立即移出候选。排查某账户为何不接某模型时两处都要看 —— Dashboard 的「生效候选」列就是为此存在，它由服务端按 `CopilotEndpoint.supports()` 算出（客户端算不出来，需要同时拿到 capability、`allowed_models` 和启用态）。

选中的账户恰好都服务不了该模型时，GHCP 候选为空，请求落到 Foundry —— 与那些账户全部熔断同效，不是错误。响应的 `copilot_candidates_empty` 会标明这一点，Dashboard 保存后给出警告但**不阻止**：那是运维的决定。

**会话亲和不受影响。** 筛选加在 `CopilotEndpoint.supports()`，那是所有候选判断的唯一收口，含亲和用的「配置态合格集合」。限定让这个集合变小但仍然稳定（它只随策略变，不随熔断变），所以同一会话在同一策略下仍落同一账户。注意换一组限定会改变集合大小，亲和哈希因此重新分布 —— 这是既有哈希的固有性质，也是为什么它是运维的显式决定而非自动行为。

存储用独立的 `model_copilot_account` 表而不是给 `model_provider_preference` 加列：后者在生产已经存在，加列是真迁移；新表由首次使用惰性创建，零迁移。发布预检校验它的列与主键（主键丢了会让两行声明同一个模型，限定变成取决于行序），表不存在时放行。

### `claude-*` 走 GHCP：逐模型 opt-in

`claude-*` 默认仍被 OpenAI 风格入口拒绝（400）。给**某个具体模型**配置指向 `copilot`/`azure_openai` 的偏好后，该模型在这些入口放行 —— Copilot 原生服务它的 Responses/Chat 形状。这是逐模型开关，不是整族开闸：开放 `claude-opus-5.5` 不会同时开放 `claude-sonnet-5.5`。清除偏好即恢复拒绝。

**作用范围限制：`/v1/messages` 仍然只走 Databricks。** 从 Copilot 提供 Anthropic Messages 协议需要双向适配器 —— 本仓库能解析 Anthropic SSE（`main.py` 的流观测与 `release_probe`），但没有合成器，`chat_adapter` 是 Chat→Responses 的 OpenAI 内部转换。所以通过 Claude Code 这类走 `/v1/messages` 的客户端**用不到**这条通路；它面向 OpenAI 风格客户端（Codex、OpenAI SDK）。

**计价必须分渠道**，否则费用会算错。区分依据是记录的模型名：Databricks 记映射后的 `databricks-claude-*`，Copilot 记自己的 id（裸 `claude-*`）。前者按 Anthropic 公开 API 价与「input 不含 cache」口径，后者按 `copilot_pricing` 分档价与 inclusive-input 口径。2026-10-12 核对生产 `usage_daily`：14 条 Claude 行全部带 `databricks-` 前缀、零条裸 `claude-*`，所以这项修正不改变任何既有行的计价基准。

### GHCP 多账户：启用与按账户勾选模型

账户凭据不进配置。`endpoints[]` 里只写 `name`，token 会按 `copilot-auth-<name>.json` 从 device-flow 缓存目录解析（AKS 上由 Secret 挂载），`token_source` 记为 `file`，因此 `/admin/copilot/reload` 可以重读它 —— 写死在配置里的明文（`literal`）做不到。

每个账户有两层模型信息：

- **capability** —— 由 `POST /admin/copilot/accounts/{account}/discover` 向上游发现并存快照，只读、带时间戳。**它是时点发现，不是能力认证**，上游随时会变
- **allowed_models** —— 运维在 capability 内勾选的子集，`null` 表示不额外限制

有效可用集 = `capability ∩ allowed_models`。capability 本身就是外层边界：上游说某账户没有该模型时，路由过去没有意义，所以即使没设 allowlist 也不会落过去。从未发现过 capability（空）时不据此限制，避免把「还没发现」当成「什么都没有」。写入时超出 capability 的模型被拒，所以**不可能**把账户配成服务它没有的模型。

停用账户用启用开关，不要靠清空勾选 —— 空列表会被拒，否则两种语义会混淆。

#### 先落地、确认后再启用

账户的启用态是**三态**，不是开关：

| 状态 | 来自 | 接流量 |
|---|---|---|
| 未启用 | 配置写了 `enabled: false`，且运维从未做过决定 | 否 |
| 已启用 | 运维通过 `PUT /admin/copilot/accounts/{account}` 显式启用 | 是 |
| 已停用 | 运维显式停用 | 否 |

`enablement_source` 字段说明这个值现在由谁决定（`config` 或 `operator`）。配置只给**基线**，一次显式写入之后由存储接管，两个方向都归它裁决。

需要 `enabled: false` 是因为别无他法：token 来自只读的 Secret 挂载，端点又只在启动时构建，所以不把账户写进配置就根本没有可供 Dashboard 操作的对象 —— 而写进去在过去就等于立刻投入服务。

两点容易被误读，但都是[统一降级规则](#policy-degradation-config-is-the-baseline)的直接结果，不是特例：

- **存储不可用时保持配置声明的基线。** 规则是「回落基线」而不是「放行」；这里的基线是 `enabled: false`，所以结果是保持 inert。放行一个从未被启用过的账户不是「恢复既有行为」，而是把一个按设计不该接流量的账户推上线。
- **发现能力不等于启用。** `discover` 是对上游的只读查询，它会创建记录但**不写** `enabled` —— 沿用本节开头「capability 是上游事实且只读，allowed_models 是运维意图」的同一条分离。存储层因此把 `enabled` 持久化为可空列（`NULL` = 尚无决定）；否则一次「刷新能力」就会把 inert 账户推上线。

启用一个配置声明为 inert 且没有 `models`、也从未发现过 capability 的账户会被拒（400 `capability_unknown`）：它没有任何边界，启用即通配，会成为每个模型的候选 —— 包括自己席位没有的那些。顺序是**先发现能力 → 勾选 → 启用**。

与 `/models` 目录的关系：目录由启用账户的有效集派生。通配端点（`models: []`）若已发现 capability，就用 capability 而非静态核验表，因为前者更准。未启用的账户不贡献任何模型，也不会触发通配回落。

降级按[统一规则](#policy-degradation-config-is-the-baseline)回落到配置基线：`allowed_models` 这一层不再施加约束（端点的 `models` 仍然生效），启用态保持配置声明的那个值。发布预检会校验 `copilot_account_policy` 的列与主键，表不存在时放行（它由首次使用惰性创建）。

实测提示：同一 SKU 的账户模型集合可能完全一致（2026-10-12 实测 4 个 business seat 账户各 38 个模型、零差异），差异主要在 SKU 层。逐账户勾选在同 SKU 之间不产生区分效果。

## Azure OpenAI

```yaml
azure_openai:
  endpoints:
    - name: azure-eastus
      endpoint: https://YOUR-RESOURCE.openai.azure.com
      api_key: ${AZURE_OPENAI_KEY}
      deployments: [YOUR_DEPLOYMENT_NAME]
      weight: 1
```

`deployments` 必须使用资源中已存在的部署名。Copilot 和 Azure 共享 OpenAI 风格入口，Copilot 优先；只有既有兼容性/明确未准入分支才可能转向 Azure。任意 503、读超时、写失败或中途断流都不能被当成安全 fallback 的证明。

## GitHub Copilot

```yaml
github_copilot:
  endpoints:
    - name: gh-account-1
      weight: 1
      models: []
      api_types: [responses, chat]
```

`name` 是稳定且唯一的账户端点标识，关联日志、Dashboard、CLI 和缓存文件。`models: []` 为通配，并不证明每个模型/API 都受上游支持。`api_types` 可限定真实支持的入口协议。

凭据解析顺序为：显式 `github_token`（支持环境变量）、项目 Device Flow 缓存、兼容旧缓存。多账户不得复用同一份解析出的凭据；显式配置来源失效时不能靠隐式换账户恢复会话。

```bash
python main.py --copilot-login --endpoint gh-account-1
```

按终端给出的官方 verification URL 和 user code 完成登录。缓存位于 `~/.config/databricks-claude-lb/copilot-auth-<sanitized-name>.json`，权限 0600。AKS 挂载时文件名须对应端点名，具体 Secret 方式见部署指南。

OAuth 凭据用于交换短期 session token，后者的有效期以上游 `expires_at` 为准。后台刷新和有界 401 修复不能保证凭据永不被撤销或账户额度永不受限。多账户的 opaque state、pinning 和会话亲和约束见 [状态说明](OPAQUE_STATE.md) 与 [可靠性契约](RESILIENCE.md)。

Copilot 实时费用采用 GHCP token 价格表及适用上下文档位；未知价格保留未知。套餐额度、折扣和历史计费模式请核对 GitHub 账单，不从 LB 计数推算剩余额度。完整口径见 [计费说明](COPILOT_PRICING.md)。

## 用量存储

JSON 适合单实例本地使用，目录需对运行用户可写。Docker 镜像以 UID 1000 运行，绑定挂载时核对目录权限。JSON 不支持多个 writer 共享同一文件。

```yaml
usage_storage:
  type: mysql
  host: YOUR_MYSQL_HOST
  port: 3306
  user: YOUR_MYSQL_USER
  password: ${MYSQL_PASSWORD}
  database: claude_lb
  pool_size: 5
  retention_days: 90
```

MySQL 使用 InnoDB 日汇总与批次账本。提交成功后 ACK 丢失可以按批次幂等重投；尚未落盘的内存事件没有 durable outbox。历史金额缺少逐次渠道与上下文信息，不能重算成准确的 Copilot 分档账单。

Dashboard 的数据维护需要有效 LB Key 和明确确认。密钥仅驻留本次交互，操作后清空，不写入 localStorage。超时或响应未知时先查实际历史，不自动重发删除。

## 客户端接入

### Claude Code

```bash
export ANTHROPIC_BASE_URL='http://127.0.0.1:8000'
export ANTHROPIC_API_KEY="$LB_API_KEY"
claude
```

### OpenAI 兼容 SDK

使用 `base_url=http://127.0.0.1:8000/v1`，API Key 为网关密钥。模型必须在已配置渠道上可用。Responses-only 模型应优先走 `/responses`；Chat 适配可能缓冲完整 Responses 结果，不能据此期望相同 TTFT。

### 使用自定义 Responses provider 的客户端

支持自定义 provider 的客户端可使用如下配置形态；配置键以所用客户端版本为准：

```toml
model_provider = "claude-lb"
model = "YOUR_CONFIGURED_MODEL"

[model_providers.claude-lb]
name = "claude-lb"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
env_key = "OPENAI_API_KEY"
```

将 `OPENAI_API_KEY` 设为 LB 密钥。发生代理或 localhost 解析问题时，先按 [排障指南](TROUBLESHOOTING.md) 核对网络路径，避免把客户端中断直接归因于上游。

## 参数与监控

完整默认值以鉴权后的 `/config/effective` 为准。预算、参数严格模式、图片裁剪、渠道能力目录、v2/v3 指标和安全日志分别见 [当前契约](OBSERVABILITY_AND_CONTEXT.md)。未知或过期的渠道能力不自动成为硬限额。
