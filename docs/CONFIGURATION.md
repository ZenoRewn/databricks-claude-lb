# 渠道、存储与客户端配置

Author: Zeno Ren

从 [完整配置示例](../config.yaml.example) 复制本地 `config.yaml`，只启用实际使用的渠道。`CONFIG_PATH` 可以指定其他路径；`${VARIABLE}` 从服务进程环境展开，容器不会自动继承宿主机全部环境变量。

**引用解析不出来就拒绝启动。** 凭据字段（`auth.api_key`、`auth.api_keys[*]`、各 provider 的 `token`/`api_key`、`usage_storage.password`）经 `resolve_credential()` 解析：`${X}` 指向一个未设置或为空的变量时报错并带上 config 路径与变量名，**不**带任何已解析出的值。**字面空值仍合法** —— 多租户配置里 `auth.api_key` 本就缺省，所以全内联的配置加载行为分毫不变。

这一条的由来：早先缺失变量被静默替换成空串，而没有任何一层拒绝空 token —— 端点照常构建、照常打 `Loaded`、照常进 `/config/effective`，唯一症状是上游 401 而配置和日志里毫无线索。凭据全部内联时这无所谓（没有引用可写错），改成从 Secret 注入后「变量名拼错」就是首要失败模式。

**唯一例外是 Copilot 的 `github_token`**：它会回落到挂载的凭据缓存，并已经明确警告「刚 rotate 的 Secret 没有在用」。那是有落点的韧性；其他 provider 没有落点，空凭据只是坏的。

代价要说清：单副本 + `maxSurge:0/maxUnavailable:1` 意味着拒绝启动就是停服直到回滚。所以改 Secret 或 config 前要先用一个短期校验 Pod 跑一遍 `load_config`（挂新 config + 新 Secret），确认退出码 0 再动生产 —— 这与镜像预检是同一种关卡。

## 认证与凭据

`auth.api_key` 用于客户端访问 LB，与上游 Databricks、Azure、Copilot 凭据分开。多租户 `auth.api_keys` 的映射语义见配置示例及 [诊断契约](OBSERVABILITY_AND_CONTEXT.md)。不要把真实配置、`.env`、token 缓存或 Kubernetes Secret 放进 Git。

上游凭据可使用环境变量或已有 Secret 挂载。在现有 AKS 中改凭据来源、挂载路径或 `envFrom` 必须核对现场配置，不能只复制示例的一半。见 [AKS 指南](AKS.md)。

<a id="policy-layers-overview"></a>

### 策略层总览：五层，谁做减法、谁只排序、在哪收口

运行时策略有五层，都存在用量库里、都叠在配置之上。**改任何一层之前先看清这张表**，
因为「为什么这个模型不走这个账户」几乎总是多层叠加的结果，而不是某一层单独的锅。

| 层 | 粒度 | 作用 | 存储 | 读取函数 |
|---|---|---|---|---|
| 渠道 allowlist | per-key（租户） | **减法**：移除 provider 候选 | `api_key_policy` | `allowed(tenant)` |
| provider 优先序 | per-model（按**模型身份**，三种拼法共用一条） | **只排序，不减** | `model_provider_preference` | `model_preference(model)` |
| GHCP 账户选择 | per-model（同上） | **减法**：限定该模型用哪些账户 | `model_copilot_account` | `model_allows_account(model, account)` |
| `allowed_models` | per-account | **减法**：限定该账户服务哪些模型 | `copilot_account_policy` | `account_allows(account, model)` |
| 账户启用态 | per-account | **三态开关** | 同上（`enabled` 可空列） | `account_policy(account)` |

两条容易混淆的边界：

- **第 2 层只排序。** 它把列出的 provider 提前、未列出的按默认顺序接在后面，
  **fallback 永不丢失**。provider 维度的减法只有第 1 层做。
- **第 3 层是减法，但筛的是「同一 provider 内的哪个账户」**，不是哪个 provider。
  所以它与上一条不冲突，provider 级 fallback 不受影响。

第 3、4 层是同一件事的两个方向（模型→账户 / 账户→模型），**取交集**：任一侧成立
即移出候选。排查某账户为何不接某模型时两处都要看 —— Dashboard 路由规则表的
「生效候选」列就是为此存在，它由服务端算出，客户端算不出来。

<a id="supports-is-the-choke-point"></a>

#### 收口在 `CopilotEndpoint.supports()`

第 3、4、5 层**全部**在 `main.py` 的 `CopilotEndpoint.supports()` 里生效，加上 config
自己的 `models` 列表与 `api_types`。新增任何 Copilot 候选条件都必须加在那里，
原因不是整洁而是正确性：

- `supports()` 是所有候选判断的**唯一**入口，包括会话亲和用的「配置态合格集合」
  （`main.py` 的 `_select_endpoint`）。漏掉亲和那一处，会在某个账户回落时把
  **所有**会话重新洗牌，而不只是原本落在它上面的那些。
- `can_handle()` / `supports_model()` / `/models` 目录都走它。**2026-10-14 的一个真实缺陷
  就出在这里**：`_collect_model_catalog` 当时直接问 `account_allows` 而不是过
  `supports()`，于是目录继续宣传一个「唯一被选中的账户根本服务不了」的模型 ——
  客户端选了就路由失败。回归断言在 `tests/test_model_routing.py` 的
  `test_the_model_catalogue_follows_the_selection` 与 `…_a_wildcard_endpoint_also_honours…`。

所以：**要给候选加条件，改 `supports()`；要读候选，问 `supports()`。** 两边都不要绕。

<a id="exclusion-reasons"></a>

#### 排除原因：与候选判断同一份代码

Dashboard 要回答「这个账户为什么不接这个模型的流量」。这个解释一旦与候选判断分成
两份实现就必然漂移，而漂移的那一份恰好是**被人当真的诊断** —— 比没有诊断更糟。所以
反过来写：**reason 是主实现，布尔是派生**。

```python
# main.py
def _copilot_support_reason(endpoint, model, api_type=None, *, ignore_model_selection=False)
class CopilotEndpoint:
    def supports(self, model, api_type=None):
        return _copilot_support_reason(self, model, api_type) is None
```

`key_policy` 两个布尔读者同样委托给各自的 reason 读者（`account_allows` →
`account_exclusion_reason`，`model_allows_account` → `model_accounts_exclusion_reason`）。
防漂移的守卫是对**所有 (账户 × 模型 × api_type × 存储状态) 组合**断言两者永不相左，见
`tests/test_key_policy.py::ExclusionReasonTests` 与 `tests/test_model_routing.py::SupportReasonTests`。

原因是**闭集**（`main.COPILOT_EXCLUSION_REASONS`），不是自由文本 —— 否则 UI 无从分支、
测试无从断言：

| reason | 含义 | 谁管这一层 |
|---|---|---|
| `null` | 在候选里，接流量 | — |
| `account_disabled` | 运维显式停用过 | GHCP 账户页的启用开关 |
| `account_never_enabled` | 配置声明 inert，从没有人决定过 | 同上（先刷新能力再启用） |
| `not_selected_for_model` | 本模型被限定到了别的账户 | 路由规则页的账户选择 |
| `excluded_by_allowed_models` | 账户侧 `allowed_models` 把该模型摘掉了 | GHCP 账户页的模型勾选 |
| `not_in_capability` | 上游的时点发现里没有这个模型 | 只读事实，可重新「刷新能力」 |
| `not_in_configured_models` | config 的 `models` 列表限定 | 改配置 + 滚动重启 |
| `api_type_unsupported` | 账户的 `api_types` 不含该入口形状 | 改配置 |

两处区分必须保留，因为它们指向**不同的修法**：

- `not_in_capability`（上游事实，重新发现）vs `excluded_by_allowed_models`（运维意图，改勾选）
- `account_disabled`（有人关过）vs `account_never_enabled`（从没开过）—— 合并会让人去找一个
  自己从未拨过的开关
- `not_in_configured_models` 是唯一需要改配置并重启的，不能混进那些改一下勾选就生效的 overlay 原因

**`capability_unknown` 故意不在这个集合里。** 没发现过能力的账户服务**所有**模型，那是通配
而不是排除；把它当排除原因会让诊断说反话。它由 `capability_known: false` 单独表达，是启用
前该看见的警告。

`ignore_model_selection=True` 回答的是「若把它选上，它能服务吗」，只跳过账户选择这一层。
Dashboard 用它在**保存之前**算出生效候选：服务端给「能否服务」，客户端只做一次
`勾选 ∩ 能服务` 的平凡交集，不复制任何策略逻辑。

reason 与 capability 一样是**时点**判断，不是能力认证。

<a id="enabled-versus-receiving-traffic"></a>

#### 「已启用」与「实际接流量」是两个状态

账户可以开着却一个模型都服务不了 —— 因为**另一层**（per-model 账户选择）把它从每个
已知模型上摘掉了。只显示启用开关会让人读成「开着就是在用」。

`/admin/copilot/accounts` 因此单独返回 `receiving_traffic` 与 `serves_models`，Dashboard
把它渲染成独立一列：**已启用但闲置**。计数口径是「时点目录 ∪ 各账户已发现能力」，
页面上标注了基数 —— 不是「世上所有模型」。

外部先例是 AWS ALB 的 `unused`：已注册但没有任何监听规则路由到它的 target，被报成一个
与 `unhealthy` **并列**的独立状态，理由正是成因在另一层。反面教材是 LiteLLM 的 Admin UI
列出已被 router 丢弃的模型行（显示配置态而非生效态），有一例 17 天后才被发现。

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

这一页需要凭据。点**解锁**一次并发读出各块，之后每次保存自动复用同一个会话；某一块失败不影响其余，未配置的渠道显示为「该渠道未配置」而不是报错。

**锁定时只渲染解锁卡片，被门禁的内容整块 `hidden`。** 这不只是版面选择：`hidden` 就是
`display:none`，于是那一块**同时**退出 tab 序与无障碍树。

> 纯 CSS 模糊 / 毛玻璃**不是门禁**。模糊后的内容仍然可以 Tab 到，屏幕阅读器仍然完整朗读它
> （W3C/MDN/WHATWG 一致：移出无障碍树的只有 `hidden` 属性、`display:none`、`visibility:hidden`，
> `filter` 与 `backdrop-filter` 都不在其中）。而最常见的补救 —— 给被模糊的外层加
> `aria-hidden="true"` —— **本身是 WCAG 4.1.2 Level A 失败**，因为该属性不得用于可聚焦元素
> 及其任何祖先，而设置面板必然含 input/button（W3C ACT 规则 6cfa84、axe-core
> `aria-hidden-focus`，impact Serious）。

所以毛玻璃只用在**解锁卡片自身**的观感上，不拿它冒充门禁；这一块也不标 `aria-modal`
—— 它是页面区域而不是对话框，而 APG 要求只有在「代码真正阻断所有用户与外部内容的交互」
**且**「视觉上遮蔽外部内容」同时成立时才可声明 modality。

解锁后分**五个子页**（客户端密钥 / 路由规则 / GHCP 账户 / 上游凭据 / 面板密码），每页只做
一件事，**每页的操作反馈落在自己的区域内** —— 之前四块共用顶部一行提示，在底部操作时
结果滚到屏幕外，看起来像什么都没发生。

**页面脚本不持有任何凭据。** 解锁是把密码换成一个 **HttpOnly 会话 cookie**，此后由浏览器携带：

```
POST /admin/session  {"password": "…"}        → Set-Cookie: lb_panel=…; HttpOnly; SameSite=Strict; Path=/admin
GET  /admin/session                           → {authenticated, expires_in, password_configured, password_source}
DELETE /admin/session                         → 服务端失效 + 清浏览器那一份
```

- 密码来自**存储的覆盖值**或 `LB_DASHBOARD_PASSWORD`（env，来自 Secret），见下一节。**两者都没有或都短于 12 字符 → 面板登录整体不可用**，绝不退化为无密码放行 —— `/admin/*` 由没有 OAuth 的那个 ingress 提供，对公网可达
- 会话是服务端进程内的 **30 分钟滑动窗口**（任何操作续期），**重启全部失效**，多副本之间不共享（换 Pod 要重新解锁，不是故障）
- **刷新页面保留会话** —— 这正是目的：过去刷新就等于重输。页面加载时先问 `GET /admin/session`，已认证就直接恢复四块，全程零交互
- 锁定会**在服务端一并失效**并清空已读出的内容（掩码、指纹、账户名本身就是会话授权才能看到的）。只清浏览器那一份会留下一个仍可用的会话
- 连续 5 次失败按来源**指数退避**（30s 起翻倍，有上限）。退避期内**正确密码也拒**且不区分原因，否则它就成了一个旁路的密码预言机。退避**只计密码路径**：让它挡住高熵的 API Key 等于把一次猜测攻击变成自伤
- `Secure` 只在 https 下设置，否则本地 http 调试根本拿不到 cookie
- `SameSite=Strict` + `Path=/admin` + 同源 XHR 已覆盖 CSRF，不再加 token

**LB API Key 始终可以换会话**，而它**也换成同一种会话**而不是留在页面里 —— 否则「脚本不持凭据」只在一半情况下成立。安全上等价：能出示有效 key 的人本来就能直接调用每个 admin 端点。配了密码时它收在「改用 LB API Key 解锁」折叠项里；没配密码时它是唯一入口，所以默认展开。这条路径也是**存储的密码被忘记之后唯一的回头路**。

这一页在入口层已由 oauth2-proxy 保护（`lb.zeno.ink/stats*`，限定单个 Google 账号），但 `/admin/*` 走的是另一个没有 OAuth 的 ingress —— 所以凭据仍然必要，不能把「页面已登录」当作授权。

**上游凭据不在此列。** `credSecret` 保持提交后立即清除的单次请求契约 —— 会话是便利，上游凭据是秘密。

<a id="changing-the-panel-password"></a>

### 在页面上改面板密码

「面板密码」子页可以设置或更改它。**存的是单向验证器，不是密码。**

```
PUT    /admin/panel-password  {"current_password": "…", "new_password": "…"}
DELETE /admin/panel-password  {"current_password": "…"}     → 清除覆盖值，回落 env
```

存储形态是 `pbkdf2_sha256$<迭代数>$<盐>$<哈希>`，单行表 `panel_password`（`id` 恒为 1，
惰性创建、无需迁移；`release_probe.py` 对它沿用「不存在放行、存在严格匹配」）。

为什么是哈希而不是像上游凭据那样 AES 封装：上游凭据必须**重放**给上游，所以必须可解开；
面板密码只需要**验证**。于是哈希严格更好 —— **即使 `LB_CREDENTIAL_KEY` 泄露也推不出密码**，
而且这条路径压根不依赖那把密钥（没配它也能改密码）。代价要写明：**忘记即不可恢复**，
回头路是用一把有效 LB API Key 解锁后重设。

授权与边界：

- **会话不够**，还要出示**当前密码** —— 会话可能是用 API Key 换来的，那并不证明调用方
  知道密码。首次设置（两边都没有密码时）用一把有效 API Key 代替
- 错误的当前密码走**登录那同一套**按来源退避。否则这个端点就是一条绕过登录退避的
  猜测通道
- 新密码下限 12 个**字符**（非字节）。`kubectl describe secret` 报的是字节数，多字节密码
  会在那边看着够长却被判为不合格
- 成功后**其他会话立即失效**，调用方自己那条保留（刚改完就被踢出去没有意义）
- 写失败则旧密码仍然有效，不假装成功
- 响应体、响应头与日志都不回显密码或验证器；它也不进 `LB_SETTINGS`，所以不会出现在
  `/config/effective` 里

**优先级沿用那条降级规则**：存储是 overlay，`LB_DASHBOARD_PASSWORD` 是基线。有覆盖值就
用它；overlay 缺失或未加载回落 env；**两者都没有则登录整体不可用而不是放行**。`GET
/admin/session` 的 `password_source` 回答「现在用的是哪一份」（`store` / `env` / `none`），
只回来源不回内容。

> **多副本边界。** 策略存储**只在启动时加载，没有周期刷新**。所以页内改密码只对**当前
> 副本**即时生效，其余副本在重启前**仍然接受 env 里的旧密码**。这与其他每一层策略同
> 性质，但密码的后果更尖锐 —— 改完请同步更新 Secret 再滚动重启。当前为单副本部署。

<a id="per-key-provider-allowlist"></a>

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

<a id="entering-upstream-credentials"></a>

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

<a id="routing-rules"></a>

### 路由规则：模型 → Provider → GHCP 账户

`PUT /admin/model-routing/{model}` 设置单个模型的规则。两段都可选：省略 `providers` 只改账户，省略 `copilot_accounts` 不动账户。**没有规则的模型维持原本行为。**

两段都在**任何写入之前**校验完。第三段打错一个账户名却让 provider 那半边已经生效，是最难排查的一种半应用。

Dashboard 的编辑器分三步，并支持点已有规则那一行直接载入（改一条规则不必重填整张表单）。
三步里有两处是刻意做成「结构可见」而不是靠提示文案的：

**第二步把完整的生效顺序画出来**，两个 provider 始终都在，没被提前的那个标着
「fallback · 保留」。这样「偏好只重排、永不筛除」是控件本身展示的事实，而不是说明文字里的
一句主张。外部先例是 Vercel AI Gateway，它把 `order`（只定顺序，未列出的仍在候选里、只是
靠后）与 `only`（真筛除）做成两个互斥字段名，把语义写进 API 表面；OpenRouter 同样强调
「优先级列表是偏好，不是 allowlist」。

<a id="candidate-matrix"></a>

**第三步是候选矩阵**，一行一个账户，三列：

| 列 | 作用域 | 写哪个端点 |
|---|---|---|
| 账户启用 | **对所有模型生效** | `PUT /admin/copilot/accounts/{account}` |
| 本模型选中 | **仅当前模型** | `PUT /admin/model-routing/{model}` |
| 当前结果 | 计算值，带 [排除原因](#exclusion-reasons) 与「去哪改」 | — |

把两层放在一张表上是运维视角的选择（一次看完「这个模型谁在服务」），但它有一个真实
风险：**左边那一列是全局的**，改了影响所有模型，却和右边那列长得一样。缓解不靠说明文字：

- 列头各带作用域徽标（「对所有模型」vs「仅 `<模型名>`」）
- 确认框**逐行列明**哪些是全局改动、哪些仅限本模型
- 勾一个服务不了该模型的账户时，那一行直接标「勾了也不会服务该模型」

**保存是两步写，分别回报。** 两个端点做不到原子提交，就不假装做得到：先写全局启用态，
再写模型规则。第一步失败则不做第二步；第一步成功而第二步失败时明说「账户启用态已生效、
模型规则未写入」。

**生效候选在保存前就算出来**（`生效候选 = 勾选 ∩ 能服务`），空集立即告警。数据来自
`GET /admin/model-routing/{model}`：服务端给出每个账户「能否服务该模型」，客户端只做一次
平凡交集，不复制策略逻辑。之前这件事只能等写入响应才知道 —— 也就是规则已经上线之后。
反面教材是 one-api：候选集为空时只抛一个不区分原因的 503，错误文案只回显 group 与 model，
从不说明是哪个谓词把候选排除了。

#### 第一段：provider 优先序

**顺序即偏好**，不去重也不重排，重复项会被拒。

**偏好只重排，不筛除。** 列出的 provider 提到前面，未列出的按默认顺序（`copilot` → `azure_openai`）接在后面，所以 fallback 能力完整保留。provider 维度的筛除是 per-key allowlist 的职责；两者叠加后的候选是：

```
(能服务该模型的 provider) ∩ (该 key 允许的 provider)，先按偏好排序，再按默认顺序
```

**一份偏好，两个协议面各自过滤。** 偏好里的 `databricks` 不参与 OpenAI 风格入口的顺序（它服务 `/v1/messages`），`azure_openai` 反过来不参与 messages 面的顺序（它说 Azure OpenAI 形状）。所以 Dashboard 有**两个**顺序下拉，各用自己的选项来源：

| 下拉 | 选项来源 | 可选 |
|---|---|---|
| OpenAI 风格入口 | `default_openai_order` | GHCP、Foundry（`azure_openai`，gpt 那一面） |
| Anthropic Messages 面 | `default_messages_order` | ADB、GHCP、**Foundry Claude**（`azure_anthropic`） |

此前只有前一个下拉，于是唯一的「Foundry」是 `azure_openai` —— 给 Claude 选它**毫无作用**（messages 面会把它过滤掉），而真正想选的 `azure_anthropic` 根本无法命名。一个没有选项的面就是配不了的面。

两段合成**一份** `providers` 时有三条规则：

- 两面都选「保持当前顺序」就**省略 `providers`**，绝不重写已存的偏好；
- 只编辑一个面时要带上另一个面**已保存**的选择 —— `providers` 是整份替换，只发被编辑的那一段会把另一面悄悄打回默认；
- 两面都有选择时以 `copilot`（两面唯一共享的 provider）为锚，把 OpenAI 段插进 messages 段，否则为一面指定首位会连带挪动另一面的 `copilot`。

OpenAI 风格面只有两个 provider，所以指定首位就是完整排序。Messages 面有三个，因此多一个「第二跳」下拉：**选定前两位即确定全序**（第三位只剩一个），不需要拖拽排序控件。第二跳的选项按第一跳动态生成并排除它本身 —— 重复的 provider 会被 `ordered_providers` 拒（400），而那是一个本可以不发生的失败。不选第二跳就回到「只指首位、其余按默认序补齐」，写入的仍然只有一个 provider。

`Foundry Claude → GHCP → ADB` 因此是可表达的：第一跳 Foundry Claude、第二跳 GHCP。顺序偏好是运行时 overlay，**保存即生效，不需要重启**（只有端点构建才需要）。

**规则按模型身份匹配，不按拼法。** 同一个 Claude 模型有三种拼法（`databricks-claude-opus-5-5` / `claude-opus-5.5` / `claude-opus-5-5`），`known_models` 里它们是三个条目，但共用**一条**规则 —— 规则命名的是模型，不是拼法。实现是 `key_policy.rule_key()`：精确拼法优先（最不意外，也让身份匹配之前写的规则继续按原样工作），否则按 `claude_model_identity.canonical()` 归并；非 Claude 模型没有 canonical 身份，仍只精确匹配（没有任何依据说 `gpt-5.4` 与 `gpt-5-4` 是同一个模型）。写入会把其他拼法下的同一条规则**收敛掉**，所以存储里一个模型只有一条规则，Dashboard 不会把同一个模型列两次、给两个互相覆盖的答案。

这一条修掉的不只是体验：`/v1/messages` 的失败转移在调用池**之前**就把模型改写成了各 provider 的拼法，所以 `supports()` 看到的是 `claude-opus-5.5`。按 ADB 或 Foundry 拼法保存的 GHCP 账户限定因此永远查不到，**这一层静默地不再做减法**。

#### 第二段：限定 GHCP 账户

`copilot_accounts` 把该模型限定到指定的 GHCP 账户，**空列表或 `null` = 不限定**（这是基线，也是撤销限定的唯一表达方式）。账户名必须是已配置的 Copilot 端点，否则 400 `unknown_copilot_account` —— 写错的名字静默生效就等于把该模型的 GHCP 候选清空，而界面上看着像配好了。

这一段**是筛除**，但筛的是**同一渠道内的哪个账户**，不是哪个 provider，所以上面那条「只重排不筛除」不受影响，provider 级 fallback 完整保留。

它与按账户的 `allowed_models` 是**交集**：前者把账户从模型上摘掉，后者把模型从账户上摘掉，任一侧成立即移出候选。排查某账户为何不接某模型时两处都要看 —— Dashboard 的「生效候选」列就是为此存在，它由服务端按 `CopilotEndpoint.supports()` 算出（客户端算不出来，需要同时拿到 capability、`allowed_models` 和启用态）。

选中的账户恰好都服务不了该模型时，GHCP 候选为空，请求落到 Foundry —— 与那些账户全部熔断同效，不是错误。响应的 `copilot_candidates_empty` 会标明这一点，Dashboard 保存后给出警告但**不阻止**：那是运维的决定。

**会话亲和不受影响。** 筛选加在 `CopilotEndpoint.supports()`，那是所有候选判断的唯一收口，含亲和用的「配置态合格集合」。限定让这个集合变小但仍然稳定（它只随策略变，不随熔断变），所以同一会话在同一策略下仍落同一账户。注意换一组限定会改变集合大小，亲和哈希因此重新分布 —— 这是既有哈希的固有性质，也是为什么它是运维的显式决定而非自动行为。

存储用独立的 `model_copilot_account` 表而不是给 `model_provider_preference` 加列：后者在生产已经存在，加列是真迁移；新表由首次使用惰性创建，零迁移。发布预检校验它的列与主键（主键丢了会让两行声明同一个模型，限定变成取决于行序），表不存在时放行。

### `claude-*` 走 GHCP：逐模型 opt-in

`claude-*` 默认仍被 OpenAI 风格入口拒绝（400）。给**某个具体模型**配置指向 `copilot`/`azure_openai` 的偏好后，该模型在这些入口放行 —— Copilot 原生服务它的 Responses/Chat 形状。这是逐模型开关，不是整族开闸：开放 `claude-opus-5.5` 不会同时开放 `claude-sonnet-5.5`。清除偏好即恢复拒绝。

**作用范围：这一段只放开 OpenAI 风格入口。** 这里早先写着「`/v1/messages` 仍然只走 Databricks，需要双向适配器，所以 Claude Code 这类客户端用不到这条通路」—— 2026-10-17 的实测把它**推翻**了：GHCP 自己的 `/models` 对每个 claude 条目声明 `supported_endpoints: ["/v1/messages", ...]`，Foundry 的 `POST /anthropic/v1/messages` 返回标准 `type: "message"`，所以**不需要合成器**，而走 `/v1/messages` 的客户端恰恰是三条路径的主要受益者。本段讲的 opt-in 与那件事无关：它只决定一个 `claude-*` 模型能不能从 `/v1/responses`、`/v1/chat/completions` 进来。messages 面的优先序见上面「一份偏好，两个协议面各自过滤」。

**计价必须分渠道**，否则费用会算错。区分依据是记录的模型名：Databricks 记映射后的 `databricks-claude-*`，Copilot 记自己的 id（裸 `claude-*`）。前者按 Anthropic 公开 API 价与「input 不含 cache」口径，后者按 `copilot_pricing` 分档价与 inclusive-input 口径。2026-10-12 核对生产 `usage_daily`：14 条 Claude 行全部带 `databricks-` 前缀、零条裸 `claude-*`，所以这项修正不改变任何既有行的计价基准。

### GHCP 多账户：启用与按账户勾选模型

账户凭据不进配置。`endpoints[]` 里只写 `name`，token 会按 `copilot-auth-<name>.json` 从 device-flow 缓存目录解析（AKS 上由 Secret 挂载），`token_source` 记为 `file`，因此 `/admin/copilot/reload` 可以重读它 —— 写死在配置里的明文（`literal`）做不到。

每个账户有两层模型信息：

- **capability** —— 由 `POST /admin/copilot/accounts/{account}/discover` 向上游发现并存快照，只读、带时间戳。**它是时点发现，不是能力认证**，上游随时会变
- **allowed_models** —— 运维在 capability 内勾选的子集，`null` 表示不额外限制

有效可用集 = `capability ∩ allowed_models`。capability 本身就是外层边界：上游说某账户没有该模型时，路由过去没有意义，所以即使没设 allowlist 也不会落过去。从未发现过 capability（空）时不据此限制，避免把「还没发现」当成「什么都没有」。写入时超出 capability 的模型被拒，所以**不可能**把账户配成服务它没有的模型。

停用账户用启用开关，不要靠清空勾选 —— 空列表会被拒，否则两种语义会混淆。

<a id="ship-inert-then-enable"></a>

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
