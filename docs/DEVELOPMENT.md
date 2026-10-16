# 开发与验证

Author: Zeno Ren

Python 3.11+，推荐 3.12；前端为独立 HTML、CSS 和原生 JavaScript，没有 React/npm 构建链。Node.js 仅用于 UI 行为回归，Chart.js 使用固定版本及 SRI 校验。

## 模块边界

| 范围 | 主要文件 |
|---|---|
| HTTP 路由、provider 代理与负载均衡 | `main.py` |
| 准入、请求预算、生命周期 | `admission.py`、`request_budget.py`、`gateway_lifecycle.py` |
| 请求与发送观测、安全诊断、阶段时间 | `request_telemetry.py`、`safe_diagnostics.py`、`request_timing.py` |
| 协议结果、Chat 适配、上游错误体 | `response_semantics.py`、`chat_adapter.py`、`upstream_body.py` |
| 参数与渠道能力 | `effort_compat.py`、`model_capabilities.py` |
| 运行时策略存储（渠道 allowlist、模型偏好、模型→账户、账户启用与 allowed_models） | `key_policy.py` |
| 上游凭据封装、存储与 device flow | `provider_credentials.py` |
| 面板密码与会话（进程内，不落存储） | `main.py` 的 `PANEL_*` 段 |
| 用量与 Copilot 估算 | `usage_store.py`、`copilot_pricing.py` |
| 发布前的存储 schema 门控 | `release_probe.py` |
| Dashboard 与构建身份 | `dashboard.html`、`build_metadata.py` |
| 报告、实验、构建与受控发布 | `operations/` |

策略层有五层、互相叠加，改动前先读 [策略层总览](CONFIGURATION.md#policy-layers-overview) —— 那里有「谁做减法、谁只排序、在哪收口」的地图。

## 回归

```bash
python -m pip install -r requirements-test.lock
python -m pytest tests/ -q --tb=short
node tests/test_copilot_pricing_ui.cjs
node tests/test_dashboard_ui.cjs
```

改策略或面板时，这几个文件是对应的回归入口：

| 改了什么 | 对应测试 |
|---|---|
| 渠道 allowlist、降级规则 | `tests/test_key_policy.py`（`DegradationFallsBackToTheConfigBaselineTests` 横跨五层断言） |
| provider 偏好、模型→账户、`/v1/models` 目录一致性 | `tests/test_model_routing.py` |
| 账户启用三态、capability 与 `allowed_models` | `tests/test_copilot_accounts.py` |
| 面板密码、会话 cookie、退避、页内改密码 | `tests/test_panel_session.py` |
| 排除原因（reason 闭集）与它和 `supports()` 的一致性 | `tests/test_key_policy.py::ExclusionReasonTests`、`tests/test_model_routing.py::SupportReasonTests`（组合式断言，防两份逻辑漂移） |
| 候选诊断端点、候选矩阵的数据源 | `tests/test_model_routing.py::CandidateDiagnosticEndpointTests` |
| 「已启用但闲置」与「已停用」是不同状态 | `tests/test_copilot_accounts.py::EnabledButIdleIsItsOwnStateTests` |
| 凭据封装、device flow | `tests/test_provider_credentials.py` |
| 任何新路由或鉴权变更 | `tests/test_api_surface_contract.py`（双向核验 `CLAUDE.md` 的 API 表） |
| 存储 schema | `tests/test_release_schema.py`（「不存在放行、存在严格匹配」） |
| Dashboard 行为 | `tests/test_dashboard_ui.cjs`（按 `// ---------- X` 注释切片取 JS，改动要保留这些标记） |

MySQL 的 opt-in 测试必须使用一次性隔离实例，不指向生产。发布工具的 Kind/中断/接管场景有独立环境与证据要求，见 [发布测试说明](../tests/release/README.md)。不为通过 CI 削弱协议、账本或退出码断言。

## UI 预览

```bash
python tests/dashboard_preview.py
```

访问终端打印的 loopback 地址。全部渠道、账本和版本均为合成数据，没有生产数据库或上游调用；模拟 DELETE 不删除任何数据。`?scenario=empty`、`error`、`unknown`、`modified` 用于未配置、请求失败和版本状态验收。

界面采用语义 CSS 变量、清晰的信息层级、键盘导航和 reduced-motion 处理，参考 BoardUI 设计规则并保留现有技术栈；不将其称为 BoardUI React 组件实现。核验深浅主题、表格溢出、四个视图、手动/暂停刷新、版本链接和失败提示。Chart.js 无法加载时仍显示表格。

版本 API `/version` 仅返回提交及文件匹配状态。前端链接只由合法 40 位 Git SHA 和固定仓库地址构造；未知构建不猜版本。管理凭据只用于用户明确确认的清理请求，不持久保存或发往 GitHub。

## 镜像与来源

```bash
python -m operations.build_identity \
  --build-tag claude-lb:local \
  --platform linux/amd64
```

工具把 Git SHA、运行文件清单及本地修改状态绑定到构建。未提交代码会标注修改状态；不能靠手写源码注解伪装一致。新增运行模块时同步 `build_metadata.RUNTIME_FILES`、发布器 `APP_FILES`、Docker COPY 和打包验证。

本地运行示例：

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  --env LB_API_KEY --env DATABRICKS_TOKEN_1 \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/usage_data:/app/usage_data" \
  claude-lb:local
```

先创建可写的数据目录，并传入配置中所有实际引用的环境变量；不要挂入与另一个 writer 共用的 JSON 存储。Copilot 缓存使用单独只读凭据挂载，细节见 [配置指南](CONFIGURATION.md)。

`.wheelhouse`、`.release-wheelhouse` 和 `.test-wheelhouse` 是被忽略的离线构建/测试输入。已有构建确实使用过这些目录，不能仅因体积大就当作无用文件删除。普通 Python/pytest 缓存可以再生；私有发布目录包含备份和恢复材料，保持独立、限权且不进入 Git。

## 改动前值得知道的几处收口与约束

都是真实踩过的，不是预防性清单。

**给 Copilot 候选加条件只能改 `CopilotEndpoint.supports()`。** 它是唯一收口，
`can_handle` / `supports_model` / `/models` 目录 / 会话亲和的「配置态合格集合」全走它。
绕过去的后果见 [收口说明](CONFIGURATION.md#supports-is-the-choke-point) —— 目录曾因此
宣传过一个实际没有候选的模型。注意条件本体现在住在 `_copilot_support_reason()` 里，
`supports()` 只是「没有排除原因」—— 收口没变，新条件仍然加在那个函数里。

**诊断与判断必须是同一份代码，而且方向是「reason 为主、布尔为派生」。** Dashboard 要说出
是哪一层摘掉了某个账户；把那个解释另写一份，漂移的必然是诊断，而诊断恰好是被人当真的
那一份。所以 `supports()` ≡ `reason is None`，`key_policy` 的两个布尔读者也委托给各自的
reason 读者。守卫是对所有 (账户 × 模型 × api_type × 存储状态) 组合断言两者永不相左 ——
加新 reason 时那组断言会一并覆盖它。原因必须进
[闭集](CONFIGURATION.md#exclusion-reasons)，不要返回自由文本。

**`CopilotEndpoint` 新字段要追加在 dataclass 末尾。** 文件里有一行既有注释写明了这点：
插在中间会让 `main.CopilotEndpoint('name', '', 1, [], {...})` 这类位置参数构造静默错位，
测试里就有这种写法。

**新增运行时文件要同步三处并重建发布控制器镜像**：`build_metadata.RUNTIME_FILES`、
`operations/release/model.APP_FILES`、`Dockerfile` 的 COPY。`model.py` 对 `APP_FILES`
做**严格集合相等**校验，少一个多一个都会让发布红。只在 `main.py` 里加代码则不涉及这三处 ——
这也是面板会话逻辑没有拆成独立模块的原因。

**秘密不要进 `LB_SETTINGS`。** `/config/effective` 把整个 settings 字典返回给任何持有效
key 的调用方。`LB_CREDENTIAL_KEY` 与 `LB_DASHBOARD_PASSWORD` 都是直接 `os.environ.get`，
各自有测试断言它们不出现在该端点里。

**长度下限判字符数，不是字节数。** 面板密码的 `len(pw) >= 12` 是 Python 字符数，而
`kubectl describe secret` 报字节数；多字节密码可能看着够长却被判为未配置。
排查入口见 [排障 19](TROUBLESHOOTING.md)。

**新表用新表，别给既有表加列。** 生产里的策略表是惰性创建的，一旦存在，加列就是真迁移。
`model_copilot_account` 与 `panel_password` 之所以独立成表而不是挂到既有表上，就是
为了零迁移。`release_probe.py` 对每张表沿用「不存在放行、存在则严格匹配」，加表要同步它。

**要验证的秘密存哈希，要重放的秘密才存密文。** 上游凭据必须重放给上游，所以 AES-GCM 封装；
面板密码只需要验证，所以存 PBKDF2-SHA256 单向验证器 —— 即使 `LB_CREDENTIAL_KEY` 泄露也
推不出密码，而且这条路径不依赖那把密钥。代价要在 UI 和文档里写明：**忘记即不可恢复**。
PBKDF2 是同步 CPU 操作（六十万轮 HMAC），**必须 `asyncio.to_thread`**，直接算会卡住事件循环。

**门禁用 `hidden`，不要用模糊，也不要给容器加 `aria-hidden`。** 纯 CSS 模糊后的内容仍可
Tab 到、仍被屏幕阅读器完整朗读；而给含可聚焦控件的外层加 `aria-hidden` 是 WCAG 4.1.2
Level A 失败（axe `aria-hidden-focus`，impact Serious）。`display:none` 一次解决两件事。
毛玻璃可以用在解锁卡片自身，但别拿它冒充门禁；页面区域也不要标 `aria-modal`，那是对话框
的声明，APG 要求「代码真正阻断交互」与「视觉遮蔽」同时成立才可声明。

**Dashboard 预览里有 `/admin/*` 的合成夹具**（`tests/dashboard_preview.py`）。改了这一页
就顺手在那边走一遍：锁定 → 解锁 → 候选矩阵 → 两步保存 → 改密码。夹具故意包含一个
「已启用但闲置」的账户和一个未发现能力的 inert 账户，那两个状态只有在那里才看得到。
注意预览的会话是进程内全局的、不校验 cookie —— 它不是鉴权测试。

## 文档维护

README 只保留入口、快速开始和关键边界；操作细节归到对应指南，阶段证据归到 `docs/reviews/`。新增路由同步 [完整 API 表](../CLAUDE.md#api-端点)，已有测试会核对文档与真实路由及鉴权行为。

API 表的「认证」列目前有**四种**取值，加路由时要选对：`需要`（有效 LB API Key）、
`不需要`、`可选`（带了就必须有效、不带放行）、`密码`（凭据是面板密码而非 key）。
`tests/test_api_surface_contract.py` 会对每种做行为核验 —— 把密码端点错标成「需要」，
会让做安全复核的人以为轮换 API key 就能关掉那扇门。

生产发布是独立验收层。已合并 GitHub、镜像构建通过、Pod Ready 和完整业务发布成功分别记录，参见 [AKS](AKS.md) 与 [发布流程](RELEASE_TOOL.md)。
