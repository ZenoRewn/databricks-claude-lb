# 路由规则三段式 + 面板解锁 AKS 发布回执

Author: Zeno Ren

2026-10-14 将 `7229327412e0442d2f5f1daff9bccea9144f05a4`（功能提交 `abc0009`，[#36](https://github.com/ZenoRewn/databricks-claude-lb/pull/36)）发布到 AKS。线上此前停在 `49c142b`。

**一次成功**，北京时间 **16:41:23.080–16:42:36.021** 完成维护并恢复流量，**72.94 秒**，十项 action 全部 verified，三个业务探针通过且 `persistence_verified=True`（记录 3 条用量事件），`cleanup_errors` 为空。

与上次不同，本批**没有触发 preflight 的两个已知坑**，这是发布前刻意复核的结果：Deployment 模板未新增任何 ConfigMap/Secret 引用（上次 403 的根因），且本批没有 ConfigMap 变更（上次那个「先发布还是先 apply」的顺序问题）。控制器 Role 对模板引用的三个对象覆盖完整，已在 plan 之前逐项确认。

## 本次发布改变了什么

### 「密钥与渠道」页：解锁一次，四块自动列出

这一页原来有四块、每块一个「读取」按钮，而输入框在每次操作后被清空 —— 读一下账户、改个勾选、再保存要输三次 key。现在输一次 key 点「解锁」并发读出四块，之后每次写入复用同一把 key，成功后自动重读所属那块；某一块失败不拖垮其余三块。

**这是契约变更，必须写明**：LB API Key 在解锁期间**驻留页面内存**，而非只用于当次请求。这不是改 UI 能绕开的 —— `/admin/api-keys` 需要鉴权而 `/stats/dashboard` 本身无鉴权，所以「直接列出密钥」只有这一条路。边界没有放宽：不写入 localStorage / sessionStorage / cookie、不进 URL、刷新即失效、可手动锁定、闲置 10 分钟自动锁定，锁定会同时清空已读出的掩码、指纹与账户名 —— 那些本来就是这把 key 授权才能看到的东西。从源重读客户端密钥后自动锁定，因为那次轮换可能正好作废手里这把。

代价说清楚：10 分钟内离开座位且未锁定，等于把这些运维端点交给能碰这台机器的人。

**上游凭据不在此列。** `credSecret` 保持提交后立即清除 —— 驻留 LB key 是便利，上游凭据是秘密。

### 路由规则：模型 → Provider → GHCP 账户

三块里的路由配置收成一行的下拉表单。其中第三段是**新增的一层策略**：per-model 的 GHCP 账户选择此前不存在，只有反向的 `allowed_models`（账户 → 允许哪些模型）。从模型侧表达一次意图不该要求去改每个账户的 allowlist，那会连带动到别的模型。两者取**交集**。

这一段**是筛除**，但筛的是同一渠道内的哪个账户，不是哪个 provider，所以既有的「偏好只重排不筛除」不受影响，provider 级 fallback 完整保留。筛选加在 `CopilotEndpoint.supports()` —— 所有候选判断的唯一收口，含会话亲和用的「配置态合格集合」—— 所以**会话亲和自动保持**：限定让集合变小但仍然稳定。

两段都在任何写入之前校验完。账户名打错返回 400 `unknown_copilot_account`，provider 那半边不会已经生效 —— 第三个字段打错一个字母却让前两段生效，是最难排查的一种半应用。

### 补上一个一直缺的反向操作

新增 `DELETE /admin/api-keys/{tenant}/providers`。`clear_providers` 从一开始就在存储层，但从未暴露端点，而设置接口不接受空集（允许空集等于把租户锁在门外），所以设错的渠道限制此前只能去改数据库。

### 降级规则不变，现在横跨五层断言

全项目仍然只有一条规则：**overlay 缺失或未加载时回落配置基线**。新层的基线是「不限定」。`tests/test_key_policy.py::DegradationFallsBackToTheConfigBaselineTests` 扩到五层，任何一层改成真正的 fail-open 都会红。

## 验证中发现并修掉的三个缺陷

1. **`/v1/models` 目录绕过了新层。** 它直接问 `_copilot_account_allows` 而非 `supports()`，于是会继续宣传一个「唯一被选中的账户根本服务不了」的模型 —— 客户端选了就路由失败。**这是端到端验证才暴露的**，单元测试没抓到；capability 路径与通配回落两条路径现在都有回归断言。
2. **解锁失败仍保留旧 key。** 页面显示「API Key 无效」但旧 key 还在授权写入，而那把旧 key 可能正是运维来替换的。改为解锁先丢弃旧 key，失败后处于锁定态。
3. **503「未配置」被报成「读取失败」。** 没配凭据存储的部署每次解锁都看到警告。`fetchJSON` 现在附上服务端 error code —— 「没装」和「坏了」都是 503，只有 code 能区分。

## 存储：新表，零迁移

新增 `model_copilot_account`，由首次使用惰性创建。用独立表而不是给 `model_provider_preference` 加列，因为后者在生产已经存在，加列是真迁移。

发布后在 Pod 内核对了线上表结构，与 `release_probe.py` 的期望逐项一致：`model` varchar(190) NOT NULL、`accounts` text NOT NULL、`updated_at` timestamp NOT NULL，主键 `(model)`，0 行。主键这一项值得单独校验 —— 丢了会让两行声明同一个模型，账户限定变成取决于行序。既有四张表的 schema 未动。

## 上次遗留项在本次重启中完成

2026-10-14 上一次发布后追加到 ConfigMap 的 4 个 Copilot 账户，当时需要等一次滚动重启才能出现在 Dashboard。**本次发布的重启让它们生效了**：启动日志显示 5 个 Copilot 端点，其中 4 个标记 `[INERT (enabled: false in config)]`，`/admin/copilot/accounts` 与新的 `/admin/model-routing` 都能看到它们且 `enabled=false`。

它们**仍然不接流量**。启用各自还需：刷新能力 → 勾选模型 → 启用 → 再一次重启。

## 构建

`linux/amd64`、`dependencies-offline`、工作树干净（`source_tree_dirty=false`、`out_of_tree=false`），镜像内 `source_attested=true`、23 个运行时文件齐全。**本批无新依赖**，`.wheelhouse` 直接复用；`operations/` 与 `APP_FILES` 集合未变，所以**不需要重建发布控制器镜像**。

推送前在镜像内自检：build-info attestation、23 个运行时文件、新层的往返与降级、`seal`/`unseal` 往返与 AAD 绑定（密文挪到另一行、换另一把密钥都必须失败）、`main` 可导入、新增与变更的三条路由存在。

## 发布后核验

公网 `/version` 报 `7229327`、`verification: matched`；Pod `claude-lb-74dd94f7b5-6rqgp` Ready、重启数 0；`/health/accepting` 200；两个 Service selector 干净无发布门标签残留。`[Policy] backend=mysql state=loaded`、`[Credentials] backend=mysql state=loaded key=present records=0`。9 个 Databricks、4 个 Azure、5 个 Copilot 端点加载。

新端点走公网实测：`GET /admin/model-routing` 200，返回 30 个 `known_models`、5 个账户及其启用态、`model_accounts=0`。

独立业务抽样（发布流程之外）：`databricks-claude-opus-5-5` 与 `databricks-claude-sonnet-5-5` 均 HTTP 200，上游模型名分别为 `claude-opus-5-5`、`claude-sonnet-5-5`，`stop_reason=end_turn`。抽样从 Pod 内对公网 URL 发起，所以客户端密钥没有离开集群。

业务探针用 `databricks-claude-opus-5` 而非 `opus-5-5`：后者上游容量间歇受限，2026-10-10 那次回滚就是把它当强制门槛造成的。

**本次发布不改变任何路由行为** —— 没有配置任何规则（`model_preferences=0`、`model_accounts=0`），所有模型保持默认顺序与全部启用账户。

## 我自己的两处操作失误

- **`status` 的参数记错了。** 用 `--directory` 轮询了三分钟，而该子命令要 `--release-id` —— 那三分钟全是参数错误，没有真的在读状态。集群内的状态机不受影响，发布照常完成。
- **凭据与 MySQL 环境变量的签名靠猜。** 镜像自检脚本里把 `CredentialStore` 的构造签名和 MySQL env 变量名写错了两次，都是先读源码确认后才改对。

## 未完成

- **把路由决策在请求准入时冻结**仍未实现。单次请求内由 `select_endpoint(pinned=…)` 保住同一账户、跨请求由亲和哈希保住，两者都在，但「策略在请求执行途中被改」这个竞态仍靠 `pinned` 兜而非结构保证。新层给这个竞态多加了一个可变维度，影响上限是一次重试落到别的账户。
- **Azure Foundry 凭据仍然全路径 401、历史请求数为 0**，凭据本身得换；换好后可用 `POST /admin/credentials/azure_openai/{account}/validate` 确认。
- 4 个 Copilot 账户已加载可见但 inert，启用各自还需「刷新能力 → 勾选 → 启用」再加一次重启。
- Copilot 价格刷新记了一条 `ValueError` 并保留了上一次的价格。这是既有的 fail-safe 行为，不是本次引入，本轮未深究。

机器可读回执见 [receipt.json](receipt.json)。
