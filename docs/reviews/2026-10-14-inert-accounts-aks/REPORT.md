# 账户先落地后启用 + 凭据录入 AKS 发布回执

Author: Zeno Ren

2026-10-14 将 `49c142b0c88ba559d9c54ccd756d3da2365c9588` 发布到 AKS，一次带上线上落后的六次合并（[#28](https://github.com/ZenoRewn/databricks-claude-lb/pull/28)、[#30](https://github.com/ZenoRewn/databricks-claude-lb/pull/30)、[#31](https://github.com/ZenoRewn/databricks-claude-lb/pull/31)、[#32](https://github.com/ZenoRewn/databricks-claude-lb/pull/32)、[#33](https://github.com/ZenoRewn/databricks-claude-lb/pull/33)、[#34](https://github.com/ZenoRewn/databricks-claude-lb/pull/34)）。线上此前停在 `c318724`。

**本次执行两轮：第一轮在 preflight 阶段 403 中止，第二轮成功。** 第一轮的原因是我自己在发布前准备时漏了一步权限，不是应用缺陷，详见下文。

第二轮回执 **succeeded**，北京时间 **14:32:48.461–14:33:38.084** 完成维护并恢复流量，**49.623 秒**，十项 action 全部 verified，三个业务探针通过且 `persistence_verified=True`（记录 3 条用量事件），`cleanup_errors` 为空。

## 第一轮为什么 403

控制器在 `zeno-apps` 的 Role 对 `secrets` 的 `get` 限定了 `resourceNames: ['claude-lb-copilot-auth']`——这是有意的最小授权。

而我在 2026-10-13 的发布前准备里给 Deployment 加了第二个 Secret 引用 `claude-lb-credential-key`（AES-GCM 封装密钥，以 `secretKeyRef` 注入）。preflight 的 `_verify_references()` 要逐个读取模板引用的 ConfigMap 与 Secret，用它们的摘要确认「计划与执行之间配置或凭据没有被改动过」。新 Secret 不在白名单里，于是 403。

**我当时在私有准备记录里把这一步写成了「已就位，不用再动」，那是错的**：密钥接好了，但控制器读不到它。`plan` 阶段没暴露这个问题，因为 plan 由我的用户身份执行，权限不同。

生产完全未被改动，有据可查：镜像未变、Deployment `generation` 仍是 115（dry-run PATCH 不递增）、Pod 未被替换且重启数 0、两个 Service 的 selector 无发布门标签残留、无遗留校验 Pod。回滚时 `cleanup_errors` 里那条 `ApiException` 是同一个 403，而当时本来无物可清。

### 修法

把 `claude-lb-credential-key` 加进该 Role 既有的 `secrets` 规则，**只加一个名字**，server-side diff 只有一行新增。

用 `kubectl apply` 而不是 JSON patch：这个 Role 由 apply 管理，patch 会让 `last-applied-configuration` 失配，下一次 apply 就会把改动吞回去。所以我从它自己的 last-applied 重建对象，只改那一处再 apply。

授权范围已逐项复核：两个具名 Secret 可读，未具名的 `get secrets` 仍是 `no`，无关 Secret 名也是 `no`。控制器只拿到摘要——`_verify_references` 比较的是 `digest({data, binaryData, type, immutable})`，不留明文。

## 本次发布改变了什么

### Copilot 账户「先落地、确认后再启用」

`github_copilot.endpoints[]` 新增 `enabled` 字段。`enabled: false` 表示**部署但不投入服务**：端点照常构建、在 Dashboard 可见可操作，但在被显式启用前不进任何候选集。

这是账户唯一的先落地途径——token 来自只读 Secret 挂载，端点又只在启动时构建，所以不把账户写进配置就根本没有可供 Dashboard 操作的对象，而写进去在过去就等于立刻投入服务。

启用态因此是三态（未启用／已启用／已停用），存储层把 `enabled` 持久化为可空列（`NULL` = 尚无决定）。**发现能力不等于启用**：`discover` 是对上游的只读查询，它会建记录但不写 `enabled`，否则一次「刷新能力」就会把 inert 账户推上线。

启用一个没有 `models`、也从未发现过 capability 的 inert 账户会被拒（400 `capability_unknown`）：它没有任何边界，启用即通配，会成为每个模型的候选，包括自己席位没有的。

### 一条统一的降级规则

早先我把上述两点写成「刻意与项目 fail-open 姿态相反的例外」。那个框架是错的，已删除。全项目只有一条规则：**overlay 缺失或未加载时回落到配置基线**。既有三层（按密钥限渠道、模型→provider 优先序、`allowed_models`）看着像 fail-open，只是因为它们的基线恰好宽松。账户启用态的基线可以是 `enabled: false`，于是同一条规则给出「保持不投入服务」。

`tests/test_key_policy.py::DegradationFallsBackToTheConfigBaselineTests` 把四层放在一个类里断言，任何一层改成真正的 fail-open 都会红。

### 凭据录入与 Foundry v1

同批上线的 #30–#33 带来按密钥限渠道、Dashboard 密钥身份与轮换、Copilot 账户能力发现、per-model provider 优先序、运维录入上游凭据（AES-256-GCM 封装），以及 Azure 切到 v1 GA 路径。

发布后状态：`[Credentials] backend=mysql state=loaded key=present records=0`——**封装密钥可读，录入功能已可用**，尚未录入任何凭据。`provider_policy` 同为 `loaded`/mysql。四张表（`provider_credential`、`copilot_account_policy`、`model_provider_preference`、`api_key_policy`）都由首次使用惰性创建。

`copilot_account_policy` 在生产从未存在过（`key_policy.py` 不在 `c318724` 里），所以把 `enabled` 改为可空列是首次创建而不是迁移。

## ConfigMap 的顺序是反过来的

我原先记的顺序是「先 apply ConfigMap，再发布」。动手前复核时发现这会开一个窗口：**线上镜像 `c318724` 不认识 `enabled` 字段**，它会照常解析那 4 个端点、忽略 `enabled: false`，并因为没有 `models` 而当成通配端点。如果旧 Pod 在发布前因任何原因重启，4 个 business 账户就会带着通配上线，而 `gpt-5.4` / `gpt-5.6-luna`（两个业务探针模型）恰好是它们席位没有的。

所以改为**先发布、后 apply**。代价是账户要到之后某次滚动重启才出现在 Dashboard——它们本来就是 inert、没有任何东西依赖它们。

ConfigMap 变更：端点从 1 个变 5 个，追加的四个全部 `enabled: false` 且**不含 token**（只写 name，运行时按 `copilot-auth-<name>.json` 从挂载的 Secret 解析）。apply 前逐字段校验过：现有 `ZenoRewn` 条目逐字节未变、其它配置段完全相同、未引入任何凭据材料。

**本次发布不改变任何路由行为**——运行中的 Pod 仍只有 1 个 Copilot 端点。

## 构建

`linux/amd64`、`dependencies-offline`、工作树干净（`source_tree_dirty=false`、`out_of_tree=false`），镜像内 `source_attested=true`、23 个运行时文件，控制器侧 21 个 `APP_FILES` 严格集合校验通过。

本机到 PyPI 文件 CDN 不通，`.wheelhouse` 经企业代理索引填充，所以三个新依赖的轮子**逐个与 PyPI JSON API 公布的 sha256 对过**：`cryptography==50.0.1`（`ff838d62…`）、`cffi==2.1.1`（`c1453022…`）、`pycparser==3.0`（`b7274141…`）。推送前在镜像内实测 `seal`/`unseal` 往返与 `build-info.json` 的 attestation。

## 发布后核验

公网 `/version` 报 `49c142b`、`verification: matched`；Pod 重启数 0；`accepting=200`；`lb_usage_pending_events=0`、`lb_usage_backend_ready=1`；两个 Service selector 干净。

独立业务抽样（走公网，发布流程之外）：`databricks-claude-opus-5-5` 与 `databricks-claude-sonnet-5-5` 均 HTTP 200，上游模型名分别为 `claude-opus-5-5`、`claude-sonnet-5-5`，`stop_reason=end_turn`。

业务探针用 `databricks-claude-opus-5` 而非 `opus-5-5`：后者上游容量间歇受限，2026-10-10 那次回滚就是把它当强制门槛造成的。

## 未完成

- 4 个 Copilot 账户要到下次滚动重启才出现在 Dashboard；启用各自还需「刷新能力 → 勾选 → 启用」再加一次重启才服务流量
- **Azure Foundry 凭据仍然全路径 401、历史请求数为 0**。#33 的 v1 GA 切换不解决这件事，凭据本身得换；换好后可用 `POST /admin/credentials/azure_openai/{account}/validate` 确认
- per-key 的模型→provider 优先序、per-key Copilot 账户选择、以及把路由决策在请求准入时冻结（让 sticky 从「靠 opaque-state pin 偶然保住」变成结构保证）均未实现

机器可读回执见 [receipt.json](receipt.json)。
