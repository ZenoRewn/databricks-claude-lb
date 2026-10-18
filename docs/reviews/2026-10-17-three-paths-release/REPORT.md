# 2026-10-17 AKS 发布回执：Claude 三路径上线 + Foundry 投产配置

Author: Zeno Ren

源码 `0a9f830`，线上此前 `34acc39`。本次把 `/v1/messages` 从单一 Databricks 扩成
ADB / GHCP / Foundry 三条路径，并**第一次把 Foundry 凭据配进生产**。

| | |
|---|---|
| 源码 | `0a9f830cdd9ccc1bfdaeff3b80a0de05c6463dbc`（PR #47–#51） |
| 镜像 | `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:173cba6b…`（本地 image ID `9a1801be…`，两者是不同证据） |
| source manifest | `2a1a3c6ba7fbedb7870655ef2f681dd5f8fbb3829d5e7682b6dc0e89002095da`（**24** 个运行文件，比上次多 `claude_model_identity.py`） |
| 线上此前 | `34acc39`，generation 136，镜像 `sha256:b332e2e8…` |
| 线上现在 | `0a9f830` / `matched`，generation 140 |
| 维护窗口 | **48.0 秒**，11 个阶段全过，一次通过（未回滚、未重试） |
| 业务探针 | messages / responses / chat 三协议均 `completed` + `marker_found`，`persistence_verified: true`，3 条用量事件入账 |
| 清理 | `verified: true`，`cleanup_errors: []` |

阶段序列：`preflight → gating → pausing → draining → stopping → starting →
verifying_backend → restoring_routes → verifying_business → finalizing → succeeded`

## 本次有两个前置，与上次不同

### ① 发布控制器必须重建 —— 否则 submit 必然被拒

上次的结论是「`operations/release/*` 变了但零行为变更，不需要重建控制器」。**这次不成立**：
新增运行文件 `claude_model_identity.py` 要同步三处清单，其中一处是
`operations/release/model.py` 的 `APP_FILES`，而 `validate_plan()` 断言

```python
if set(hashes)!=set(APP_FILES) or ...: raise ValueError('Complete runtime file hashes are required')
```

且 `controller.py:50` 在**控制器进程内**也跑这条校验。于是：本地 CLI（新代码）生成 22 个文件
hash 的 plan，运行中的控制器（旧代码，21 个）读到后必然抛错。

发布前实测确认，不是推断：

| 证据 | 结果 |
|---|---|
| 运行中控制器 Pod 内 `len(APP_FILES)` | **21**，`'claude_model_identity.py' in APP_FILES` → `False` |
| 上次 `plan.json` 的 `source_hashes` 条目数 | **21**（与控制器一致，反证口径同源） |
| 新镜像内对 22 文件 plan 跑 `validate_plan` | 通过文件集合这一关（只卡在合成 plan 缺的 `legacy_bootstrap`） |
| 新镜像内对 21 文件 plan 跑 `validate_plan` | `Complete runtime file hashes are required` |

控制器新镜像 `lb-release-controller@sha256:5eb866f5…`（单平台 manifest v2），滚动后两副本
各自自检均为 `22 files / claude_model_identity: True / rev 0a9f830`，跨两个节点、均 Ready
（两副本未分别 Ready 时预检不放行，所以这步必须等）。

构建时 pip 走公网被本机网络拦截（`SSL: UNEXPECTED_EOF_WHILE_READING`，与同机 `curl` 正常
对照），改用既有的 `dependencies-offline` + `.release-wheelhouse`（19 个 wheel 对应 lock 的
19 个包）完成。

### ② Foundry 生产配置（ConfigMap 变更，不在发布工具管辖范围内）

发布工具 v1 明确不做凭据更新，所以这是一次**独立的 ConfigMap 变更**，在 `plan` 之前完成，
让 plan 读到最终态、指纹一致。

配置前先对上游复验，不照抄两周前的证据：

| deployment | 结果 |
|---|---|
| `claude-opus-5-5` / `claude-opus-5` / `claude-sonnet-5-5` / `claude-haiku-5-5` | **200** |
| `claude-opus-5.5`（点号写法） | **404** |

响应是标准 `type: "message"` / `stop_reason: end_turn`，带 Azure 的三个额外字段
（`container` / `stop_details` / `diagnostics`，实测均 `null`），usage 含
`cache_creation` / `inference_geo` / `service_tier` / `speed` —— 与
[2026-10-17 实测证据](../2026-10-17-claude-three-paths/EVIDENCE.md) 一致。那个 404 正是
`claude_model_identity` 存在的理由：**点号拼法必须改写成短横才能打到 Foundry。**

变更手法上刻意用了 `kubectl patch --type=json --patch-file` 而不是 `apply`：这个 ConfigMap
的凭据是明文字面值（既有形态），而 `apply` 会把整份内容写进
`last-applied-configuration` 注解。读回核对确认该注解长度仍是 3654（未被改写），
**新凭据没有进注解**。

补丁前做了 11 项结构校验（YAML 可解析、顶层 key 只多 `azure_anthropic`、9 个 ADB 端点与
`azure_openai` / `github_copilot` / `auth` / `usage_storage` 逐段不变、api_base 无路径、
4 个 deployment 为短横拼法）。ConfigMap 由 3907B / `2af897bf…` 变为 4463B / `aaaa55ec…`。

回滚前提也验过：旧镜像用 `config.get("azure_openai")` 这类按需读取、`yaml.safe_load`
不做 schema 校验，所以 `34acc39` 读到新增的 `azure_anthropic` 顶层段会**完全忽略**而不是崩溃
—— 回滚路径成立。回滚材料（变更前 ConfigMap + 补丁）留在 `.release-private/cm-backup/`
（0700/0600，gitignored，不入 Git）。

## 发布前验证

本地 `python3 -m pytest tests/ -q`：**1400 passed, 6 skipped, 1362 subtests**。

镜像内自检 **42 项全过**（验的是镜像里的代码，不是工作区）：构建身份（`0a9f830` /
manifest `2a1a3c6b…` / 24 文件 / tree 不脏）、三种拼法互转含点号→短横那一条、
默认序 ADB→GHCP→Foundry、Foundry 用 `x-api-key` 且**不**带 `api-key`（实测它回 401）、
ADB 用 Bearer、端点 token 不进 `repr`（上一轮真实泄露过）、Foundry 计价为 `None` 而非 0、
三个新诊断字段在 `safe_diagnostics` 白名单内、Dashboard 非占位（208KB）且含 messages 路径块、
`azure_openai` 不在 messages 面。

`plan` 阶段 `production_changed: false`、`server_dry_run_verified: true`；两个入口
（`claude-lb` / `claude-lb-internal`，selector 均 `app=claude-lb`）、preStop 45 秒、
三个探针与 90 秒 grace 全部保留。

## 线上实测

- `/version` → `0a9f830` / `verification: matched`；`/health/{live,ready,accepting}` 全 200
- Pod `claude-lb-698c979c9c-8pzfr`，1/1 Ready，**0 重启**
- **Foundry 已加载**：`Loaded Foundry Anthropic endpoint: eastus2-foundry (deployments:
  ['claude-opus-5-5', 'claude-opus-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'])`
- provider 装配齐全：9 个 Databricks + 1 个 Foundry + 4 个 Azure OpenAI + 5 个 Copilot
  （`ZenoRewn` in service，另外四个 `INERT (enabled: false in config)`）
- `copilot_h2_stream_id_watermark` → **1**（全新连接，离 400 的轮换阈值很远）；
  `copilot_h2_connections_retired_total` 只有 HELP/TYPE、无样本 —— 重启后未发生退休，符合预期
- 18 个 `circuit_open` 样本**全部为 0**
- `/v1/models` 30 个条目，12 个 claude 条目全部是 `databricks-claude-*` 拼法并声明
  `['/v1/messages']` —— Foundry 的短横 deployment 名**不**单独出现在清单里，这是对的：
  它是内部改写目标，暴露出去会让客户端以为多了一批模型
- 顺带解决一个待办：`zenoren-msft` 轮换后旧 token 还在上一个进程内存里，本次重启后新进程
  从文件读取（日志 `resolved token for 'zenoren-msft' from ...`），`/admin/copilot/reload`
  不必再跑

## 路径视图线上实测：候选确实按模型逐个算

`/admin/model-routing/{model}` 的 `messages_paths` 在线上逐模型给出三条路径，
`effective_messages_order` = `["databricks", "copilot", "azure_anthropic"]`（默认序生效）：

| 模型 | ADB | GHCP | Foundry |
|---|---|---|---|
| `databricks-claude-opus-5-5` | 9/9 端点，`reason: null` | 1/5，改写为 `claude-opus-5.5` | 1/1，改写为 `claude-opus-5-5` |
| `databricks-claude-sonnet-5` | 9/9 | 1/5，`claude-sonnet-5` | **0/1，`no_endpoint_permits_model`** |
| `databricks-claude-opus-4-8` | 9/9 | 1/5，`claude-opus-4.8` | **0/1，`no_endpoint_permits_model`** |

三件事由此得到线上确认：模型 id 按 provider 正确改写（GHCP 点号、Foundry 短横、单段版本如
`sonnet-5` 两侧一致且没被补成 `.0`）；候选集**按模型逐个算**而不是按 provider 整体假设
（Sonnet 5 与 Opus 4.8 在 Foundry 没有 deployment，视图直接说出原因）；默认优先序生效。

### 但 GHCP 那一跳的 1/5 是通配乐观，不是已证实的能力

逐账户判定：`ZenoRewn` 的 `reason` 为 `null`、`serves_model: true`，而
**`capability_known: false`** —— 它进候选是因为「从没发现过能力的账户服务所有模型」这条通配
规则，不是因为证实了它能服务 Claude。而 2026-10-17 实测该 enterprise seat 的 claude 条目是
**0 个**。四个 business 账户则是 `account_never_enabled`。

所以 ADB 故障时的实际链路是 **ADB → GHCP（打到 `ZenoRewn`，会失败）→ Foundry**，中间多一跳
注定失败的调用。这符合既有设计（`capability_unknown` 是通配警告而非排除原因，由
`capability_known` 单独表达），面板上也能看见，但它意味着三路径里**真正能兜底的只有 Foundry**。

两条干净的改法，都属运维决定，本次未动：对 `ZenoRewn` 跑一次 `discover`（只读查询上游、
建能力快照、**不写** `enabled`），此后 `not_in_capability` 会把它从 claude 模型上正确摘掉；
或者启用一个 business 账户，让 GHCP 这一跳真的有能力。

## 缺口：Foundry 不在 `/stats` 与 `/metrics` 里

**这是本次发布的已知缺口，不是边界说明。** `/stats` 顶层有 `endpoints`（9 个 ADB）、
`azure_openai`、`github_copilot` 三段，**没有 `azure_anthropic`**；`/metrics` 的 18 个
`circuit_open` 样本同样只覆盖 9 + 4 + 5，Foundry 的 `eastus2-foundry` 一个都没有。

后果：Foundry 一旦接流量（ADB 与 GHCP 都不可用时），它的请求数、错误数、熔断状态与成本
在工作台和 Prometheus 上都看不见。按既有不变量，Foundry 的费用应当报为**未知**（CCU 计量，
套 Anthropic 公开价会得出自信的错数字）—— 但「未知」也要显示出来才算表达，当前是根本没有这一列。

现在影响为零：Foundry 是 failover 第三跳，ADB 健康时不接流量。但这件事必须在它第一次接流量
**之前**修，否则第一次用到它恰好是最需要可观测性的时刻。

## 边界

- **「通过网关真的打到 Foundry」这条端到端路径在生产上未验证。** 已验证的是：端点已加载
  （日志）、凭据与 4 个 deployment 直连可用（本轮实测）、模型 id 改写正确（镜像内自检）。
  要在生产触发这一跳需要 ADB+GHCP 同时不可用，或给某个模型配 Foundry 优先的 per-model 顺序
  —— 后者属于运维决定且已约定「后续在 Dashboard 里配」，所以本次没有擅自改路由策略
- **GHCP 的 messages 路径仍然不接流量**：唯一启用的 `ZenoRewn` 是 enterprise seat，实测
  0 个 claude 条目；四个各有 10 个 claude 的 business 账户仍 `INERT`。启用哪个是运维决定，
  本次未动
- **进程内累计计数器随本次重启归零**（请求数、错误数、token 累计等），这是预期的
- ConfigMap 里的上游凭据是**明文字面值**（既有形态，非本次引入）。本次只做到不让状况恶化
  （没有把新凭据写进 `last-applied-configuration`）。迁到 Secret 是一次独立变更
- `_image_preflight` 的已知缺陷（首次拉取慢会把通过了的预检读成硬失败）**本次没有触发**，
  也仍未修复 —— 需要重建控制器镜像，而本次重建只同步了 `APP_FILES`，没有捎带修它
