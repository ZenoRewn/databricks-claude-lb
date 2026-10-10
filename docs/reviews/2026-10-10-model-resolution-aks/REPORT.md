# Claude 5.5 模型解析修复 AKS 发布回执

Author: Zeno Ren

2026-10-10 将 [PR #26](https://github.com/ZenoRewn/databricks-claude-lb/pull/26) 合并后的 `c3187243ebbfbab956c83d0b2ec721c905267fd5` 发布到 AKS。**本次发布执行了两轮：第一轮回滚，第二轮成功。** 回滚原因是我的探针选择错误，不是应用缺陷，详见下文。

第二轮回执 **succeeded**，北京时间 **18:28:40.452–18:29:27.002** 维护并恢复流量，**46.550 秒**，九项 action 全部 verified，三个常规业务探针通过且 `persistence_verified=True`，`cleanup_errors` 为空。

## 修复的缺陷

显式请求 `databricks-claude-opus-5-5` 实际跑在 **Opus 5** 上，只有一行 INFO，没有任何错误信号。版本正则 `opus[-_.]?5(?:[-_.]|$)` 的分隔符字符类同时匹配 `5.5` 里的第二个点，于是 5.5 命中 5 的分支。同一机制把 `claude-haiku-5.5` 降两档到 Haiku 4.5，把 `claude-fable-5-1` 跨家族替换成 Sonnet 4.6。

根因不止是正则。既有契约测试要求 `claude-opus-5.1` 作为「小版本号（次要修订）」回落到 `opus-5`，但实测 `opus-5-1`、`sonnet-5-1`、`opus-6`、`*-fast` 在上游**都不存在**，而 `5-5` 是独立模型 —— 这条假设本身是错的，只改正则不够。

### 改为规范化而不猜测

定家族 → 剥离非身份片段（8 位日期戳与 `latest`）→ 统一 `-_.` 分隔符 → 拼名字。34 行 if/elif 变成 10 行。

- 写了版本号就按写的转发，**上游是「该版本是否存在」的唯一权威**。新版本上线当天无需改码；不存在的版本得到上游明确的 `passthrough is not supported`，不再被静默降级
- 只有完全没写版本号才落到家族默认值，这是代码里唯一的版本假设
- 日期戳与 `latest` 之外的尾缀保留（如 `-fast`），吞掉会导致按另一档价格计费

家族默认升到 5.5：更新且更便宜（opus 4/20 vs 4-7 的 5/25，sonnet 2/10 vs 3/15，haiku 0.10/0.50 vs 1/5）。

### 一并修的三处连带缺陷

`haiku-5-5` 与 `fable-*` 没有定价条目，`get_model_pricing` 返 None、成本静默漏统计；`opus-5-5` 子串命中 `opus-5`，按 5/25 计而非 4/20。价格取自 `claude.com/pricing`（2026-10-10 取）。

`output_config.effort` 对 Opus 5.5 被丢弃（发布前实测响应头 `x-lb-dropped-parameters`）。允许列表改为**精确成员判断**，因为 `...opus-5-5` 包含 `...opus-5`，未验证变体不能继承支持。

`historical_model_cost` 缺 `'fable-'` 前缀，裸 fable 行走了 OpenAI 口径的 inclusive-input 分支。

实际 Deployment spec 仅改变 image 与 source-revision 注解；单副本、资源 requests、Service selector、Ingress、配置/凭据、三探针、preStop、grace 与环境变量均未改动。

## 第一轮回滚：探针选择错误，非应用缺陷

| 项 | 值 |
|---|---|
| release id | `lb-20261010-models-37f523` |
| 结果 | **rolled_back** |
| 失败 action | `verify_business`，`error_type=ReadTimeout`，`code=business_or_persistence_verification_failed` |
| 维护窗口 | 18:07:28.144–18:11:10.386（222.242 秒，含回滚） |
| 计划 hash | `8394e86c52b49230f1e7ed47c902c2e7624a27d5c9e225715bfb06fd8c811349` |
| cleanup_errors | `["RuntimeError"]` |

我把 `databricks-claude-opus-5-5` 设成了强制业务探针。`verify_backend` 已 verified、`restore_routes` 已 verified，失败发生在业务探针。

归因过程（没有停在「超时」就下结论）：直接复现该请求，上游返回 **`TEMPORARILY_UNAVAILABLE`** —— `Databricks is unable to satisfy this request due to unexpected capacity constraints`，`upstream_status=503`，`upstream_endpoint=southcentralus-adb-ws`。随后分离「请求变大」与「时间变化」两个变量：`max_tokens` 4 → 1/3 成功、16 → 3/3、128 → 2/3，证明**与请求大小无关，是间歇性容量限制**。

**因此本报告修正一项先前表述。** PR #26 里写的 Opus/Sonnet/Haiku 5.5「全部端点 6/6 稳定」在测量当时为真，但不成立于所有时间。发布后复测：opus-5-5 为 5/6（1 次容量拒绝），sonnet-5-5 与 haiku-5-5 为 6/6，opus-5 为 6/6。

错在让发布门禁去赌上游容量 —— 业务探针的职责是验证网关能否服务流量，不是验证上游余量。第二轮改用历次发布均稳定的 `databricks-claude-opus-5` 做门禁，5.5 的验证移到发布后单独执行。

回滚本身干净：`rollback.verified=true`、`public_health_verified=true`、`storage_backend_ready=true`，复核时 Pod Ready、0 重启、两个 Service selector 均为 `app=claude-lb` 无维护标识残留、`accepting` 200、`backend_ready=1`。

## 固定身份与验收

| 项 | 值 |
|---|---|
| release id | `lb-20261010-models-16ff0f` |
| registry manifest | `@sha256:e6b00f998d6547154953c645a048f3c565c6ac271bad098bc9c3fd90bdfc6293` |
| 平台 | linux/amd64 单平台 manifest v2（已确认无 `manifests` 字段，非多架构 index） |
| config digest | `sha256:5f6233a9fb51e6d78526142e6fcea6b99cb07743e9eee201738aa6d112988163` |
| source manifest | `aff8eca6f39405c652f551b74d379f81fa43b958452a8cc117240e46d301b31e` |
| source_tree_dirty | `false` |
| 回滚锚点 | 前一版本 `5f2463e` 的 `@sha256:691109f9…` |
| 计划 hash | `a623605532dec8219c28894f26458512852ee7b8504aabec9920c8bc92f1f89a` |
| 公网 `/version` | `c318724` / `verification=matched`，零重启 |

PR #26 六项 CI 全绿后合并，main 为 `c318724`。本机全量 **897 passed、834 subtests passed、6 skipped**（较上一版 +27）。正式镜像内断网 **767 passed、692 subtests passed、7 skipped、零失败**；本次新增及修改的三个测试文件在镜像内 77 passed。

镜像内测试使用继承发布镜像的临时镜像，仅叠加 pytest/pytest-subtests 层，并逐文件比对 `main.py`、`effort_compat.py` 的 sha256 与发布镜像一致，应用源码未经挂载替换。镜像内与本机数差额来自 `operations` 发布工具包和 `openai`/`anthropic` 测试 SDK 按设计不进运行镜像，以及 `CLAUDE.md`、`docs/`、`model-capabilities.example.json` 不是运行文件 —— 已逐项核对为环境性缺失，无一是回归。

### 发布后在运行进程内读回

```
claude-opus-5-5   -> databricks-claude-opus-5-5     claude-opus-5.1 -> databricks-claude-opus-5-1
claude-haiku-5.5  -> databricks-claude-haiku-5-5    claude-opus-6   -> databricks-claude-opus-6
claude-fable-5-1  -> databricks-claude-fable-5-1    opus-5-5-fast   -> ...opus-5-5-fast（尾缀保留）
claude-opus       -> databricks-claude-opus-5-5     DEFAULT_MODEL   = ...sonnet-5-5
opus-5-5 价 = 4.0/20.0   haiku-5-5 价 = 0.1/0.5   effort 列表 = (opus-5, opus-5-5)
```

### 端到端（客户端实际会发的短名字）

| 请求 | 上游实际模型 | 修复前 |
|---|---|---|
| `claude-opus-5-5` | `claude-opus-5-5` | `claude-opus-5` |
| `claude-sonnet-5-5` | `claude-sonnet-5-5` | `claude-sonnet-5` |
| `claude-haiku-5.5` | `claude-haiku-5-5` | `anthropic.claude-haiku-4-5-…`（降两档） |

### 关闭了一项先前的未验证项

PR #26 把「Opus 5.5 是否接受 `output_config.effort`」列为合并前无法关闭 —— 旧网关在请求发出前就剥掉该字段，隔着网关探不到。发布后实测：`claude-opus-5-5` 返回 `x-claude-effort-forwarded: low` 且 HTTP 200，`claude-opus-5` 同样；对照 `claude-sonnet-5-5` 正确返回 `x-lb-dropped-parameters: output_config.effort`。**上游接受该字段，精确匹配按预期工作，不需要回退。**

## 未验证边界

**Opus 5.5 当前受 Databricks 间歇性容量限制**，发布后采样 5/6，被拒请求返回 `TEMPORARILY_UNAVAILABLE`。这是上游容量，网关侧无可用开关；不要把它当作本次改动的回归。Sonnet 5.5 与 Haiku 5.5 当前未观测到该拒绝。容量随时间变化，上述比例是时点采样，不是稳定性承诺。

**Fable 5.1 受 region 限制**：6 次采样仅 2 次成功，其余返回 `NOT_FOUND ... not available in your region`，即只有部分 workspace 有该模型，按端点轮询会间歇失败。需用 `endpoints[].models` 限定到确有该模型的端点 —— 配置动作，代码层面修不了。本次未执行该配置变更。

**Haiku 5.5 官方按 prompt 大小分两档**，本表为平价结构，只收 ≤100K 档；>100K 档为 5×，会低估。沿用表内 gpt-5.4/5.5 只收短上下文价的既有处理；不收录则成本完全不计入。

行为变更需知悉：`claude-opus-5.1` 从「静默给 opus-5」变成上游 400；`claude-opus` / `claude-sonnet` 等裸别名从 4-7 / 4-6 上调到 5-5；`claude-3-opus-20240229` 这类仅带日期的旧名日期被剥离后无版本，落到家族默认（5-5）。

本次未执行数据库快照与隔离恢复验证，也未安排独立 QA 复核。上一轮遗留的上游三类拒绝（120 秒 `CANCEL`、`REFUSED_STREAM` 限流、裸 `response.failed`）本次未触碰，仍未解决。
