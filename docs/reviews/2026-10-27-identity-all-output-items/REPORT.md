# Codex 重复复发、输出项身份补齐，与一次被探针挡下的发布

Author: Zeno Ren

日期：2026-10-27（北京时间）

## 一、Codex 重复复发：不是回归

运维报「之前修好的 Codex 重复又出现了，看看是不是什么被冲突掉了」。**代码完好无损**：

| 检查 | 结果 |
|---|---|
| Pod 内 `responses_item_identity.py` 指纹 | `2c6cabef…`，与 2026-10-22 发布回执记录**完全一致** |
| 该模块提交历史 | 仅 `8e1bd8a`、`51507c5` 两次，均已发布 |
| 门控代码 | `main.py` 的 `identity_mode = ... if model in copilot_responses_item_id_models else 'off'` 原样 |
| 同时段 identity 事件 | `gpt-6.1-sol` 103 次、`gpt-6-astra` 82 次、`gpt-5.6-sol` 35 次，全部 `reason: none` |

**真因是逐模型 opt-in 的名单没跟上新模型。** 当天刚开通的 `gpt-6.1-terra` 不在
`COPILOT_RESPONSES_ITEM_ID_MODELS` 里，79 次流式请求**零** identity 事件 —— 名单外的
模型连事件都不产生，所以「日志干净」恰好是最坏的情况。

该项是 Deployment 的 env，已直接 patch 并滚动重启（发布工具的 `target_env` 要求目标
镜像必须不同，纯 env 改动走不了那条路）。改后实测 terra 的 message 项在
added / delta / done / completed 四处同一个 id。

## 二、排查中量出第二个缺陷：reasoning 项的身份

拿一直在名单里的 `gpt-6.1-sol` 做对照，message 稳定而 **reasoning 项三处三个 id**：

```
idx=0 reasoning  added=6023a19e  done=2c705db7  completed=6a84a5ed
idx=1 message    added=56a0c0d0  delta=56a0c0d0  done=56a0c0d0  completed=56a0c0d0
```

客户端按 `findIndex(existing.id === incoming.id)` 找不到就 `push`，所以漂移的
reasoning id 会被追加成第二个思考块 —— 与 message 当初的重复是同一个机制。2026-10-20
批准的方案原话也是「维护稳定的**输出项**身份」，落地时收窄成了 message。

## 三、第一次发布被业务探针挡下（`lb-identity-all-items-fd38e8a`）

把改写扩到全部输出项后发布，在 `verifying_business` 被 **`grok-4.7`** 挡下
（`completed: false`、`marker_found: false`、`text_chars: 0`）并整体回滚。前两条探针
（`messages/databricks-claude-opus-5`、`responses/gpt-6-astra`）均通过。

回滚干净：`rolled_back`、`public_health_verified: true`、`storage_backend_ready: true`，
`/version` 回到 `c638187`，维护 117.4 秒。**工具之外设的两个 env 也都保住了**
（`REQUEST_BODY_TIMEOUT_SECONDS=300`、14 项模型名单）—— 它们在私有快照的原环境里，
回滚只还原镜像与注解。`cleanup_errors: ['RuntimeError']` 出现过一次，但
`cleanup.verified: true`，事后检查命名空间无任何遗留 Pod 或 ConfigMap。

**真因：改写了 reasoning 项，却没把引用它的事件认全。** 实测 grok-4.7 一条流里
reasoning 项带**六个**不同 id：

```
output_item.added              item.id=825c77c2
reasoning_summary_part.added   item_id=f0c960b4
reasoning_summary_text.done    item_id=3aeeabc6
reasoning_summary_part.done    item_id=c1c17a01
output_item.done               item.id=796ec442
response.completed  output[0]  id=b9db59c1
```

`reasoning_summary_*` 不在 `_REFERENCE_EVENTS` 里，于是走 `_unknown()`，而它对
「需要改写但不认识的事件」直接 `_fail()` → 流被切断。**fail-closed 本身是对的**
（半稳定的身份比不稳定更糟），缺的是把事件认全。

## 四、修正（`461de98`）

- **`_REFERENCE_EVENTS` 补齐** reasoning 与工具的引用事件（官方 streaming-events
  schema 里都带顶层 `item_id`）：`reasoning_summary_part.added/done`、
  `reasoning_summary_text.delta/done`、`reasoning_text.delta/done`、
  `function_call_arguments.delta/done`。后两个同样必须 —— 改写了 `function_call`
  的 id 却不认识它的 arguments 事件，第一条漂移的工具流会以完全相同的方式被中断。
- **新增 `_REWRITABLE_KINDS = {message, reasoning, function_call}`**：只改写引用事件
  已经认全的类型。这比「改写全部类型」更强 —— 上游哪天多出一种 item 并让它的 id
  漂移，宁可原样转发（客户端最多多显示一条），也不能打断整条流。`mapped_items`
  的口径随之与可改写集合对齐。
- 只改写顶层 `id`；`call_id` / `arguments` / `encrypted_content` 一律不动（实测三者
  互相绑定，错配会 400）。安全性依据与 message 相同：`_strip_input_item_ids` 剥掉
  每个 input item 的顶层 `id`，所以发给下游的 id 永不回放给上游。

**用真帧验证**：把生产抓到的 grok-4.7 原始 18 帧原样重放过新实现 ——
`failure_reason: None`、`unsupported: None`、`drift_items: 1`、`rewritten_frames: 7`，
两个输出项各自只剩一个 id。单元测试的形状是手写的，这一份是把发布打回去的那份真输入。

| 检查 | 结果 |
|---|---|
| 完整回归 | 1751 passed、6 skipped、1416 subtests |
| 新增测试 | 9 条（5 条覆盖全类型身份，4 条覆盖引用事件与白名单），均先在旧实现上红 |
| node UI 测试 5 个、`git diff --check` | 通过 |

## 五、第二次发布（`lb-identity-refs-461de98`）成功

镜像 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:2f08642acfd0459a4b1d920ff16d8ceef96e6feaa06e2500739df949ae4d33ae`，
`source_tree_dirty: false`，运行文件清单未变（23 个），控制器无需重建。

- 11 个阶段一次通过，维护 **63.8 秒**（12:35:28 → 12:36:32 UTC，预算 300），含探针全程 102.1 秒。
- **5 条业务探针全部 `completed` + `marker_found`**，含上次挡下发布的 `grok-4.7`
  与本轮新增的 `gpt-6.1-terra`。
- 旧 writer exitCode 143，有终止证据；`cleanup_errors: []`。
- `/version` = `461de98` / `matched`；Pod 内 `responses_item_identity.py`
  `51f61f7e…`、`main.py` `c4f1d642…`、`copilot_pricing.py` `8dc7553f…` 与构建清单逐一相符。
- 发布后线上实测三个模型，reasoning 与 message 两项的 id 均为**各一个**：
  `grok-4.7`、`gpt-6.1-sol`、`gpt-6.1-terra`。

本次同时上线 2026-10-26 的目录重新核验（`gpt-6.1-terra` / `gpt-6-astra` /
`gpt-5.4-nano` / `grok-4.6` / `gemini-4-argon` 收录，三个下架条目移出，价目基线轮换），
以及「已发现 capability 不再抹掉核验过的入口」那一处修复。

## 六、未验证边界

- **「Codex 桌面端不再显示重复」没有端到端验证。** 已验证的是网关交付的同一输出项
  在全生命周期只有一个 id；客户端渲染行为只能由桌面端确认。
- **`reasoning_text.delta/done` 两个事件没有实测样本**，是按官方 schema 加入的
  （已观察到的是 `reasoning_summary_*` 四个）。它们同样声明顶层 `item_id`，但
  「照 schema 加」与「实测过」不是一回事。
- **未认全引用事件的 item 类型仍会漂移**，只是不再打断流。真出现这种上游类型时，
  客户端可能多显示一条 —— 那是有意选的方向。
- `REQUEST_BODY_TIMEOUT_SECONDS` 由 120 提到 300 是**预算形状未改**的权宜之计：
  它仍是「上传总时长上限」，把「传得慢但在传」与「客户端卡死」同等对待，约 18 MB
  的请求体会再次撞墙。空闲超时的改法尚未实现。
