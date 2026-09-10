# Copilot opaque state：拒绝分类、有界恢复、发布前验收

给发布前复核用。回答三个问题：**改了什么**、**哪些是实测的哪些是推断**、**发布要检查什么**。

对应交接文档 `lb-complete-analysis-20260910.md`（§8 设计约束、§9 验证矩阵、§10 发布约定）。
本文不是部署授权。

---

## 1. 一句话

携带 opaque state 的请求被 GHCP 拒绝时，LB 过去**必死**且**把每个计数器记两遍**、还白刷一次
session token。现在走一条预算显式的两级恢复阶梯，走不通则返回明确的 `orphaned_conversation_state`，
永不静默成功。

---

## 2. 改动前的行为（实测，非推断）

生产 `/metrics`（单账号 `ZenoRewn`，`lb.zeno.ink` 免认证抓取）。三次抓取，同一个 Pod：

| 指标 | 抓取 ① | 抓取 ② | 抓取 ③（最后一次核验） |
|---|---|---|---|
| `copilot_endpoint_requests_total` | ~95 | 218 | **381** |
| `copilot_endpoint_completed_requests_total` | 87 | 209 | 362 |
| `copilot_endpoint_neutral_requests_total` | 7 | 8 | 9 |
| `copilot_upstream_401_total{scope="request"}` | 6 | 6 | **6** |
| `copilot_upstream_401_total{scope="endpoint"}` | 0 | 0 | 0 |
| `copilot_orphaned_item_id_events_total{detected}` | 12 | 12 | **12** |
| `copilot_orphaned_item_id_events_total{recovered}` | 0 | 0 | 0 |
| `copilot_orphaned_item_id_events_total{unrecoverable}` | 12 | 12 | **12** |
| `copilot_token_refresh_total` | 7 | 7 | 7 |
| `copilot_input_item_ids_stripped_total` | 3383 | 18049 | 27634 |
| `copilot_input_item_ids_stripped_requests_total` | 58 | 167 | 321 |

**关键观察 1：流量从 ~95 涨到 381（4 倍），主动剥离从 3383 涨到 27634（8 倍），而 orphan 三个
计数器与 `token_refresh_total` 一个都没动。** 三次独立抓取给出同一结论：这是**一段有界的历史
插曲，不是持续的失败率**。分母口径也随之变化 —— 6/95 ≈ 6.3% 只是插曲发生时那一小段的比例，
按最后一次核验算是 6/381 ≈ 1.6%，且**已冻结**。严重性不在频率，而在「一旦发生，该会话永久
废掉且毫无出路」。

**关键观察 2（2026-09-10 实测 Codex CLI 0.145.0 后的重新解读）：那 6 个是 LB 请求，不是 6 个用户
轮次。** Codex 对 401 会重试 **6 次**（5 次 `Reconnecting... N/5` + 1 次原始），且**每次重试的请求
体语义完全相同**（实测：6 次 body 的 `input[]` 逐字一致，0 处字段差异）。而生产形态下每个 LB 请求
= 2 次 GHCP 调用 + 1 次强制 token 刷新。于是：

```
1 个中毒的 Codex 轮次
  → 6 个 LB 请求（客户端重试）
  → 12 次 detected / 12 次 unrecoverable / 6 次 request-401 / 6 次强制刷新
```

生产实测值恰好是 `12 / 12 / 6 / 6`（`token_refresh 7 = 6 + 1 warmup`）。**所以最可能的读法是
「一个用户轮次被客户端重试 6 次」，而不是「6 个独立的倒霉请求」。** 用户可见影响 = 1 个会话；
上游代价 = 12 次 GHCP 调用。

这条修正很重要：它说明**放大的主要来源是客户端重试**，所以「让客户端别重试」和「从根上不产生
跨账户回放」（会话亲和）比「把恢复做得更快」价值更大。

抓取 ① 的 `requests_total` 是由 `completed 87 + neutral 7 + errors 1` 推算的（当时没抓那一行），
抓取 ② 是直接读到的 218 = 209 + 8 + 1。

两个结构性缺陷（都用真实代码路径实测，只 mock httpx client 与 token 交换）：

**① `recovered` 结构性不可达。** `_proxy` 已在发送前主动剥掉 `input[*].id`，事后的兜底再调同一个
**幂等** helper 必然返 0，永远落到 `unrecoverable`。

**② orphan 拒绝继续掉进 401 auth repair。** 强制刷一次 session token（注定无用：实测轮换不使
opaque state 失效，`docs/TROUBLESHOOTING.md` §16）再原样重发：

| 一个客户端请求（旧行为） | 流式 | 非流式 |
|---|---|---|
| `detected` / `unrecoverable` | 2 | 2 |
| `upstream_401{request}` | 1 | 1 |
| 强制 session token 交换 | 1 | 1 |
| 上游调用次数 | 2 | 2 |

生产数字全部对上：`12 = 6×2`、`6 = 6×1`、`token_refresh 7 = 6 + 1 warmup`、`neutral 7 = 6 + 1`。
**唯一请求数就是 6**，无需按 request_id 去重。

---

## 3. 上游行为的实测结果

直连 GHCP Enterprise（`gpt-5.4-mini`，`store:false`，带 function tool 回路，隔离测试会话）。

### 3.1 两级独立校验

篡改 `encrypted_content` 中间 20 字符 → **不是** orphan 报错：

```json
{"error":{"message":"The encrypted content Zb+H...LQ== could not be verified. Reason: Encrypted content could not be decrypted or parsed.","code":"invalid_request_body"}}
```

结论：先解密/解析，再校验归属。orphan 报错 ⇒ blob 能解开但归属对不上。这削弱了「状态格式不兼容
/ 过期」这条根因候选，并暴露出**第一类错误此前完全没有兜底**。

### 3.2 删掉 blob 的代价

下表是**两次独立隔离会话**的合并结果，不是一次实验：会话 ①（默认参数）给出 A–E，
会话 ②（显式 `store: false` + function tool 回路，更贴近 Codex 实际请求形态）给出 B–F 的对照。
两次的 reasoning item 都是 **420 字符 `id`**，`encrypted_content` 分别是 **5160 / 5128 字符**
（同一账号不同轮次长度会略有差异，所以下表只写量级）。

| 组 | 处理 | 结果 |
|---|---|---|
| A | 原样回放（420 字符 `id` + ~5.1KB `encrypted_content`） | 200，答出暗号 |
| B | 剥 `id`、保留 `encrypted_content`（旧行为） | 200，答出暗号 |
| C | 剥 `id` + 整条删除 reasoning item | 200，答出暗号 |
| D | 剥 `id` + 保留 item 但删 `encrypted_content` | 200，答出暗号 |
| E | 剥 `id` + 篡改 `encrypted_content` | 400 `invalid_request_body` |
| F | 保留 `id` + 删 `encrypted_content` | 200，答出暗号 |

- `call_id` 回路在 C/D/F 全部正确配对
- `input_tokens=1637 / cached_tokens=1280` 在 B/C/D **完全一致** → token 与 prompt cache 代价为零
- 选 D 而非 C：两者都 200，但删 blob 之后已无 opaque 载体，整条删 item 救不回任何东西

### 3.3 为什么不会静默丢服务端历史

GHCP **不支持** `previous_response_id`（带上直接 400 `previous_response_id is not supported`，
2026-09-09 实测，`docs/TROUBLESHOOTING.md` §11）。所以客户端必然把完整历史放在 `input[]` 里，
删掉的只是加密的推理链。这消掉了交接文档 §8.1.2 的主要顾虑。

### 3.4 被实测判掉的两条根因候选

交接文档 §7 候选表里这两条已有控制实验（2026-09-09，`docs/TROUBLESHOOTING.md` §16）：

| 候选 | 实验 | 结果 |
|---|---|---|
| 同账号 token/session 更替造成状态失效 | 换全新交换的 session token 回放 | 200 |
| 物理连接 / HTTP2 生命周期导致状态失效 | 换全新 httpx client（新 TCP/TLS）回放 | 200 |

⇒ 「未控制 token 与账号之前不要因 connection 字样去改 HTTPX 连接池」这条建议成立，且现在有实测支撑。
**本次没有改任何连接池参数。**

---

### 3.5 跨账户回放：把最大的那条推断变成实测（2026-09-10）

用**两个真实 GHCP 账户**（`ZenoRewn` → `api.enterprise.githubcopilot.com`，
`zenoren-icloud` → `api.business.githubcopilot.com`，连 GHCP 部署都不同），模型 `gpt-6-astra`
（生产出问题的正是这个），A 账户铸造 reasoning `encrypted_content` 后拿到 B 账户回放：

| 处理 | 上游 | 答出暗号 |
|---|---|---|
| 原样（`id` + blob） | **401 `input item ID does not belong to this connection`** | — |
| **只剥 `id`（LB 生产现行行为）** | **401 `input item does not belong to this connection`** | — |
| **剥 `id` + 删 blob（rung 2）** | **200** | **✓** |
| 剥 `id` + 整条删 reasoning item | 200 | ✓ |

**双向（A→B 与 B→A）× 3 次重复 = 24/24，零例外。**

三条结论：

1. **生产那条错误被精确复现。** 「只剥 id」组的原文与生产日志**逐字一致** —— 因为线上默认就会
   主动剥 id，所以线上看到的必然是不含 `ID` 的那个变体。
2. **rung 2 确定能救回归属类拒绝，且上下文保真** —— 200 之后仍能答出上一轮种下的暗号。
   这条从「推断」升级为「实测」。
3. **多账号的会话亲和是硬前置条件，不是理论顾虑。** 两个账户之间任何一次跨账户回放都必然 401。
   `least_requests` 会把同一会话的连续轮次分到不同账户，所以只要配了第二个账户，这个 401 就是
   **必然事件**而不是偶发。

回归测试 `CrossAccountReplayShapeTests` 用上面两种**实测原文**跑完整 LB 路径，锁住分类 → 阶梯 →
恢复这条链对实测字节成立。

### 3.6 Codex CLI 的重试策略（实测，决定错误码怎么选）

用本地桩端点 + 隔离 `CODEX_HOME` 实测 codex-cli **0.145.0**（每格是「上游被打的次数」）：

| LB 返回 | 上游命中 | 用时 | 客户端表现 |
|---|---|---|---|
| **400** | **1** | 0.5s | 立即放弃，**把错误体逐字显示给用户**（含 `code` 与指引）|
| 401 | 6 | 6.8s | 5 次 `Reconnecting... N/5` |
| 403 | 6 | 6.9s | 同上 |
| **409** | **6** | 7.2s | 同上，且只显示 `unexpected status 409 Conflict`，**我们的 `code` 被吞掉** |
| **422** | **6** | 6.8s | 同上 |
| 429（无 `Retry-After`） | 1 | 0.5s | 立即放弃 |
| 429（带 `Retry-After: 2`） | 1 | 0.5s | 立即放弃 |
| 500 / 503 | **30**（25s 截断，可能更多）| 25s | 重试风暴 |
| 已提交 200 流 + SSE error | 6 | 7.3s | 5 次 reconnect |

**这推翻了交接文档 §8.2 的假设**：409/422 的重试次数与 401 **完全相同**，换过去毫无改善，还会让
客户端把我们精心写的 `orphaned_conversation_state` 指引丢掉。真正能让客户端立即放弃的是 **400**，
而且它是唯一会把完整错误体透给用户的状态码。

⇒ 非流式的阶梯耗尽改用 **400**（见 §4.3）。**流式无解**：ASGI 的 `http.response.start` 在生成器
被迭代之前就发了 200，所以流式路径只能发 SSE error 帧，实测仍是 6 次重试。要改必须把
response-start 延后到第一个真 chunk —— 那是 `docs/STREAM_OWNERSHIP.md` 覆盖的架构改动，不在本次
范围；现在有实测数据支撑将来做这个决定。**流式路径真正的解法是会话亲和：从根上不产生跨账户回放。**

## 4. 改动后的行为

### 4.1 恢复阶梯

```
rung 0  发送前主动剥 input[*].id                  无条件（COPILOT_STRIP_INPUT_ITEM_IDS，默认 true）
rung 1  被拒后再剥一次 id                         只在 rung 0 关掉 / 出现新 id 通道时有料
rung 2  删 reasoning item 的 encrypted_content    COPILOT_OPAQUE_STATE_RECOVERY，默认 true
走完    orphaned_conversation_state 错误           明确要求客户端重建，绝不静默成功
```

触发条件（对应交接文档 §8.1.6）：`status ∈ {400, 401}`、流式 `not sent_any_chunk`、非流式尚未产出
任何字节。即**上游明确拒绝 + 下游未提交任何响应**，符合 `docs/RESILIENCE.md` 的 POST replay 约束。
同 endpoint、同 lease，绝不换账户。

预算是**显式计数**（`_OpaqueStateRecovery.MAX_RUNGS = 2`），不靠「改写幂等所以最多重试一次」——
外层 attempt 循环、401 auth repair、`_normal_request` 递归是三条能叠加的通道，一个客户端请求共用
一份预算（交接文档 §8.1.5）。**实测上界**：流式 ≤3 次上游调用（`for attempt in range(3)` 是硬顶），
非流式 ≤4（1 原始 + 1 auth repair + 2 级恢复，递归深度 ≤2）。外层 attempt 重试**不会**让阶梯重新
装弹（回归测试 `test_outer_attempt_retry_does_not_reload_the_ladder` 锁住），否则一个中毒会话会成倍
放大上游负载。

### 4.2 opaque-state 拒绝不再喂 401 auth repair

那次注定无用的强制 token 交换消失。**生产形态下上游调用次数不变（2 次，已实测）** —— 原来白打的
那次现在换成一次真正的恢复尝试。

顺带把 auth repair 的预算从 `attempt == 0` 改成**每请求一次的独立 flag**，修掉之前记录为「已知次要
交互，未修」的那条（恢复分支 `continue` 推进 `attempt` 会吃掉 auth repair 预算）。
**副作用**：换到新 endpoint 后出现的 401 现在也能拿到一次刷新。上界不变：每请求最多一次。

### 4.2b 循环不变量与 `retry_budget_exhausted`（自审查发现的回归，已修）

**每个 `continue` 都必须留下一轮来消费修好的请求。** 流式循环里端点切换类的三处（5xx /
PoolTimeout / 网络错误）本来就带 `attempt < max_retries - 1`；原地修复类过去靠巧合满足这条
（剥离幂等 + auth repair 限死 `attempt == 0`），恰好留一轮。

阶梯做成两级 + auth repair 与 attempt 解耦之后，那个巧合没了：三次原地修复能吃满三轮，而
**流式循环体就是 async generator 体的最后一段**，耗尽即静默返回 —— 下游拿到 HTTP 200 + 0 字节。
那正是 `docs/TROUBLESHOOTING.md` §14 里会把 Codex 会话永久毒化的「无终端事件断流」，而且日志会把它
误记成 `outcome=client_disconnect`。**这是本次改动引入的回归，在自审查阶段用真实代码路径复现后修掉。**

修法：两处原地修复显式带上同一条守卫；另加一层循环后兜底终端 `retry_budget_exhausted`（按不变量
不可达，只为防后人加 `continue` 时漏守卫），并配一个必须恒 0 的金丝雀指标
`copilot_stream_retry_budget_exhausted_total`（带零样本，告警写 `> 0`，不必 grep 日志）。回归测试通过把 `MAX_RUNGS` 调高来验证**不变量**而不是
「2」这个数字。**运维含义：生产日志出现 `outcome=retry_budget_exhausted` 即不变量被破坏，按回归处理。**

### 4.3 客户端看到什么

`code = orphaned_conversation_state`，`opaque_state_kind`、`recovery_rungs_attempted`，message 说明
要求开新会话或丢弃历史后重试，保留 `upstream_ids` / `lb_request_id`。

**非流式的 HTTP 状态码改写为 400**（基于 §3.6 的实测）：401/403/409/422 都会让 codex-cli 重试
**6 次**，而每次重试都要跑一整条阶梯（2 次 GHCP 调用）—— 对一个**确定不可重试**的拒绝来说是 12 次
无谓的上游调用。400 只被打 1 次，而且是唯一会把我们的错误体逐字显示给用户的状态码。

**流式仍是 SSE error 帧**（实测 6 次重试）：ASGI 的 `http.response.start` 在生成器被迭代之前就
发了 200，所以流式路径无法把它变成真正的 HTTP 400。

> **这条不是「没想到」，是评估后明确不做。** `_LifecycleStreamingResponse` 是我们自己的
> `StreamingResponse` 子类，技术上可以覆写 `stream_response`、先 `anext()` 拿到第一项再决定
> 发什么 status。不做的三条理由：
> 1. **会把响应头延后到第一项**。生成器在等上游 headers 期间会 yield 心跳，长 thinking 场景下
>    第一项可能是 15s 后的心跳 → 响应头延后 15s。ingress（`proxy_read_timeout: 600`）与
>    Cloudflare（100s）能容忍，但我**没有办法在本地验证**真实客户端与中间链路对「头延后」的
>    容忍度，而这是全链路行为。
> 2. **要动最脆的子系统**。rung 3 必须从「yield SSE error」改成「raise」，这会和
>    `end_current_request` / `_emit_stream_end` / lease 归属交织 —— 正是
>    `docs/STREAM_OWNERSHIP.md` 那份契约管的窗口。
> 3. **根因已被 §4.6 的会话亲和消除**。放大只在真发生跨账户（或跨多账号时期）回放时才出现，
>    亲和从根上不产生它。为一个已被上游堵住的路径去动流式所有权，风险收益比不成立。
>
> **什么情况下重新考虑**：亲和已开启、`copilot_session_affinity_total{hit}` 正常，但
> `copilot_opaque_state_requests_total` 仍持续增长（说明还有别的跨账户来源），且
> `copilot_endpoint_requests_total` 的增速显示重试放大真的在压上游配额。届时先在验证环境
> 量「头延后」对真实客户端的影响，再动。

### 4.4 指标

| 指标 | 语义 |
|---|---|
| `copilot_opaque_state_requests_total` | 受影响的**唯一客户端请求**数 —— 故障规模看这个 |
| `copilot_opaque_state_rejections_total{kind}` | 拒绝**事件**数；`orphaned_id` = 归属对不上，`unverifiable_content` = blob 解不开 |
| `copilot_opaque_state_recovery_total{outcome}` | `attempted` / `succeeded` / `exhausted` |

`succeeded` **只在最终有效完成后**记（交接文档 §9）：流式挂在终端事件 `completed`，非流式挂在真正
返回 `JSONResponse` 之后。上游接受了改写后的请求但流仍被截断 → 不记 `succeeded`，也不记
`exhausted`。

⇒ `attempted - succeeded - exhausted` 是**第三桶**：恢复被上游接受、但流随后断了。读
`succeeded/attempted` 当命中率时要知道分母含这一类；那一桶的量看
`copilot_stream_truncated_no_completion_total`，不要误算成「恢复失败」。回归测试
`test_upstream_accepting_the_retry_is_not_success_without_completion` 锁住这条恒等式。

Legacy `copilot_orphaned_item_id_events_total{stage}` 保留（不改名不删除），但**修正了计数时机**：

| stage | 旧 | 新 |
|---|---|---|
| `detected` | 每次拒绝 +1 | 语义与**数值都不变**（生产形态仍是每请求 2）。第二次拒绝仍然真实存在，只是从「注定无用的 token 刷新后原样重发」变成「一次真正的恢复尝试」。要按请求计数请读 `copilot_opaque_state_requests_total` |
| `recovered` | 事后剥到 id 才 +1（默认配置下不可达） | 不变 —— **这才是「还有别的 id 通道」的金丝雀** |
| `unrecoverable` | 恢复还没走完就 +1 | **只在阶梯彻底走完后 +1**，每请求一次 |

> **告警影响**（实测值，非推断）：
> - `detected` **不变**，不要按「减半」调阈值
> - `unrecoverable` 生产形态从每请求 2 变 1，按它配的阈值需要减半
> - 「`detected` 恒 0 才正常」这条要撤掉 —— 已被生产反证（`detected=12` 而日志明写
>   `no input[*].id to strip`：拒绝的是 blob 归属，剥 id 结构上防不住）

三个新指标都走 `_labeled_counter_samples` 补零，一上线就有 0 序列，告警可写 `> 0` 而非 `absent()`。

### 4.5 改了哪些文件

| 文件 | 改动 |
|---|---|
| `main.py` | `_OpaqueStateRecovery`；`_classify_opaque_state_rejection`；`_drop_reasoning_encrypted_content`；`_mark_opaque_state_exhausted`；`_note_opaque_state_recovery`；`_proxy` / `_stream_response` / `_normal_request` 三处穿 `recovery=`；两处 auth-repair 守卫改一次性 flag 并补 `attempt < max_retries - 1`；循环后兜底终端 + `stream_retry_budget_exhausted_total` 金丝雀；`LBSettings.copilot_opaque_state_recovery`；四个新指标 + 两组 label 常量 |
| `tests/test_copilot_opaque_state_recovery.py` | 新建 |
| `tests/test_copilot_session_affinity.py` | 新建（会话亲和） |
| `tests/test_copilot_request_lifecycle.py` | 夹具补三个新计数器 |
| `CLAUDE.md` / `docs/TROUBLESHOOTING.md` / `docs/RESILIENCE.md` / `docs/AKS.md` | 文档 |

**没有改**：ADB / Azure 的任何分类逻辑、403/429 归属、连接池参数、`Dockerfile`、`deploy/k8s/` 下
任何 manifest、`config.yaml`。

---

### 4.6 会话亲和（多账户的硬前置条件，本次实现）

`_session_affinity_key(body, api_type)` 取 Codex 的 `prompt_cache_key`，
`_affinity_index(key, n)` 用 **blake2b**（不是 Python 的 `hash()` —— 后者对 str 加 per-process
随机盐，多副本之间结果不同，正好破坏这个机制的唯一目的）映射到配置态合格端点集合。

四个设计点：

1. **哈希打在「配置态合格集合」上，不是「当前可用集合」。** 打在可用集合上，任何一次熔断都会把
   **所有**会话重新洗牌 —— 等于给每个活跃会话制造一次跨账户回放。打在配置集合上只影响原本映射到
   那个端点的会话（回归测试 `test_mapping_is_stable_when_an_unrelated_account_goes_down` 锁住）。
2. **对所有 Responses 请求生效，不只 stateful 的。** 第一轮无状态，但它铸造的 state 第二轮就要
   回放；等到 stateful 才钉已经晚了。
3. **与 pinning 组合而非竞争。** pinning 管「同一请求内不许换」，亲和管「下一个请求回到同一个」；
   两者都适用时 pinning 优先。
4. **单账户下是完全的 no-op**（合格端点 ≤1 时不走亲和分支），所以默认开对当前生产零影响。

端到端实测（两个真实账户 + 真实 codex-cli 0.145.0，同一会话三轮，模型 `gpt-6-astra`）：

| | 亲和开 | 亲和关 |
|---|---|---|
| 各账户请求数 | 3 / 0 | 3 / 3 |
| `copilot_opaque_state_requests_total` | **0** | **3** |
| `copilot_opaque_state_rejections_total{orphaned_id}` | 0 | 3 |
| `copilot_opaque_state_recovery_total{attempted}` / `{succeeded}` | 0 / 0 | **3 / 3** |
| `copilot_session_affinity_total{hit}` | 3 | 0 |
| Codex 轮次成功 | 3/3 | 6/6 |

**亲和从根上消除故障；漏过去的由恢复阶梯救回** —— 关掉亲和时 3 次跨账户回放全部被 rung 2 救回，
答案（`17*23=391`）在推理链被删、请求换到另一账户之后仍然正确，用户一次都没看到错误。

指标：`copilot_session_affinity_total{outcome="hit|unavailable|absent"}`。
`unavailable` = 亲和目标熔断/冷却 → 该轮跨账户 → 由阶梯兜住（代价是丢一次推理链），
所以 `unavailable` 与 `opaque_state_requests_total` 应当同步增长；两者背离说明有别的
跨账户来源。开关 `COPILOT_SESSION_AFFINITY`（默认 true）。

## 5. 测试覆盖

`python3 -m pytest tests/ -q` → **469 passed / 421 subtests**（改动前基线 320 / 325，零回归）。
四种 `COPILOT_SESSION_AFFINITY` × `COPILOT_OPAQUE_STATE_RECOVERY` 组合结果相同。

> 这个数字包含本次交付里与 opaque state 无关的其他加固（图片准入、HTTP 入口契约、
> token 三级回退、Databricks 终端契约、PoolTimeout 文档漂移守卫、API 端点表守卫）。
> 只看 opaque state 自身：`tests/test_copilot_opaque_state_recovery.py` +
> `tests/test_copilot_session_affinity.py`。

新文件覆盖（对应交接文档 §9 验证矩阵）：

| 矩阵条目 | 覆盖 |
|---|---|
| 确定性复现 | `test_orphan_401_no_longer_wastes_a_forced_token_exchange`（流式 + 非流式各一）把实测到的旧行为数字钉死 |
| 非流式：无 opaque / 有 blob / 无可删字段 / 再次失败 | `NonStreamRecoveryTests` 7 例 |
| 流式：首包前可恢复、已提交后不得重放、终止事件完整 | `StreamRecoveryTests` 6 例，含 `test_no_recovery_after_any_chunk_was_committed` |
| 上下文保真 | `DropReasoningEncryptedContentTests` 断言 message / `call_id` / `function_call_output` / `summary` 全部保留 |
| 服务端历史不得静默丢 | 结构上不可能（§3.3）；`previous_response_id` 不在 rung 2 射程内 |
| 资源与并发 | 每个用例断言 `active_requests == 0` |
| 鉴权与熔断 | `AuthRepairIsolationTests` 3 例：真 401 仍修复、stateful 401 仍中性、预算不被吃掉 |
| 指标 | `attempted` 在尝试时记；`succeeded` 只在终端事件后记（`test_upstream_accepting_the_retry_is_not_success_without_completion`）；零样本齐全 |
| 回归 | 既有 `test_copilot_input_item_ids.py` / `test_copilot_401_circuit_scope.py` 全过，未改断言 |
| 重试预算不被吃穿 | `RetryBudgetFloorTests` 4 例（3 次原地修复 / 混合端点重试 / 提高 `MAX_RUNGS` / 纯恢复路径不变） |
| 上游调用上界 | `UpstreamCallBudgetTests` 4 例，把流式 ≤3、非流式 ≤4、生产形态 =2、chat 不触发钉死 |
| 终端帧形状 | `TerminalShapeTests` 3 例：Responses 无 `[DONE]`、Chat 有 `[DONE]`、兜底帧两协议都合法 |
| lease 所有权 | `RecoveryResourceOwnershipTests` 3 例：恢复中取消 / 成功收尾 / 一帧不读就关，`on_request_start` 与 `on_request_end` 都恰好一次 |
| 分类误判 | `ClassifyOpaqueStateRejectionTests` 断言「API key could not be verified」等 4 类不相关 400 不被误判 |
| 不外溢到别的 provider | `OtherProvidersMustNotBePatchedTests` 结构守卫：新 helper 调用点计数 + ADB/Azure 那 4 处字面量保持原样 |
| 跨账户实测形态 | `CrossAccountReplayShapeTests` 用两种**实测原文**（带/不带 `ID`）跑完整 LB 路径 |
| 会话亲和 | `tests/test_copilot_session_affinity.py` 19 例：键提取、blake2b 跨 `PYTHONHASHSEED` 一致（子进程验证）、同会话恒定、熔断只影响相关会话、单账户 no-op、pinned 优先、端到端三轮同账户 |

测试**不依赖环境变量**：三个异步类通过 `_PinnedSettings` 把依赖到的开关钉成显式值，
`COPILOT_OPAQUE_STATE_RECOVERY` 取 unset / `true` / `false` 三种情况下都是 27 passed。

kill switch 端到端实测（真跑代码路径，非 mock 断言）：

```
COPILOT_OPAQUE_STATE_RECOVERY=true   上游调用=2  rung2触发=True   强制token刷新=0  recovery={attempted:1, exhausted:1}
COPILOT_OPAQUE_STATE_RECOVERY=false  上游调用=1  rung2触发=False  强制token刷新=0  recovery={exhausted:1}
```

---

## 5.5 明确评估后决定不修的

「决定不修」也要有据可查，否则和「没发现」无法区分。

| 项 | 为什么不修 | 重新考虑的触发条件 |
|---|---|---|
| 流式路径的 6× 客户端重试放大 | 要覆写 `stream_response` 延后 `http.response.start`，会把响应头延后到第一项（长 thinking 下可能 15s），且要动 `docs/STREAM_OWNERSHIP.md` 管的所有权窗口；根因已被会话亲和消除。详见 §4.3 的方框 | 亲和正常但 `opaque_state_requests_total` 仍增长，且重试放大在压上游配额 |
| `usage_store` 的 `is_error` 死链路 | 生产没有任何调用点传 `is_error=True`。接活需要在失败路径调 `record()`，而 `_flush` 会无条件 `requests += 1` —— 那会把 per-model `requests` 的语义从「成功数」改成「尝试数」，影响 `/stats` 与历史成本口径。**这是语义决策，不是机械修复**，不该以「修缺陷」的名义悄悄改 | 需要「按天持久化错误数」这个能力时，先定义清楚错误请求算哪个模型、算不算一次 request |
| `global_stats.total_errors` 重启归零 | 与上一条同源：token/requests 会从磁盘恢复，errors 不会（那个字段结构性恒 0）。`/stats` 因此在重启后显示 `total_errors: 0`。行为已在 CLAUDE.md 写明 | 同上 |

## 6. 未实测 / 属于推断的部分

**必须读这一节再决定发布。**

1. ~~rung 2 能否救回 `does not belong to this connection`：推断~~ → **已于 2026-09-10 实测确认**，
   见 §3.5。两个真实 GHCP 账户跨账户回放，双向 × 3 次重复 = **24/24 确定性**。此项不再是未验证项。
2. ~~多账号场景未测~~ → **已实现并端到端实测**（§4.6）。用两个真实 GHCP 账户 + 真实 codex-cli
   跑通了「亲和开 = 零拒绝」与「亲和关 = 3 次拒绝全部被阶梯救回」两组对照。
3. ~~未跑真实上游 e2e~~ → **已跑**：§4.6 的对照是经过完整 LB 的真实端到端（真实两账户、真实
   Codex CLI、真实 GHCP）。仍未做的是**生产环境**的 e2e（那需要发布，见 §7）。
4. **Mac 客户端 / VS Code 等价性未验证。**
5. ~~`orphaned_conversation_state` 的客户端行为未验证~~ → **已实测**（§3.6）：400 时 codex-cli
   把整个错误体逐字显示给用户（含 `code` 与「开新会话」指引），且只打 1 次上游。

### 6.1 自审查阶段发现并修掉的两个缺陷（发布复核请重点看这两处）

不是「顺手优化」，是**本次改动引入的真缺陷**，用真实代码路径复现后修的：

| 缺陷 | 后果 | 修法 | 回归测试 |
|---|---|---|---|
| 三次原地修复吃满 attempt 循环 → 生成器静默结束 | 下游 HTTP 200 + **0 字节**、无终端事件，日志误记为 `client_disconnect`；按 §14 机制会毒化会话 | 两处原地修复带 `attempt < max_retries - 1`；循环后兜底 `retry_budget_exhausted` | `RetryBudgetFloorTests`（含提高 `MAX_RUNGS` 验证不变量） |
| `unverifiable_content` 只匹配 `could not be verified` | 误吞不相关 400（如 `The provided API key could not be verified`）→ **无理由删掉用户的推理 blob**，并把真实原因改写成 `orphaned_conversation_state` 藏起来 | 判据要求同时命中「主题是 encrypted content」与「校验失败措辞」 | `test_unrelated_verification_failures_are_not_opaque_state` |

第一条尤其要注意：它把「一次可恢复的失败」变成「静默断流」，而静默断流正是这一整条问题链的**根源**。

---

## 7. 发布前要做的（对应交接文档 §10）

1. 在隔离验证环境跑经 LB 的 e2e：`/v1/responses` 流式与非流式各若干轮多轮对话，确认无回归。
   **`8/8 回显通过`不能代替状态恢复验收** —— 那条路径根本不触发本次改动。
2. 抓改动前后的指标差值，重点：
   - `copilot_opaque_state_requests_total`（规模）。**注意基线可能长期是 0** —— 观测窗口内
     那 6 个事件已停止增长（三次抓取跨流量 ~95 → 218 → 381 零新增）。所以「上线后这个指标是 0」
     **既可能是修复生效、也可能只是没再发生**，不能据此判定成败；要判定得等
     `attempted` 非 0 之后再看 `succeeded/attempted`
   - `copilot_stream_retry_budget_exhausted_total` **必须为 0**（见 §4.2b：非 0 说明循环不变量
     被破坏，客户端本会收到空 SSE 流；按回归处理，先 `COPILOT_OPAQUE_STATE_RECOVERY=false` 再排查）
   - `copilot_opaque_state_recovery_total{outcome="succeeded"}` 对 `{outcome="attempted"}`（命中率）
   - `copilot_orphaned_item_id_events_total{stage="detected"}` **应基本不变**（每次拒绝仍 +1；
     第二次拒绝来自恢复尝试而不是无用的 token 刷新）
   - `copilot_orphaned_item_id_events_total{stage="unrecoverable"}` 相对每请求应**约减半**
     （只在阶梯走完后记一次）
   - `copilot_token_refresh_total` 增速应下降（不再有浪费的强制刷新）
   - `copilot_upstream_401_total{scope="endpoint"}` 必须仍为 0
   - `copilot_session_affinity_total{outcome=...}`：**单账户下 hit/unavailable 必须恒为 0**
     （非 0 说明配置里意外多了合格端点）；`absent` 会随非 Codex 客户端流量增长，属正常
3. 按完整 digest 固定镜像，surge-first 保证 Ready ≥ 1，保存可回滚的旧 digest。
4. 不改 ConfigMap / Secret / `openclaw.json`；不动 Copilot endpoint 数量。
5. 出问题的快速回退顺序：
   1. `COPILOT_OPAQUE_STATE_RECOVERY=false`（关 rung 2，不用重建镜像）
   2. 仍有问题 → 回滚到旧 digest
6. **不要**先上线再试是否修好。

---

## 8. 观察清单（上线后）

```promql
# 携带坏状态的会话有多少救不回来
copilot_opaque_state_recovery_total{outcome="exhausted"}
  / copilot_opaque_state_requests_total

# 恢复命中率 —— 把 §6.1 的推断变成实测
copilot_opaque_state_recovery_total{outcome="succeeded"}
  / copilot_opaque_state_recovery_total{outcome="attempted"}

# 真金丝雀：事后还能剥到 id ⇒ 主动剥离漏了一个通道
copilot_orphaned_item_id_events_total{stage="recovered"} > 0

# 账户级健康：必须恒 0
copilot_upstream_401_total{scope="endpoint"} > 0

# 循环不变量：必须恒 0（非 0 = 客户端本会收到 HTTP 200 + 空 SSE 流）
copilot_stream_retry_budget_exhausted_total > 0

# 单账户下必须恒 0；加了第二个账户后 hit 应该远大于 unavailable
copilot_session_affinity_total{outcome="hit"}
copilot_session_affinity_total{outcome="unavailable"}

# unavailable 与受影响请求数应同步；背离说明有别的跨账户来源
copilot_session_affinity_total{outcome="unavailable"} / copilot_opaque_state_requests_total

# 根源：orphan 的上游是我们自己的断流（剥离只阻止它升级为会话中毒）
copilot_stream_truncated_no_completion_total
copilot_pool_timeout_total
copilot_upstream_html_events_all_total
```

`succeeded/attempted` 长期接近 0 ⇒ §6 的推断不成立，应关掉 rung 2 并回到根因调查（跨账户状态
回放的亲和方案）。接近 1 ⇒ 推断成立，可以把这段文字从「推断」改写为「实测」。

**但先看 `attempted` 有没有非 0。** 那 6 个事件在观测窗口内已停止增长，所以完全可能上线很久
`attempted` 仍是 0 —— 那既不能证明修复有效、也不能证明无效，只说明故障没再复现。这条推断只能
等真实事件到来才能验证；在此之前不要把「指标全 0」当成验收通过。
