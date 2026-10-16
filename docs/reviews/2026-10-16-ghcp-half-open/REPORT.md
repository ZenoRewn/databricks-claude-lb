# 2026-10-16 GHCP HALF_OPEN 事件：证据口径与两个缺陷修正

Author: Zeno Ren

> 范围：本地源码改动与文档，**未发布 AKS**，未改生产配置，未启用任何账户，未轮换凭据。
> 事件取证来自独立复盘（`GHCP_HALF_OPEN_复盘与优化评审.md`，未纳入本仓库），本回执只记
> 「核验结论 + 本轮改了什么 + 有意没改什么」。

## 事件（取自复盘，已对代码核验）

| UTC+8 | 事件 |
|---|---|
| 11:20:50.738 | `ZenoRewn` 端点 CLOSED→OPEN，`failure_threshold`，连续错误 5 |
| 11:21:55.699 | 大请求（very_large、流式、16 张图、带 opaque state）准入为半开试探 |
| 11:23:25.824 | 该试探 `client_disconnected`、未取得上游响应头 → `trial_inconclusive` → 又一轮 60s |
| 11:29:49.144 | 一次人工 `say ok` 小请求获得试探资格 |
| 11:29:51.424 | `trial_succeeded`，熔断关闭 |

9 分钟不可用；恢复**靠人**，不是自动。打满阈值的 5 次错误来自 **2 个**业务请求
（一个 small 重试 3 次 + 一个 very_large 失败 2 次）。

## 核验：复盘里要修正的一条

复盘称「日志缺真实事件时间」。**数据没丢**：生产是 `LOG_FORMAT=json`
（`deploy/k8s/deployment.yaml`），该 formatter 输出 `ts`（取 `record.created`）。

真正的缺陷窄一些但确实存在：**事件载荷自身**只有 `started_at_unix`（请求起点，同一请求
所有事件共用），于是读 JSON 体的人自然把它当事件时间。这正是前一轮分析里 **8 条表述中
4 条**被撤回的原因。默认的 text 格式则连时间都没有。

## 本轮改了什么

### 1. 事件自描述（`event_time` / `event_monotonic`）

`log_event()` 为每条结构化事件注入两者；`started_at_unix` 含义不动。锚定测试直接复现那个
坑：三次相隔两分钟的 attempt 必须有三个不同时间，而 `started_at_unix` 只有一个。

实现中撞到一处：新字段**必须**加进 `safe_diagnostics` 白名单，否则被静默过滤。

### 2. 重试继承剩余预算（唯一的行为修正）

`effective_startup_seconds()` = `min(固定预算, 剩余 deadline)`，`startup_budget_source`
说明是否被 deadline 绑住；剩余不足 `UPSTREAM_MIN_RETRY_SECONDS`（默认 5s）则**不开始**重试。

此前 `startup_seconds()` 返回固定值，`remaining_seconds()` 虽已存在却**只用于一个日志
字段**。

**说清楚：这条修正不会改变本次事件的结果。** 总预算默认 1800s、启动预算 180s，而本次
整笔请求只用了 367s —— 第三次 attempt 开始时剩余约 1558s，远超 180s，clamp 不触发，
`retry_within_budget()` 也不会拦。每次 attempt 约 121s 是**上游 RST** 决定的，不是预算。

它修的是结构：上游若从「121s 后 RST」变成「挂住不答」，三次重试就是 3×180s；而配了
`LB_STARTUP_BUDGET_OVERRIDES`（例如某路由 600s）时，三次重试能直接吃掉整个 1800s 总预算，
最后一次必然在 deadline 之后才开始 —— 那是纯粹的浪费。这条是「本来就该这样」，不是本次
事件的解药。

另一个口径差异值得记下但本轮未动：**客户端约 90s 就放弃了，而网关总预算是 1800s。**
网关无法知道调用方的 deadline，只能等它断开；断开时现有代码确实会取消上游（本次
`client_disconnected` 正常传播、试探也正常结算），所以这一侧是工作的。要进一步对齐需要
客户端侧日志，属复盘列为「尚未确认」的第 4 项。

安全边界未动：可重放性仍只由 `_replay_safe_transport_failure` 判定（它还决定终态状态码，
与剩余时间无关），预算由 `retry_within_budget()` 判定 —— 两个问题，两个函数。

### 3. 恢复可见性

- 试探准入事件补 `small_input_evidence` / `preference_window_remaining` /
  `admitted_payload_small` / `admitted_estimate_complete` —— 小请求优先机制本来就存在，
  但没有任何字段说明它当时**有没有证据可依**，所以「它生效了吗」无从回答
- inconclusive 结算补 `inconclusive_kind`（cancelled/neutral）与 `upstream_headers_received`
  —— 未收到响应头的取消对上游**零证据**，收到了的至少证明上游接了流。无测量上下文时为
  `null` 而不是 `False`：缺席不等于否
- `resilience_stats`（因而 `/stats` 与 `/metrics`）补 `cooldown_elapsed_seconds` /
  `seconds_since_last_trial` / `last_trial_outcome` / `last_trial_age_seconds` /
  `trial_in_flight_age_seconds`。那 9 分钟从此是一个**数字**
- 跳闸事件补 `contributing_attempts` / `contributing_requests`（本次会是 5 / 2）

`trial_in_flight_age_seconds` 为 `None` 时**不发 metric 样本**：发 0 会让「没有在途」与
「刚开始」变成同一个样本。

### 4. 图片判定收敛为一个实现

仓库里有三个图片探测器、三套规则，而最窄的 `_has_image` 恰好门控
`Copilot-Vision-Request` 头 —— **漏判等于把带图请求裸发给上游**。

| 形状 | 旧 `_has_image` |
|---|---|
| 图片在 tool 输出等更深层 | 漏 |
| `content` 是 mapping 而非 list | 漏 |
| Anthropic 的 `type: "image"` | 漏 |
| 裸 `data:image/...` 字符串 | 漏 |

同一个 body 因此能同时记下 `image_count=16` 与 `has_image=False`（复盘实测）。现在
`_has_image` 委托 `model_capabilities.payload_has_image()`，与 `estimate_payload` 同一套
遍历；假阳性那侧由遍历本身兜住（`tools` 等 definition 子树不计图片，描述图片的 schema
不是图片）。超量裁剪的 `collect` 未动 —— 它求的是 `(parent, index)` 定位，是另一个问题。

**这不是根因结论。** 它是独立成立的缺陷，可能影响带图的那一部分；但触发最终熔断的是一个
**没有图**的小请求，同样的 code 7。

## 有意没改什么

三项都评估过、都有支持它的理由、都被留下，并由
`tests/test_circuit_admission.py::DeliberatelyUnchangedTests` 钉住，理由见
[docs/RESILIENCE.md](../../RESILIENCE.md#deliberately-unchanged)。

| 没改 | 理由 |
|---|---|
| 熔断仍按 attempt 而非业务请求计数 | 按请求计数会让端点真死时保护晚得多（每个失败请求先烧掉自己的 3 次长等待）；调大阈值更糟，只是让更多请求去经历长时间失败。改为把放大倍数报出来 |
| inconclusive 试探仍重置整轮冷却 | 缩短它恰好能救本次那 60s（客户端断开、上游零证据），但整轮冷却正是防 cancel/reprobe 风暴的东西，**一次事件的证据不足以放宽保护边界**。判据已记录，策略等数据 |
| 不加后台恢复探测 | 它会发**真实、计费**的推理请求 —— 那次关掉熔断的 `say ok` 本身就是真调用真计费，不是只读健康检查。改为把等待暴露成数字，介入与否仍是人的决定 |

## 仍未确认

1. RST_STREAM 来自哪一层、为何发生（上游容量/背压、网络路径、连接复用、特定负载交互）
2. 多个递增 stream_id 是否来自同一上游连接
3. 输入大小、图片数量、opaque state 与错误的因果关系（无同负载受控 A/B）
4. 90 秒取消的责任层（客户端、SDK、入口代理还是别处）
5. 单账户故障域 —— 启用备用账户需先验模型/API/鉴权/会话亲和，本轮未动

## 验证

```
python3 -m pytest tests/ -q      → 1297 passed, 6 skipped, 1287 subtests
node tests/test_copilot_pricing_ui.cjs
node tests/test_dashboard_ui.cjs
```

新增测试：`tests/test_vision_header_detection.py`（12 项 + 10 subtests）、
`test_request_budget.py::RetryBudgetTests`（8 项）、
`test_operational_evidence.py::EventTimeTests`（2 项）、
`test_circuit_admission.py` 的 `RecoveryEvidenceTests`（10 项）与
`DeliberatelyUnchangedTests`（3 项）。

**未发布 AKS。** 本轮全部是源码与文档改动；发布时机等通知。
