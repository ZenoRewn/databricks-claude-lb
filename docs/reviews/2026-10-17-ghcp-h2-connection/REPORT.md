# 2026-10-17 GHCP 再次 HALF_OPEN：一条 h2 连接被用到服务端不再接受新流

Author: Zeno Ren

> 范围：本地源码与文档，**未发布 AKS**。生产取证来自 live Pod 日志与 `/metrics`，未改生产配置、
> 未启用账户、未轮换凭据。恢复动作是运维侧的一次小请求试探，没有重启。

## 症状

Codex 调用间歇失败，客户端看到：

```
RemoteProtocolError: Upstream request could not complete; execution may have occurred.
(connection_id=... chunks_yielded=0 first_event=None last_event=None
 probe.ok=True dns_ms=27 tcp_ms=229 ips=140.82.114.22)
```

`ZenoRewn` 熔断 OPEN，`consecutive_errors=5`，9 分钟内两轮试探都以 `trial_inconclusive` 收场。

## 根因：一条连接用了约 340 个流，服务端开始拒绝新流，而池子毫不知情

10 次失败的 `http2_stream_id` 严格 +2 递增，`httpx_pool` 全程 `total=1` —— **所有 Copilot 流量
复用同一条 HTTP/2 连接**：

```
683 code=8 headers=True  frames=469 idle=120.1s
685 code=7 headers=False
687 code=8 headers=True  idle=120.0s
689 code=7 headers=False
691 code=8 headers=True  idle=120.0s
695 697 699 701 code=7   703 code=8 idle=120.0s
```

两种模式在同一条连接上交替：

| h2 code | 形状 | 含义 |
|---|---|---|
| **7 REFUSED_STREAM** | `headers=false`，瞬间失败 | 服务端**不再接受这条连接上的新流** |
| **8 CANCEL** | `headers=true`，空闲精确 120.0s | 已接受的流被饿死后取消 |

**日志里 GOAWAY 出现 0 次。** 这是全部问题的关键：httpcore 的
`AsyncHTTP2Connection.is_available()` 只看 `_used_all_stream_ids`、`_connection_error` 和 h2 状态机，
静默的 REFUSED_STREAM 一个都不触碰，所以池子认为这条连接完全健康，**继续把它交给每一个新请求**。

### 重试全部打回同一条死连接

```
req 1605f0cc  attempt0 → 685 REFUSED   attempt1 → 689 REFUSED
req b409791b  attempt0 → 695 REFUSED   attempt1 → 699 REFUSED
req 6848db2f  attempt0 → 697 REFUSED   attempt1 → 701 REFUSED
```

REFUSED_STREAM 按 RFC 7540 §8.1.4 判为可重放是**对的**，但在刚刚拒绝过的连接上重试，结构上必然
再被拒。它唯一的效果是把错误数翻倍 —— 转换事件实测 `contributing_attempts=5 /
contributing_requests=3`，正是这个把阈值 5 填满的。

## 排除的两个假设

**不是请求体积。** 断崖前十三分钟，同样 0.9–3.0 MB 的 body 完成了 **112 个**，
`send_to_headers` p50 5.9s / max 17.4s，传输失败 0；06:37 之后大请求完成数直接变成 **0**。

| 时间 | ≥0.9MB 大请求 |
|---|---|
| 06:24–06:36 | completed=112，失败 0（单分钟峰值 15 个完成） |
| 06:37–06:50 | completed=**0**，failed=6 / http_error=12 / 放弃=3 |

所以**压缩请求体不会有任何帮助** —— 上传路径已被证明能吃下 3 MB。体积只是**占用时长**的代理：
`gpt-6.1-sol` 是推理模型，第一条失败的 `frames=469`、`decoded_bytes=275016` 但
`downstream_content_started=false`（发了 275 KB 思考流还没出内容），大请求把流按分钟占着，于是
连接进入死状态时正在占流的、以及要开新流的都是它们。小请求 5 秒就结束，大多躲过了窗口。
这也解释了「单独调用 GHCP 没问题」：新连接、低 stream id，碰不到上限。

**不是多路复用压力。** 并发度与失败负相关：6 次传输失败全发生在在途 1–2 个时，而在途 ≥3 个的
90 个请求里传输失败是 **0**。

## 本轮改了什么

### 连接退休（核心）

`_retire_h2_connections()` 把池中 h2 连接的 `_used_all_stream_ids` 置真。这**只**让
`is_available()` 返回假，即把连接从「可分配给新请求」里移出；httpcore 关闭连接的条件是
`is_closed()` / `has_expired()` / 空闲过量，**不包含「不可用」**，所以：

- 还在上面跑的流**不受影响**，继续跑完
- 每个新请求去开一条新连接
- 已经空闲的连接立刻过期（没有流要保护，没必要留到 keepalive 到期再被拒一次）

这正是服务端没发的那个 GOAWAY 的语义。相比之下重建整个 client（`/admin/copilot/reset-pool`
的做法）会打断在途响应，所以不适合自动触发。

三个触发条件：

| reason | 判据 |
|---|---|
| `refused_stream` | 远端 REFUSED_STREAM（code 7）—— 连接层面的声明，**在重试之前**退休 |
| `prefirst_content_stall` | 收到响应头后静默超过 `COPILOT_STREAM_PREFIRST_CONTENT_WARN_SECONDS`（默认 90s）。不动那条流（服务端 120s 自己会取消），只拦新流 |
| `stream_id_cap` | 出向 stream id 达到 `COPILOT_H2_RETIRE_STREAM_ID`（默认 400）。主动轮换，不去发现服务端那个没有文档的上限 |

本地 reset 不触发退休（`remote_reset=False` 对对端毫无证明），CANCEL / INTERNAL_ERROR /
PROTOCOL_ERROR 等也不触发 —— 它们描述单条流的结局，不是下一条流的。

### 用私有属性这件事说清楚

httpx / httpcore **没有**「停止把新流给这条连接」的公开 API，唯一的公开替代是重建 client，而那会
打断在途响应。所以这是一个有意的窄赌，并且只有在「库升级后会在测试里大声失败」的前提下才可接受：
`tests/test_h2_connection_lifecycle.py::PoolContractTests` 钉住了 5 条假设 —— 包括
`is_available()` 仍然看 `_used_all_stream_ids`，以及**池子不会因为「不可用」就关连接**（整个优雅
退休的设计全压在这一条上）。任一条变了，测试红，而不是生产上静默退化成空操作。

所有内省都包在 try/except 里，失败时返回 `error` 字段并记日志，绝不让诊断逻辑打断请求路径。

### 记录（下一次要能直接认出来）

| 信号 | 用途 |
|---|---|
| `copilot_h2_connection_retired` 事件 | `retirement_reason` / `retired_connections` / `h2_stream_id_watermark` |
| `copilot_h2_connections_retired_total{reason}` | 按原因计数，区分「被拒」与「我们主动轮换」 |
| `copilot_h2_stream_id_watermark` | **故障前就能看见的信号**：连接复用时每请求 +2，逼近上限时上升，退休后回落。无 h2 连接时不发样本（发 0 会让「没有连接」与「全新连接」同形） |

新字段已加入 `safe_diagnostics` 白名单 —— 实现时测试立刻抓到了漏加（第三次踩同一个坑，它现在
是 CLAUDE.md 里的一条不变量）。

## 有意没做

| 没做 | 理由 |
|---|---|
| 首事件 45 秒截止线（中止请求） | 成功样本首事件 p50 6.4s / max **18.0s**，而失败在 91.5s 仍静默，中间 72 秒空白，阈值很安全 —— 但那些静默的流**都在那条将死的连接上**。先修连接，再看这个模式是否还存在；在一晚 n=6 的证据上引入「杀用户请求」的超时，是会反咬的那类改动 |
| 关掉 h2（`COPILOT_HTTP2=false`） | 零代码的缓解手段，但会失去多路复用。连接退休保留了 h2 又消除了共享毒化，更对症。env 开关仍在，作为退路 |
| 改熔断计数单位 | 2026-10-16 的反论未变：按业务请求计数会让端点真死时保护晚得多。退休已经从源头上消掉了「重试撞死连接」这个放大器 |

## 仍未确认

1. GitHub 那个拒绝上限究竟是流数、连接寿命还是字节数 —— `stream_id_watermark` 会随时间给出答案
2. `idle=120.0s` 那个精确超时属于哪一层（边缘代理还是推理后端）
3. 单账户故障域未动：`gpt-6.1-sol` 占本窗口请求的四分之三，而只有一个账户启用，Azure 不服务该模型，
   所以熔断一开就是 `upstream_unavailable`、`endpoint=null`

## 验证

```
python3 -m pytest tests/ -q      → 1325 passed, 6 skipped, 1305 subtests
node tests/test_copilot_pricing_ui.cjs
node tests/test_dashboard_ui.cjs
pyflakes（一方代码）             → 零告警
```

新增 `tests/test_h2_connection_lifecycle.py`（27 项 + 18 subtests），含一条端到端用例：让真实的
REFUSED_STREAM 走完 `_stream_response`，断言连接被退休且**只报告一次**（不是每次 attempt 一次）。

**未发布 AKS。** 发布时机等通知。
