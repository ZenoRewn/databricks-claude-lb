# 120 秒上游取消的生产取证

Author: Zeno Ren

日期：2026-10-09（北京时间）。本页记录 `fea7d60` 上线后首次拿到 typed protocol code 的生产事件。证据来自运行 Pod 的结构化日志与 `/stats` 读回，不含请求正文、prompt、工具内容或凭据。

## 触发与症状

客户端（Codex）收到 `503 recovery_prefers_small_request`，request id `5006eada9b4290e9dba33ee451b82b5e`。该请求 `body_size_bytes=2198326`（2.2 MB）、`admissions=0`、`upstream_sends=0`、`error_origin=local`，是本地拒绝，从未发往上游。它是最外层症状，不是故障源。

## 三类故障的 typed 分类

| 时间（CST） | protocol_error_kind | http2_error_code | 次数 | 关键字段 |
|---|---|---|---|---|
| 12:23:09–12:23:30 | `http2_goaway` | **0** (NO_ERROR) | 6 | `last_stream_id=2147483647`，`chunks_yielded` 223–3692，`upstream_idle_seconds` 0.008–0.066 |
| 13:16:51 起 | `http2_stream_reset` | **7** (REFUSED_STREAM) | 8 | `upstream_headers_received=false`，`chunks_yielded=0` |
| 13:19:05–13:29:47 | `http2_stream_reset` | **8** (CANCEL) | 5 | `upstream_headers_received=true`，`chunks_yielded=0`，`first_event=null` |

第一类不是故障。`NO_ERROR` 配合 `last_stream_id` 为 `2**31-1` 是优雅关闭，上游在回收共享连接；当时几条流已输出数千 chunk，空闲间隔都在 70 ms 以内。

第二类的协议语义保证上游**未执行**该请求：`REFUSED_STREAM` 表示流被拒绝建立，且本次全部未收到响应头。这是账户级过载/限流信号。

第三类是本轮要定位的形态。

## CANCEL(8) 与请求体积的关联

五次 `CANCEL` 的 `upstream_idle_seconds` 与对应请求体积：

| upstream_idle_seconds | body_size_bytes |
|---|---|
| 120.103448 | 365,215 |
| 120.000004 | 770,018 |
| 120.000406 | 2,198,326 |
| 120.000871 | 2,198,326 |
| 120.000985 | 2,198,676 |

形态与 2026-10-08 那次未归因的断流一致：上游返回 200、`upstream_headers_received=true`，随后在约 120 秒内没有任何正文，然后流被重置。区别是这次有 typed code，可以确定是 **stream 范围的上游取消**，而不是连接级关闭或裸 EOF。

> **2026-10-10 修正**：本节原文称「四个读数落在同一毫秒内，这是定时器而非网络抖动」。次日事件把样本扩到 12 个，其中 3 个明显偏高（120.051215、124.776939、134.360068）。原表述基于 5 个样本，过度概括了聚集程度。成立的证据是**硬下界**：12 个读数没有一个低于 120，而 `upstream_idle_seconds` 量的是最近活动到本地捕获异常的间隔，调度与 pump 积压只会放大它。结论「存在 120 秒定时器」不变，依据改为下界而非聚集度。合并样本与判读方式见 [排障指南](../../TROUBLESHOOTING.md)。

同一个 `2,198,326` 字节的会话在本次事件中既撞了 2 次 `CANCEL(8)`、又撞了 3 次 `REFUSED_STREAM(7)`，最后撞上 `recovery_prefers_small_request`。客户端在重试同一请求体。

**边界**：`protocol_scope=stream` 证明重置只影响该流，不证明哪个组件持有这个 120 秒定时器 —— GHCP 服务端与支持 HTTP/2 的中间代理都可能发出 `RST_STREAM(CANCEL)`。五个样本上的体积相关性不构成体积阈值，也不授权据此设置准入上限。未验证"减小请求体积可避免该取消"的因果。

## 传导链与两项保护的实测局限

1. 大请求撞 120 秒 `CANCEL`，客户端重试同一请求体。
2. 重试叠加后上游开始 `REFUSED_STREAM`。
3. 失败累计至 `consecutive_errors=7`，**共享 endpoint 熔断打开**，`circuit_generation` 推进到 7，期间反复 `trial_admitted → trial_failed / trial_inconclusive`。
4. 每次进入 HALF_OPEN 都重新开启 10 秒偏好窗口；2.2 MB ≫ 64 KiB 阈值，该工作负载在每个窗口期都被推迟。

事件中段的现场读回：

```
ZenoRewn        circuit_open=true  consecutive_errors=7   (598 requests / 19 errors)
  gpt-6.1-sol   HALF_OPEN  errors=5
  gpt-5.4       CLOSED     errors=0
  gpt-5.6-luna  CLOSED     errors=0
  gpt-6-astra   CLOSED     errors=0
```

**分层熔断没有隔离这个形态。** `CANCEL(8)` 确实按设计记入了 `gpt-6.1-sol` 的局部层，但主导失败是 `REFUSED_STREAM(7)`，它作为账户级过载信号保留共享保护。结果是三个零错误的模型被共享熔断一起挡住。该设计判断本身未被此事件推翻 —— GHCP 确实在限流账户 —— 但"局部熔断能隔离单模型故障"的预期在此形态下不成立。

**偏好窗口的过期保证是按窗口成立的，不跨越反复开合的熔断。** 原文档称窗口过期后任何合格请求都能取用试探，以避免大请求无限饥饿；实测中熔断每 30 秒冷却后重新 HALF_OPEN，于是又产生一个新窗口。只有大请求的工作负载因此被系统性延后。失败分布为 `upstream_unavailable` 79、`recovery_preference` 10，说明多数拒绝来自熔断本身，偏好窗口是叠加的额外惩罚而非主因。

## 请求体积与成功率

同期 3 小时窗口内按请求体积分组（`lb_request_received.body_size_bytes` 关联 `lb_request_end.outcome`）：

| 体积 | completed | 总计 | 成功率 |
|---|---|---|---|
| > 1 MB | 103 | 179 | 57.5% |
| ≤ 1 MB | 494 | 595 | 83.0% |

这是同一时间窗、同一上游状态下的观测对比，不是控制实验；大请求也更多来自同一个重试中的会话，样本不独立。不能据此给出体积与成功率的函数关系。

## 处置与后续

现场将 `COPILOT_RECOVERY_PREFERENCE_SECONDS` 设为 `0`，仅追加该一条环境变量，镜像、`source-revision` 注解、副本数、preStop 与 grace 均未改动，server dry-run 核对后应用。新 Pod 读回该值为 `0.0`，`COPILOT_SCOPED_CIRCUITS` 保持 `true`。此后 `recovery_preference` 归零。

该处置只消除一类本地 503。上游的 120 秒 `CANCEL` 与 `REFUSED_STREAM` 均未解决。新 Pod 上熔断计数归零是**进程重启**所致，熔断状态为进程内状态，不表示上游已恢复。

由本次取证驱动的代码改动见同一 PR：偏好窗口改为依赖近期小输入证据，以及首个内容前停滞的可观测信号。两者都不改变推理 POST 重放白名单，不自动中止流，也不从体积推断准入上限。
