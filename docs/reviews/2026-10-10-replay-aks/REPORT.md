# 协议保证落地与重放白名单扩展 AKS 发布回执

Author: Zeno Ren

2026-10-10 已将 [PR #23](https://github.com/ZenoRewn/databricks-claude-lb/pull/23) 与 [PR #24](https://github.com/ZenoRewn/databricks-claude-lb/pull/24) 合并后的 `5f2463e0124f0d18e699aea64e01ff672dad074e` 发布到 AKS。集群回执 **succeeded**，北京时间 **16:07:00–16:08:07** 维护并恢复流量，**66.097 秒**，全部 action verified，三个常规业务探针、账本与清理通过，未回滚，清理错误为空。

**本次是推理 POST 重放白名单自编写以来的首次扩展。** 该变更有 RFC 依据与多重约束，但在生产上未经验证，见下文边界。

## 生效内容

### REFUSED_STREAM 进入重放白名单

RFC 7540 §8.1.4 规定远端 `RST_STREAM` 携带 `REFUSED_STREAM` 表示流在「任何处理发生之前」就被关闭，请求「可以安全重试」。[可靠性契约](../../RESILIENCE.md) 的不变量禁止的是执行状态**不明**时重放；此处执行状态是**已知的（未执行）**，因此不构成不变量冲突。

2026-10-09/10 观测到 8～12 次该错误码，全部 `upstream_headers_received=false`、`chunks_yielded=0`，却逐个变成客户端可见失败。

约束保持完整：证据只取自 typed h2 事件；本地发起的重置（`remote_reset=False`）不算；连接级 `GOAWAY` 携带 7 不算；`CANCEL(8)`、`INTERNAL_ERROR(2)`、`PROTOCOL_ERROR(1)`、`ENHANCE_YOUR_CALM(11)` 一律排除。调用方仍要求 `response is None`、未向下游输出内容、以及既有 attempt 预算。该错误码**仍然计入共享熔断**，所以上游持续拒绝时熔断会打开并终止重试 —— 这是防止重放放大过载的闸门。

### GOAWAY + NO_ERROR 不再累积熔断错误

RFC 7540 §6.8 的 `GOAWAY` 携带 `NO_ERROR` 是有序关闭，不是故障。2026-10-09 在 21 秒内观测到 6 次，其中几条已输出至多 3692 chunks，当时正把一次正常的连接回收推向 `failure_threshold`。

流确实丢失、客户端确实看到失败；但端点并不不健康（新连接可用），所以记为 neutral。已应用到全部 6 个 provider 调用点。它**不可重放** —— 与 `REFUSED_STREAM` 不同，连接干净关闭并不说明该流是否已被处理。携带真实错误码的 `GOAWAY` 仍按故障计入，neutral 也不改变影响范围（仍为 `endpoint`）。

### 其余两项

裸 `response.failed`（无 error 对象、无 usage）的 `failure_reason` 由 `unknown` 改为 `upstream_failure`。该值是既有兜底，语义为「上游失败、无进一步分类」，不编造原因；`response.incomplete` 不动。

新增 `elevated_input` 提示层，`LB_CONTEXT_ELEVATED_INPUT_BYTES` 默认 1 MiB。**该默认值是启发式而非证据阈值** —— 目的只是让 1.58/2.20/3.21 MB 请求不再与 256 KiB 共用标签；同窗口内 >1 MB 请求仍有 103/179 成功。不拒绝、不裁剪、不摘要，已核验限额比值仍优先于体积分级。

实际 Deployment spec 仅改变 image 与 source-revision 注解；单副本、资源 requests、Service selector、Ingress、配置/凭据、三探针、preStop、grace 与环境变量均未改动。

## 固定身份与验收

| 项 | 值 |
|---|---|
| registry manifest | `@sha256:691109f90c658a78073e31c6d6d1a3af174f529cc34fe0126108a3b9ef41e003` |
| 平台 | linux/amd64 单平台 manifest v2 |
| config digest | `sha256:f74d092e2f70f614037488cad35242951e6031dd1080daa02161c49cb0f40ecf` |
| source manifest | `d3cbcb194f6c10576531a0c30fa5675aee59954a78398f0370e77488da4171f2` |
| 回滚锚点 | 前一版本 `97c7e64` 的 `@sha256:6cf3b76a…` |
| 计划 hash | `0975a4d3b12d71fb54794d87a56b068da3ba165e3947eec50def5e7ae2366b21` |
| 公网 `/version` | `5f2463e` / `verification=matched`，零重启 |

两个 PR 与合并后 `main` 的六项 CI 均通过。本机全量 870 passed、772 subtests passed、6 skipped。正式镜像内 787 passed、712 subtests passed、7 skipped、零失败，应用源码未经挂载替换、容器 `--network none`。

镜像内与发布后运行进程均逐项读回新行为，而非仅依赖单元测试：

```
REFUSED_STREAM(7) 可重放 = True      CANCEL(8) 可重放        = False
ENHANCE_YOUR_CALM(11)    = False     本地重置(7)             = False
GOAWAY NO_ERROR neutral  = True      GOAWAY code=1 neutral   = False
ELEVATED_INPUT_BYTES = 1048576       3.21MB advice = elevated_input
```

两个 Service 各恢复一个 Ready 新 Pod 后端，selector 读回 `app=claude-lb`，无维护标识残留；公网三个健康入口均 200。

## 未验证边界

**重放 `REFUSED_STREAM` 是否真能减少客户端可见失败、以及是否会在上游限流期间加重过载，均未经生产验证。** 依据是协议文本与本地注入测试，不是现网观测。熔断仍计数是设计上的兜底，但其实际效果同样未验收。

`GOAWAY` neutral 是否改变恢复行为未验证。上游的三类拒绝 —— 120 秒 `CANCEL`、`REFUSED_STREAM` 限流、裸 `response.failed` —— 均未解决，本次发布只让它们可诊断或可恢复，不是修复。`elevated_input` 的 1 MiB 边界是否合适未验证。

本次未执行数据库快照与隔离恢复验证，也未安排独立 QA 复核。熔断状态仍为进程内状态，Pod 重启使计数归零不表示上游恢复。
