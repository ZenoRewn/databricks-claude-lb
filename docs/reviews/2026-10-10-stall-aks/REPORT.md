# 停滞信号与证据门控 AKS 发布回执

Author: Zeno Ren

2026-10-10 已将 [PR #21](https://github.com/ZenoRewn/databricks-claude-lb/pull/21) 合并后的 `97c7e64fea6b80afa60636a231f357ac82e8042b` 发布到 AKS。集群回执 **succeeded**，北京时间 **12:32:06–12:32:55** 维护并恢复流量，**48.488 秒**，三个常规业务探针、账本与清理通过，未回滚，清理错误为空。

本次发布**首次尝试被 preflight 正当拒绝**，详见下节；该拒绝未改动生产。本页同时记录发布后移除 `COPILOT_RECOVERY_PREFERENCE_SECONDS` 的独立变更。

## 首次尝试被拒绝：账本后端不可用

首次提交（`lb-20261010-stall-6cc1a0`）在 preflight 阶段 0.62 秒内失败并进入 `rolled_back`：

```
last_failure: {action: preflight, code: old_backend_not_healthy, error_type: RuntimeError}
maintenance_started_at: null
cleanup: verified=true
```

`maintenance_started_at` 为 null 意味着从未进入维护 —— 未暂停、未排空、未停止旧 writer，生产未被改动。判定条件是 `operations/release/backend.py` 中的 `diagnostics['accepting_status'] != 200 or diagnostics['metrics'].get('lb_usage_backend_ready') != 1`。当时现场：

```
lb_usage_backend_ready    0
lb_usage_pending_events   646
/health/accepting         200
/health/ready             200
```

即本地就绪而账本后端不可用。Pod 内探测确认：

```
zeno-sea-mysql.mysql.database.azure.com:3306
  DNS ok  35ms  ips=['135.171.164.113']
  TCP FAIL after 8008ms: TimeoutError
```

`az mysql flexible-server show` 返回 `state: Stopped`。同命名空间依赖同一 MySQL 的 `ghost-blog` 与 `umami` 当时分别重启 35 和 36 次且 not ready，与该结论一致。

**这次拒绝是正确的。** 维护必然重启 Pod，而 646 条待写用量事件只存在于内存；按 [可靠性契约](../../RESILIENCE.md) 未 flush 的事件不承诺零 RPO，强行发布会直接丢失它们。发布器没有把「本地 accepting」误当作「可以安全重启」。

启动 MySQL 后复测：TCP 21ms、`lb_usage_backend_ready=1`、`lb_usage_pending_events` 由 646 降至 0，**646 条事件全部落盘，无丢失**。`ghost-blog` 随后恢复 1/1；`umami` 仍在其既有 CrashLoopBackOff 退避中，不属本次发布范围。MySQL 为何处于 Stopped 不在本页断言范围 —— 活动日志显示 2026-10-10T02:26:16Z 有一次管理员 `Create/Update MySQL Server` 操作，是否与该状态因果相关未经验证。

## 本次发布生效内容

- 首个内容前停滞信号：`copilot_stream_prefirst_content_stall_total` 与 `kind=copilot_stream_prefirst_content_stall`，在已收到响应头但超过 `COPILOT_STREAM_PREFIRST_CONTENT_WARN_SECONDS`（默认 90）仍无正文时每流一次。纯观测，不中止流、不重试、不影响熔断。
- 恢复优先窗口改为依赖证据：仅当本轮冷却期内真的到达过估算完整且不超过 `COPILOT_RECOVERY_SMALL_INPUT_BYTES` 的请求时才推迟大请求。证据记录在真实拒绝路径上并绑定观察到它的 circuit generation，新冷却期不继承，管理重置即失效；readiness 不记录证据。
- `seconds_since_headers` 与 `threshold_seconds` 加入诊断字段白名单 —— 此前事件会说明发生了停滞但静默丢弃持续时长。
- 实际 Deployment spec 仅改变 image 与 source-revision 注解；单副本、资源 requests、Service selector、Ingress、配置/凭据、三探针、preStop 45 秒、90 秒 grace 均保留。

## 固定身份与验收

| 项 | 值 |
|---|---|
| registry manifest | `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:6cf3b76a435fc4fd7a9d32047361d01409b84bb163af3013044a84e0d2ad7f85` |
| 平台 | linux/amd64 单平台 manifest v2（非多平台 index） |
| config digest | `sha256:65568cddb0cad155b0a2546bc9062ee1d7370853d923e77d2f67720c18e22743`，与本地构建 image ID 相符 |
| source manifest | `dd4cab6f07b7a44758864bc7c5dfe06c08ece26d485438cd10248fe3fb416d38` |
| 回滚锚点 | 前一版本 `fea7d60` 的 `@sha256:7199477b…` |
| 计划 hash | `84b3fb897c678f72847e3e4c6f13066d71cd969782daaad9ecd3a4208fbdccc6` |
| 公网 `/version` | `97c7e64` / `verification=matched` |

PR 与合并后 `main` 的六项 CI 均通过。本机全量 833 passed、747 subtests passed、6 skipped。正式镜像内 750 passed、687 subtests passed、7 skipped，应用源码未经挂载替换、容器 `--network none`；排除的 8 个文件依赖 `operations` 模块或仓库文档，应用镜像按设计不含它们。镜像内另确认 `h2==4.4.1`、三个 stall 方法与三个证据门控方法存在、`seconds_since_headers` 与 `threshold_seconds` 可通过诊断过滤器。

业务探针：Messages(databricks-claude-opus-5)、Responses SSE(gpt-5.4)、Chat SSE(gpt-5.6-luna) 三项 `completed=true`、`marker_found=true`，`persistence_verified=true`。两个 Service 各恢复一个 Ready 新 Pod 后端，selector 读回 `app=claude-lb`，无维护标识残留。新 Pod 零重启，`/health/{live,ready,accepting}` 均 200。

## 发布后移除 preference 环境变量

发布器不改动 env，因此 2026-10-09 事件中追加的 `COPILOT_RECOVERY_PREFERENCE_SECONDS=0` 在发布后仍然生效，而证据门控在 preference 整体关闭时是 no-op。移除该条是**独立的第二次 rollout**，不可与发布合并到同一维护窗口。

顺序为先发布、后移除：若先移除，旧镜像会带着 10 秒窗口却没有证据门控，短暂重现 2026-10-09 的大请求推迟行为。采用带 `test` 断言的 JSON patch 只删除该条目，server dry-run 核对剩余 env 与 image 未变后应用。新 Pod 读回 `copilot_recovery_preference_seconds=10.0`、`copilot_scoped_circuits=True`、`copilot_stream_prefirst_content_warn_seconds=90`。

## 未验证边界

上游的 120 秒取消与 `REFUSED_STREAM` 限流均未解决，本次发布不针对它们。证据门控是否在真实负载下改善恢复、停滞信号的 90 秒默认值是否合适、`umami` 的崩溃是否已随 MySQL 恢复而消除，均未验收。熔断状态仍为进程内状态，Pod 重启使计数归零不表示上游恢复。本次未执行数据库快照与隔离恢复验证，也未安排独立 QA 复核；账本仅经只读接口确认可读与 flush 恢复。
