# 流诊断与分层保护 AKS 发布回执

Author: Zeno Ren

2026-10-09 已将 [PR #19](https://github.com/ZenoRewn/databricks-claude-lb/pull/19) 合并到 GitHub `main`，并将应用源码 `fea7d604d230dd71347cdf9d6145cb8429f5d729` 发布到 AKS。集群发布回执为 **succeeded**；北京时间 **19:36:38–19:39:13** 维护并恢复流量，时长 **155.711 秒**。三个常规业务探针、账本与清理通过，未回滚，清理错误为空。

**本次上线的是取证能力与局部保护，不是对断流故障的修复。** 发布后约 30 分钟观察期内没有出现任何上游协议错误，这不构成故障已消除的证据 —— 原故障为偶发，且观察期未复现。结构化结果见 [receipt.json](receipt.json)。

## 生效内容与范围

- 协议取证从真实 h2 原因对象提取流重置/GOAWAY 错误码、流 ID、实际 HTTP 版本与上游空闲间隔，并新增 `upstream_headers_received` 与 `upstream_idle_seconds`。异常原文、GOAWAY debug data、prompt 与工具内容不写日志。
- Copilot endpoint/model/API 局部熔断置于共享 endpoint 熔断之下。仅明确的远端 RST_STREAM（1/2/8）与已分类的非 HTML 缓冲协议不匹配进入局部层；GOAWAY、REFUSED_STREAM、ENHANCE_YOUR_CALM、未知协议失败、缺终态、鉴权与过载仍进入共享保护。
- 冷却后 10 秒优先接纳语义输入不超过 64 KiB 且估算完整的既有入站请求；被推迟的请求收到 503 `recovery_prefers_small_request` 与 Retry-After，不占试探、不触发推理/鉴权、不授权跨 provider 回退。
- 推理与 count_tokens 响应提供规模、估算可信度、未知组件与整理建议；低可信度估算不构成拒绝、静默裁剪或自动摘要。
- 实际 Deployment spec 仅改变应用 image 与 source-revision 注解。单副本、500m/2Gi、完整 Service selector、Ingress、配置/凭据、三个探针、preStop 45 秒、90 秒 grace 与双协调器均保留；不改数据库 schema，不放宽推理 POST 重放白名单。

配置按代码默认值上线，未注入环境变量覆盖：`COPILOT_SCOPED_CIRCUITS=true`、`COPILOT_RECOVERY_SMALL_INPUT_BYTES=65536`、`COPILOT_RECOVERY_PREFERENCE_SECONDS=10.0`、`LB_CONTEXT_LARGE_INPUT_BYTES=262144`。四项均在运行进程中读回确认。源码与本地验证见 [发布前报告](../2026-10-09-stream-reliability/REPORT.md)。

## 固定身份与维护前验收

应用单平台 registry manifest 为 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:7199477bdae5cf79dffb6ea9b41723a99420feaff96391354585b1fe4074560d`，平台 linux/amd64，mediaType 为单平台 manifest v2 而非多平台 index；config digest 与本地构建 image ID 相符，运行 Pod imageID 一致。21 个运行文件与依赖清单 hash 与合并源码一致，manifest SHA-256 为 `a6e675f86836667d18588d24a770e5066e3e4ba2580e78c64847004ffd8426d5`。公网 `/version` 返回 `fea7d60` 与 `verification=matched`，新 Pod 零重启。

PR 与合并后 `main` 的六项 CI 均通过：Python 3.11/3.12/3.13、Docker、隔离 MySQL 8.0/8.4；[PR CI](https://github.com/ZenoRewn/databricks-claude-lb/actions/runs/37923144166)、[main CI](https://github.com/ZenoRewn/databricks-claude-lb/actions/runs/37923530559)。本机全量回归 816 passed、745 subtests passed、6 skipped。正式镜像内另跑 733 passed、685 subtests passed、7 skipped，应用源码未经挂载替换；排除的 8 个测试文件依赖 `operations` 模块或仓库文档，应用镜像按设计不含它们，这些文件已在本机全量与 CI 中通过。镜像内另确认 `h2==4.4.1` 存在，协议取证不会因缺少可选依赖降级为 `unknown`。

维护前完成现场发现、完整 spec 最小差异比较、server dry-run 与恢复材料。计划 hash 为 `708f07bd61d8f100715ec69f5ef98a4a8769cbf35a45d31b1829bb78e0ab224a`，回滚锚点为前一版本 `284ca53` 的 `@sha256:2b0e273e26f181ad7258c9ead871a86242d91151122f85c4630eb38a0161b9c6`。既有跨节点协调器以互斥 Lease 和持久回执持有维护责任；总预算 600 秒、前向 360 秒、排空 180 秒。没有强杀长请求。

**本次未执行数据库快照与隔离恢复验证，也未安排独立 QA 复核。** 上一轮的那两项证据不转移到本次发布。账本经只读接口确认可读、返回 2026-07-12 至 2026-10-09 共 90 天历史，`/health/ready` 为 `ready` 且 issues 为空；这证明账本后端在新 Pod 下工作，不等于已验证全部历史行与 payload hash 逐条保留。

## 公网业务与收尾

| 验收 | 模型 / provider | 结果 |
|---|---|---|
| Messages JSON | databricks-claude-opus-5 / Databricks | 非空 LB_OK、成功终态、账本事件已落盘 |
| Responses SSE | gpt-5.4 / Copilot | 重组后的非空 LB_OK、成功终态、账本事件已落盘 |
| Chat SSE | gpt-5.6-luna / Copilot | 非空 LB_OK、成功终态、账本事件已落盘 |

两个 Service 各恢复一个 Ready、非 terminating 的新 Pod 后端，selector 均读回为 `app=claude-lb`，无维护标识残留。Deployment/Service 维护标识、Pod finalizer 与应用 pause/drain 均已清理，`cleanup_errors` 为空。公网三个健康入口连续 30 轮采样全部 200、零失败；无凭据调用 `/config/effective` 仍为 401，`/stats` 与 Dashboard 保持 OAuth 302。该短时结果不是长期 SLA。

## 观察期结果与边界

发布后约 30 分钟的结构化日志共 56 条 `lb_request_end`：**50 completed、6 client_disconnected**，provider 分布为 Copilot 43、Databricks 13。`copilot_stream_network_error`、`lb_circuit_transition` 与 `lb_recovery_deferred` 均为 **0 条**，即观察期内没有上游协议错误、没有熔断转换、恢复优先窗口未被触发。新的上下文提示在真实流量上工作，`context_advice` 分布为 none 41、large_input 25，`estimate_confidence` 全部为 low。

6 条 `client_disconnected` 的 `error_origin` 均为 `client`，是下游客户端在收到内容前自行断开，与本轮要定位的上游 `RemoteProtocolError` 是不同 outcome。其中两条为带图的 large 请求，`send_to_headers` 分别为 21.4 秒和 26.2 秒。这是观察记录，不足以归因到上游、客户端超时或图片处理中的任何一项。

触发本轮工作的原始故障具有 `chunks_yielded=0`、`first_event=None`、`probe.ok=true` 的形态，其日志不含任何 typed protocol code，无法回溯归类。观察期零协议错误**不证明**该故障已消除；下一次复现时应读取 `kind=copilot_stream_network_error` 的 `protocol_error_kind`、`http2_error_code`、`protocol_scope`、`upstream_headers_received` 与 `upstream_idle_seconds` 再定方向。未验证项包括：原故障复现与归因、GOAWAY 与中间代理的因果、跨账户/模型的真实服务等价性、Mac 客户端对新响应头的呈现、长期负载与故障演练、多副本全局熔断（局部状态仍为进程内）。

完整计划、配置/Secret、Pod/Lease 身份与请求诊断保留于本机 0700/0600 私有发布目录，不进入公开 GitHub。本回执仅更新文档，不再次滚动应用。
