# LB 合约优化验证回执

Author: Zeno Ren

日期：2026-09-30。基线：`16de51d29d1fbb3e81c65434ce55fadc3573da61`。最终应用源码：`1d75685655f672177374cf3327cd16c4bcffe2db`。本文件更新当前验收汇总，旧失败与中间候选证据原样保留。

本轮实现见 [当前契约](../../OBSERVABILITY_AND_CONTEXT.md) 与 [实验/耐久性边界](../../OPERATIONS_EXPERIMENTS.md)。完成代码不等于部署完成。

| 层次 | 已获得的证据 | 边界 |
|---|---|---|
| 主机完整回归 | `host-full-r5.txt`：Python 3.14.5，722 passed、6 skipped、697 subtests passed | 对应最终应用提交 1d75685；旧 R3/R4 日志保留 |
| QA 修复目标回归 | `qa-contracts-green.txt`：120 passed、73 subtests；`qa-stderr-green.txt`：34 passed、12 subtests | 覆盖三 provider 的 12 个流式重试分支、真实入口、能力特性和日志背压；合成输入 |
| Linux/amd64 镜像 | `image-tests-r5.txt`：Python 3.12.14，717 passed、7 skipped、697 subtests passed；21 个运行文件/依赖锁匹配，安装测试依赖未改变运行依赖版本 | 本地 image ID，不是已推送 registry manifest digest；4 个 Git 构建工具测试在主机另测 |
| MySQL 8.4.11 | `mysql84-tests-r2.txt`：6 项隔离事务测试通过 | arm64 MySQL，amd64 应用 client；无网络出口/host port；最后候选的 usage_store.py 与测试版本一致 |
| SDK 协议 | OpenAI Python SDK 读取合成文本、工具索引/ID、usage、finish_reason；现有 OpenAI/Anthropic 不安全重试回归保留 | 不是真实账号、真实客户端应用或上游模型 E2E |
| 微基准 | `microbenchmark-r5.json`：160/160 完成、160 次发送、结束 active=0、无诊断丢弃 | 最终候选的 context observe/off 比较，4 并发、MockTransport/ASGI、内存诊断 sink；不是生产容量或默认 stderr I/O 成本测试 |
| 独立 QA | 7 项 findings 修复后，在最终 1d75685 上获只读独立复核通过，见 [QA 回执](QA.md) | 代码及本地合成验证范围；生产与真实供应商未验证 |
| GitHub | [PR #8](https://github.com/ZenoRewn/databricks-claude-lb/pull/8) 已合并为 d88d742，PR/main 各六项 CI 通过；发布身份修复 [PR #9](https://github.com/ZenoRewn/databricks-claude-lb/pull/9) 已合并为 80da865，PR/main 各六项 CI 通过 | 应用运行文件仍为已核验的 1d75685；发布工具与应用镜像分别绑定身份 |
| AKS | 2026-09-30 发布 succeeded，维护约 75 秒，运行文件、三协议、账本、路由和清理通过，见 [部署回执](../2026-09-30-aks-release/REPORT.md) | 公网实际覆盖 Databricks/Copilot；Azure 另有候选镜像独立进程验证，不等于全部端点或真实客户端验收 |
| OpenClaw | 外部自动化、脚本、调度、baseline 和历史未修改 | 本轮部署 LB，不代表外部 watcher 已采用新增采集/报告逻辑 |

## 可复现入口

```bash
python3 -m pytest tests/ -q --tb=short
node tests/test_copilot_pricing_ui.cjs
python3 -m operations.build_identity --build-tag claude-lb:local-review --platform linux/amd64
```

镜像回归只挂载 tests、operations 工具、文档、配置示例和测试依赖；未覆盖镜像内任何应用运行模块。安装测试依赖后再次检查运行依赖版本无变化、runtime manifest 完整且匹配。`tests/test_build_identity.py` 是需要 Git 的宿主构建工具测试，在主机执行，不伪装为目标应用运行能力。

最终镜像 `claude-lb:contracts-20260930-r5` 的本地 image ID 为 `sha256:fb615b508cc31ed88e4cf5808164fcbb7e18535ebf1d0b8f8ed5fc22e64465b7`，source manifest 为 `87b7a17d60a24d1aa78648d343bbbbb6462cb8f41aed4138be144d119fdb7ffd`，dirty=false、out_of_tree=false。完整身份见 [镜像回执](application-image-r5.json)。包下载端出现 TLS EOF 后使用既有离线 wheel 与相同 lockfile 构建；没有关闭 TLS 校验。最终镜像测试容器 network=none，安装测试依赖后以 uid/gid 1000 执行套件，测试后自动删除。

主机 6 个 skip 是显式 opt-in 的隔离 MySQL 测试，已在独立容器另行执行。stock 镜像不包含可选 OTel 依赖；相关可选测试的跳过不表示 tracing 已启用。详细 skip 回执保留在镜像日志中。

## 红绿证据

- A：`diagnostics-red.txt` → `diagnostics-green.txt`，7 个初始失败；后续主键、原因和模型来源反例各有单独 red/green 文件。
- B：`phases-red.txt` → `phases-green.txt`，验证 header/body 计时、未知阶段、嵌套 cleanup、精确路由预算和版本化指标。
- C：`adapter-red.txt` → `adapter-green.txt`，以及 `adapter-hardening-red.txt` → `adapter-hardening-green.txt`。schema、工具配对、角色、图片同意、incomplete/refusal、模型来源均有负例。
- D：`context-red.txt` → `context-green.txt`，以及 `capability-shapes-red.txt` → `capability-shapes-green.txt`。来源、有效期、未知状态、鉴权和合法 fixture 保护均有覆盖。
- 构建/运维：`identity-red.txt` → `identity-green.txt`；`experiment-red.txt` → `experiment-green.txt`；`operational-red.txt` → `operational-green.txt`。
- 首轮整体 `host-full-r1.txt` 暴露新加入的事件时间戳反例，未标作通过；后续修复及完整回归保留。
- 独立 QA 的诊断、真实 stderr 背压、换端点能力检查、输出类型、图片能力及来源门禁反例和复核见 [QA.md](QA.md)。包括中间失败与修正 fixture 的基线复现，没有删除失败证据或放宽断言。

公开文本日志会将本机仓库绝对路径替换成 `<repository>`；断言、计数与失败信息不删除。所有 canary、会话、凭据字样和请求数据均为合成 fixture。

## 真实剩余边界

最终候选微基准的 4 KiB 请求 p50 从 2.97ms（off）到 3.92ms（observe），2 MiB 从 34.43ms 到 32.78ms；p95 分别为 7.24→6.82ms、57.67→62.96ms（nearest-rank）。使用内存诊断 sink，默认 stderr 背压由独立故障测试覆盖，不能从这组时延推断真实日志出口成本。单轮数值不证明优化收益、内存改善或生产开销上限；旧 R2/R3 记录保留。`LB_CONTEXT_BUDGET_MODE=off` 可关闭附加预算扫描，已有安全诊断与计时仍保留。

离线 CLI 已用合成完整 cohort 执行，最终重跑回执 `experiment-synthetic-r5.json` 为 eligible_for_review、production_acceptance=false；未知或混合 scope 的拒绝另有回归。这只证明分析工具按输入计划工作，没有生产优化收益结论。

子进程硬退出实验确认：flush 前内存事件可丢，确认 JSON flush 后可重读。MySQL batch 幂等解决重投，不解决尚未耐久接受的事件。当前仍无 durable outbox，不承诺节点丢失 RPO=0；没有通过降低断言来隐藏这一事实。

没有新的晨间生产 cohort、权威 Copilot 窗口或费用上限，单路由实验只完成配置能力和离线分析器。`eligible_for_review` 不触发发布，任何输出都保持 `production_acceptance=false`。多副本/全局额度/存储介质与保留期仍需单独定义和验收。

未运行新的完整 Kind 发布故障矩阵：发布器运行状态机未修改，变更限于应用源文件清单和独立本地构建 helper。应用测试不能替代后续实际发布的入口/互斥/接管/恢复验收。

隔离 MySQL fixture 已按本轮标签及 container ID 核验后删除，tmpfs 数据随之移除，见 [清理回执](cleanup.json)。没有清理用户原有镜像、工作树、未提交事故材料或 OpenClaw 资产。
