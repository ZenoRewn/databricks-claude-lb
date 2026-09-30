# LB 合约优化验证回执

Author: Zeno Ren

日期：2026-09-30。基线：`16de51d29d1fbb3e81c65434ce55fadc3573da61`。当前应用候选源码：`c137f7bc37719f7b84f71aa23490c8c397ded387`。本文件持续追加当前验收结果，旧失败与中间候选证据不改写。

本轮实现见 [当前契约](../../OBSERVABILITY_AND_CONTEXT.md) 与 [实验/耐久性边界](../../OPERATIONS_EXPERIMENTS.md)。完成代码不等于部署完成。

| 层次 | 已获得的证据 | 边界 |
|---|---|---|
| 主机完整回归 | `host-full-r3.txt`：Python 3.14.5，711 passed、6 skipped、660 subtests passed | 对应 bceddc0；后续 capability shape 修复另有 36 项目标回归，最终镜像运行完整应用套件 |
| 最新 capability shape | `capability-shapes-green.txt`：36 passed、22 subtests passed | 验证普通文本和 tool schema 示例不会被 enforce 误判；合成输入 |
| Linux/amd64 镜像 | `image-tests-r3.txt`：Python 3.12.14，707 passed、7 skipped、662 subtests passed；21 个运行文件/依赖锁匹配，安装测试依赖未改变运行依赖版本 | 本地 image ID，不是已推送 registry manifest digest；4 个 Git 构建工具测试在主机另测 |
| MySQL 8.4.11 | `mysql84-tests-r2.txt`：6 项隔离事务测试通过 | arm64 MySQL，amd64 应用 client；无网络出口/host port；最后候选的 usage_store.py 与测试版本一致 |
| SDK 协议 | OpenAI Python SDK 读取合成文本、工具索引/ID、usage、finish_reason；现有 OpenAI/Anthropic 不安全重试回归保留 | 不是真实账号、真实客户端应用或上游模型 E2E |
| 微基准 | `microbenchmark-r3.json`：160/160 完成、160 次发送、结束 active=0、无诊断丢弃 | 候选的 context observe/off 比较，4 并发、MockTransport/ASGI；不是旧版本或生产容量比较 |
| 独立 QA | 尚未执行 | 主代理测试与自查不能冒充独立评审 |
| GitHub | 尚未上传本轮分支 | 发布前保留独立评审门禁，CI 状态另行记录 |
| AKS / OpenClaw | 未操作 | 未部署、未发付费推理、未改 scheduler、baseline、历史或生产数据库 |

## 可复现入口

```bash
python3 -m pytest tests/ -q --tb=short
node tests/test_copilot_pricing_ui.cjs
python3 -m operations.build_identity --build-tag claude-lb:local-review --platform linux/amd64
```

镜像回归只挂载 tests、operations 工具、文档、配置示例和测试依赖；未覆盖镜像内任何应用运行模块。安装测试依赖后再次检查运行依赖版本无变化、runtime manifest 完整且匹配。`tests/test_build_identity.py` 是需要 Git 的宿主构建工具测试，在主机执行，不伪装为目标应用运行能力。

主机 6 个 skip 是显式 opt-in 的隔离 MySQL 测试，已在独立容器另行执行。stock 镜像不包含可选 OTel 依赖；相关可选测试的跳过不表示 tracing 已启用。详细 skip 回执保留在镜像日志中。

## 红绿证据

- A：`diagnostics-red.txt` → `diagnostics-green.txt`，7 个初始失败；后续主键、原因和模型来源反例各有单独 red/green 文件。
- B：`phases-red.txt` → `phases-green.txt`，验证 header/body 计时、未知阶段、嵌套 cleanup、精确路由预算和版本化指标。
- C：`adapter-red.txt` → `adapter-green.txt`，以及 `adapter-hardening-red.txt` → `adapter-hardening-green.txt`。schema、工具配对、角色、图片同意、incomplete/refusal、模型来源均有负例。
- D：`context-red.txt` → `context-green.txt`，以及 `capability-shapes-red.txt` → `capability-shapes-green.txt`。来源、有效期、未知状态、鉴权和合法 fixture 保护均有覆盖。
- 构建/运维：`identity-red.txt` → `identity-green.txt`；`experiment-red.txt` → `experiment-green.txt`；`operational-red.txt` → `operational-green.txt`。
- 首轮整体 `host-full-r1.txt` 暴露新加入的事件时间戳反例，未标作通过；后续修复及完整回归保留。

公开文本日志会将本机仓库绝对路径替换成 `<repository>`；断言、计数与失败信息不删除。所有 canary、会话、凭据字样和请求数据均为合成 fixture。

## 真实剩余边界

微基准的 4 KiB 请求 p50 从 5.15ms（off）到 5.48ms（observe），2 MiB 从 49.65ms 到 54.73ms；p95 分别为 12.25→13.20ms、89.17→93.29ms（nearest-rank）。新增观测有可测开销，不能将单轮 RSS 差值解释成内存优化，也不承诺生产开销小于某个比例。`LB_CONTEXT_BUDGET_MODE=off` 可关闭附加预算扫描，已有安全诊断与计时仍保留。

离线 CLI 已用合成完整 cohort 执行，回执 `experiment-synthetic.json` 为 eligible_for_review、production_acceptance=false；这只证明分析工具按输入计划工作，没有生产优化收益结论。

子进程硬退出实验确认：flush 前内存事件可丢，确认 JSON flush 后可重读。MySQL batch 幂等解决重投，不解决尚未耐久接受的事件。当前仍无 durable outbox，不承诺节点丢失 RPO=0；没有通过降低断言来隐藏这一事实。

没有新的晨间生产 cohort、权威 Copilot 窗口或费用上限，单路由实验只完成配置能力和离线分析器。`eligible_for_review` 不触发发布，任何输出都保持 `production_acceptance=false`。多副本/全局额度/存储介质与保留期仍需单独定义和验收。

未运行新的完整 Kind 发布故障矩阵：发布器运行状态机未修改，变更限于应用源文件清单和独立本地构建 helper。应用测试不能替代后续实际发布的入口/互斥/接管/恢复验收。

隔离 MySQL fixture 已按本轮标签及 container ID 核验后删除，tmpfs 数据随之移除，见 [清理回执](cleanup.json)。没有清理用户原有镜像、工作树、未提交事故材料或 OpenClaw 资产。
