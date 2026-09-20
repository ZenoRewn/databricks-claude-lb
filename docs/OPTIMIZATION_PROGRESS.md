# 本地可靠性优化进度

Author: Zeno Ren

范围：只优化 `databricks-claude-lb` 本地代码、测试、构建与文档。暂不部署 AKS，不改 OpenClaw Gateway 任务、脚本、历史数据或调度。用户提供的 OpenClaw 输出是运行背景，不是执行清理或修改自动化的授权。

## 已完成的本地批次

| 批次 | 状态 | 提交/证据 |
|---|---|---|
| 线上 effort 基线、Docker/CI 打包清单 | 完成 | `3a807af`；主程序与 effort 模块字节 hash 对齐线上 |
| request/admission/send 独立观测 | 完成 | `211d706`；完整 ASGI 生命周期、JSON/SSE 结果分类 |
| 上游错误体时间/字节双限额 | 完成 | `c40a96f`；真实 TCP、无限正文、gzip、重复取消测试 |
| 确定性窗口和本地归档 | 完成，未接入 OpenClaw | `operations.reporting`；12 项测试与 2 个子测试 |

本地 Python 3.14 全量：501 passed、433 subtests passed。Linux/amd64/Python 3.12 镜像：500 passed、1 skipped（默认镜像没有可选 OTel 依赖）、433 subtests passed。镜像应用代码全部来自 baked files，没有挂载主程序覆盖；监控工具和文档仅作为测试材料挂载。

正常 Docker 构建的 PyPI 下载遇到 TLS EOF；离线验证构建使用宿主机经 TLS 从 PyPI 下载的 wheel，仅替换依赖安装来源，保留相同应用 COPY 清单。第一批验证镜像 `claude-lb:review-20260920-amd64` 尚未推送或部署。第一批构建时新运行模块的镜像 hash 与当时本地一致；后续批次需要重新构建验收，不能复用旧镜像结论。

后续本地进度（尚未最终发布验收）：

- `6ccfa10`：请求总预算 1800s、上游响应头预算 180s，真实 TCP 停滞/取消回收测试；全量 511 passed、438 subtests。
- `5cb5329`：模型兼容白名单、请求内 tried-set、重试决策观测；全量 520 passed、446 subtests 后，补回空池 503 契约并单独验证相关 10 项测试。
- 最新准入批次：进程并发/队列/tenant 限制、128 MiB 输入预留、120s 上传预算、JSON object 校验；全量 533 passed、457 subtests。
- `eeb3aee` 为上述准入批次提交。
- usage 失败恢复、事务批次幂等、跨午夜事件归属、JSON 错误传播/原子写、source 元数据和新持久化指标已实现。全量 551 passed、6 skipped（隔离 MySQL 场景）、462 subtests；另用 Linux/amd64 baked image 在真实 MySQL 8.0.46 跑通全部 6 个事务/回执/并发/retention 场景。CI 已加独立 mysql-usage job；正常 CI unit job 保持隔离场景跳过。
- 后续需本地 readiness/drain、参数兼容可见性、最终镜像验收与更新交接说明。所有生产/OpenClaw修改仍未执行。

工作分支为 `codex/service-reliability-20260920`。离线 PyPI wheel 在 `/tmp/lb-wheelhouse-20260920`（linux/amd64/Python 3.12），pytest wheel 在 `/tmp/lb-test-wheels-20260920`。旧镜像构建目录路径保存于 `/tmp/lb-offline-build-path-20260920.txt`；不得将旧源码副本用于后续最终构建。Docker 已启动；所有生产访问均只读。

最新 storage 验证镜像为 `claude-lb:usage-review-20260920-v2`，源码副本路径记录在 `/tmp/lb-usage-v2-build-path-20260920.txt`。隔离数据库容器 `lb-usage-test-20260920` 使用 network=none，无挂载用户数据；通过 `--network container:lb-usage-test-20260920` 运行测试客户端。收尾时只清理这个本任务创建的测试容器及其匿名卷，不动其他容器。

## 接下来按顺序推进

1. 请求总预算、上游启动预算与超时终态，保持现有 cleanup ownership。
2. 兼容候选、tried-set 和有界准入；精确容量 503 保持独立开关，不扩大通用 POST 重放。
3. usage 失败恢复与幂等持久化；readiness/drain 与双副本的本地支持。
4. 逐批回归与镜像验收，整理给 OpenClaw 的指标/行为更新摘要。

## OpenClaw 后续同步边界

运行环境：`zeno-oc` 的 `/openclaw`；配置备份 remote 为 `ZenoRewn/openclaw_configuration_backup`，不是完整监控系统备份。

| 任务 | ID |
|---|---|
| LB 异常 watcher | `29abb4b6-a517-4ab5-99be-bed87049a6cf` |
| LB 小时数据归档 | `d33f2cb3-09bc-407e-a5f1-318b8cfc9abb` |
| LB 每日汇总 | `b268ddf5-5c72-4dd9-92c5-51b9ec1ab00f` |

此表仅用于后续交接，不代表已读取或修改任务。当前权威 watcher 是 Gateway 的 `trigger.script`/`payload.message`；不存在的旧 `tmp/lb-watcher-trigger-v2.js` 不能当作来源。保留 `tmp/lb-monitor-probe.sh`、未跟踪的采集脚本、`data/lb-monitor/` 及既有任务。后续更新说明需要明确新指标单位、默认值、终态和仍未完成的验证。
