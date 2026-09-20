# 本地可靠性优化进度

Author: Zeno Ren

日期：2026-09-20

**本地核心优化已完成并验证；AKS、OpenClaw 和生产数据库未修改。** 工作分支为 `codex/service-reliability-20260920`，运行代码候选提交为 `e4dde8d`。当前部署仍需另行确认，本文不是发布回执。

## 已完成

| 范围 | 主要提交 |
|---|---|
| 线上 effort 基线与打包 | `3a807af` |
| 请求/准入/实际发送观测 | `211d706`、`70ee7f3` |
| 错误体字节/时间上限 | `c40a96f` |
| 本地确定性窗口与归档组件 | `f396800`，未接入 OpenClaw |
| 请求总预算和启动预算 | `6ccfa10` |
| 兼容候选与请求内 tried-set | `5cb5329` |
| 有界准入、输入预留和上传预算 | `eeb3aee` |
| 失败用量恢复、事务幂等与事件归属 | `05cc69f` |
| 本地 readiness/drain 和 shutdown | `76eb1ed` |
| 参数移除可见性及严格选项 | `025796b` |
| 复核修正：重连所有权、取消清理、排空归属 | `089cae9`、`e4dde8d` |

## 最终验证

- 本地 Python 3.14：568 passed、6 skipped、481 subtests passed。六个跳过项为单独运行的 MySQL 场景。
- Linux/amd64/Python 3.12 baked image：567 passed、7 skipped、481 subtests passed。除上述六项外，默认镜像未安装可选 OTel SDK，因此跳过该启用测试；没有宣称 exporter 链路通过。
- 隔离真实 MySQL 8.0.46：6/6 通过，覆盖部分写入、ACK 丢失、并发写者、同批并发重投、提交后取消与 retention。
- 容器启动、就绪→draining、拒绝新请求、零上游调用与应用 shutdown complete 检查通过。docker stop 后 exit=143，为 SIGTERM 信号退出记录，不伪写为 exit 0。
- 应用源文件 hash 与验证镜像逐一一致。只挂载测试/运维工具/文档，没有挂载主程序覆盖镜像。

最终本地镜像为 `claude-lb:reliability-review-20260920-r2`，image ID 为 `sha256:eefe8f7e02bf3c983a31f7498965bbbbe1b948b3a5fe0b739712c6869c488347`。这是本地 image ID，尚未推送 ACR，不是已发布的 registry digest。

普通在线构建遇到容器访问 PyPI 的 TLS EOF。验证构建使用宿主机经 TLS 下载的离线 wheel，仅替换依赖安装来源；未关闭证书验证。原 Dockerfile、验证 Dockerfile、wheel 和源码 hash 及日志见 [验证记录](reviews/2026-09-20-service-optimization/implementation-r2/validation.json)。

## 实施边界

运行时没有放宽 503/执行不明 POST 重放，也没有自动更换模型/provider。新预算和并发值尚未按真实负载校准；新增 MySQL 账本需在未来部署前审阅权限与迁移。pending usage 仍是内存队列，不承诺 Pod 硬丢失零 RPO。

真实 OpenClaw/SDK/模型端到端、长期压力、供应商能力/容量域、节点维护和生产灰度仍属后续运行验证。管理权限细分、公网 metrics 访问控制、共享配额与 durable outbox 根据接入范围和环境另行安排，不把本地测试当生产认证。

独立发布复核按现有 RESILIENCE.md 门禁保留；本轮只完成本地实现与技术检查，没有 push、PR、部署或对外消息。

## OpenClaw 交接

参见 [OPENCLAW_UPDATE_HANDOFF.md](OPENCLAW_UPDATE_HANDOFF.md)。其 Gateway task 定义仍是 watcher 的权威来源；`/openclaw/tmp/lb-monitor-probe.sh`、未跟踪采集脚本、历史数据和三个既有自动化均未修改或清理。用户提供的“清理 subagent”文字是背景，不是本次执行指令。
