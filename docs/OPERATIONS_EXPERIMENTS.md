# 单路由实验、重放与用量耐久性

Author: Zeno Ren

日期：2026-09-30。本页提供本地工具与实施边界，不是一份 AKS 发布回执，也不表示已完成晨间生产实验。

## 实验先有可比较证据

`python -m operations.experiment` 只读取离线 JSON，不调用模型、编辑配置、重启服务或投递通知。它比较 provider、requested/forwarded model、API、stream、原始体积桶、可信 tenant 和开始时段相同的 cohort。

```bash
python -m operations.experiment \
  --baseline baseline.json --candidate candidate.json \
  --plan experiment-plan.json --output new-run/summary.json
```

输出拒绝覆盖既有文件，保留输入 SHA-256。公开示例和测试全部为 synthetic。私人请求记录应放私有目录，不提交原始生产数据；输出只包含 cohort 聚合，不包含请求 ID 或正文。

计划必须明确：timezone、hours、每组最低样本量、最小完成率改善、p95 比例上限、断开率增长上限、资源增长上限及发送放大量上限。没有生产默认阈值；测试里的数字只是合成 fixture。

数据需包含完整/部分采集状态、pending、cohort_started、诊断丢弃数、镜像 digest、时间窗、资源峰值、安全事件，以及唯一请求的 outcome、开始时间、时长和发送数。`cohort_started = 已结束请求行数 + pending`；重复 ID 或分母不符拒绝分析。输入的生产 scope 和覆盖声明是采集方证据，不由工具自行认证。

两组 `reported_scope` 必须一致且明确。任一组为 `unknown`，或 synthetic 与 reported_production 混合比较，均输出 `inconclusive` 并列出来源原因；有已报告安全事件时仍优先输出 `stop`。不能把合成数据与生产数据拼成改善证据。

| 结果 | 含义 |
|---|---|
| `inconclusive` | 覆盖、来源、资源、安全证据、样本量、闭合请求或匹配 cohort 不足；缺失不当 0 |
| `not_supported` | 有足够配对样本，但未同时满足完成率、尾时延、断开和发送放大量门槛 |
| `stop` | 重放/隐私/重复用量等已报告安全事件，或资源越界 |
| `eligible_for_review` | 样本符合预先给出的保守统计条件，仍需独立业务/客户端审阅 |

所有输出的 `production_acceptance` 都为 false。完成率保留 failed、rejected、overloaded、client_disconnected 等全部结束结果作为分母，不能靠拒绝或过滤失败请求制造改善。Wilson 95% 区间是描述性估计，不证明流量独立或随机化，更不证明因果关系。

实验前还需确定真实路由、费用上限、维护责任、停止条件和客户可接受的等待。只有阶段证据支持等待 headers 预算不足时，才改变单一路由的一个参数；没有默认把 180 秒改成某个更大值，也不启动收费保活探针。重复晨间窗口的样本尚未提供，本轮没有声称线上超时改善。

## 重放与熔断

原 [重放契约](RESILIENCE.md) 保持：上下文/参数错误不原样重试；已输出、EOF、read/write timeout、协议错和执行不明的 POST 不透明重放。既有 429、一次鉴权刷新、受限 opaque 修复和 Connect/Pool 条件保留；Retry-After、total deadline、账户 pinning 和 session affinity 继续约束行为。

本轮新增 `lb_circuit_transition`，提供打开、试探准入、恢复、失败/中性试探和管理重置的原因、generation 与安全端点别名。HALF_OPEN 的事件在真正试探准入时记录，纯健康检查和轮询不会消费 slot 或制造状态历史。当前 open=0 不等于从未熔断；诊断丢弃也必须计入历史覆盖判断。

## 用量承诺与实测边界

现有模型调用与记账结果独立。每个用量事件首先进入**易失内存**；固定 batch ID 和 MySQL 事务账本处理重投与 ACK 丢失，不能让尚未 flush 的内存事件在进程消失后自动恢复。

v3 增加：

| 指标 | 含义 |
|---|---|
| `lb_usage_accepted_events_total` | 本进程成功进入 buffer 的事件数；不代表耐久提交 |
| `lb_usage_persisted_events_total` | 本进程确认批次保存成功的事件数，幂等重试只确认一次 |
| `lb_usage_oldest_pending_age_seconds` | 已知最旧待写事件年龄；时间未知输出 NaN，空队列为 0 |
| `lb_usage_volatile_buffer` | 当前为 1，表示入队到确认保存之间存在易失边界 |

这些计数随进程生命周期重置，不能跨 Pod 无条件相减。约 30 秒的正常 flush 周期不是硬丢失上限，数据库故障、积压和进程/节点丢失会扩大窗口。

`tests/test_operational_evidence.py` 的独立子进程实测表明：`record()` 后、flush 前 `os._exit()` 会失去该事件；确认 JSON flush 后硬退出，重读可恢复合计。这个测试**证明剩余易失边界**，不是宣告零丢账已修复，也不是 AKS 节点故障演练。

真实隔离 MySQL 验收覆盖：部分写入回滚、提交 ACK 丢失、重复批次、取消后重投、多 writer 增量合计与回执保留。具体本轮执行结果见 [验证记录](reviews/2026-09-30-lb-contracts/VALIDATION.md)；本地事务测试不等于供应商账单对账。

## Durable outbox 与多副本的决策边界

如果业务要求“已确认记录的用量事件在进程丢失后仍能恢复”，应单独实现 durable outbox：先耐久保存不可变 event/batch ID，再向 MySQL 投递，ACK 丢失重投同批次，确认后再清理。存储必须覆盖所要求的故障域；容器临时文件或随 Pod 消失的本地盘不能承诺节点丢失 RPO=0。记录前进程已消失、上游没返回 usage 或真实供应商账单差异仍需另行对账，不能重新推理补账。

本轮没有擅自选择零 RPO、存储介质、保留期或双写迁移。当前没有持久 outbox 实现；这是待明确业务承诺后的独立架构项。数据库保持原有追加账本与写入语义，回滚不清空账本或覆盖新增用量。

增加副本前还需确认：全局并发/内存/速率预算、账户列表和亲和一致性、共享状态、token 刷新、实际容量、writer/账本幂等和长流排空。两个进程各 128 active 可能变成总计 256，不能把单进程参数当全局配额。单副本改双副本不是已证实的 upstream startup 慢的修复。

生产采用任何候选前，先按 [AKS 指南](AKS.md) 和 [受控发布指南](RELEASE_TOOL.md) 完成当前现场、最小补丁、备份/恢复、互斥与完整业务验收；不要对现有部署整份 apply 示例模板。
