# GHCP Responses 消息 ID 稳定化：本地实现与验证

Author: Zeno Ren

日期：2026-10-20（北京时间）

## 结果与范围

已基于“排查 GHCP 重复输出”中的方案完成本地优化。对于显式启用的 GHCP 原生 Responses 流式请求，同一 message 在 added、delta、done 与终态快照中使用首次 added 的 ID，避免按 ID 更新消息的客户端把完成快照另加为第二条消息。相同文字的不同 message 仍保留，未按文字去重。

基线为 `15eff26546ae0407402cc451a411ff1c3cd44f52`，以下记录保留本地验收时的工作区状态。新增模块、应用镜像 COPY、运行来源清单、发布器 APP_FILES 与 CI 文件检查已同步；验收时尚未提交的运行文件哈希见 [working-tree-identity.json](working-tree-identity.json)。该文件不是镜像或线上来源证明；GitHub 交付以本修复分支的提交和关联 PR 为准。原有未跟踪的 AGENTS.md 与历史事故目录未纳入修改。

## 实现

- 每次上游 attempt 独立维护有界映射；请求开始时固定模式和精确模型命中结果。仅 message 的已知 ID 字段可改写，reasoning/tool ID、call_id、参数、encrypted_content、phase、正文、用量和 response.id 保留。
- 转换发生在成功终态判断/结算前。协议冲突和本地资源/转换错误返回明确、不可重试的错误，不重放推理、不扩大账户熔断。若冲突帧含合法用量，原有幂等记账路径仍记录一次。
- 原始帧未改写时保持 bytes 不变；改写仅重建 SSE data。保留 event/id/retry/comment 控制字段，复用现有解压、分帧、背压和 shielded cleanup。
- 每 attempt 限制 4096 项、16384 个 ID、单 ID 8 KiB、映射字符串 2 MiB。改写帧继续受每事件和全进程 retained-buffer 预算约束。取消、异常与清理释放状态；诊断自身失败不跳过上游关闭或 lease 释放。
- 新增有限维度的漂移、改写、失败和容量指标及安全请求摘要，不记录原始 ID、正文或凭据。

完整配置和字段白名单见 [配置说明](../../CONFIGURATION.md#原生-responses-流式-message-id-兼容) 与 [协议契约](../../STREAM_PROTOCOL.md#opt-in-copilot-responses-message-identity)。

## 验证证据

| 检查 | 结果 |
|---|---|
| 修复前最小集成复现 | 下游实际为 `A,B,B,C`，与预期 `A,A,A,A` 不符；测试失败 |
| 规范化器单元测试 | 64 passed，含同文消息隔离、工具/reasoning 交错、拒绝回答、annotation、终态、SSE 行结束符/控制字段、非法 JSON/Unicode、别名冲突和容量限制 |
| 新增集成测试 | 13 passed、11 subtests，含单次上游发送、用量、失败/未完成、gzip、配置冻结、取消、故障注入、指标及 provider/非流式隔离 |
| 冻结代码后的完整回归 | `python3 -m pytest tests/ -q --tb=short -rs`：**1587 passed、6 skipped、1400 subtests passed，48.16 秒** |
| 独立 QA 与相关协议回归 | **205 passed、164 subtests passed**，包含额外 18 项独立故障/边界用例以及 ownership、JSON、input-ID stripping、opaque recovery 回归 |
| 打包与发布工具回归 | 52 passed、39 subtests；运行文件集合、镜像 COPY 与发布计划清单一致 |
| 文档 | 链接、代码块、Author 与浏览器渲染检查通过；`git diff --check` 通过 |

完整回归日志见 [full-regression.txt](full-regression.txt)。本机没有 `python` 命令，使用已有 `python3`，版本为 **3.14.5**。6 个跳过项均为 `tests/integration/test_mysql_usage.py`，需要独立的 loopback MySQL 测试服务器；没有通过削弱断言绕过测试。

独立 QA 发现并推动修复了未知事件中 `item` 字段遮蔽 `item_id`、以及诊断摘要异常时仍须独立关闭映射的边界。相关失败已补回归，最终没有剩余代码阻塞。首轮全量测试曾在并行修复过程中失败 1 项，以上最终全量结果来自代码冻结后的重跑。

GitHub 交付补充：[首次 CI](https://github.com/ZenoRewn/databricks-claude-lb/actions/runs/39387430225) 的 Docker 构建/运行 smoke 与隔离 MySQL 8.0/8.4 检查均通过；三个 Python 作业唯一失败项是新回执未加入评审索引。该检查只枚举已跟踪文件，本地生成回执时尚未覆盖此目录；提交后已复现并补齐索引，未修改测试断言。最新完整 CI 状态以 [PR #65](https://github.com/ZenoRewn/databricks-claude-lb/pull/65) 的检查为准。

有界容量 smoke（Python 3.14，tracemalloc，非 RSS）：4096 项加到 16384 个 ID 用时 0.662 秒，映射字符串 249856 B、跟踪分配峰值 2560741 B，关闭后映射归零；8 KiB 单 ID 和 2 MiB 映射字符串上限触发正确。128 个 64 KiB 文本增量（累计 8 MiB）仅保留 9 B 身份字符串，未保存完整回答。这些本地合成测量不能代表生产负载或整进程内存上限。

## 启用与真实边界

默认保持 `COPILOT_RESPONSES_ITEM_ID_MODE=off`，模型名单为空。后续在已授权、隔离的验收环境可先配置：

```text
COPILOT_RESPONSES_ITEM_ID_MODE=observe
COPILOT_RESPONSES_ITEM_ID_MODELS=gpt-6-astra
COPILOT_STRIP_INPUT_ITEM_IDS=true
```

确认样本契约后，`stabilize` 才会实际改写；`observe` 只观察。`stabilize` 与关闭 input ID stripping 的配置组合拒绝启动。配置在进程启动时加载；关闭需设置 `off` 并按受控流程重启，不是热开关。新增运行模块也要求未来发布时同步重建/更新发布协调器镜像，详情见 [发布指南](../../RELEASE_TOOL.md#安装材料)。

未验证项：

- **目标镜像构建/运行**：本机 Docker daemon 未运行，未启动或安装运行环境；不能将静态打包清单检查称为镜像验收。
- **真实 GHCP/Foundry 调用、Codex 桌面显示、三轮工具回路及生产部署**：本次未执行，因此不能声明线上重复已消失。新增工具保护测试证明字段不变，不证明真实工具执行次数。
- **快照顺序**：当多个同类型项在终态中全部换成未知 ID 时，依赖 Responses 完整 output 数组按 output_index 排序的协议契约。可检测已知别名重排、数量/类型冲突，不能独立识别所有未知 ID 的任意重排。
- **其他客户端**：依赖 item_reference/retrieve 的客户端、工具/reasoning 自身 ID 漂移未在本批修复范围；Foundry、Messages、Chat、非流式和 buffered adapter 保持原路径。

本地验收阶段未提交或推送；后续按用户要求同步代码与证据到 GitHub。GitHub CI 结果与上述本地测试分别记录，不能据此推断功能已在生产启用。本次未发起真实模型推理或部署；生产验收须另行核实现场并完成真实客户端与完整业务终态检查。
