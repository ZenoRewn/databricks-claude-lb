# 独立 QA 与修复回执

Author: Zeno Ren

日期：2026-09-30。用户授权的只读 QA 子代理 `/root/independent_qa` 独立审查基线 `16de51d` 至候选 `dc2a2e4`，随后复核 `cd2ab85` 和最终应用提交 `1d75685655f672177374cf3327cd16c4bcffe2db`。评审未编辑代码或访问生产、真实供应商、付费接口、私有配置及 GitHub 写入。

结论：原 7 项 P2 findings 全部关闭；本次代码及合成验证范围内无剩余合并阻塞。主机完整回归、最终镜像和 GitHub Checks 分别记录，不由这份评审代替，更不代表生产验收。

| 初审发现 | 修复与复核结论 |
|---|---|
| 默认 text 日志丢关联字段 | 文本出口保留安全结构化字段，JSON 字段继续存在；既有 Copilot summary 文本标记保留 |
| 日志 sink 卡住导致解释器退出阻塞 | 默认 sink 不另注册 logging.Handler，并通过直接 fd 写入避开 Python stdio 缓冲锁；保留自定义 stream 支持及写异常计数 |
| Chat→Responses 的 send start/end 协议不一致 | 同一 attempt 的发送及错误事件使用实际 responses；请求终态仍为入口 chat |
| 流式换端点绕过能力 enforce | 三个 provider 在新 lease/POST 前复查；不兼容时发合法、不可重试 SSE 终态并清理旧响应 |
| 未知/混合来源被列为实验候选 | unknown 或 synthetic/production 范围不一致进入 inconclusive；安全事件仍优先 stop |
| 小数/布尔 token budget 在入口被删除 | 不再强制转整数后删除，适配器在真实入口明确拒绝非法类型 |
| 工具输出中的图片未检查能力 | 只遍历实际 function_call_output.output 协议位置，继续排除 schema/example 示例 |

## 独立复核证据

对 `cd2ab85` 的独立定向验证为 8 passed、35 subtests passed，其中包括 12 个流式重试分支：Databricks Messages、Azure Responses/Chat、Copilot Responses/Chat，覆盖适用的 429、connect、pool 条件。核验仅一次上游发送、B 端点无新 lease/错误计账、两端 active=0、响应关闭且终态不可重试。

该轮复核进一步发现，移除 logging.Handler 注册仍不足以解决真实 stderr 管道背压；旧 stdio 写入仍会让解释器退出等待缓冲锁。主代理保留失败输出后改为直接 fd 写入，未通过读取 stderr 来掩盖背压。

对最终 `1d75685` 的独立日志出口测试为 8 passed、4 subtests passed。另行运行真实满管道子进程：父进程等待退出期间完全不读取 stderr，关闭标记后约 0.234 秒正常退出（returncode=0），退出后读取到 65536 bytes，没有 Fatal Python error。短写、零进展写入、异常计数、自定义 StringIO 输出及正常/失败 worker 收尾均检查通过。满管道下允许诊断丢失，不宣称日志持久性。

## 主代理回归与失败保留

- `qa-diagnostics-red.txt` / `qa-diagnostics-green.txt`：最初三项诊断问题。
- `qa-contracts-red.txt` / `qa-contracts-green.txt`：其余语义/能力/实验问题；最终目标回归 120 passed、73 subtests passed。
- `qa-contracts-attempt1.txt` 保留中间失败。Databricks fixture 初始使用的 synthetic-model 被原有路由映射成默认模型，后改为明确的 databricks-synthetic-model；没有降低断言。
- `qa-capability-retries-baseline-red.txt`：从受审提交 dc2a2e4 的干净 archive 加载当前测试，12 个 retry 子用例全部复现失败；未覆盖基线应用模块。
- `qa-text-compat-red.txt`：保留旧 Copilot summary 文本标记的反例。
- `qa-stderr-pipe-red.txt` / `qa-stderr-fd-red.txt` / `qa-stderr-green.txt`：真实满管道退出、短写及错误计数，最终诊断/请求/阶段回归 34 passed、12 subtests passed。

最终完整套件、文件 hash、镜像及数据库证据见 [VALIDATION.md](VALIDATION.md)。评审未要求修改 OpenClaw 或扩展生产 POST 重放，也不把 durable outbox、多副本、真实客户端与生产窗口的已披露边界写成已完成。
