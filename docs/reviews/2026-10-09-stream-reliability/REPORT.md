# Codex 断流后的四项本地优化

Author: Zeno Ren

日期：2026-10-09（北京时间）。本页记录仓库实现、合成故障注入、本地回归和 Linux/amd64 镜像验收，当时未部署 AKS，未调用真实付费模型。该候选随后以 [PR #19](https://github.com/ZenoRewn/databricks-claude-lb/pull/19) 合并为 `fea7d60` 并发布到 AKS，发布与真实流量观察见 [部署回执](../2026-10-09-stream-aks/REPORT.md)。

## 结果与证据边界

四项优化按协议诊断、局部熔断、恢复试探、上下文提示的顺序实现。原请求在上游 HTTP 200 后约 120 秒没有正文并发生 RemoteProtocolError；旧日志缺少 reset/GOAWAY 类型，无法唯一归因或证明这些候选改动已消除原故障。

1. **协议诊断**：提取真实 h2 原因对象中的流重置/GOAWAY 错误码、流 ID、实际 HTTP 版本和最近响应头/解码正文 delivery 的空闲间隔。异常文本、GOAWAY debug data、prompt 和工具内容不写日志。有限深度、循环原因链和缺 h2 的情况保持未知。
2. **分层熔断**：明确的远端 RST_STREAM（1/2/8）与已分类的非 HTML 缓冲协议不匹配进入 endpoint/model/API 局部熔断；连接、鉴权、过载、GOAWAY、未知断流和缺终态仍进入共享保护。各层独立 generation 与单试探槽，保留累计错误、取消中性、旧结果防覆盖及管理重置。局部成功不能替其他模型解锁。每个 balancer 最多 128 条局部状态，容量耗尽保留共享保护。
3. **恢复试探**：只使用正常入站请求。冷却后默认 10 秒优先接纳语义输入不超过 64 KiB 且估算完整的请求；较大或含未知组件的请求收到 503 和 Retry-After。等待不占试探、不触发鉴权/推理、不授权 provider 回退；窗口过后仍可正常试探。既有 pinning、会话亲和、请求预算及不重放策略保留。
4. **上下文提示**：推理响应头、诊断与受保护 count_tokens 接口提供规模、估算可信度、未知组件和整理建议。默认 256 KiB 仅触发规模提示；near/over 只使用未过期且已核验的渠道限额。低可信度输入估算始终不成为拒绝、静默裁剪或自动摘要依据。

## 配置与运行观察

| 配置 | 默认与作用 |
|---|---|
| `COPILOT_SCOPED_CIRCUITS` | true；false 恢复 endpoint-only 保护 |
| `COPILOT_RECOVERY_SMALL_INPUT_BYTES` | 65536；恢复优先窗口的语义输入规模，范围 1 B～4 MiB |
| `COPILOT_RECOVERY_PREFERENCE_SECONDS` | 10；范围 0～60 秒，0 关闭优先窗口 |
| `LB_CONTEXT_LARGE_INPUT_BYTES` | 262144；提示阈值，范围 1 B～64 MiB |

配置在 `/config/effective` 可读，启动时校验范围。`lb_circuit_transition` 增加 scope；endpoint stats 增加 model_api_circuits；本地等待单列 `lb_recovery_deferred` 和 `recovery_preference`。没有新增高基数 Prometheus 维度，也未删除旧指标。状态仍为进程内，不能作为多进程/多副本全局熔断。

## 验证

新增行为均先建立失败测试，再实现；既有测试断言未削弱。新增覆盖：协议原因链的安全提取、真实本地 TCP HTTP/2 reset/GOAWAY 经 HTTPX/httpcore 的转换、单次发送与资源回收、模型/API 隔离、共享保护、过期结果、取消和恢复、局部缓存上限、管理重置、恢复窗口到期、pinning、不新增 provider 回退，以及 JSON/SSE 提示和输入保真。

| 验证层 | 结果 |
|---|---|
| 本机 Python 3.14.5 全量回归 | `python3 -m pytest tests/ -q --tb=short`：816 passed，6 skipped，745 subtests passed，56.85 秒 |
| Linux/amd64、Python 3.12 最终应用镜像 | 9 个相关测试文件：100 passed，77 subtests passed，4.20 秒；包含本地真实 TCP HTTP/2 故障注入 |
| 源码与镜像 | main 导入成功；21 个运行文件与本地 hash 一致，runtime manifest 覆盖完整 |
| 文档与差异 | Author、Markdown fence、相对链接与 `git diff --check` 通过 |

六项跳过为既有隔离 MySQL 验收，当前未配置专用测试数据库；不宣称数据库集成通过。本轮未修改存储写入行为。镜像测试使用只读挂载的纯 Python pytest 测试库和测试目录，应用及其运行依赖来自实际构建镜像，容器设置 `--network none`。

本地镜像 tag：`claude-lb:stream-reliability-20261009-local`。本地 image ID：`sha256:9666a2136783886448b23694cdd29e4a95e2a1d80b54b6471d2c159073d0b0ec`；它不是 registry manifest digest。

运行文件 manifest SHA-256：`a6e675f86836667d18588d24a770e5066e3e4ba2580e78c64847004ffd8426d5`。构建基线 HEAD 为 `13641ad190850ee50584701093b22ccc05d51c19`，该镜像在本地候选尚未提交时构建，`source_tree_dirty=true` / `out_of_tree=true` 如实保留。构建成功和文件匹配不等于提交已合并或线上已升级。

## 未验证与交付状态

昨天的 120 秒关闭究竟来自模型服务还是中间代理仍未知。局部 reset 分类证明流的范围，不证明服务端根因；未知协议错误保守保留共享熔断。跨账户/模型真实服务、Mac 客户端呈现自定义头、长期负载和线上改善均未验收。GitHub 发布及 AKS 部署仍须遵循既有独立评审与发布门禁。

本轮未修改存储 schema、发布工具、部署参数、监控资产或 OpenClaw 自动化。
