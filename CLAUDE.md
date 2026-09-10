# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Databricks Claude Load Balancer - 一个智能负载均衡代理，支持：
- **Databricks Claude**: 将 Claude API 请求分发到多个 Databricks workspace 端点（`/anthropic/v1/messages`）
- **Azure OpenAI**: 将 OpenAI API 请求分发到多个 Azure OpenAI 区域端点（可选，支持 Responses API 和 Chat Completions API）
- **GitHub Copilot**: 将非 Anthropic 模型（gpt-*、gemini-* 等）通过 Copilot 上游 `https://api.githubcopilot.com` 转发，支持多 GitHub 账号、Device Flow 登录、token 自动刷新
- **统一路由**: `/v1/chat/completions` 与 `/v1/responses` 按模型自动分流：`claude-*` 永远走 Databricks；其他模型 **优先 Copilot，失败/不支持时 fallback Azure**
- **用量持久化**: 按天存储 token 用量，支持 JSON 文件或 MySQL 8.x 后端
- **成本追踪**: 内置模型定价，自动计算使用成本（GHCP 是订阅制，但用 OpenAI 标准价格表算 "假想成本" 用于 API 层成本对照）

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt
# MySQL 存储后端（可选）
pip install aiomysql

# 本地开发运行
python main.py
# 或带热重载
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# GitHub Copilot 登录（device flow，token 写入 ~/.config/databricks-claude-lb/copilot-auth-<name>.json，权限 0600）
python main.py --copilot-login --endpoint gh-account-1
# 移除某个 endpoint 的本地缓存
python main.py --copilot-logout --endpoint gh-account-1

# Docker 构建和运行
docker build -t claude-lb .
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml -v $(pwd)/usage_data:/app/usage_data claude-lb
```

## 架构概览

整个项目是单文件架构 (`main.py`)，包含以下核心模块：

### 模型映射与定价
- `DATABRICKS_MODELS`: Databricks 模型名常量映射
- `MODEL_PRICING`: Anthropic 官方 API 定价（USD/MTok）
- `get_databricks_model()`: 将 Claude 模型名映射到 Databricks 模型
  - `claude-*-sonnet-*` → `databricks-claude-sonnet-4-6`（默认），支持显式 4-5/4-6/5（`claude-sonnet-5*` 直通 `databricks-claude-sonnet-5`；正则 `sonnet[-_.]?5(?:[-_.]|$)` 优先于 `sonnet-*-*`）
  - `claude-*-opus-*` → `databricks-claude-opus-4-7`（默认），支持显式 4-5/4-6/4-7/4-8/5（`claude-opus-5*` 直通 `databricks-claude-opus-5`；正则 `opus[-_.]?5(?:[-_.]|$)` 优先于 `opus-*-*`）
  - `claude-*-haiku-*` → `databricks-claude-haiku-4-5`
- `get_model_pricing()` / `calculate_cost()`: 按模型名子串匹配定价并计算成本。定价与 Anthropic / OpenAI 2026-09-08 官方文档对齐（`MODEL_PRICING` 头部注释注明数据源）。新加：Sonnet 5、Opus 4.8/4.1、o4-mini、o3-pro、gpt-5-pro、gpt-5.3-codex、gpt-5.6-cyber、gpt-5.5-pro、gpt-5.4-{mini,nano,pro}、gpt-5.2-pro；修正：gpt-5.5/5.4/5.2/5.6-* 的 input/output 单价。

### 请求兼容性处理
- `strip_cache_control_extras()`: 清理 `cache_control` 中 Databricks 不支持的额外字段（如 `scope`），保留 `type: ephemeral`
- `proxy_request()` 中依次执行：模型名映射 → 移除不支持的顶层字段 → 清理 tools 字段 → 清理 content types → strip cache_control → thinking 参数转换

### 图片自动压缩（避开 ADB 4MB / GHCP 上游 limit）
- `compress_images_in_payload()` / `compress_images_async()`：递归遍历 payload，对超过 200KB 的 base64 图片解码 → 等比缩到 ≤1280px → JPEG q=82 重编码。压不小自动保留原图，无图请求几乎零开销
- 同时支持 **Anthropic 格式**（`{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`）和 **OpenAI 格式**（任意 `data:image/...;base64,...` data URL）；OpenAI 路径走 `data:image/...` 字符串扫描，Anthropic 路径走 image block 显式识别后改 `source.data` + `media_type=image/jpeg`
- 在三个入口（`/v1/messages`、`/v1/chat/completions`、`/v1/responses`）原始 body 大小超过 200KB 时触发，使用 `asyncio.to_thread` 跑 PIL，避免阻塞 event loop
- `MAX_REQUEST_SIZE = 4MB`（压缩后仍超 → 413，提示 `/clear`），`MAX_RAW_REQUEST_SIZE = 64MB`（入口粗暴上限保护 OOM）
- **不能解决 Claude Code 客户端 32MB 限制**：CC 在按 enter 之前就拦截，需要客户端绕过（小截图、Codex CLI 等）。LB 侧的压缩是给"32MB 以下、4MB 以上"的请求兜底
- Pillow 是必需依赖（已加入 `requirements.txt`）；未安装时压缩自动跳过（仅日志告警）
- **内存上界来自「顺序解码 + 请求级限流」，不是像素预算**：`compress_images_in_payload` 在单线程里逐张解码、每张位图用完即释放，所以单请求峰值是「一张位图」；`compress_images_async` 的 Semaphore 限的是**并发压缩的请求数**，于是全局峰值 ≈ `IMG_COMPRESS_CONCURRENCY` 张位图。`IMG_MAX_TOTAL_PIXELS` 约束的是累计解码工作量（延迟/CPU），两者目的不同，别混用来调参
- **准入被拒后的优雅降级会重跑准入**：`trim_excess_images` 丢掉最早的图后必须再过一遍 `check_image_admission`。只按张数 trim 到上限不保证像素预算也满足（50 张 4K 仍远超），不重跑等于让降级路径把预算整个绕过去。仍超则按新原因 413。**这是行为变化**：以往这类请求会被放行（慢但能跑），现在会 413
- **张数预算不可被畸形 base64 绕过**：`account()` 在尝试解码**之前**就记张数（从前 `b64decode` 抛异常会整体跳过计数，于是可以塞任意多个 image 节点）。解不开的图不计入像素数（PIL 同样解不开，压缩阶段不会解码，不构成 OOM 风险）
- 行为覆盖在 `tests/test_image_admission.py`（准入预算、trim 的占位类型与索引位移、压缩的降采样/转 JPEG/坏输入原样保留）

### 负载均衡
- `WorkspaceEndpoint` / `AzureOpenAIEndpoint` / `CopilotEndpoint`: 端点数据类
- `GlobalStats`: 全局统计数据类
- `LoadBalancer`: 支持 `least_requests` / `round_robin` / `random` 三种策略
- 熔断器机制: 错误达阈值自动禁用端点，超时后自动恢复；429 也触发熔断，4xx 客户端错误不触发
- `LoadBalancer.select_endpoint_for_model()` 仅给 Azure 用（按 `ep.deployments` 过滤）；Copilot 在 `CopilotProxy._select_endpoint()` 内自己实现选择，原因是 Copilot 的 `models` 字段语义为白名单且空列表 = 通配（"上游模型列表会变，不强制静态配置"），不能直接复用 Azure 那套

### Databricks 代理 (ClaudeProxy)
- `proxy_request()`: 代理请求入口，最多 3 次重试
- `_stream_request()`: SSE 流式响应处理，支持流内重试和 token 用量嗅探
- `_normal_request()`: 普通 JSON 响应处理
- `_record_usage()`: 记录 token 用量到内存统计 + `usage_store`
- 请求路径: `{endpoint.api_base}/anthropic/v1/messages`
- Thinking 参数自动转换: 由模块级 `supports_adaptive_thinking(databricks_model)` 判定。**黑名单策略**——只有 `opus-4-5` / `sonnet-4-5` 走 `enabled` + `budget_tokens`；其它模型（含 Opus 4.6/4.7、Sonnet 4.6、Opus 5 及未来更高版本）默认走 `adaptive`（移除多余 `budget_tokens`）。这样 Anthropic 后续发新模型无需改代码。旧模型分支保留 max_tokens 与 budget_tokens 冲突时的自动调整

### Azure OpenAI 代理 (AzureOpenAIProxy)
- `proxy_responses()`: Responses API 代理，URL: `{endpoint}/openai/v1/responses`
- `proxy_chat_completions()`: Chat Completions API 代理，URL: `{endpoint}/openai/deployments/{model}/chat/completions`
- Auth Header: `api-key: {endpoint.api_key}`
- 端点选择通过 `select_endpoint_for_model()` 按模型过滤

### GitHub Copilot 代理 (CopilotProxy)
- `proxy_chat_completions()`: URL `{session_base_url}/chat/completions`，OpenAI 风格
- `proxy_responses()`: URL `{session_base_url}/responses`，OpenAI Responses 风格（GPT-5 系列）
- 必带 headers (`COPILOT_HEADERS` 常量): `Editor-Version: vscode/1.104.0`（默认）、`Editor-Plugin-Version: copilot-chat/0.30.0`（默认）、`Copilot-Integration-Id: vscode-chat`（写死）、`User-Agent: GitHubCopilotChat/0.30.0`（默认）、`Openai-Organization: github-copilot`（写死）、`Openai-Intent: conversation-edits`、`X-Initiator: user`。前三项版本号由 env `COPILOT_EDITOR_VERSION` / `COPILOT_EDITOR_PLUGIN_VERSION` / `COPILOT_USER_AGENT` 覆盖，允许运维随 VS Code / Copilot Chat 官方版本自行滚动，无需改代码。业务请求另外按 stream/non-stream 附上 `Accept: text/event-stream` 或 `Accept: application/json`，并无条件加 `Accept-Encoding: gzip, deflate` + `Accept-Language: en-US,en;q=0.9` —— 真实 VS Code Copilot Chat 扩展会带这些字段，缺失即被 Cloudflare bot management 判为客户端指纹异常。模仿 VS Code Copilot Chat 扩展，**这些字段必须保留**，否则上游会 401/403 或返 CDN 挑战页 HTML
- 视觉请求自动加 `Copilot-Vision-Request: true`（检测 messages.content 中是否含 `image_url`）
- Auth Header: `Authorization: Bearer {short_lived_session_token}`

#### Token 双层模型（生产级自愈，AKS 友好）

- **Long-lived OAuth token**（GitHub 端基本不过期，除非用户撤销）：
  - 来源优先级 = config.yaml `github_token` > 本项目 device-flow 缓存 `~/.config/databricks-claude-lb/copilot-auth-<name>.json` > 兼容 copilot-lb 旧缓存 `~/.config/copilot-lb/auth.json`
  - **运行时按需重读**：`resolve_github_token()` 返回 `(token, source_dict)`，`source_dict` 记录可重读的来源（`env` / `file` / `literal`）。`reload_github_token(endpoint)` 从源重读，配合 K8s Secret rotation：mounted secret 文件被 kubelet 异步同步（约 1 min 周期），**Pod 内自动 pick up，零重启**
- **Short-lived Copilot session token**：**TTL 由上游 `expires_at` 决定，不要假定固定值**。2026-09-09 对本账户三次独立交换实测均 ≈ **24 h（86400 s）**（旧文档写「30 min」是错的，已订正）。以 `copilot_session_token_remaining_seconds` 为准
  - 从 long-lived token 调 `https://api.github.com/copilot_internal/v2/token` 交换
  - **内存缓存** + 过期前 60 s 自动刷新；并发请求由 per-endpoint `asyncio.Lock` 串行化
  - **后台主动刷新 task**（`background_refresh_loop`）：每 `COPILOT_REFRESH_INTERVAL`（默认 300s）秒扫一遍，剩 ≤`COPILOT_REFRESH_THRESHOLD`（默认 600s）就主动刷新；即使长时间无请求也保持新鲜。在 24 h TTL 下这等于**每天刷新 1 次**，属正常 —— 别照「30 min TTL」的假设去调这两个值
  - **不持久化是有意的**：session token 只在内存（`CopilotEndpoint.session_token`）。TTL 既然是 24 h，重启代价 = 每 endpoint 一次 HTTP GET；把短期凭证落盘是负收益。且**实测 session token 轮换不会使 opaque state 失效**（见 `docs/TROUBLESHOOTING.md` 第 15 节），所以重启也不会让客户端手里的会话状态作废
  - **请求级 401 自愈**：上游 `/chat/completions`、`/responses` 返回 401 → `force=True` 重刷 session token → 同 endpoint 重试一次（`_proxy` 和 `_stream_response` 内都做了）
  - **token-exchange 401 自愈**：`get_session_token()` 收到 401 → `_LongLivedTokenInvalidError` → 调 `reload_github_token()` 从源重读 long-lived token → 再交换一次；仍失败则该 endpoint 进熔断池（`circuit_open=True` + readiness probe 反映）
- Device Flow CLI: `python main.py --copilot-login --endpoint <name>` 使用 VS Code 公开 client_id `Iv1.b507a08c87ecfe98` 走标准 GitHub Device Flow，token 写入 0600 权限文件
- 启动时 `asyncio.create_task(copilot_proxy.warmup())` + 启动 `background_refresh_loop`
- `POST /admin/copilot/reload`（运维端点）：立即从源重读所有 endpoint 的 long-lived token + 强制刷新 session token，配合 K8s Secret rotation 实现"零延迟"生效（不等 kubelet 同步周期）
- `POST /admin/copilot/reset-pool`（运维端点）：怀疑 httpx 连接池泄漏或 keepalive 卡半开时的自救按钮。新建 client、原子替换、后台 30s 后 aclose 旧 client、再 warmup 一次；返回 `previous_pool` 快照给对照。
- `CopilotEndpoint.models` 语义：**空列表 = 通配**（接受所有模型）；填值则只服务列表内模型。和 Azure 的 `deployments` 语义不同（Azure 必须列出可用部署），所以 Copilot 自己实现 `_select_endpoint(model)` 不复用 `LoadBalancer.select_endpoint_for_model()`
- `_UnsupportedModelError`: non-stream/buffered 请求可在 HTTP response start 前由路由层捕获并 fallback Azure；direct streaming 已提交 HTTP 200 后不能透明切 provider，此时返回明确的 `unsupported_model` SSE error + `[DONE]`
- 路由分流入口: `_route_openai_chat()` / `_route_openai_responses()`：先检查 `claude-*` 名拒绝（必须走 `/v1/messages`）；再 try Copilot；如抛 `_UnsupportedModelError` 或 HTTPException 404/503 就 fallback Azure

#### 可观测性 / 探针

- **`/health/live`**：仅检查进程能响应（K8s livenessProbe，避免上游故障导致 Pod 被 kill）
- **`/health/ready`**：检查依赖就绪 — 至少 1 个 ADB endpoint 可用 + 至少 1 个 Copilot endpoint token 有效（K8s readinessProbe）
- **`/metrics`**：Prometheus 文本格式，除 token/circuit/provider 指标外，还暴露 `copilot_stream_connections_active`、`copilot_stream_connection_oldest_seconds`、`copilot_stream_upstream_idle_max_seconds`、`copilot_stream_disconnects_detected_total`、`copilot_stream_forced_releases_total`、`copilot_pool_timeout_total` 等连接生命周期指标
- **JSON 日志**：`LOG_FORMAT=json` 切换；字段 `ts`/`level`/`logger`/`message`，AKS Log Analytics 可直接 KQL 解析；接管 `uvicorn`/`uvicorn.error`/`uvicorn.access`/`httpx` logger 统一格式
- **环境变量控制**：除日志/token 刷新变量外，连接监控支持 `COPILOT_STREAM_HIGH_WATERMARK=400`、`COPILOT_STREAM_OVERLOAD_GRACE=30`、`COPILOT_STREAM_DISCONNECT_GRACE=15`、`COPILOT_STREAM_MONITOR_INTERVAL=5`；httpx 池另有 `COPILOT_POOL_MAX_CONNECTIONS=500`、`COPILOT_POOL_MAX_KEEPALIVE=200`、`COPILOT_POOL_KEEPALIVE_EXPIRY=30`、`COPILOT_POOL_ACQUIRE_TIMEOUT=20`（默认从 60→20，配合"单端点池获取超时快速失败"给客户端更快 error 让 Codex 自己 retry；恢复旧值设 `60`）、`COPILOT_POOL_READ_TIMEOUT`（默认无上限；设 `None`/`0`/空 = 无上限，设正数则强制封顶，仅在极端诊断场景使用）；SSE 心跳间隔 `STREAM_HEARTBEAT_INTERVAL=15`；PoolTimeout 触发时的 DNS+TCP 探针 `COPILOT_UPSTREAM_PROBE_TIMEOUT=3` / `COPILOT_UPSTREAM_PROBE_CACHE_TTL=5`；请求头版本号 `COPILOT_EDITOR_VERSION=vscode/1.104.0` / `COPILOT_EDITOR_PLUGIN_VERSION=copilot-chat/0.30.0` / `COPILOT_USER_AGENT=GitHubCopilotChat/0.30.0`；HTML 上游诊断软熔断 `COPILOT_HTML_SOFT_COOLDOWN=30`（秒；0 = 关闭）

#### 上游 HTML 挑战页识别 + 软熔断

Cloudflare / GHCP CDN 偶发会给单条请求返 `Content-Type: text/html`（"Just a moment..." 挑战页），有时甚至 status=200。旧版 LB 只识别 `status ≥ 400` 的 HTML，pump 会把 200-with-HTML 原样透传给客户端 SSE 消费者，客户端 SDK 解析失败自己报 "HTML error page"。

- **早期 content-type 校验**：`_stream_response` 与 `_normal_request` 在拿到 upstream response headers 后立即检查 `Content-Type`。凡 `text/html*`（无论 status），一律走 `_build_upstream_error_detail` 的 `upstream_html_error` 分支，生成规范化 JSON 错误 + `_apply_html_cooldown` 触发软熔断。
- **软熔断窗口 30s**（`COPILOT_HTML_SOFT_COOLDOWN`）：命中的 endpoint 在窗口内被 `_select_endpoint` 优先跳过；所有 endpoint 都在冷却时降级到"最小活跃"选一个而不是拒绝服务；不动 `total_errors` 也不置 `circuit_open`。
- **上游诊断 header 记录**：`_extract_upstream_ids` 抽 `cf-ray` / `x-github-request-id` / `x-request-id` / `server` / `x-served-by` / `x-cache` 六个字段，塞进结构化日志 (`kind=copilot_upstream_html`) 与 SSE error message 尾部（客户端可见），运维拿这些去找 GitHub Support 反馈最有效。
- **Metrics**：`copilot_upstream_html_events_total{endpoint}` (per-endpoint counter)、`copilot_upstream_html_events_all_total` (aggregate)、`copilot_upstream_html_events_by_status_total{status_bucket}` (按上游 HTTP 桶拆分 200/4xx/5xx/other，用来区分 Cloudflare 200-HTML 挑战页 vs 上游服务错误)、`copilot_html_soft_cooldown_active{endpoint}` (gauge 0/1)、`copilot_html_soft_cooldown_remaining_seconds{endpoint}` (gauge)。
- **SSE error 结构化 upstream_ids**：除 6 个上游诊断 header，还额外挂 `lb_request_id`（即本 LB 生成的 X-Request-Id），运维在客户端错误面板 → LB 结构化日志一次跳转就能完成关联，不需要靠 message 前缀 `[req=…]` 手动扒。

#### GHCP connection-bound `input[*].id` 剥离（修 "input item does not belong to this connection"）

**2026-09-09 直连 GHCP Enterprise 上游实测的结论**：GHCP 给 Responses output item 铸造的 `id` **不是** OpenAI 的 `msg_xxx` 短 id，而是 **424~428 字符的签名不透明 blob**，且被密码学校验 —— 篡改任意 20 字符即 400（验签失败后落到通用 schema 的「max length 64」分支）。这个 blob 绑定在 GHCP 服务端某个 "connection" 上；该 connection 消亡后，客户端仍在回放的旧 id 会让**整个会话永久被拒**：

```json
{"code":"bad_request","type":"websocket_error",
 "message":"input item ID does not belong to this connection"}
```

上游多仓库交叉印证：`github/copilot-cli#2147` 官方根因是 "stale WebSocket state being reused after a reconnection"；`#4505` 记录中断后恢复旧会话使该会话永久失败（`/fork` 也救不回，新会话正常）；`caozhiyuan/copilot-api#235` 是与本 LB 同形态的代理，多实例分流下 `/responses` 多轮几乎必挂。

**对 LB 的意义**：我们自己每一次中途断流（silent truncation / PoolTimeout / 换端点 / HTML 软熔断 / HTTP/2 `ConnectionTerminated`）都会让 Codex 留下半成品 item 并持续回放其 id ——「一次断流」于是升级成「该会话此后每轮都失败」。客户端同时看到的 `stream disconnected before completion` 与 `input item does not belong to this connection` 正是**因→果**关系。

**实现**：
- `CopilotProxy._strip_input_item_ids(body, api_type)`：在 `_proxy` 里对所有 Responses 请求剥掉 `input[*].id`。**只剥 item 顶层 `id`**；`call_id` 与 `encrypted_content` 一律保留（实测二者互相密码学绑定，错配会 400）。幂等、对 Chat 协议返 0
- 放在 `is_stateful` 计算**之后**：statefulness 只看 `previous_response_id` / `encrypted_content`，不看 `id`，所以剥离不改变 pinning 判定（有回归测试锁住）
- 兜底：`_is_orphaned_item_id_error(status, body)` 识别 400/401 + `does not belong to this connection`（上游大小写不稳定，故只匹配这个稳定子串）。命中且 `not sent_any_chunk` 时就地剥 id 重试一次 —— 流式沿用 401 auth-repair 的「同 endpoint、同 lease」形状，非流式递归一次。因剥离幂等，天然最多重试一次
- 开关 `COPILOT_STRIP_INPUT_ITEM_IDS`（默认 **true**）；`false` 恢复原样透传，仅用于排查
- **实测代价为零**（逐维度验证）：message ✅ / reasoning 剥 id 保留 `encrypted_content` 上下文不丢 ✅ / `function_call` 回路靠 `call_id` 正确配对 ✅ / **prompt cache `cached_tokens` 与保留 id 完全相同**（4063/4066）✅
- **Metrics**：`copilot_input_item_ids_stripped_total`、`copilot_input_item_ids_stripped_requests_total`（两者相除 = 客户端平均每请求回放多少个 id）、`copilot_orphaned_item_id_events_total{stage="detected|recovered|unrecoverable"}`。**stage 语义在 2026-09-10 被生产反证并更正**：真正的「还有别的 id 通道」金丝雀是 **`recovered`**（事后还能剥到 id），不是 `detected` —— 生产 `detected=12` 而日志明写 `no input[*].id to strip`，因为上游拒绝的是 `encrypted_content` 的**归属**，主动剥 id 结构上防不住。`unrecoverable` 现在只在恢复阶梯彻底走完后记一次（详见下一节）

#### 有状态请求（opaque state）跨账户重放保护

Handoff §7.2 实证：同一 Responses opaque reasoning state（`previous_response_id` / `input[*].encrypted_content`）在原账户 200，换另一账户 401。而 LB 在 5xx / PoolTimeout / 网络错误 / HTML cooldown 时会自动切 endpoint，一旦客户端带着 opaque state 走到备份 endpoint 就必然 401。修复：

- **`_request_has_opaque_state(body, api_type)`**：检测 `previous_response_id` 或 `input[*].content[*].encrypted_content`（Chat 协议返 False）
- **`_select_endpoint(model, *, pinned=None)`**：pinned 非空时仅当 pinned 仍在可用池内才返回它（**忽略 HTML cooldown**——宁可再踩一次挑战页也不换账户），否则返 None
- **`_proxy` 层**：stateful 请求首次选中后钉住，后续 `_select_endpoint` 传 `pinned=first_endpoint`；pinned 掉线时抛 503 `stateful_pinned_endpoint_unavailable` 让客户端重构会话，绝不静默换账户
- **`_stream_response` 内 3 处换 endpoint 分支**（5xx retry / PoolTimeout retry / 网络错误 retry）同样传 `pinned=stateful_pin`；无法换端点时 `_note_stateful_pin(reason)` 计数 + 明确 SSE error 让客户端 retry
- **无状态请求维持原 failover** —— pinning 不影响普通请求的正常端点切换
- **Metrics**：`copilot_stateful_request_pinned_total{reason}`，reason ∈ `{http_5xx, network_error, pinned_unavailable, pool_acquire_timeout}`（即 `_STATEFUL_PIN_REASONS`，与 `_note_stateful_pin` 的四个调用点一一对应；旧文档写的 `pool_timeout` 不存在）；非零就意味着 pinning 成功挡下了会 401 的跨账户重放

#### 上游 401 按来源分类（防一个坏会话熔断整个账户）

`CopilotProxy._classify_upstream_failure(status, request_is_stateful=...)` 取代了 Copilot 两处的原地 `is_client_error` 字面量（`_proxy` 与 `_stream_response`，**仅此两处**）。除 401 外所有状态码行为逐位不变。

**为什么要分**（2026-09-09 实测）：401 被排除在 `is_client_error` 之外 ⇒ `failed=True` ⇒ `consecutive_errors += 1` ⇒ 连续 5 次打开 **endpoint 级**熔断。而一个 Copilot endpoint 承载该账户全部模型，所以「一个会话持续 401」会升级成「该账户所有模型下线」。实测计数关系：一个持续 401 的请求贡献 **1** 次（首次 401 走 auth repair 的 `continue`，在 `end_current_request` 之前，不计数）；第 5 个请求打开熔断；熔断后 `_select_endpoint` 对任意模型、任意 api_type 都返 `None`。

**判据 = `_request_has_opaque_state`**：

| 401 来源 | 分类 | 处理 |
|---|---|---|
| 请求携带 opaque state（跨账户重放、失效 item id） | request-scoped | 中性，不计入 `consecutive_errors` |
| 无 opaque state（凭证/席位坏了） | endpoint-scoped | 照旧计数 → 照旧熔断 |

**为什么这个判据对而不只是够用**：`_select_endpoint` 对 stateful 请求返 `pinned if pinned in matched else None`，`matched` 已排除熔断端点 —— 所以**对 stateful 请求熔断只有害无益**：杀掉该会话唯一可能服务的 endpoint 并带走该账户其他全部流量，而 failover 本来就被 pinning 禁止，熔断换不来任何可用性。

**覆盖不会变窄**：seat/policy 被撤时 `_exchange_token` 仍成功 ⇒ `auth_unhealthy` 恒 False ⇒ `_mark_endpoint_unhealthy` 永不触发 ⇒ **只有计数能熔断它**。但那种情况下**所有**请求都 401，包括每个新会话第一轮和全部 Chat Completions 流量（`_request_has_opaque_state` 对 chat 恒返 False），这些是无状态的，照旧计数。（对比：long-lived token 真失效时 `_mark_endpoint_unhealthy` 会**直接** `_open()`，实测 `circuit_open=True` 而 `consecutive_errors=0`，不依赖计数，因此本改动不影响它。）

**403 / 429 不动**：429 是服务端过载，403 多为 Cloudflare bot management，两者确是 endpoint 级信号。无任何实测到的 request-scoped 403 实例，证据不足就不改。

- 开关 `COPILOT_STATEFUL_401_NEUTRAL`（默认 **true**）；`false` 恢复旧行为，仅回退排查用
- **Metrics**：`copilot_upstream_401_total{scope="request|endpoint"}`。`request` 高而 `endpoint` 为 0 = 客户端在回放坏状态、账户是好的；`endpoint` 持续增长 = 真凭证/席位问题
- **ADB / Azure 那 4 处不要改**：401 在那里确等于 `dapi` token / `api-key` 坏 = endpoint 不健康，计数是对的。`tests/test_copilot_401_circuit_scope.py::OtherProvidersMustNotBePatchedTests` 有结构守卫锁住调用点数量

#### opaque state 被上游拒绝时的有界恢复阶梯

上一节保住了「一个坏会话不熔断整个账户」，但那些请求**一个都救不回**。2026-09-10 生产：**6 个唯一请求**以不可恢复的 401 收场（末次核验累计 6/381 ≈ 1.6%），`recovered` 恒 0，`detected` / `unrecoverable` 各被记了两遍。注意这 6 个是**有界的历史插曲而非持续失败率** —— 三次抓取跨流量 ~95 → 218 → 381（主动剥离 3383 → 27634）期间那几个计数器一个都没动；它的严重性在于「一旦发生该会话就永久废掉且毫无出路」，不在频率。完整证据链与发布验收在 `docs/OPAQUE_STATE.md`，排查手册在 `docs/TROUBLESHOOTING.md` 第 17 节。

**上游是两级独立校验**（实测：篡改 `encrypted_content` 中间 20 字符）：

| 阶段 | 失败报错 | 分类 kind |
|---|---|---|
| 先解密 / 解析 | `invalid_request_body` "could not be verified" | `unverifiable_content` |
| 再校验归属 | "input item does not belong to this connection" | `orphaned_id` |

所以 orphan 报错意味着 blob **能解开但不属于这条 connection**；第一类此前完全没有兜底。判据是 `CopilotProxy._classify_opaque_state_rejection(status, body_text)`（仅 400/401）。

`unverifiable_content` 要求**同时**命中「主题是 encrypted content」与「校验失败措辞」两半。只匹配 `could not be verified` 会误吞别的 400（如 `The provided API key could not be verified`），代价很实：无理由删掉用户的推理 blob，还把真实原因改写成 `orphaned_conversation_state` 藏起来。

**阶梯**（`_OpaqueStateRecovery`，`MAX_RUNGS = 2`）：

```
rung 0  发送前主动剥 input[*].id                  COPILOT_STRIP_INPUT_ITEM_IDS（默认 true）
rung 1  被拒后再剥一次 id                         只在 rung 0 关掉 / 出现新 id 通道时有料
rung 2  删 reasoning item 的 encrypted_content    COPILOT_OPAQUE_STATE_RECOVERY（默认 true）
走完    orphaned_conversation_state 错误           明确要求客户端重建，绝不静默成功
```

- **预算是显式计数，不靠幂等**：外层 attempt 循环、401 auth repair、`_normal_request` 递归是三条能叠加的通道，一个客户端请求共用一份 `recovery`。**实测上界**：流式 ≤3 次上游调用（`for attempt in range(3)` 是硬顶），非流式 ≤4（1 原始 + 1 auth repair + 2 级恢复，递归深度 ≤2）；生产形态（主动剥离已开、纯 orphan）= 2 次，与改动前相同
- **循环不变量：每个 `continue` 都必须留下一轮来消费修好的请求。** 端点切换类的三处（5xx / PoolTimeout / 网络错误）本来就带 `attempt < max_retries - 1`；原地修复类（恢复 rung、auth repair）过去靠「剥离幂等 + auth 限死 `attempt == 0`」凑出上限 2 恰好留一轮，rung 提到 2 级并把 auth 与 attempt 解耦后那个巧合没了。**流式循环体就是生成器体的最后一段，耗尽即静默结束**（下游 HTTP 200 + 0 字节 —— 正是上一节里会毒化会话的无终端断流）。所以两处原地修复显式带上同一条守卫，另加一层循环后兜底终端（`retry_budget_exhausted`，按不变量不可达，只为防后人漏守卫）+ 一个**必须恒 0 的金丝雀指标** `copilot_stream_retry_budget_exhausted_total`（有零样本，告警写 `> 0`，不要靠 grep 日志）。回归测试把 `MAX_RUNGS` 调高来验证不变量而不是验证「2」这个数字
- **重放安全**：只在 `status ∈ {400,401}` 且流式 `not sent_any_chunk` / 非流式尚未产出字节时触发 —— 上游明确拒绝 + 下游未提交任何响应，符合 `docs/RESILIENCE.md` 的 POST replay 约束。同 endpoint、同 lease，绝不换账户
- **删 blob 代价实测为零**：`input_tokens=1637 / cached_tokens=1280` 在「保留 / 删 blob / 整条删 reasoning」三组完全一致（blob 不计入 input tokens）；`call_id` 回路正常；GHCP 不支持 `previous_response_id`，所以历史必然全在 `input[]` 里，**不可能静默丢服务端历史**
- **不做主动删除**：只在被拒绝后删，否则每个请求都白丢推理连续性
- **rung 2 之后不再加「整条删 reasoning item」**：实测同样 200，但删 blob 后已无 opaque 载体，救不回任何东西

**opaque-state 拒绝不再喂 401 auth repair**：刷 session token 对它注定无用（实测轮换不使 opaque state 失效），只会白打一次上游并把 orphan 计数器翻倍。顺带把 auth repair 的预算从 `attempt == 0` 改成**每请求一次的独立 flag**，修掉之前记录为「已知次要交互，未修」的那条。副作用：换新 endpoint 后的 401 现在也能拿到一次刷新（上界不变）。

- **Metrics**：`copilot_opaque_state_requests_total`（受影响的**唯一客户端请求**数，运维该读这个）、`copilot_opaque_state_rejections_total{kind}`（拒绝**事件**数）、`copilot_opaque_state_recovery_total{outcome="attempted|succeeded|exhausted"}`。**`succeeded` 只在最终有效完成后记** —— 流式挂终端事件 `completed`、非流式挂真正返回；上游收下改写后的请求但流被截断不算成功
- **诚实边界**：`does not belong to this connection` 需要另一个账户铸造的 blob 才能复现，单账号造不出，所以「rung 2 能救那一类」是**推断**。已实测的是删后形态合法/上下文保真，以及另一类 opaque-state 拒绝确定被救回。上线后看 `succeeded/attempted` 把推断变成实测；不成立就关掉开关回到根因调查
- 开关 `COPILOT_OPAQUE_STATE_RECOVERY`（默认 **true**）；`false` 只关 rung 2，仍记 `exhausted` 并返回结构化错误

#### 会话亲和（多账户的硬前置条件，2026-09-10 实测后实现）

**为什么必须有**（两个真实 GHCP 账户实测，`api.enterprise.*` + `api.business.*`，模型 `gpt-6-astra`）：A 账户铸造的 reasoning `encrypted_content` 拿到 B 账户回放**必然** 401 `input item does not belong to this connection` —— **双向 × 3 次重复 = 24/24 确定性**。而 `least_requests` 会把同一会话的连续轮次分到不同账户，所以只要配了第二个账户，这个 401 就是**必然事件**而非偶发。

- **键 = `prompt_cache_key`**（`_session_affinity_key`）：Codex CLI 用它做 prompt cache，实测跨 6 次客户端重试一致、跨同一会话的多个轮次一致，等于 Codex 的 `session_id` / `thread_id`。**故意不用 api_key 做键** —— 同一 key 承载多会话，那既没用又引入跨会话关联。Chat 协议没有这个字段也没有 opaque state，返 None
- **哈希必须稳定**（`_affinity_index` 用 **blake2b**）：Python 内置 `hash()` 对 str 加 per-process 随机盐，多副本之间结果不同 —— 那正好破坏这个机制的唯一目的。有子进程测试用不同 `PYTHONHASHSEED` 验证
- **哈希打在「配置态合格集合」上，不是「当前可用集合」**：打在可用集合上会让任何一次熔断把**所有**会话重新洗牌，等于给每个活跃会话制造一次跨账户回放；打在配置集合上只影响原本映射到那个端点的会话
- **对所有 Responses 请求生效，不只是 stateful 的**：第一轮是无状态的，但它铸造的 state 第二轮就要回放，等到 stateful 才钉已经晚了
- **与 pinning 是组合而非竞争**：pinning 管「同一请求内不许换」，亲和管「下一个请求回到同一个」。两者都适用时 pinning 优先
- **单账户下是完全的 no-op**（只有一个合格端点时不走亲和分支），所以默认开对当前生产零影响
- 开关 `COPILOT_SESSION_AFFINITY`（默认 **true**）
- **Metrics**：`copilot_session_affinity_total{outcome="hit|unavailable|absent"}`。`hit` = 挡下了一次必然的 401；`unavailable` = 亲和目标熔断，该轮会跨账户但由恢复阶梯兜住（代价是丢一次推理链）；`absent` = 非 Codex 客户端没带 `prompt_cache_key`。**单账户下 hit/unavailable 恒为 0**

端到端实测（两个真实账户 + 真实 codex-cli 0.145.0，同一会话三轮）：

| | 亲和开 | 亲和关 |
|---|---|---|
| 各账户请求数 | 3 / 0 | 3 / 3 |
| `opaque_state_requests_total` | **0** | **3** |
| `recovery{succeeded}` | 0 | **3** |
| 轮次成功 | 3/3 | 6/6 |

亲和从根上消除故障；漏过去的由恢复阶梯救回（推理链被删、请求换账户之后答案仍然正确）。

#### 客户端重试放大（实测 codex-cli 0.145.0，决定错误码怎么选）

| LB 返回 | 客户端打上游次数 | 表现 |
|---|---|---|
| **400** | **1** | 立即放弃，**把错误体逐字显示给用户** |
| 401 / 403 / 409 / 422 | **6** | 5 次 `Reconnecting... N/5`；409/422 还会把我们的 `code` 吞掉 |
| 429（有无 `Retry-After` 都一样） | 1 | 立即放弃 —— 所以 LB 自己内部重试 429 是必要的 |
| 500 / 503 | **30+** | 重试风暴 |
| 已提交 200 流 + SSE error | 6 | 5 次 reconnect |

- **交接文档把 409/422 列为候选是基于未验证的假设**：它们与 401 同为 6 次，换过去毫无改善。非流式的阶梯耗尽因此改用 **400**（确定不可重试的拒绝就该让客户端立刻停手）
- **流式改不了**：ASGI 的 `http.response.start` 在生成器被迭代前就发了 200，只能发 SSE error 帧（实测仍 6 次）。要改必须把 response-start 延后到第一个真 chunk，属 `docs/STREAM_OWNERSHIP.md` 的架构改动。流式路径的真正解法是会话亲和
- 生产那 `12/12/6/6` 的重新解读：**6 个是 LB 请求，很可能只是 1 个用户轮次被客户端重试 6 次**（实测 6 次重试的请求体语义完全相同）

#### dict 驱动的 label 指标必须有零样本

`orphaned_item_id_events` / `stateful_pinned_events` / `upstream_html_events_by_status` / `upstream_401_events` / `opaque_state_rejections` / `opaque_state_recovery` / `session_affinity_events` 都是运行期事件填充的 dict，健康态下为空。直接按 dict 生成样本会导致 `/metrics` 里**只有 HELP/TYPE、零条 sample** —— 运维分不清「零事件」和「LB 没部署」，只能写 `absent()`；尤其 `orphaned_item_id_events` 是 `67e91cd` 的金丝雀（要求恒 0），没有 0 序列就等于没有金丝雀。

- `_labeled_counter_samples(name, label, counts, known)` 在 **exposition 层**补零，已知 label 全集是模块级常量 `_ORPHANED_ITEM_ID_STAGES` / `_STATEFUL_PIN_REASONS` / `_UPSTREAM_HTML_BUCKETS` / `_UPSTREAM_401_SCOPES` / `_OPAQUE_STATE_KINDS` / `_OPAQUE_STATE_RECOVERY_OUTCOMES` / `_SESSION_AFFINITY_OUTCOMES`
- **只补零，不给 dict 播种**：在 `__init__` 里播种会改变 dict 自身语义并打破既有断言（如 `assertNotIn("unrecoverable", ...)`、`assertEqual(upstream_html_events_by_status, {"5xx": 1})`）
- 未预置的新 label 仍会照常出现（补零在 `merged.update(counts)` 之前），所以加了新 reason 忘了更新常量只会少一条零样本，不会吞掉真实计数
- `stream_truncated_no_completion_by_model` 的 label 是开放的模型名集合，**不预置**

#### PoolTimeout 诊断（分类是**中立的**，不做因果断言）

- `httpx.PoolTimeout` 触发时 `_describe_pool_timeout` 会同时抓 httpx 内部池状态（`_transport._pool` introspection：`total/active/idle/closing/requests_waiting`）+ 对 upstream host 做一次 3s 内的 DNS + TCP 探针（5s TTL cache，`_probe_upstream_connect`），把结果写进 structured log（`kind=copilot_pool_timeout`，字段含 `classification / httpx_pool / httpx_pool_observed_full / upstream_probe / per_endpoint / request_id`）并塞到给 Codex 的 SSE `error.message` 尾部（`probe.ok=... dns_ms=... tcp_ms=... ips=...`）
- **`classification` 恒为 `"pool_acquire_timeout"` 这一个值**，SSE `error.code` 同名。`PoolTimeout` 只说明「等池分配超时」，**不能证明** TCP/TLS 建连失败，事后补做的探针也无法确立当时的因果 —— 所以从前那套 `upstream_connect_stalled` / `local_pool_saturated` 二分类被**撤回**了（迁移记录见 `docs/STREAM_OWNERSHIP.md`「Diagnostic/API compatibility」）。取而代之的是一个纯观测字段 `httpx_pool_observed_full`（bool）与计数器 `copilot_pool_timeout_saturated_total`（观测到池满时 +1）
- `copilot_pool_timeout_upstream_stall_total` 是那次撤回的遗留，**结构性恒 0、没有任何自增点**，HELP 文本已标 `Deprecated`。**不要**基于它写告警或做判断；权威计数是 `copilot_pool_timeout_total`
- **单 Copilot endpoint 一律快速失败**（判据是 `len(endpoints) <= 1`，**不看** classification）：所有 endpoint 共享同一个 `httpx.AsyncClient`，重试还是打同一个 pool，只会让用户多等一个 `POOL_ACQUIRE_TIMEOUT` — 直接给客户端 error，Codex 自己 retry 得更快
- 多端点仍走原退避 + `_select_endpoint` 换端点重试逻辑（受 `attempt < max_retries - 1` 约束）

详细 AKS 部署步骤、Token rotation 流程、监控告警建议见 `docs/AKS.md`。常见客户端 / 上游异常排查见 `docs/TROUBLESHOOTING.md`（含 macOS 系统代理拦截 localhost、CC 32MB 限制、ADB 4MB / GHCP 模型 API 约束、token exchange 池隔离等）。

**设计契约文档**（生产分支合入 2026-09-08）：
- `docs/STREAM_OWNERSHIP.md`：streaming response 所有权、pool timeout 语义、shielded cleanup 契约 —— 说明 `_finish_cleanup` / `_close_stream_resources` / `_OwnedResponseStream` 三层保护如何确保 ASGI 取消与 upstream close join。
- `docs/OPAQUE_STATE.md`：opaque state 拒绝分类、有界恢复阶梯、**实测证据与推断分栏**、发布前验收与观察清单。发布复核先读这份。
- `docs/RESILIENCE.md`：本地 admission、CLOSED/OPEN/HALF_OPEN 熔断状态机、`RequestAttempt` per-lease identity、POST replay 禁令、cross-provider fallback 边界。**stateful pinning 与 admission 组合语义**：pinned endpoint 必须通过 circuit（不能 bypass HALF_OPEN 锁），HTML cooldown 对 pin 无效，POST replay 禁令覆盖 pinning（`sent_any_chunk=True` 后一律不重试）。
- `docs/STREAM_PROTOCOL.md`：WHATWG SSE 完整帧解析、`PER_PENDING_EVENT` / `PER_PROCESS_TOTAL_RETAINED_STREAM_BUFFER` 字节预算、Responses JSON `type` discriminator 优先于 event header、gzip incremental decompression、Copilot endpoint `api_types` 契约。**api_types 与 pinning 的组合**：pinned endpoint 若不支持当前 api_type 视为不可用（返 None、抛 503），不允许静默换 account。

### 流式代理健壮性（`_stream_request` / `_stream_response`）
- **httpx 客户端**: Databricks/Azure 保持 `200/50` 与 `pool=30s`；Copilot 独立使用 `500/200` 与 `pool=60s`。三者 `read` 默认都为 `None`，长 thinking 不能用整体 read timeout 误杀。Copilot 的 `read` 可通过 `COPILOT_POOL_READ_TIMEOUT` 强制封顶（罕见场景需要），设为 `None`/`0`/空表示无上限
- **Copilot lifecycle**: 每个 streaming attempt 使用 exactly-once lease；自定义 `StreamingResponse` 在 ASGI downstream send 失败/取消时显式 `aclose()` body iterator，确保 pump、upstream response 和 `active_requests` 一起释放
- **高水位兜底**: 后台 monitor 仅在 active requests ≥400 持续 30s 后检查异常 lease；只回收 owner 已结束或 downstream 连续确认断开 15s 的连接。请求总年龄和 upstream idle 仅做观测，绝不单独作为 kill 条件
- **异常覆盖**: 网络异常 `except` 分支同时捕获 `ConnectTimeout / ReadTimeout / WriteTimeout / ConnectError / RemoteProtocolError / ReadError / WriteError`，上游中途断流（最常见症状即是客户端报 "socket connection closed unexpectedly"）也走熔断 + 切端点重试路径；`httpx.ReadTimeout` 在 Copilot 侧另计 `stream_read_timeout_total`，默认 `read=None` 下应恒为 0
- **响应清理**: `response = None` 局部变量 + `finally: await response.aclose()`，避免 httpx 连接池堆积半开连接
- **后台 pump + 心跳**: 将 `aiter_bytes()` 放入 `asyncio.create_task(_pump(response))`，主循环 `await asyncio.wait_for(queue.get(), timeout=15.0)`；15 秒无 chunk 则同时 `_touch_stream(connection_id)` + yield 一次 `: keep-alive\n\n`（SSE 注释，Anthropic/OpenAI SDK 会忽略），刷新中间链路 idle 计时且让 monitor 的 upstream_idle gauge 保持真实。`_await_with_heartbeat` 在等待 headers 阶段也同样心跳 + touch。pump 端观察 `queue.full()` 事件并累计 `stream_pump_queue_full_events_total`。pump 退出时由 `finally` 分支 `pump_task.cancel()` 回收
- **SSE 终止规范化 (Anthropic)**：不变量是**每条错误路径都必须发出 `event: error`**。2026-09-10 用 anthropic SDK 1.3.0 实测三种收尾：只发 `error` → 抛 `APIStatusError` 带我们的消息（干净）；**先补 `message_stop` 再发 `error` → 结果完全相同，补发毫无作用**；**直接断流、无任何终端 → SDK 不抛异常，静默返回部分文本**（客户端以为拿到了一个短答案）——只有第三种是真正危险的。
  - 因此旧文档写的「先补发 `message_stop` 再发 `error`」是**不必要的**，用来门控它的 `sent_message_start` 一直是死变量（赋值但从未被读），已删除。`ClaudeProxy._stream_request` 的五个错误出口都发 `event: error`，回归测试 `tests/test_databricks_stream_termination.py::StreamTerminalContractTests` 按「一定有 error 帧」这条不变量锁住
  - 同一条不变量在 Copilot 侧对应 `retry_budget_exhausted` 兜底（见下文循环不变量一节）——两条路径防的是同一种静默故障
- **SSE 终止规范化 (Responses / Chat)**: Copilot 和 Azure 都用 `saw_completion` 跟踪终端事件是否见到。Responses 认三种：`event: response.completed|response.failed|response.incomplete` header **或** `data: {"type": "..."}` payload，两种形态由 `_parse_sse_event_block` 统一识别；Chat 认 `data: [DONE]`。上游 EOF 时 `sent_any_chunk and not saw_completion` 触发 silent truncation 分支，补发 `data: {"type":"response.failed","response":{"error":{"code":"upstream_truncated","message":"..."}}}\n\n`（Responses，无 `[DONE]`）或 `data: {"error":{...}}\n\ndata: [DONE]\n\n`（Chat）。所有错误分支的 SSE bytes 由 `_sse_terminal_error` / `_sse_terminal_from_upstream_detail` 统一产出，确保 Responses 流永远不出现 `data: [DONE]`（Codex/OpenAI JS SDK `serde_json` 会把它报为 `error decoding response body`）
- **truncation 观测**: `stream_truncated_no_completion_total` 是总计数，`stream_truncated_no_completion_by_model` 按 `(model, api_type)` 拆分，`/metrics` 输出 `copilot_stream_truncated_no_completion_by_model_total{model="...",api_type="responses|chat"}`；每次 truncation 日志附带 `connection_id / chunks_yielded / first_event / last_event / input_bytes / has_image` 便于回溯
- **重试守卫**: 已 yield 过任意 chunk（Claude 用 `sent_any_content`，Azure/Copilot 用 `sent_any_chunk`）后，不再切换端点重试，避免客户端看到两段拼接的响应

### 服务启动参数
- `uvicorn.run(..., timeout_keep_alive=600, timeout_graceful_shutdown=30)`，覆盖最长合理的 thinking 响应时间，避免 uvicorn 默认 5s keep-alive 中断下游连接

### Token 用量持久化
- `UsageDataStore` (基类): 缓冲 + 30 秒定时刷盘框架，子类实现存储后端
  - `JsonUsageStore`: JSON 文件后端，目录结构 `{path}/{YYYY}/{MM}/{YYYY-MM-DD}.json`
  - `MysqlUsageStore`: MySQL 8.x 后端，`usage_daily` 表（`date, model` 联合主键），`aiomysql` 连接池
- `create_usage_store(config)`: 工厂函数，根据配置创建对应后端实例
- 内存缓冲 → 30 秒批量刷盘（零请求延迟影响）
- **落盘走 `_save_day_delta(d, delta, cumulative)` 钩子，`delta` 只含本批增量**：
  - 基类默认实现写 `cumulative`（整天绝对覆盖）→ 单写者语义，`JsonUsageStore` 走这条
  - `MysqlUsageStore` override 成 `col = col + VALUES(col)` 增量累加 → **多副本安全**
  - 为什么必须这样：`_today_cache` 是当天累计，多个 Pod 各自从 `_load_day` 同一起点累加、每 30s 各自落盘；若写累计绝对值，两 Pod 互相覆盖，用量/成本静默丢失（回归测试 `tests/test_usage_store.py::test_two_writers_sum_instead_of_clobber`）
  - **JSON 后端永远只能单副本**（文件无法跨进程原子累加）；多副本必须 `usage_storage.type: mysql`
  - 基类 `_save_day` 现在**抛 `NotImplementedError`**（从前是 `pass`）。MySQL 后端只 override delta 钩子、不实现 `_save_day`，所以误调它等于「静默丢弃当天全部用量」，从前只有一行注释挡着 —— 现在会炸（`tests/test_usage_store.py::test_base_save_day_fails_loud_instead_of_dropping_a_whole_day`）
  - **`global_stats.total_errors` 在重启后归零**（`/stats` 会显示 `total_errors: 0`），而 token / requests 会从磁盘恢复。原因见下一条：那个持久化字段结构性恒 0。这是已知的不对称，不是 bug 报错点
  - per-model `errors` 已与 `totals` 对称累计，两个后端往返一致。**注意 `is_error=True` 在生产里没有任何调用点**（`record()` 只在成功路径被调），所以这一列现实中恒为 0；修的是一致性——从前 MySQL 侧恒写字面量 0，而 `totals["errors"]` 又由这些 per-model 列求和重建，于是 MySQL 后端重启即丢、JSON 后端不丢。把死链路接活属于新功能（错误请求算哪个模型、有没有 token 都要先定义），不在本次范围
- 原子写入：JSON 用 temp file + `os.replace()`；MySQL 用 `INSERT ON DUPLICATE KEY UPDATE`（增量形式）
- 服务重启自动恢复当天数据到 `GlobalStats` 及 `ClaudeProxy.today_model_stats`
- `ClaudeProxy.today_model_stats` 缓存当天 per-model 累计（启动时从磁盘恢复 + 运行期 `_record_usage` 累加），跨 0 点自动重置；`/stats` 的 KPI `estimated_total_cost_usd` 与 Anthropic Models 表均以该字段为准，因此重启后 Est. Cost 仍会包含重启前的数据（与 Usage History 来源一致）。端点表格的 per-model `estimated_cost_usd` 保留为本次会话内的负载分布视图
- 历史数据清理: 配置 `retention_days` 自动清理 + `DELETE /stats/history?keep_days=N` 手动清理 + Dashboard UI

### 配置管理
- `load_config()`: 加载 YAML 配置，返回 `(ClaudeProxy, Optional[AzureOpenAIProxy], Optional[CopilotProxy], storage_config)`
- `expand_env_vars()`: 支持 `${VAR_NAME}` 环境变量语法
- `resolve_github_token()`: Copilot 端点的 token 三级回退解析（config > 本项目 cache > copilot-lb cache）
  - **静默凭证替换会告警**：config 里显式写了 `${ENV}` 但该变量未设时，会回退到本地缓存文件。K8s 里那个目录正是 `copilot-cache` secret 的挂载点，所以运维可能以为在用刚 rotate 的 Secret、实际跑的是缓存里的旧 token。回退行为**不变**（提供韧性），但现在会打 WARNING 明说「新值没生效」。嵌入式引用（`ghu_${SUFFIX}`）未解析时会返回**残缺** token（上游必 401），同样告警。优先级链的完整覆盖在 `tests/test_copilot_token_resolution.py`
- 存储配置优先级: `usage_storage` > `usage_data_dir` > 默认 `./usage_data`
- **`LBSettings` dataclass**（模块级 `LB_SETTINGS` 单例）：中心化所有 env vars（含默认值 + 类型），启动时一次性加载。`GET /config/effective`（auth-gated）返回全部字段用于运维 introspection。旧的散点 `os.getenv(...)` 调用点仍在（作为 backing store），LBSettings 是 introspection 层。
  - **双来源的默认值必须一致**，否则不设该 env 时 introspection 就在说谎。`IMG_MAX_TOTAL_PIXELS` 曾漂移（报 200M / 执行 100M），已对齐向执行值；防漂移守卫 `tests/test_databricks_payload_compat.py::LBSettingsTests::test_no_dual_sourced_env_default_drift` 按**运行期实际值**比较全部 4 个双来源 env

### 多租户认证（`auth.api_keys`）
- 兼容两种配置：
  - 旧：`auth.api_key: "single-key"` → 内部注册为 tenant `"default"`
  - 新：`auth.api_keys: {tenant-a: key-a, tenant-b: key-b}` → 每 key 记名到租户
  - 两者可共存（旧 key 依然做 `default`，新 dict 额外注册）
- 中心化查表：`API_KEY_TO_TENANT: Dict[str, str]` 由 `_register_api_keys()` 在 `load_config()` 里构建；`_lookup_tenant(key)` 返 tenant 名或 None
- Handler（`/v1/messages` / `/v1/responses` / `/v1/chat/completions`）拿到 tenant 后：
  1. 写 `request.state.tenant`（middleware 可读）
  2. 调 `_CURRENT_TENANT.set(...)` —— `contextvars.ContextVar` 沿 await 链自动传播
  3. `LatencyHistogram.observe(..., tenant=_CURRENT_TENANT.get())` 把 tenant 变成 histogram label
- 空 key 从来不匹配（proxy.verify_api_key 里 `bool(key) and key == self.api_key`），防止 auth bypass

### OpenTelemetry Tracing（opt-in）
- 通过 `otel_setup.py` 提供，`setup_tracing(app)` 在 `lifespan` 里调用
- 默认关闭：`OTEL_ENABLED=true` 才启用；未装 `opentelemetry-*` packages 时 log WARNING + 继续跑（graceful degrade）
- **依赖在 `requirements-otel.txt`，故意不并入 `requirements.txt` → 默认镜像里没有这些包**。因此在 K8s 里只设 `OTEL_ENABLED=true` **不会生效**（打一条 WARNING 后静默禁用）；容器场景真实动作是把 `requirements-otel.txt` 并进 Dockerfile 重新构建镜像（该文件头部有确切的两行）。`otel_setup.py` 的 warning 文案同时给出 local 与 container 两条修法，不要只按 `pip install` 排查
- 测试的 skip guard 必须检查**真实子模块**（`opentelemetry.sdk.trace` / `.instrumentation.fastapi` / `.instrumentation.httpx` …），不能只 `import opentelemetry` —— 那是 namespace package，装了任意一个 otel 发行包（如 `azure-monitor-opentelemetry` 传递带入的 `opentelemetry-api`）就 import 成功，导致「装了一部分包」的机器跳不过去、直接断言失败
- Env 变量走标准 OTel 契约：`OTEL_SERVICE_NAME`、`OTEL_EXPORTER_OTLP_ENDPOINT`、`OTEL_EXPORTER_OTLP_HEADERS`、`OTEL_RESOURCE_ATTRIBUTES`
- Auto-instruments FastAPI（server span per request）+ httpx（client span per upstream call）
- 排除 `/health*` `/metrics` 减少高基数噪音

### Dashboard 前端（单文件抽出到 `dashboard.html`）
- `dashboard.html`（1286+ 行）作为独立文件；`main.py` 通过 `_load_dashboard_html()` 加载
- **设计 token**: `:root` 定义 CSS 变量。旧的 `--card-bg`/`--tooltip-bg` 与新的 BoardUI 对齐命名 `--text-primary`/`--separator-border`/`--border-focus-ring` 共存。禁止硬编码 rgba
- **半径 scale**：`--radius-md/-xl/-2xl/-3xl` = 12/16/20/24px（4px grid）
- **动效 tokens**：`--motion-hover 150ms`、`--motion-entrance 250ms`、`--ease-out cubic-bezier(0.16, 1, 0.3, 1)`
- **复合排版**：`.text-title-1-medium` 等单类携带 size+weight+line-height+letter-spacing
- **动效尊重**：`@media (prefers-reduced-motion: reduce)` 折起所有动画
- **主题切换**: hero 区 `#themeToggle` 按钮；`initTheme()` 先读 `localStorage['lb-theme']`，未设置则回退到 `prefers-color-scheme`；`setTheme(t)` 写 `data-theme` 属性 + localStorage + 调用 `applyChartTheme()`
- **Chart.js 主题同步**: 创建图表时用 `cssVar()` 读当前 CSS 变量作为初始 tooltip/grid/border 颜色；切主题时 `applyChartTheme()` 更新 `Chart.defaults` 并 walk `charts` 单例字典，刷新每个实例的 `plugins.tooltip/legend.labels/scales.*.grid/ticks/title` 颜色及 doughnut `dataset.borderColor`，然后 `update('none')` 无动画重绘
- **Badge / 输入框强调色**: 使用 `color-mix(in srgb, var(--accent) 12%, transparent)` 让强调色在两主题下自动衰减为合适的背景透明度
- **数据刷新**: `refresh()` 每 5 秒轮询 `/stats`，通过 `tickTo()` 平滑过渡 KPI 数字、`diffEndpoints()` 按 name 增删改端点行，避免 innerHTML 全量重绘闪烁

## API 端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `/v1/messages` | POST | 需要 | Databricks Claude 消息 API（仅 `claude-*` 模型） |
| `/v1/messages/count_tokens` | POST | 不需要 | Token 估算 |
| `/v1/responses` | POST | 需要 | OpenAI Responses API（按模型分流：Copilot 优先 → Azure fallback；`claude-*` 拒绝） |
| `/v1/responses`、`/v1/responses/{tail}` | GET | 不需要 | **501** + `Allow: POST`：Codex 的 background/polling 模式未实现。必须是 501 而不是 404/405 —— 后两者会触发客户端指数重试风暴 |
| `/v1/chat/completions` | POST | 需要 | OpenAI Chat Completions API（按模型分流：Copilot 优先 → Azure fallback；`claude-*` 拒绝） |
| `/v1/models`、`/v1/models/{id}`、`/models`、`/models/{id}` | GET | **可选** | 模型清单（客户端发现用）。第三种鉴权模式：**带了 key 就必须有效（否则 401），完全不带则放行** —— 见 `_verify_optional_models_auth` |
| `/health`、`/health/live` | GET | 不需要 | Liveness probe（仅检查进程） |
| `/health/ready` | GET | 不需要 | Readiness probe（检查依赖就绪；故障返回 503 + issues 数组） |
| `/metrics` | GET | 不需要 | Prometheus 文本格式 metrics（K8s / Azure Monitor 抓取） |
| `/admin/copilot/reload` | POST | 需要 | 运维端点：从源重读所有 Copilot endpoint 的 long-lived token + 强制刷新 session（K8s Secret rotation 后立刻生效） |
| `/admin/copilot/reset-pool` | POST | 需要 | 运维端点：重建共享 httpx.AsyncClient，逐出所有 keepalive/半开连接 |
| `/config/effective` | GET | 需要 | 返回 `LBSettings` 全部字段（32 项 env 生效值），运维 introspection 用 |
| `/stats` | GET | 不需要 | 端点统计（含成本估算、Azure OpenAI、GitHub Copilot） |
| `/stats/history` | GET | 不需要 | 历史用量数据（`?days=7`，含每日成本） |
| `/stats/history` | DELETE | **需要** | 清理历史数据（`?keep_days=30`）—— P0.3 加 auth |
| `/stats/dashboard` | GET | 不需要 | 可视化监控面板（四标签页：Anthropic / Azure / GitHub Copilot / 历史，支持深色/浅色主题切换）；response 头带 CSP + X-Content-Type-Options: nosniff |
| `/reset` | POST | **需要** | 重置内存统计（持久化数据保留）—— P0.3 加 auth |
| `/api/event_logging/batch` | POST | 不需要 | **故意的空 sink**：handler 不接 `Request`、**从不读 body**、无条件返 `{"status":"ok"}`。存在的唯一目的是让 Codex 的遥测 POST 不拿到 404（否则客户端刷错误日志）。**不要给它加鉴权**，也不要让它去解析 body —— 它不落盘、不转发、不计数，所以「无鉴权」在这里不构成暴露面 |

> 这张表是**完整**的路由清单（`/openapi.json`、`/docs`、`/redoc` 除外），由
> `tests/test_api_surface_contract.py` 双向机械核验：表里的每一行必须真实存在，
> 每条真实路由必须在表里，且「认证」列与实际行为一致。加了新路由忘了写文档会红。

## 配置文件

`config.yaml` 结构：
```yaml
load_balancer:
  strategy: least_requests        # 负载均衡策略
  circuit_breaker_threshold: 5    # 熔断器错误阈值
  circuit_breaker_timeout: 60     # 熔断器恢复超时（秒）

auth:
  # 二选一 or 共存：
  api_key: your-key               # 旧配置：单 key（默认租户名 "default"）
  # api_keys:                     # 新配置：多租户（P3.2）
  #   team-a: key-a               # tenant 名 -> api key
  #   team-b: ${TEAM_B_KEY}       # 支持 ${ENV_VAR}
  # /metrics 里 proxy_request_latency_seconds{tenant="..."} 会按 tenant 拆分

endpoints:
  - name: workspace-1
    api_base: https://adb-xxx.azuredatabricks.net/serving-endpoints
    token: dapi_xxx               # Databricks 访问令牌，支持 ${ENV_VAR}
    weight: 1

# 用量持久化 - 简单模式（JSON 文件）
usage_data_dir: ./usage_data

# 用量持久化 - 高级模式（支持 json/mysql + 自动清理）
# usage_storage:
#   type: mysql                   # json 或 mysql
#   host: localhost
#   port: 3306
#   user: root
#   password: ${MYSQL_PASSWORD}
#   database: claude_lb
#   pool_size: 5
#   retention_days: 90            # 自动清理超过 N 天的数据

# Azure OpenAI 配置（可选）
azure_openai:
  load_balancer:                    # 可选，不填则使用全局 load_balancer 配置
    strategy: least_requests
  endpoints:
    - name: eastus-region
      endpoint: https://my-openai-eastus.openai.azure.com  # 完整 Azure 端点 URL
      api_key: ${AZURE_KEY_EASTUS}  # 支持环境变量
      deployments:                  # 该资源上可用的部署列表
        - gpt-4o
        - gpt-5
      weight: 1

# GitHub Copilot 配置（可选）
github_copilot:
  load_balancer:                    # 可选，不填则继承全局 load_balancer
    strategy: least_requests
  endpoints:
    - name: gh-account-1
      # github_token 留空则按优先级从 device-flow 缓存或 copilot-lb 缓存读取
      # github_token: ${GITHUB_COPILOT_TOKEN_1}
      weight: 1
      models: []                    # 空 = 通配；填了就只服务列表内模型
```

## 环境变量

> **提示**：所有 env 在 `LBSettings` dataclass 里定义（`load()` classmethod），运行时通过 `GET /config/effective` 可以看当前 effective 值。

**基础**
- `CONFIG_PATH`: 配置文件路径（默认 `config.yaml`）
- `ANTHROPIC_BASE_URL`: Claude Code 需设为 `http://localhost:8000`
- `ANTHROPIC_API_KEY`: Claude Code 需设为与 `config.yaml` 中 `api_key`（或 `api_keys.*` 中任一 key）一致的值
- `AZURE_KEY_*`: Azure OpenAI API 密钥（按区域配置）
- `GITHUB_COPILOT_TOKEN_*`: GitHub OAuth long-lived token（可选；不设也能从 device-flow 缓存读）
- `MYSQL_PASSWORD`: MySQL 密码（如使用 MySQL 存储后端）

**日志 / 流**
- `LOG_FORMAT`: `json` 或 `text`（默认 text）
- `LOG_LEVEL`: `INFO` / `DEBUG` / `WARNING`（默认 INFO）
- `STREAM_HEARTBEAT_INTERVAL`: SSE 心跳间隔秒（默认 15）

**图片压缩**
- `IMG_ADMISSION_ENABLED`: 图片入 admission gate 总开关（默认 true）
- `IMG_COMPRESS_CONCURRENCY`: 并发 PIL 压缩上限（默认 2）
- `IMG_MAX_COUNT`: 单请求图片张数上限（默认 50）
- `IMG_MAX_TOTAL_PIXELS`: 单请求总像素上限（默认 **1 亿**，≈8 张 4K）。内存充裕（≥2Gi）可放宽到 `200000000`，紧张（<1Gi）调到 `50000000` 并把 `IMG_COMPRESS_CONCURRENCY` 设 `1`

**Copilot / GHCP**
- `COPILOT_EDITOR_VERSION` / `COPILOT_EDITOR_PLUGIN_VERSION` / `COPILOT_USER_AGENT`: 请求头版本，随 VS Code 官方滚动
- `COPILOT_HTML_SOFT_COOLDOWN`: HTML 挑战页软熔断窗口秒（默认 30；`0` 关闭）
- `COPILOT_STRIP_INPUT_ITEM_IDS`: 是否剥掉 Responses 请求的 `input[*].id`（默认 **true**）。GHCP 的 item id 绑在服务端 connection 上，回放已失效的 id 会让会话永久 400；`false` 恢复原样透传，仅排查用
- `COPILOT_STATEFUL_401_NEUTRAL`: 携带 opaque state 的请求收到上游 401 时是否视为 request-scoped（默认 **true** = 不计入 endpoint `consecutive_errors`）。防「一个坏会话把整个账户熔断」；`false` 恢复旧行为（所有 401 都计数），仅回退排查用
- `COPILOT_SESSION_AFFINITY`: 是否把同一会话（Codex 的 `prompt_cache_key`）的每一轮钉在同一个 Copilot 账户上（默认 **true**）。**单账户下是完全的 no-op**；多账户下关掉它意味着同一会话的连续轮次会跨账户，必然触发 401（实测 24/24），只能靠恢复阶梯兜住并丢掉推理链
- `COPILOT_OPAQUE_STATE_RECOVERY`: 上游拒绝 opaque state 后是否走恢复阶梯的 rung 2（删 reasoning item 的 `encrypted_content` 再重试一次，默认 **true**）。`false` 只关这一级，仍会记 `exhausted` 并返回结构化 `orphaned_conversation_state` 错误；rung 0/1（剥 `input[*].id`）由 `COPILOT_STRIP_INPUT_ITEM_IDS` 单独控制
- `COPILOT_HTTP2`: 是否启用 HTTP/2（默认 **true**；`false` 强制 HTTP/1.1）—— 需要 `h2` 包（已在 requirements.txt）
- `COPILOT_POOL_MAX_CONNECTIONS` / `COPILOT_POOL_MAX_KEEPALIVE` / `COPILOT_POOL_KEEPALIVE_EXPIRY`
- `COPILOT_POOL_ACQUIRE_TIMEOUT`: 池获取超时（默认 20；旧为 60）
- `COPILOT_POOL_READ_TIMEOUT`: 读超时（默认无上限；`None`/`0`/`""` = 无上限）
- `COPILOT_REFRESH_INTERVAL` / `COPILOT_REFRESH_THRESHOLD`: 后台 token 刷新扫间隔 / 剩余阈值
- `COPILOT_STREAM_HIGH_WATERMARK` / `COPILOT_STREAM_OVERLOAD_GRACE` / `COPILOT_STREAM_DISCONNECT_GRACE` / `COPILOT_STREAM_MONITOR_INTERVAL`
- `COPILOT_UPSTREAM_PROBE_TIMEOUT` / `COPILOT_UPSTREAM_PROBE_CACHE_TTL`

**Observability (OpenTelemetry, opt-in)**
- `OTEL_ENABLED`: 主开关（默认 `false`）—— 打开需要 `pip install -r requirements-otel.txt`（5 个包）。**容器里默认镜像不含这些包，只设本变量无效**，须把该文件并进 Dockerfile 重建镜像
- `OTEL_SERVICE_NAME`: 服务名（默认 `databricks-claude-lb`）
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP collector 地址（如 `http://otel-col:4318`）；未设则用 `ConsoleSpanExporter`（本地调试）
- `OTEL_EXPORTER_OTLP_HEADERS` / `OTEL_RESOURCE_ATTRIBUTES`: 标准 OTel 变量都自动生效
