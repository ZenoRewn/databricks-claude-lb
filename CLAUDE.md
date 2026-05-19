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
  - `claude-*-sonnet-*` → `databricks-claude-sonnet-4-6`（默认），支持显式指定 4-5/4-6
  - `claude-*-opus-*` → `databricks-claude-opus-4-7`（默认），支持显式指定 4-5/4-6/4-7
  - `claude-*-haiku-*` → `databricks-claude-haiku-4-5`
- `get_model_pricing()` / `calculate_cost()`: 按模型名子串匹配定价并计算成本

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
- Thinking 参数自动转换: 新模型（Opus 4.6+、Sonnet 4.6+）使用 `adaptive`（移除多余 `budget_tokens`）；旧模型将 `adaptive` 转为 `enabled` + 自动计算 `budget_tokens`

### Azure OpenAI 代理 (AzureOpenAIProxy)
- `proxy_responses()`: Responses API 代理，URL: `{endpoint}/openai/v1/responses`
- `proxy_chat_completions()`: Chat Completions API 代理，URL: `{endpoint}/openai/deployments/{model}/chat/completions`
- Auth Header: `api-key: {endpoint.api_key}`
- 端点选择通过 `select_endpoint_for_model()` 按模型过滤

### GitHub Copilot 代理 (CopilotProxy)
- `proxy_chat_completions()`: URL `{session_base_url}/chat/completions`，OpenAI 风格
- `proxy_responses()`: URL `{session_base_url}/responses`，OpenAI Responses 风格（GPT-5 系列）
- 必带 headers (`COPILOT_HEADERS` 常量): `Editor-Version: vscode/1.95.3`、`Editor-Plugin-Version: copilot-chat/0.22.4`、`Copilot-Integration-Id: vscode-chat`、`User-Agent: GitHubCopilotChat/0.22.4`、`Openai-Organization: github-copilot`、`Openai-Intent: conversation-edits`、`X-Initiator: user`。模仿 VS Code Copilot Chat 扩展，**这些值写死且必须保留**，否则上游会 401/403
- 视觉请求自动加 `Copilot-Vision-Request: true`（检测 messages.content 中是否含 `image_url`）
- Auth Header: `Authorization: Bearer {short_lived_session_token}`

#### Token 双层模型（生产级自愈，AKS 友好）

- **Long-lived OAuth token**（GitHub 端基本不过期，除非用户撤销）：
  - 来源优先级 = config.yaml `github_token` > 本项目 device-flow 缓存 `~/.config/databricks-claude-lb/copilot-auth-<name>.json` > 兼容 copilot-lb 旧缓存 `~/.config/copilot-lb/auth.json`
  - **运行时按需重读**：`resolve_github_token()` 返回 `(token, source_dict)`，`source_dict` 记录可重读的来源（`env` / `file` / `literal`）。`reload_github_token(endpoint)` 从源重读，配合 K8s Secret rotation：mounted secret 文件被 kubelet 异步同步（约 1 min 周期），**Pod 内自动 pick up，零重启**
- **Short-lived Copilot session token**（约 30 min 过期）：
  - 从 long-lived token 调 `https://api.github.com/copilot_internal/v2/token` 交换
  - **内存缓存** + 过期前 60 s 自动刷新；并发请求由 per-endpoint `asyncio.Lock` 串行化
  - **后台主动刷新 task**（`background_refresh_loop`）：每 `COPILOT_REFRESH_INTERVAL`（默认 300s）秒扫一遍，剩 ≤`COPILOT_REFRESH_THRESHOLD`（默认 600s）就主动刷新；即使长时间无请求也保持新鲜
  - **请求级 401 自愈**：上游 `/chat/completions`、`/responses` 返回 401 → `force=True` 重刷 session token → 同 endpoint 重试一次（`_proxy` 和 `_stream_response` 内都做了）
  - **token-exchange 401 自愈**：`get_session_token()` 收到 401 → `_LongLivedTokenInvalidError` → 调 `reload_github_token()` 从源重读 long-lived token → 再交换一次；仍失败则该 endpoint 进熔断池（`circuit_open=True` + readiness probe 反映）
- Device Flow CLI: `python main.py --copilot-login --endpoint <name>` 使用 VS Code 公开 client_id `Iv1.b507a08c87ecfe98` 走标准 GitHub Device Flow，token 写入 0600 权限文件
- 启动时 `asyncio.create_task(copilot_proxy.warmup())` + 启动 `background_refresh_loop`
- `POST /admin/copilot/reload`（运维端点）：立即从源重读所有 endpoint 的 long-lived token + 强制刷新 session token，配合 K8s Secret rotation 实现"零延迟"生效（不等 kubelet 同步周期）
- `CopilotEndpoint.models` 语义：**空列表 = 通配**（接受所有模型）；填值则只服务列表内模型。和 Azure 的 `deployments` 语义不同（Azure 必须列出可用部署），所以 Copilot 自己实现 `_select_endpoint(model)` 不复用 `LoadBalancer.select_endpoint_for_model()`
- `_UnsupportedModelError`: 上游返回 `model_not_found` / `unsupported_model` / `unknown model` 等关键字时抛出，路由层捕获后 fallback 到 Azure（仅在未发任何 chunk 时有效，由 `sent_any_chunk` 守卫）
- 路由分流入口: `_route_openai_chat()` / `_route_openai_responses()`：先检查 `claude-*` 名拒绝（必须走 `/v1/messages`）；再 try Copilot；如抛 `_UnsupportedModelError` 或 HTTPException 404/503 就 fallback Azure

#### 可观测性 / 探针

- **`/health/live`**：仅检查进程能响应（K8s livenessProbe，避免上游故障导致 Pod 被 kill）
- **`/health/ready`**：检查依赖就绪 — 至少 1 个 ADB endpoint 可用 + 至少 1 个 Copilot endpoint token 有效（K8s readinessProbe）
- **`/metrics`**：Prometheus 文本格式，暴露 `copilot_session_token_remaining_seconds`、`copilot_token_refresh_total`、`copilot_token_refresh_failed_total`、`copilot_token_reload_total`、`copilot_endpoint_circuit_open`、`databricks_endpoint_*`、`azure_openai_endpoint_*` 等
- **JSON 日志**：`LOG_FORMAT=json` 切换；字段 `ts`/`level`/`logger`/`message`，AKS Log Analytics 可直接 KQL 解析；接管 `uvicorn`/`uvicorn.error`/`uvicorn.access`/`httpx` logger 统一格式
- **环境变量控制**：`LOG_FORMAT`（text|json）、`LOG_LEVEL`（INFO 等）、`COPILOT_REFRESH_INTERVAL`（秒）、`COPILOT_REFRESH_THRESHOLD`（秒）

详细 AKS 部署步骤、Token rotation 流程、监控告警建议见 `docs/AKS.md`。常见客户端 / 上游异常排查见 `docs/TROUBLESHOOTING.md`（含 macOS 系统代理拦截 localhost、CC 32MB 限制、ADB 4MB / GHCP 模型 API 约束等）。

### 流式代理健壮性（`_stream_request` / `_stream_response`）
- **httpx 客户端**: `timeout=Timeout(connect=10.0, read=None, write=60.0, pool=30.0)`；`limits=Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=30.0)`；`http2=False`。`read=None` 必要 —— 流式请求不能整体 read 超时，由逐 chunk 节奏决定
- **异常覆盖**: 网络异常 `except` 分支同时捕获 `ConnectTimeout / ReadTimeout / WriteTimeout / ConnectError / RemoteProtocolError / ReadError / WriteError`，上游中途断流（最常见症状即是客户端报 "socket connection closed unexpectedly"）也走熔断 + 切端点重试路径
- **响应清理**: `response = None` 局部变量 + `finally: await response.aclose()`，避免 httpx 连接池堆积半开连接
- **后台 pump + 心跳**: 将 `aiter_bytes()` 放入 `asyncio.create_task(_pump(response))`，主循环 `await asyncio.wait_for(queue.get(), timeout=15.0)`；15 秒无 chunk 则 yield 一次 `: keep-alive\n\n`（SSE 注释，Anthropic SDK 会忽略），刷新中间链路 idle 计时。pump 退出时由 `finally` 分支 `pump_task.cancel()` 回收
- **SSE 终止规范化**: `sent_message_start` 跟踪是否已向客户端发出 `event: message_start`；若在已发送后 upstream 失败，先补发 `event: message_stop\ndata: {"type":"message_stop"}\n\n` 再发 `event: error`，让 Anthropic SDK 得到合法的流终止序列而不是 socket 异常关闭
- **重试守卫**: 已 yield 过 `message_start`（Claude）或任意 chunk（Azure `sent_any_chunk`）后，不再切换端点重试，避免客户端看到两段拼接的响应

### 服务启动参数
- `uvicorn.run(..., timeout_keep_alive=600, timeout_graceful_shutdown=30)`，覆盖最长合理的 thinking 响应时间，避免 uvicorn 默认 5s keep-alive 中断下游连接

### Token 用量持久化
- `UsageDataStore` (基类): 缓冲 + 30 秒定时刷盘框架，子类实现存储后端
  - `JsonUsageStore`: JSON 文件后端，目录结构 `{path}/{YYYY}/{MM}/{YYYY-MM-DD}.json`
  - `MysqlUsageStore`: MySQL 8.x 后端，`usage_daily` 表（`date, model` 联合主键），`aiomysql` 连接池
- `create_usage_store(config)`: 工厂函数，根据配置创建对应后端实例
- 内存缓冲 → 30 秒批量刷盘（零请求延迟影响）
- 原子写入：JSON 用 temp file + `os.replace()`；MySQL 用 `INSERT ON DUPLICATE KEY UPDATE`
- 服务重启自动恢复当天数据到 `GlobalStats` 及 `ClaudeProxy.today_model_stats`
- `ClaudeProxy.today_model_stats` 缓存当天 per-model 累计（启动时从磁盘恢复 + 运行期 `_record_usage` 累加），跨 0 点自动重置；`/stats` 的 KPI `estimated_total_cost_usd` 与 Anthropic Models 表均以该字段为准，因此重启后 Est. Cost 仍会包含重启前的数据（与 Usage History 来源一致）。端点表格的 per-model `estimated_cost_usd` 保留为本次会话内的负载分布视图
- 历史数据清理: 配置 `retention_days` 自动清理 + `DELETE /stats/history?keep_days=N` 手动清理 + Dashboard UI

### 配置管理
- `load_config()`: 加载 YAML 配置，返回 `(ClaudeProxy, Optional[AzureOpenAIProxy], Optional[CopilotProxy], storage_config)`
- `expand_env_vars()`: 支持 `${VAR_NAME}` 环境变量语法
- `resolve_github_token()`: Copilot 端点的 token 三级回退解析（config > 本项目 cache > copilot-lb cache）
- 存储配置优先级: `usage_storage` > `usage_data_dir` > 默认 `./usage_data`

### Dashboard 前端（单文件内嵌）
- Dashboard HTML/CSS/JS 全部内嵌在 `DASHBOARD_HTML` 字符串常量中（`main.py` 内），不引入构建系统
- **设计 token**: `:root` 定义深色主题 CSS 变量（`--card-bg`、`--tooltip-bg`、`--chart-grid`、`--brand-gradient` 等），`:root[data-theme="light"]` 覆盖为浅色；所有 CSS 均通过 `var(--xxx)` 引用，禁止直接硬编码 rgba 色值
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
| `/v1/chat/completions` | POST | 需要 | OpenAI Chat Completions API（按模型分流：Copilot 优先 → Azure fallback；`claude-*` 拒绝） |
| `/health`、`/health/live` | GET | 不需要 | Liveness probe（仅检查进程） |
| `/health/ready` | GET | 不需要 | Readiness probe（检查依赖就绪；故障返回 503 + issues 数组） |
| `/metrics` | GET | 不需要 | Prometheus 文本格式 metrics（K8s / Azure Monitor 抓取） |
| `/admin/copilot/reload` | POST | 需要 | 运维端点：从源重读所有 Copilot endpoint 的 long-lived token + 强制刷新 session（K8s Secret rotation 后立刻生效） |
| `/stats` | GET | 不需要 | 端点统计（含成本估算、Azure OpenAI、GitHub Copilot） |
| `/stats/history` | GET | 不需要 | 历史用量数据（`?days=7`，含每日成本） |
| `/stats/history` | DELETE | 不需要 | 清理历史数据（`?keep_days=30`） |
| `/stats/dashboard` | GET | 不需要 | 可视化监控面板（四标签页：Anthropic / Azure / GitHub Copilot / 历史，支持深色/浅色主题切换） |
| `/reset` | POST | 不需要 | 重置内存统计（持久化数据保留） |

## 配置文件

`config.yaml` 结构：
```yaml
load_balancer:
  strategy: least_requests        # 负载均衡策略
  circuit_breaker_threshold: 5    # 熔断器错误阈值
  circuit_breaker_timeout: 60     # 熔断器恢复超时（秒）

auth:
  api_key: your-key               # 客户端认证密钥

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

- `CONFIG_PATH`: 配置文件路径（默认 `config.yaml`）
- `ANTHROPIC_BASE_URL`: Claude Code 需设为 `http://localhost:8000`
- `ANTHROPIC_API_KEY`: Claude Code 需设为与 `config.yaml` 中 `api_key` 一致的值
- `AZURE_KEY_*`: Azure OpenAI API 密钥（按区域配置）
- `GITHUB_COPILOT_TOKEN_*`: GitHub OAuth long-lived token（可选；不设也能从 device-flow 缓存读）
- `MYSQL_PASSWORD`: MySQL 密码（如使用 MySQL 存储后端）
