# Databricks Claude Load Balancer

一个智能负载均衡代理，支持将 Claude API 请求分发到多个 Databricks workspace 端点，以及将 OpenAI API 请求分发到多个 Azure OpenAI 区域端点。

## 为什么需要这个项目？

- **突破单一 workspace/区域限制**：通过多个端点分散请求，提高整体吞吐量
- **高可用性**：内置熔断器机制，自动检测故障端点并切换到健康端点
- **成本优化**：利用多个 workspace/区域的配额，避免单点瓶颈
- **成本追踪**：内置模型定价，自动计算每日使用成本，支持 JSON/MySQL 持久化
- **统一代理**：Databricks Claude 和 Azure OpenAI 共用一套负载均衡基础设施

## 功能特性

- **负载均衡** - 支持最少请求数 (least_requests)、轮询 (round_robin)、随机 (random) 策略
- **多端点支持** - Databricks workspace 端点 + Azure OpenAI 区域端点
- **熔断器** - 自动检测故障端点并临时禁用，超时后自动恢复
- **API Key 认证** - 支持自定义 API Key 验证
- **流式响应** - 完整支持 SSE 流式输出
- **Extended Thinking** - 支持 Claude Opus/Sonnet 的 adaptive 思考模式
- **自动重试** - 请求失败时自动切换端点重试
- **Azure OpenAI** - 支持 Responses API 和 Chat Completions API，按模型智能路由
- **Prompt Caching** - 自动清理 `cache_control` 额外字段，兼容 Databricks prompt caching
- **用量持久化** - 按天存储 token 用量数据，支持 JSON 文件或 MySQL 8.x 后端
- **成本估算** - 内置 Anthropic 模型定价，Dashboard 实时展示使用成本
- **历史数据** - Dashboard 可视化历史用量趋势，支持自动/手动清理
- **双主题 Dashboard** - 支持深色 / 浅色主题切换，默认跟随系统 `prefers-color-scheme`，选择持久化到 `localStorage`

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/databricks-claude-lb.git
cd databricks-claude-lb
```

### 2. 安装依赖

```bash
pip install -r requirements.txt

# 如需 MySQL 存储后端（可选）
pip install aiomysql
```

### 3. 配置端点

复制示例配置文件：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，填入你的端点信息：

```yaml
load_balancer:
  strategy: least_requests        # 负载均衡策略
  circuit_breaker_threshold: 5    # 熔断器错误阈值
  circuit_breaker_timeout: 60     # 熔断器恢复超时（秒）

auth:
  api_key: your-secret-api-key    # 自定义的 API Key

# Databricks 端点配置
endpoints:
  - name: workspace-1
    api_base: https://adb-xxx.azuredatabricks.net/serving-endpoints
    token: dapi_xxx               # Databricks Personal Access Token
    weight: 1

  - name: workspace-2
    api_base: https://adb-yyy.azuredatabricks.net/serving-endpoints
    token: ${DATABRICKS_TOKEN_2}  # 支持环境变量
    weight: 1

# Token 用量持久化（可选）
# 简单模式：JSON 文件
usage_data_dir: ./usage_data

# 高级模式：支持 JSON 或 MySQL + 自动清理
# usage_storage:
#   type: mysql                   # json 或 mysql
#   host: localhost
#   port: 3306
#   user: root
#   password: ${MYSQL_PASSWORD}
#   database: claude_lb
#   retention_days: 90            # 自动清理超过 N 天的数据

# Azure OpenAI 端点配置（可选）
# azure_openai:
#   endpoints:
#     - name: eastus-region
#       endpoint: https://my-openai-eastus.openai.azure.com
#       api_key: ${AZURE_KEY_EASTUS}
#       deployments: [gpt-4o, gpt-5]
#       weight: 1
```

### 4. 启动服务

```bash
python main.py
```

或使用 uvicorn 带热重载：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务将在 `http://localhost:8000` 启动。

### 5. 配置 Claude Code

设置环境变量后启动 Claude Code：

```bash
export ANTHROPIC_BASE_URL='http://localhost:8000'
export ANTHROPIC_API_KEY='your-secret-api-key'  # 与 config.yaml 中的 api_key 一致

claude
```

## Docker 部署

### 构建镜像

```bash
docker build -t claude-lb .
```

### 运行容器

```bash
docker run -d \
  --name claude-lb \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/usage_data:/app/usage_data \
  claude-lb
```

### Docker Compose（可选）

创建 `docker-compose.yaml`：

```yaml
version: '3.8'
services:
  claude-lb:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./usage_data:/app/usage_data
    restart: unless-stopped
```

运行：

```bash
docker-compose up -d
```

## API 端点

| 端点 | 方法 | 认证 | 描述 |
|------|------|------|------|
| `/v1/messages` | POST | 需要 | Databricks Claude 消息 API |
| `/v1/messages/count_tokens` | POST | 不需要 | Token 计数估算 |
| `/v1/responses` | POST | 需要 | Azure OpenAI Responses API |
| `/v1/chat/completions` | POST | 需要 | Azure OpenAI Chat Completions API |
| `/health` | GET | 不需要 | 健康检查 |
| `/stats` | GET | 不需要 | 端点统计（含成本估算） |
| `/stats/history` | GET | 不需要 | 历史用量数据（`?days=7`） |
| `/stats/history` | DELETE | 不需要 | 清理历史数据（`?keep_days=30`） |
| `/stats/dashboard` | GET | 不需要 | 可视化监控面板 |
| `/reset` | POST | 不需要 | 重置内存统计（持久化数据保留） |

### Dashboard

![Dashboard](pictures/dashboard.png)

Dashboard 包含三个标签页：
- **Anthropic Models** - 实时端点状态、模型统计、成本估算
- **Azure OpenAI Models** - Azure 端点状态和统计
- **Usage History** - 历史用量表格、成本趋势、数据清理操作

**主题切换**：右上角的太阳 / 月亮按钮可在深色与浅色主题间切换。首次打开跟随操作系统 `prefers-color-scheme`；手动切换后写入 `localStorage['lb-theme']`，下次访问自动恢复。图表（Chart.js）会同步更新 legend、tooltip、grid、ticks 颜色，无需刷新页面。

## 支持的模型

### Databricks Claude

| Claude 模型 | Databricks 模型 |
|------------|-----------------|
| claude-*-sonnet-* | databricks-claude-sonnet-4-6（默认），支持显式指定 4-5/4-6 |
| claude-*-opus-* | databricks-claude-opus-4-7（默认），支持显式指定 4-5/4-6/4-7 |
| claude-*-haiku-* | databricks-claude-haiku-4-5 |

### Azure OpenAI

Azure OpenAI 端点无需模型名映射，请求中的 `model` 字段直接对应 Azure 上的 deployment 名称（如 `gpt-4o`, `gpt-5`, `o4-mini`）。代理会自动将请求路由到拥有该 deployment 的端点。

## 请求兼容性处理

代理会自动处理 Claude Code 请求与 Databricks 之间的兼容性差异：

| 处理项 | 说明 |
|--------|------|
| 模型名映射 | `claude-opus-4-7` → `databricks-claude-opus-4-7` |
| Thinking 参数 | Opus 4.6+/Sonnet 4.6+ 使用 `adaptive`；旧模型自动转换为 `enabled` + `budget_tokens` |
| cache_control | 自动 strip `scope` 等额外字段，保留 `{"type": "ephemeral"}` 支持 prompt caching |
| 不支持的字段 | 自动移除 `context_management`、`output_config`、`defer_loading`、`input_examples`、`tool_reference` |

## 用量持久化与成本追踪

### 存储后端

**JSON 文件（默认）**
- 目录结构：`{usage_data_dir}/{YYYY}/{MM}/{YYYY-MM-DD}.json`
- 无需额外依赖

**MySQL 8.x（可选）**
- 需要安装 `aiomysql`：`pip install aiomysql`
- 自动创建 `usage_daily` 表
- 配置 `usage_storage.type: mysql`

### 成本估算

内置 Anthropic 官方定价，自动计算各模型的使用成本：

| 模型系列 | Input $/MTok | Output $/MTok | Cache Write $/MTok | Cache Read $/MTok |
|---------|-------------|---------------|--------------------|--------------------|
| Opus | $5.00 | $25.00 | $6.25 | $0.50 |
| Sonnet | $3.00 | $15.00 | $3.75 | $0.30 |
| Haiku | $1.00 | $5.00 | $1.25 | $0.10 |

### 历史数据清理

- **自动清理**：配置 `retention_days`，启动时和每日自动执行
- **API 清理**：`DELETE /stats/history?keep_days=30`
- **Dashboard**：History 标签页内置清理面板

## 负载均衡策略

| 策略 | 说明 |
|------|------|
| `least_requests` | 选择当前活跃请求数最少的端点（默认，推荐） |
| `round_robin` | 轮询方式分配请求 |
| `random` | 随机选择端点 |

## 熔断器机制

- 当某个端点连续发生 `circuit_breaker_threshold` 次服务端错误时，熔断器开启
- 熔断器开启后，该端点在 `circuit_breaker_timeout` 秒内不会收到新请求
- 超时后自动恢复，错误计数重置
- 客户端错误（4xx，除 429）不触发熔断器
- 可通过 `/reset` 端点手动重置所有熔断器

## 架构说明

```
                                                 ┌─────────────────────┐
                                           ┌────►│ Databricks          │
                                           │     │ Workspace 1         │
┌─────────────┐     ┌──────────────────┐   │     └─────────────────────┘
│             │     │                  │   │     ┌─────────────────────┐
│ Claude Code │────►│  /v1/messages    │───┼────►│ Databricks          │
│   (Claude)  │     │                  │   │     │ Workspace 2         │
└─────────────┘     │  Load Balancer   │   │     └─────────────────────┘
                    │  Proxy           │   │
┌─────────────┐     │  (localhost:8000)│   │     ┌─────────────────────┐
│  OpenAI     │     │                  │   │     │ Azure OpenAI        │
│  compatible │────►│  /v1/responses   │───┼────►│ East US             │
│  client     │     │  /v1/chat/       │   │     └─────────────────────┘
└─────────────┘     │   completions    │   │     ┌─────────────────────┐
                    └──────────────────┘   └────►│ Azure OpenAI        │
                                                 │ West Europe         │
                                                 └─────────────────────┘
```

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `CONFIG_PATH` | 配置文件路径 | `config.yaml` |
| `ANTHROPIC_BASE_URL` | Claude Code 指向代理地址 | - |
| `ANTHROPIC_API_KEY` | 与 config.yaml 中 api_key 一致 | - |
| `AZURE_KEY_*` | Azure OpenAI API 密钥 | - |
| `MYSQL_PASSWORD` | MySQL 密码（如使用 MySQL 存储） | - |

配置文件中的 token/api_key/password 支持环境变量语法：`${ENV_VAR_NAME}`

## 前置条件

- Python 3.10+
- 至少一个启用了 Claude 模型的 Databricks workspace
- Databricks Personal Access Token
- （可选）Azure OpenAI 资源及 API Key
- （可选）MySQL 8.x（如使用 MySQL 存储后端）

## 常见问题

**Q: 为什么三个端点的请求数不均衡？**

A: `least_requests` 策略基于当前活跃请求数分配，而非总请求数。处理速度快的端点会接收更多请求，这是合理的负载均衡行为。

**Q: 熔断器触发后如何恢复？**

A: 等待 `circuit_breaker_timeout` 秒后自动恢复，或调用 `POST /reset` 手动重置。

**Q: 支持哪些 Claude 模型？**

A: 支持 Opus 4.5/4.6/4.7、Sonnet 4.5/4.6、Haiku 4.5。默认映射到各系列最新版本（Opus → 4.7，Sonnet → 4.6）。

**Q: Prompt caching 是否生效？**

A: 是的。代理会保留 `cache_control: {"type": "ephemeral"}`，仅 strip 掉 Databricks 不支持的额外字段（如 `scope`），因此 prompt caching 正常工作。

**Q: 用量数据在服务重启后会丢失吗？**

A: 不会。用量数据按天持久化到 JSON 文件或 MySQL，重启时自动恢复当天数据；Dashboard 的 KPI Est. Cost 和 Anthropic Models 表也基于该恢复值，不会因重启回零。

**Q: 如何切换到 MySQL 存储？**

A: 安装 `pip install aiomysql`，然后在 config.yaml 中配置 `usage_storage.type: mysql` 及数据库连接信息。

## License

MIT
