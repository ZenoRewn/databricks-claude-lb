# Databricks Claude Load Balancer

一个智能负载均衡代理，支持将 Claude API 请求分发到多个 Databricks workspace 端点，以及将 OpenAI API 请求分发到多个 Azure OpenAI 区域端点。

## 为什么需要这个项目？

- **突破单一 workspace/区域限制**：通过多个端点分散请求，提高整体吞吐量
- **高可用性**：内置熔断器机制，自动检测故障端点并切换到健康端点
- **成本优化**：利用多个 workspace/区域的配额，避免单点瓶颈
- **统一代理**：Databricks Claude 和 Azure OpenAI 共用一套负载均衡基础设施

## 功能特性

- **负载均衡** - 支持最少请求数 (least_requests)、轮询 (round_robin)、随机 (random) 策略
- **多端点支持** - Databricks workspace 端点 + Azure OpenAI 区域端点
- **熔断器** - 自动检测故障端点并临时禁用，超时后自动恢复
- **API Key 认证** - 支持自定义 API Key 验证
- **流式响应** - 完整支持 SSE 流式输出
- **Extended Thinking** - 支持 Claude Opus 的思考模式
- **自动重试** - 请求失败时自动切换端点重试
- **Azure OpenAI** - 支持 Responses API 和 Chat Completions API，按模型智能路由

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/databricks-claude-lb.git
cd databricks-claude-lb
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
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

# Azure OpenAI 端点配置（可选）
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
        - o4-mini
      weight: 1
    - name: westeurope-region
      endpoint: https://my-openai-westeu.openai.azure.com
      api_key: ${AZURE_KEY_WESTEU}
      deployments:
        - gpt-4o
        - gpt-5
      weight: 2
```

> **注意**: `azure_openai` 配置段是可选的。不配置则仅启用 Databricks Claude 功能，完全向后兼容。

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
| `/v1/responses` | POST | 需要 | Azure OpenAI Responses API（需配置 azure_openai） |
| `/v1/chat/completions` | POST | 需要 | Azure OpenAI Chat Completions API（需配置 azure_openai） |
| `/health` | GET | 不需要 | 健康检查 |
| `/stats` | GET | 不需要 | 查看所有端点统计信息 |
| `/stats/dashboard`| GET | 不需要 | 查看 Dashboard |
| `/reset` | POST | 不需要 | 重置所有熔断器状态 |

### 查看统计信息

```bash
curl http://localhost:8000/stats
```

返回示例：

```json
{
  "endpoints": [
    {
      "name": "workspace-1",
      "active_requests": 1,
      "total_requests": 100,
      "total_errors": 2,
      "circuit_open": false
    },
    {
      "name": "workspace-2",
      "active_requests": 0,
      "total_requests": 98,
      "total_errors": 1,
      "circuit_open": false
    }
  ]
}
```

### Dashboard查看使用信息
![alt text](pictures/dashboard.png)

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

- **Databricks Claude**: 使用原生 Anthropic 端点 (`/anthropic/v1/messages`)，直接透传请求
- **Azure OpenAI**: 支持 Responses API (`{endpoint}/openai/v1/responses`) 和 Chat Completions API (`{endpoint}/openai/deployments/{model}/chat/completions`)，按模型智能路由到拥有该 deployment 的端点

## 支持的模型

### Databricks Claude

| Claude 模型 | Databricks 模型 |
|------------|-----------------|
| claude-*-sonnet-* | databricks-claude-sonnet-4-5 |
| claude-*-opus-* | databricks-claude-opus-4-6（默认），支持显式指定 4-5/4-6 |
| claude-*-haiku-* | databricks-claude-haiku-4-5 |

### Azure OpenAI

Azure OpenAI 端点无需模型名映射，请求中的 `model` 字段直接对应 Azure 上的 deployment 名称（如 `gpt-4o`, `gpt-5`, `o4-mini`）。代理会自动将请求路由到拥有该 deployment 的端点。

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

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `CONFIG_PATH` | 配置文件路径 | `config.yaml` |
| `AZURE_KEY_*` | Azure OpenAI API 密钥（按区域配置） | - |

配置文件中的 token/api_key 支持环境变量语法：`${ENV_VAR_NAME}`

## 前置条件

- Python 3.10+
- 至少一个启用了 Claude 模型的 Databricks workspace（用于 Claude 代理）
- Databricks Personal Access Token
- （可选）Azure OpenAI 资源及 API Key（用于 OpenAI 代理）

## 获取 Databricks Token

1. 登录 Databricks workspace
2. 点击右上角用户图标 → User Settings
3. 选择 Developer → Access tokens
4. 点击 Generate new token

## 常见问题

**Q: 为什么三个端点的请求数不均衡？**

A: `least_requests` 策略基于当前活跃请求数分配，而非总请求数。处理速度快的端点会接收更多请求，这是合理的负载均衡行为。

**Q: 熔断器触发后如何恢复？**

A: 等待 `circuit_breaker_timeout` 秒后自动恢复，或调用 `POST /reset` 手动重置。

**Q: 支持哪些 Claude 功能？**

A: 支持所有 Databricks Claude 端点支持的功能，包括流式输出、Extended Thinking 等。

**Q: Azure OpenAI 是必须配置的吗？**

A: 不是。`azure_openai` 配置段完全可选，不配置则仅启用 Databricks Claude 功能，与之前版本完全兼容。

**Q: Azure OpenAI 的 Responses API 和 Chat Completions API 有什么区别？**

A: Responses API (`/v1/responses`) 是 OpenAI 推荐的新 API，Chat Completions API (`/v1/chat/completions`) 是传统 API 作为降级路径。两者都支持流式和非流式响应。

**Q: 如何确定请求会路由到哪个 Azure 端点？**

A: 代理会根据请求中的 `model` 字段，从配置的端点中筛选出 `deployments` 列表包含该模型的端点，然后按负载均衡策略选择。如果没有端点支持请求的模型，返回 404 错误。

## License

MIT
