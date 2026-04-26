# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Databricks Claude Load Balancer - 一个智能负载均衡代理，支持：
- **Databricks Claude**: 将 Claude API 请求分发到多个 Databricks workspace 端点（`/anthropic/v1/messages`）
- **Azure OpenAI**: 将 OpenAI API 请求分发到多个 Azure OpenAI 区域端点（可选，支持 Responses API 和 Chat Completions API）
- **用量持久化**: 按天存储 token 用量，支持 JSON 文件或 MySQL 8.x 后端
- **成本追踪**: 内置模型定价，自动计算使用成本

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

### 负载均衡
- `WorkspaceEndpoint` / `AzureOpenAIEndpoint`: 端点数据类
- `GlobalStats`: 全局统计数据类
- `LoadBalancer`: 支持 `least_requests` / `round_robin` / `random` 三种策略
- 熔断器机制: 错误达阈值自动禁用端点，超时后自动恢复；429 也触发熔断，4xx 客户端错误不触发

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
- `load_config()`: 加载 YAML 配置，返回 `(ClaudeProxy, Optional[AzureOpenAIProxy], storage_config)`
- `expand_env_vars()`: 支持 `${VAR_NAME}` 环境变量语法
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
| `/v1/messages` | POST | 需要 | Databricks Claude 消息 API |
| `/v1/messages/count_tokens` | POST | 不需要 | Token 估算 |
| `/v1/responses` | POST | 需要 | Azure OpenAI Responses API（需配置 azure_openai） |
| `/v1/chat/completions` | POST | 需要 | Azure OpenAI Chat Completions API（需配置 azure_openai） |
| `/health` | GET | 不需要 | 健康检查 |
| `/stats` | GET | 不需要 | 端点统计（含成本估算、Azure OpenAI） |
| `/stats/history` | GET | 不需要 | 历史用量数据（`?days=7`，含每日成本） |
| `/stats/history` | DELETE | 不需要 | 清理历史数据（`?keep_days=30`） |
| `/stats/dashboard` | GET | 不需要 | 可视化监控面板（三标签页：实时统计 / Azure / 历史，支持深色/浅色主题切换） |
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
```

## 环境变量

- `CONFIG_PATH`: 配置文件路径（默认 `config.yaml`）
- `ANTHROPIC_BASE_URL`: Claude Code 需设为 `http://localhost:8000`
- `ANTHROPIC_API_KEY`: Claude Code 需设为与 `config.yaml` 中 `api_key` 一致的值
- `AZURE_KEY_*`: Azure OpenAI API 密钥（按区域配置）
- `MYSQL_PASSWORD`: MySQL 密码（如使用 MySQL 存储后端）
