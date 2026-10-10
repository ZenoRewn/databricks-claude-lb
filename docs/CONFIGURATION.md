# 渠道、存储与客户端配置

Author: Zeno Ren

从 [完整配置示例](../config.yaml.example) 复制本地 `config.yaml`，只启用实际使用的渠道。`CONFIG_PATH` 可以指定其他路径；`${VARIABLE}` 从服务进程环境展开，容器不会自动继承宿主机全部环境变量。

## 认证与凭据

`auth.api_key` 用于客户端访问 LB，与上游 Databricks、Azure、Copilot 凭据分开。多租户 `auth.api_keys` 的映射语义见配置示例及 [诊断契约](OBSERVABILITY_AND_CONTEXT.md)。不要把真实配置、`.env`、token 缓存或 Kubernetes Secret 放进 Git。

上游凭据可使用环境变量或已有 Secret 挂载。在现有 AKS 中改凭据来源、挂载路径或 `envFrom` 必须核对现场配置，不能只复制示例的一半。见 [AKS 指南](AKS.md)。

## Databricks

每个 `endpoints[]` 配置独立 `name`、`api_base`、`token` 和权重。可用 `models` 限定转换后的原生模型名；没有白名单表示保留通配路由，不是上游能力认证。

Claude 模型使用 `/v1/messages`。在 OpenAI 风格入口传入 Claude 模型会被拒绝，不会隐式换协议或供应商。模型映射和支持范围以代码配置及上游实际可用性为准。

### 模型名解析

`get_databricks_model` 规范化而不猜测版本：定家族（opus/sonnet/haiku/fable）→ 剥离日期戳与 `latest` → 统一 `-_.` 分隔符 → 拼成 `databricks-claude-<家族>-<版本>`。

- **写了版本号就按写的转发**，由上游裁决是否存在。新版本上线当天无需改码即可用；不存在的版本得到上游明确的 `passthrough is not supported for model ...`，不会被静默降级成另一个模型。
- **只有完全没写版本号**才用 `DATABRICKS_FAMILY_DEFAULTS` 里的家族默认值，该默认值是代码里唯一的版本假设。
- 日期戳与 `latest` 之外的尾缀一律保留进模型名（如 `-fast`），避免被吞掉后按另一档价格计费。
- `databricks-` 前缀的名字原样透传，不做任何解析。

历史上这里曾用正则嗅探版本，其分隔符字符类同时匹配 `5.5` 里的第二个点，导致显式请求 Opus 5.5 静默跑在 Opus 5 上；因此不要恢复按版本分支的 if/elif 写法。

解析成功不代表每个端点都有该模型。2026-10-10 实测 Opus/Sonnet/Haiku 的 5.5 在全部端点稳定，而 `claude-fable-5-1` 仅部分 workspace 可用，其余返回 `NOT_FOUND ... not available in your region`，按端点轮询会间歇失败。区域受限模型需要用 `endpoints[].models` 限定到确有该模型的端点。

## Azure OpenAI

```yaml
azure_openai:
  endpoints:
    - name: azure-eastus
      endpoint: https://YOUR-RESOURCE.openai.azure.com
      api_key: ${AZURE_OPENAI_KEY}
      deployments: [YOUR_DEPLOYMENT_NAME]
      weight: 1
```

`deployments` 必须使用资源中已存在的部署名。Copilot 和 Azure 共享 OpenAI 风格入口，Copilot 优先；只有既有兼容性/明确未准入分支才可能转向 Azure。任意 503、读超时、写失败或中途断流都不能被当成安全 fallback 的证明。

## GitHub Copilot

```yaml
github_copilot:
  endpoints:
    - name: gh-account-1
      weight: 1
      models: []
      api_types: [responses, chat]
```

`name` 是稳定且唯一的账户端点标识，关联日志、Dashboard、CLI 和缓存文件。`models: []` 为通配，并不证明每个模型/API 都受上游支持。`api_types` 可限定真实支持的入口协议。

凭据解析顺序为：显式 `github_token`（支持环境变量）、项目 Device Flow 缓存、兼容旧缓存。多账户不得复用同一份解析出的凭据；显式配置来源失效时不能靠隐式换账户恢复会话。

```bash
python main.py --copilot-login --endpoint gh-account-1
```

按终端给出的官方 verification URL 和 user code 完成登录。缓存位于 `~/.config/databricks-claude-lb/copilot-auth-<sanitized-name>.json`，权限 0600。AKS 挂载时文件名须对应端点名，具体 Secret 方式见部署指南。

OAuth 凭据用于交换短期 session token，后者的有效期以上游 `expires_at` 为准。后台刷新和有界 401 修复不能保证凭据永不被撤销或账户额度永不受限。多账户的 opaque state、pinning 和会话亲和约束见 [状态说明](OPAQUE_STATE.md) 与 [可靠性契约](RESILIENCE.md)。

Copilot 实时费用采用 GHCP token 价格表及适用上下文档位；未知价格保留未知。套餐额度、折扣和历史计费模式请核对 GitHub 账单，不从 LB 计数推算剩余额度。完整口径见 [计费说明](COPILOT_PRICING.md)。

## 用量存储

JSON 适合单实例本地使用，目录需对运行用户可写。Docker 镜像以 UID 1000 运行，绑定挂载时核对目录权限。JSON 不支持多个 writer 共享同一文件。

```yaml
usage_storage:
  type: mysql
  host: YOUR_MYSQL_HOST
  port: 3306
  user: YOUR_MYSQL_USER
  password: ${MYSQL_PASSWORD}
  database: claude_lb
  pool_size: 5
  retention_days: 90
```

MySQL 使用 InnoDB 日汇总与批次账本。提交成功后 ACK 丢失可以按批次幂等重投；尚未落盘的内存事件没有 durable outbox。历史金额缺少逐次渠道与上下文信息，不能重算成准确的 Copilot 分档账单。

Dashboard 的数据维护需要有效 LB Key 和明确确认。密钥仅驻留本次交互，操作后清空，不写入 localStorage。超时或响应未知时先查实际历史，不自动重发删除。

## 客户端接入

### Claude Code

```bash
export ANTHROPIC_BASE_URL='http://127.0.0.1:8000'
export ANTHROPIC_API_KEY="$LB_API_KEY"
claude
```

### OpenAI 兼容 SDK

使用 `base_url=http://127.0.0.1:8000/v1`，API Key 为网关密钥。模型必须在已配置渠道上可用。Responses-only 模型应优先走 `/responses`；Chat 适配可能缓冲完整 Responses 结果，不能据此期望相同 TTFT。

### 使用自定义 Responses provider 的客户端

支持自定义 provider 的客户端可使用如下配置形态；配置键以所用客户端版本为准：

```toml
model_provider = "claude-lb"
model = "YOUR_CONFIGURED_MODEL"

[model_providers.claude-lb]
name = "claude-lb"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
env_key = "OPENAI_API_KEY"
```

将 `OPENAI_API_KEY` 设为 LB 密钥。发生代理或 localhost 解析问题时，先按 [排障指南](TROUBLESHOOTING.md) 核对网络路径，避免把客户端中断直接归因于上游。

## 参数与监控

完整默认值以鉴权后的 `/config/effective` 为准。预算、参数严格模式、图片裁剪、渠道能力目录、v2/v3 指标和安全日志分别见 [当前契约](OBSERVABILITY_AND_CONTEXT.md)。未知或过期的渠道能力不自动成为硬限额。
