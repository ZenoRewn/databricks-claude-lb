# MAI-Decision-1.1 接入与 AKS 发布回执（2026-11-26）

Author: Zeno Ren

## 结论

`lb.zeno.ink` 现在可以调用 Azure Foundry 的 `Microsoft-Decision-1.1`。新入口是 `POST /v1/decision`，模型对外名为 `MAI-Decision-1.1`，`/v1/models` 和 `/models` 都能查到它。应用源码为 `2aace34`（PR #77），镜像是 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:39683bee3d2e1e7ea02bf66391b761d0e26b23dd5c84dde07daf52f32839b91e`。

上线后的公网实测结果如下：
- 带 choice、score、noul 三类问题的请求返回 200，耗时约 2.3 秒，带 `usage: {input_tokens: 102, output_tokens: 3}`。
- `/stats` 中 `eastus2-azure-ai` 的 `model_stats` 记下了这次请求和估算费用。
- 判定模型走 chat 入口会返回 400 `unsupported_api_for_model`，提示改用 `/v1/decision`。
- 问题类型不合法时返回 400，没有密钥时返回 401。

## 为什么要单开一个入口

这个模型不是生成式的，能用的推理入口只有一个：
- 部署能力声明是 `chatCompletion: false`、`decision: true`。
- 直连 eastus2 时，`/openai/v1/chat/completions` 返回 404，`/openai/v1/responses` 返回 «The requested operation is unsupported»。
- 唯一可用的是 `POST {resource}/providers/microsoft/v1/systemone`：请求体为 `state` 加类型化的 `questions`，响应为 `answers` 加 `usage`，不支持流式。

把它塞进对话类协议会做出一个没人用得对的接口，所以网关按上游原样透传，只把 base URL 换成网关，并把 `model` 改写成部署名。

## 设计要点

- **候选集分开**：Azure 端点新增 `decision_deployments` 配置（格式为 `{对外名: 部署名}`），和 `deployments` 是两张不相交的表。这样 chat 和 responses 的候选集永远选不到判定部署。如果写错了，或者某个名字同时出现在两张表里，网关拒绝启动。
- **不重放**：复用 Azure 现有的重试、熔断和结算路径。执行状态不明时仍然不重放，因为每次执行都要计费。
- **成功的判定标准**：返回 200 但没有 `answers` 也算失败。
- **权限**：受 per-key allowlist 控制，provider 记为 `azure_openai`。
- **计价**：取自 Azure Retail Prices API，eastus2 的「Decision 1.1 Gl 1M Tokens」为每 100 万 token $0.042，输入和输出同价。计价按对外名匹配。

## 上线经过

1. **改 ConfigMap**：先备份，再给 eastus2 加上 `decision_deployments`。用校验 Pod 在新镜像上跑过 `load_config` 后才改生产。旧版本会忽略这个键，所以可以先改配置、后发版。
2. **第一次提交 `lb-decision-2aace34`**：排空超时，自动回滚。原因是旧实例上的在途请求在 60 秒内没有结束，与本次代码无关。
3. **第二次提交 `-r2`**：业务探针失败，自动回滚，维护约 287 秒。原因是探针用的 `databricks-claude-opus-5` 在 northcentralus 工作区被上游拒绝（403「Invalid access token」），旧版本上同样复现，也与本次代码无关。**这个 ADB 凭据问题还没修**。
4. **第三次提交 `-r3`**：把探针换成 `databricks-claude-opus-5-5`，发布成功。
5. **上线后实测失败（401）**：网关托管身份（`b96083d5…`）原来只有 `Cognitive Services OpenAI User` 角色，缺少调用 `/providers/*` 所需的 `Microsoft.CognitiveServices/accounts/AIServices/providers/action` 权限。为它在 eastus2 账户上加了 **`Foundry User`** 角色。中途试过一个只含这一条 data action 的自定义角色，仍然返回 401，已删除。
6. **换域名后返回 200**：加角色后，`cognitiveservices.azure.com` 域名仍然返回 401「Principal does not have access to API/Operation」，而 `services.ai.azure.com` 域名返回 200。我在 Pod 内用同一个 token 直接对比过这两个域名，结果一致。所以把 eastus2 的 `endpoint` 改成 `services.ai.azure.com`，同样先用校验 Pod 跑过，再滚动重启。gpt-5.4 在这个域名上经 Responses 调用也是 200。

## 运维变更（不在 Git 里）

- ConfigMap `claude-lb-config` 改了两处：eastus2 加 `decision_deployments: {MAI-Decision-1.1: Decision-1.1}`；eastus2 的 endpoint 改为 `https://eastus2-azureai-service-507154.services.ai.azure.com`。每次修改前的备份在本地 `.release-private/cm-backup/`。
- 在 eastus2 账户上给托管身份 `b96083d5-ae62-4486-88a0-b8ca2349577f` 加了 `Foundry User` 角色。

## 未验证与遗留

- **`databricks-claude-opus-5` 在 northcentralus 返回 403**：ADB 访问令牌失效或权限不足，与本次发布无关，需要单独处理。
- **`cognitiveservices.azure.com` 域名的权限差异**：为什么同一个 token、同一个角色，换到这个域名就被拒，原因没查清。
- **镜像构建**：本机连不上 `files.pythonhosted.org`，所以改用 `dependencies-offline` 构建；`.wheelhouse` 补了 azure-identity 依赖链上的 8 个 wheel，按 lock 文件的 hash 校验。
- **价格边界**：计价表只收录 Global 部署价；如果以后用 DataZone 部署（每 100 万 token $0.046–0.05），费用会被低估。
