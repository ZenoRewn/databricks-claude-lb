# 2026-11-18 GPT-6.2 接入与 AKS 发布 `31b84fc`

Author: Zeno Ren

## 一、价格刷新被整份拒绝（真正的阻塞）

当日官方价格页有两处新内容，解析器按设计 fail-closed，整份快照被拒，生产价目停在 10-26 基线，此后上线的模型（含 GPT-6.2）全是「未知」：

1. Gemini 4 Carbon 的 `until` 脚注追加了促销后价格句（「Starting December 1, 2026, pricing is …」）；
2. 新增 GPT Image 2.5 Flare 图像模型表（`Text input` / `Image output` 等列）。

修正：追加句只在重述同一边界日时接受，且**不采用**其中的促销后价格；图像表只按那组确切的列跳过，其它陌生列仍整体拒绝。内置基线由解析 11-18 快照生成（fixture `tests/fixtures/copilot-pricing-20261118.html`）：新增 7 个模型（`claude-fable-5.2`、`gemini-4-carbon`、`gemini-4-hydrogen`、`gpt-6.2-astra`、`gpt-6.2-sol`、`mai-code-2`、`mai-code-2-flash`），`claude-opus-5`、`gpt-5.6-terra` 已不在官方页，存活条目无改价。

| 模型 | 默认档（≤272K）入/出/缓存读/缓存写 | 长上下文（>272K） |
|---|---|---|
| gpt-6.2-astra | $8 / $40 / $0.40 / $10 | $16 / $60 / $0.80 / $20 |
| gpt-6.2-sol | $2 / $10 / $0.10 / $2.50 | $4 / $15 / $0.20 / $5 |

## 二、运维变更（发布前已在线完成）

- `COPILOT_RESPONSES_ITEM_ID_MODELS` 增加 `gpt-6.2-astra,gpt-6.2-sol`（patch Deployment + 滚动重启）。流式实测出现 `copilot_response_identity`（`identity_mode: stabilize`）。
- 两个 GHCP 账户重新 discover：`ZenoRewn` 33 个模型，只有 `gpt-6.2-sol`；`zenoren-icloud` 42 个，两个都有。`gpt-6.2-astra` 仅 business seat 有，因此 icloud 的 `allowed_models` 加入 `gpt-6.2-astra`（否则无账户可服务）；`gpt-6.2-sol` 由不限定的 ZenoRewn 服务。
- 入口实测：两个模型 `/v1/responses` 均 `completed`，`/chat/completions` 均 400 `unsupported_api_for_model` —— Responses-only，与目录登记一致。

## 三、发布 `lb-gpt62-31b84fc`

- 镜像 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:36db73ad82f4daabbcfe50653150004d9a7457e4c7768ed67b8c2e0d044b8832`，`operations.build_identity` 本地构建。`az acr build` 失败：Dockerfile 的 `RUN --mount` 需要 BuildKit，ACR 经典构建器不支持。
- 阶段 preflight → … → verifying_business → succeeded，维护约 64 秒。
- 7 条业务探针全部通过，新增 `gpt-6.2-astra` / `gpt-6.2-sol`；为满足 2048 token 总上限，`gpt-6-astra` 探针由 512 调到 384。
- `/version` = `31b84fc` / `matched`；价格刷新 `status: ok`、`catalog_origin: refreshed`、34 个模型、`last_success_at` 2026-11-18T02:06Z、`stale: false`。
- `target_env` 携带同一份 16 项名单，发布后仍含两个 6.2 模型。

## 四、未验证边界

- Codex 桌面端对 6.2 是否不再重复显示，未端到端验证；已验证的只是网关对其启用了稳定化。
- `mai-code-2-flash` 已有公开价，但未加入对外目录：此前其 chat 入口实测失败，需另行核验入口。
- `claude-opus-5` 在 Copilot 价目里回到未知（官方页不再列出），历史用量对应行的 GHCP 估算随之为未知。
