# 同一会话反复 `response.failed` 熔断整个账户：修复与 AKS 发布

Author: Zeno Ren

日期：2026-10-22（北京时间）

## 事件

03:32–03:34 UTC，同一个 Codex 会话把完全相同的请求体（4,428,689 B，带图）发了 5 次，`gpt-6-astra` / `zenoren-icloud` 每次都只返回 `response.created → response.failed`，没有 error 详情。网关把这 5 次都按账户失败记账，凑满阈值后 `zenoren-icloud` 被熔断，03:35 两个无关请求（包括 `claude-opus-5-5`）拿到 503，03:41 小请求试探成功后才恢复。同一模型当天成功处理过 4–13 MB 的带图请求，所以单靠请求体积解释不了这次失败。

## 修复（PR #66，合并提交 `09a9abb`）

- 上游接受流后发出 `response.failed`，且请求体指纹在本轮连续失败里已经计过：不再增加 `consecutive_errors`，`total_errors` 照常累计，并打 `lb_repeat_payload_failure`。401、传输错误、5xx 等其他失败计数方式不变。指纹用每个进程随机生成的 key 计算 BLAKE2b，取 12 位十六进制，只放内存，每轮最多 64 个，请求成功后清空。
- `response.failed` 记录 `upstream_error_code` / `upstream_error_type`（只保留安全标识符，不记 message）和 `payload_fingerprint`。
- 返回 413 `image_trim_requires_consent` 时记录 `image_count` / `image_limit`。Codex 客户端已加 `X-LB-Image-Trim: allow` 请求头，按客户端开启图片裁剪；网关的全局策略仍是 `reject`。

## 验证

- 全量回归 1695 passed、6 skipped（需要本地 MySQL）；node UI 测试通过；PR CI 6 项全部通过。
- 首轮回归有 4 个既有测试失败：最初的写法把同一请求体的 401 也去重了。修正为只对 `response.failed` 去重，没有修改那些测试的断言。

## AKS 发布

- 镜像 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:abfba37ba4a66682f65b3b8be2774b7285f4b015aed0e0cfe1c0fd8d41f7e14f`，源码干净。
- 发布编号 `lb-failed-scope-09a9abb`：只换镜像，identity 相关 env 不变，发布控制器无需重建。各阶段一次通过，结果 `succeeded`。
- 发布后：`/version` 为 `09a9abb`、`matched`；新 Pod 重启次数 0；前 10 分钟内 5 个请求全部 completed。

## 未验证边界

- 去重逻辑还没被真实的重复 `response.failed` 触发过。上游错误码也还没有样本，所以还不能决定是否把某类错误直接改成请求级处理。
- 这次修复不会让出问题的请求变成成功，也不会自动重放它。
