# 大图请求 413：现场诊断与本地候选修复

Author: Zeno Ren

本报告记录 2026-10-08 发布前的只读现场诊断、本地修复、回归、Linux/amd64 镜像验证与独立 QA。该阶段**未提交 Git、未上传镜像、未发布生产、未调用真实模型**；观测到线上为 `d676026`。后续 GitHub/AKS 交付以对应发布回执为准，本报告不是生产发布回执。

## 现场结论

北京时间 2026-10-08 15:32:43 的请求 `00e665d5b87ab6deccde9b6c56858611` 命中 LB 图片准入保护。现场日志确认：

- `/v1/responses`，Codex 客户端，`gpt-6-astra`，stream=true。
- 原始请求体 21,445,629 字节，约 20.45 MiB；HTTP 413，`upstream_sends=0`。本次请求尚未发送给上游模型。
- 日志不包含原始图片及完整图片数量/尺寸，因此没有重放原请求，也不能保证它在候选版本下必然通过全部限制。

核对目标为 context `zeno-sg-aks`、namespace `zeno-apps`、Deployment/container `claude-lb`。运行单副本，资源上限 CPU 500m、内存 2Gi；观测时约 60m/394Mi，零重启，接流健康检查正常。该瞬时资源值不是峰值容量证明。

线上 imageID 为 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:23e1a9129939ba017c315eae19c0dda3fd4641b812d1b4fde99cab5801015760`；公网及 Pod 内版本均为 `d67602685626182162c8b38f67435123b7877ba6`、`verification=matched`。线上 `main.py` SHA256 `bc676a2ca3e4979859e2dd682a79cc67f8cf4dceef7c6537c44a048e2299a0cb` 与本轮修改前源码一致。

Ingress 已为 64m；两个 Service 是 `claude-lb` 与 `claude-lb-internal`，selector 均为 `app=claude-lb`。现有显式图片环境变量只有 `IMG_MAX_COUNT=50`，其余使用代码默认值。旧代码在压缩前累计原图像素，超过 100M 直接拒绝，压缩无法执行；因此调大 Ingress 无效。

## 候选行为

| 限制 | 默认值 | 处理阶段 |
|---|---:|---|
| `IMG_MAX_COUNT` | 50 张 | 解码前 |
| `IMG_MAX_SINGLE_PIXELS` | 40M | 单张源图在完整解码前检查 |
| `IMG_MAX_SOURCE_PIXELS` | 400M | 源图累计解码工作量 |
| `IMG_MAX_TOTAL_PIXELS` | 100M | 缩图后再次检查 |
| `IMG_COMPRESS_CONCURRENCY` | 2 个请求 | 逐张处理；实际线程结束后释放槽 |

三个 HTTP 入口共用流程。内嵌图片长边超过 1280px 时等比例缩图，包括编码不足 200KB 的大尺寸 PNG；保留图片节点、顺序及 detail 等字段。必要时允许缩图后的编码略大，以保证像素量下降。缩图有损，文字细节可能下降，响应头标明 `images.compressed`。

默认不删除历史图。仅张数超限时沿用明确的 trim consent；strict 模式仍阻止丢图。单图或源累计超限在完整解码前拒绝；缩图失败且最终超预算时拒绝，不向模型发送超预算图片。Pillow bomb 异常不再被当成未知尺寸跳过。

图片及转换缓冲显式 close。取消、重复取消、AnyIO level cancellation 时仍等待实际压缩线程结束，避免 semaphore 和请求体 lease 先释放、后台线程继续占内存。没有修改推理 POST 重放、账户亲和、用量账本或数据库。

## 验证

- TDD：新增 HTTP 用例在旧代码出现 12 个失败子用例，见 [失败记录](http-before.txt)；实现方另先复现图片/取消/资源释放失败，再修复。
- 最终全量本地回归：**791 passed，741 subtests passed，6 skipped**；跳过的六项需一次性隔离 MySQL。本轮没有存储变更或数据库验收，见 [完整日志](full-regression-final.txt)。
- 最终 Python 3.12/Linux amd64 应用镜像内：**68 个测试通过**，覆盖图片、HTTP、adapter 和配置。只挂载测试与离线测试依赖，未挂载应用源码，网络禁用，见 [镜像日志](image-tests-with-sdk.txt)。首次镜像测试缺测试专用 OpenAI SDK，保留 [失败记录](image-tests-final.txt)；在临时容器离线安装锁定 SDK 后补测通过，没有修改应用依赖或镜像。
- 21 个运行文件/依赖清单 hash 全部与 [最终候选源码清单](candidate-identity-final.json) 相符，见 [镜像读回](image-runtime-hashes.json)。
- 独立 QA 在最终源码上未发现阻断问题：真实合成 108M PNG 的 Responses ASGI＋mock 路径、三个像素边界、51 张图及 allow/strict、线程成功/异常与各类取消均通过。测试没有真实 provider 调用。
- Python 3.14 本地在取消后线程抛错时会额外记录 shield 诊断，但资源所有权断言通过；独立 QA 已在目标镜像 Python 3.12.14 补查两种取消＋异常场景，均无该日志且资源归零，见 [复现脚本](qa-cancellation.py) 和 [目标运行时记录](qa-cancellation.txt)。

最终 `main.py` SHA256：`2dbd73d9db841043b52f9d6b308fbdd511e0488f0d2e829b369d3acf46032a57`。本地候选 image ID 为 `sha256:cc41bf524e7761ee319c7105df321d34b7c3a2b80ad98e42496814645e4fcf23`，**不是 registry manifest digest**。构建诚实标注基于 `d676026` 的未提交修改，不可当作已固定的新发布 SHA。

## 合成资源样本

[复现脚本](benchmark.py) 使用 14 张 3840×2160 图，源累计 116,121,600 像素、JSON 2,059,189 字节。旧生产版本镜像准确返回同一 100M 错误，见 [旧版结果](benchmark-before.json)。

候选均在本地 Docker linux/amd64、0.5 CPU/2Gi 限额、禁用网络下执行：

| 样本 | 结果 | 处理耗时 | 整个进程峰值 RSS |
|---|---|---:|---:|
| 压缩并发 1，单请求 | 保留 14 图；变为 1280×720，共 12,902,400 像素 | 2.902 秒 | 127.96 MiB |
| 默认压缩并发 2，同时两个请求 | 两个请求均保留全部 14 图、通过最终预算 | 10.996 / 10.775 秒 | 203.68 MiB |

单请求 JSON 降至 632,827 字节。详情见 [单请求](benchmark-after.json) 和 [双请求](benchmark-concurrent.json)。这是本地合成样本，宿主为 Apple Silicon，包含 amd64 执行开销；峰值含测试构图，不含生产当前工作负载，不证明最坏情况 RSS、生产延迟或长期 SLA。

## 进入生产前

建议最小发布只更新应用镜像和源码注解，沿用当前 2Gi/500m、单副本、两个 Service 和既有受控发布器，不改数据库、Ingress 或监控。默认并发 2 的样本已验证；如另行调整为 1，应单独审阅环境变量补丁与发布器支持，不能假定只替换镜像会改变显式环境配置。

用户确认发布范围后，仍需固定源码提交与 registry digest、刷新当次现场、私有备份及恢复验证、最小补丁/server dry-run、可信公网基线和发布预算。单副本有维护窗口，必须由既有协调器持有完整恢复责任。验收使用固定的小型合成探针，检查 Messages/Responses/Chat 终态、账本、路由和清理；不自动重放这次原始会话。

原请求图片分布、模型实际可用性、供应商图片/token 限制、真实客户端效果和生产负载尚未验证。未上线前，要立即继续当前会话，可在客户端缩小截图或移除不再需要的历史图片后重发。
