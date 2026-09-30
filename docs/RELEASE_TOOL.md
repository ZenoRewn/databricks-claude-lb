# 集群内发布与恢复工具

Author: Zeno Ren

2026-09-30：应用文件清单扩展了诊断、时序、adapter、能力和 build metadata 模块。使用 `python -m operations.build_identity --build-tag ... --platform linux/amd64` 可从本地 Git/source hash 生成构建参数；镜像构建时校验 manifest，`/config/effective` 读回运行匹配状态。该 helper 只构建本地镜像，不推送 registry 或部署。最终 registry/platform digest 仍在推送后绑定到发布回执；本地 image ID 不替代它。

本工具将计划、执行、恢复和验收放在同一持久状态机中。本轮实现只在本地和一次性 Kind 集群验证；没有安装到 AKS，没有执行生产发布。测试结果与未验证边界见 [实施与验证记录](reviews/2026-09-20-post-upgrade-hardening/IMPLEMENTATION_PROGRESS.md)。

## 支持范围

- v1 支持本项目单应用容器、单副本、MySQL 用量账本的维护发布；不做数据库迁移、数据恢复、凭据更新、LB 扩容或蓝绿切换。
- JSON 存储仍可用于 LB 服务，但发布器拒绝将其视为已经具备逐请求账本验收的后端。
- HPA 管理的目标、已识别的 GitOps 管理、注入 sidecar、版本绑定的 Service selector、selectorless 直连路由均需独立策略，v1 会拒绝进入维护。
- 全部入口由 Kubernetes 资源发现并对照计划。不存在直接 Pod 调用仍需运维确认；Kubernetes API 不能证明所有客户端的网络行为。
- 首次升级的旧实现只接受已核对的 892e397 运行文件指纹。旧版采用临时关闭全部 Service 的兼容路径；新版使用带所有权和执行代次的可恢复暂停。
- 旧版恢复引用必须是不可变的单平台 manifest digest，并与运行 Pod 的 imageID 相符；准备、集群预检和恢复开流前都核验。可变 tag、缺失 imageID、无法直接核对的多平台 index/config digest 均拒绝。工具不自动修改旧引用或猜测 index 到平台的关系。

## 架构与恢复原则

本地 CLI 只负责生成/提交计划及查询或发出取消、恢复请求。两个跨节点协调器中只有 Lease 持有者执行；工作负载另有互斥 Lease。本机关闭或提交连接丢失，不会结束集群内流程。

计划写入不可变 ConfigMap；完整 Deployment、Service 和恢复材料写入不可变 Secret。可变回执记录阶段、操作意图、读回结果及错误。配置和凭据内容不复制到公开回执；工具检查其身份/内容指纹，拒绝覆盖并发变更。

```text
preflight → gating → pausing → draining → stopping → starting
  → verifying_backend → restoring_routes → verifying_business
  → finalizing → succeeded

失败/取消 → recovering → finalizing → rolled_back / cancelled
所有权冲突、无法确认 writer 或恢复失败 → needs_attention
```

每次写入带对象 UID、resourceVersion、发布 ID 和执行代次条件。接管者重新读回现场；API ACK 丢失不等于写入未发生。前一次执行的暂停/恢复命令会被文件 revision 和 epoch 拒绝。

单写者停止使用受控 Pod finalizer 保留容器终止证据，再允许启动替代实例；不通过 force delete 应用 Pod 冒充排空。未调度且已进入删除的 Pod 单独记录“未分配运行节点”的证据。节点失联而没有终止证据时不会启动另一 writer。

默认维护预算为 300 秒，前向切换最多 150 秒、排空最多 60 秒，至少保留 150 秒恢复时间。回滚到已经接过流量的新版之前，同样先暂停、排空并确认退出。预算是自动化恢复目标，不能保证控制平面或节点故障时仍在五分钟内恢复，也不授权强杀长请求。

日志持久化失败会独立尝试安全恢复，并在工作负载留下 recovery-only 标记，防止下一任执行者再次关闭已经恢复的旧路由。恢复动作不能撤销永久 drain，不能覆盖人工接管，也不能按 TTL 无条件打开流量。

## 安装材料

协调器使用独立镜像及 [独立依赖锁](../operations/release/requirements.lock)。构建：

```bash
docker build --platform linux/amd64 -f deploy/release/Dockerfile \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t YOUR_REGISTRY/lb-release-controller:YOUR_VERSION .
```

生成安装清单只输出 JSON，不访问或修改集群：

```bash
python -m operations.release.manifests \
  --namespace YOUR_OPERATIONS_NAMESPACE \
  --target-namespace YOUR_TARGET_NAMESPACE \
  --target YOUR_DEPLOYMENT \
  --image YOUR_REGISTRY/lb-release-controller@sha256:YOUR_DIGEST \
  --configmap YOUR_CONFIGMAP \
  --secret YOUR_REFERENCED_SECRET > coordinator.json
```

`--configmap`、`--secret` 可重复，必须列出实际引用。安装应作为明确的独立变更，核对清单和权限后执行；不能拿本仓库的通用 `deploy/k8s/` 覆盖现网。

协调器默认两副本、跨节点反亲和、滚动更新 surge=0/unavailable=1。两个实例未分别 Ready 时，发布预检不会放行。工具不会降低其他工作负载的资源 requests，也不会自动扩容节点。

RBAC 仅允许目标 Deployment 的读取/补丁、目标命名空间的必要 Pod/Service/EndpointSlice/Ingress/HPA 操作、指定配置与 Secret 的读取，以及运维命名空间的计划/回执/Lease。另需只读 Node get 来检查终止证据。Kubernetes RBAC 无法按标签限制动态 Pod 名称的 exec 权限，建议使用专用目标命名空间；代码工作负载白名单不是 RBAC 隔离的替代品。

生产安装不得启用 `--lab`。该选项仅允许 `lb-lab*` 目标命名空间，用于真实 API 写入后的 ACK 丢失、日志失败和受控中断测试；默认安装清单不启用。

## 准备、审阅与提交

安装 CLI 的独立锁定依赖：

```bash
python -m pip install -r operations/release/requirements.lock
```

计划配置示例中的值必须替换为当次真实目标。源 SHA 必须对应本地可读取的 Git 提交；非 lab 模式从该提交计算运行文件 hash，拒绝不一致的人工 manifest。

```yaml
release_id: your-unique-release-id
namespace: your-target-namespace
deployment: your-deployment
container: your-container
source_revision: YOUR_FULL_40_CHARACTER_GIT_SHA
image: YOUR_REGISTRY/YOUR_IMAGE@sha256:YOUR_64_CHARACTER_DIGEST
direct_pod_access: false
storage_compatibility: additive-compatible
legacy_bootstrap: false
public_urls:
  - https://YOUR_LB_HOST
business_probes:
  - api: messages
    model: YOUR_CONFIGURED_CLAUDE_MODEL
    stream: false
    max_tokens: 64
  - api: responses
    model: YOUR_CONFIGURED_OPENAI_MODEL
    stream: true
    max_tokens: 128
  - api: chat
    model: YOUR_CONFIGURED_CHAT_MODEL
    stream: true
    max_tokens: 64
```

公开探针目标必须属于实际指向目标 Service 的 Ingress host；不向任意配置 URL 发送 LB 凭据，不跟随推理请求重定向。计划限制 1～8 个固定合成探针，每个最多 512 输出 token、合计最多 2048；`max_tokens` 是输出限制，不是供应商费用报价。

```bash
python -m operations.release --context YOUR_CONTEXT --kubeconfig YOUR_KUBECONFIG \
  --controller-namespace YOUR_OPERATIONS_NAMESPACE \
  plan --profile release.yaml --output .release-private/YOUR_RELEASE_ID
```

`plan` 读取现场并执行 API Server dry-run，不修改应用。新目录为 0700，文件为 0600，不覆盖既有计划。审阅 `review.json` 的旧版本、目标 digest、全部入口、固定探针/生命周期设置、预算与业务探针；私有 `snapshot.json` 不进入 Git、聊天或公开 artifact。

预检还会使用不匹配业务 Service 的短期 Pod 验证新旧镜像可拉取及导入、目标运行文件/真实 Dashboard、配置和只读 MySQL schema。检查列宽、collation、索引与引擎，不执行 DDL。验证 Pod 保留目标资源 requests 和调度条件；没有调度余量会停在维护前。

授权范围和具体计划明确后提交：

```bash
python -m operations.release --context YOUR_CONTEXT --kubeconfig YOUR_KUBECONFIG \
  --controller-namespace YOUR_OPERATIONS_NAMESPACE \
  submit --directory .release-private/YOUR_RELEASE_ID
```

重复提交相同 release ID/相同计划只返回原状态；不同内容不能复用 ID。`submit` 不构建或上传镜像，目标必须已经是可拉取的不可变 digest。

## 状态、取消和恢复

```bash
python -m operations.release --context YOUR_CONTEXT --kubeconfig YOUR_KUBECONFIG \
  --controller-namespace YOUR_OPERATIONS_NAMESPACE status --release-id YOUR_RELEASE_ID

python -m operations.release --context YOUR_CONTEXT --kubeconfig YOUR_KUBECONFIG \
  --controller-namespace YOUR_OPERATIONS_NAMESPACE cancel --release-id YOUR_RELEASE_ID
```

取消是发送安全收尾请求，不是杀进程。旧 writer 未停止时恢复它的准入和原路由；已进入切换则按兼容条件回滚。状态为 `needs_attention` 时先核对实际对象、所有权、writer 和数据状态；人工接管后旧执行者不得继续写入。

在冲突已由责任人解决、恢复目标明确后，`recover --release-id YOUR_RELEASE_ID` 为该回执请求新的受控恢复窗口。它不重新发送执行结果不明的业务探针，不恢复旧数据库快照，不绕过 CAS 或节点失联条件。

`succeeded` 要求：目标身份和后端检查、所有路由读回、公网三协议中计划所列的实际终态、逐请求账本关联与 hash、维护清理全部通过。业务 exec ACK 丢失时读取同一 Pod 内的回执，不自动重复推理。

业务标记通过协议文本重组后检查：Messages 的 text delta、Responses 的 output-text delta、Chat 的首个 choice 内容会先拼接，再匹配 `LB_OK`，不能直接搜索原始 SSE。元数据、工具参数、推理内容不作为回答；有标记但缺少成功终态仍失败。Responses 完整文本快照替代 delta，不重复累加。

此次分段误判的本地复现、修复和生产归因边界见 [验收器修复记录](reviews/2026-09-21-probe-sse/REPORT.md)。

每条已解析检查保留 `completed`、`marker_found`、文本长度/hash 和请求 ID，不保留原始回答。业务 exec 非零后，新协调器通过受限的只读 `receipt` 命令取回同一 Pod 的缓存，并在恢复前写入发布 ConfigMap。应用与协调器须使用包含该能力的配套版本；若 Pod 在取回前消失或状态存储不可用，仍可能缺证据，不能宣称已保存完整原始报文。

`business` 完成失败（包括命中失败缓存）仍返回非零；`receipt` 的退出码 0 仅表示读取成功。尚在运行的缓存返回 `pending=true, verified=false`，协调器继续等待；任何退出码都不能替代对 `verified=true` 及完整发布终态的检查。

`rolled_back` 表示旧版本、原路由、公开健康和持久化就绪状态恢复；不会为了回滚验收再次执行可能已完成的模型请求。它不是“新版发布成功”，也不等于对所有供应商能力重新认证。失败预检 Pod 可保留诊断，回执列出其名称；历史计划和私有备份不自动删除。

协调器提供 `/metrics`，包括需要接管的数量、未结束维护的最老年龄和最近扫描时间，并写入 `ReleaseNeedsAttention` Kubernetes Warning Event。该能力不自动接入 OpenClaw 或任何外部消息系统。

## 验证边界

发布状态机、日志失败、ACK 丢失及真实 Kind 场景分别留有回执。Kind 使用合成上游与隔离 MySQL，不能替代 AKS、真实供应商、长期压力、控制平面灾难或独立发布复核。Lease 与持久时限依赖合理同步的节点时钟；API 不可达时停止未经确认的写操作。usage 仍是内存待写队列，没有零 RPO 承诺。
