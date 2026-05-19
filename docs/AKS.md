# AKS 部署指南

把 `databricks-claude-lb` 部署到 Azure Kubernetes Service，目标是**单 Pod 长期稳定运行 + GitHub Copilot token 完全自动刷新 + 无需重启 Pod 即可滚动 Secret**。

---

## 整体架构

```
                  ┌─────────────────────────────────────┐
                  │  AKS Pod (claude-lb)                │
   客户端          │  ┌─────────────────┐  ┌──────────┐ │      ┌───────────┐
 (Codex/Claude  ──┼──▶ FastAPI :8000   ├──┤ session  ├─┼──────▶ GitHub    │
  Code/openclaw)  │  │ Anthropic /     │  │ token    │ │      │ Copilot   │
                  │  │ OpenAI routes   │  │ cache    │ │      └───────────┘
                  │  │                 │  │ (mem)    │ │
                  │  │ background      │  └────┬─────┘ │      ┌───────────┐
                  │  │ refresh task    │       │       │      │ Databricks│
                  │  │ (5min scan)     │       │       ├──────▶ x6 region │
                  │  └────────┬────────┘       │       │      └───────────┘
                  │           │ readiness      │       │      ┌───────────┐
                  │           ▼                │       ├──────▶ Azure AI  │
                  │  /health/{live,ready}      │       │      │ x4 region │
                  │  /metrics  (Prometheus)    │       │      └───────────┘
                  └────┬───────────────────────┘       │
                       │                               │
              ┌────────┴────────┐         ┌────────────┴──────────┐
              │ ConfigMap       │         │ Secret claude-lb-     │
              │ claude-lb-config│         │  copilot-cache         │
              │ (config.yaml)   │         │ (long-lived OAuth     │
              └─────────────────┘         │   token JSON file)    │
                       ▲                  └────────────┬──────────┘
              ┌────────┴────────┐                      │
              │ Secret claude-  │       ┌──────────────▼──────────┐
              │  lb-secrets     │       │ kubelet 自动同步         │
              │ (env vars: ADB, │       │ (~1 min 周期)            │
              │  Azure, LB key) │       │ Pod 内 reload_github_   │
              └─────────────────┘       │  token() 自动 pick up   │
                                        └─────────────────────────┘
```

---

## 一、镜像构建与推送

### 1. 本地构建并推到 ACR

```bash
# 登录 ACR
ACR=myacr  # 你的 Azure Container Registry 名字
az acr login --name $ACR

# 构建（多架构推荐：AKS 默认 amd64；如果你用 ARM 节点改 --platform linux/arm64）
docker buildx build --platform linux/amd64 \
  -t $ACR.azurecr.io/databricks-claude-lb:$(git rev-parse --short HEAD) \
  -t $ACR.azurecr.io/databricks-claude-lb:latest \
  --push .
```

### 2. （可选）让 ACR 自动从 git 触发构建

```bash
az acr task create \
  --registry $ACR \
  --name build-claude-lb \
  --image databricks-claude-lb:{{.Run.ID}} \
  --image databricks-claude-lb:latest \
  --context https://github.com/<you>/databricks-claude-lb \
  --branch main --file Dockerfile \
  --git-access-token <PAT>
```

---

## 二、准备 Secret 和 ConfigMap

### 1. 一次性生成 GitHub Copilot long-lived token

**这一步必须在能开浏览器的机器上做（device flow 需要交互授权）**。生成后 token 永不过期（除非用户撤销），可重复用于所有 Pod。

```bash
# 项目目录下
python main.py --copilot-login --endpoint gh-account-1
# 终端会打印 verification URL + user code，去浏览器输入即可
# 完成后 token 落到 ~/.config/databricks-claude-lb/copilot-auth-gh-account-1.json (mode 0600)
```

### 2. 创建命名空间和 Secrets

```bash
NS=claude-lb
kubectl create ns $NS

# Secret 1：所有 ${ENV_VAR} 占位的实际值（API key、Databricks tokens、Azure keys）
kubectl create secret generic claude-lb-secrets -n $NS \
  --from-literal=LB_API_KEY="sk-xxxxx" \
  --from-literal=DATABRICKS_TOKEN_CENTRALUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_EASTUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_EASTUS2="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_NORTHCENTRALUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_SOUTHCENTRALUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_WESTUS="dapi_xxx" \
  --from-literal=AZURE_KEY_EASTUS2="..." \
  --from-literal=AZURE_KEY_SWEDENCENTRAL="..." \
  --from-literal=AZURE_KEY_POLANDCENTRAL="..." \
  --from-literal=AZURE_KEY_SOUTHCENTRALUS="..."

# Secret 2：Copilot OAuth token JSON（被 Pod 内 reload_github_token() 监听重读）
kubectl create secret generic claude-lb-copilot-cache -n $NS \
  --from-file=copilot-auth-gh-account-1.json=$HOME/.config/databricks-claude-lb/copilot-auth-gh-account-1.json
```

### 3. （可选）把 ConfigMap 的镜像引用替换成你的 ACR

`deploy/k8s/kustomization.yaml` 改：
```yaml
images:
  - name: <REGISTRY>/databricks-claude-lb
    newName: myacr.azurecr.io/databricks-claude-lb
    newTag: latest
```

---

## 三、部署

```bash
kubectl apply -k deploy/k8s/ -n claude-lb

# 观察启动
kubectl logs -n claude-lb -l app=claude-lb -f
# 期望日志（JSON 格式）：
#   {"ts":"...","level":"INFO","logger":"main","message":"Loaded Databricks endpoint: ..."}
#   {"ts":"...","level":"INFO","logger":"main","message":"[Copilot] resolved token for 'gh-account-1' from /home/app/.config/..."}
#   {"ts":"...","level":"INFO","logger":"main","message":"[Copilot] session token cached for gh-account-1: base=..., expires in 1800s"}
#   {"ts":"...","level":"INFO","logger":"main","message":"[Copilot] background refresh started: interval=300s, threshold=600s"}
```

### Service 暴露

默认 ClusterIP。对外访问的常见做法：

- **集群内消费者**（Codex CLI 跑在另一个 Pod 里）：直接 `http://claude-lb.claude-lb.svc.cluster.local:8000`
- **VPN / 私有 LB**：`kubectl patch svc claude-lb -n claude-lb -p '{"spec":{"type":"LoadBalancer"}}'`，配合 Azure Internal LB annotation
- **公网 + TLS**：上 Ingress（推荐 nginx-ingress 或 application gateway），配 cert-manager

---

## 四、Token 自动刷新机制（核心）

### Session token（30 min）— 全自动

| 触发点 | 行为 |
|---|---|
| 启动 warmup | 初始化时为每个 endpoint 交换一次 |
| 后台 task | 每 `COPILOT_REFRESH_INTERVAL=300` 秒扫一遍，剩 `COPILOT_REFRESH_THRESHOLD=600` 秒就主动刷新 |
| 请求触发 | 每次请求前检查，剩 ≤60s 就刷新 |
| 上游 401 | 强制刷新一次后重试同一请求 |

→ **任何情况下都不需要人工干预**。

### Long-lived OAuth token — 配合 K8s Secret rotation

**正常场景**：long-lived token 永不过期，无需任何操作。

**异常场景**（你 GitHub 那边主动 revoke 了 token）：

1. 本地重新跑：
   ```bash
   python main.py --copilot-login --endpoint gh-account-1
   ```

2. 替换 K8s Secret：
   ```bash
   kubectl create secret generic claude-lb-copilot-cache -n claude-lb \
     --from-file=copilot-auth-gh-account-1.json=$HOME/.config/databricks-claude-lb/copilot-auth-gh-account-1.json \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

3. **零重启生效**（推荐 — 但有 1 min 延迟）：等 kubelet 自动把新 Secret 同步到 mounted file，下次后台刷新（≤5min）会自然 pick up。

3'. **立刻生效**：
   ```bash
   # 调 admin endpoint 强制重读 + 重新交换
   curl -X POST http://<svc>:8000/admin/copilot/reload \
     -H "Authorization: Bearer $LB_API_KEY"
   # 返回 JSON 显示每个 endpoint 是否成功 reload
   ```

3''. **保底**：如果上述都不行（比如 Secret 同步异常），最后手段：
   ```bash
   kubectl rollout restart deployment/claude-lb -n claude-lb
   ```

---

## 五、监控告警

### Prometheus 抓取（已 annotated）

`/metrics` 暴露：

| Metric | 告警建议 |
|---|---|
| `copilot_session_token_remaining_seconds` | < 120 持续 5 min → P2 告警（后台刷新失效） |
| `copilot_token_refresh_failed_total` | rate 5min > 0 → P2 告警 |
| `copilot_endpoint_circuit_open` | == 1 → P1 告警（token 失效，需重新登录 + rotate Secret） |
| `databricks_endpoint_circuit_open` | == 1 持续 5 min → P2 告警（单 endpoint 不影响整体） |
| `copilot_endpoint_errors_total` | rate 5min > 0.1 req/s → P3 监控 |

### Azure Monitor managed Prometheus

AKS 启用 [Azure Monitor managed Prometheus](https://learn.microsoft.com/azure/azure-monitor/essentials/prometheus-metrics-overview)：
```bash
az aks update -n <cluster> -g <rg> --enable-azure-monitor-metrics
```

`prometheus.io/scrape=true` 注解已在 deployment.yaml 配好，自动被采集。

### 日志（JSON 格式 → Log Analytics）

```kusto
ContainerLogV2
| where PodName startswith "claude-lb-"
| extend p = parse_json(LogMessage)
| where p.level == "WARNING" or p.level == "ERROR"
| where p.message contains "[Copilot]"
| project TimeGenerated, p.level, p.message
```

---

## 六、容量规划

| 资源 | requests | limits | 说明 |
|---|---|---|---|
| CPU | 200m | 2000m | 主要消耗在 SSE 流式转发；峰值低 |
| 内存 | 256Mi | 1Gi | usage_data 缓冲 + httpx 连接池；MySQL 后端可降到 512Mi |
| 网络出口 | — | — | 需访问 `*.databricks.net`、`*.openai.azure.com`、`api.github.com`、`api.githubcopilot.com`、`api.enterprise.githubcopilot.com` |

单 Pod 处理能力：约 50 RPS（瓶颈在上游 token 速率限制）。如需扩容：

- replicas → 多副本时把 `usage_storage` 切到 MySQL（避免 PVC 挂载冲突）
- HPA 基于 `kube_pod_container_resource_requests{resource="cpu"}` 或自定义 metric `copilot_endpoint_active_requests`

---

## 七、故障排查

### Pod 不 ready

```bash
kubectl describe pod -n claude-lb -l app=claude-lb
# 看 Readiness probe 失败原因
curl http://<pod-ip>:8000/health/ready
# 返回 issues 字段定位到底是哪个 provider 没就绪
```

常见 issues：

- `github_copilot: no healthy endpoint` → Copilot token 失效，按"四"重新登录 + rotate
- `databricks: no available endpoints` → 所有 ADB workspace 都熔断，检查 token 是否被旋转
- `azure_openai: no available endpoints` → 同上，4xx 不熔断，所以一般是 5xx 持续

### Copilot 突然 401

```bash
# 看是不是 long-lived token 失效
curl http://<svc>:8000/metrics | grep copilot_token_refresh_failed_total
# 上升 → 走"四"的 rotation 流程
```

### usage 数据丢失

PVC 没挂载或 storageClassName 错误：
```bash
kubectl get pvc -n claude-lb
# Status 应为 Bound；Pending → 检查 storageClassName 是否存在
```

切到 MySQL 后端：编辑 ConfigMap，`usage_storage.type=mysql` + 加连接配置；同时 requirements.txt 加 `aiomysql>=0.2.0` 重新构建镜像。

---

## 八、运维 Cheat Sheet

```bash
# 看实时日志
kubectl logs -n claude-lb -l app=claude-lb -f

# 进容器调试
kubectl exec -it -n claude-lb deploy/claude-lb -- /bin/bash

# 看当前 token 剩余时间（在 Pod 外，靠 metrics）
kubectl run -i --rm --restart=Never -n claude-lb tmp-curl --image=curlimages/curl -- \
  -s http://claude-lb:8000/metrics | grep copilot_session_token_remaining_seconds

# 强制 reload Copilot token
TOKEN=$(kubectl get secret -n claude-lb claude-lb-secrets -o jsonpath='{.data.LB_API_KEY}' | base64 -d)
kubectl run -i --rm --restart=Never -n claude-lb tmp-curl --image=curlimages/curl -- \
  -X POST -H "Authorization: Bearer $TOKEN" http://claude-lb:8000/admin/copilot/reload

# 查 dashboard（端口转发到本地）
kubectl port-forward -n claude-lb svc/claude-lb 8000:8000
# 浏览器开 http://localhost:8000/stats/dashboard
```
