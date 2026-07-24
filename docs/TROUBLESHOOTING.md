# 故障排查 Cheat Sheet

记录使用本项目时常见且不直观的坑，按"症状 → 真因 → 解决"组织。

---

## 1. 客户端报 "unexpected status 503 Service Unavailable: \<!doctype html>...Connection Closed"

### 症状

Codex Desktop / 其他 Mac 客户端调 `http://localhost:8000/v1/responses` 收到 503 + 一段 HTML "Connection Closed" 错误页。日志关键 header：
```
content-type: text/html;charset=utf-8
proxy-connection: close
```

### 真因

**LB 没收到请求**——本地的 HTTP 系统代理（Surge / Charles / ClashX / mitmproxy 等）拦截了 `localhost:8000`。

很多 Rust 写的客户端（Codex Desktop 用 hyper/reqwest）只读 `HTTP_PROXY/HTTPS_PROXY/NO_PROXY` 环境变量，**不读 macOS 系统代理设置里的 ExceptionsList**，所以即使 ExceptionsList 有 `localhost`，请求还是会走系统代理。代理对未配置的目标返回静态 HTML 错误页 → 客户端误以为是上游 503。

诊断：
```bash
scutil --proxy | grep -E "HTTPProxy|HTTPPort"
# HTTPProxy : 127.0.0.1
# HTTPPort  : 6152      ← 有代理在拦
```

### 解决

任选其一：

1. **改 base_url 用 IP**（最快）：把 `~/.codex/config.toml` / 客户端配置里的 `http://localhost:8000` 改成 `http://127.0.0.1:8000`。部分库会优先把 `127.0.0.1` 当做直连。
2. **设 NO_PROXY**：`launchctl setenv NO_PROXY "localhost,127.0.0.1,::1"` + 重启 GUI 应用。
3. **代理软件加直连规则**：Surge 配置里加 `DOMAIN,localhost,DIRECT` 和 `IP-CIDR,127.0.0.1/32,DIRECT,no-resolve`；Charles "Proxy → External Proxy Settings → Bypass" 加 localhost。

### 如何分辨"真上游 HTML"和"代理拦截 HTML"

LB 自己已经在 `_build_upstream_error_detail()` 把上游 HTML 错误页转成 JSON（`code: upstream_html_error`）。如果客户端真的收到 HTML，**100% 不是 LB 经手的响应**——肯定是中间代理拦的。

---

## 2. Claude Code 报 "Request too large (max 32MB). Double press esc to go back"

### 症状

CC 在读取大图、Chrome MCP fullPage 截图时直接拒绝发送，不让你按 Enter。

### 真因

**Claude Code 客户端的本地硬限制**，在请求出 CC 之前就拦了。LB 完全无法干预——请求根本没出 CC。

### 解决

LB 侧已经在 `compress_images_in_payload()` 做了图片自动压缩（>200KB base64 → ≤1280px JPEG@82），但**只能解决"32MB 以下、4MB 以上"的请求**。对于 CC 的 32MB 上限，绕路办法：

- 截图小一点（不要 fullPage / 限制可视区域）
- 切到 **Codex CLI** / **openclaw** 等没有 32MB 客户端拦截的工具
- 把图先本地压缩再粘贴

---

## 3. 413 Payload Too Large — 两种来源要分清

线上 LB 链路是 **Cloudflare → ingress-nginx → claude-lb Pod**，413 可能在两层之一被拒。**先看错误体格式**再决定怎么修：

| 来源 | 响应体 / 头特征 | 修法 |
|---|---|---|
| **ingress-nginx**（最常见） | `Content-Type: text/html`，body 是 `<center>413 Request Entity Too Large</center><center>nginx</center>`；带 `cf-ray` 头 | 改 ingress annotation `proxy-body-size: 64m`（见 3.1） |
| **LB 自身** | `Content-Type: application/json`，body 是 `{"error": {"type": "request_too_large", "message": "Request size (X.XXmb) exceeds Databricks 4MB limit even after image compression..."}}` | 减少图片 / `/clear` 新会话 / 启用 prompt cache（见 3.2） |

判断口诀：**HTML = ingress 拦的，JSON = LB 拦的**。LB 的图片压缩链路只有在请求穿过 ingress 后才有机会跑，所以 ingress 上限设太低（默认 1MB）会让所有大图请求拿到 HTML 错误页 —— LB 完全不知情。

### 3.1 ingress-nginx 默认 1MB 拦截

#### 症状

客户端报：
```
unexpected status 413 Payload Too Large: <html>
<head><title>413 Request Entity Too Large</title></head>
<body><center><h1>413 Request Entity Too Large</h1></center>
<hr><center>nginx</center></body></html>
url: https://lb.example.com/v1/responses, cf-ray: a0d0c0b69fad1a58-SIN
```

#### 真因

ingress-nginx 默认 `client_max_body_size = 1m`。带图、带长上下文的请求只要原始 body > 1MB 就被入口直接拒，**LB 的图片自动压缩根本没运行**。

#### 解决

仓库 `deploy/k8s/ingress.yaml` 已经预置正确 annotation；如果你是已有 ingress：
```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "64m"      # 与 LB MAX_RAW_REQUEST_SIZE 对齐
    nginx.ingress.kubernetes.io/proxy-buffering: "off"      # SSE 流式必备
    nginx.ingress.kubernetes.io/proxy-request-buffering: "off"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
```

```bash
kubectl apply -k deploy/k8s/
kubectl describe ingress -n claude-lb claude-lb | grep proxy-body-size   # 确认 64m
```

如果 annotation 加了还报 413，看集群级 ConfigMap：
```bash
kubectl -n ingress-nginx get cm ingress-nginx-controller -o yaml | grep -E "body-size|proxy-buffering"
```

详见 `docs/AKS.md` 的 "Ingress 配置（重要）" 段。

### 3.2 LB 压缩后仍超 Databricks 4MB

#### 症状

```json
{"error": {"type": "request_too_large", "message": "Request size (X.XXmb) exceeds Databricks 4MB limit even after image compression..."}}
```

#### 真因

LB 自带图片自动压缩（>200KB base64 → ≤1280px JPEG@82）；如果压缩后仍 > 4MB，说明：
- 一次塞了多张大图
- 累积上下文（工具调用历史、长 prompt）已经很大

#### 解决

- 客户端 `/clear` 开新会话
- 减少同时附带的图片数
- 用 prompt cache（`cache_control`）减少重复内容
- 检查 `[image-compress]` 日志确认是否真的压了

#### 软上限保护（防 OOM，v2 新增）

字节数上限（`MAX_RAW_REQUEST_SIZE`）管不住真正的内存杀手：**图片解码后的位图**。一张 4000×3000 图 base64 才 ~2MB，PIL 解码成位图却要 ~48MB；多张高分辨率图同时解码就能打爆 Pod（历史 OOM 真凶）。故在 PIL 解码**之前**加了三层准入保护（全部只读 header peek `img.size`，不触发全量解码，内存开销极小）：

| 保护 | 环境变量 | 默认 | 说明 |
|------|----------|------|------|
| 图片张数上限 | `IMG_MAX_COUNT` | `15` | 单请求图片超此数 → 413 |
| 总像素预算 | `IMG_MAX_TOTAL_PIXELS` | `100000000` | 所有图 w×h 累加超此值 → 413（≈8 张 4K）|
| 压缩并发削峰 | `IMG_COMPRESS_CONCURRENCY` | `2` | 同时解码的图片数，`Semaphore` 限流 |
| 总开关 | `IMG_ADMISSION_ENABLED` | `1` | 设 `0` 关闭准入检查（仅保留压缩）|

触发时客户端拿到 JSON 413（非 ingress 的 HTML），错误体 `type: request_too_large`，`message` 说明是张数还是像素超限。日志前缀 `[image-admission]`。

**调参建议**：Pod 内存充裕（≥2Gi）可放宽 `IMG_MAX_TOTAL_PIXELS` 到 `200000000`；内存紧张（<1Gi）调低到 `50000000` 并把 `IMG_COMPRESS_CONCURRENCY` 设 `1`。改环境变量后重启 Pod 生效,无需改代码。

---

## 4. GHCP 返回 "model X is not accessible via the /chat/completions endpoint"

### 症状

```json
{"detail":{"error":{"code":"unsupported_api_for_model","message":"model \"gpt-5.5\" is not accessible via the /chat/completions endpoint"}}}
```

### 真因

GHCP 部分新模型（gpt-5.5、gpt-5.6-sol、gpt-5.6-luna、gpt-5.6-terra、gpt-5-codex 等）**只允许走 `/responses` API**，不允许走 `/chat/completions`。这是 GHCP 上游的强约束，LB 无法绕过。

### 解决

优先把客户端配置成 `wire_api = "responses"`（Codex 默认就是）。如果客户端只支持 Chat Completions，LB 默认会把 `gpt-5.5,gpt-5-codex,gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra` 的 `/v1/chat/completions` 请求 buffered 转到 `/v1/responses`，再包装回 Chat Completions 响应；这种适配会牺牲首字延迟。可用 `OPENAI_CHAT_TO_RESPONSES_MODELS` 覆盖模型列表，或改用 GHCP 原生接受 Chat Completions 的模型（gpt-4o、gpt-4.1 等）。

如果游戏选择 `gpt-5.5` 后报 `No provider available for model 'gpt-5.5': Copilot configured but failing; Azure not configured`，通常不是客户端 key 错，而是 Copilot 上游拒绝了该模型且没有 Azure fallback。新版 LB 会返回更明确的 `unsupported_model`，并在日志中记录 request id、客户端 header 形态、provider 选择、Copilot endpoint 是否熔断、模型是否命中白名单、session token 是否存在等信息。不要先做 silent fallback；先看同一 request id 的 `[OpenAICompat]`、`[route]`、`[Copilot responses]` 日志，确认上游到底拒绝了哪个模型。

---

## 5. Pod ready 但客户端报 "no healthy endpoint for model X"

### 症状

`/health/ready` 返回 200，但 `/v1/chat/completions` 返回 404 + "No provider available for model X"。

### 真因

Provider 整体活着，但**没有任何 endpoint 在 `deployments`/`models` 列表里包含该 model**。例如 Azure 配的 deployment 是 gpt-5.4，客户端发了 gpt-5.5。

### 解决

- 看 `/build_no_provider_message` 给出的诊断（每个 provider 的具体状态）
- 编辑 `config.yaml` 加 deployment / model；或换 GHCP（其 `models: []` 通配，只要上游支持就能跑）

---

## 6. Copilot endpoint 反复进熔断

### 症状

`copilot_endpoint_circuit_open == 1`；`docker logs` 里看到 401 自愈失败 + endpoint 被剔除。

### 真因

long-lived OAuth token 失效（用户在 GitHub 端 revoke 了或 token 过期）。

### 解决

参考 `docs/AKS.md` 第四节的 token rotation 流程：
```bash
python main.py --copilot-login --endpoint gh-account-1
# 替换 K8s Secret，等 kubelet 同步（≤1min）或调 /admin/copilot/reload 立即生效
```

---

## 7. usage 数据丢失 / Dashboard KPI 重启后归零

### 症状

重启 LB 后 Dashboard 的 KPI Est. Cost / Anthropic Models 表里今日数据没有了。

### 真因

- PVC 没挂载（K8s 场景）
- `usage_storage` 配错（路径不可写 / MySQL 连不上）

### 解决

- 检查 `usage_data/` 目录权限和挂载
- MySQL 后端：`pip install aiomysql`，确认 `usage_storage` 配置 host/user/password
- `/stats/history?days=7` 可手工查历史

---

## 8. 流式响应客户端报 "stream disconnected before completion"

### 症状

SSE 流提前断开，客户端拼接到一半挂了。

### 真因（已修复）

Pod 内 ingress / service mesh 的 idle timeout 与 LB 心跳不同步。LB 现在每 15 秒发一次 SSE keep-alive 注释（`: keep-alive`），覆盖大部分 idle 检测。

### 解决

如果还是遇到（极小概率，超长 thinking + 慢上游），调高 ingress idle timeout 或 LB 心跳间隔；不要在 LB 上方再加短 idle 的反代。

---

## 9. Cloudflare 504 Gateway time-out

### 症状

客户端收到 Cloudflare 生成的 504 JSON / HTML，常见字段包括 `origin_gateway_timeout`、`cloudflare_error: true`、`retry_after: 120`，域名指向 LB 前面的 Cloudflare zone。

### 真因

这是 Cloudflare 到 LB origin 的等待超时，不是 Databricks/Azure/Copilot 直接返回给客户端的业务错误。长 thinking 或上游排队时，如果 LB 在等待上游响应头期间没有向客户端写出任何字节，Cloudflare 会认为 origin 太久没响应并主动返回 504。

### 解决

- 对长请求优先使用 streaming。LB 的 streaming 路径会在等待上游响应头期间也发送 SSE 注释心跳（默认每 15 秒 `: keep-alive\n\n`），响应头返回后的空闲阶段也继续心跳。
- 可用 `STREAM_HEARTBEAT_INTERVAL` 调整心跳间隔；建议保持小于 Cloudflare / ingress / service mesh 的 idle timeout。
- 非 streaming 请求无法在同一个 HTTP 响应里提前写 heartbeat；如果模型处理超过 Cloudflare origin timeout，只能改走 streaming、降低请求复杂度，或调整 Cloudflare/Ingress 超时策略。
