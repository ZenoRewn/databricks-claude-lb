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
| 图片张数上限 | `IMG_MAX_COUNT` | `50` | 单请求图片超此数 → 413 |
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

## 8. 流式响应客户端报 "stream disconnected before completion" / "stream closed before response.completed"

### 症状

SSE 流提前断开，客户端拼接到一半挂了。Codex Desktop / OpenAI JS SDK / OpenAI Python SDK 有时会具体报 `stream disconnected before completion: stream closed before response.completed`（Responses API 客户端 SDK 自己在 SSE EOF 后没看到 `response.completed` 事件时抛出）。

### 真因链（按发生概率排序）

1. **LB 内部曾经缺 `saw_completion = False` 初始化**（Copilot 侧 `_stream_response`）。上游至少吐了 1 个 chunk 但没有任何 `data: {"type":"response.completed"...}` payload 时，LB 触发 `UnboundLocalError` → `finally` 只关 response、不 yield 任何终止事件 → 客户端观察到 socket 直接断。**已修复**（v2026-09 修复：初始化 + Azure/Copilot 两侧对齐）。
2. **GHCP 上游 SSE 用 `event: response.completed\ndata: {...}` header 形式**发送终止事件，data payload 里 `type` 字段缺失。原先只识别 payload `type` 会把这类流误判为 silent truncation → 客户端看到 `response.failed{code: upstream_truncated}` 而非真正的 `response.completed`。**已修复**（`_parse_sse_event_block` 兼容两种形态）。
3. **httpx `read` timeout 太短**打断长 thinking：`COPILOT_POOL_READ_TIMEOUT` 曾默认 `300s`，Codex `reasoning_effort=high` 常见上游 idle >300s。现在默认 `None`（无限），与 Databricks/Azure 一致；有多层兜底：`_await_with_heartbeat` 在等 headers 时也发 SSE 心跳、`connection_monitor_loop` 高水位 400 + 客户端断开 15s 才强制回收、`Request.is_disconnected` 快速路径。
4. **上游 Ingress / service mesh idle timeout 与心跳不同步**：LB 每 `STREAM_HEARTBEAT_INTERVAL`（默认 15s）发 SSE `: keep-alive`，见 Cloudflare 章节。
5. **Codex 客户端自身 timeout**（与 LB 无关）：检查 `~/.codex/config.toml` 里的 `request_max_retries` / `stream_max_retries` / `stream_idle_timeout_ms`。

### 排查步骤

1. `curl -s :8000/metrics | grep -E 'copilot_stream_(truncated|read_timeout|forced|disconnects|pump_queue)'`
2. 查日志 grep `[Copilot] upstream closed stream on` — 现在带 `connection_id / chunks_yielded / first_event / last_event` 完整上下文，能直接定位是哪个模型的哪种事件断的
3. 若 `copilot_stream_read_timeout_total` >0,说明有人把 `COPILOT_POOL_READ_TIMEOUT` 设成了有限值。恢复成 `None`（或删掉环境变量）
4. 若 `copilot_stream_truncated_no_completion_by_model_total{model="X"}` 集中在某个模型,该模型的 GHCP 端可能在长 reasoning 期间会主动 EOF —— 目前只能降 reasoning_effort 或切换模型

### Copilot 连接池高水位 / active requests 不下降

新版 Copilot streaming 路径在客户端中断后会从 ASGI response 边界关闭 body iterator，取消 upstream pump、执行 `response.aclose()` 并 exactly-once 归还 request slot。`httpx.PoolTimeout` 被归类为 LB 本地容量压力，不再累计 endpoint circuit breaker 错误。

连接 monitor 默认每 5 秒采样：active requests ≥400 持续 30 秒后进入保护状态，只对 owner task 已结束或下游连续确认断开 15 秒的 stream 做强制回收。它不会因为请求运行时间长或 upstream 长时间没有 token 就终止连接，因此 `read=None` 和超长 thinking 仍受支持。

排查时同时看：

- `copilot_stream_connections_active` / `copilot_stream_connection_oldest_seconds`
- `copilot_stream_upstream_idle_max_seconds`（仅诊断）
- `copilot_pool_timeout_total` / `copilot_pool_read_timeout_seconds`
- `copilot_stream_disconnects_detected_total` / `copilot_stream_forced_releases_total`
- `copilot_stream_truncated_no_completion_by_model_total{model="...",api_type="responses|chat"}`
- `copilot_stream_read_timeout_total`（`read=None` 时应该恒为 0）
- `copilot_stream_pump_queue_full_events_total`（backpressure 观察）
- `copilot_endpoint_active_requests` / `copilot_endpoint_circuit_open`

---

## 9. Codex/Copilot 报 "Copilot upstream connect stalled … no new connection returned within Ns"

### 症状

Codex Desktop / CLI 抛：

```
stream disconnected before completion: [req=61153d56...] Copilot upstream connect stalled for gh-account-1 (no new connection returned within 20.0s); probe.ok=... dns_ms=... tcp_ms=... ips=...
```

这段字符串是 LB 自己产出的（`_describe_pool_timeout`）。触发链：`httpx.PoolTimeout` 被抛出 → LB 判断本地 `sum(active_requests) < POOL_MAX_CONNECTIONS` → 归类为 `upstream_connect_stalled`（本地池还有余量、httpx 却 acquire 不到新连接）。

### 真因（按优先级排查）

从 20.0s 起，LB 附带的 DNS+TCP 探针 + httpx 内部池快照直接告诉你是哪一层出问题：

| `probe` 字段 | 含义 | 常见根因 |
|---|---|---|
| `probe.ok=true` + httpx `active/total` 都 <500 | 探针正常，本地池也不满 → 上游对本地池"新建连接"这一步慢，可能是 GHCP 服务端限流 / 排队 | 现象常见于短时上游波动；20s 快速失败让 Codex 层重试更快 |
| `probe.ok=true` + httpx `active` ≈ `total` ≈ 500 | 本地池实际已经饱和，但 LB 自己的 `active_requests` 掉队 —— 说明存在连接泄漏或 keepalive 卡住 | 观察 `copilot_stream_connections_active`、找长时间挂着不释放的 stream；检查 `copilot_stream_pump_queue_full_events_total` |
| `probe.error=dns_timeout>3s` | Pod 里到 upstream host 的 DNS 解析卡住 | AKS CoreDNS / NodeLocalDNS 故障，或 upstream host 换 IP 但 DNS TTL 未过；`kubectl exec` 进 pod `dig api.githubcopilot.com` 复现 |
| `probe.error=tcp_timeout>3s` | DNS 拿到 IP 了但 TCP 到 :443 拒绝 / 超时 | 出口 NAT / firewall 限流；确认 pod egress ACL、Azure NAT gateway 端口耗尽 |
| `probe.error=dns_error: ...` / `tcp_error: ...` | 具体 IO 异常直接打出来 | 按异常类型走 |

### 结构化日志字段

LB 触发这类错误时会发一行结构化日志（`extra.kind=copilot_pool_timeout`），字段包括：

- `request_id / endpoint / attempt / api_type / model` — 谁的哪一次尝试
- `lb_pool_active / lb_pool_max` — LB 侧计数
- `httpx_pool` — `{total, active, idle, closing, requests_waiting}`，直接反映 httpx 内部池状态
- `upstream_probe` — 完整 probe 结果，包括 `resolved_ips`、`dns_ms`、`tcp_ms`
- `per_endpoint` — 每个端点的 `active_requests / circuit_open / total_errors`
- `classification` — `local_pool_saturated` 或 `upstream_connect_stalled`
- `exc` — httpcore 原始异常字符串（多半是空的，凭 classification 判断）

KQL / Loki 直接过：`ContainerLog | where LogEntry contains "copilot_pool_timeout"` 拿 JSON 字段即可。

### 关键设计

- **单端点 + `upstream_connect_stalled` 快速失败**：所有 Copilot endpoint 共享同一个 `httpx.AsyncClient`，"换 endpoint 重试"仍打同一个 pool，等于让用户再等一个 `POOL_ACQUIRE_TIMEOUT`。所以 `len(endpoints) <= 1 and classification == upstream_connect_stalled` 时不重试、直接给客户端 SSE error，让 Codex 自己 retry。
- **本地饱和**（`local_pool_saturated`）**仍然重试**：因为存在"某个 stream 刚好结束正在归还 slot"的概率，且换 endpoint 也可能命中不同的 httpcore origin pool。
- **多端点仍走原退避 + `_select_endpoint(model)` 换端点**流程。

### 相关配置

- `COPILOT_POOL_ACQUIRE_TIMEOUT`（默认 20s，从 60s 下调）：httpx 池 acquire 超时。想恢复旧行为设 `60`。
- `COPILOT_UPSTREAM_PROBE_TIMEOUT`（默认 3s）：DNS + TCP 探针的每一步超时。
- `COPILOT_UPSTREAM_PROBE_CACHE_TTL`（默认 5s）：探针结果缓存 TTL，防止密集失败风暴。
- `COPILOT_POOL_MAX_CONNECTIONS` / `COPILOT_POOL_MAX_KEEPALIVE` / `COPILOT_POOL_KEEPALIVE_EXPIRY`：池上限与 keepalive；调大池上限对"本地饱和"有效，对"upstream stall"无效。

### 相关 metrics

- `copilot_pool_timeout_total` — 所有 PoolTimeout 累计（含两类）
- `copilot_pool_timeout_saturated_total` — 本地池真饱和的次数
- `copilot_pool_timeout_upstream_stall_total` — 上游握手挂的次数（**用户报错这一条**）

---

## 10. Codex APP / Openclaw 报 "provider returned an HTML error page" 或 "Session Send failed"

### 症状

- Mac 上的 Codex APP、Openclaw 客户端通过 `lb.zeno.ink` 反代 GitHub Copilot 一段时间后偶发失败
- 报错文案：`The provider returned an HTML error page instead of an API response. This usually means a CDN or gateway (e.g. Cloudflare) blocked the request.` 或 `⚠️ Session Send failed`
- **VSCode 内置 Copilot Chat 直连 GitHub 一切正常** —— 说明账号 / long-lived token 没被封，问题在 LB 与上游之间

### 真因（按优先级排查）

1. **Cloudflare 挑战页 / bot management 拦截**
   - GHCP 后端由 Cloudflare 承接。同一 long-lived token 在多客户端高并发时，Cloudflare 可能给某次请求返 `text/html` 挑战页（"Just a moment..."）。之前旧版 LB 只在 `status ≥ 400` 时识别 HTML，`status = 200` + HTML 会被 pump 原样透传给客户端 SSE 消费者，客户端 SDK 自己解析失败报 "HTML error page"。
   - **本次修复**：LB 在收到 upstream response headers 后立刻检查 `Content-Type`。凡是 `text/html*`（无论 status 是 200/403/502），一律走 `upstream_html_error` 规范化路径 + 触发 30s 软熔断（跳过该 endpoint 直到自动脱敏）+ 主动关闭底层 stream 避免同 socket 复用踩同一 challenge。
2. **Editor-Version / User-Agent 版本陈旧**
   - 老版本 LB 写死 `vscode/1.95.3` + `copilot-chat/0.22.4`（2024-11 版本）。Cloudflare 会根据客户端指纹判断是否 legacy client，命中概率随时间上升。
   - **本次修复**：默认升级到 `vscode/1.104.0` + `copilot-chat/0.30.0`。三个 env 变量 `COPILOT_EDITOR_VERSION` / `COPILOT_EDITOR_PLUGIN_VERSION` / `COPILOT_USER_AGENT` 允许运维随官方版本自行滚动。
3. **业务请求缺少 Accept / Accept-Encoding / Accept-Language**
   - 真实 VS Code Copilot Chat 扩展会带这些字段，缺失即被 CDN 判为客户端指纹异常。
   - **本次修复**：`_build_headers` 现在按 stream/non-stream 分别附上 `Accept: text/event-stream` 或 `Accept: application/json`，同时补 `Accept-Encoding: gzip, deflate` 与 `Accept-Language: en-US,en;q=0.9`。
4. **本地代理拦截**（Mac 特有；详见 §1）
   - 如果客户端配置成 `http://localhost:8000` 而 Mac 系统代理正在拦截 localhost，客户端拿到的 HTML 根本不是 LB 送的。判据：**LB 已经把 HTML 兜底转 JSON `upstream_html_error`；客户端看到 raw HTML → 100% 是中间代理干的**。诊断命令见 §1。

### 结构化日志字段

- `kind=copilot_upstream_html`：LB 检测到 HTML 时打的每一行都带这个标签。附带字段：
  - `endpoint`：命中 CDN 拦截的 Copilot endpoint
  - `api_type`：`chat` 或 `responses`
  - `upstream_status`：上游 HTTP status（200 / 403 / 502 等）
  - `html_soft_cooldown_seconds`：本次软熔断窗口秒数
  - `upstream_ids`：字典，包含上游返回的 `cf-ray` / `x-github-request-id` / `x-request-id` / `server` / `x-served-by` / `x-cache`（拿这些去找 GitHub Support 反馈最有效）
- KQL 示例（Azure Log Analytics）：`AppTraces | where LogLevel == "Warning" and Properties.kind == "copilot_upstream_html" | project TimeGenerated, Properties.endpoint, Properties.upstream_status, Properties.upstream_ids`

### 相关 metrics

- `copilot_upstream_html_events_total{endpoint="..."}`：per-endpoint HTML 事件累计
- `copilot_upstream_html_events_all_total`：全局聚合，方便告警
- `copilot_html_soft_cooldown_active{endpoint="..."}`：0/1，当前是否在冷却窗口
- `copilot_html_soft_cooldown_remaining_seconds{endpoint="..."}`：剩余秒数

告警建议：`increase(copilot_upstream_html_events_all_total[10m]) > 5` 触发告警时先看 log 里 `upstream_ids.cf-ray`，能直接提交给 GitHub Support。

### 相关配置

- `COPILOT_HTML_SOFT_COOLDOWN=30`（默认，秒；0 = 关闭软熔断）
- `COPILOT_EDITOR_VERSION=vscode/1.104.0`（可自行滚动到当前 stable）
- `COPILOT_EDITOR_PLUGIN_VERSION=copilot-chat/0.30.0`
- `COPILOT_USER_AGENT=GitHubCopilotChat/0.30.0`

### 关键设计

- HTML 事件不清 `total_errors` 也不置 `circuit_open`，只用独立的软熔断窗口。这样 CDN 抖动不会连带触发硬熔断（60s+ 全禁用），也不打乱现有 error rate 面板。
- 单 endpoint 场景全部 endpoint 都进冷却时，`_select_endpoint` 会退回到"最小活跃"选一个而不是拒绝服务；下一次请求如果又踩 HTML 会自动刷新窗口，形成"重试式退避"。

---

## 11. Codex 报 401 且 Responses 请求带 `previous_response_id` 或 `encrypted_content`

### 症状

- 多 Copilot endpoint 池下偶发单次 401
- 出错的请求 body 里能看到 `previous_response_id` 非空，或 `input[*].content[*].encrypted_content` 存在
- 同一账户重发**无状态**探测（不带这些字段）→ 恢复 200

### 真因

Handoff §7.2 实证：同一 Responses opaque reasoning state 只能被生成它的账户/会话解密。LB 在 5xx / PoolTimeout / 网络错误 / HTML cooldown 时会**自动 failover** 到另一 Copilot endpoint —— 备份 endpoint 上没有那个 session 状态，上游立刻返 401，客户端看不到"其实是我们换账户了"。

### 修复行为（已内建）

请求携带 opaque state → 首次 endpoint 选定后钉住，`_select_endpoint` 之后所有换端点尝试都被拒绝：
- pinned endpoint 仍健康 → 保持在同一账户，即使它触发了 HTML soft cooldown 也不换（避免更严重的 401）
- pinned endpoint 被硬熔断 → 抛 503 `stateful_pinned_endpoint_unavailable`，客户端应重构会话（新的 request 不带 opaque state）而不是等 LB 换账户

### 相关 metrics

- `copilot_stateful_request_pinned_total{reason=...}` —— 拒绝换 endpoint 的次数，按原因（`http_5xx` / `pool_timeout` / `network_error` / `pinned_unavailable`）分。**非零就是修复在生效**。
- `/stats` 里 pool 层聚合展示

### 什么时候例外处理

无状态请求（纯 input，无 previous / encrypted）**不受影响**，仍走正常 endpoint failover。判定逻辑在 `CopilotProxy._request_has_opaque_state()`，Chat Completions 协议不适用一律返 False。

---

## 12. Cloudflare 504 Gateway time-out

### 症状

客户端收到 Cloudflare 生成的 504 JSON / HTML，常见字段包括 `origin_gateway_timeout`、`cloudflare_error: true`、`retry_after: 120`，域名指向 LB 前面的 Cloudflare zone。

### 真因

这是 Cloudflare 到 LB origin 的等待超时，不是 Databricks/Azure/Copilot 直接返回给客户端的业务错误。长 thinking 或上游排队时，如果 LB 在等待上游响应头期间没有向客户端写出任何字节，Cloudflare 会认为 origin 太久没响应并主动返回 504。

### 解决

- 对长请求优先使用 streaming。LB 的 streaming 路径会在等待上游响应头期间也发送 SSE 注释心跳（默认每 15 秒 `: keep-alive\n\n`），响应头返回后的空闲阶段也继续心跳。
- 可用 `STREAM_HEARTBEAT_INTERVAL` 调整心跳间隔；建议保持小于 Cloudflare / ingress / service mesh 的 idle timeout。
- 非 streaming 请求无法在同一个 HTTP 响应里提前写 heartbeat；如果模型处理超过 Cloudflare origin timeout，只能改走 streaming、降低请求复杂度，或调整 Cloudflare/Ingress 超时策略。

---

## 11. Token exchange 被共享 streaming pool 卡住

### 症状与原因

`token exchange network error` / `header build failed` 后异常文本为空，随后 endpoint 熔断；httpx 内部 `total/active` 达到池上限，但业务 `active_requests` 很低。独立连接访问 token endpoint 正常、reset-pool 后恢复，说明不能仅凭业务计数将其判断为 GitHub 鉴权故障。

旧版 `_exchange_token()` 使用推理长流的 `self.client.get()`。httpx 的连接上限跨 origin 共享，即使 token GET 访问 `api.github.com` 而非推理 host，也会等待同一个满池。空文本的 `PoolTimeout` 在 header-build 路径被记为请求失败，可能触发熔断。此处修复的是 **token 控制面与推理池的耦合**；不宣称解决导致 streaming 池堆满的所有泄漏/半开连接问题。

### 修复行为

- 每次 token GET 尝试创建独立短生命周期 `httpx.AsyncClient`，`max_connections=1`、`max_keepalive_connections=0`；buffered GET 读完后立即关闭，异常和取消也经过 async context 清理。不新增常驻池，无需修改 `close()` 或 reset-pool 的生命周期；不会关闭进行中的推理流。
- 保持 token 请求 connect/read/write/pool **各 10 秒**，不继承 streaming 的 `read=None`、池上限或 HTTP/2 开关。这里是 httpx 每阶段/每次 I/O timeout，不是总墙钟 deadline；既有代理环境设置仍生效。
- 每次 `_exchange_token()` 最多 **3 次尝试**，间隔 **0.2、0.4 秒**。仅重试 `ConnectTimeout/ReadTimeout/WriteTimeout/PoolTimeout/ConnectError/ReadError/WriteError/RemoteProtocolError`；仅应用于这个可安全重复的 GET，不重放推理 POST。
- HTTP 状态错误（含 401/403/429/5xx）、解析错误、非白名单异常不进入上述网络重试。**401 仅保留原有自愈链**：从既有来源读到变化的 OAuth token 才再交换一次；没有变化或仍为 401 则熔断。既有业务层换端点/重试逻辑不变，因此一次业务请求可能包含多个有界的 token exchange。
- 缓存、per-endpoint lock、60 秒提前刷新、后台刷新、admin reload 和成功后的 circuit 恢复保持原语义。

### 观测与上线检查

- 新日志 `kind=copilot_token_exchange_error` 含 `endpoint/error_type/attempt/max_attempts/retry`；文本日志同样带异常类型。重试为 WARNING、最终失败为 ERROR；此日志不输出异常原文、请求头、响应体或 token。
- `copilot_token_refresh_failed_total` 按失败交换尝试计数（包括被重试恢复的网络失败），`copilot_token_refresh_total` 按成功交换计数；失败计数增长不一定代表最终刷新失败。取消本身不算失败。stream pool/read-timeout 指标不混入 token client 的异常。
- 无新增依赖、配置或 K8s manifest 变更。按现有流程部署新代码后，关注 token 剩余有效期、refresh 成败计数、circuit 和 readiness；token 刷新不再依赖 reset-pool。仍需独立观察推理池是否饱和，不能用 token 恢复推断长流泄漏已经解决。
- 取舍：每次刷新增加一次短连接/TLS 建连成本；刷新低频且有缓存/锁，优先选易于清理、不会被旧池污染的短生命周期 client，而非引入另一个共享常驻池。

回归测试：`python -m unittest discover -s tests -v`。`test_copilot_token_exchange.py` 仅用合成 token、MockTransport 和 localhost；真实单槽 httpcore 池被未结束的流占满时，先验证共享 GET 触发 PoolTimeout，再验证隔离刷新成功且不关闭原流，无需真实凭据或访问 GitHub。
