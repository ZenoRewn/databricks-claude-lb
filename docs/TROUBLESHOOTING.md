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
| 压缩并发削峰 | `IMG_COMPRESS_CONCURRENCY` | `2` | **并发压缩的请求数**。单请求内部是逐张顺序解码、每张用完即释放，所以全局位图峰值 ≈ 这个值 |
| 总开关 | `IMG_ADMISSION_ENABLED` | `1` | 设 `0` 关闭准入检查（仅保留压缩）|

触发时客户端拿到 JSON 413（非 ingress 的 HTML），错误体 `type: request_too_large`，`message` 说明是张数还是像素超限。日志前缀 `[image-admission]`。

**调参建议**：Pod 内存充裕（≥2Gi）可放宽 `IMG_MAX_TOTAL_PIXELS` 到 `200000000`；内存紧张（<1Gi）调低到 `50000000` 并把 `IMG_COMPRESS_CONCURRENCY` 设 `1`。改环境变量后重启 Pod 生效,无需改代码。

**别把这两个变量混着理解**（2026-09-10 核实）：真正的内存上界来自「顺序解码 + 请求级并发限流」——
单请求内部逐张解码、每张位图用完即释放，所以全局峰值 ≈ `IMG_COMPRESS_CONCURRENCY` 张位图，**与图片
张数无关**。`IMG_MAX_TOTAL_PIXELS` 约束的是累计解码工作量（延迟/CPU）。内存吃紧优先调
`IMG_COMPRESS_CONCURRENCY`，请求太慢才调像素预算。

**超预算时的降级**：LB 会先 `trim_excess_images` 丢掉最早的图（替换成 `[image omitted for context
length]` 占位），然后**重跑准入**；仍超才 413。重跑是必须的 —— 只按张数 trim 到上限不保证像素预算
也满足（50 张 4K 仍远超）。所以一个「图很多且都很大」的请求现在会拿到 413，而以往会被放行（慢但能跑）。
日志前缀 `[image-trim]` / `[image-admission]`。

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

> ⚠️ **若错误后半句是 `input item does not belong to this connection`，看 [第 14 节](#14-codex-报-input-item-does-not-belong-to-this-connection含-input-item-id)**。那是**两个叠加的问题**：本节的断流是**因**，第 14 节的会话中毒是**果** —— 断流让 Codex 留下半成品 item 并持续回放其 GHCP connection-bound id，于是该会话此后每轮都失败。只按本节查断流会漏掉「为什么新会话好、旧会话永久坏」。

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
stream disconnected before completion: [req=61153d56...] Copilot connection pool acquisition timed out for gh-account-1 (configured wait 20.0s); this does not establish a TCP/TLS failure; probe.ok=... dns_ms=... tcp_ms=... ips=...
```

这段字符串是 LB 自己产出的（`_describe_pool_timeout`），SSE `error.code` = `pool_acquire_timeout`。

> **旧文案已不再出现。** 早期版本报的是 `Copilot upstream connect stalled for ...`，分类分 `upstream_connect_stalled` / `local_pool_saturated` 两种。那套措辞把「等池分配超时」当成了「上游建连失败」的证据，属于无根据的因果断言，已被撤回（迁移记录见 `docs/STREAM_OWNERSHIP.md`「Diagnostic/API compatibility」）。**按旧字符串搜日志会一无所获**，请用上面的新文案或 `kind=copilot_pool_timeout`。

触发链：`httpx.PoolTimeout` 被抛出（等待池分配超过 `POOL_ACQUIRE_TIMEOUT`）→ LB 抓一次 httpx 内部池快照 + DNS/TCP 探针 → 归类为中立的 `pool_acquire_timeout`，并另记一个纯观测布尔 `httpx_pool_observed_full`。**`PoolTimeout` 本身不区分「本地池满」与「上游不给连接」**，事后补做的探针也无法确立当时的因果，所以 LB 只给观测、不给结论。

### 真因（按优先级排查）

LB 附带的 DNS+TCP 探针 + httpx 内部池快照**收窄假设范围**，但都不是当时的因果证据（探针发生在超时之后）：

| `probe` 字段 | 观测到的事实 | 值得优先查的方向 |
|---|---|---|
| `probe.ok=true` + httpx `active/total` 都 <500（`httpx_pool_observed_full=false`） | 事后探针能建连，且池快照没满 —— 池没满却 acquire 不到 | GHCP 服务端限流 / 排队；或本地池有卡住的条目（见下一节的 `/admin/copilot/reset-pool`）。20s 快速失败让 Codex 层重试更快 |
| `probe.ok=true` + httpx `active` ≈ `total` ≈ 500（`httpx_pool_observed_full=true`） | 快照里池确实满了，而 LB 自己的 `active_requests` 掉队 | 连接泄漏或 keepalive 卡住：观察 `copilot_stream_connections_active`、找长时间挂着不释放的 stream；检查 `copilot_stream_pump_queue_full_events_total` |
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
- `classification` — **恒为 `pool_acquire_timeout`**（中立命名；见上文「旧文案已不再出现」）
- `httpx_pool_observed_full` — bool，超时那一刻的池快照是否已满。这是替代旧二分类的**观测**字段，判断请用它而不是 `classification`
- `exc` — httpcore 原始异常字符串（多半是空的）

KQL / Loki 直接过：`ContainerLog | where LogEntry contains "copilot_pool_timeout"` 拿 JSON 字段即可。

### 关键设计

- **单端点一律快速失败**：所有 Copilot endpoint 共享同一个 `httpx.AsyncClient`，"换 endpoint 重试"仍打同一个 pool，等于让用户再等一个 `POOL_ACQUIRE_TIMEOUT`。所以 `len(endpoints) <= 1` 时不重试、直接给客户端 SSE error，让 Codex 自己 retry。**判据只有端点数**——从前这里还要求 `classification == upstream_connect_stalled`，那个分类撤回后判据一并简化，于是单端点场景不再有「按误导性分类走进重试」的分支。
- **多端点仍走原退避 + `_select_endpoint` 换端点**流程（受 `attempt < max_retries - 1` 约束）。

### 相关配置

- `COPILOT_POOL_ACQUIRE_TIMEOUT`（默认 20s，从 60s 下调）：httpx 池 acquire 超时。想恢复旧行为设 `60`。
- `COPILOT_UPSTREAM_PROBE_TIMEOUT`（默认 3s）：DNS + TCP 探针的每一步超时。
- `COPILOT_UPSTREAM_PROBE_CACHE_TTL`（默认 5s）：探针结果缓存 TTL，防止密集失败风暴。
- `COPILOT_POOL_MAX_CONNECTIONS` / `COPILOT_POOL_MAX_KEEPALIVE` / `COPILOT_POOL_KEEPALIVE_EXPIRY`：池上限与 keepalive；只在 `httpx_pool_observed_full=true` 的那类观测下调大池上限才有意义。

### 相关 metrics

- `copilot_pool_timeout_total` — **权威计数**，所有 PoolTimeout 累计
- `copilot_pool_timeout_saturated_total` — 其中「超时那一刻池快照已满」的次数（观测，非根因）
- ~~`copilot_pool_timeout_upstream_stall_total`~~ — **已废弃，结构性恒 0**。没有任何自增点，HELP 文本标了 `Deprecated`；保留只为不破坏既有抓取。**不要基于它写告警**，否则得到一条永远不会触发的规则（回归守卫：`tests/test_pool_timeout_classification.py`）

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

> 与 [第 14 节](#14-codex-报-input-item-does-not-belong-to-this-connection含-input-item-id) 的区别：本节是**跨账户**重放 opaque state（换 endpoint 触发，单账户下不成立），靠 pinning 解决；第 14 节是**同账户内** item id 所属 connection 已消亡，与账户数无关，靠剥离 `input[*].id` 解决。错误文案带 `does not belong to this connection` 的一律看第 14 节。
>
> 补充实测（2026-09-09）：GHCP **不支持 `previous_response_id`**，带上直接 400 `previous_response_id is not supported`，所以 Copilot 路径上实际起作用的 opaque state 只有 `encrypted_content`。
>
> 熔断归属（2026-09-09 起）：本节这类 401 是 **request-scoped**，已**不再计入** endpoint `consecutive_errors` —— 否则一个坏会话能把整个账户熔断，见 [第 15 节](#15-一个账户下所有模型突然不可用401-熔断污染)。

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

## 13. Token exchange 被共享 streaming pool 卡住

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

---

## 14. Codex 报 `input item does not belong to this connection`（含 "input item ID"）

### 症状

Codex APP / codex-cli 在多轮对话中报（常与前半句同时出现）：

```
stream disconnected before completion: [req=164f73752ca9867d4932ace454e868ca]
input item does not belong to this connection
```

特征：**一旦出现，这个会话往后每一轮都失败**；新开会话正常。上游原文形态是

```json
{"error":{"code":"bad_request","type":"websocket_error",
          "message":"input item ID does not belong to this connection"}}
```

### 真因（2026-09-09 直连 GHCP Enterprise 上游实测 + 上游 issue 交叉印证）

GHCP 给 Responses output item 铸造的 `id` **不是** OpenAI 的 `msg_xxx` 短 id，而是 **424~428 字符的签名不透明 blob**，并且被密码学校验：

| 实验 | 结果 |
|---|---|
| 篡改 id 任意 20 字符 | 400（验签失败后落到通用 schema「max length 64」） |
| id 与 `encrypted_content` 错配 | 400 `invalid_request_body` |
| 换 session token / 换 HTTP 连接 / 跨 response 混合 item / 8 分钟旧 id | 全部 200（**不是**这些维度） |

这个 blob 绑在 GHCP 服务端某个 "connection" 上。connection 消亡后，客户端仍在回放的旧 id 就成了 orphan，被永久拒绝。上游印证：

- `github/copilot-cli#2147` — GitHub 官方结论：*"stale WebSocket state being reused after a reconnection"*
- `github/copilot-cli#4505` — 中断后恢复旧会话触发；该会话永久失败，连 `/fork` 都救不回
- `caozhiyuan/copilot-api#235` — 与本 LB 同形态的代理，多实例分流下 `/responses` 多轮几乎必挂

**LB 在其中的角色是「制造 orphan」**：我们每一次中途断流都会让 Codex 留下半成品 item 并持续回放其 id ——
`stream_truncated_no_completion` / `PoolTimeout` / 换端点重试 / HTML 软熔断 / HTTP/2 `ConnectionTerminated`。
所以那条错误里的两句话是**因→果**：前半句（断流）造成后半句（会话中毒）。

### 解决（已内置，默认开启）

LB 在转发前一律剥掉 `input[*].id`（`CopilotProxy._strip_input_item_ids`），因此不存在可被拒绝的 orphan id。
**只剥 item 顶层 `id`**，`call_id` 与 `encrypted_content` 原样保留。实测代价为零：

- reasoning item 剥 id 后上下文不丢（保留 `encrypted_content`）
- `function_call` 回路正常（配对靠 `call_id`，与 `id` 无关）
- **prompt cache 不受影响** —— `cached_tokens` 与保留 id 完全相同

另有兜底：若上游仍以该错误拒绝且尚未发出任何 SSE 帧，走**有界恢复阶梯**（先再剥一次 id，再删
reasoning 的 `encrypted_content`），见 [第 17 节](#17-上游拒绝-opaque-state-时的有界恢复阶梯)。

### 排查命令

```bash
# 1) 剥离是否在工作（分子/分母 = 客户端平均每请求回放多少个 id）
curl -s $LB/metrics | grep copilot_input_item_ids_stripped

# 2) 看 stage 而不只看总量。**recovered > 0 才是「还有别的 id 通道」的金丝雀**；
#    detected > 0 只说明上游拒绝了本请求携带的状态，未必与 id 有关（见第 17 节）
curl -s $LB/metrics | grep copilot_orphaned_item_id_events_total

# 3) 找造成中毒的那次断流（用客户端错误里的 req= 值）
kubectl -n <ns> logs deploy/claude-lb | \
  jq -c 'select(.kind=="copilot_stream_end" and .request_id=="<req 值>")'

# 4) 断流总量趋势 —— 这是根源，剥离只是让它不再升级为会话中毒
curl -s $LB/metrics | grep -E "stream_truncated_no_completion_total|pool_timeout_total|upstream_html_events_all"
```

### 临时关闭（仅排查用）

```bash
COPILOT_STRIP_INPUT_ITEM_IDS=false    # 恢复原样透传
```

### 注意

剥离**不会**削弱跨账户重放保护：statefulness 由 `previous_response_id` / `encrypted_content` 判定，与 `id` 无关，
`copilot_stateful_request_pinned_total` 的语义不变（有回归测试锁住这条不变量）。

`copilot_orphaned_item_id_events_total` 此前在健康态下**没有 0 序列**（空 dict → 无 sample），现已修好，
告警可直接写 `> 0`，见 [第 15 节](#15-一个账户下所有模型突然不可用401-熔断污染)末尾。

**旧文档说「`detected` 恒为 0 才说明剥离奏效、非 0 = 还有别的 id 通道」——这条已被生产反证并更正。**
2026-09-10 生产 `detected=12` 而同一条日志明写 `no input[*].id to strip`：上游拒绝的其实是 reasoning
`encrypted_content` 的**归属**，主动剥 id 结构上防不住。正确读法见 [第 17 节](#17-上游拒绝-opaque-state-时的有界恢复阶梯)。

另外：会话被永久毒化的原因**不是** LB 重启换 session token —— 那条假设已被实测否证，见
[第 16 节](#16-重启会让客户端手里的会话状态失效--实测不成立)。

---

## 15. 一个账户下**所有模型**突然不可用（401 熔断污染）

### 症状

某个 Copilot 账户下**全部**模型同时不可用，而不只是出问题的那个会话：

```
503  {"error":{"message":"No available Copilot endpoint for model ..."}}
```

`/metrics` 上同时看到：

```
copilot_endpoint_circuit_open{endpoint="..."} 1
copilot_endpoint_consecutive_errors{endpoint="..."} 5
```

日志里通常是同一批 401 反复出现，而 token exchange 一切正常（`copilot_token_refresh_failed_total` 不涨）。

### 机制（2026-09-09 实测）

401 曾被排除在 `is_client_error` 之外 ⇒ `failed=True` ⇒ `consecutive_errors += 1` ⇒ 连续 5 次打开
**endpoint 级**熔断。而一个 Copilot endpoint 承载该账户全部模型，所以「一个会话持续 401」会升级成
「该账户所有模型下线」。实测的计数关系：

| 实测项 | 结果 |
|---|---|
| 一个持续 401 的请求（流式与非流式都一样） | 上游被调 2 次，`consecutive_errors` 只 +1 |
| 为什么只 +1 | 首次 401 走 auth repair 的 `continue`，位置在 `end_current_request` **之前**，不计数 |
| 阈值 5 | 第 5 个这样的请求打开熔断 |
| 爆炸半径 | 熔断后 `_select_endpoint` 对**任意**模型、**任意** api_type 都返 `None` |

所以「5 个请求 × 每个 2 条 401 日志 = 10 条日志，而 `consecutive_errors=5`」不是巧合，是这个结构的必然结果。

### 现在的行为

`_classify_upstream_failure` 把 401 按来源分开（详见 CLAUDE.md 的「上游 401 按来源分类」一节）：

| 401 来源 | 分类 | 熔断 |
|---|---|---|
| 请求携带 opaque state（跨账户重放 / 失效 item id） | request-scoped | **不计数** |
| 无 opaque state（凭证、席位、策略） | endpoint-scoped | 照旧计数 → 照旧熔断 |

判据是 `_request_has_opaque_state`。这与 pinning 是同一设计的两面：`_select_endpoint` 对 stateful 请求
只返回 pinned endpoint，而 pinned 必须通过 circuit —— 所以对 stateful 请求熔断**只有害无益**（杀掉该会话
唯一可能服务的 endpoint 并带走该账户其他全部流量，而 failover 本来就被 pinning 禁止）。

### 诊断

```bash
# 1. 401 到底算在谁头上
curl -s $LB/metrics | grep copilot_upstream_401_total
#   scope="request" 高 + scope="endpoint" 为 0  → 客户端在回放坏状态，账户是好的
#   scope="endpoint" 持续增长                    → 真凭证/席位问题，熔断是对的

# 2. 熔断现状与恢复倒计时
curl -s $LB/metrics | grep -E 'copilot_endpoint_(circuit_open|consecutive_errors)'

# 3. 区分「long-lived token 失效」（走另一条更快的路）
curl -s $LB/metrics | grep copilot_token_refresh_failed_total
#   涨 → _mark_endpoint_unhealthy 会直接 _open()，与 consecutive_errors 无关

# 4. 已熔断时怎么恢复：一次干净的最小请求打到 HALF_OPEN 试探槽即可
curl -s $LB/v1/chat/completions -H "Authorization: Bearer $LB_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"ok"}]}'
```

### 临时关闭（仅排查用）

```bash
COPILOT_STATEFUL_401_NEUTRAL=false    # 恢复旧行为：所有 401 都计入熔断
```

### 已知次要交互（**2026-09-10 已修**）

原文记录：`_stream_response` 的重试循环只有一个 `for attempt in range(max_retries)`，而**恢复分支与 401
auth repair 分支都用 `continue`**，两者都会推进 `attempt`。所以若一次请求的首个失败是 orphan-401，
恢复重试后 `attempt` 已经是 1；此时若紧接着撞上真正的 session token 过期 401，`attempt == 0` 不成立 →
**拿不到那次免费的 auth repair**。

已修：auth repair 的预算改成**每请求一次的独立 flag**（`auth_repaired`），与 `attempt` 计数解耦，
流式与非流式两条路径都改了。回归测试
`tests/test_copilot_opaque_state_recovery.py::AuthRepairIsolationTests::test_auth_repair_budget_survives_a_recovery_rung`
锁住这条：恢复用掉一个 rung 之后，紧随的真 401 仍能拿到那次刷新。

**副作用要知道**：解耦之后，「换到新 endpoint 后才出现的 401」现在也能拿到一次刷新（旧守卫下拿不到）。
这是有意的 —— 新 endpoint 的 401 更该刷 token —— 但它是一处超出最小改动的行为变化。上界不变：
每个客户端请求最多一次 auth repair。

### 顺带修好的：金丝雀指标此前「缺失 ≠ 零」

`copilot_orphaned_item_id_events_total` / `copilot_stateful_request_pinned_total` /
`copilot_upstream_html_events_by_status_total` / `copilot_upstream_401_total` 由运行期事件填充的 dict 驱动，
健康态下 dict 是空的，于是 `/metrics` 里**只有 HELP/TYPE、零条 sample**。第 14 节要求 `stage="detected"`
**恒为 0**，可当时根本没有 0 序列可看 —— 只能靠 `absent()` 猜「零事件」还是「LB 没部署」。现已在 exposition
层补零，**告警可以直接写 `> 0`**：

```promql
copilot_orphaned_item_id_events_total{stage="detected"} > 0    # 第 14 节的金丝雀
copilot_upstream_401_total{scope="endpoint"} > 0               # 真凭证问题
```

---

## 16. 「重启会让客户端手里的会话状态失效」—— 实测不成立

### 结论

**Copilot session token 轮换不会使 opaque state 失效。** 所以 LB 重启（或后台刷新换 token）不会让 Codex
手里的会话作废，也不会因此产生 401。曾经有过相反的假设，2026-09-09 打真实 GHCP 上游否证了它。

### 实验

Turn 1 拿到一个带 **424 字符 item id + 5324 字符 `encrypted_content`** 的 reasoning item，并在提示里埋一个
暗号。Turn 2 用**全新交换的 session token** + **全新 httpx client**（新 TCP/TLS 连接，模拟进程重启）回放
完整历史并追问暗号：

| Turn 2 组合 | 结果 |
|---|---|
| 新 session token + 保留 `id` | **200**，答出暗号 |
| 新 session token + 剥掉 `id`（LB 现行为） | **200**，答出暗号 |
| 旧 session token + 保留 `id`（对照） | **200**，答出暗号 |

三次独立交换的 TTL 实测均 ≈ **86400 s（24 h）**，不是旧文档写的 30 min。

### 推论

- **不持久化 session token 是有意的**：TTL 既然 24 h，重启代价 = 每 endpoint 一次 HTTP GET；把短期凭证
  落盘是负收益。以 `copilot_session_token_remaining_seconds` 为准，别假定固定 TTL
- **`COPILOT_REFRESH_INTERVAL=300` / `COPILOT_REFRESH_THRESHOLD=600` 在 24 h TTL 下 = 每天刷新 1 次**，
  这是正常的，不要照「30 min TTL」的假设去调
- 真正值得做的是让 **long-lived** 凭证跨重启存活（挂 `copilot-cache` secret 到
  `/home/app/.config/databricks-claude-lb`，repo manifest 已就绪，收敛步骤见 `docs/AKS.md`）
- 会话被永久毒化的原因**不是**重启换 token，而是第 14 节的 connection-bound `input[*].id`

---

## 17. 上游拒绝 opaque state 时的有界恢复阶梯

### 症状

Codex 报 `input item does not belong to this connection`（第 14 节那条），但 `/metrics` 显示

```
copilot_orphaned_item_id_events_total{stage="detected"}      12
copilot_orphaned_item_id_events_total{stage="recovered"}      0    ← 恒 0
copilot_orphaned_item_id_events_total{stage="unrecoverable"} 12
copilot_upstream_401_total{scope="request"}                   6
copilot_token_refresh_total{endpoint="ZenoRewn"}              7
```

同时日志明写 `orphaned-item-id rejection but request carries no input[*].id to strip — cause is elsewhere`。

### 为什么 `recovered` 恒 0（不是巧合，是结构性的）

`_proxy` 已在**发送前**主动剥掉 `input[*].id`（默认开启）。事后的兜底路径再调同一个**幂等** helper，
必然返 0 → 走不进 `recovered` 分支 → 永远落到 `unrecoverable`。所以旧版的兜底在默认配置下是死路。

### 为什么计数是 2 倍（2026-09-10 实测，流式与非流式都一样）

记完 `unrecoverable` 之后代码**继续掉进 401 auth repair**：强制刷一次 session token，再原样重发。
那次刷新**注定无用**——实测 session token 轮换不使 opaque state 失效（第 16 节）。于是一个客户端请求
把每个 orphan 计数器都记了两遍：

| 一个客户端请求（旧行为） | 流式 | 非流式 |
|---|---|---|
| `detected` / `unrecoverable` | 2 | 2 |
| `upstream_401{request}` | 1 | 1 |
| 强制 session token 交换 | 1 | 1 |
| 上游调用次数 | 2 | 2 |

对照上面的生产数字：`12 = 6×2`，`6 = 6×1`，`token_refresh 7 = 6 次浪费的强制刷新 + 1 次 warmup`，
`neutral_requests 7 = 6 + 1`。**唯一请求数就是 6**，不需要按 request_id 去重也能算出来。

### 上游其实有两级独立校验（2026-09-10 直连 GHCP 实测）

篡改 `encrypted_content` 中间 20 个字符，拿到的**不是** orphan 报错：

```json
{"error":{"message":"The encrypted content Zb+H...LQ== could not be verified. Reason: Encrypted content could not be decrypted or parsed.","code":"invalid_request_body"}}
```

所以：**先解密/解析**（失败 → `invalid_request_body`），**再校验归属**（失败 → `does not belong to this
connection`）。orphan 报错意味着那个 blob 能正常解密、只是不属于这条 connection。两点推论：

- 「状态格式不兼容 / 过期」这条根因候选被削弱 —— 那更可能报第一类错误
- 第一类错误此前**完全没有兜底**，现在与 orphan 走同一条恢复阶梯

### 删掉 blob 安全吗

下表是**两次独立隔离会话**的合并（会话 ① 默认参数给 A–E，会话 ② `store:false` + function tool
回路给 B–F），模型均为 `gpt-5.4-mini`。两次的 `id` 都是 420 字符，`encrypted_content` 分别
5160 / 5128 字符 —— 同账号不同轮次长度会变，故表里只写量级。

| 组 | 处理 | 结果 |
|---|---|---|
| A | 原样回放（420 字符 `id` + ~5.1KB `encrypted_content`） | 200，答出暗号 |
| B | 剥 `id`、保留 `encrypted_content`（LB 旧行为） | 200，答出暗号 |
| C | 剥 `id` + 整条删除 reasoning item | **200，答出暗号** |
| D | 剥 `id` + 保留 item 但删 `encrypted_content` | **200，答出暗号** |
| E | 剥 `id` + 篡改 `encrypted_content` | **400 `invalid_request_body`** |
| F | 保留 `id` + 删 `encrypted_content` | 200，答出暗号 |

- `call_id` 回路在 C/D/F 全部正确配对，`function_call_output` 不受影响
- **token 与 prompt cache 代价为零**：`input_tokens=1637 / cached_tokens=1280` 在 B/C/D 三组完全一致
  —— 那个 5000+ 字符的 blob 根本不计入 input tokens
- **不会静默丢历史**：GHCP 不支持 `previous_response_id`（第 11 节），客户端必然把完整历史放在
  `input[]` 里，删掉的只是加密的推理链，message / 工具调用 / 工具结果 / 用户约束一律保留

### 解决（已内置）

```
rung 0  发送前主动剥 input[*].id                 —— 无条件，COPILOT_STRIP_INPUT_ITEM_IDS
rung 1  被拒后再剥一次 id                        —— 只在 rung 0 关掉 / 出现新 id 通道时有料
rung 2  删 reasoning item 的 encrypted_content   —— COPILOT_OPAQUE_STATE_RECOVERY
走完    返回 orphaned_conversation_state 错误      —— 明确要求客户端重建，绝不静默成功
```

预算是**显式计数**（`_OpaqueStateRecovery.MAX_RUNGS = 2`），不靠「改写幂等所以最多重试一次」——
外层 attempt 循环、401 auth repair、`_normal_request` 递归是三条能叠加的通道，一个客户端请求共用一份预算。
**实测上界**：流式 ≤3 次上游调用，非流式 ≤4（1 原始 + 1 auth repair + 2 级恢复）；生产形态（主动剥离
已开、纯 orphan）= 2 次，与改动前**相同** —— 原来那次白打的强制 token 刷新换成了一次真正的恢复尝试。

同时 **opaque-state 拒绝不再喂给 401 auth repair**，那次注定无用的强制 token 交换消失了。
逐个计数器的前后变化（每个唯一客户端请求，生产形态，**实测**）：

| 计数器 | 旧 | 新 |
|---|---|---|
| `orphaned…{detected}` | 2 | **2（不变）** —— 第二次拒绝仍真实存在，只是换成了一次真正的恢复尝试 |
| `orphaned…{unrecoverable}` | 2 | **1** —— 只在阶梯走完后记一次 |
| `upstream_401{request}` | 1 | 1 |
| 强制 session token 交换 | 1 | **0** |
| 上游调用次数 | 2 | 2 |

要按「受影响请求数」看规模，读新指标 `copilot_opaque_state_requests_total`，不要拿 `detected` 当请求数。

### 还有一个错误码：`retry_budget_exhausted`

流式重试循环里每个 `continue` 都要求「还剩一轮来消费修好的请求」。原地修复（恢复 rung、auth repair）
过去靠巧合满足这条（剥离幂等 + auth 限死 `attempt == 0`）；阶梯做成两级并把 auth 与 attempt 解耦后
巧合没了，三次原地修复能吃满三轮，而**流式循环体就是生成器体的最后一段**，耗尽就静默结束 ——
下游拿到 HTTP 200 + 0 字节，也就是本文第 14 节那种会毒化会话的无终端断流。

现在两处原地修复都显式带上 `attempt < max_retries - 1`，另外加了一层循环后兜底：真走到那里就发
`retry_budget_exhausted` 终端。按不变量它不可达，纯粹是防后人加 `continue` 时漏掉守卫。
**若 `copilot_stream_retry_budget_exhausted_total > 0`（或日志里见到
`outcome=retry_budget_exhausted`），说明不变量被破坏了，按回归处理。** 该指标健康态下有 `0` 序列，
告警直接写 `> 0`：

```bash
curl -s $LB/metrics | grep copilot_stream_retry_budget_exhausted_total   # 必须恒 0
```

### 排查命令

```bash
# 1) 规模：受影响的唯一客户端请求数（不是事件数）
curl -s $LB/metrics | grep copilot_opaque_state_requests_total

# 2) 是哪一类拒绝
#    orphaned_id         = blob 归属对不上（跨账户 / 旧 connection 的状态）
#    unverifiable_content = blob 解不开（被截断或篡改）
curl -s $LB/metrics | grep copilot_opaque_state_rejections_total

# 3) 阶梯效果。succeeded/attempted = 恢复命中率；exhausted 才是用户真正受损的部分
#    注意 attempted - succeeded - exhausted 是第三桶：上游收下了改写后的请求但流被截断。
#    那一桶的量看 copilot_stream_truncated_no_completion_total，不要误算进「恢复失败」
curl -s $LB/metrics | grep copilot_opaque_state_recovery_total

# 4) 旧指标的正确读法：recovered > 0 才是「还有别的 id 通道」的金丝雀
curl -s $LB/metrics | grep copilot_orphaned_item_id_events_total

# 5) 逐请求看阶梯走了几级
kubectl -n <ns> logs deploy/claude-lb | grep -E "applied (strip_input_item_ids|drop_reasoning_encrypted_content)"
kubectl -n <ns> logs deploy/claude-lb | grep "no recovery step left"
```

### 客户端看到什么

阶梯走完后错误的 `code` 是 `orphaned_conversation_state`，message 说明已尝试几级恢复、要求开新会话或
丢弃历史后重试，并保留 `upstream_ids` / `lb_request_id`。**HTTP 状态码保持上游原值（400/401）**。

> 未决项：把它换成 409/422 语义上更准，但需要先验证 Codex / OpenAI JS SDK 对这两个码的重试与
> fallback 行为。未验证不动 —— 只改 `code` 与 message 对 SDK 行为零风险。

### 临时关闭（仅排查用）

```bash
COPILOT_OPAQUE_STATE_RECOVERY=false   # 只关 rung 2；仍会记 exhausted 并返回结构化错误
COPILOT_STRIP_INPUT_ITEM_IDS=false    # 关 rung 0（会让 rung 1 变得有料）
```

### 诚实边界

`does not belong to this connection` 需要另一个账户/连接铸造的 blob 才能复现，单账号造不出来。所以
**「删 blob 能救那一类拒绝」是推断**。已实测的是：

- 删了之后请求形态合法、上下文保真、工具回路正常、token/cache 代价为零
- 另一类 opaque-state 拒绝（E 组，blob 解不开）**确定**能被它救回

这就是 rung 2 带 kill switch 的原因。上线后看 `succeeded/attempted` 就能把推断变成实测。

---

## 18. 配了第二个 Copilot 账户之后会话报 `does not belong to this connection`

### 症状

单账户时好好的，加了第二个账户就开始出现第 14 / 17 节那条错误，而且**频率很高**。

### 真因（2026-09-10 用两个真实 GHCP 账户实测，双向 × 3 次 = 24/24）

A 账户铸造的 reasoning `encrypted_content` 拿到 B 账户回放**必然** 401。`least_requests` 会把同一
会话的连续轮次分到不同账户，所以这个 401 是**必然事件**而不是偶发 —— 不是「运气不好」，是配置必然
导致的。

### 解决（已内置，默认开启）

`COPILOT_SESSION_AFFINITY=true`：按 Codex 的 `prompt_cache_key`（= `session_id`/`thread_id`，实测
跨重试跨轮次稳定）用 blake2b 哈希到固定账户。**单账户下是完全的 no-op。**

```bash
# hit 应该占绝大多数；单账户下 hit/unavailable 恒为 0
curl -s $LB/metrics | grep copilot_session_affinity_total
```

| outcome | 含义 | 处理 |
|---|---|---|
| `hit` | 同一会话被钉回原账户 —— 每一次都是挡下了一次必然的 401 | 正常 |
| `unavailable` | 亲和目标熔断/冷却，该轮跨账户 → 恢复阶梯兜住，丢一次推理链 | 看那个账户为什么熔断 |
| `absent` | 请求没带 `prompt_cache_key`（非 Codex 客户端） | 该客户端拿不到亲和保护 |

`unavailable` 与 `copilot_opaque_state_requests_total` 应同步增长。**两者背离**（受影响请求远多于
`unavailable`）说明有别的跨账户来源，需要查。

### 端到端实测对照

| | 亲和开 | 亲和关 |
|---|---|---|
| 各账户请求数 | 3 / 0 | 3 / 3 |
| `opaque_state_requests_total` | **0** | **3** |
| `recovery{succeeded}` | 0 | **3** |
| Codex 轮次成功 | 3/3 | 6/6 |

关掉亲和时用户**仍然没看到错误**（阶梯 3/3 全救回，答案正确），但每轮都白打一次上游并丢掉推理链。

### 非 Codex 客户端怎么办

亲和依赖客户端提供稳定会话键。若你的客户端不发 `prompt_cache_key`（`absent` 一直涨），它就拿不到
亲和保护，只能靠恢复阶梯。这种情况下要么让客户端带上该字段，要么**保持单账户**。
