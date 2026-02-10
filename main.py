"""
Databricks Claude Load Balancer Proxy for Claude Code
使用 Databricks 原生 Anthropic 端点 (/anthropic/v1/messages)
"""

import os
import re
import asyncio
import json
import time
import logging
from typing import Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import yaml
import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 模型名称映射 ====================

DATABRICKS_MODELS = {
    "sonnet": "databricks-claude-sonnet-4-5",
    "opus": "databricks-claude-opus-4-6",  # 默认使用 4-6
    "opus-4-5": "databricks-claude-opus-4-5",
    "opus-4-6": "databricks-claude-opus-4-6",
    "haiku": "databricks-claude-haiku-4-5",
}

DEFAULT_MODEL = "databricks-claude-sonnet-4-5"


def get_databricks_model(model: str) -> str:
    """将 Claude 模型名称映射到 Databricks 模型名称"""
    model_lower = model.lower()

    # 已经是 Databricks 模型名称，直接返回
    if model_lower.startswith("databricks-"):
        return model

    # 检查是否指定了具体版本 (如 claude-opus-4-5, opus-4-5)
    if "opus" in model_lower:
        if "4-5" in model_lower or "4.5" in model_lower:
            mapped = DATABRICKS_MODELS["opus-4-5"]
        elif "4-6" in model_lower or "4.6" in model_lower:
            mapped = DATABRICKS_MODELS["opus-4-6"]
        else:
            mapped = DATABRICKS_MODELS["opus"]  # 默认 4-6
    elif "sonnet" in model_lower:
        mapped = DATABRICKS_MODELS["sonnet"]
    elif "haiku" in model_lower:
        mapped = DATABRICKS_MODELS["haiku"]
    else:
        logger.warning(f"Unknown model '{model}', using default: {DEFAULT_MODEL}")
        mapped = DEFAULT_MODEL

    if mapped != model:
        logger.info(f"Model mapping: {model} -> {mapped}")

    return mapped


# ==================== Load Balancer ====================

@dataclass
class WorkspaceEndpoint:
    name: str
    api_base: str
    token: str
    weight: int = 1
    
    active_requests: int = field(default=0, repr=False)
    total_requests: int = field(default=0, repr=False)
    total_errors: int = field(default=0, repr=False)
    last_error_time: Optional[float] = field(default=None, repr=False)
    circuit_open: bool = field(default=False, repr=False)

    # Token 用量
    total_input_tokens: int = field(default=0, repr=False)
    total_output_tokens: int = field(default=0, repr=False)

    # 延迟追踪
    total_response_time: float = field(default=0.0, repr=False)
    successful_requests: int = field(default=0, repr=False)

    # 按模型维度统计: {"model_name": {"input_tokens": int, "output_tokens": int, "requests": int}}
    model_stats: dict = field(default_factory=dict, repr=False)


@dataclass
class GlobalStats:
    total_requests: int = 0
    total_errors: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_response_time: float = 0.0
    successful_requests: int = 0
    start_time: float = field(default_factory=time.time)


class LoadBalancer:
    def __init__(
        self, 
        endpoints: list[WorkspaceEndpoint],
        strategy: str = "least_requests",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60,
    ):
        self.endpoints = endpoints
        self.strategy = strategy
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
    
    def get_available_endpoints(self) -> list[WorkspaceEndpoint]:
        now = time.time()
        available = []
        
        for ep in self.endpoints:
            if ep.circuit_open:
                if ep.last_error_time and now - ep.last_error_time > self.circuit_breaker_timeout:
                    ep.circuit_open = False
                    ep.total_errors = 0
                    logger.info(f"Circuit breaker reset for {ep.name}")
                else:
                    continue
            available.append(ep)
        
        return available
    
    def select_endpoint(self) -> Optional[WorkspaceEndpoint]:
        available = self.get_available_endpoints()
        if not available:
            logger.error("No available endpoints!")
            return None
        
        if self.strategy == "least_requests":
            return min(available, key=lambda ep: ep.active_requests)
        elif self.strategy == "round_robin":
            return available[0]
        else:  # random
            return available[int(time.time() * 1000) % len(available)]
    
    async def on_request_start(self, endpoint: WorkspaceEndpoint):
        endpoint.active_requests += 1
        endpoint.total_requests += 1
    
    async def on_request_end(self, endpoint: WorkspaceEndpoint, success: bool, is_client_error: bool = False):
        endpoint.active_requests = max(0, endpoint.active_requests - 1)

        if not success and not is_client_error:
            # 只有服务端错误才计入错误数，客户端错误（4xx）不触发熔断器
            endpoint.total_errors += 1
            endpoint.last_error_time = time.time()

            if endpoint.total_errors >= self.circuit_breaker_threshold:
                endpoint.circuit_open = True
                logger.warning(f"Circuit breaker opened for {endpoint.name}")
    
    def get_stats(self) -> dict:
        return {
            "endpoints": [
                {
                    "name": ep.name,
                    "active_requests": ep.active_requests,
                    "total_requests": ep.total_requests,
                    "total_errors": ep.total_errors,
                    "successful_requests": ep.successful_requests,
                    "error_rate": round(ep.total_errors / ep.total_requests * 100, 2) if ep.total_requests > 0 else 0,
                    "circuit_open": ep.circuit_open,
                    "total_input_tokens": ep.total_input_tokens,
                    "total_output_tokens": ep.total_output_tokens,
                    "total_tokens": ep.total_input_tokens + ep.total_output_tokens,
                    "avg_response_time_ms": round(ep.total_response_time / ep.successful_requests * 1000, 1) if ep.successful_requests > 0 else 0,
                    "model_stats": ep.model_stats,
                }
                for ep in self.endpoints
            ]
        }


# ==================== Claude Proxy (使用原生 Anthropic 端点) ====================

class ClaudeProxy:
    def __init__(self, load_balancer: LoadBalancer, api_key: str):
        self.load_balancer = load_balancer
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        self.global_stats = GlobalStats()

    def _record_usage(self, endpoint: WorkspaceEndpoint, model: str, input_tokens: int, output_tokens: int, elapsed: float):
        """记录 token 用量和延迟指标"""
        # 端点级别
        endpoint.total_input_tokens += input_tokens
        endpoint.total_output_tokens += output_tokens
        endpoint.total_response_time += elapsed
        endpoint.successful_requests += 1
        if model not in endpoint.model_stats:
            endpoint.model_stats[model] = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        endpoint.model_stats[model]["input_tokens"] += input_tokens
        endpoint.model_stats[model]["output_tokens"] += output_tokens
        endpoint.model_stats[model]["requests"] += 1
        # 全局级别
        self.global_stats.total_input_tokens += input_tokens
        self.global_stats.total_output_tokens += output_tokens
        self.global_stats.total_response_time += elapsed
        self.global_stats.successful_requests += 1
        self.global_stats.total_requests += 1

    async def close(self):
        await self.client.aclose()
    
    def verify_api_key(self, key: str) -> bool:
        return key == self.api_key
    
    async def proxy_request(self, body: dict, stream: bool = False):
        """代理请求到 Databricks 原生 Anthropic 端点"""
        
        # 转换模型名称
        if "model" in body:
            original_model = body["model"]
            body["model"] = get_databricks_model(original_model)
        
        # 移除 Databricks 不支持的字段（如新版 Claude Code 发送的 context_management 等）
        unsupported_fields = ["context_management"]
        for field in unsupported_fields:
            if field in body:
                logger.info(f"Removing unsupported field: {field}")
                del body[field]

        # 处理 thinking 参数兼容性
        # Opus 4.6: 支持 adaptive（推荐），enabled + budget_tokens 已废弃
        # 旧模型 (Sonnet 4.5, Opus 4.5 等): 仅支持 enabled + budget_tokens
        if "thinking" in body and isinstance(body["thinking"], dict):
            thinking_type = body["thinking"].get("type")
            model = body.get("model", "")
            is_opus_4_6 = "opus-4-6" in model

            if is_opus_4_6:
                # Opus 4.6: 使用 adaptive，移除多余的 budget_tokens
                if thinking_type == "adaptive" and "budget_tokens" in body["thinking"]:
                    del body["thinking"]["budget_tokens"]
                    logger.info("Removed budget_tokens for adaptive thinking (Opus 4.6)")
            else:
                # 旧模型: 不支持 adaptive，需转换为 enabled + budget_tokens
                if thinking_type == "adaptive":
                    body["thinking"]["type"] = "enabled"
                    max_tokens = body.get("max_tokens", 16000)
                    budget = body["thinking"].pop("budget_tokens", None) or max(1024, int(max_tokens * 0.8))
                    if max_tokens <= budget:
                        body["max_tokens"] = budget + 1
                    body["thinking"]["budget_tokens"] = budget
                    logger.info(f"Converted adaptive -> enabled with budget_tokens={budget} for {model}")
                elif thinking_type == "enabled" and "budget_tokens" not in body["thinking"]:
                    max_tokens = body.get("max_tokens", 16000)
                    budget = max(1024, int(max_tokens * 0.8))
                    if max_tokens <= budget:
                        body["max_tokens"] = budget + 1
                    body["thinking"]["budget_tokens"] = budget
                    logger.info(f"Added missing budget_tokens: {budget}")
        
        max_retries = 3
        last_error = None
        model = body.get("model", "unknown")
        start_time = time.time()

        for attempt in range(max_retries):
            endpoint = self.load_balancer.select_endpoint()
            if not endpoint:
                raise HTTPException(status_code=503, detail={"error": {"message": "No available endpoints"}})

            await self.load_balancer.on_request_start(endpoint)

            # 使用原生 Anthropic 端点
            url = f"{endpoint.api_base}/anthropic/v1/messages"

            logger.info(f"[{body.get('model')}] -> {endpoint.name} (attempt {attempt + 1})")

            try:
                headers = {
                    "Authorization": f"Bearer {endpoint.token}",
                    "Content-Type": "application/json",
                }

                if stream:
                    return await self._stream_request(endpoint, url, body, headers, model=model, start_time=start_time)
                else:
                    return await self._normal_request(endpoint, url, body, headers, model=model, start_time=start_time)
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                self.global_stats.total_errors += 1
                # 429 rate limit 也应该触发熔断，因为表示服务端过载
                is_client_error = 400 <= e.response.status_code < 500 and e.response.status_code != 429
                await self.load_balancer.on_request_end(endpoint, success=False, is_client_error=is_client_error)

                if e.response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"{endpoint.name} returned {e.response.status_code}, retrying...")
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                else:
                    try:
                        error_body = e.response.json()
                    except:
                        error_body = {"error": {"message": e.response.text}}

                    logger.error(f"Request failed with {e.response.status_code}: {json.dumps(error_body, ensure_ascii=False)}")
                    raise HTTPException(status_code=e.response.status_code, detail=error_body)

            except Exception as e:
                last_error = e
                self.global_stats.total_errors += 1
                logger.error(f"{endpoint.name} failed: {e}")
                await self.load_balancer.on_request_end(endpoint, success=False)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
        
        raise HTTPException(status_code=503, detail={"error": {"message": f"All retries failed: {last_error}"}})
    
    async def _normal_request(self, endpoint, url, body, headers, model: str = "unknown", start_time: float = 0) -> JSONResponse:
        """非流式请求 - 直接透传"""
        response = await self.client.post(url, json=body, headers=headers)
        response.raise_for_status()
        await self.load_balancer.on_request_end(endpoint, success=True)

        elapsed = time.time() - start_time
        resp_json = response.json()
        usage = resp_json.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        self._record_usage(endpoint, model, input_tokens, output_tokens, elapsed)

        return JSONResponse(content=resp_json, status_code=response.status_code)
    
    async def _stream_request(self, endpoint, url, body, headers, max_retries: int = 3, model: str = "unknown", start_time: float = 0) -> StreamingResponse:
        """流式请求 - 直接透传 Databricks 的 Anthropic 格式响应，支持重试"""

        proxy_self = self

        async def stream_generator():
            current_endpoint = endpoint
            current_url = url
            current_headers = headers
            input_tokens = 0
            output_tokens = 0

            for attempt in range(max_retries):
                success = False
                is_client_error = False

                try:
                    req = proxy_self.client.build_request("POST", current_url, json=body, headers=current_headers)
                    response = await proxy_self.client.send(req, stream=True)

                    if response.status_code >= 400:
                        error_body = await response.aread()
                        # 429 rate limit 也触发熔断
                        is_client_error = 400 <= response.status_code < 500 and response.status_code != 429

                        try:
                            error_json = json.loads(error_body)
                            error_msg = error_json.get('message', 'Request failed')
                            logger.error(f"Stream request failed ({response.status_code}): {error_json}")
                        except:
                            error_msg = error_body.decode('utf-8') if isinstance(error_body, bytes) else str(error_body)
                            logger.error(f"Stream request failed ({response.status_code}): {error_msg}")

                        await proxy_self.load_balancer.on_request_end(current_endpoint, success=False, is_client_error=is_client_error)

                        # 可重试的状态码
                        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                            logger.warning(f"{current_endpoint.name} returned {response.status_code}, retrying stream...")
                            await asyncio.sleep(min(2 ** attempt, 8))
                            # 选择新的 endpoint 重试
                            new_endpoint = proxy_self.load_balancer.select_endpoint()
                            if new_endpoint:
                                current_endpoint = new_endpoint
                                current_url = f"{current_endpoint.api_base}/anthropic/v1/messages"
                                current_headers = {
                                    "Authorization": f"Bearer {current_endpoint.token}",
                                    "Content-Type": "application/json",
                                }
                                await proxy_self.load_balancer.on_request_start(current_endpoint)
                                logger.info(f"[{body.get('model')}] -> {current_endpoint.name} (stream attempt {attempt + 2})")
                                continue

                        yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_msg}})}\n\n".encode()
                        return

                    # 透传响应，同时嗅探 token 用量（先 yield 再解析，零延迟影响）
                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        yield chunk  # 立即转发，不影响流式体验
                        try:
                            buffer += chunk.decode("utf-8", errors="ignore")
                            while "\n\n" in buffer:
                                event_str, buffer = buffer.split("\n\n", 1)
                                if '"message_start"' in event_str or '"message_delta"' in event_str:
                                    for line in event_str.split("\n"):
                                        if line.startswith("data: "):
                                            data = json.loads(line[6:])
                                            if data.get("type") == "message_start":
                                                input_tokens = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                                            elif data.get("type") == "message_delta":
                                                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                            if len(buffer) > 65536:
                                buffer = ""  # 防止内存泄漏
                        except Exception:
                            pass  # 指标提取绝不能阻断流

                    success = True
                    await proxy_self.load_balancer.on_request_end(current_endpoint, success=True)
                    elapsed = time.time() - start_time
                    proxy_self._record_usage(current_endpoint, model, input_tokens, output_tokens, elapsed)
                    return

                except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
                    # 网络超时/连接错误 - 应该触发熔断并重试
                    error_detail = f"{type(e).__name__}: {str(e) or 'Unknown error'}"
                    logger.error(f"Stream network error on {current_endpoint.name}: {error_detail}")
                    await proxy_self.load_balancer.on_request_end(current_endpoint, success=False, is_client_error=False)

                    if attempt < max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 8))
                        new_endpoint = proxy_self.load_balancer.select_endpoint()
                        if new_endpoint:
                            current_endpoint = new_endpoint
                            current_url = f"{current_endpoint.api_base}/anthropic/v1/messages"
                            current_headers = {
                                "Authorization": f"Bearer {current_endpoint.token}",
                                "Content-Type": "application/json",
                            }
                            await proxy_self.load_balancer.on_request_start(current_endpoint)
                            logger.info(f"[{body.get('model')}] -> {current_endpoint.name} (stream retry {attempt + 2})")
                            continue

                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_detail}})}\n\n".encode()
                    return

                except Exception as e:
                    import traceback
                    error_detail = f"{type(e).__name__}: {str(e) or 'Unknown error'}"
                    logger.error(f"Stream error: {error_detail}\n{traceback.format_exc()}")
                    await proxy_self.load_balancer.on_request_end(current_endpoint, success=False, is_client_error=False)
                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_detail}})}\n\n".encode()
                    return

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    def get_stats(self) -> dict:
        gs = self.global_stats
        uptime = time.time() - gs.start_time
        return {
            "global": {
                "uptime_seconds": round(uptime, 1),
                "total_requests": gs.total_requests,
                "total_errors": gs.total_errors,
                "total_input_tokens": gs.total_input_tokens,
                "total_output_tokens": gs.total_output_tokens,
                "total_tokens": gs.total_input_tokens + gs.total_output_tokens,
                "avg_response_time_ms": round(gs.total_response_time / gs.successful_requests * 1000, 1) if gs.successful_requests > 0 else 0,
                "requests_per_minute": round(gs.total_requests / (uptime / 60), 2) if uptime > 0 else 0,
            },
            **self.load_balancer.get_stats(),
        }


# ==================== Config ====================

def expand_env_vars(value: str) -> str:
    pattern = r'\$\{([^}]+)\}'
    def replace(match):
        return os.getenv(match.group(1), "")
    return re.sub(pattern, replace, value)


def load_config(config_path: str = "config.yaml") -> ClaudeProxy:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lb_config = config.get("load_balancer", {})
    api_key = expand_env_vars(config.get("auth", {}).get("api_key", ""))
    
    endpoints = []
    for ep in config.get("endpoints", []):
        endpoints.append(WorkspaceEndpoint(
            name=ep["name"],
            api_base=ep["api_base"],
            token=expand_env_vars(ep["token"]),
            weight=ep.get("weight", 1),
        ))
        logger.info(f"Loaded endpoint: {ep['name']}")
    
    load_balancer = LoadBalancer(
        endpoints=endpoints,
        strategy=lb_config.get("strategy", "least_requests"),
        circuit_breaker_threshold=lb_config.get("circuit_breaker_threshold", 5),
        circuit_breaker_timeout=lb_config.get("circuit_breaker_timeout", 60),
    )
    
    return ClaudeProxy(load_balancer, api_key)


# ==================== FastAPI App ====================

proxy: Optional[ClaudeProxy] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    proxy = load_config(config_path)
    logger.info(f"Proxy started with {len(proxy.load_balancer.endpoints)} endpoints (using native Anthropic endpoint)")
    yield
    if proxy:
        await proxy.close()


app = FastAPI(title="Databricks Claude Proxy (Native Anthropic)", lifespan=lifespan)


MAX_REQUEST_SIZE = 4 * 1024 * 1024  # 4MB Databricks limit


@app.post("/v1/messages")
async def messages(request: Request, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    auth_header = request.headers.get("authorization", "")
    actual_key = x_api_key or (auth_header[7:] if auth_header.startswith("Bearer ") else "")

    if not proxy.verify_api_key(actual_key):
        raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API key"}})

    # 获取原始请求体并检查大小
    body_bytes = await request.body()
    body_size = len(body_bytes)

    if body_size > MAX_REQUEST_SIZE:
        size_mb = body_size / (1024 * 1024)
        logger.warning(f"Request too large: {size_mb:.2f}MB (limit: 4MB)")
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "type": "request_too_large",
                    "message": f"Request size ({size_mb:.2f}MB) exceeds Databricks 4MB limit. Please use /clear to start a new conversation or remove large content from context."
                }
            }
        )

    body = json.loads(body_bytes)
    stream = body.get("stream", False)

    logger.info(f"Request: model={body.get('model')}, stream={stream}, thinking={body.get('thinking')}, size={body_size/1024:.1f}KB")

    return await proxy.proxy_request(body, stream=stream)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    content = json.dumps(body.get("messages", []))
    estimated_tokens = len(content) // 4
    return {"input_tokens": estimated_tokens}


@app.post("/api/event_logging/batch")
async def event_logging():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/stats")
async def stats():
    return proxy.get_stats()


@app.post("/reset")
async def reset():
    for ep in proxy.load_balancer.endpoints:
        ep.circuit_open = False
        ep.total_errors = 0
        ep.total_input_tokens = 0
        ep.total_output_tokens = 0
        ep.total_response_time = 0.0
        ep.successful_requests = 0
        ep.model_stats = {}
    proxy.global_stats = GlobalStats()
    return {"status": "reset"}


@app.get("/stats/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Databricks Claude LB Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}
h1{font-size:1.5rem;margin-bottom:16px;color:#f8fafc}
.refresh-info{font-size:.8rem;color:#64748b;margin-bottom:16px}
.global-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}
.card{background:#1e293b;border-radius:10px;padding:16px;border:1px solid #334155}
.card .label{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.card .value{font-size:1.5rem;font-weight:700;margin-top:4px;color:#f1f5f9}
.card .value.green{color:#4ade80}
.card .value.blue{color:#60a5fa}
.card .value.amber{color:#fbbf24}
table{width:100%;border-collapse:collapse;margin-bottom:24px;font-size:.85rem}
th{background:#1e293b;padding:10px 12px;text-align:left;color:#94a3b8;font-weight:600;border-bottom:2px solid #334155}
td{padding:10px 12px;border-bottom:1px solid #1e293b}
tr:hover td{background:#1e293b80}
.badge{display:inline-block;padding:2px 8px;border-radius:9999px;font-size:.7rem;font-weight:600}
.badge.ok{background:#065f4620;color:#4ade80;border:1px solid #4ade8040}
.badge.open{background:#7f1d1d20;color:#f87171;border:1px solid #f8717140}
.model-section{margin-bottom:24px}
.model-section h2{font-size:1.1rem;margin-bottom:10px;color:#cbd5e1}
.error-banner{background:#7f1d1d40;color:#fca5a5;padding:12px;border-radius:8px;margin-bottom:16px;display:none}
</style>
</head>
<body>
<h1>Databricks Claude LB Dashboard</h1>
<div class="refresh-info" id="refreshInfo">Updating...</div>
<div class="error-banner" id="errorBanner"></div>
<div class="global-grid" id="globalGrid"></div>
<h2 style="color:#cbd5e1;margin-bottom:10px">Endpoints</h2>
<div style="overflow-x:auto"><table id="endpointTable"><thead><tr>
<th>Name</th><th>Active</th><th>Total</th><th>Success</th><th>Errors</th><th>Error Rate</th><th>Circuit</th>
<th>Input Tokens</th><th>Output Tokens</th><th>Total Tokens</th><th>Avg Latency</th>
</tr></thead><tbody id="endpointBody"></tbody></table></div>
<div class="model-section"><h2>Model Stats</h2><div id="modelStats"></div></div>
<script>
function fmt(n){if(n>=1e6)return(n/1e6).toFixed(2)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return n.toString()}
function fmtTime(s){if(s>=3600)return(s/3600).toFixed(1)+'h';if(s>=60)return(s/60).toFixed(1)+'m';return s.toFixed(0)+'s'}

async function refresh(){
  try{
    const r=await fetch('/stats');
    const d=await r.json();
    const g=d.global;
    document.getElementById('errorBanner').style.display='none';
    document.getElementById('refreshInfo').textContent='Last updated: '+new Date().toLocaleTimeString()+' (auto-refresh 5s)';

    const cards=[
      {label:'Uptime',value:fmtTime(g.uptime_seconds),cls:''},
      {label:'Total Requests',value:fmt(g.total_requests),cls:'blue'},
      {label:'Input Tokens',value:fmt(g.total_input_tokens),cls:''},
      {label:'Output Tokens',value:fmt(g.total_output_tokens),cls:''},
      {label:'Total Tokens',value:fmt(g.total_tokens),cls:'amber'},
      {label:'Avg Latency',value:g.avg_response_time_ms.toFixed(0)+'ms',cls:''},
      {label:'RPM',value:g.requests_per_minute.toFixed(2),cls:'green'},
    ];
    const grid=document.getElementById('globalGrid');
    grid.innerHTML=cards.map(c=>'<div class="card"><div class="label">'+c.label+'</div><div class="value '+(c.cls||'')+'">'+c.value+'</div></div>').join('');

    const tbody=document.getElementById('endpointBody');
    tbody.innerHTML=d.endpoints.map(e=>'<tr>'+
      '<td><strong>'+e.name+'</strong></td>'+
      '<td>'+e.active_requests+'</td>'+
      '<td>'+fmt(e.total_requests)+'</td>'+
      '<td>'+fmt(e.successful_requests)+'</td>'+
      '<td>'+e.total_errors+'</td>'+
      '<td>'+e.error_rate+'%</td>'+
      '<td><span class="badge '+(e.circuit_open?'open':'ok')+'">'+(e.circuit_open?'OPEN':'OK')+'</span></td>'+
      '<td>'+fmt(e.total_input_tokens)+'</td>'+
      '<td>'+fmt(e.total_output_tokens)+'</td>'+
      '<td>'+fmt(e.total_tokens)+'</td>'+
      '<td>'+e.avg_response_time_ms.toFixed(0)+'ms</td>'+
    '</tr>').join('');

    // Model stats aggregation
    const models={};
    d.endpoints.forEach(e=>{Object.entries(e.model_stats||{}).forEach(([m,s])=>{
      if(!models[m])models[m]={input_tokens:0,output_tokens:0,requests:0};
      models[m].input_tokens+=s.input_tokens;models[m].output_tokens+=s.output_tokens;models[m].requests+=s.requests;
    })});
    const ms=document.getElementById('modelStats');
    const entries=Object.entries(models);
    if(entries.length===0){ms.innerHTML='<div style="color:#64748b">No data yet</div>';return;}
    ms.innerHTML='<table><thead><tr><th>Model</th><th>Requests</th><th>Input Tokens</th><th>Output Tokens</th><th>Total Tokens</th></tr></thead><tbody>'+
      entries.map(([m,s])=>'<tr><td>'+m+'</td><td>'+fmt(s.requests)+'</td><td>'+fmt(s.input_tokens)+'</td><td>'+fmt(s.output_tokens)+'</td><td>'+fmt(s.input_tokens+s.output_tokens)+'</td></tr>').join('')+
    '</tbody></table>';
  }catch(err){
    const b=document.getElementById('errorBanner');b.textContent='Failed to fetch stats: '+err.message;b.style.display='block';
  }
}
refresh();
setInterval(refresh,5000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_v2:app", host="0.0.0.0", port=8000, reload=True)
