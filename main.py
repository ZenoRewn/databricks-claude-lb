"""
Databricks Claude Load Balancer Proxy for Claude Code
使用 Databricks 原生 Anthropic 端点 (/anthropic/v1/messages)
"""

import os
import re
import asyncio
import json
import time
import random
import logging
from typing import Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

import yaml
import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 模型名称映射 ====================

DATABRICKS_MODELS = {
    "sonnet": "databricks-claude-sonnet-4-6",  # 默认使用最新版本
    "sonnet-4-5": "databricks-claude-sonnet-4-5",
    "sonnet-4-6": "databricks-claude-sonnet-4-6",
    "opus": "databricks-claude-opus-4-7",  # 默认使用最新版本
    "opus-4-5": "databricks-claude-opus-4-5",
    "opus-4-6": "databricks-claude-opus-4-6",
    "opus-4-7": "databricks-claude-opus-4-7",
    "haiku": "databricks-claude-haiku-4-5",
}

DEFAULT_MODEL = "databricks-claude-sonnet-4-6"

# Anthropic 官方 API 定价 (USD per 1M tokens)
MODEL_PRICING = {
    "opus-4-7": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "opus-4-6": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "opus-4-5": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "sonnet-4-5": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
}


def get_model_pricing(model_name: str) -> Optional[dict]:
    model_lower = model_name.lower()
    for key, pricing in MODEL_PRICING.items():
        if key in model_lower:
            return pricing
    return None


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int,
                   cache_creation_tokens: int = 0, cache_read_tokens: int = 0) -> Optional[float]:
    pricing = get_model_pricing(model_name)
    if not pricing:
        return None
    cost = (
        input_tokens * pricing["input"] / 1_000_000 +
        output_tokens * pricing["output"] / 1_000_000 +
        cache_creation_tokens * pricing["cache_write"] / 1_000_000 +
        cache_read_tokens * pricing["cache_read"] / 1_000_000
    )
    return round(cost, 6)


def get_databricks_model(model: str) -> str:
    """将 Claude 模型名称映射到 Databricks 模型名称"""
    model_lower = model.lower()

    # 已经是 Databricks 模型名称，直接返回
    if model_lower.startswith("databricks-"):
        return model

    # 检查是否指定了具体版本 (如 claude-opus-4-7, opus-4-5)
    if "opus" in model_lower:
        if "4-5" in model_lower or "4.5" in model_lower:
            mapped = DATABRICKS_MODELS["opus-4-5"]
        elif "4-6" in model_lower or "4.6" in model_lower:
            mapped = DATABRICKS_MODELS["opus-4-6"]
        elif "4-7" in model_lower or "4.7" in model_lower:
            mapped = DATABRICKS_MODELS["opus-4-7"]
        else:
            mapped = DATABRICKS_MODELS["opus"]  # 默认最新版本
    elif "sonnet" in model_lower:
        if "4-5" in model_lower or "4.5" in model_lower:
            mapped = DATABRICKS_MODELS["sonnet-4-5"]
        elif "4-6" in model_lower or "4.6" in model_lower:
            mapped = DATABRICKS_MODELS["sonnet-4-6"]
        else:
            mapped = DATABRICKS_MODELS["sonnet"]  # 默认最新版本
    elif "haiku" in model_lower:
        mapped = DATABRICKS_MODELS["haiku"]
    else:
        logger.warning(f"Unknown model '{model}', using default: {DEFAULT_MODEL}")
        mapped = DEFAULT_MODEL

    if mapped != model:
        logger.info(f"Model mapping: {model} -> {mapped}")

    return mapped


def strip_cache_control_extras(body: dict) -> int:
    """Strip extra fields from cache_control, keeping only 'type'.
    Databricks 仅接受 cache_control: {"type": "ephemeral"}，
    但 Claude Code 会发送额外字段如 scope。
    返回被清理的 cache_control 对象数量。"""
    stripped_count = 0

    def _clean(obj: dict):
        nonlocal stripped_count
        cc = obj.get("cache_control")
        if isinstance(cc, dict):
            extra_keys = [k for k in cc if k != "type"]
            if extra_keys:
                for k in extra_keys:
                    del cc[k]
                stripped_count += 1

    # system blocks（system 为数组格式时）
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                _clean(block)

    # tools
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if isinstance(tool, dict):
                _clean(tool)

    # messages -> content blocks（含 tool_result 内嵌 content）
    if isinstance(body.get("messages"), list):
        for msg in body["messages"]:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        _clean(block)
                        if block.get("type") == "tool_result":
                            inner = block.get("content")
                            if isinstance(inner, list):
                                for item in inner:
                                    if isinstance(item, dict):
                                        _clean(item)

    return stripped_count


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
    total_cache_creation_tokens: int = field(default=0, repr=False)
    total_cache_read_tokens: int = field(default=0, repr=False)

    # 延迟追踪
    total_response_time: float = field(default=0.0, repr=False)
    successful_requests: int = field(default=0, repr=False)

    # 按模型维度统计: {"model_name": {"input_tokens": int, "output_tokens": int, "cache_creation_tokens": int, "cache_read_tokens": int, "requests": int}}
    model_stats: dict = field(default_factory=dict, repr=False)


@dataclass
class AzureOpenAIEndpoint:
    name: str
    endpoint: str   # 完整的 Azure 端点 URL，如 https://xxx.cognitiveservices.azure.com
    api_key: str
    deployments: list = field(default_factory=list)  # 可服务的模型/部署列表
    weight: int = 1

    # 运行时统计字段（与 WorkspaceEndpoint 保持一致）
    active_requests: int = field(default=0, repr=False)
    total_requests: int = field(default=0, repr=False)
    total_errors: int = field(default=0, repr=False)
    last_error_time: Optional[float] = field(default=None, repr=False)
    circuit_open: bool = field(default=False, repr=False)
    total_input_tokens: int = field(default=0, repr=False)
    total_output_tokens: int = field(default=0, repr=False)
    total_cache_creation_tokens: int = field(default=0, repr=False)
    total_cache_read_tokens: int = field(default=0, repr=False)
    total_response_time: float = field(default=0.0, repr=False)
    successful_requests: int = field(default=0, repr=False)
    model_stats: dict = field(default_factory=dict, repr=False)


@dataclass
class GlobalStats:
    total_requests: int = 0
    total_errors: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_response_time: float = 0.0
    successful_requests: int = 0
    start_time: float = field(default_factory=time.time)


# ==================== Usage Data Persistence ====================

class UsageDataStore:
    """基类：缓冲 + 定时刷盘框架，子类实现存储后端"""

    def __init__(self, retention_days: int = 0):
        self._buffer: list = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._today_cache: dict = {}
        self._today_date: Optional[date] = None
        self.retention_days = retention_days
        self._last_cleanup_date: Optional[date] = None

    async def start(self):
        try:
            await self._backend_start()
            today = date.today()
            self._today_date = today
            self._today_cache = await self._load_day(today)
            self._flush_task = asyncio.create_task(self._periodic_flush())
            if self.retention_days > 0:
                deleted = await self._delete_before(today - timedelta(days=self.retention_days))
                if deleted:
                    logger.info(f"Auto-cleanup: deleted {deleted} expired records (retention={self.retention_days}d)")
                self._last_cleanup_date = today
        except Exception as e:
            logger.error(f"Failed to initialize usage data store: {e}")

    async def stop(self):
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()
        await self._backend_stop()

    def record(self, model: str, input_tokens: int, output_tokens: int,
               cache_creation_tokens: int = 0, cache_read_tokens: int = 0,
               is_error: bool = False):
        self._buffer.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "is_error": is_error,
        })

    async def cleanup(self, keep_days: int) -> int:
        cutoff = date.today() - timedelta(days=keep_days)
        return await self._delete_before(cutoff)

    async def _periodic_flush(self):
        while True:
            await asyncio.sleep(30)
            try:
                await self._flush()
                if self.retention_days > 0:
                    today = date.today()
                    if self._last_cleanup_date != today:
                        deleted = await self._delete_before(today - timedelta(days=self.retention_days))
                        if deleted:
                            logger.info(f"Daily cleanup: deleted {deleted} expired records")
                        self._last_cleanup_date = today
            except Exception as e:
                logger.error(f"Usage data flush error: {e}")

    async def _flush(self):
        async with self._lock:
            pending, self._buffer = self._buffer, []
            if not pending:
                return

            today = date.today()
            if self._today_date != today:
                self._today_cache = await self._load_day(today)
                self._today_date = today

            if "models" not in self._today_cache:
                self._today_cache = {
                    "date": today.isoformat(),
                    "models": {},
                    "totals": {"input_tokens": 0, "output_tokens": 0,
                               "cache_creation_tokens": 0, "cache_read_tokens": 0,
                               "requests": 0, "errors": 0},
                }

            models = self._today_cache["models"]
            totals = self._today_cache["totals"]
            for delta in pending:
                m = delta["model"]
                if m not in models:
                    models[m] = {"input_tokens": 0, "output_tokens": 0,
                                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                                 "requests": 0}
                models[m]["input_tokens"] += delta["input_tokens"]
                models[m]["output_tokens"] += delta["output_tokens"]
                models[m]["cache_creation_tokens"] += delta["cache_creation_tokens"]
                models[m]["cache_read_tokens"] += delta["cache_read_tokens"]
                models[m]["requests"] += 1
                totals["input_tokens"] += delta["input_tokens"]
                totals["output_tokens"] += delta["output_tokens"]
                totals["cache_creation_tokens"] += delta["cache_creation_tokens"]
                totals["cache_read_tokens"] += delta["cache_read_tokens"]
                totals["requests"] += 1
                if delta.get("is_error"):
                    totals["errors"] += 1

            self._today_cache["last_updated"] = datetime.now().astimezone().isoformat()
            await self._save_day(today, self._today_cache)

    def get_today_data(self) -> dict:
        return self._today_cache if self._today_cache else {}

    async def get_day_data(self, d: date) -> Optional[dict]:
        if d == self._today_date:
            return self.get_today_data()
        return await self._load_day(d) or None

    # 子类实现
    async def _backend_start(self): pass
    async def _backend_stop(self): pass
    async def _save_day(self, d: date, data: dict): pass
    async def _load_day(self, d: date) -> dict: return {}
    async def _delete_before(self, cutoff: date) -> int: return 0


class JsonUsageStore(UsageDataStore):

    def __init__(self, base_dir: str, retention_days: int = 0):
        super().__init__(retention_days)
        self.base_dir = base_dir

    async def _backend_start(self):
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"JSON usage store initialized: {self.base_dir}")

    async def _backend_stop(self):
        pass

    def _day_path(self, d: date) -> str:
        return os.path.join(self.base_dir, str(d.year), f"{d.month:02d}", f"{d.isoformat()}.json")

    async def _save_day(self, d: date, data: dict):
        path = self._day_path(d)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except IOError as e:
            logger.error(f"Failed to write usage data {path}: {e}")

    async def _load_day(self, d: date) -> dict:
        path = self._day_path(d)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load usage data {path}: {e}")
            return {}

    async def _delete_before(self, cutoff: date) -> int:
        deleted = 0
        if not os.path.exists(self.base_dir):
            return 0
        for year_dir in sorted(os.listdir(self.base_dir)):
            year_path = os.path.join(self.base_dir, year_dir)
            if not os.path.isdir(year_path) or not year_dir.isdigit():
                continue
            for month_dir in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, month_dir)
                if not os.path.isdir(month_path):
                    continue
                for fname in sorted(os.listdir(month_path)):
                    if not fname.endswith(".json"):
                        continue
                    try:
                        file_date = date.fromisoformat(fname.replace(".json", ""))
                        if file_date < cutoff:
                            os.remove(os.path.join(month_path, fname))
                            deleted += 1
                    except ValueError:
                        continue
        return deleted


class MysqlUsageStore(UsageDataStore):

    def __init__(self, mysql_config: dict, retention_days: int = 0):
        super().__init__(retention_days)
        self._mysql_config = mysql_config
        self._pool = None

    async def _backend_start(self):
        import aiomysql
        self._pool = await aiomysql.create_pool(
            host=self._mysql_config.get("host", "localhost"),
            port=self._mysql_config.get("port", 3306),
            user=self._mysql_config.get("user", "root"),
            password=self._mysql_config.get("password", ""),
            db=self._mysql_config.get("database", "claude_lb"),
            minsize=1,
            maxsize=self._mysql_config.get("pool_size", 5),
            autocommit=True,
            charset="utf8mb4",
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS usage_daily (
                        date DATE NOT NULL,
                        model VARCHAR(128) NOT NULL,
                        input_tokens BIGINT NOT NULL DEFAULT 0,
                        output_tokens BIGINT NOT NULL DEFAULT 0,
                        cache_creation_tokens BIGINT NOT NULL DEFAULT 0,
                        cache_read_tokens BIGINT NOT NULL DEFAULT 0,
                        requests INT NOT NULL DEFAULT 0,
                        errors INT NOT NULL DEFAULT 0,
                        last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (date, model)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
        logger.info(f"MySQL usage store initialized: {self._mysql_config.get('host')}:{self._mysql_config.get('port')}/{self._mysql_config.get('database')}")

    async def _backend_stop(self):
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()

    async def _save_day(self, d: date, data: dict):
        if not self._pool:
            return
        models = data.get("models", {})
        if not models:
            return
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                for model_name, mstats in models.items():
                    await cur.execute("""
                        INSERT INTO usage_daily (date, model, input_tokens, output_tokens,
                            cache_creation_tokens, cache_read_tokens, requests, errors)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            input_tokens = VALUES(input_tokens),
                            output_tokens = VALUES(output_tokens),
                            cache_creation_tokens = VALUES(cache_creation_tokens),
                            cache_read_tokens = VALUES(cache_read_tokens),
                            requests = VALUES(requests),
                            errors = VALUES(errors)
                    """, (d.isoformat(), model_name,
                          mstats.get("input_tokens", 0), mstats.get("output_tokens", 0),
                          mstats.get("cache_creation_tokens", 0), mstats.get("cache_read_tokens", 0),
                          mstats.get("requests", 0), 0))

    async def _load_day(self, d: date) -> dict:
        if not self._pool:
            return {}
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT model, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, requests, errors FROM usage_daily WHERE date = %s",
                    (d.isoformat(),))
                rows = await cur.fetchall()
        if not rows:
            return {}
        models = {}
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0, "requests": 0, "errors": 0}
        for model, inp, out, cc, cr, reqs, errs in rows:
            models[model] = {"input_tokens": inp, "output_tokens": out, "cache_creation_tokens": cc, "cache_read_tokens": cr, "requests": reqs}
            totals["input_tokens"] += inp
            totals["output_tokens"] += out
            totals["cache_creation_tokens"] += cc
            totals["cache_read_tokens"] += cr
            totals["requests"] += reqs
            totals["errors"] += errs
        return {"date": d.isoformat(), "models": models, "totals": totals}

    async def _delete_before(self, cutoff: date) -> int:
        if not self._pool:
            return 0
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM usage_daily WHERE date < %s", (cutoff.isoformat(),))
                return cur.rowcount


def create_usage_store(storage_config: dict) -> UsageDataStore:
    store_type = storage_config.get("type", "json")
    retention = storage_config.get("retention_days", 0)
    if store_type == "mysql":
        return MysqlUsageStore(storage_config, retention)
    return JsonUsageStore(storage_config.get("path", "./usage_data"), retention)


class LoadBalancer:
    def __init__(
        self,
        endpoints: list,
        strategy: str = "least_requests",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60,
    ):
        self.endpoints = endpoints
        self.strategy = strategy
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self._rr_index = 0  # round_robin 计数器

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

        if len(available) == 1:
            return available[0]

        if self.strategy == "least_requests":
            # 加权最少请求: active_requests / weight 最小的优先
            # 平局时用 total_requests / weight 作为二级排序（均摊历史负载）
            # 仍然平局则随机打散，避免总是选第一个
            min_active = min(ep.active_requests / ep.weight for ep in available)
            # 筛选出 active_requests 最少的一组
            candidates = [ep for ep in available if ep.active_requests / ep.weight == min_active]
            if len(candidates) == 1:
                return candidates[0]
            # 在候选中选 total_requests / weight 最少的（均摊历史负载）
            min_total = min(ep.total_requests / ep.weight for ep in candidates)
            finalists = [ep for ep in candidates if ep.total_requests / ep.weight == min_total]
            return random.choice(finalists)

        elif self.strategy == "round_robin":
            # 加权轮询: 按 weight 展开
            total_weight = sum(ep.weight for ep in available)
            idx = self._rr_index % total_weight
            self._rr_index += 1
            cumulative = 0
            for ep in available:
                cumulative += ep.weight
                if idx < cumulative:
                    return ep
            return available[-1]

        else:  # random / weighted_random
            # 加权随机
            weights = [ep.weight for ep in available]
            return random.choices(available, weights=weights, k=1)[0]

    def select_endpoint_for_model(self, model: str):
        """从可用端点中筛选支持指定模型的端点，再按策略选择"""
        available = self.get_available_endpoints()
        matched = [ep for ep in available if model in ep.deployments]
        if not matched:
            return None

        if len(matched) == 1:
            return matched[0]

        if self.strategy == "least_requests":
            min_active = min(ep.active_requests / ep.weight for ep in matched)
            candidates = [ep for ep in matched if ep.active_requests / ep.weight == min_active]
            if len(candidates) == 1:
                return candidates[0]
            min_total = min(ep.total_requests / ep.weight for ep in candidates)
            finalists = [ep for ep in candidates if ep.total_requests / ep.weight == min_total]
            return random.choice(finalists)
        elif self.strategy == "round_robin":
            total_weight = sum(ep.weight for ep in matched)
            idx = self._rr_index % total_weight
            self._rr_index += 1
            cumulative = 0
            for ep in matched:
                cumulative += ep.weight
                if idx < cumulative:
                    return ep
            return matched[-1]
        else:
            weights = [ep.weight for ep in matched]
            return random.choices(matched, weights=weights, k=1)[0]

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
        endpoints_stats = []
        for ep in self.endpoints:
            stat = {
                "name": ep.name,
                "active_requests": ep.active_requests,
                "total_requests": ep.total_requests,
                "total_errors": ep.total_errors,
                "successful_requests": ep.successful_requests,
                "error_rate": round(ep.total_errors / ep.total_requests * 100, 2) if ep.total_requests > 0 else 0,
                "circuit_open": ep.circuit_open,
                "total_input_tokens": ep.total_input_tokens,
                "total_output_tokens": ep.total_output_tokens,
                "total_cache_creation_tokens": ep.total_cache_creation_tokens,
                "total_cache_read_tokens": ep.total_cache_read_tokens,
                "total_tokens": ep.total_input_tokens + ep.total_output_tokens,
                "avg_response_time_ms": round(ep.total_response_time / ep.successful_requests * 1000, 1) if ep.successful_requests > 0 else 0,
                "model_stats": ep.model_stats,
            }
            if hasattr(ep, "deployments"):
                stat["deployments"] = ep.deployments
            endpoints_stats.append(stat)
        return {"endpoints": endpoints_stats}


# ==================== Claude Proxy (使用原生 Anthropic 端点) ====================

class ClaudeProxy:
    def __init__(self, load_balancer: LoadBalancer, api_key: str):
        self.load_balancer = load_balancer
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        self.global_stats = GlobalStats()
        self.today_model_stats: dict = {}
        self.today_date: str = date.today().isoformat()

    def _record_usage(self, endpoint: WorkspaceEndpoint, model: str, input_tokens: int, output_tokens: int, elapsed: float,
                      cache_creation_tokens: int = 0, cache_read_tokens: int = 0):
        """记录 token 用量和延迟指标"""
        # 端点级别
        endpoint.total_input_tokens += input_tokens
        endpoint.total_output_tokens += output_tokens
        endpoint.total_cache_creation_tokens += cache_creation_tokens
        endpoint.total_cache_read_tokens += cache_read_tokens
        endpoint.total_response_time += elapsed
        endpoint.successful_requests += 1
        if model not in endpoint.model_stats:
            endpoint.model_stats[model] = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0, "requests": 0}
        endpoint.model_stats[model]["input_tokens"] += input_tokens
        endpoint.model_stats[model]["output_tokens"] += output_tokens
        endpoint.model_stats[model]["cache_creation_tokens"] += cache_creation_tokens
        endpoint.model_stats[model]["cache_read_tokens"] += cache_read_tokens
        endpoint.model_stats[model]["requests"] += 1
        # 全局级别
        self.global_stats.total_input_tokens += input_tokens
        self.global_stats.total_output_tokens += output_tokens
        self.global_stats.total_cache_creation_tokens += cache_creation_tokens
        self.global_stats.total_cache_read_tokens += cache_read_tokens
        self.global_stats.total_response_time += elapsed
        self.global_stats.successful_requests += 1
        self.global_stats.total_requests += 1
        # 当天 per-model 累加（跨 0 点自动滚动）
        today_iso = date.today().isoformat()
        if today_iso != self.today_date:
            self.today_model_stats = {}
            self.today_date = today_iso
        tm = self.today_model_stats.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0, "requests": 0})
        tm["input_tokens"] += input_tokens
        tm["output_tokens"] += output_tokens
        tm["cache_creation_tokens"] += cache_creation_tokens
        tm["cache_read_tokens"] += cache_read_tokens
        tm["requests"] += 1
        if usage_store:
            usage_store.record(model, input_tokens, output_tokens,
                               cache_creation_tokens, cache_read_tokens)

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
        
        # 移除 Databricks 不支持的顶层字段（如新版 Claude Code 发送的 context_management 等）
        unsupported_fields = ["context_management", "output_config"]
        for field in unsupported_fields:
            if field in body:
                logger.info(f"Removing unsupported field: {field}")
                del body[field]

        # 清理 tools 中 Databricks 不支持的嵌套字段
        # Claude Code 2.1.69+ 新增了 defer_loading, input_examples 等字段
        unsupported_tool_fields = ["defer_loading", "input_examples"]
        if "tools" in body and isinstance(body["tools"], list):
            cleaned_count = 0
            for tool in body["tools"]:
                if isinstance(tool, dict):
                    for tf in unsupported_tool_fields:
                        if tf in tool:
                            del tool[tf]
                            cleaned_count += 1
                        # 也检查 custom 子对象（错误信息指向 tools.0.custom.defer_loading）
                        if "custom" in tool and isinstance(tool["custom"], dict) and tf in tool["custom"]:
                            del tool["custom"][tf]
                            cleaned_count += 1
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} unsupported fields from tools")

        # 清理 messages 中 Databricks 不支持的 content type
        # Databricks 仅支持: document, image, search_result, text
        # Claude Code 2.1.69+ 新增了 tool_reference 等类型
        # 注意: tool_reference 可能嵌套在 tool_result.content 内部
        # 错误路径示例: messages.3.content.0.tool_result.content.0
        unsupported_content_types = {"tool_reference"}
        if "messages" in body and isinstance(body["messages"], list):
            for msg in body["messages"]:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    # 清理顶层 content blocks
                    original_len = len(content)
                    msg["content"] = [
                        block for block in content
                        if not (isinstance(block, dict) and block.get("type") in unsupported_content_types)
                    ]
                    removed = original_len - len(msg["content"])
                    if removed > 0:
                        logger.info(f"Removed {removed} unsupported content blocks from message")
                    # 清理 tool_result 内部嵌套的 content blocks
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            inner_content = block.get("content")
                            if isinstance(inner_content, list):
                                original_inner = len(inner_content)
                                block["content"] = [
                                    inner for inner in inner_content
                                    if not (isinstance(inner, dict) and inner.get("type") in unsupported_content_types)
                                ]
                                inner_removed = original_inner - len(block["content"])
                                if inner_removed > 0:
                                    logger.info(f"Removed {inner_removed} unsupported content blocks from tool_result")

        # 清理 cache_control 中 Databricks 不支持的额外字段
        # Claude Code 发送 {"type": "ephemeral", "scope": "turn"}，Databricks 仅接受 {"type": "ephemeral"}
        cc_cleaned = strip_cache_control_extras(body)
        if cc_cleaned > 0:
            logger.info(f"Stripped extra fields from {cc_cleaned} cache_control objects")

        # 处理 thinking 参数兼容性
        # 新模型 (Opus 4.6+, Sonnet 4.6+): 支持 adaptive（推荐），enabled + budget_tokens 已废弃
        # 旧模型 (Sonnet 4.5, Opus 4.5 等): 仅支持 enabled + budget_tokens
        if "thinking" in body and isinstance(body["thinking"], dict):
            thinking_type = body["thinking"].get("type")
            model = body.get("model", "")
            supports_adaptive = any(x in model for x in ["opus-4-6", "opus-4-7", "sonnet-4-6"])

            if supports_adaptive:
                # 新模型: 使用 adaptive，移除多余的 budget_tokens
                if thinking_type == "adaptive" and "budget_tokens" in body["thinking"]:
                    del body["thinking"]["budget_tokens"]
                    logger.info(f"Removed budget_tokens for adaptive thinking ({model})")
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
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
        cache_read_tokens = usage.get("cache_read_input_tokens", 0)
        self._record_usage(endpoint, model, input_tokens, output_tokens, elapsed,
                          cache_creation_tokens=cache_creation_tokens, cache_read_tokens=cache_read_tokens)

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
            cache_creation_tokens = 0
            cache_read_tokens = 0

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
                                                msg_usage = data.get("message", {}).get("usage", {})
                                                input_tokens = msg_usage.get("input_tokens", 0)
                                                cache_creation_tokens = msg_usage.get("cache_creation_input_tokens", 0)
                                                cache_read_tokens = msg_usage.get("cache_read_input_tokens", 0)
                                            elif data.get("type") == "message_delta":
                                                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                            if len(buffer) > 65536:
                                buffer = ""  # 防止内存泄漏
                        except Exception:
                            pass  # 指标提取绝不能阻断流

                    success = True
                    await proxy_self.load_balancer.on_request_end(current_endpoint, success=True)
                    elapsed = time.time() - start_time
                    proxy_self._record_usage(current_endpoint, model, input_tokens, output_tokens, elapsed,
                                            cache_creation_tokens=cache_creation_tokens, cache_read_tokens=cache_read_tokens)
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
                "total_cache_creation_tokens": gs.total_cache_creation_tokens,
                "total_cache_read_tokens": gs.total_cache_read_tokens,
                "total_tokens": gs.total_input_tokens + gs.total_output_tokens,
                "avg_response_time_ms": round(gs.total_response_time / gs.successful_requests * 1000, 1) if gs.successful_requests > 0 else 0,
                "requests_per_minute": round(gs.total_requests / (uptime / 60), 2) if uptime > 0 else 0,
            },
            **self.load_balancer.get_stats(),
        }


# ==================== Azure OpenAI Proxy ====================

class AzureOpenAIProxy:
    def __init__(self, load_balancer: LoadBalancer, api_key: str):
        self.load_balancer = load_balancer
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        self.global_stats = GlobalStats()

    def verify_api_key(self, key: str) -> bool:
        return key == self.api_key

    async def close(self):
        await self.client.aclose()

    def _record_usage(self, endpoint: AzureOpenAIEndpoint, model: str, input_tokens: int, output_tokens: int, elapsed: float,
                      cache_creation_tokens: int = 0, cache_read_tokens: int = 0):
        """记录 token 用量和延迟指标"""
        endpoint.total_input_tokens += input_tokens
        endpoint.total_output_tokens += output_tokens
        endpoint.total_cache_creation_tokens += cache_creation_tokens
        endpoint.total_cache_read_tokens += cache_read_tokens
        endpoint.total_response_time += elapsed
        endpoint.successful_requests += 1
        if model not in endpoint.model_stats:
            endpoint.model_stats[model] = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0, "requests": 0}
        endpoint.model_stats[model]["input_tokens"] += input_tokens
        endpoint.model_stats[model]["output_tokens"] += output_tokens
        endpoint.model_stats[model]["cache_creation_tokens"] += cache_creation_tokens
        endpoint.model_stats[model]["cache_read_tokens"] += cache_read_tokens
        endpoint.model_stats[model]["requests"] += 1
        self.global_stats.total_input_tokens += input_tokens
        self.global_stats.total_output_tokens += output_tokens
        self.global_stats.total_cache_creation_tokens += cache_creation_tokens
        self.global_stats.total_cache_read_tokens += cache_read_tokens
        self.global_stats.total_response_time += elapsed
        self.global_stats.successful_requests += 1
        self.global_stats.total_requests += 1
        if usage_store:
            usage_store.record(model, input_tokens, output_tokens,
                               cache_creation_tokens, cache_read_tokens)

    async def proxy_responses(self, body: dict, stream: bool = False):
        """代理 Azure OpenAI Responses API"""
        model = body.get("model", "unknown")
        max_retries = 3
        last_error = None
        start_time = time.time()

        for attempt in range(max_retries):
            endpoint = self.load_balancer.select_endpoint_for_model(model)
            if not endpoint:
                raise HTTPException(status_code=404, detail={"error": {"message": f"No endpoint available for model '{model}'"}})

            await self.load_balancer.on_request_start(endpoint)
            url = f"{endpoint.endpoint}/openai/v1/responses"
            headers = {
                "api-key": endpoint.api_key,
                "Content-Type": "application/json",
            }

            logger.info(f"[Azure Responses][{model}] -> {endpoint.name} (attempt {attempt + 1})")

            try:
                if stream:
                    return await self._stream_response(endpoint, url, body, headers, model, "responses", start_time)
                else:
                    return await self._normal_request(endpoint, url, body, headers, model, "responses", start_time)
            except httpx.HTTPStatusError as e:
                last_error = e
                self.global_stats.total_errors += 1
                is_client_error = 400 <= e.response.status_code < 500 and e.response.status_code != 429
                await self.load_balancer.on_request_end(endpoint, success=False, is_client_error=is_client_error)
                if e.response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"{endpoint.name} returned {e.response.status_code}, retrying...")
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                else:
                    try:
                        error_body = e.response.json()
                    except Exception:
                        error_body = {"error": {"message": e.response.text}}
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

    async def proxy_chat_completions(self, body: dict, stream: bool = False):
        """代理 Azure OpenAI Chat Completions API"""
        model = body.get("model", "unknown")
        max_retries = 3
        last_error = None
        start_time = time.time()

        for attempt in range(max_retries):
            endpoint = self.load_balancer.select_endpoint_for_model(model)
            if not endpoint:
                raise HTTPException(status_code=404, detail={"error": {"message": f"No endpoint available for model '{model}'"}})

            await self.load_balancer.on_request_start(endpoint)
            url = f"{endpoint.endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-10-21"
            headers = {
                "api-key": endpoint.api_key,
                "Content-Type": "application/json",
            }

            # 流式请求注入 stream_options 以获取 usage
            if stream and "stream_options" not in body:
                body["stream_options"] = {"include_usage": True}

            logger.info(f"[Azure Chat][{model}] -> {endpoint.name} (attempt {attempt + 1})")

            try:
                if stream:
                    return await self._stream_response(endpoint, url, body, headers, model, "chat", start_time)
                else:
                    return await self._normal_request(endpoint, url, body, headers, model, "chat", start_time)
            except httpx.HTTPStatusError as e:
                last_error = e
                self.global_stats.total_errors += 1
                is_client_error = 400 <= e.response.status_code < 500 and e.response.status_code != 429
                await self.load_balancer.on_request_end(endpoint, success=False, is_client_error=is_client_error)
                if e.response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"{endpoint.name} returned {e.response.status_code}, retrying...")
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                else:
                    try:
                        error_body = e.response.json()
                    except Exception:
                        error_body = {"error": {"message": e.response.text}}
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

    async def _normal_request(self, endpoint, url, body, headers, model: str, api_type: str, start_time: float) -> JSONResponse:
        """非流式请求"""
        response = await self.client.post(url, json=body, headers=headers)
        response.raise_for_status()
        await self.load_balancer.on_request_end(endpoint, success=True)

        elapsed = time.time() - start_time
        resp_json = response.json()
        usage = resp_json.get("usage", {})
        if api_type == "chat":
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cache_read_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        else:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        self._record_usage(endpoint, model, input_tokens, output_tokens, elapsed,
                          cache_read_tokens=cache_read_tokens)

        return JSONResponse(content=resp_json, status_code=response.status_code)

    async def _stream_response(self, endpoint, url, body, headers, model: str, api_type: str, start_time: float) -> StreamingResponse:
        """流式请求 - 透传 SSE，支持流内重试"""

        proxy_self = self
        max_retries = 3

        async def stream_generator():
            current_endpoint = endpoint
            current_url = url
            current_headers = headers
            input_tokens = 0
            output_tokens = 0
            cache_read_tokens = 0

            for attempt in range(max_retries):
                try:
                    req = proxy_self.client.build_request("POST", current_url, json=body, headers=current_headers)
                    response = await proxy_self.client.send(req, stream=True)

                    if response.status_code >= 400:
                        error_body = await response.aread()
                        is_client_error = 400 <= response.status_code < 500 and response.status_code != 429
                        try:
                            error_json = json.loads(error_body)
                            error_msg = json.dumps(error_json)
                        except Exception:
                            error_msg = error_body.decode("utf-8") if isinstance(error_body, bytes) else str(error_body)

                        logger.error(f"Azure stream failed ({response.status_code}): {error_msg}")
                        await proxy_self.load_balancer.on_request_end(current_endpoint, success=False, is_client_error=is_client_error)

                        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                            logger.warning(f"{current_endpoint.name} returned {response.status_code}, retrying stream...")
                            await asyncio.sleep(min(2 ** attempt, 8))
                            new_endpoint = proxy_self.load_balancer.select_endpoint_for_model(model)
                            if new_endpoint:
                                current_endpoint = new_endpoint
                                if api_type == "responses":
                                    current_url = f"{current_endpoint.endpoint}/openai/v1/responses"
                                else:
                                    current_url = f"{current_endpoint.endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-10-21"
                                current_headers = {"api-key": current_endpoint.api_key, "Content-Type": "application/json"}
                                await proxy_self.load_balancer.on_request_start(current_endpoint)
                                continue

                        yield f"data: {json.dumps({'error': {'message': error_msg}})}\n\n".encode()
                        return

                    # 透传响应，同时嗅探 usage
                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        try:
                            buffer += chunk.decode("utf-8", errors="ignore")
                            while "\n\n" in buffer:
                                event_str, buffer = buffer.split("\n\n", 1)
                                for line in event_str.split("\n"):
                                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                                        continue
                                    data = json.loads(line[6:])
                                    if api_type == "responses":
                                        # Responses API: usage 在 response.completed 事件中
                                        if data.get("type") == "response.completed":
                                            usage = data.get("response", {}).get("usage", {})
                                            input_tokens = usage.get("input_tokens", 0)
                                            output_tokens = usage.get("output_tokens", 0)
                                            cache_read_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                                    else:
                                        # Chat Completions: usage 在末尾 chunk
                                        usage = data.get("usage")
                                        if usage:
                                            input_tokens = usage.get("prompt_tokens", 0)
                                            output_tokens = usage.get("completion_tokens", 0)
                                            cache_read_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                if len(buffer) > 65536:
                                    buffer = ""
                        except Exception:
                            pass

                    await proxy_self.load_balancer.on_request_end(current_endpoint, success=True)
                    elapsed = time.time() - start_time
                    proxy_self._record_usage(current_endpoint, model, input_tokens, output_tokens, elapsed,
                                            cache_read_tokens=cache_read_tokens)
                    return

                except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
                    error_detail = f"{type(e).__name__}: {str(e) or 'Unknown error'}"
                    logger.error(f"Azure stream network error on {current_endpoint.name}: {error_detail}")
                    await proxy_self.load_balancer.on_request_end(current_endpoint, success=False)

                    if attempt < max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 8))
                        new_endpoint = proxy_self.load_balancer.select_endpoint_for_model(model)
                        if new_endpoint:
                            current_endpoint = new_endpoint
                            if api_type == "responses":
                                current_url = f"{current_endpoint.endpoint}/openai/v1/responses"
                            else:
                                current_url = f"{current_endpoint.endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-10-21"
                            current_headers = {"api-key": current_endpoint.api_key, "Content-Type": "application/json"}
                            await proxy_self.load_balancer.on_request_start(current_endpoint)
                            continue

                    yield f"data: {json.dumps({'error': {'message': error_detail}})}\n\n".encode()
                    return

                except Exception as e:
                    error_detail = f"{type(e).__name__}: {str(e) or 'Unknown error'}"
                    logger.error(f"Azure stream error: {error_detail}")
                    await proxy_self.load_balancer.on_request_end(current_endpoint, success=False)
                    yield f"data: {json.dumps({'error': {'message': error_detail}})}\n\n".encode()
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
                "total_cache_creation_tokens": gs.total_cache_creation_tokens,
                "total_cache_read_tokens": gs.total_cache_read_tokens,
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


def load_config(config_path: str = "config.yaml") -> tuple:
    """返回 (ClaudeProxy, Optional[AzureOpenAIProxy])"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    lb_config = config.get("load_balancer", {})
    api_key = expand_env_vars(config.get("auth", {}).get("api_key", ""))

    # Databricks 端点
    endpoints = []
    for ep in config.get("endpoints", []):
        endpoints.append(WorkspaceEndpoint(
            name=ep["name"],
            api_base=ep["api_base"],
            token=expand_env_vars(ep["token"]),
            weight=ep.get("weight", 1),
        ))
        logger.info(f"Loaded Databricks endpoint: {ep['name']}")

    load_balancer = LoadBalancer(
        endpoints=endpoints,
        strategy=lb_config.get("strategy", "least_requests"),
        circuit_breaker_threshold=lb_config.get("circuit_breaker_threshold", 5),
        circuit_breaker_timeout=lb_config.get("circuit_breaker_timeout", 60),
    )
    claude_proxy = ClaudeProxy(load_balancer, api_key)

    # Azure OpenAI 端点（可选）
    azure_proxy = None
    azure_config = config.get("azure_openai")
    if azure_config:
        azure_lb_config = azure_config.get("load_balancer", lb_config)
        azure_endpoints = []
        for ep in azure_config.get("endpoints", []):
            azure_endpoints.append(AzureOpenAIEndpoint(
                name=ep["name"],
                endpoint=ep["endpoint"],
                api_key=expand_env_vars(ep["api_key"]),
                deployments=ep.get("deployments", []),
                weight=ep.get("weight", 1),
            ))
            logger.info(f"Loaded Azure OpenAI endpoint: {ep['name']} (deployments: {ep.get('deployments', [])})")

        azure_lb = LoadBalancer(
            endpoints=azure_endpoints,
            strategy=azure_lb_config.get("strategy", "least_requests"),
            circuit_breaker_threshold=azure_lb_config.get("circuit_breaker_threshold", 5),
            circuit_breaker_timeout=azure_lb_config.get("circuit_breaker_timeout", 60),
        )
        azure_proxy = AzureOpenAIProxy(azure_lb, api_key)

    # 存储配置：usage_storage 优先，否则回退到 usage_data_dir
    storage_config = config.get("usage_storage")
    if storage_config:
        if isinstance(storage_config, dict):
            for k in ["password", "path"]:
                if k in storage_config and isinstance(storage_config[k], str):
                    storage_config[k] = expand_env_vars(storage_config[k])
        else:
            storage_config = {"type": "json", "path": "./usage_data"}
    else:
        storage_config = {"type": "json", "path": expand_env_vars(config.get("usage_data_dir", "./usage_data"))}
    return claude_proxy, azure_proxy, storage_config


# ==================== FastAPI App ====================

proxy: Optional[ClaudeProxy] = None
azure_proxy: Optional[AzureOpenAIProxy] = None
usage_store: Optional[UsageDataStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy, azure_proxy, usage_store
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    proxy, azure_proxy, storage_config = load_config(config_path)

    usage_store = create_usage_store(storage_config)
    await usage_store.start()

    restore = usage_store.get_today_data()
    if restore and restore.get("models"):
        for model_name, mstats in restore["models"].items():
            proxy.global_stats.total_input_tokens += mstats.get("input_tokens", 0)
            proxy.global_stats.total_output_tokens += mstats.get("output_tokens", 0)
            proxy.global_stats.total_cache_creation_tokens += mstats.get("cache_creation_tokens", 0)
            proxy.global_stats.total_cache_read_tokens += mstats.get("cache_read_tokens", 0)
            proxy.global_stats.total_requests += mstats.get("requests", 0)
            proxy.global_stats.successful_requests += mstats.get("requests", 0)
            # 按模型恢复当天快照（供 /stats 计算 KPI Est. Cost 及 Anthropic Models 表使用）
            proxy.today_model_stats[model_name] = {
                "input_tokens": mstats.get("input_tokens", 0),
                "output_tokens": mstats.get("output_tokens", 0),
                "cache_creation_tokens": mstats.get("cache_creation_tokens", 0),
                "cache_read_tokens": mstats.get("cache_read_tokens", 0),
                "requests": mstats.get("requests", 0),
            }
        if restore.get("totals", {}).get("errors"):
            proxy.global_stats.total_errors += restore["totals"]["errors"]
        logger.info(f"Restored today's usage data: {restore['totals'].get('requests', 0)} requests")

    logger.info(f"Proxy started with {len(proxy.load_balancer.endpoints)} Databricks endpoints")
    logger.info(f"Usage storage: {storage_config.get('type', 'json')}")
    if azure_proxy:
        logger.info(f"Azure OpenAI proxy enabled with {len(azure_proxy.load_balancer.endpoints)} endpoints")
    yield
    if usage_store:
        await usage_store.stop()
    if proxy:
        await proxy.close()
    if azure_proxy:
        await azure_proxy.close()


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


def _extract_api_key(request: Request, x_api_key: Optional[str] = None) -> str:
    auth_header = request.headers.get("authorization", "")
    return x_api_key or (auth_header[7:] if auth_header.startswith("Bearer ") else "")


@app.post("/v1/responses")
async def responses(request: Request, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    if not azure_proxy:
        raise HTTPException(status_code=404, detail={"error": {"message": "Azure OpenAI is not configured"}})

    actual_key = _extract_api_key(request, x_api_key)
    if not azure_proxy.verify_api_key(actual_key):
        raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API key"}})

    body = await request.json()
    stream = body.get("stream", False)
    logger.info(f"[Azure Responses] model={body.get('model')}, stream={stream}")
    return await azure_proxy.proxy_responses(body, stream=stream)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    if not azure_proxy:
        raise HTTPException(status_code=404, detail={"error": {"message": "Azure OpenAI is not configured"}})

    actual_key = _extract_api_key(request, x_api_key)
    if not azure_proxy.verify_api_key(actual_key):
        raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API key"}})

    body = await request.json()
    stream = body.get("stream", False)
    logger.info(f"[Azure Chat] model={body.get('model')}, stream={stream}")
    return await azure_proxy.proxy_chat_completions(body, stream=stream)


@app.post("/api/event_logging/batch")
async def event_logging():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/stats")
async def stats():
    result = proxy.get_stats()
    # 端点级 per-model Est. Cost 仍反映本次会话内的负载分布
    for ep in result.get("endpoints", []):
        for model_name, mstats in (ep.get("model_stats") or {}).items():
            cost = calculate_cost(model_name, mstats["input_tokens"], mstats["output_tokens"],
                                  mstats.get("cache_creation_tokens", 0), mstats.get("cache_read_tokens", 0))
            if cost is not None:
                mstats["estimated_cost_usd"] = cost
    # KPI Est. Cost 和 Anthropic Models 表格统一以当天（per-model）累计为准
    today_models = {}
    total_cost = 0.0
    for model_name, mstats in (proxy.today_model_stats or {}).items():
        entry = dict(mstats)
        cost = calculate_cost(model_name, mstats["input_tokens"], mstats["output_tokens"],
                              mstats.get("cache_creation_tokens", 0), mstats.get("cache_read_tokens", 0))
        if cost is not None:
            entry["estimated_cost_usd"] = cost
            total_cost += cost
        today_models[model_name] = entry
    result["today_model_stats"] = today_models
    result["today_date"] = proxy.today_date
    result["global"]["estimated_total_cost_usd"] = round(total_cost, 4)
    if azure_proxy:
        result["azure_openai"] = azure_proxy.get_stats()
    return result


@app.get("/stats/history")
async def stats_history(days: int = 7):
    if not usage_store:
        return {"error": "Usage data persistence is not configured"}
    end = date.today()
    start = end - timedelta(days=days - 1)
    history = []
    current = start
    while current <= end:
        day_data = await usage_store.get_day_data(current)
        if day_data and day_data.get("models"):
            daily_cost = 0.0
            for model_name, mstats in day_data["models"].items():
                cost = calculate_cost(model_name, mstats.get("input_tokens", 0), mstats.get("output_tokens", 0),
                                      mstats.get("cache_creation_tokens", 0), mstats.get("cache_read_tokens", 0))
                if cost is not None:
                    mstats["estimated_cost_usd"] = cost
                    daily_cost += cost
            day_data["estimated_total_cost_usd"] = round(daily_cost, 4)
            history.append(day_data)
        else:
            history.append({"date": current.isoformat(), "models": {}, "totals": {}, "estimated_total_cost_usd": 0})
        current += timedelta(days=1)
    return {"history": history, "days": days}


@app.delete("/stats/history")
async def delete_history(keep_days: int = 30):
    if not usage_store:
        return {"error": "Usage data persistence is not configured"}
    deleted = await usage_store.cleanup(keep_days)
    return {"deleted": deleted, "keep_days": keep_days}


@app.post("/reset")
async def reset():
    for ep in proxy.load_balancer.endpoints:
        ep.circuit_open = False
        ep.total_errors = 0
        ep.total_input_tokens = 0
        ep.total_output_tokens = 0
        ep.total_cache_creation_tokens = 0
        ep.total_cache_read_tokens = 0
        ep.total_response_time = 0.0
        ep.successful_requests = 0
        ep.model_stats = {}
    proxy.global_stats = GlobalStats()
    proxy.today_model_stats = {}
    proxy.today_date = date.today().isoformat()
    if azure_proxy:
        for ep in azure_proxy.load_balancer.endpoints:
            ep.circuit_open = False
            ep.total_errors = 0
            ep.total_input_tokens = 0
            ep.total_output_tokens = 0
            ep.total_cache_creation_tokens = 0
            ep.total_cache_read_tokens = 0
            ep.total_response_time = 0.0
            ep.successful_requests = 0
            ep.model_stats = {}
        azure_proxy.global_stats = GlobalStats()
    if usage_store:
        await usage_store._flush()
    return {"status": "reset", "note": "In-memory stats reset. Persisted usage data preserved."}


@app.get("/stats/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Databricks Claude LB Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
@property --angle { syntax: '<angle>'; initial-value: 0deg; inherits: false; }
:root {
  color-scheme: dark;
  --bg-0: #0b1020;
  --bg-1: #0f172a;
  --surface-1: #111a2e;
  --surface-2: #17223b;
  --border: #1f2a44;
  --border-hi: #334155;
  --text-0: #f1f5f9;
  --text-1: #cbd5e1;
  --text-2: #94a3b8;
  --text-3: #64748b;
  --accent: #60a5fa;
  --accent-2: #c084fc;
  --ok: #4ade80;
  --warn: #fbbf24;
  --err: #f87171;
  --radius: 14px;
  --radius-sm: 10px;
  --ease: cubic-bezier(.4,0,.2,1);
  --bg-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(120, 119, 198, 0.22), transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 10%, rgba(96, 165, 250, 0.12), transparent 70%),
    radial-gradient(ellipse 50% 40% at 10% 0%, rgba(192, 132, 252, 0.10), transparent 70%);
  --card-bg: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(17, 26, 46, 0.6));
  --chart-card-bg: linear-gradient(135deg, rgba(30, 41, 59, 0.55), rgba(17, 26, 46, 0.55));
  --table-bg: rgba(17, 26, 46, 0.5);
  --th-bg: rgba(15, 23, 42, 0.7);
  --row-hover: rgba(96, 165, 250, 0.06);
  --tag-bg: rgba(30, 41, 59, 0.5);
  --pill-bg: rgba(17, 26, 46, 0.6);
  --input-bg: rgba(15, 23, 42, 0.6);
  --info-bg: rgba(17, 26, 46, 0.5);
  --card-hover-glow: rgba(96, 165, 250, 0.10);
  --chart-border: rgba(15, 23, 42, 0.8);
  --chart-grid: rgba(51, 65, 85, 0.25);
  --chart-grid-strong: rgba(51, 65, 85, 0.4);
  --tooltip-bg: rgba(15, 23, 42, 0.95);
  --tooltip-border: #334155;
  --deployments-bg: rgba(96, 165, 250, 0.1);
  --deployments-border: rgba(96, 165, 250, 0.2);
  --err-banner-bg: linear-gradient(135deg, rgba(127, 29, 29, 0.35), rgba(127, 29, 29, 0.18));
  --err-banner-border: rgba(248, 113, 113, 0.25);
  --err-banner-text: #fca5a5;
  --btn-primary-bg: linear-gradient(135deg, #3b82f6, #6366f1);
  --btn-primary-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
  --btn-danger-bg: rgba(239, 68, 68, 0.15);
  --btn-danger-hover: rgba(239, 68, 68, 0.25);
  --btn-danger-border: rgba(239, 68, 68, 0.3);
  --btn-danger-text: #fca5a5;
  --brand-gradient: linear-gradient(90deg, #e2e8f0 0%, #60a5fa 30%, #c084fc 55%, #60a5fa 80%, #e2e8f0 100%);
  --alert-bar: linear-gradient(180deg, var(--err), #fb923c);
  --alert-glow: 0 0 8px rgba(248, 113, 113, 0.6);
  --tab-indicator-glow: 0 0 12px rgba(96, 165, 250, 0.5);
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg-0: #f8fafc;
  --bg-1: #f5f7fb;
  --surface-1: #ffffff;
  --surface-2: #f1f5f9;
  --border: #e2e8f0;
  --border-hi: #cbd5e1;
  --text-0: #0f172a;
  --text-1: #1e293b;
  --text-2: #475569;
  --text-3: #94a3b8;
  --accent: #2563eb;
  --accent-2: #7c3aed;
  --ok: #16a34a;
  --warn: #d97706;
  --err: #dc2626;
  --bg-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99, 102, 241, 0.14), transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 10%, rgba(37, 99, 235, 0.10), transparent 70%),
    radial-gradient(ellipse 50% 40% at 10% 0%, rgba(124, 58, 237, 0.09), transparent 70%);
  --card-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(248, 250, 252, 0.85));
  --chart-card-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(248, 250, 252, 0.88));
  --table-bg: rgba(255, 255, 255, 0.72);
  --th-bg: rgba(241, 245, 249, 0.9);
  --row-hover: rgba(37, 99, 235, 0.06);
  --tag-bg: rgba(255, 255, 255, 0.85);
  --pill-bg: rgba(255, 255, 255, 0.8);
  --input-bg: rgba(255, 255, 255, 0.92);
  --info-bg: rgba(255, 255, 255, 0.7);
  --card-hover-glow: rgba(37, 99, 235, 0.08);
  --chart-border: rgba(255, 255, 255, 0.95);
  --chart-grid: rgba(148, 163, 184, 0.25);
  --chart-grid-strong: rgba(148, 163, 184, 0.35);
  --tooltip-bg: rgba(15, 23, 42, 0.92);
  --tooltip-border: #475569;
  --deployments-bg: rgba(37, 99, 235, 0.08);
  --deployments-border: rgba(37, 99, 235, 0.2);
  --err-banner-bg: linear-gradient(135deg, rgba(254, 226, 226, 0.85), rgba(254, 242, 242, 0.65));
  --err-banner-border: rgba(220, 38, 38, 0.3);
  --err-banner-text: #991b1b;
  --btn-primary-bg: linear-gradient(135deg, #2563eb, #6366f1);
  --btn-primary-shadow: 0 4px 12px rgba(37, 99, 235, 0.22);
  --btn-danger-bg: rgba(239, 68, 68, 0.08);
  --btn-danger-hover: rgba(239, 68, 68, 0.16);
  --btn-danger-border: rgba(239, 68, 68, 0.28);
  --btn-danger-text: #b91c1c;
  --brand-gradient: linear-gradient(90deg, #0f172a 0%, #2563eb 30%, #7c3aed 55%, #2563eb 80%, #0f172a 100%);
  --alert-bar: linear-gradient(180deg, var(--err), #f97316);
  --alert-glow: 0 0 8px rgba(220, 38, 38, 0.4);
  --tab-indicator-glow: 0 0 10px rgba(37, 99, 235, 0.35);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: var(--text-1);
  background: var(--bg-1);
  background-image: var(--bg-image);
  background-attachment: fixed;
  padding: 24px clamp(16px, 3vw, 32px) 48px;
  font-feature-settings: 'cv11', 'ss01';
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
  transition: background-color .3s var(--ease), color .3s var(--ease);
}
.hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.brand {
  display: flex;
  align-items: baseline;
  gap: 14px;
  flex-wrap: wrap;
}
.brand h1 {
  font-size: clamp(1.35rem, 2.4vw, 1.7rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  background: var(--brand-gradient);
  background-size: 250% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: shimmer 6s linear infinite;
}
.brand .tag {
  font-size: .72rem;
  font-weight: 500;
  color: var(--text-2);
  padding: 3px 10px;
  border: 1px solid var(--border-hi);
  border-radius: 9999px;
  background: var(--tag-bg);
  backdrop-filter: blur(8px);
}
.hero-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px; height: 38px;
  border: 1px solid var(--border-hi);
  border-radius: 9999px;
  background: var(--pill-bg);
  backdrop-filter: blur(8px);
  color: var(--text-1);
  cursor: pointer;
  transition: color .2s var(--ease), border-color .2s var(--ease), transform .2s var(--ease), background-color .3s var(--ease);
  font-family: inherit;
}
.theme-toggle:hover { color: var(--accent); border-color: var(--accent); transform: translateY(-1px); }
.theme-toggle:active { transform: scale(0.94); }
.theme-toggle svg { width: 18px; height: 18px; }
.theme-toggle .icon-sun { display: none; }
.theme-toggle .icon-moon { display: block; }
:root[data-theme="light"] .theme-toggle .icon-sun { display: block; }
:root[data-theme="light"] .theme-toggle .icon-moon { display: none; }
@keyframes shimmer {
  from { background-position: 250% 0; }
  to { background-position: -250% 0; }
}
.refresh-pill {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border: 1px solid var(--border-hi);
  border-radius: 9999px;
  background: var(--pill-bg);
  backdrop-filter: blur(8px);
  font-size: .78rem;
  color: var(--text-2);
  transition: background-color .3s var(--ease);
}
.refresh-pill svg { width: 18px; height: 18px; }
.refresh-pill .ring-bg { fill: none; stroke: var(--border-hi); stroke-width: 3; }
.refresh-pill .ring-fg { fill: none; stroke: var(--accent); stroke-width: 3; stroke-linecap: round; transform: rotate(-90deg); transform-origin: center; transition: stroke-dashoffset 1s linear; }
.refresh-pill.paused .ring-fg { stroke: var(--text-3); }

.error-banner {
  background: var(--err-banner-bg);
  border: 1px solid var(--err-banner-border);
  color: var(--err-banner-text);
  padding: 12px 14px;
  border-radius: var(--radius-sm);
  margin-bottom: 16px;
  display: none;
  font-size: .88rem;
}
.error-banner.show { display: block; animation: fadeIn .2s var(--ease); }

.tab-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  position: relative;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
  overflow-x: auto;
  scrollbar-width: none;
}
.tab-bar::-webkit-scrollbar { display: none; }
.tab-btn {
  background: none; border: none;
  padding: 12px 20px;
  font-size: .9rem;
  font-weight: 600;
  color: var(--text-3);
  cursor: pointer;
  font-family: inherit;
  transition: color .2s var(--ease);
  white-space: nowrap;
  position: relative;
}
.tab-btn:hover { color: var(--text-1); }
.tab-btn.active { color: var(--accent); }
.tab-indicator {
  position: absolute;
  bottom: -1px;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border-radius: 2px;
  transition: transform .35s var(--ease), width .35s var(--ease);
  pointer-events: none;
  box-shadow: var(--tab-indicator-glow);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; animation: fadeIn .25s var(--ease); }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

.section-title {
  font-size: .95rem;
  font-weight: 600;
  color: var(--text-1);
  margin: 28px 0 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  letter-spacing: -0.01em;
}
.section-title::before {
  content: '';
  width: 3px; height: 14px;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
  border-radius: 2px;
}

.global-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.card {
  position: relative;
  background: var(--card-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  overflow: hidden;
  transition: border-color .2s var(--ease), transform .2s var(--ease), background-color .3s var(--ease);
}
.card::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(420px circle at var(--mx, 50%) var(--my, 50%), var(--card-hover-glow), transparent 45%);
  opacity: 0;
  transition: opacity .25s var(--ease);
  pointer-events: none;
}
.card:hover { border-color: var(--border-hi); transform: translateY(-1px); }
.card:hover::before { opacity: 1; }
.card .label {
  font-size: .68rem;
  color: var(--text-2);
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 600;
}
.card .value {
  font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
  font-size: 1.55rem;
  font-weight: 600;
  margin-top: 6px;
  color: var(--text-0);
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}
.card .value.green { color: var(--ok); }
.card .value.blue { color: var(--accent); }
.card .value.amber { color: var(--warn); }
.card .value.purple { color: var(--accent-2); }

.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}
@media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

.chart-card {
  position: relative;
  background: var(--chart-card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
  min-height: 260px;
  transition: background-color .3s var(--ease);
}
.chart-card h3 {
  font-size: .82rem;
  font-weight: 600;
  color: var(--text-2);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: .06em;
}
.chart-card .canvas-wrap { position: relative; height: 230px; }

.table-wrap {
  background: var(--table-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 16px;
  transition: background-color .3s var(--ease);
}
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .84rem; }
th {
  background: var(--th-bg);
  padding: 11px 14px;
  text-align: left;
  color: var(--text-2);
  font-weight: 600;
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .06em;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
td {
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text-1);
  font-variant-numeric: tabular-nums;
}
tr:last-child td { border-bottom: none; }
tr.row { transition: background .15s var(--ease); }
tr.row:hover { background: var(--row-hover); }
tr.row.alert { position: relative; }
tr.row.alert td:first-child { position: relative; }
tr.row.alert td:first-child::before {
  content: ''; position: absolute; left: 0; top: 12%; bottom: 12%;
  width: 3px; border-radius: 2px;
  background: var(--alert-bar);
  box-shadow: var(--alert-glow);
  animation: pulseAlert 1.8s ease-in-out infinite;
}
@keyframes pulseAlert { 0%, 100% { opacity: .6; } 50% { opacity: 1; } }

td.mono, .mono { font-family: 'JetBrains Mono', monospace; font-size: .8rem; }
td strong { color: var(--text-0); font-weight: 600; }

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: .7rem;
  font-weight: 600;
  letter-spacing: .02em;
}
.badge::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
}
.badge.ok { background: color-mix(in srgb, var(--ok) 12%, transparent); color: var(--ok); border: 1px solid color-mix(in srgb, var(--ok) 28%, transparent); }
.badge.ok::before { background: var(--ok); box-shadow: 0 0 8px var(--ok); }
.badge.open { background: color-mix(in srgb, var(--err) 14%, transparent); color: var(--err); border: 1px solid color-mix(in srgb, var(--err) 32%, transparent); }
.badge.open::before { background: var(--err); box-shadow: 0 0 8px var(--err); animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: .3; } }

.err-rate { display: inline-flex; align-items: center; gap: 8px; }
.err-rate .bar { flex: 0 0 42px; height: 5px; background: var(--border); border-radius: 9999px; overflow: hidden; }
.err-rate .bar > span { display: block; height: 100%; border-radius: 9999px; transition: width .4s var(--ease), background .2s var(--ease); }

.deployments-tag {
  display: inline-block;
  font-size: .7rem;
  padding: 2px 8px;
  border-radius: 6px;
  background: var(--deployments-bg);
  color: var(--accent);
  border: 1px solid var(--deployments-border);
  margin-right: 4px;
  margin-bottom: 2px;
}

.info-box {
  background: var(--info-bg);
  border: 1px dashed var(--border-hi);
  border-radius: var(--radius);
  padding: 36px;
  text-align: center;
  color: var(--text-2);
  font-size: .92rem;
}
.info-box code { font-family: 'JetBrains Mono', monospace; font-size: .85rem; color: var(--accent); background: color-mix(in srgb, var(--accent) 12%, transparent); padding: 1px 6px; border-radius: 4px; }

.history-toolbar {
  display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap;
  padding: 12px 16px;
  background: var(--table-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: background-color .3s var(--ease);
}
.history-toolbar label { color: var(--text-2); font-size: .82rem; }
.history-toolbar input[type=number] {
  width: 72px;
  padding: 7px 10px;
  background: var(--input-bg);
  border: 1px solid var(--border-hi);
  border-radius: 8px;
  color: var(--text-0);
  font-size: .85rem;
  font-family: 'JetBrains Mono', monospace;
  outline: none;
  transition: border-color .15s var(--ease);
}
.history-toolbar input[type=number]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent); }
.btn {
  padding: 7px 16px;
  border: 1px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  font-size: .82rem;
  font-weight: 600;
  font-family: inherit;
  transition: transform .1s var(--ease), filter .15s var(--ease), background-color .2s var(--ease);
}
.btn:active { transform: scale(0.97); }
.btn-primary { background: var(--btn-primary-bg); color: #fff; box-shadow: var(--btn-primary-shadow); }
.btn-primary:hover { filter: brightness(1.1); }
.btn-danger { background: var(--btn-danger-bg); color: var(--btn-danger-text); border-color: var(--btn-danger-border); }
.btn-danger:hover { background: var(--btn-danger-hover); }

@media (max-width: 768px) {
  body { padding: 16px 12px 32px; }
  .global-grid { grid-template-columns: repeat(2, 1fr); }
  .card .value { font-size: 1.25rem; }
  .table-wrap { overflow: visible; }
  table, thead, tbody, th, td, tr { display: block; }
  thead { display: none; }
  tr.row { padding: 12px; border-bottom: 1px solid var(--border); }
  tr.row td { border: none; padding: 5px 0; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  tr.row td::before {
    content: attr(data-label);
    font-size: .7rem;
    font-weight: 600;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: .05em;
  }
  tr.row.alert td:first-child::before { display: none; }
  tr.row.alert { border-left: 3px solid var(--err); }
}
</style>
</head>
<body>
<div class="hero">
  <div class="brand">
    <h1>Databricks Claude LB</h1>
    <span class="tag" id="uptimeTag">-</span>
  </div>
  <div class="hero-actions">
    <button class="theme-toggle" id="themeToggle" type="button" title="Toggle theme" aria-label="Toggle theme">
      <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
    </button>
    <div class="refresh-pill" id="refreshPill" title="Auto refresh every 5s">
      <svg viewBox="0 0 36 36">
        <circle class="ring-bg" cx="18" cy="18" r="15"/>
        <circle class="ring-fg" id="refreshRing" cx="18" cy="18" r="15" stroke-dasharray="94.25" stroke-dashoffset="0"/>
      </svg>
      <span id="refreshLabel">Updating</span>
    </div>
  </div>
</div>
<div class="error-banner" id="errorBanner"></div>
<div class="tab-bar">
  <button class="tab-btn active" data-tab="anthropic">Anthropic Models</button>
  <button class="tab-btn" data-tab="azure">Azure OpenAI Models</button>
  <button class="tab-btn" data-tab="history">Usage History</button>
  <span class="tab-indicator" id="tabIndicator"></span>
</div>

<div class="tab-panel active" id="tab-anthropic">
  <div class="global-grid" id="globalGrid"></div>
  <div class="grid-2">
    <div class="chart-card"><h3>Model Usage Share</h3><div class="canvas-wrap"><canvas id="anthropicModelChart"></canvas></div></div>
    <div class="chart-card"><h3>Endpoint Latency (ms)</h3><div class="canvas-wrap"><canvas id="anthropicLatencyChart"></canvas></div></div>
  </div>
  <h2 class="section-title">Databricks Endpoints</h2>
  <div class="table-wrap"><div class="table-scroll"><table id="endpointTable">
    <thead><tr>
      <th>Name</th><th>Status</th><th>Active</th><th>Total</th><th>Error Rate</th><th>Avg Latency</th><th>Total Tokens</th><th>Est. Cost</th>
    </tr></thead>
    <tbody id="endpointBody"></tbody>
  </table></div></div>
  <h2 class="section-title">Model Stats</h2>
  <div class="table-wrap"><div class="table-scroll"><table>
    <thead><tr>
      <th>Model</th><th>Requests</th><th>Input</th><th>Output</th><th>Cache Create</th><th>Cache Read</th><th>Total</th><th>Est. Cost</th>
    </tr></thead>
    <tbody id="modelBody"></tbody>
  </table></div></div>
</div>

<div class="tab-panel" id="tab-azure"><div id="azureContent"><div class="info-box">Loading...</div></div></div>

<div class="tab-panel" id="tab-history">
  <div class="history-toolbar">
    <label>Show last</label>
    <input id="histDays" type="number" value="7" min="1" max="365">
    <label>days</label>
    <button class="btn btn-primary" onclick="loadHistory(+document.getElementById('histDays').value)">Refresh</button>
    <span style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <label>Clean data older than</label>
      <input id="cleanDays" type="number" value="30" min="1" max="3650">
      <label>days</label>
      <button class="btn btn-danger" onclick="cleanupHistory()">Clean Up</button>
    </span>
  </div>
  <div id="historyContent"><div class="info-box">Loading history...</div></div>
</div>

<script>
// ---------- Formatters ----------
function fmt(n) { n = +n || 0; if (n >= 1e9) return (n/1e9).toFixed(2)+'B'; if (n >= 1e6) return (n/1e6).toFixed(2)+'M'; if (n >= 1e3) return (n/1e3).toFixed(1)+'K'; return n.toFixed(0); }
function fmtTime(s) { s = +s || 0; if (s >= 86400) return (s/86400).toFixed(1)+'d'; if (s >= 3600) return (s/3600).toFixed(1)+'h'; if (s >= 60) return (s/60).toFixed(1)+'m'; return s.toFixed(0)+'s'; }
function fmtCost(v) { if (v == null) return '-'; v = +v; if (v >= 1) return '$'+v.toFixed(2); if (v >= 0.01) return '$'+v.toFixed(4); if (v > 0) return '$'+v.toFixed(6); return '$0'; }
function fmtMs(v) { return (+v || 0).toFixed(0) + 'ms'; }
function fmtPct(v) { return (+v || 0).toFixed(1) + '%'; }

// ---------- Number Ticker ----------
function tickTo(el, target, formatter) {
  if (!el) return;
  target = +target || 0;
  const start = parseFloat(el.dataset.cur || '0');
  if (Math.abs(start - target) < 0.0001) { el.textContent = formatter(target); el.dataset.cur = target; return; }
  cancelAnimationFrame(el._raf || 0);
  const t0 = performance.now();
  const dur = 700;
  const step = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - p, 3);
    const cur = start + (target - start) * e;
    el.textContent = formatter(cur);
    el.dataset.cur = cur;
    if (p < 1) el._raf = requestAnimationFrame(step);
    else el.dataset.cur = target;
  };
  el._raf = requestAnimationFrame(step);
}

// ---------- Tab handling ----------
let activeTab = 'anthropic';
let historyLoaded = false;
function switchTab(t) {
  activeTab = t;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === t));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + t));
  updateTabIndicator();
  if (t === 'history' && !historyLoaded) loadHistory();
}
function updateTabIndicator() {
  const active = document.querySelector('.tab-btn.active');
  const ind = document.getElementById('tabIndicator');
  if (!active || !ind) return;
  ind.style.width = active.offsetWidth + 'px';
  ind.style.transform = 'translateX(' + active.offsetLeft + 'px)';
}
document.querySelectorAll('.tab-btn').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));
window.addEventListener('resize', updateTabIndicator);

// ---------- Magic card hover ----------
function bindMagicHover(scope) {
  (scope || document).querySelectorAll('.card:not([data-magic])').forEach(c => {
    c.dataset.magic = '1';
    c.addEventListener('mousemove', e => {
      const r = c.getBoundingClientRect();
      c.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      c.style.setProperty('--my', (e.clientY - r.top) + 'px');
    });
  });
}

// ---------- Theme ----------
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function applyChartDefaults() {
  Chart.defaults.color = cssVar('--text-2') || '#94a3b8';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = cssVar('--chart-grid-strong') || 'rgba(51, 65, 85, 0.4)';
}

function applyThemeToChart(c) {
  if (!c || !c.options) return;
  const text1 = cssVar('--text-1');
  const text2 = cssVar('--text-2');
  const tooltipBg = cssVar('--tooltip-bg');
  const tooltipBorder = cssVar('--tooltip-border');
  const grid = cssVar('--chart-grid');
  const chartBorder = cssVar('--chart-border');
  const p = c.options.plugins || {};
  if (p.tooltip) { p.tooltip.backgroundColor = tooltipBg; p.tooltip.borderColor = tooltipBorder; p.tooltip.titleColor = '#f1f5f9'; p.tooltip.bodyColor = '#e2e8f0'; }
  if (p.legend && p.legend.labels) { p.legend.labels.color = text1; }
  if (c.options.scales) {
    Object.values(c.options.scales).forEach(s => {
      if (s && s.grid && s.grid.color !== undefined && s.grid.display !== false) s.grid.color = grid;
      if (s && s.ticks) s.ticks.color = text2;
      if (s && s.title) s.title.color = text2;
    });
  }
  if (c.config.type === 'doughnut' && c.data.datasets[0]) {
    c.data.datasets[0].borderColor = chartBorder;
  }
  c.update('none');
}

function applyChartTheme() {
  applyChartDefaults();
  Object.values(charts).forEach(applyThemeToChart);
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('lb-theme', theme); } catch (e) {}
  const btn = document.getElementById('themeToggle');
  if (btn) { btn.title = theme === 'light' ? 'Switch to dark' : 'Switch to light'; btn.setAttribute('aria-label', btn.title); }
  applyChartTheme();
}

function initTheme() {
  let t = null;
  try { t = localStorage.getItem('lb-theme'); } catch (e) {}
  if (t !== 'light' && t !== 'dark') {
    t = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
  document.documentElement.setAttribute('data-theme', t);
  applyChartDefaults();
}

// ---------- Chart registry ----------
const charts = {};
const chartColors = ['#60a5fa', '#c084fc', '#4ade80', '#fbbf24', '#f87171', '#22d3ee', '#f472b6', '#a78bfa', '#fb923c', '#34d399'];

function ensureDoughnut(canvasId) {
  if (charts[canvasId]) return charts[canvasId];
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  charts[canvasId] = new Chart(ctx, {
    type: 'doughnut',
    data: { labels: [], datasets: [{ data: [], backgroundColor: chartColors, borderColor: cssVar('--chart-border') || 'rgba(15, 23, 42, 0.8)', borderWidth: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '62%',
      plugins: {
        legend: { position: 'right', labels: { boxWidth: 10, boxHeight: 10, padding: 10, color: cssVar('--text-1'), font: { size: 11 } } },
        tooltip: {
          backgroundColor: cssVar('--tooltip-bg'), borderColor: cssVar('--tooltip-border'), borderWidth: 1, padding: 10, cornerRadius: 8,
          callbacks: {
            label: (c) => {
              const total = c.dataset.data.reduce((a, b) => a + b, 0);
              const v = c.parsed;
              const pct = total > 0 ? ((v / total) * 100).toFixed(1) : '0';
              return c.label + ': ' + fmt(v) + ' (' + pct + '%)';
            }
          }
        }
      },
      animation: { duration: 500, easing: 'easeOutCubic' }
    }
  });
  return charts[canvasId];
}

function ensureBar(canvasId) {
  if (charts[canvasId]) return charts[canvasId];
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Avg Latency (ms)', data: [], backgroundColor: 'rgba(96, 165, 250, 0.7)', borderColor: '#60a5fa', borderWidth: 1, borderRadius: 6 }] },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: cssVar('--tooltip-bg'), borderColor: cssVar('--tooltip-border'), borderWidth: 1, padding: 10, cornerRadius: 8, callbacks: { label: (c) => fmtMs(c.parsed.x) } }
      },
      scales: {
        x: { beginAtZero: true, grid: { color: cssVar('--chart-grid') }, ticks: { callback: (v) => v + 'ms' } },
        y: { grid: { display: false } }
      },
      animation: { duration: 500, easing: 'easeOutCubic' }
    }
  });
  return charts[canvasId];
}

function ensureHistoryLine(canvasId) {
  if (charts[canvasId]) return charts[canvasId];
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Input', data: [], borderColor: '#60a5fa', backgroundColor: 'rgba(96, 165, 250, 0.15)', tension: 0.35, fill: true, yAxisID: 'y', pointRadius: 3, pointHoverRadius: 5, borderWidth: 2 },
        { label: 'Output', data: [], borderColor: '#c084fc', backgroundColor: 'rgba(192, 132, 252, 0.12)', tension: 0.35, fill: true, yAxisID: 'y', pointRadius: 3, pointHoverRadius: 5, borderWidth: 2 },
        { label: 'Cache Read', data: [], borderColor: '#4ade80', backgroundColor: 'rgba(74, 222, 128, 0.08)', tension: 0.35, fill: false, yAxisID: 'y', pointRadius: 3, pointHoverRadius: 5, borderWidth: 2, borderDash: [4, 4] },
        { label: 'Cost (USD)', data: [], borderColor: '#fbbf24', backgroundColor: 'rgba(251, 191, 36, 0.1)', tension: 0.35, fill: false, yAxisID: 'y1', pointRadius: 3, pointHoverRadius: 5, borderWidth: 2 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, boxHeight: 12, padding: 14, color: cssVar('--text-1'), usePointStyle: true, pointStyle: 'circle' } },
        tooltip: {
          backgroundColor: cssVar('--tooltip-bg'), borderColor: cssVar('--tooltip-border'), borderWidth: 1, padding: 12, cornerRadius: 8,
          callbacks: { label: (c) => c.dataset.label + ': ' + (c.dataset.yAxisID === 'y1' ? fmtCost(c.parsed.y) : fmt(c.parsed.y)) }
        }
      },
      scales: {
        x: { grid: { color: cssVar('--chart-grid') } },
        y: { type: 'linear', position: 'left', beginAtZero: true, grid: { color: cssVar('--chart-grid') }, ticks: { callback: (v) => fmt(v) }, title: { display: true, text: 'Tokens', color: cssVar('--text-2'), font: { size: 11 } } },
        y1: { type: 'linear', position: 'right', beginAtZero: true, grid: { display: false }, ticks: { callback: (v) => '$' + v.toFixed(2) }, title: { display: true, text: 'Cost', color: cssVar('--text-2'), font: { size: 11 } } }
      },
      animation: { duration: 600, easing: 'easeOutCubic' }
    }
  });
  return charts[canvasId];
}

// ---------- KPI cards (diff update) ----------
const KPI_DEFS = [
  { key: 'total_requests', label: 'Total Requests', fmt: fmt, cls: 'blue' },
  { key: 'total_input_tokens', label: 'Input Tokens', fmt: fmt, cls: '' },
  { key: 'total_output_tokens', label: 'Output Tokens', fmt: fmt, cls: '' },
  { key: 'total_cache_creation_tokens', label: 'Cache Create', fmt: fmt, cls: '' },
  { key: 'total_cache_read_tokens', label: 'Cache Read', fmt: fmt, cls: '' },
  { key: 'total_tokens', label: 'Total Tokens', fmt: fmt, cls: 'amber' },
  { key: 'avg_response_time_ms', label: 'Avg Latency', fmt: fmtMs, cls: '' },
  { key: 'requests_per_minute', label: 'RPM', fmt: (v) => (+v).toFixed(2), cls: 'green' },
  { key: 'estimated_total_cost_usd', label: 'Est. Cost', fmt: fmtCost, cls: 'purple' }
];

function renderKpiGrid(gridId, defs, data, prefix) {
  const grid = document.getElementById(gridId);
  if (!grid.dataset.built) {
    grid.innerHTML = defs.map(d => '<div class="card"><div class="label">' + d.label + '</div><div class="value ' + d.cls + '" id="' + prefix + '-' + d.key + '">0</div></div>').join('');
    grid.dataset.built = '1';
    bindMagicHover(grid);
  }
  defs.forEach(d => {
    const el = document.getElementById(prefix + '-' + d.key);
    tickTo(el, data[d.key] || 0, d.fmt);
  });
}

// ---------- Endpoint table (diff update) ----------
function errBarColor(rate) {
  if (rate <= 1) return 'var(--ok)';
  if (rate <= 5) return 'var(--warn)';
  return 'var(--err)';
}

function renderEndpointRow(e, prefix) {
  const rowId = prefix + '-row-' + e.name;
  let tr = document.getElementById(rowId);
  const isAlert = e.circuit_open || (e.error_rate || 0) > 5;
  const deploymentsHtml = e.deployments ? '<div style="font-size:.7rem;color:var(--text-3);margin-top:3px">' + (e.deployments || []).map(d => '<span class="deployments-tag">' + d + '</span>').join('') + '</div>' : '';
  const barColor = errBarColor(+e.error_rate || 0);
  const barWidth = Math.min(100, (+e.error_rate || 0) * 10);
  const html =
    '<td data-label="Name"><strong>' + e.name + '</strong>' + deploymentsHtml + '</td>' +
    '<td data-label="Status"><span class="badge ' + (e.circuit_open ? 'open' : 'ok') + '">' + (e.circuit_open ? 'OPEN' : 'OK') + '</span></td>' +
    '<td data-label="Active" class="mono">' + e.active_requests + '</td>' +
    '<td data-label="Total" class="mono">' + fmt(e.total_requests) + '</td>' +
    '<td data-label="Error Rate"><span class="err-rate"><span class="bar"><span style="width:' + barWidth + '%;background:' + barColor + '"></span></span><span class="mono">' + fmtPct(e.error_rate) + '</span></span></td>' +
    '<td data-label="Avg Latency" class="mono">' + fmtMs(e.avg_response_time_ms) + '</td>' +
    '<td data-label="Total Tokens" class="mono">' + fmt(e.total_tokens) + '</td>' +
    '<td data-label="Est. Cost" class="mono">' + (e.estimated_cost_usd != null ? fmtCost(e.estimated_cost_usd) : '-') + '</td>';
  if (!tr) {
    tr = document.createElement('tr');
    tr.id = rowId;
    tr.className = 'row';
    tr.innerHTML = html;
    return { tr, isAlert, isNew: true };
  }
  tr.innerHTML = html;
  return { tr, isAlert, isNew: false };
}

function diffEndpoints(bodyId, endpoints, prefix) {
  endpoints = endpoints || [];
  const tbody = document.getElementById(bodyId);
  const seen = new Set();
  endpoints.forEach(e => {
    seen.add(prefix + '-row-' + e.name);
    const { tr, isAlert, isNew } = renderEndpointRow(e, prefix);
    tr.classList.toggle('alert', isAlert);
    if (isNew) tbody.appendChild(tr);
  });
  Array.from(tbody.children).forEach(tr => { if (!seen.has(tr.id)) tr.remove(); });
}

// ---------- Model tables ----------
function aggregateModels(endpoints) {
  const models = {};
  (endpoints || []).forEach(e => {
    Object.entries(e.model_stats || {}).forEach(([m, s]) => {
      if (!models[m]) models[m] = { input_tokens: 0, output_tokens: 0, cache_creation_tokens: 0, cache_read_tokens: 0, requests: 0, estimated_cost_usd: 0 };
      models[m].input_tokens += s.input_tokens || 0;
      models[m].output_tokens += s.output_tokens || 0;
      models[m].cache_creation_tokens += s.cache_creation_tokens || 0;
      models[m].cache_read_tokens += s.cache_read_tokens || 0;
      models[m].requests += s.requests || 0;
      models[m].estimated_cost_usd += s.estimated_cost_usd || 0;
    });
  });
  return models;
}

function renderModelTable(bodyId, models, withCost) {
  const tbody = document.getElementById(bodyId);
  const entries = Object.entries(models);
  if (entries.length === 0) {
    tbody.innerHTML = '<tr><td colspan="' + (withCost ? 8 : 7) + '" style="text-align:center;color:var(--text-3);padding:24px">No data yet</td></tr>';
    return;
  }
  tbody.innerHTML = entries.map(([m, s]) =>
    '<tr class="row">' +
    '<td data-label="Model"><strong>' + m + '</strong></td>' +
    '<td data-label="Requests" class="mono">' + fmt(s.requests) + '</td>' +
    '<td data-label="Input" class="mono">' + fmt(s.input_tokens) + '</td>' +
    '<td data-label="Output" class="mono">' + fmt(s.output_tokens) + '</td>' +
    '<td data-label="Cache Create" class="mono">' + fmt(s.cache_creation_tokens) + '</td>' +
    '<td data-label="Cache Read" class="mono">' + fmt(s.cache_read_tokens) + '</td>' +
    '<td data-label="Total" class="mono">' + fmt(s.input_tokens + s.output_tokens) + '</td>' +
    (withCost ? '<td data-label="Est. Cost" class="mono">' + fmtCost(s.estimated_cost_usd) + '</td>' : '') +
    '</tr>'
  ).join('');
}

// ---------- History ----------
async function loadHistory(days) {
  days = days || 7;
  const hc = document.getElementById('historyContent');
  hc.innerHTML = '<div class="info-box">Loading...</div>';
  try {
    const r = await fetch('/stats/history?days=' + days);
    const d = await r.json();
    if (d.error) { hc.innerHTML = '<div class="info-box">' + d.error + '</div>'; return; }
    let totalCost = 0, totalTokens = 0, totalReq = 0;
    d.history.forEach(h => {
      totalCost += h.estimated_total_cost_usd || 0;
      const t = h.totals || {};
      totalTokens += (t.input_tokens || 0) + (t.output_tokens || 0);
      totalReq += t.requests || 0;
    });
    const avgCost = d.history.length ? totalCost / d.history.length : 0;
    hc.innerHTML =
      '<div class="global-grid">' +
      '<div class="card"><div class="label">Period</div><div class="value blue">' + days + ' Days</div></div>' +
      '<div class="card"><div class="label">Total Requests</div><div class="value">' + fmt(totalReq) + '</div></div>' +
      '<div class="card"><div class="label">Total Tokens</div><div class="value amber">' + fmt(totalTokens) + '</div></div>' +
      '<div class="card"><div class="label">Total Cost</div><div class="value purple">' + fmtCost(totalCost) + '</div></div>' +
      '<div class="card"><div class="label">Avg Daily Cost</div><div class="value">' + fmtCost(avgCost) + '</div></div>' +
      '</div>' +
      '<div class="grid-2">' +
      '<div class="chart-card" style="min-height:320px"><h3>Daily Tokens &amp; Cost</h3><div class="canvas-wrap" style="height:280px"><canvas id="historyLineChart"></canvas></div></div>' +
      '<div class="chart-card" style="min-height:320px"><h3>Model Breakdown (' + days + 'd)</h3><div class="canvas-wrap" style="height:280px"><canvas id="historyModelChart"></canvas></div></div>' +
      '</div>' +
      '<h2 class="section-title">Daily Usage</h2>' +
      '<div class="table-wrap"><div class="table-scroll"><table>' +
      '<thead><tr><th>Date</th><th>Requests</th><th>Input</th><th>Output</th><th>Cache Create</th><th>Cache Read</th><th>Total</th><th>Est. Cost</th></tr></thead>' +
      '<tbody>' +
      d.history.slice().reverse().map(h => {
        const t = h.totals || {};
        return '<tr class="row">' +
          '<td data-label="Date" class="mono">' + h.date + '</td>' +
          '<td data-label="Requests" class="mono">' + fmt(t.requests) + '</td>' +
          '<td data-label="Input" class="mono">' + fmt(t.input_tokens) + '</td>' +
          '<td data-label="Output" class="mono">' + fmt(t.output_tokens) + '</td>' +
          '<td data-label="Cache Create" class="mono">' + fmt(t.cache_creation_tokens) + '</td>' +
          '<td data-label="Cache Read" class="mono">' + fmt(t.cache_read_tokens) + '</td>' +
          '<td data-label="Total" class="mono">' + fmt((t.input_tokens || 0) + (t.output_tokens || 0)) + '</td>' +
          '<td data-label="Est. Cost" class="mono">' + fmtCost(h.estimated_total_cost_usd) + '</td>' +
          '</tr>';
      }).join('') +
      '</tbody></table></div></div>' +
      '<h2 class="section-title">Model Breakdown</h2>' +
      '<div class="table-wrap"><div class="table-scroll"><table id="historyModelTable">' +
      '<thead><tr><th>Model</th><th>Requests</th><th>Input</th><th>Output</th><th>Cache Create</th><th>Cache Read</th><th>Total</th><th>Est. Cost</th></tr></thead>' +
      '<tbody id="historyModelBody"></tbody>' +
      '</table></div></div>';
    bindMagicHover(hc);

    // Populate charts
    const lineChart = ensureHistoryLine('historyLineChart');
    if (lineChart) {
      const labels = d.history.map(h => h.date);
      const input = d.history.map(h => (h.totals || {}).input_tokens || 0);
      const output = d.history.map(h => (h.totals || {}).output_tokens || 0);
      const cacheRead = d.history.map(h => (h.totals || {}).cache_read_tokens || 0);
      const cost = d.history.map(h => h.estimated_total_cost_usd || 0);
      lineChart.data.labels = labels;
      lineChart.data.datasets[0].data = input;
      lineChart.data.datasets[1].data = output;
      lineChart.data.datasets[2].data = cacheRead;
      lineChart.data.datasets[3].data = cost;
      lineChart.update();
    }

    // Aggregate model stats
    const allModels = {};
    d.history.forEach(h => {
      Object.entries(h.models || {}).forEach(([m, s]) => {
        if (!allModels[m]) allModels[m] = { input_tokens: 0, output_tokens: 0, cache_creation_tokens: 0, cache_read_tokens: 0, requests: 0, estimated_cost_usd: 0 };
        allModels[m].input_tokens += s.input_tokens || 0;
        allModels[m].output_tokens += s.output_tokens || 0;
        allModels[m].cache_creation_tokens += s.cache_creation_tokens || 0;
        allModels[m].cache_read_tokens += s.cache_read_tokens || 0;
        allModels[m].requests += s.requests || 0;
        allModels[m].estimated_cost_usd += s.estimated_cost_usd || 0;
      });
    });
    renderModelTable('historyModelBody', allModels, true);
    const modelChart = ensureDoughnut('historyModelChart');
    if (modelChart) {
      const entries = Object.entries(allModels);
      modelChart.data.labels = entries.map(e => e[0]);
      modelChart.data.datasets[0].data = entries.map(e => e[1].input_tokens + e[1].output_tokens);
      modelChart.update();
    }
    historyLoaded = true;
  } catch (err) {
    hc.innerHTML = '<div class="info-box">Failed to load history: ' + err.message + '</div>';
  }
}

async function cleanupHistory() {
  const days = +document.getElementById('cleanDays').value;
  if (!confirm('Confirm: delete all usage data older than ' + days + ' days?')) return;
  try {
    const r = await fetch('/stats/history?keep_days=' + days, { method: 'DELETE' });
    const d = await r.json();
    alert('Deleted ' + d.deleted + ' records');
    historyLoaded = false;
    loadHistory(+document.getElementById('histDays').value);
  } catch (err) { alert('Cleanup failed: ' + err.message); }
}

// ---------- Refresh ----------
const REFRESH_MS = 5000;
let refreshDeadline = Date.now() + REFRESH_MS;

async function refresh() {
  try {
    const r = await fetch('/stats');
    const d = await r.json();
    const g = d.global;
    document.getElementById('errorBanner').classList.remove('show');
    document.getElementById('uptimeTag').textContent = 'uptime ' + fmtTime(g.uptime_seconds);
    document.getElementById('refreshLabel').textContent = new Date().toLocaleTimeString();

    // Attach cost to each endpoint by summing its model cost
    (d.endpoints || []).forEach(e => {
      let c = 0;
      Object.values(e.model_stats || {}).forEach(s => { c += +s.estimated_cost_usd || 0; });
      e.estimated_cost_usd = c;
    });

    renderKpiGrid('globalGrid', KPI_DEFS, g, 'kpi-anthropic');
    diffEndpoints('endpointBody', d.endpoints, 'anthropic');

    // Anthropic Models 表格以“当天(含重启前)累计”为准，与 KPI Est. Cost 保持一致
    const models = d.today_model_stats && Object.keys(d.today_model_stats).length
      ? d.today_model_stats
      : aggregateModels(d.endpoints);
    renderModelTable('modelBody', models, true);

    // Update anthropic charts
    const modelChart = ensureDoughnut('anthropicModelChart');
    if (modelChart) {
      const entries = Object.entries(models);
      modelChart.data.labels = entries.map(e => e[0]);
      modelChart.data.datasets[0].data = entries.map(e => e[1].input_tokens + e[1].output_tokens);
      modelChart.update('none');
    }
    const latencyChart = ensureBar('anthropicLatencyChart');
    if (latencyChart) {
      latencyChart.data.labels = (d.endpoints || []).map(e => e.name);
      latencyChart.data.datasets[0].data = (d.endpoints || []).map(e => +e.avg_response_time_ms || 0);
      latencyChart.update('none');
    }

    // Azure section
    const azContent = document.getElementById('azureContent');
    if (d.azure_openai) {
      if (!azContent.dataset.built) {
        azContent.innerHTML =
          '<div class="global-grid" id="azureGlobalGrid"></div>' +
          '<div class="grid-2">' +
          '<div class="chart-card"><h3>Model Usage Share</h3><div class="canvas-wrap"><canvas id="azureModelChart"></canvas></div></div>' +
          '<div class="chart-card"><h3>Endpoint Latency (ms)</h3><div class="canvas-wrap"><canvas id="azureLatencyChart"></canvas></div></div>' +
          '</div>' +
          '<h2 class="section-title">Azure OpenAI Endpoints</h2>' +
          '<div class="table-wrap"><div class="table-scroll"><table>' +
          '<thead><tr><th>Name</th><th>Status</th><th>Active</th><th>Total</th><th>Error Rate</th><th>Avg Latency</th><th>Total Tokens</th><th>Deployments</th></tr></thead>' +
          '<tbody id="azureEndpointBody"></tbody>' +
          '</table></div></div>' +
          '<h2 class="section-title">Model Stats</h2>' +
          '<div class="table-wrap"><div class="table-scroll"><table>' +
          '<thead><tr><th>Model</th><th>Requests</th><th>Input</th><th>Output</th><th>Cache Create</th><th>Cache Read</th><th>Total</th></tr></thead>' +
          '<tbody id="azureModelBody"></tbody>' +
          '</table></div></div>';
        azContent.dataset.built = '1';
      }
      const AZURE_KPIS = KPI_DEFS.filter(k => k.key !== 'estimated_total_cost_usd');
      renderKpiGrid('azureGlobalGrid', AZURE_KPIS, d.azure_openai.global, 'kpi-azure');
      // Azure endpoint rows (render inline; table slightly different — deployments col)
      const tbody = document.getElementById('azureEndpointBody');
      const seen = new Set();
      (d.azure_openai.endpoints || []).forEach(e => {
        const rowId = 'azure-row-' + e.name;
        seen.add(rowId);
        let tr = document.getElementById(rowId);
        const isAlert = e.circuit_open || (e.error_rate || 0) > 5;
        const barColor = errBarColor(+e.error_rate || 0);
        const barWidth = Math.min(100, (+e.error_rate || 0) * 10);
        const html =
          '<td data-label="Name"><strong>' + e.name + '</strong></td>' +
          '<td data-label="Status"><span class="badge ' + (e.circuit_open ? 'open' : 'ok') + '">' + (e.circuit_open ? 'OPEN' : 'OK') + '</span></td>' +
          '<td data-label="Active" class="mono">' + e.active_requests + '</td>' +
          '<td data-label="Total" class="mono">' + fmt(e.total_requests) + '</td>' +
          '<td data-label="Error Rate"><span class="err-rate"><span class="bar"><span style="width:' + barWidth + '%;background:' + barColor + '"></span></span><span class="mono">' + fmtPct(e.error_rate) + '</span></span></td>' +
          '<td data-label="Avg Latency" class="mono">' + fmtMs(e.avg_response_time_ms) + '</td>' +
          '<td data-label="Total Tokens" class="mono">' + fmt(e.total_tokens) + '</td>' +
          '<td data-label="Deployments">' + (e.deployments || []).map(x => '<span class="deployments-tag">' + x + '</span>').join('') + '</td>';
        if (!tr) {
          tr = document.createElement('tr');
          tr.id = rowId;
          tr.className = 'row';
          tr.innerHTML = html;
          tbody.appendChild(tr);
        } else { tr.innerHTML = html; }
        tr.classList.toggle('alert', isAlert);
      });
      Array.from(tbody.children).forEach(tr => { if (!seen.has(tr.id)) tr.remove(); });

      const azModels = aggregateModels(d.azure_openai.endpoints);
      renderModelTable('azureModelBody', azModels, false);
      const azModelChart = ensureDoughnut('azureModelChart');
      if (azModelChart) {
        const entries = Object.entries(azModels);
        azModelChart.data.labels = entries.map(e => e[0]);
        azModelChart.data.datasets[0].data = entries.map(e => e[1].input_tokens + e[1].output_tokens);
        azModelChart.update('none');
      }
      const azLatencyChart = ensureBar('azureLatencyChart');
      if (azLatencyChart) {
        azLatencyChart.data.labels = (d.azure_openai.endpoints || []).map(e => e.name);
        azLatencyChart.data.datasets[0].data = (d.azure_openai.endpoints || []).map(e => +e.avg_response_time_ms || 0);
        azLatencyChart.update('none');
      }
    } else {
      azContent.innerHTML = '<div class="info-box">Azure OpenAI is not configured. Add <code>azure_openai</code> section to your config.yaml to enable this tab.</div>';
      azContent.dataset.built = '';
    }
  } catch (err) {
    const b = document.getElementById('errorBanner');
    b.textContent = 'Failed to fetch stats: ' + err.message;
    b.classList.add('show');
  } finally {
    refreshDeadline = Date.now() + REFRESH_MS;
  }
}

// Refresh ring countdown
function updateRing() {
  const ring = document.getElementById('refreshRing');
  if (!ring) return;
  const remain = Math.max(0, Math.min(REFRESH_MS, refreshDeadline - Date.now()));
  const p = remain / REFRESH_MS;
  const circumference = 2 * Math.PI * 15;
  ring.setAttribute('stroke-dasharray', circumference);
  ring.setAttribute('stroke-dashoffset', circumference * (1 - p));
  requestAnimationFrame(updateRing);
}

// ---------- Bootstrap ----------
initTheme();
document.getElementById('themeToggle').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme') || 'dark';
  setTheme(cur === 'light' ? 'dark' : 'light');
});
requestAnimationFrame(() => { updateTabIndicator(); updateRing(); });
refresh();
setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
