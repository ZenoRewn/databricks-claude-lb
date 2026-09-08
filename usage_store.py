"""P2.2 extracted from main.py: usage data persistence backends.

- UsageDataStore: buffered + periodic-flush base
- JsonUsageStore: JSON file backend, atomic replace, per-day sharding
- MysqlUsageStore: aiomysql-backed with INSERT ON DUPLICATE KEY UPDATE
- create_usage_store: factory dispatching on storage_config['type']

The classes preserve their identity when imported back into main.py, so any
`main.UsageDataStore` / `main.create_usage_store` references keep working via
the re-export in main.py.
"""
import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("main")

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
        # P2.5 self-heal: 内层 try 已经把 flush + cleanup 包住；外层新增 sleep 也包
        # 在 try 里以防 CancelledError 之外的异常从 sleep 抛出（罕见但曾发生过）。
        while True:
            try:
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
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Usage data periodic loop unexpected error, backing off 5s: {type(e).__name__}: {e}"
                )
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break

    @staticmethod
    def _empty_day(d: date) -> dict:
        return {
            "date": d.isoformat(),
            "models": {},
            "totals": {"input_tokens": 0, "output_tokens": 0,
                       "cache_creation_tokens": 0, "cache_read_tokens": 0,
                       "requests": 0, "errors": 0},
        }

    @staticmethod
    def _empty_model() -> dict:
        return {"input_tokens": 0, "output_tokens": 0,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "requests": 0}

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
                self._today_cache = self._empty_day(today)

            # 两份数据同步累：
            #   _today_cache —— 当天累计，/stats 与 dashboard 读它，重启时由
            #                    _load_day 从后端种回，语义不变
            #   batch        —— 仅本次 flush 的增量，交给 _save_day_delta。
            #                    多写后端（MySQL）用它做 col = col + VALUES(col)，
            #                    这样多副本不会互相覆盖对方的量
            batch = self._empty_day(today)
            models = self._today_cache["models"]
            totals = self._today_cache["totals"]
            b_models = batch["models"]
            b_totals = batch["totals"]
            for delta in pending:
                m = delta["model"]
                if m not in models:
                    models[m] = self._empty_model()
                if m not in b_models:
                    b_models[m] = self._empty_model()
                for field in ("input_tokens", "output_tokens",
                              "cache_creation_tokens", "cache_read_tokens"):
                    models[m][field] += delta[field]
                    b_models[m][field] += delta[field]
                    totals[field] += delta[field]
                    b_totals[field] += delta[field]
                models[m]["requests"] += 1
                b_models[m]["requests"] += 1
                totals["requests"] += 1
                b_totals["requests"] += 1
                if delta.get("is_error"):
                    totals["errors"] += 1
                    b_totals["errors"] += 1

            stamp = datetime.now().astimezone().isoformat()
            self._today_cache["last_updated"] = stamp
            batch["last_updated"] = stamp
            await self._save_day_delta(today, batch, self._today_cache)

    def get_today_data(self) -> dict:
        return self._today_cache if self._today_cache else {}

    async def get_day_data(self, d: date) -> Optional[dict]:
        if d == self._today_date:
            return self.get_today_data()
        return await self._load_day(d) or None

    # 子类实现
    async def _backend_start(self): pass
    async def _backend_stop(self): pass
    # _save_day 写「当天累计绝对值」，只被下面 _save_day_delta 的默认实现调用。
    # MysqlUsageStore 不实现它 —— 它 override 的是 _save_day_delta（增量累加）。
    # 因此**不要**在别处直接调 _save_day，MySQL 后端上那是个静默 no-op。
    async def _save_day(self, d: date, data: dict): pass
    async def _load_day(self, d: date) -> dict: return {}
    async def _delete_before(self, cutoff: date) -> int: return 0

    async def _save_day_delta(self, d: date, delta: dict, cumulative: dict):
        """落盘钩子。``delta`` 只含本次 flush 的增量，``cumulative`` 是当天累计。

        默认实现写 ``cumulative``（整天绝对覆盖），这是**单写者**语义 ——
        文件后端无法跨进程原子累加，所以 JsonUsageStore 保持这条路径，
        代价是 JSON 后端只能跑单副本。

        支持原子累加的后端（MySQL）override 本方法、改用 ``delta``，这样多个
        副本各自 flush 时不会把对方的量抹掉（否则 DB 行只保留最后一次写入的
        「该副本起点 + 该副本增量」，其余副本的量永久丢失）。
        """
        await self._save_day(d, cumulative)


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
        import ssl as _ssl
        ssl_ctx = _ssl.create_default_context()
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
            ssl=ssl_ctx,
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

    async def _save_day_delta(self, d: date, delta: dict, cumulative: dict):
        """增量累加落盘 —— 多副本安全。

        写 ``delta``（本次 flush 的增量）而非 ``cumulative``（当天累计），
        并用 ``col = col + VALUES(col)`` 让 InnoDB 在行锁内做累加。多个 pod
        同时跑时，DB 行 = 所有副本增量之和；若改回 ``= VALUES(col)`` 的绝对
        覆盖，各副本每 30s 就把对方的量抹掉一次（回归测试见
        ``tests/test_usage_store.py::test_two_writers_sum_instead_of_clobber``）。

        副作用之一：单副本下也更安全 —— 旧写法若某次 ``_load_day`` 因瞬时 DB
        异常返回空，``_flush`` 会以零值重建缓存并把当天整行覆盖掉；增量写没有
        这条路径。
        """
        if not self._pool:
            return
        models = delta.get("models", {})
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
                            input_tokens = input_tokens + VALUES(input_tokens),
                            output_tokens = output_tokens + VALUES(output_tokens),
                            cache_creation_tokens = cache_creation_tokens + VALUES(cache_creation_tokens),
                            cache_read_tokens = cache_read_tokens + VALUES(cache_read_tokens),
                            requests = requests + VALUES(requests),
                            errors = errors + VALUES(errors)
                    """, (d.isoformat(), model_name,
                          mstats.get("input_tokens", 0), mstats.get("output_tokens", 0),
                          mstats.get("cache_creation_tokens", 0), mstats.get("cache_read_tokens", 0),
                          mstats.get("requests", 0),
                          # per-model errors 目前不被跟踪（models[m] 里没有 errors
                          # 字段，只有 totals 有），故恒为 0 增量。已知缺口，独立议题。
                          0))

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
        try:
            import aiomysql  # noqa: F401
        except ImportError:
            logger.critical(
                "usage_storage.type=mysql but aiomysql is not installed. "
                "Run: pip install aiomysql>=0.2.0 — refusing to start to avoid silent data loss."
            )
            sys.exit(1)
        return MysqlUsageStore(storage_config, retention)
    return JsonUsageStore(storage_config.get("path", "./usage_data"), retention)

