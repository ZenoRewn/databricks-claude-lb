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
import copy
from collections import deque
import hashlib
import itertools
import json
import logging
import math
import os
import importlib.util
import sys
import time
import uuid
import anyio
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger("main")

IO_TIMEOUT = float(os.getenv('USAGE_IO_TIMEOUT_SECONDS','10'))
MAX_BUFFER_EVENTS = int(os.getenv('USAGE_MAX_BUFFER_EVENTS','100000'))
FLUSH_BATCH_EVENTS = int(os.getenv('USAGE_FLUSH_BATCH_EVENTS','1000'))
if not math.isfinite(IO_TIMEOUT) or IO_TIMEOUT <= 0 or MAX_BUFFER_EVENTS < 1 or FLUSH_BATCH_EVENTS < 1:
    raise ValueError('Usage persistence budgets must be finite and positive')


async def _storage_cleanup(awaitable):
    """Retain transaction/pool cleanup across repeated or level cancellation."""
    task = asyncio.create_task(awaitable)
    cancelled = None
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancelled = exc
        result = task.result()
    if cancelled is not None:
        raise cancelled
    return result

# ==================== Usage Data Persistence ====================

class UsageDataStore:
    """基类：缓冲 + 定时刷盘框架，子类实现存储后端"""

    def __init__(self, retention_days: int = 0):
        self._buffer: list = []
        self._pending_groups = deque()
        self._inflight_batch = None
        self._flush_failures = 0
        self._flush_successes = 0
        self._accepted_events = 0
        self._persisted_events = 0
        self._rejected_events = 0
        self._last_success = 0.0
        self._backend_ready = False
        self._last_error_type = None
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._today_cache: dict = {}
        self._today_date: Optional[date] = None
        self.retention_days = retention_days
        self._last_cleanup_date: Optional[date] = None

    async def start(self):
        try:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_start()
                today = date.today()
                self._today_date = today
                self._today_cache = await self._load_day(today)
                self._backend_ready = True
                if self.retention_days > 0:
                    deleted = await self._delete_before(today - timedelta(days=self.retention_days))
                    if deleted:
                        logger.info(f"Auto-cleanup: deleted {deleted} expired records (retention={self.retention_days}d)")
                    self._last_cleanup_date = today
        except Exception as e:
            self._backend_ready = False
            self._last_error_type = type(e).__name__
            logger.error(f"Failed to initialize usage data store: {e}")
        # Statistics failure must not permanently disable the recovery loop.
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self):
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        try:
            while self._buffer or self._pending_groups:
                await self._flush()
        finally:
            await self._backend_stop()

    def record(self, model: str, input_tokens: int, output_tokens: int,
               cache_creation_tokens: int = 0, cache_read_tokens: int = 0,
               is_error: bool = False, *, provider: str = 'unknown',
               tenant: str = 'default', request_id: Optional[str] = None,
               generation_outcome: str = 'unknown', usage_fields=None):
        if generation_outcome not in ('unknown','completed','failed','incomplete','cancelled','client_disconnected'):
            raise ValueError('Invalid usage generation outcome')
        if not isinstance(model,str) or not model or len(model)>128:
            raise ValueError('Usage model must fit the persisted model identifier')
        model.encode('utf-8')
        if not all(type(value) is int and 0 <= value < 2**63 for value in
                   (input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens)):
            raise ValueError('Usage token counts must be nonnegative int64 values')
        if self.persistence_stats()['pending_events'] >= MAX_BUFFER_EVENTS:
            self._rejected_events += 1
            raise BufferError('Usage event buffer is full; inference must not be replayed')
        event_time = datetime.now().astimezone()
        self._buffer.append({
            'event_id':str(uuid.uuid4()),
            'event_date':event_time.date().isoformat(),
            'recorded_at':event_time.isoformat(),
            'recorded_at_unix':event_time.timestamp(),
            'provider':provider,'tenant':tenant,'request_id':request_id,
            'generation_outcome':generation_outcome,
            'usage_fields':sorted(usage_fields) if usage_fields is not None else None,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "is_error": is_error,
        })
        self._accepted_events += 1

    async def cleanup(self, keep_days: int) -> int:
        cutoff = date.today() - timedelta(days=keep_days)
        return await self._delete_before(cutoff)

    async def _periodic_flush(self):
        # P2.5 self-heal: 内层 try 已经把 flush + cleanup 包住；外层新增 sleep 也包
        # 在 try 里以防 CancelledError 之外的异常从 sleep 抛出（罕见但曾发生过）。
        delay = 30
        while True:
            try:
                await asyncio.sleep(delay)
                try:
                    await self._flush()
                    delay = 0 if self.persistence_stats()['pending_events'] >= FLUSH_BATCH_EVENTS else 30
                    if self.retention_days > 0:
                        today = date.today()
                        if self._last_cleanup_date != today:
                            deleted = await self._delete_before(today - timedelta(days=self.retention_days))
                            if deleted:
                                logger.info(f"Daily cleanup: deleted {deleted} expired records")
                            self._last_cleanup_date = today
                except Exception as e:
                    delay = 30
                    logger.error(f"Usage data flush error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                delay = 30
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
        # errors 与 totals 对称地按模型累计。注意：生产里**没有任何调用点**传
        # is_error=True（record() 只在成功路径被调），所以这一列目前恒为 0。
        # 保留并保持两个后端一致，是为了「哪天真的接上错误记账」时 JSON 与 MySQL
        # 行为相同 —— 从前 MySQL 侧恒写字面量 0，而 totals["errors"] 又是由这些
        # per-model 列求和重建的，于是 MySQL 后端重启即丢，JSON 后端不丢。
        return {"input_tokens": 0, "output_tokens": 0,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "requests": 0, "errors": 0}

    async def _flush(self):
        async with self._lock:
            if not self._buffer and not self._pending_groups:
                return
            # Retain moved events before any await. New record() calls append to
            # the independent buffer while this bounded snapshot is persisted.
            if not self._pending_groups:
                pending = self._buffer[:FLUSH_BATCH_EVENTS]
                del self._buffer[:len(pending)]
                groups = {}
                for event in pending:
                    groups.setdefault(event['event_date'],[]).append(event)
                self._pending_groups.extend((date.fromisoformat(day),events) for day,events in sorted(groups.items()))
            try:
                await self._ensure_backend()
                while self._pending_groups:
                    day,events = self._pending_groups[0]
                    if self._inflight_batch is None:
                        if day == self._today_date and self._today_cache:
                            base = self._today_cache
                        else:
                            async with asyncio.timeout(IO_TIMEOUT):
                                base = await self._load_day(day)
                        cumulative = copy.deepcopy(base) if base.get('models') else self._empty_day(day)
                        batch = self._empty_day(day)
                        batch.update(batch_id=str(uuid.uuid4()),schema_version=1,events=copy.deepcopy(events))
                        for event in events:
                            for target in (batch,cumulative):
                                m = target['models'].setdefault(event['model'],self._empty_model())
                                totals = target['totals']
                                for key in ('input_tokens','output_tokens','cache_creation_tokens','cache_read_tokens'):
                                    m[key] += event[key];totals[key] += event[key]
                                m['requests'] += 1;totals['requests'] += 1
                                if event.get('is_error'):
                                    m['errors'] += 1;totals['errors'] += 1
                        stamp = datetime.now().astimezone().isoformat()
                        batch['last_updated'] = cumulative['last_updated'] = stamp
                        self._inflight_batch = (day,batch,cumulative)
                    day,batch,cumulative = self._inflight_batch
                    async with asyncio.timeout(IO_TIMEOUT):
                        await self._save_day_delta(day,batch,cumulative)
                    # No await between acknowledgement and local settlement.
                    if day == date.today():
                        self._today_date,self._today_cache = day,cumulative
                    self._pending_groups.popleft()
                    self._inflight_batch = None
                    self._flush_successes += 1
                    self._persisted_events += len(events)
                    self._last_success = time.time()
                    self._backend_ready = True
                    self._last_error_type = None
            except Exception as exc:
                self._flush_failures += 1
                self._backend_ready = False
                self._last_error_type = type(exc).__name__
                raise

    async def _ensure_backend(self):
        pass

    def persistence_stats(self):
        return {'pending_events':len(self._buffer)+sum(len(events) for _,events in self._pending_groups),
                'pending_groups':len(self._pending_groups),'inflight_batch':int(self._inflight_batch is not None),
                'flush_failures_total':self._flush_failures,'flush_successes_total':self._flush_successes,
                'rejected_events_total':self._rejected_events,'last_success_timestamp_seconds':self._last_success,
                'backend_ready':int(self._backend_ready),'last_error_type':self._last_error_type}

    def extended_persistence_stats(self):
        now, oldest, unknown = time.time(), None, False
        events = itertools.chain(self._buffer, itertools.chain.from_iterable(group for _, group in self._pending_groups))
        for event in events:
            stamp = event.get('recorded_at_unix')
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0 or stamp > now:
                unknown = True; break
            oldest = stamp if oldest is None else min(oldest, stamp)
        return {'accepted_events_total': self._accepted_events, 'persisted_events_total': self._persisted_events,
                'oldest_pending_age_seconds': None if unknown else now-oldest if oldest is not None else 0,
                'volatile_buffer': 1}

    def render_extended_metrics(self):
        lines = []
        for key, value in self.extended_persistence_stats().items():
            name = 'lb_usage_' + key
            kind = 'counter' if key.endswith('_total') else 'gauge'
            lines.extend([f'# HELP {name} Accepted versus acknowledged usage; memory acceptance is not crash durability',
                          f'# TYPE {name} {kind}', f'{name} {"NaN" if value is None else value}'])
        return '\n'.join(lines) + '\n'

    def render_metrics(self):
        lines = []
        for key,value in self.persistence_stats().items():
            if key == 'last_error_type':
                continue
            name = 'lb_usage_'+key
            kind = 'counter' if key.endswith('_total') else 'gauge'
            lines.extend([f'# HELP {name} Usage persistence state; separate from generation outcome',
                          f'# TYPE {name} {kind}',f'{name} {value}'])
        return '\n'.join(lines)+'\n'

    def get_today_data(self) -> dict:
        return self._today_cache if self._today_cache else {}

    async def get_day_data(self, d: date) -> Optional[dict]:
        if d == self._today_date:
            return self.get_today_data()
        return await self._load_day(d) or None

    # 子类实现
    async def _backend_start(self): pass
    async def _backend_stop(self): pass
    async def _save_day(self, d: date, data: dict):
        """写「当天累计绝对值」（整天覆盖，单写者语义）。

        只被 ``_save_day_delta`` 的默认实现调用。支持原子累加的后端（MySQL）
        override 的是 ``_save_day_delta``，**不实现本方法**，所以在那些后端上直接
        调它是个 bug。这里显式抛错而不是 ``pass`` —— 从前的 ``pass`` 让误调变成
        「静默丢当天全部用量」，只有一行注释挡着。宁可启动/刷盘时炸掉。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _save_day; a backend that "
            "overrides _save_day_delta must never route through it"
        )
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
        def write():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + '.' + uuid.uuid4().hex + '.tmp'
            try:
                with open(tmp_path, 'x') as f:
                    json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
                    f.flush();os.fsync(f.fileno())
                os.replace(tmp_path,path)
                fd = os.open(os.path.dirname(path),os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
            finally:
                if os.path.exists(tmp_path):os.unlink(tmp_path)
        # Keep the worker owned on cancellation: a late old overwrite must not
        # race a newer batch. Disk work no longer blocks the application's loop.
        await _storage_cleanup(asyncio.to_thread(write))

    async def _load_day(self, d: date) -> dict:
        path = self._day_path(d)
        def read():
            try:
                with open(path,'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                return {}
            if not isinstance(data,dict) or not all(isinstance(data.get(k),dict) for k in ('models','totals')):
                raise ValueError('Invalid persisted usage day shape; refusing to overwrite')
            return data
        return await _storage_cleanup(asyncio.to_thread(read))

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
        self._backend_lock = asyncio.Lock()

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
            connect_timeout=IO_TIMEOUT,
        )
        try:
            await self._create_schema()
        except BaseException:
            await self._backend_stop()
            raise
        logger.info(f"MySQL usage store initialized: {self._mysql_config.get('host')}:{self._mysql_config.get('port')}/{self._mysql_config.get('database')}")

    async def _ensure_backend(self):
        if self._pool is None or getattr(self._pool,'closed',False):
            async with self._backend_lock:
                if self._pool is None or getattr(self._pool,'closed',False):
                    async with asyncio.timeout(IO_TIMEOUT):
                        await self._backend_start()

    async def _create_schema(self):
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
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS usage_batch_ledger (
                        batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
                        event_date DATE NOT NULL,
                        payload_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
                        payload JSON NULL,
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        PRIMARY KEY (batch_id),
                        KEY usage_batch_event_date (event_date)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

    async def _backend_stop(self):
        if self._pool:
            pool,self._pool = self._pool,None
            pool.close()
            await _storage_cleanup(pool.wait_closed())

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
            raise RuntimeError('MySQL usage pool is unavailable; batch remains pending')
        models = delta.get("models", {})
        if not models:
            return
        batch_id = delta.get('batch_id')
        if not isinstance(batch_id,str) or not batch_id:
            raise ValueError('A stable usage batch ID is required')
        payload = json.dumps(delta,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        payload_hash = hashlib.sha256(payload.encode()).hexdigest()
        import aiomysql
        async with self._pool.acquire() as conn:
            try:
                await conn.begin()
                async with conn.cursor() as cur:
                    try:
                        await cur.execute("""
                            INSERT INTO usage_batch_ledger (batch_id, event_date, payload_sha256, payload)
                            VALUES (%s, %s, %s, %s)
                        """,(batch_id,d.isoformat(),payload_hash,payload))
                    except aiomysql.IntegrityError as exc:
                        if exc.args[0] != 1062:
                            raise
                        await cur.execute('SELECT payload_sha256 FROM usage_batch_ledger WHERE batch_id = %s FOR UPDATE',(batch_id,))
                        row = await cur.fetchone()
                        if not row or row[0] != payload_hash:
                            raise ValueError('Usage batch ID was reused with different content') from exc
                        await _storage_cleanup(conn.rollback())
                        return  # The matching transaction was already committed.
                    # Stable lock ordering reduces multi-writer deadlock risk.
                    for model_name in sorted(models):
                        mstats = models[model_name]
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
                              mstats.get("errors", 0)))
                await conn.commit()
            except BaseException:
                try:
                    await _storage_cleanup(conn.rollback())
                except asyncio.CancelledError:
                    conn.close()
                    raise
                except BaseException:
                    conn.close()  # Never return a connection with uncertain transaction state.
                raise

    async def _load_day(self, d: date) -> dict:
        if not self._pool:
            raise RuntimeError('MySQL usage pool is unavailable')
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
            # 形状必须与 _empty_model() 一致，否则「进程内新建」与「从 DB 载入」
            # 的 models[m] 键集不同，消费方一旦不用 .get() 就会 KeyError。
            models[model] = {"input_tokens": inp, "output_tokens": out, "cache_creation_tokens": cc,
                             "cache_read_tokens": cr, "requests": reqs, "errors": errs}
            totals["input_tokens"] += inp
            totals["output_tokens"] += out
            totals["cache_creation_tokens"] += cc
            totals["cache_read_tokens"] += cr
            totals["requests"] += reqs
            totals["errors"] += errs
        return {"date": d.isoformat(), "models": models, "totals": totals}

    async def get_day_data(self, d: date) -> Optional[dict]:
        # The backend, rather than a per-process cache, owns the multi-pod total.
        await self._ensure_backend()
        async with asyncio.timeout(IO_TIMEOUT):
            return await self._load_day(d) or None

    async def _delete_before(self, cutoff: date) -> int:
        if not self._pool:
            raise RuntimeError('MySQL usage pool is unavailable')
        async with self._pool.acquire() as conn:
            try:
                await conn.begin()
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM usage_daily WHERE date < %s", (cutoff.isoformat(),))
                    deleted = cur.rowcount
                    # Keep only the minimum idempotency receipt. Event details
                    # follow the same retention cutoff as the daily aggregate.
                    await cur.execute('UPDATE usage_batch_ledger SET payload = NULL WHERE event_date < %s', (cutoff.isoformat(),))
                await conn.commit()
                return deleted
            except BaseException:
                try:
                    await _storage_cleanup(conn.rollback())
                except asyncio.CancelledError:
                    conn.close()
                    raise
                except BaseException:
                    conn.close()
                raise


def create_usage_store(storage_config: dict) -> UsageDataStore:
    store_type = storage_config.get("type", "json")
    retention = storage_config.get("retention_days", 0)
    if store_type == "mysql":
        # 只探测可用性，不真的导入 —— 从前写成 `import aiomysql  # noqa` 会被静态
        # 检查报「未使用」，把真正的告警埋在噪音里。
        if importlib.util.find_spec("aiomysql") is None:
            logger.critical(
                "usage_storage.type=mysql but aiomysql is not installed. "
                "Run: pip install aiomysql>=0.2.0 — refusing to start to avoid silent data loss."
            )
            sys.exit(1)
        return MysqlUsageStore(storage_config, retention)
    return JsonUsageStore(storage_config.get("path", "./usage_data"), retention)
