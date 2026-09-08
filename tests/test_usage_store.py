"""usage_store 持久化语义测试.

重点是**多写者安全**：多副本部署下每个 pod 各自持有 `_today_cache`（当天累计），
若落盘走「绝对覆盖」，两个 pod 每 30s 互相把对方的量抹掉，当天用量/成本静默丢失。
`MysqlUsageStore` 必须走 `col = col + VALUES(col)` 增量累加；`JsonUsageStore`
文件后端无法跨进程原子累加，仍是单写全文件覆盖（因此 JSON 后端只能单副本）。

本机无 MySQL，用 fake pool/cursor 捕获 SQL 与参数。
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usage_store import JsonUsageStore, MysqlUsageStore, UsageDataStore


class FakeCursor:
    """记录每次 execute 的 (sql, params)，供断言检查。"""

    def __init__(self, sink: list, rows=None):
        self._sink = sink
        self._rows = rows or []
        self.rowcount = 0

    async def execute(self, sql, params=None):
        self._sink.append((sql, params))

    async def fetchall(self):
        return self._rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, sink: list, rows=None):
        self._sink = sink
        self._rows = rows

    def cursor(self):
        return FakeCursor(self._sink, self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakePoolAcquire:
    def __init__(self, sink: list, rows=None):
        self._sink = sink
        self._rows = rows

    async def __aenter__(self):
        return FakeConn(self._sink, self._rows)

    async def __aexit__(self, *exc):
        return False


class FakePool:
    """只实现 acquire()，够 _save_day_delta / _load_day / _delete_before 用。"""

    def __init__(self, rows=None):
        self.executed: list = []
        self._rows = rows

    def acquire(self):
        return FakePoolAcquire(self.executed, self._rows)


def _mysql_store(rows=None) -> MysqlUsageStore:
    """跳过 _backend_start（不连真库），直接塞 fake pool。"""
    store = MysqlUsageStore({"host": "fake", "database": "fake"})
    store._pool = FakePool(rows=rows)
    store._today_date = date.today()
    store._today_cache = {}
    return store


def _upsert_calls(pool: FakePool) -> list:
    return [(sql, params) for sql, params in pool.executed if "INSERT INTO usage_daily" in sql]


class MysqlIncrementalUpsertTests(unittest.IsolatedAsyncioTestCase):
    """MySQL 后端必须增量累加，不能绝对覆盖。"""

    async def test_upsert_sql_accumulates_instead_of_overwriting(self):
        store = _mysql_store()
        store.record("gpt-5.6-sol", input_tokens=10, output_tokens=5)
        await store._flush()

        calls = _upsert_calls(store._pool)
        self.assertEqual(len(calls), 1, "一个 model 应产生一条 upsert")
        sql = calls[0][0]
        # 累加形式必须在
        for col in ("input_tokens", "output_tokens", "cache_creation_tokens",
                    "cache_read_tokens", "requests"):
            self.assertIn(f"{col} = {col} + VALUES({col})", sql,
                          f"{col} 必须增量累加，否则多副本互相覆盖")
        # 绝对覆盖形式必须消失（注意 `col = col + VALUES(col)` 不含 `= VALUES(col)`
        # 这个子串，所以下面的断言是有效的）
        self.assertNotIn("input_tokens = VALUES(input_tokens)", sql,
                         "绝对覆盖写法会在多副本下丢数据")

    async def test_flush_persists_batch_delta_not_running_total(self):
        """第二次 flush 必须只写本批增量，而不是当天累计。"""
        store = _mysql_store()

        store.record("gpt-5.6-sol", input_tokens=10, output_tokens=5)
        await store._flush()
        store.record("gpt-5.6-sol", input_tokens=3, output_tokens=2)
        await store._flush()

        calls = _upsert_calls(store._pool)
        self.assertEqual(len(calls), 2)
        # params 顺序: (date, model, input, output, cache_creation, cache_read, requests, errors)
        first_input, second_input = calls[0][1][2], calls[1][1][2]
        self.assertEqual(first_input, 10)
        self.assertEqual(second_input, 3, "第二批应写增量 3，写 13 说明退回了累计语义")
        self.assertEqual(calls[1][1][6], 1, "requests 同样是本批增量")

        # 内存里的当天视图仍是累计（/stats 依赖它）
        self.assertEqual(store.get_today_data()["models"]["gpt-5.6-sol"]["input_tokens"], 13)
        self.assertEqual(store.get_today_data()["totals"]["requests"], 2)

    async def test_two_writers_sum_instead_of_clobber(self):
        """双副本回归：两个进程各自从同一起点累加，落盘之和必须等于真实总量。

        旧的绝对覆盖实现下，两次 execute 的参数分别是各自的「起点 + 自己的量」，
        DB 最终只保留最后写入的那份 → 断言失败。
        """
        # 两个 pod 启动时都从 DB 读到同一天已有 100 input tokens
        existing_rows = [("gpt-5.6-sol", 100, 50, 0, 0, 4, 0)]

        pod_a = _mysql_store(rows=existing_rows)
        pod_b = _mysql_store(rows=existing_rows)
        today = date.today()
        pod_a._today_cache = await pod_a._load_day(today)
        pod_b._today_cache = await pod_b._load_day(today)
        # 两者起点一致，且都看到 100
        self.assertEqual(pod_a._today_cache["models"]["gpt-5.6-sol"]["input_tokens"], 100)
        self.assertEqual(pod_b._today_cache["models"]["gpt-5.6-sol"]["input_tokens"], 100)

        pod_a.record("gpt-5.6-sol", input_tokens=10, output_tokens=1)
        pod_b.record("gpt-5.6-sol", input_tokens=7, output_tokens=1)
        await pod_a._flush()
        await pod_b._flush()

        a_written = _upsert_calls(pod_a._pool)[0][1][2]
        b_written = _upsert_calls(pod_b._pool)[0][1][2]
        # 增量语义下 DB 行 = 100 + 10 + 7 = 117
        self.assertEqual(100 + a_written + b_written, 117,
                         "两副本的落盘增量之和必须等于真实总量；"
                         f"实际 a={a_written} b={b_written}（绝对覆盖下会是 110/107）")

    async def test_errors_column_stays_zero_known_gap(self):
        """per-model errors 目前恒写 0（models[m] 里没有 errors 字段）。

        这是既有缺口，本次不修；锁住行为避免无意改动。
        """
        store = _mysql_store()
        store.record("gpt-5.6-sol", input_tokens=1, output_tokens=1, is_error=True)
        await store._flush()
        calls = _upsert_calls(store._pool)
        self.assertEqual(calls[0][1][7], 0, "per-model errors 仍写 0")
        # 但 totals 层面记到了
        self.assertEqual(store.get_today_data()["totals"]["errors"], 1)

    async def test_empty_batch_writes_nothing(self):
        store = _mysql_store()
        await store._flush()
        self.assertEqual(_upsert_calls(store._pool), [])


class JsonStoreUnchangedTests(unittest.IsolatedAsyncioTestCase):
    """JSON 后端走基类默认路径，仍是整天绝对覆盖（单写者语义）。"""

    async def test_json_file_holds_full_day_cumulative(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonUsageStore(tmp)
            store._today_date = date.today()
            store._today_cache = {}

            store.record("claude-opus-5", input_tokens=10, output_tokens=5)
            await store._flush()
            store.record("claude-opus-5", input_tokens=3, output_tokens=2)
            await store._flush()

            today = date.today()
            path = os.path.join(tmp, str(today.year), f"{today.month:02d}",
                                f"{today.isoformat()}.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            # 文件是当天累计（不是最后一批增量）
            self.assertEqual(data["models"]["claude-opus-5"]["input_tokens"], 13)
            self.assertEqual(data["totals"]["requests"], 2)

    async def test_base_default_delta_hook_falls_back_to_save_day(self):
        """基类默认 _save_day_delta 必须把累计值交给 _save_day，
        这样任何未 override 的后端行为都不变。"""
        seen = {}

        class Recording(UsageDataStore):
            async def _save_day(self, d, data):
                seen["cumulative"] = json.loads(json.dumps(data))

        store = Recording()
        store._today_date = date.today()
        store._today_cache = {}
        store.record("m", input_tokens=4, output_tokens=1)
        await store._flush()
        store.record("m", input_tokens=6, output_tokens=1)
        await store._flush()

        self.assertEqual(seen["cumulative"]["models"]["m"]["input_tokens"], 10)


class DeltaShapeTests(unittest.IsolatedAsyncioTestCase):
    """delta 与 cumulative 是两份独立数据，不能共享同一个可变对象。"""

    async def test_delta_is_not_the_same_object_as_cumulative(self):
        captured = {}

        class Capturing(UsageDataStore):
            async def _save_day_delta(self, d, delta, cumulative):
                captured["delta"] = delta
                captured["cumulative"] = cumulative

        store = Capturing()
        store._today_date = date.today()
        store._today_cache = {}
        store.record("m", input_tokens=5, output_tokens=1)
        await store._flush()

        self.assertIsNot(captured["delta"], captured["cumulative"])
        self.assertIsNot(captured["delta"]["models"]["m"],
                         captured["cumulative"]["models"]["m"])
        # 第一批时两者数值相同
        self.assertEqual(captured["delta"]["models"]["m"]["input_tokens"], 5)
        self.assertEqual(captured["cumulative"]["models"]["m"]["input_tokens"], 5)

        # 第二批时 delta 只含本批
        store.record("m", input_tokens=2, output_tokens=1)
        await store._flush()
        self.assertEqual(captured["delta"]["models"]["m"]["input_tokens"], 2)
        self.assertEqual(captured["cumulative"]["models"]["m"]["input_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
