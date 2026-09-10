"""usage_store 持久化语义测试.

重点是**多写者安全**：多副本部署下每个 pod 各自持有 `_today_cache`（当天累计），
若落盘走「绝对覆盖」，两个 pod 每 30s 互相把对方的量抹掉，当天用量/成本静默丢失。
`MysqlUsageStore` 必须走 `col = col + VALUES(col)` 增量累加；`JsonUsageStore`
文件后端无法跨进程原子累加，仍是单写全文件覆盖（因此 JSON 后端只能单副本）。

本机无 MySQL，用 fake pool/cursor 捕获 SQL 与参数。
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import usage_store as usage_store_mod
from usage_store import (JsonUsageStore, MysqlUsageStore, UsageDataStore,
                         create_usage_store)


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

    async def test_per_model_errors_are_persisted_not_hardcoded_zero(self):
        """per-model errors 必须写真实增量。

        从前这里恒写字面量 0，而 MySQL 的 `_load_day` 又是**由这些 per-model 列求和**
        重建 `totals["errors"]` 的 —— 于是 MySQL 后端上错误计数重启即归零，JSON 后端
        却保留。两个后端对同一份数据给出不同答案，是比「计数不准」更麻烦的问题。

        诚实边界：生产里没有任何调用点传 `is_error=True`（`record()` 只在成功路径被
        调用），所以这一列现实中恒为 0。修的是**一致性**，不是把死链路接活 —— 接活
        属于新功能（错误请求算哪个模型、有没有 token 都要先定义）。
        """
        store = _mysql_store()
        store.record("gpt-5.6-sol", input_tokens=1, output_tokens=1, is_error=True)
        store.record("gpt-5.6-sol", input_tokens=1, output_tokens=1)
        await store._flush()
        calls = _upsert_calls(store._pool)
        self.assertEqual(calls[0][1][7], 1, "per-model errors 必须是真实增量")
        self.assertEqual(store.get_today_data()["models"]["gpt-5.6-sol"]["errors"], 1)
        self.assertEqual(store.get_today_data()["totals"]["errors"], 1)

    async def test_mysql_load_day_shape_matches_empty_model(self):
        """`_load_day` 造出的 models[m] 键集必须与 `_empty_model()` 完全一致。

        否则「进程内新建」与「从 DB 载入」两条路径的字典形状不同，消费方一旦不用
        `.get()` 取值就 KeyError —— 而这种差异只会在重启后才暴露。
        """
        store = _mysql_store(rows=[("gpt-5.6-sol", 10, 5, 0, 3, 2, 1)])
        loaded = await store._load_day(date.today())
        self.assertEqual(set(loaded["models"]["gpt-5.6-sol"]),
                         set(UsageDataStore._empty_model()))
        self.assertEqual(loaded["models"]["gpt-5.6-sol"]["errors"], 1)
        self.assertEqual(loaded["totals"]["errors"], 1)

    async def test_base_save_day_fails_loud_instead_of_dropping_a_whole_day(self):
        """基类 `_save_day` 从前是 `pass` —— 未实现它的后端（MySQL）上误调等于
        「静默丢弃当天全部用量」，只有一行注释挡着。现在必须炸。
        """
        store = _mysql_store()
        with self.assertRaises(NotImplementedError):
            await store._save_day(date.today(), UsageDataStore._empty_day(date.today()))
        # 正常路径（override 了 _save_day_delta）不受影响
        store.record("m", input_tokens=1, output_tokens=1)
        await store._flush()
        self.assertEqual(len(_upsert_calls(store._pool)), 1)

    async def test_empty_batch_writes_nothing(self):
        store = _mysql_store()
        await store._flush()
        self.assertEqual(_upsert_calls(store._pool), [])


class JsonStoreUnchangedTests(unittest.IsolatedAsyncioTestCase):
    """JSON 后端走基类默认路径，仍是整天绝对覆盖（单写者语义）。"""

    async def test_json_round_trips_per_model_errors(self):
        """两个后端对同一份数据必须给出相同答案 —— 这是修 per-model errors 的目的。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonUsageStore(tmp)
            store._today_date = date.today()
            store._today_cache = {}
            store.record("m", input_tokens=1, output_tokens=1, is_error=True)
            store.record("m", input_tokens=1, output_tokens=1)
            await store._flush()

            reloaded = await store._load_day(date.today())
            self.assertEqual(reloaded["models"]["m"]["errors"], 1)
            self.assertEqual(reloaded["totals"]["errors"], 1)
            self.assertEqual(set(reloaded["models"]["m"]) - {"estimated_cost_usd"},
                             set(UsageDataStore._empty_model()),
                             "JSON 往返后的键集也要与 _empty_model() 一致")

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


class StoreLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """`start` / `stop` / `_periodic_flush` / retention —— 之前全部零覆盖。

    这四块的失败模式都是**静默丢用量**：
      - `stop` 不做最终 flush → 每次重启丢最后 ≤30s
      - `start` 的宽 `except` 吞掉 `_load_day` 失败 → 当天缓存从 0 开始，
        随后一次 flush 就把整天覆盖成「只有这一批」（多副本那节警告的模式）
      - `_periodic_flush` 抛出未捕获异常 → 循环死掉，之后再也不落盘
    """

    async def test_stop_flushes_the_tail_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonUsageStore(tmp)
            await store.start()
            store.record("m", input_tokens=7, output_tokens=3)
            await store.stop()   # 不能等 30s 周期
            reloaded = await store._load_day(date.today())
            self.assertEqual(reloaded["models"]["m"]["input_tokens"], 7,
                             "stop 必须落盘尾批，否则每次重启都丢最后一段用量")

    async def test_start_seeds_today_cache_from_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = JsonUsageStore(tmp)
            await first.start()
            first.record("m", input_tokens=10, output_tokens=1)
            await first.stop()

            second = JsonUsageStore(tmp)
            await second.start()
            try:
                self.assertEqual(second.get_today_data()["models"]["m"]["input_tokens"], 10,
                                 "重启必须把当天已有用量种回缓存")
                second.record("m", input_tokens=5, output_tokens=1)
                await second._flush()
                self.assertEqual(second.get_today_data()["models"]["m"]["input_tokens"], 15,
                                 "新增量要累加到种回的起点上，不是覆盖")
            finally:
                await second.stop()

    async def test_start_survives_a_backend_failure_but_says_so(self):
        """`_load_day` 失败被吞掉是有意的（不让统计拖垮启动），但必须留日志 ——
        否则「当天数据凭空归零」查不出原因。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonUsageStore(tmp)
            with patch.object(store, "_load_day", side_effect=OSError("disk gone")):
                with self.assertLogs(usage_store_mod.logger, level="ERROR"):
                    await store.start()
            self.assertEqual(store.get_today_data(), {})
            await store.stop()

    async def test_periodic_flush_survives_a_flush_error(self):
        """一次 flush 抛异常不能让整个周期任务死掉（P2.5 自愈）。"""
        store = JsonUsageStore("/nonexistent-should-not-be-written")
        store._today_date = date.today()
        store._today_cache = {}
        calls = []

        async def boom():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient")

        sleeps = []

        async def fake_sleep(sec):
            sleeps.append(sec)
            if len(sleeps) >= 3:
                raise asyncio.CancelledError()

        with patch.object(store, "_flush", side_effect=boom), \
             patch.object(usage_store_mod.asyncio, "sleep", side_effect=fake_sleep):
            await store._periodic_flush()

        self.assertGreaterEqual(len(calls), 2,
                                "第一次 flush 抛异常后循环必须继续调用 flush")

    async def test_retention_deletes_only_files_before_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonUsageStore(tmp)
            for iso in ("2026-01-01", "2026-01-02", "2026-06-15"):
                d = date.fromisoformat(iso)
                p = os.path.join(tmp, str(d.year), f"{d.month:02d}")
                os.makedirs(p, exist_ok=True)
                with open(os.path.join(p, f"{iso}.json"), "w") as f:
                    json.dump(UsageDataStore._empty_day(d), f)
            # 放一个非日期文件名和一个非 json，必须被跳过而不是崩
            os.makedirs(os.path.join(tmp, "2026", "01"), exist_ok=True)
            open(os.path.join(tmp, "2026", "01", "notes.txt"), "w").close()
            open(os.path.join(tmp, "2026", "01", "garbage.json"), "w").close()

            deleted = await store._delete_before(date.fromisoformat("2026-02-01"))
            self.assertEqual(deleted, 2, "只删 cutoff 之前的两天")
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "2026", "06", "2026-06-15.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "2026", "01", "notes.txt")))

    async def test_retention_ignores_nonexistent_dir(self):
        store = JsonUsageStore("/definitely/not/here")
        self.assertEqual(await store._delete_before(date.today()), 0)


class FactoryTests(unittest.TestCase):

    def test_defaults_to_json_backend(self):
        store = create_usage_store({})
        self.assertIsInstance(store, JsonUsageStore)
        self.assertEqual(store.base_dir, "./usage_data")
        self.assertEqual(store.retention_days, 0)

    def test_json_backend_honours_path_and_retention(self):
        store = create_usage_store({"type": "json", "path": "/tmp/x", "retention_days": 30})
        self.assertEqual(store.base_dir, "/tmp/x")
        self.assertEqual(store.retention_days, 30)

    def test_mysql_backend_refuses_to_start_without_aiomysql(self):
        """缺驱动时必须 exit 而不是静默退回 JSON —— 后者会让多副本部署悄悄丢用量
        （JSON 后端永远单副本）。"""
        with patch.dict(sys.modules, {"aiomysql": None}):
            with self.assertRaises(SystemExit):
                create_usage_store({"type": "mysql", "host": "h", "database": "d"})
