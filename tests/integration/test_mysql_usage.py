"""Real MySQL transaction checks against a disposable loopback test server.

Opt in only with LB_TEST_MYSQL_ISOLATED=1. Tests create/drop their own UUID-named
database; they never read LB config or production credentials. Author: Zeno Ren.
"""
import asyncio
import os
import unittest
import uuid
from datetime import date, timedelta

import aiomysql
from usage_store import MysqlUsageStore


class CursorWrapper:
    def __init__(self, cursor, faults):
        self.cursor, self.faults = cursor, faults
    def __getattr__(self, name):return getattr(self.cursor,name)
    async def execute(self, sql, params=None):
        if 'INSERT INTO usage_daily' in sql and params[1] == self.faults.get('model'):
            raise OSError('injected error before second model write')
        return await self.cursor.execute(sql,params)


class CursorContext:
    def __init__(self, context, faults):self.context,self.faults=context,faults
    async def __aenter__(self):return CursorWrapper(await self.context.__aenter__(),self.faults)
    async def __aexit__(self,*exc):return await self.context.__aexit__(*exc)


class ConnectionWrapper:
    def __init__(self, connection, faults):self.connection,self.faults=connection,faults
    def __getattr__(self,name):return getattr(self.connection,name)
    def cursor(self):return CursorContext(self.connection.cursor(),self.faults)
    async def commit(self):
        await self.connection.commit()
        if self.faults.pop('ack',False):
            raise OSError('injected lost commit acknowledgement')
        if self.faults.get('pause_after_commit'):
            self.faults['committed'].set()
            await asyncio.Event().wait()


class AcquireContext:
    def __init__(self, context, faults):self.context,self.faults=context,faults
    async def __aenter__(self):return ConnectionWrapper(await self.context.__aenter__(),self.faults)
    async def __aexit__(self,*exc):return await self.context.__aexit__(*exc)


class PoolWrapper:
    def __init__(self,pool,faults):self.pool,self.faults=pool,faults
    def __getattr__(self,name):return getattr(self.pool,name)
    def acquire(self):return AcquireContext(self.pool.acquire(),self.faults)


@unittest.skipUnless(os.getenv('LB_TEST_MYSQL_ISOLATED')=='1','requires disposable loopback MySQL test server')
class RealMysqlUsageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.database='lb_usage_test_'+uuid.uuid4().hex[:16]
        # Passwordless access is only for the isolated, network=none fixture;
        # production MysqlUsageStore._backend_start still requires verified TLS.
        self.admin=await aiomysql.create_pool(host='127.0.0.1',user='root',password='',autocommit=True)
        async with self.admin.acquire() as conn:
            async with conn.cursor() as cur:await cur.execute('CREATE DATABASE '+self.database)
        self.pool=await aiomysql.create_pool(host='127.0.0.1',user='root',password='',db=self.database,
                                           autocommit=True,minsize=1,maxsize=5)
        self.faults={}
        self.store=MysqlUsageStore({'database':self.database})
        self.store._pool=PoolWrapper(self.pool,self.faults)
        await self.store._create_schema()

    async def asyncTearDown(self):
        self.pool.close();await self.pool.wait_closed()
        async with self.admin.acquire() as conn:
            async with conn.cursor() as cur:await cur.execute('DROP DATABASE '+self.database)
        self.admin.close();await self.admin.wait_closed()

    async def counts(self):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute('SELECT model,input_tokens FROM usage_daily ORDER BY model')
                tokens=dict(await cur.fetchall())
                await cur.execute('SELECT COUNT(*) FROM usage_batch_ledger')
                ledger=(await cur.fetchone())[0]
        return tokens,ledger

    async def test_partial_write_rolls_back_both_rows_and_ledger(self):
        self.store.record('a',7,1);self.store.record('b',3,1)
        self.faults['model']='b'
        with self.assertRaises(OSError):await self.store._flush()
        self.assertEqual(await self.counts(),({},0))
        self.faults.clear();await self.store._flush()
        self.assertEqual(await self.counts(),({'a':7,'b':3},1))

    async def test_commit_ack_loss_is_idempotent(self):
        self.store.record('m',5,1);self.faults['ack']=True
        with self.assertRaises(OSError):await self.store._flush()
        self.assertEqual(await self.counts(),({'m':5},1))
        await self.store._flush()
        self.assertEqual(await self.counts(),({'m':5},1))
        self.assertEqual(self.store.persistence_stats()['pending_events'],0)

    async def test_two_writers_read_the_shared_daily_total(self):
        other=MysqlUsageStore({'database':self.database});other._pool=self.pool
        self.store.record('m',5,1);other.record('m',7,1)
        await asyncio.gather(self.store._flush(),other._flush())
        self.assertEqual(await self.counts(),({'m':12},2))
        from datetime import date
        self.assertEqual((await self.store.get_day_data(date.today()))['models']['m']['input_tokens'],12)
        self.assertEqual((await other.get_day_data(date.today()))['models']['m']['input_tokens'],12)

    async def test_concurrent_retries_of_same_batch_apply_once(self):
        self.store.record('m',5,1);self.faults['model']='m'
        with self.assertRaises(OSError):await self.store._flush()
        day,batch,cumulative=self.store._inflight_batch
        self.faults.clear()
        other=MysqlUsageStore({'database':self.database});other._pool=self.pool
        await asyncio.gather(self.store._flush(),other._save_day_delta(day,batch,cumulative))
        self.assertEqual(await self.counts(),({'m':5},1))

    async def test_cancel_after_commit_is_recoverable_without_duplicate_usage(self):
        self.store.record('m',5,1)
        self.faults.update(pause_after_commit=True,committed=asyncio.Event())
        task=asyncio.create_task(self.store._flush())
        await self.faults['committed'].wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.faults.clear();await self.store._flush()
        self.assertEqual(await self.counts(),({'m':5},1))

    async def test_retention_erases_event_details_but_keeps_retry_receipt(self):
        self.store.record('m',5,1);self.faults['ack']=True
        with self.assertRaises(OSError):await self.store._flush()
        await self.store._delete_before(date.today()+timedelta(days=1))
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute('SELECT payload FROM usage_batch_ledger')
                self.assertIsNone((await cur.fetchone())[0])
        await self.store._flush()
        self.assertEqual(await self.counts(),({},1))
