"""Persistence failure, uncertain commit and midnight tests; no production database."""
import asyncio
import copy
from datetime import date, datetime, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import usage_store
from usage_store import UsageDataStore, JsonUsageStore, MysqlUsageStore
import aiomysql


class RecordingStore(UsageDataStore):
    def __init__(self):
        super().__init__();self.fail=True;self.saved=[];self.attempts=[]
    async def _save_day_delta(self,day,delta,cumulative):
        self.attempts.append(copy.deepcopy(delta))
        if self.fail:raise OSError('synthetic persistence failure')
        self.saved.append((day,copy.deepcopy(delta),copy.deepcopy(cumulative)))


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_backlog_drains_without_sleeping_a_full_period_between_batches(self):
        store=RecordingStore();store.fail=False
        for _ in range(3):store.record('m',1,1)
        sleeps=[]
        async def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps)==3:raise asyncio.CancelledError
        with patch.object(usage_store,'FLUSH_BATCH_EVENTS',1),patch.object(usage_store.asyncio,'sleep',side_effect=sleep):
            await store._periodic_flush()
        self.assertEqual(sleeps,[30,0,0])

    async def test_invalid_usage_is_rejected_without_poisoning_valid_pending_events(self):
        store=RecordingStore();store.record('m',1,1)
        for value in (-1,None,'1',True,float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):store.record('m',value,1)
        self.assertEqual(store.persistence_stats()['pending_events'],1)

    async def test_provider_and_tenant_are_recorded_by_each_proxy(self):
        import main
        sink=RecordingStore();sink.fail=False
        endpoints=[main.WorkspaceEndpoint('d','https://fixture.invalid','synthetic'),
                   main.AzureOpenAIEndpoint('a','https://fixture.invalid','synthetic',deployments=['m']),
                   main.CopilotEndpoint('c','synthetic')]
        proxies=[cls(main.LoadBalancer([ep]),'synthetic') for cls,ep in zip(
            (main.ClaudeProxy,main.AzureOpenAIProxy,main.CopilotProxy),endpoints)]
        token=main._CURRENT_TENANT.set('monitor')
        try:
            with patch.object(main,'usage_store',sink):
                for proxy,ep in zip(proxies,endpoints):proxy._record_usage(ep,'m',1,2,.1)
            await sink._flush()
            events=sink.saved[0][1]['events']
            self.assertEqual({e['provider'] for e in events},{'databricks','azure_openai','copilot'})
            self.assertEqual({e['tenant'] for e in events},{'monitor'})
        finally:
            main._CURRENT_TENANT.reset(token)
            for proxy in proxies:await proxy.close()

    async def test_startup_does_not_assign_shared_gpt_history_to_databricks(self):
        import main
        from unittest.mock import AsyncMock
        proxy=main.ClaudeProxy(main.LoadBalancer([]),'synthetic')
        class History:
            start=AsyncMock();stop=AsyncMock()
            def get_today_data(self):
                return {'models':{'gpt-test':{'requests':7,'errors':5},'databricks-claude-opus-5':{'requests':2,'errors':1}},'totals':{'requests':9,'errors':6}}
        with patch.object(main,'load_config',return_value=(proxy,None,None,{})),patch.object(main,'create_usage_store',return_value=History()),\
                patch('otel_setup.setup_tracing',return_value=False),patch.object(main,'proxy',None),\
                patch.object(main,'azure_proxy',None),patch.object(main,'copilot_proxy',None),patch.object(main,'usage_store',None):
            async with main.lifespan(main.app):
                self.assertEqual(proxy.global_stats.total_requests,2)
                self.assertNotIn('gpt-test',proxy.today_model_stats)
                self.assertEqual(proxy.global_stats.total_errors,0)

    async def test_failed_batch_is_retried_without_losing_new_events_or_double_cache(self):
        store=RecordingStore();store.record('m',10,1)
        with self.assertRaises(OSError):await store._flush()
        store.record('m',3,2);store.fail=False
        await store._flush();await store._flush()
        self.assertEqual(sum(row[1]['models']['m']['input_tokens'] for row in store.saved),13)
        self.assertEqual(store.get_today_data()['models']['m']['input_tokens'],13)
        self.assertEqual(store.attempts[0]['batch_id'],store.attempts[1]['batch_id'])
        self.assertEqual(store.persistence_stats()['pending_events'],0)

    async def test_day_load_failure_keeps_events_for_later(self):
        store=RecordingStore();store.fail=False;store.record('m',7,1)
        with patch.object(store,'_load_day',side_effect=OSError('load unavailable')):
            with self.assertRaises(OSError):await store._flush()
        await store._flush()
        self.assertEqual(store.saved[0][1]['models']['m']['input_tokens'],7)

    async def test_cancelled_save_keeps_same_batch_identity(self):
        store=RecordingStore();store.record('m',4,1)
        entered=asyncio.Event()
        async def interrupted(day,delta,cumulative):
            store.attempts.append(copy.deepcopy(delta));entered.set();await asyncio.Event().wait()
        with patch.object(store,'_save_day_delta',side_effect=interrupted):
            task=asyncio.create_task(store._flush());await entered.wait();task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        store.fail=False;await store._flush()
        self.assertEqual(store.attempts[0]['batch_id'],store.saved[0][1]['batch_id'])

    async def test_json_ack_loss_replays_identical_cumulative_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            store=JsonUsageStore(root);store.record('m',5,1)
            save=store._save_day
            async def lose_ack(day,payload):
                await save(day,payload);raise OSError('ack lost after replace')
            with patch.object(store,'_save_day',side_effect=lose_ack):
                with self.assertRaises(OSError):await store._flush()
            await store._flush()
            saved=await store._load_day(date.today())
            self.assertEqual(saved['models']['m']['input_tokens'],5)

    async def test_json_write_failure_remains_pending(self):
        with tempfile.TemporaryDirectory() as root:
            store=JsonUsageStore(root);store.record('m',5,1)
            with patch.object(usage_store.os,'replace',side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):await store._flush()
            self.assertEqual(store.persistence_stats()['pending_events'],1)
            await store._flush()
            self.assertEqual((await store._load_day(date.today()))['models']['m']['input_tokens'],5)

    async def test_corrupt_json_is_not_treated_as_an_empty_day_and_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            store=JsonUsageStore(root);path=Path(store._day_path(date.today()))
            path.parent.mkdir(parents=True);path.write_text('{corrupt')
            store.record('m',5,1)
            with self.assertRaises(ValueError):await store._flush()
            self.assertEqual(path.read_text(),'{corrupt')
            self.assertEqual(store.persistence_stats()['pending_events'],1)

    async def test_midnight_uses_event_day_instead_of_flush_day(self):
        class Clock(datetime):
            current=datetime(2026,9,19,12,tzinfo=timezone.utc)
            @classmethod
            def now(cls,tz=None):return cls.current
        store=RecordingStore();store.fail=False
        with patch.object(usage_store,'datetime',Clock):
            store.record('m',10,1)
            Clock.current=datetime(2026,9,20,12,tzinfo=timezone.utc)
            store.record('m',3,1)
            await store._flush()
        totals={day.isoformat():batch['models']['m']['input_tokens'] for day,batch,_ in store.saved}
        self.assertEqual(totals,{'2026-09-19':10,'2026-09-20':3})

    async def test_start_failure_still_keeps_background_recovery_running(self):
        store=RecordingStore()
        with patch.object(store,'_backend_start',side_effect=OSError('temporarily unavailable')):
            await store.start()
        self.assertIsNotNone(store._flush_task)
        self.assertFalse(store._flush_task.done())
        await store.stop()

    async def test_stop_closes_backend_even_when_final_save_fails(self):
        store=RecordingStore();store.record('m',1,1);closed=[]
        async def close():closed.append(True)
        with patch.object(store,'_backend_stop',side_effect=close):
            with self.assertRaises(OSError):await store.stop()
        self.assertEqual(closed,[True])

    async def test_record_carries_source_and_event_identity(self):
        store=RecordingStore();store.fail=False
        store.record('m',1,1,provider='copilot',tenant='monitor',request_id='r-synthetic')
        await store._flush()
        event=store.saved[0][1]['events'][0]
        self.assertTrue(event['event_id'])
        self.assertEqual(event['provider'],'copilot')
        self.assertEqual(event['tenant'],'monitor')
        self.assertEqual(event['request_id'],'r-synthetic')


class TransactionPool:
    """A transactional fake: autocommit writes are deliberately irreversible."""
    def __init__(self):
        self.tokens={};self.ledger={};self.staged=None;self.begins=0;self.commits=0;self.rollbacks=0
        self.fail_model=None;self.lose_ack=False;self.row=None
    def acquire(self):return self
    async def __aenter__(self):return self
    async def __aexit__(self,*exc):return False
    def cursor(self):return self
    async def begin(self):
        self.begins+=1;self.staged=(copy.deepcopy(self.tokens),copy.deepcopy(self.ledger))
    async def commit(self):
        self.commits+=1
        self.tokens,self.ledger=self.staged;self.staged=None
        if self.lose_ack:
            self.lose_ack=False;raise OSError('commit succeeded, ack lost')
    async def rollback(self):self.rollbacks+=1;self.staged=None
    def close(self):pass
    async def execute(self,sql,params=None):
        tokens,ledger=self.staged if self.staged is not None else (self.tokens,self.ledger)
        if 'INSERT INTO usage_batch_ledger' in sql:
            if params[0] in ledger:raise aiomysql.IntegrityError(1062,'duplicate batch')
            ledger[params[0]]=params[2]
        elif 'SELECT payload_sha256' in sql:
            self.row=(ledger[params[0]],)
        elif 'INSERT INTO usage_daily' in sql:
            if params[1]==self.fail_model:raise OSError('model write failed')
            tokens[params[1]]=tokens.get(params[1],0)+params[2]
    async def fetchall(self):return []
    async def fetchone(self):return self.row


class MysqlTransactionTests(unittest.IsolatedAsyncioTestCase):
    def store(self):
        store=MysqlUsageStore({'host':'synthetic.invalid'})
        store._pool=TransactionPool();return store
    async def test_partial_failure_rolls_back_all_models_and_ledger(self):
        store=self.store();pool=store._pool;pool.fail_model='b'
        store.record('a',7,1);store.record('b',3,1)
        with self.assertRaises(OSError):await store._flush()
        self.assertEqual(pool.tokens,{})
        self.assertEqual(pool.ledger,{})
        self.assertEqual(pool.rollbacks,1)
        pool.fail_model=None;await store._flush()
        self.assertEqual(pool.tokens,{'a':7,'b':3})
        self.assertEqual(len(pool.ledger),1)
    async def test_ack_loss_reuses_batch_and_does_not_apply_delta_twice(self):
        store=self.store();pool=store._pool;pool.lose_ack=True
        store.record('m',5,1)
        with self.assertRaises(OSError):await store._flush()
        self.assertEqual(pool.tokens,{'m':5})
        await store._flush()
        self.assertEqual(pool.tokens,{'m':5})
        self.assertEqual(len(pool.ledger),1)
        self.assertEqual(store.get_today_data()['models']['m']['input_tokens'],5)
    async def test_missing_pool_is_not_acknowledged_as_a_successful_write(self):
        store=MysqlUsageStore({'host':'synthetic.invalid'});store.record('m',1,1)
        with patch.object(store,'_backend_start',side_effect=OSError('not connected')):
            with self.assertRaises(OSError):await store._flush()
        self.assertEqual(store.persistence_stats()['pending_events'],1)

    async def test_cancellation_during_rollback_keeps_cleanup_owned_and_cancellation_visible(self):
        store=self.store();pool=store._pool;pool.fail_model='m';store.record('m',1,1)
        closing,release=asyncio.Event(),asyncio.Event();rollback=pool.rollback
        async def slow_rollback():
            closing.set();await release.wait();await rollback()
        with patch.object(pool,'rollback',side_effect=slow_rollback):
            task=asyncio.create_task(store._flush());await closing.wait()
            task.cancel();task.cancel();await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(pool.tokens,{})
        self.assertEqual(store.persistence_stats()['pending_events'],1)
