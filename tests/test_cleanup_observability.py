"""Observe owned cleanup without cancelling it or double-counting waiters."""
import asyncio
import unittest


class CleanupObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_owner_is_counted_once_and_failure_is_visible(self):
        from cleanup_observability import CleanupTracker
        tracker=CleanupTracker();gate=asyncio.Event()
        async def close():await gate.wait();raise OSError('synthetic close failure')
        task=asyncio.create_task(close());tracker.watch(task);tracker.watch(task)
        self.assertEqual(tracker.snapshot()['active'],1)
        gate.set()
        with self.assertRaises(OSError):await task
        await asyncio.sleep(0)
        self.assertEqual(tracker.snapshot()['active'],0)
        self.assertEqual(tracker.results['failed'],1)
        self.assertEqual(tracker.count,1)

    async def test_oldest_age_is_observed_without_deadline_cancellation(self):
        from cleanup_observability import CleanupTracker
        clock=[1.0];tracker=CleanupTracker(clock=lambda:clock[0]);gate=asyncio.Event()
        task=asyncio.create_task(gate.wait());tracker.watch(task);clock[0]=8.0
        self.assertEqual(tracker.snapshot()['oldest_age_seconds'],7.0)
        self.assertFalse(task.cancelled());gate.set();await task;await asyncio.sleep(0)
        self.assertEqual(tracker.snapshot()['oldest_age_seconds'],0)
