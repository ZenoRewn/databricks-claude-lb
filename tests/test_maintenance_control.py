"""A release may resume its pause; it cannot undo shutdown or another owner."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gateway_lifecycle as lifecycle
from admission import AdmissionController, AdmissionError


class MaintenanceTests(unittest.IsolatedAsyncioTestCase):
    def controller(self):
        return AdmissionController(max_active=1,max_queued=1,wait_timeout=1,body_budget=100,tenant_limits={})

    async def test_pause_keeps_active_request_and_resumes_only_owned_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'maintenance'; c=self.controller(); lease=await c.acquire('tenant')
            state=lifecycle.maintenance_command('pause','release-a',1,0,path=path)
            lifecycle.sync_maintenance(c,path)
            self.assertEqual(c.active,1)
            with self.assertRaises(AdmissionError):await c.acquire('tenant')
            with self.assertRaises(ValueError):lifecycle.maintenance_command('resume','release-b',1,state['revision'],path=path)
            lifecycle.maintenance_command('resume','release-a',1,state['revision'],path=path)
            lifecycle.sync_maintenance(c,path);self.assertFalse(c.draining)
            lease.release();(await c.acquire('tenant')).release()

    async def test_resume_does_not_undo_permanent_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'maintenance';c=self.controller()
            state=lifecycle.maintenance_command('pause','r',1,0,path=path);lifecycle.sync_maintenance(c,path)
            c.drain()
            lifecycle.maintenance_command('resume','r',1,state['revision'],path=path);lifecycle.sync_maintenance(c,path)
            self.assertTrue(c.draining)

    async def test_takeover_rejects_old_epoch_and_stale_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'maintenance'
            a=lifecycle.maintenance_command('pause','r',1,0,path=path)
            b=lifecycle.maintenance_command('pause','r',2,a['revision'],path=path)
            with self.assertRaises(ValueError):lifecycle.maintenance_command('resume','r',1,b['revision'],path=path)
            with self.assertRaises(ValueError):lifecycle.maintenance_command('resume','r',2,a['revision'],path=path)
            self.assertTrue(lifecycle.maintenance_state(path)['paused'])

    async def test_corrupt_control_state_closes_admission_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'maintenance';path.write_text('{bad');c=self.controller()
            lifecycle.sync_maintenance(c,path)
            self.assertTrue(c.draining)
            with self.assertRaises(ValueError):lifecycle.maintenance_command('resume','r',1,0,path=path)

    async def test_pause_wakes_queued_request_and_does_not_leak_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'maintenance';c=self.controller();lease=await c.acquire('a')
            pending=asyncio.create_task(c.acquire('b'));await asyncio.sleep(0)
            lifecycle.maintenance_command('pause','r',1,0,path=path);lifecycle.sync_maintenance(c,path)
            with self.assertRaises(AdmissionError):await pending
            lease.release();self.assertEqual(c.active,0);self.assertEqual(len(c.waiters),0)

    async def test_resume_before_delayed_pause_fences_the_old_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'maintenance'
            state=lifecycle.maintenance_command('resume','r',2,0,path=path)
            self.assertFalse(state['paused'])
            with self.assertRaises(ValueError):lifecycle.maintenance_command('pause','r',1,0,path=path)
