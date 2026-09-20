"""Local readiness is independent of shared upstream faults; drain is explicit."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import main
import gateway_lifecycle
from admission import AdmissionController


class DrainTests(unittest.TestCase):
    def test_invalid_cli_budget_cannot_create_a_drain_marker(self):
        with tempfile.TemporaryDirectory() as root,patch.object(gateway_lifecycle,'DRAIN_MARKER_FILE',str(Path(root)/'flag')),\
                patch('sys.argv',['gateway_lifecycle','--wait-seconds','-1']):
            with self.assertRaises(SystemExit):gateway_lifecycle.main()
            self.assertFalse((Path(root)/'flag').exists())

    def test_marker_is_idempotent_and_never_overwrites_an_existing_file(self):
        with tempfile.TemporaryDirectory() as root:
            marker=Path(root)/'draining'
            gateway_lifecycle.request_drain(marker)
            original=marker.read_bytes()
            gateway_lifecycle.request_drain(marker)
            self.assertEqual(marker.read_bytes(),original)
            controller=AdmissionController(max_active=1,max_queued=1,wait_timeout=1,body_budget=100,tenant_limits={})
            self.assertTrue(gateway_lifecycle.sync_drain(controller,marker))
            self.assertTrue(controller.draining)

    def test_wait_does_not_treat_missing_or_malformed_state_as_drained(self):
        clock=[0.0]
        def sleep(seconds):clock[0]+=seconds
        for value in ({},{'draining':True,'active_requests':False,'queued_requests':0},None):
            clock[0]=0
            self.assertFalse(gateway_lifecycle.wait_until_drained(lambda:value,timeout=.1,interval=.05,
                                                                  clock=lambda:clock[0],sleep=sleep))
        self.assertTrue(gateway_lifecycle.wait_until_drained(
            lambda:{'draining':True,'active_requests':0,'queued_requests':0},timeout=0))


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_store_shutdown_still_closes_all_clients(self):
        import asyncio
        store=SimpleNamespace(stop=AsyncMock(side_effect=asyncio.CancelledError))
        clients=[SimpleNamespace(close=AsyncMock()),SimpleNamespace(close=AsyncMock())]
        with self.assertRaises(asyncio.CancelledError):await main._stop_runtime((),store,clients)
        for client in clients:client.close.assert_awaited_once()

    async def test_shared_upstream_failure_does_not_remove_a_locally_ready_gateway(self):
        ep=main.WorkspaceEndpoint('fixture','https://fixture.invalid','synthetic')
        ep.circuit_open=True;ep.circuit_retry_at=float('inf')
        controller=AdmissionController(max_active=1,max_queued=1,wait_timeout=1,body_budget=100,tenant_limits={})
        with tempfile.TemporaryDirectory() as root,patch.object(gateway_lifecycle,'DRAIN_MARKER_FILE',str(Path(root)/'flag')),\
                patch.object(main.app.state,'lb_initialized',True,create=True),patch.object(main,'INFERENCE_ADMISSION',controller),\
                patch.object(main,'proxy',SimpleNamespace(load_balancer=main.LoadBalancer([ep]))),\
                patch.object(main,'azure_proxy',None),patch.object(main,'copilot_proxy',None):
            response=await main.health_accepting()
            self.assertEqual(response.status_code,200)
            with self.assertRaises(main.HTTPException):await main.health_ready()
            gateway_lifecycle.request_drain()
            response=await main.health_accepting()
            self.assertEqual(response.status_code,503)
            self.assertTrue(json.loads(response.body)['draining'])

    async def test_gateway_is_not_accepting_before_local_initialization(self):
        with patch.object(main.app.state,'lb_initialized',False,create=True):
            self.assertEqual((await main.health_accepting()).status_code,503)

    async def test_lifespan_closes_clients_when_body_raises_and_usage_stop_fails(self):
        proxy=SimpleNamespace(load_balancer=SimpleNamespace(endpoints=[object()]),close=AsyncMock())
        store=SimpleNamespace(start=AsyncMock(),stop=AsyncMock(side_effect=OSError('synthetic store failure')),
                              get_today_data=lambda:{})
        with patch.object(main,'load_config',return_value=(proxy,None,None,{})),patch.object(main,'create_usage_store',return_value=store),\
                patch('otel_setup.setup_tracing',return_value=False),patch.object(main,'proxy',None),\
                patch.object(main,'azure_proxy',None),patch.object(main,'copilot_proxy',None),patch.object(main,'usage_store',None):
            with self.assertRaisesRegex(ValueError,'synthetic body failure'):
                async with main.lifespan(main.app):raise ValueError('synthetic body failure')
        proxy.close.assert_awaited_once()
        store.stop.assert_awaited_once()
        self.assertFalse(main.app.state.lb_initialized)
