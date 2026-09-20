"""Independent fault-oriented release invariants, without Kubernetes access."""
import copy
import unittest


class MemoryStore:
    def __init__(self, record):self.record=copy.deepcopy(record);self.fail=False
    def read(self):return copy.deepcopy(self.record)
    def save(self,record):
        if self.fail:raise OSError('journal unavailable')
        self.record=copy.deepcopy(record)


class FakeBackend:
    def __init__(self):
        self.routes=True;self.paused=False;self.old_running=True;self.new_running=False
        self.healthy=True;self.active=0;self.fenced=False;self.recovery_only=False
        self.ack_loss=None;self.fault=None;self.calls=[];self.stopped=False;self.owned=True
    def check_owner(self):
        if not self.owned:raise RuntimeError('ownership lost')
    def observe(self,record):
        return {'routes_restored':self.routes,'paused':self.paused,'old_running':self.old_running,
                'new_running':self.new_running,'backend_healthy':self.healthy,
                'quiescent':self.active==0,'old_stopped':self.stopped,'recovery_only':self.recovery_only}
    def perform(self,action,record):
        self.check_owner();self.calls.append(action)
        if self.fault==action:raise RuntimeError('injected '+action)
        if action in ('preflight','cleanup'):pass
        elif action=='gate':self.routes=False
        elif action=='pause':self.paused=True
        elif action=='stop_old':self.old_running=False;self.stopped=True
        elif action=='start_new':self.new_running=True
        elif action=='verify_backend':
            if not self.healthy:raise RuntimeError('unhealthy')
        elif action=='restore_routes':self.routes=True
        elif action=='verify_business':
            if not self.healthy:raise RuntimeError('business failed')
        elif action=='resume_old':self.paused=False;self.routes=True
        elif action=='rollback':
            if not self.healthy:raise RuntimeError('rollback target unavailable')
            self.new_running=False;self.old_running=True;self.paused=False;self.routes=True
        elif action=='mark_recovery':self.recovery_only=True
        else:raise ValueError(action)
        if self.ack_loss==action:
            self.ack_loss=None
            from operations.release.engine import UncertainOperation
            raise UncertainOperation('ACK lost')
        return {'verified':True}
    def emergency_restore(self,record):
        self.check_owner();self.recovery_only=True
        if self.old_running and self.healthy:
            self.paused=False;self.routes=True
        return self.routes and self.healthy


class ReleaseEngineTests(unittest.TestCase):
    def setup_release(self):
        from operations.release.engine import Engine, new_record
        record=new_record({'release_id':'test-r','maintenance_seconds':300,'forward_seconds':150,'drain_seconds':60})
        self.clock=[100.0];self.store=MemoryStore(record);self.backend=FakeBackend()
        self.engine=Engine(self.store,self.backend,clock=lambda:self.clock[0]);return self.engine
    def step(self):return self.engine.tick()
    def finish(self):
        for _ in range(30):
            state=self.step();self.clock[0]+=1
            if state['phase'] in ('succeeded','rolled_back','cancelled','needs_attention'):return state
        self.fail('release did not converge')
    def test_success_never_exits_after_drain_with_a_closed_route(self):
        self.setup_release();r=self.finish()
        self.assertEqual(r['phase'],'succeeded');self.assertTrue(self.backend.routes)
        self.assertIn('verify_business',self.backend.calls)
    def test_restart_in_every_phase_uses_durable_state(self):
        from operations.release.engine import Engine
        for stop_after in range(1,11):
            with self.subTest(stop_after=stop_after):
                self.setup_release()
                for _ in range(stop_after):self.step()
                self.engine=Engine(self.store,self.backend,clock=lambda:self.clock[0])
                self.assertEqual(self.finish()['phase'],'succeeded');self.assertTrue(self.backend.routes)
    def test_ack_loss_is_read_back_without_blind_rollback(self):
        self.setup_release();self.backend.ack_loss='gate';r=self.finish()
        self.assertEqual(r['phase'],'succeeded');self.assertNotIn('rollback',self.backend.calls)
    def test_drain_timeout_resumes_old_without_killing_active_request(self):
        self.setup_release()
        while self.store.read()['phase']!='draining':self.step()
        self.backend.active=1;self.clock[0]+=61
        r=self.finish();self.assertEqual(r['phase'],'rolled_back')
        self.assertTrue(self.backend.old_running);self.assertNotIn('stop_old',self.backend.calls)
    def test_journal_failure_does_not_strand_healthy_old_service(self):
        self.setup_release()
        while self.store.read()['phase']!='draining':self.step()
        self.store.fail=True;r=self.step()
        self.assertTrue(self.backend.routes);self.assertTrue(self.backend.recovery_only)
        self.assertNotEqual(r['phase'],'succeeded')
        self.store.fail=False;self.assertEqual(self.finish()['phase'],'rolled_back')
    def test_unknown_rollback_never_claims_success(self):
        self.setup_release()
        while self.store.read()['phase']!='starting':self.step()
        self.backend.healthy=False;self.backend.fault='start_new'
        r=self.finish();self.assertEqual(r['phase'],'needs_attention');self.assertFalse(self.backend.routes)
    def test_lost_owner_does_not_restore_someone_elses_routes(self):
        self.setup_release();self.backend.owned=False
        r=self.step();self.assertEqual(self.backend.calls,[]);self.assertNotEqual(r['phase'],'succeeded')
    def test_cancel_before_stopping_old_resumes_it(self):
        self.setup_release()
        while self.store.read()['phase']!='draining':self.step()
        r=self.store.read();r['cancel_requested']=True;self.store.save(r)
        self.assertEqual(self.finish()['phase'],'cancelled');self.assertTrue(self.backend.old_running)
