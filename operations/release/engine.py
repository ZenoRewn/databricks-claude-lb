"""Restartable single-writer release state machine. Author: Zeno Ren."""
import copy
import json
import re
import sys
import time

TERMINAL = {'succeeded','rolled_back','cancelled','needs_attention'}


class PendingOperation(Exception):
    """Observed operation is still converging; no inference from an ACK alone."""


class UncertainOperation(Exception):
    """An API operation may have executed; retry only through state readback."""


class OwnershipLost(Exception):
    pass


class UnsafeState(Exception):
    """Concurrent drift or unknown writer state requires explicit intervention."""


def new_record(plan):
    return {'schema_version':'lb-release-v1','plan':copy.deepcopy(plan),'phase':'preflight',
            'cancel_requested':False,'maintenance_started_at':None,'drain_started_at':None,
            'maintenance_finished_at':None,'intent':None,'events':[], 'action_receipts':{},
            'cleanup_errors':[],'final_result':'not_completed'}


class Engine:
    def __init__(self,store,backend,clock=time.time):
        self.store,self.backend,self.clock=store,backend,clock

    def _save(self,record):
        self.backend.check_owner()
        try:self.store.save(record)
        except OSError:
            # Storage failure cannot skip recovery. The adapter fences a durable
            # recovery-only marker on the workload before reopening a healthy old
            # backend. Subsequent leaders observe it and cannot gate again.
            recovery_error=None
            try:restored=self.backend.emergency_restore(record) is True
            except Exception as exc:
                restored=False;recovery_error=type(exc).__name__
            if not restored:
                # The journal is unavailable: emit a separate, sanitized failure
                # signal instead of hiding an unsuccessful recovery behind it.
                print(json.dumps({'kind':'release_emergency_recovery_unverified',
                    'release_id':record['plan']['release_id'],'phase':record['phase'],
                    'error_type':recovery_error}),file=sys.stderr,flush=True)
            raise

    def _phase(self,record,phase,reason=None):
        if phase=='recovering' and record.get('recovery_deadline') is None:
            now=self.clock()
            record['recovery_started_at']=now
            record['recovery_deadline']=(record['maintenance_started_at']+record['plan'].get('maintenance_seconds',300)
                if record['maintenance_started_at'] is not None and record['maintenance_finished_at'] is None
                else now+record['plan'].get('maintenance_seconds',300))
        record['phase']=phase
        record['events'].append({'at':self.clock(),'phase':phase,'reason':reason})
        record['events']=record['events'][-256:]
        if phase in TERMINAL:record['final_result']=phase
        self._save(record)

    def _perform(self,record,action):
        if not record.get('intent') or record['intent']['action']!=action:
            record['intent']={'action':action,'at':self.clock()}
            record['events'].append({'at':self.clock(),'phase':record['phase'],'intent':action})
            self._save(record)
        self.backend.check_owner()
        receipt=self.backend.perform(action,record)
        if action=='verify_business' and isinstance(receipt,dict) and receipt.get('failed') is True:
            record['action_receipts'][action]=receipt
            self._save(record)
            raise RuntimeError('business_or_persistence_verification_failed')
        if not isinstance(receipt,dict) or receipt.get('verified') is not True:
            raise PendingOperation(action)
        record['action_receipts'][action]=receipt
        record['intent']=None
        self._save(record)

    def tick(self):
        record=self.store.read()
        if record['phase'] in TERMINAL:return record
        try:self.backend.check_owner()
        except Exception:return record  # A stale executor is not allowed to journal either.
        try:
            observed=self.backend.observe(record)
            phase=record['phase'];now=self.clock();plan=record['plan']
            started=record['maintenance_started_at']
            if (record.get('cancel_requested') or observed.get('recovery_only')) and phase not in ('recovering','finalizing'):
                self._phase(record,'recovering','cancel_requested' if record.get('cancel_requested') else 'recovery_only')
                return record
            if started is not None and record['maintenance_finished_at'] is None and phase!='recovering':
                if now-started>=plan.get('forward_seconds',150):
                    self._phase(record,'recovering','forward_budget_exceeded');return record
            if phase=='preflight':
                self._perform(record,'preflight');self._phase(record,'gating')
            elif phase=='gating':
                if started is None:
                    record['maintenance_started_at']=now;self._save(record)
                self._perform(record,'gate');self._phase(record,'pausing')
            elif phase=='pausing':
                self._perform(record,'pause');record['drain_started_at']=now;self._phase(record,'draining')
            elif phase=='draining':
                if observed.get('quiescent') is True:
                    self._phase(record,'stopping')
                elif now-record['drain_started_at']>=plan.get('drain_seconds',60):
                    self._phase(record,'recovering','drain_timeout')
            elif phase=='stopping':
                self._perform(record,'stop_old');self._phase(record,'starting')
            elif phase=='starting':
                self._perform(record,'start_new');self._phase(record,'verifying_backend')
            elif phase=='verifying_backend':
                self._perform(record,'verify_backend');self._phase(record,'restoring_routes')
            elif phase=='restoring_routes':
                self._perform(record,'restore_routes');record['maintenance_finished_at']=now
                self._phase(record,'verifying_business')
            elif phase=='verifying_business':
                self._perform(record,'verify_business');record['completion_target']='succeeded';self._phase(record,'finalizing')
            elif phase=='finalizing':
                self._perform(record,'cleanup');self._phase(record,record['completion_target'])
            elif phase=='recovering':
                if now>=record.get('recovery_deadline',float('inf')):
                    self._phase(record,'needs_attention','maintenance_budget_exceeded');return record
                if started is None and observed.get('routes_restored'):
                    pass
                elif observed.get('old_running') and not observed.get('new_running'):
                    self._perform(record,'resume_old')
                elif observed.get('old_stopped') or record.get('writer_stop_requested'):
                    self._perform(record,'rollback')
                else:
                    raise UnsafeState('writer_state_unknown')
                record['maintenance_finished_at']=now
                record['completion_target']='cancelled' if record.get('cancel_requested') else 'rolled_back'
                self._phase(record,'finalizing')
            else:raise UnsafeState('unknown_phase')
        except PendingOperation:
            pass  # The next tick must observe actual state before acting again.
        except UncertainOperation:
            if not record.get('uncertain_operation'):
                record['uncertain_operation']=copy.deepcopy(record.get('intent'))
                try:self._save(record)
                except OSError:pass
        except OwnershipLost:
            pass
        except OSError:
            # _save already attempted safe recovery independently of its journal.
            record['final_result']='not_completed'
        except UnsafeState as exc:
            record['cleanup_errors'].append(str(exc)[:200])
            try:self._phase(record,'needs_attention',str(exc)[:200])
            except OSError:pass
        except Exception as exc:
            record['cleanup_errors'].append(type(exc).__name__)
            message=str(exc)
            record['last_failure']={'action':(record.get('intent') or {}).get('action',record['phase']),
                                    'error_type':type(exc).__name__,
                                    'code':message if re.fullmatch(r'[a-z0-9_]{1,80}',message) else None,
                                    'api_status':getattr(exc,'status',None) if type(getattr(exc,'status',None)) is int else None}
            try:
                if record['phase']=='recovering':self._phase(record,'needs_attention','recovery_failed')
                else:self._phase(record,'recovering',type(exc).__name__)
            except OSError:pass
        return record
