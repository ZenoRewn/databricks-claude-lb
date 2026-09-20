"""Bound inference concurrency, waiting callers and retained input. Author: Zeno Ren."""
import asyncio
from collections import Counter, deque
from contextvars import ContextVar
from dataclasses import dataclass
import json
import math
import os
import time

MAX_ACTIVE = int(os.getenv('INFERENCE_MAX_ACTIVE', '128'))
MAX_QUEUED = int(os.getenv('INFERENCE_MAX_QUEUED', '64'))
QUEUE_TIMEOUT = float(os.getenv('INFERENCE_QUEUE_TIMEOUT_SECONDS', '10'))
BODY_BUDGET = int(os.getenv('INFERENCE_BODY_MEMORY_BYTES', str(128*1024*1024)))
BODY_READ_TIMEOUT = float(os.getenv('REQUEST_BODY_TIMEOUT_SECONDS', '120'))
TENANT_LIMITS = json.loads(os.getenv('INFERENCE_TENANT_LIMITS', '{}'))
CURRENT_LEASE = ContextVar('inference_admission_lease', default=None)
ROUTES = {'/v1/messages','/v1/responses','/v1/chat/completions'}
REASONS = ('queue_full','queue_timeout','body_memory_budget','draining','maintenance')


class AdmissionError(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(eq=False)
class _Waiter:
    tenant: str
    future: asyncio.Future


class AdmissionLease:
    def __init__(self, controller, tenant):
        self.controller, self.tenant = controller, tenant
        self.body_bytes = 0
        self.ended = False

    def reserve_body(self, amount):
        if self.ended or type(amount) is not int or amount < 0:
            raise ValueError('Invalid reservation or closed admission lease')
        c = self.controller
        if amount > c.body_budget-c.body_bytes:
            c.rejections['body_memory_budget'] += 1
            raise AdmissionError('body_memory_budget')
        c.body_bytes += amount
        self.body_bytes += amount

    def release(self):
        if self.ended:
            return
        self.ended = True
        c = self.controller
        c.body_bytes -= self.body_bytes
        c.active -= 1
        c.tenant_active[self.tenant] -= 1
        c._dispatch()


class AdmissionController:
    def __init__(self, *, max_active, max_queued, wait_timeout, body_budget, tenant_limits):
        if (type(max_active) is not int or not 1 <= max_active <= 10000 or
                type(max_queued) is not int or not 0 <= max_queued <= 10000 or
                not math.isfinite(wait_timeout) or wait_timeout <= 0 or
                type(body_budget) is not int or body_budget <= 0):
            raise ValueError('Invalid admission limits')
        if not isinstance(tenant_limits,dict) or not all(
                isinstance(k,str) and k and type(v) is int and v > 0 for k,v in tenant_limits.items()):
            raise ValueError('Tenant limits must map configured tenant names to positive integers')
        self.max_active, self.max_queued = max_active, max_queued
        self.wait_timeout, self.body_budget = wait_timeout, body_budget
        self.tenant_limits = dict(tenant_limits)
        self.active = self.body_bytes = 0
        self.tenant_active = Counter()
        self.waiters = deque()
        self.rejections = Counter()
        self.queue_wait_count = 0
        self.queue_wait_seconds = 0.0
        self.permanent_draining = False
        self.maintenance_paused = False

    @property
    def draining(self):
        return self.permanent_draining or self.maintenance_paused

    def _can_admit(self, tenant):
        return not self.draining and self.active < self.max_active and self.tenant_active[tenant] < self.tenant_limits.get(tenant,self.max_active)

    def _allocate(self, tenant):
        self.active += 1
        self.tenant_active[tenant] += 1
        return AdmissionLease(self,tenant)

    def _dispatch(self):
        # Preserve FIFO within each tenant while bypassing a tenant whose own
        # quota is full. No await between eligibility and claiming a slot.
        for waiter in tuple(self.waiters):
            if waiter.future.done():
                self.waiters.remove(waiter)
            elif self._can_admit(waiter.tenant):
                self.waiters.remove(waiter)
                waiter.future.set_result(self._allocate(waiter.tenant))

    def drain(self):
        self.permanent_draining = True
        self._reject_waiters('draining')

    def set_maintenance(self, paused):
        self.maintenance_paused = bool(paused)
        if self.draining:
            self._reject_waiters('draining' if self.permanent_draining else 'maintenance')
        else:
            self._dispatch()

    def _reject_waiters(self, reason):
        while self.waiters:
            waiter = self.waiters.popleft()
            if not waiter.future.done():
                self.rejections[reason] += 1
                waiter.future.set_exception(AdmissionError(reason))

    async def acquire(self, tenant):
        if self.draining:
            reason = 'draining' if self.permanent_draining else 'maintenance'
            self.rejections[reason] += 1
            raise AdmissionError(reason)
        self._dispatch()
        if self._can_admit(tenant):
            return self._allocate(tenant)
        if len(self.waiters) >= self.max_queued:
            self.rejections['queue_full'] += 1
            raise AdmissionError('queue_full')
        waiter = _Waiter(tenant,asyncio.get_running_loop().create_future())
        self.waiters.append(waiter)
        started = time.monotonic()
        try:
            # Python 3.11 wait_for can consume external cancellation when the
            # offered future completes in the same turn. Own the timeout here
            # so cancellation still releases an offered lease below.
            async with asyncio.timeout(self.wait_timeout):
                return await asyncio.shield(waiter.future)
        except BaseException as exc:
            if waiter in self.waiters:
                self.waiters.remove(waiter)
            if waiter.future.done() and not waiter.future.cancelled():
                if waiter.future.exception() is None:
                    waiter.future.result().release()
            else:
                waiter.future.cancel()
            self._dispatch()
            if isinstance(exc,TimeoutError):
                self.rejections['queue_timeout'] += 1
                raise AdmissionError('queue_timeout') from None
            raise
        finally:
            self.queue_wait_count += 1
            self.queue_wait_seconds += time.monotonic()-started

    def render_metrics(self):
        values = {'lb_admission_active':self.active,'lb_admission_queued':len(self.waiters),
                  'lb_admission_body_bytes':self.body_bytes,'lb_admission_draining':int(self.draining)}
        lines = []
        for name,value in values.items():
            lines.extend([f'# HELP {name} Local process admission state',f'# TYPE {name} gauge',f'{name} {value}'])
        name = 'lb_admission_rejected_total'
        lines.extend([f'# HELP {name} Local rejections before upstream inference',f'# TYPE {name} counter'])
        lines.extend(f'{name}{{reason="{reason}"}} {self.rejections[reason]}' for reason in REASONS)
        lines.extend(['# HELP lb_admission_queue_wait_seconds Time of settled queued callers, including cancellation and timeout',
                      '# TYPE lb_admission_queue_wait_seconds summary',
                      f'lb_admission_queue_wait_seconds_count {self.queue_wait_count}',
                      f'lb_admission_queue_wait_seconds_sum {self.queue_wait_seconds}'])
        return '\n'.join(lines)+'\n'


if not math.isfinite(BODY_READ_TIMEOUT) or BODY_READ_TIMEOUT <= 0:
    raise ValueError('Request body timeout must be finite and positive')
CONTROLLER = AdmissionController(max_active=MAX_ACTIVE,max_queued=MAX_QUEUED,wait_timeout=QUEUE_TIMEOUT,
                                 body_budget=BODY_BUDGET,tenant_limits=TENANT_LIMITS)


class AdmissionMiddleware:
    def __init__(self, app, *, tenant_for_scope, controller=CONTROLLER):
        self.app, self.tenant_for_scope, self.controller = app, tenant_for_scope, controller

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') not in ROUTES:
            return await self.app(scope,receive,send)
        tenant = self.tenant_for_scope(scope)
        if tenant is None:
            # The existing handler retains its auth/provider error contract and
            # rejects before body parsing; unauthenticated traffic gets no slot.
            return await self.app(scope,receive,send)
        scope.setdefault('state',{})['lb_source_tenant'] = tenant
        try:
            lease = await self.controller.acquire(tenant)
        except AdmissionError as exc:
            scope.setdefault('state',{})['lb_overloaded'] = True
            scope['state']['lb_draining_at_finish'] = self.controller.draining
            scope['state']['lb_admission_reason'] = exc.reason
            body = json.dumps({'error':{'code':'lb_overloaded','reason':exc.reason,
                                       'message':'Local inference admission unavailable; retry after cooldown.'}}).encode()
            await send({'type':'http.response.start','status':503,
                        'headers':[(b'content-type',b'application/json'),(b'retry-after',b'1')]})
            await send({'type':'http.response.body','body':body,'more_body':False})
            return
        token = CURRENT_LEASE.set(lease)
        try:
            await self.app(scope,receive,send)
        finally:
            scope.setdefault('state',{})['lb_draining_at_finish'] = self.controller.draining
            CURRENT_LEASE.reset(token)
            lease.release()
