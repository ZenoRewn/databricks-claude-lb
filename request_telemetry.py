"""Request, admission and send accounting. Author: Zeno Ren.

Send started means an HTTP client invocation, not proof of bytes on the wire or
upstream execution. Body completed means ASGI send returned, not client receipt.
All labels are finite; identifiers and per-request details belong in logs only.
"""
import asyncio
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
import re
import time
import uuid
from request_budget import startup_budget, UpstreamStartupTimeout, remaining_seconds

logger = logging.getLogger('main')
ROUTES = {'/v1/messages':'messages', '/v1/responses':'responses', '/v1/chat/completions':'chat'}
APIS = ('messages','responses','chat')
PROVIDERS = ('databricks','azure_openai','copilot')
OUTCOMES = ('completed','failed','incomplete','unknown','http_error','rejected',
            'overloaded','cancelled','client_disconnected','internal_error','deadline_exceeded')
BUCKETS = (.1,.5,1,2,5,10,30,60,120,300,600,1800)
CURRENT = ContextVar('lb_request_lifecycle', default=None)


def safe_id(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value) else None


def log_event(fields):
    try:
        logger.info('%s %s', fields['kind'], json.dumps(fields), extra=fields)
    except Exception:
        # Observability must never cancel generation or strand an owned response.
        pass


class RequestTelemetry:
    def __init__(self):
        self.started = Counter()
        self.active = Counter()
        self.outcomes = Counter()
        self.admissions = Counter()
        self.sends = Counter()
        self.send_results = Counter()
        self.retry_decisions = Counter()
        self.latency_sum = Counter()
        self.latency_buckets = Counter()

    def render(self):
        lines = []
        def metric(name, description, kind, samples):
            lines.extend([f'# HELP {name} {description}', f'# TYPE {name} {kind}', *samples])
        for name, counts, kind, description in (
            ('lb_requests_started_total',self.started,'counter','Inference ingress calls, including local rejections'),
            ('lb_requests_active',self.active,'gauge','Inference ingress calls not yet settled'),
            ('lb_endpoint_admissions_total',self.admissions,'counter','Endpoint leases; distinct from HTTP send invocations')):
            metric(name,description,kind,[f'{name}{{api_type="{api}"}} {counts[api]}' for api in APIS])
        metric('lb_requests_finished_total','One local outcome per finished ingress call; not workflow success','counter',
               [f'lb_requests_finished_total{{api_type="{api}",outcome="{outcome}"}} {self.outcomes[(api,outcome)]}'
                for api in APIS for outcome in OUTCOMES])
        metric('lb_upstream_send_started_total','Inference HTTP client invocations, including auth and opaque repairs','counter',
               [f'lb_upstream_send_started_total{{provider="{provider}",api_type="{api}"}} {self.sends[(provider,api)]}'
                for provider in PROVIDERS for api in APIS])
        metric('lb_upstream_send_finished_total','Send result; response means headers or buffered body received, not generation success','counter',
               [f'lb_upstream_send_finished_total{{provider="{provider}",api_type="{api}",result="{result}"}} {count}'
                for (provider,api,result),count in sorted(self.send_results.items())])
        metric('lb_retry_decisions_total','HTTP retry decisions under existing replay policy','counter',
               [f'lb_retry_decisions_total{{reason="{reason}"}} {self.retry_decisions[reason]}'
                for reason in ('retry_429','upstream_cooldown','attempt_budget_exhausted','status_not_retryable')])
        samples = []
        for api in APIS:
            for outcome in OUTCOMES:
                labels=f'api_type="{api}",outcome="{outcome}"'
                for upper in BUCKETS:
                    samples.append(f'lb_request_duration_seconds_bucket{{{labels},le="{upper}"}} {self.latency_buckets[(api,outcome,upper)]}')
                count=self.outcomes[(api,outcome)]
                samples.extend([f'lb_request_duration_seconds_bucket{{{labels},le="+Inf"}} {count}',
                                f'lb_request_duration_seconds_count{{{labels}}} {count}',
                                f'lb_request_duration_seconds_sum{{{labels}}} {self.latency_sum[(api,outcome)]}'])
        metric('lb_request_duration_seconds','Full ingress lifetime including failures and cancellations','histogram',samples)
        return '\n'.join(lines)+'\n'


TELEMETRY = RequestTelemetry()


@dataclass
class RequestRecord:
    api_type: str
    metrics: RequestTelemetry
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started: float = field(default_factory=time.monotonic)
    generation: str = 'unknown'
    admissions: int = 0
    sends: int = 0
    last_admission: str = 'unknown'


def note_admission():
    record = CURRENT.get()
    if record:
        record.admissions += 1
        record.metrics.admissions[record.api_type] += 1


def note_candidate_selection(eligible_count, tried_count, untried_count):
    record = CURRENT.get()
    log_event({'kind':'lb_candidate_selection','lb_request_id':record.request_id if record else None,
               'eligible_count':eligible_count,'tried_count':tried_count,'untried_count':untried_count,
               'selection_reason':'untried_preferred' if untried_count else 'bounded_revisit'})


def note_retry_decision(status, attempt, max_attempts, retry_after_present, allowed):
    record = CURRENT.get()
    reason = ('status_not_retryable' if status != 429 else 'attempt_budget_exhausted'
              if attempt >= max_attempts-1 else 'upstream_cooldown' if retry_after_present else 'retry_429')
    (record.metrics if record else TELEMETRY).retry_decisions[reason] += 1
    log_event({'kind':'lb_retry_decision','lb_request_id':record.request_id if record else None,
               'upstream_status':status,'reason':reason,'retry_allowed':bool(allowed),
               'execution_certainty':'admission_rejected' if status==429 else 'unknown',
               'retry_after_present':bool(retry_after_present),
               'remaining_loop_attempts':max(0,max_attempts-attempt-1),'remaining_seconds':remaining_seconds()})


def note_admission_end(success, *, cancelled=False, neutral=False):
    record = CURRENT.get()
    if record:
        record.last_admission = 'completed' if success else 'cancelled' if cancelled else 'neutral' if neutral else 'failed'


def note_generation(outcome):
    record = CURRENT.get()
    if record and outcome in ('completed','failed','error','incomplete'):
        record.generation = 'failed' if outcome == 'error' else outcome


def note_json_result(payload, api_type):
    """Observe explicit result semantics without changing the response contract."""
    record = CURRENT.get()
    if not record:
        return
    record.generation = 'unknown'
    if not isinstance(payload, dict):
        return
    if payload.get('error') or payload.get('status') == 'failed':
        note_generation('failed')
    elif payload.get('status') == 'incomplete' or payload.get('incomplete_details'):
        note_generation('incomplete')
    elif api_type == 'responses':
        if payload.get('status') == 'completed' and isinstance(payload.get('id'),str) and isinstance(payload.get('output'),list):
            note_generation('completed')
    elif api_type == 'messages' and payload.get('type') == 'message':
        reason = payload.get('stop_reason')
        if reason == 'max_tokens':
            note_generation('incomplete')
        elif reason in ('end_turn','tool_use','stop_sequence'):
            note_generation('completed')
    elif api_type == 'chat':
        choices=payload.get('choices')
        if isinstance(choices,list) and choices and all(isinstance(c,dict) for c in choices):
            reasons={c.get('finish_reason') for c in choices if isinstance(c.get('finish_reason'),str)}
            if len([c for c in choices if isinstance(c.get('finish_reason'),str)]) != len(choices):
                return
            if reasons & {'length','content_filter'}:
                note_generation('incomplete')
            elif reasons <= {'stop','tool_calls','function_call'}:
                note_generation('completed')


async def inference_call(awaitable, provider, api_type):
    record = CURRENT.get()
    metrics = record.metrics if record else TELEMETRY
    metrics.sends[(provider,api_type)] += 1
    attempt_id = str(uuid.uuid4())
    if record:
        record.sends += 1
    started = time.monotonic()
    result = 'transport_error'
    status = None
    try:
        async with startup_budget():
            response = await awaitable
        status = getattr(response,'status_code',None)
        result = f'http_{status//100}xx' if type(status) is int and 100 <= status < 600 else 'unknown'
        return response
    except asyncio.CancelledError:
        result = 'cancelled'
        raise
    except UpstreamStartupTimeout:
        result = 'startup_timeout'
        raise
    finally:
        metrics.send_results[(provider,api_type,result)] += 1
        fields={'kind':'lb_upstream_send_end','lb_request_id':record.request_id if record else None,
                'upstream_attempt_id':attempt_id,'provider':provider,'api_type':api_type,
                'result':result,'upstream_status':status if type(status) is int else None,
                'duration_seconds':round(time.monotonic()-started,6)}
        log_event(fields)


class RequestTelemetryMiddleware:
    def __init__(self, app, metrics=TELEMETRY):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        api_type=ROUTES.get(scope.get('path'))
        if scope['type'] != 'http' or scope.get('method') != 'POST' or not api_type:
            return await self.app(scope,receive,send)
        record=RequestRecord(api_type,self.metrics)
        operation_id=safe_id(dict(scope.get('headers',[])).get(b'x-lb-operation-id',b'').decode('ascii',errors='replace'))
        token=CURRENT.set(record)
        self.metrics.started[api_type] += 1
        self.metrics.active[api_type] += 1
        status=None
        body_complete=False
        disconnected=False
        error=None
        async def observed_receive():
            nonlocal disconnected
            message=await receive()
            if message['type']=='http.disconnect' and not body_complete:
                disconnected=True
            return message
        async def observed_send(message):
            nonlocal status, body_complete, disconnected
            if message['type']=='http.response.start':
                status=message['status']
                message={**message,'headers':[(k,v) for k,v in message.get('headers',[]) if k.lower()!=b'x-lb-request-id']+
                         [(b'x-lb-request-id',record.request_id.encode('ascii'))]}
            try:
                await send(message)
            except OSError:
                disconnected=True
                raise
            if message['type']=='http.response.body' and not message.get('more_body',False):
                body_complete=True
        try:
            await self.app(scope,observed_receive,observed_send)
        except asyncio.CancelledError:
            error='cancelled'
            raise
        except BaseException:
            error='internal_error'
            raise
        finally:
            if disconnected:
                outcome='client_disconnected'
            elif scope.get('state',{}).get('lb_deadline_exceeded'):
                outcome='deadline_exceeded'
            elif error:
                outcome=error
            elif status == 429:
                outcome='overloaded'
            elif status is not None and status >= 400:
                outcome='rejected' if status < 500 else 'http_error'
            elif not body_complete:
                outcome='unknown'
            elif record.generation != 'unknown':
                outcome=record.generation
            elif record.last_admission in ('failed','neutral'):
                outcome='failed'
            else:
                outcome='unknown'
            elapsed=time.monotonic()-record.started
            self.metrics.outcomes[(api_type,outcome)] += 1
            self.metrics.latency_sum[(api_type,outcome)] += elapsed
            for upper in BUCKETS:
                if elapsed <= upper:
                    self.metrics.latency_buckets[(api_type,outcome,upper)] += 1
            self.metrics.active[api_type] -= 1
            CURRENT.reset(token)
            fields={'kind':'lb_request_end','lb_request_id':record.request_id,'api_type':api_type,
                    'request_id':safe_id(scope.get('state',{}).get('request_id')),
                    'operation_id':operation_id,
                    'outcome':outcome,'http_status':status,'generation_outcome':record.generation,
                    'downstream_body_completed':body_complete,'admissions':record.admissions,
                    'upstream_sends':record.sends,'duration_seconds':round(elapsed,6)}
            log_event(fields)
