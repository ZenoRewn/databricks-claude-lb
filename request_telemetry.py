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
import os
import re
import time
import uuid
from fastapi import HTTPException
from request_budget import startup_budget, startup_seconds, UpstreamStartupTimeout, remaining_seconds
from request_timing import CURRENT as CURRENT_TIMING, Timeline, PhaseMetrics, observe_event
from response_semantics import REASONS, assess_json, exception_reason, failure_reason
from safe_diagnostics import safe_fields, safe_identifier, DIAGNOSTIC_DROPS

logger = logging.getLogger('main')
ROUTES = {'/v1/messages':'messages', '/v1/responses':'responses', '/v1/chat/completions':'chat'}
APIS = ('messages','responses','chat')
PROVIDERS = ('databricks','azure_openai','copilot')
SEND_RESULTS = ('http_1xx', 'http_2xx', 'http_3xx', 'http_4xx', 'http_5xx',
                'transport_error', 'startup_timeout', 'cancelled', 'unknown')
OUTCOMES = ('completed','failed','incomplete','unknown','http_error','rejected',
            'overloaded','cancelled','client_disconnected','internal_error','deadline_exceeded')
BUCKETS = (.1,.5,1,2,5,10,30,60,120,300,600,1800)
CURRENT = ContextVar('lb_request_lifecycle', default=None)
PARAMETERS = ('context_management','output_config','output_config.effort','output_config.format',
              'output_config.other','tools.defer_loading','tools.input_examples','messages.tool_reference',
              'cache_control.extras','thinking.budget_tokens','temperature','top_p','other',
              'response_format','reasoning_effort','tools','tools.strict','tool_choice','parallel_tool_calls',
              'messages.tool_calls','messages.tool_results','messages.content','messages.role',
              'max_tokens','max_completion_tokens','stream.buffered','stream_options','images.trimmed','images.compressed',
              'adapter.unsupported')
IMAGE_TRIM_POLICY = os.getenv('LB_IMAGE_TRIM_POLICY', 'reject')
if IMAGE_TRIM_POLICY not in ('reject', 'allow'):
    raise ValueError('LB_IMAGE_TRIM_POLICY must be reject or allow')


def safe_id(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value) else None


def log_event(fields):
    try:
        fields = safe_fields({**log_context(), **fields, 'schema_version': 1})
        logger.info('%s', fields['kind'], extra=fields)
    except Exception:
        # Observability must never cancel generation or strand an owned response.
        DIAGNOSTIC_DROPS['event_error'] += 1


class RequestTelemetry:
    def __init__(self):
        self.started = Counter()
        self.active = Counter()
        self.outcomes = Counter()
        self.admissions = Counter()
        self.sends = Counter()
        self.send_results = Counter()
        self.retry_decisions = Counter()
        self.parameter_decisions = Counter()
        self.reasons = Counter()
        self.latency_sum = Counter()
        self.latency_buckets = Counter()
        self.phases = PhaseMetrics()

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
        metric('lb_request_reasons_total','Bounded reason independent of HTTP versus SSE envelope','counter',
               [f'lb_request_reasons_total{{api_type="{api}",reason="{reason}"}} {self.reasons[(api,reason)]}'
                for api in APIS for reason in REASONS])
        metric('lb_upstream_send_started_total','Inference HTTP client invocations, including auth and opaque repairs','counter',
               [f'lb_upstream_send_started_total{{provider="{provider}",api_type="{api}"}} {self.sends[(provider,api)]}'
                for provider in PROVIDERS for api in APIS])
        metric('lb_upstream_send_finished_total','Send result; response means headers or buffered body received, not generation success','counter',
               [f'lb_upstream_send_finished_total{{provider="{provider}",api_type="{api}",result="{result}"}} {self.send_results[(provider,api,result)]}'
                for provider in PROVIDERS for api in APIS for result in SEND_RESULTS])
        metric('lb_retry_decisions_total','HTTP retry decisions under existing replay policy','counter',
               [f'lb_retry_decisions_total{{reason="{reason}"}} {self.retry_decisions[reason]}'
                for reason in ('retry_429','upstream_cooldown','attempt_budget_exhausted','status_not_retryable')])
        metric('lb_parameter_policy_requests_total','Requests affected by each named local parameter decision, not token counts','counter',
               [f'lb_parameter_policy_requests_total{{action="{action}",parameter="{parameter}"}} {self.parameter_decisions[(action,parameter)]}'
                for action in ('dropped','rejected') for parameter in PARAMETERS])
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
    started_at_unix: float = field(default_factory=time.time)
    body_size_bucket: str = 'unknown'
    generation: str = 'unknown'
    admissions: int = 0
    sends: int = 0
    last_admission: str = 'unknown'
    parameter_policy: str = 'compat'
    dropped_parameters: set = field(default_factory=set)
    rejected_parameters: set = field(default_factory=set)
    transformed_parameters: set = field(default_factory=set)
    image_trim_policy: str = field(default_factory=lambda: IMAGE_TRIM_POLICY)
    failure_reason: str = 'none'
    error_origin: str = 'none'
    requested_model: str = 'unknown'
    stream: bool = False
    active_attempt: dict = field(default_factory=dict)
    last_error_key: tuple | None = None
    downstream_headers_sent: bool = False
    downstream_content_started: bool = False
    context_budget: dict = field(default_factory=dict)


def log_context():
    record = CURRENT.get()
    if record is None:
        return {}
    return {'lb_request_id': record.request_id, 'api_type': record.api_type,
            'requested_model': record.requested_model, 'stream': record.stream,
            'provider': 'unknown', 'forwarded_model': 'unknown', 'resolved_model': 'unknown',
            'started_at_unix': record.started_at_unix, 'body_size_bucket': record.body_size_bucket,
            'downstream_headers_sent': record.downstream_headers_sent,
            'downstream_content_started': record.downstream_content_started,
            **record.active_attempt}


def note_request_context(model, *, stream=False, body_size=None):
    record = CURRENT.get()
    if record:
        record.requested_model = safe_identifier(model)
        record.stream = bool(stream)
        if type(body_size) is int and body_size >= 0:
            record.body_size_bucket = ('small' if body_size < 64*1024 else 'medium' if body_size < 512*1024
                                       else 'large' if body_size < 4*1024*1024 else 'very_large')


def note_reported_model(value):
    record = CURRENT.get()
    if record is not None and isinstance(value, str):
        record.active_attempt['resolved_model'] = safe_identifier(value)


def note_content_offered(has_content):
    if has_content:
        observe_event(content=True)
        record = CURRENT.get()
        if record is not None:
            record.downstream_content_started = True


def note_stream_end(observation):
    if observation is None:
        return
    record = CURRENT.get()
    log_event({'kind': 'lb_stream_end', 'terminal_seen': observation.terminal is not None,
               'generation_outcome': record.generation if record is not None else 'unknown',
               'failure_reason': record.failure_reason if record is not None else 'unknown',
               'input_tokens': observation.usage.get('input_tokens'),
               'output_tokens': observation.usage.get('output_tokens')})


def note_failure(reason, *, origin='unknown', status=None, exception=None):
    record = CURRENT.get()
    if record is None:
        return
    note_reason(reason)
    record.error_origin = origin
    key = (record.active_attempt.get('upstream_attempt_id'), reason, origin, status)
    if key == record.last_error_key:
        return
    record.last_error_key = key
    log_event({'kind': 'lb_upstream_error', 'reason': record.failure_reason,
               'error_origin': origin, 'upstream_http_status': status,
               'upstream_code_class': record.failure_reason,
               'exc_type': type(exception).__name__ if exception else None})


def note_exception(exc):
    reason = exception_reason(exc)
    note_failure(reason, origin='local' if reason == 'internal_error' else 'transport', exception=exc)


def note_local_terminal(code):
    record = CURRENT.get()
    if record is None:
        return
    mapping = {'upstream_truncated': 'upstream_truncated', 'pool_acquire_timeout': 'pool_timeout',
               'protocol_buffer_limit': 'local_resource_limit', 'local_observer_error': 'local_observer_error',
               'request_deadline_exceeded': 'request_deadline_exceeded'}
    reason = mapping.get(code)
    if reason:
        note_failure(reason, origin='local' if reason in ('local_resource_limit', 'local_observer_error',
                                                        'request_deadline_exceeded') else 'transport')
    elif record.failure_reason == 'none':
        note_failure('unknown')
    note_generation('failed')


def set_parameter_policy(value, *, image_trim=None):
    record = CURRENT.get()
    normalized = (value or 'false').strip().lower()
    if normalized not in ('false','0','true','1'):
        if record:
            record.parameter_policy = 'invalid'
        raise HTTPException(status_code=400,detail={'error':{'code':'invalid_parameter_policy',
            'message':'X-LB-Strict-Parameters must be true, false, 1 or 0.'}})
    if record:
        record.parameter_policy = 'strict' if normalized in ('true','1') else 'compat'
        image_trim = IMAGE_TRIM_POLICY if image_trim is None else image_trim.strip().lower()
        if image_trim not in ('reject', 'allow'):
            raise HTTPException(status_code=400, detail={'error': {'code': 'invalid_image_trim_policy',
                'message': 'X-LB-Image-Trim must be reject or allow.'}})
        record.image_trim_policy = image_trim


def note_parameter_transforms(fields):
    record = CURRENT.get()
    if record is not None and fields:
        values = {value if value in PARAMETERS else 'other' for value in fields}
        record.transformed_parameters.update(values)
        log_event({'kind': 'lb_parameter_policy', 'action': 'transformed', 'parameters': sorted(values)})


def reject_parameters(fields, *, status=400, code='parameter_not_forwarded'):
    values = {value if value in PARAMETERS else 'other' for value in fields}
    record = CURRENT.get()
    if record is not None:
        for value in values - record.rejected_parameters:
            record.metrics.parameter_decisions[('rejected', value)] += 1
        record.rejected_parameters.update(values)
    note_failure('invalid_input', origin='local')
    log_event({'kind': 'lb_parameter_policy', 'action': 'rejected', 'parameters': sorted(values)})
    raise HTTPException(status_code=status, detail={'error': {'code': code,
        'type': 'request_too_large' if status == 413 else 'invalid_request_error',
        'message': 'The gateway cannot preserve the requested semantics on this route. Use the native API or adjust the named parameters.',
        'parameters': sorted(values), 'lb_request_id': current_request_id(), 'retryable': False}})


def require_image_trim_consent():
    record = CURRENT.get()
    if record is None:
        return  # Explicit standalone transformation helper, outside serving.
    if record.parameter_policy == 'strict':
        reject_parameters({'images.trimmed'})
    if record.image_trim_policy != 'allow':
        reject_parameters({'images.trimmed'}, status=413, code='image_trim_requires_consent')
    note_parameter_drops({'images.trimmed'})


def note_parameter_drops(fields):
    record = CURRENT.get()
    if not record or not fields:
        return
    fields = {value if value in PARAMETERS else 'other' for value in fields}
    strict = record.parameter_policy=='strict'
    seen = record.rejected_parameters if strict else record.dropped_parameters
    action = 'rejected' if strict else 'dropped'
    for name in fields-seen:
        record.metrics.parameter_decisions[(action,name)] += 1
    seen.update(fields)
    log_event({'kind':'lb_parameter_policy','lb_request_id':record.request_id,'action':action,'parameters':sorted(fields)})
    if strict:
        raise HTTPException(status_code=400,detail={'error':{'code':'parameter_not_forwarded',
            'message':'The local gateway compatibility policy would remove requested parameters.',
            'parameters':sorted(fields)}})


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
        if record.generation == 'completed':
            record.failure_reason = record.error_origin = 'none'
        elif record.generation == 'failed' and record.failure_reason == 'none':
            record.failure_reason = 'unknown'


def note_reason(reason):
    record = CURRENT.get()
    if record:
        record.failure_reason = reason if reason in REASONS else 'unknown'


def current_request_id():
    record = CURRENT.get()
    return record.request_id if record else None


def note_json_result(payload, api_type, *, observe_model=True):
    """Observe explicit result semantics without changing the response contract."""
    record = CURRENT.get()
    if not record:
        return
    result = assess_json(payload, api_type)
    if observe_model and isinstance(payload, dict):
        note_reported_model(payload.get('model'))
    record.generation = result.outcome
    record.failure_reason = result.reason
    return


async def inference_call(awaitable, provider, api_type, *, model=None, endpoint=None, stream=False):
    record = CURRENT.get()
    metrics = record.metrics if record else TELEMETRY
    metrics.sends[(provider,api_type)] += 1
    attempt_id = str(uuid.uuid4())
    if record:
        record.sends += 1
        record.active_attempt = {'upstream_attempt_id': attempt_id, 'provider': provider,
                                 'forwarded_model': safe_identifier(model), 'resolved_model': 'unknown',
                                 'endpoint_alias': safe_identifier(endpoint), 'upstream_stream': bool(stream)}
        record.last_error_key = None
    context = log_context()
    log_event({'kind': 'lb_upstream_send_start', **context})
    started = time.monotonic()
    result = 'transport_error'
    status = None
    budget = startup_seconds(provider, api_type, model)
    timeline = CURRENT_TIMING.get()
    timing = timeline.begin_attempt(budget) if timeline is not None else None
    try:
        async with startup_budget(budget):
            response = await awaitable
        status = getattr(response,'status_code',None)
        result = f'http_{status//100}xx' if type(status) is int and 100 <= status < 600 else 'unknown'
        return response
    except asyncio.CancelledError:
        result = 'cancelled'
        raise
    except UpstreamStartupTimeout:
        result = 'startup_timeout'
        note_failure('startup_timeout', origin='transport')
        raise
    except Exception as exc:
        note_exception(exc)
        raise
    finally:
        if timing is not None:
            timeline.end_send(timing)
        metrics.send_results[(provider,api_type,result)] += 1
        fields={**context,'kind':'lb_upstream_send_end','lb_request_id':record.request_id if record else None,
                'upstream_attempt_id':attempt_id,'provider':provider,'api_type':api_type,
                'result':result,'upstream_status':status if type(status) is int else None,
                'duration_seconds':round(time.monotonic()-started,6)}
        fields.update(startup_budget_seconds=budget,
                      upstream_headers_received=timing.header_seconds is not None if timing else None,
                      phase_seconds={'send_to_headers': timing.header_seconds if timing else None,
                                     'send_call': timing.send_seconds if timing else None})
        log_event(fields)


class RequestTelemetryMiddleware:
    def __init__(self, app, metrics=TELEMETRY):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        api_type=ROUTES.get(scope.get('path'))
        if scope['type'] != 'http' or scope.get('method') != 'POST' or not api_type:
            return await self.app(scope,receive,send)
        record=RequestRecord(api_type,self.metrics)
        timeline = Timeline()
        timing_token = CURRENT_TIMING.set(timeline)
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
                headers=[(k,v) for k,v in message.get('headers',[]) if k.lower() not in
                         (b'x-lb-request-id',b'x-lb-parameter-policy',b'x-lb-dropped-parameters',
                          b'x-lb-transformed-parameters',b'x-lb-image-trim')]
                headers.extend([(b'x-lb-request-id',record.request_id.encode('ascii')),
                                (b'x-lb-parameter-policy',record.parameter_policy.encode('ascii')),
                                (b'x-lb-image-trim',record.image_trim_policy.encode('ascii'))])
                if record.dropped_parameters:
                    headers.append((b'x-lb-dropped-parameters',','.join(sorted(record.dropped_parameters)).encode('ascii')))
                if record.transformed_parameters:
                    headers.append((b'x-lb-transformed-parameters',','.join(sorted(record.transformed_parameters)).encode('ascii')))
                if record.context_budget:
                    headers.extend([(b'x-lb-context-policy',record.context_budget['budget_mode'].encode('ascii')),
                                    (b'x-lb-capability-status',record.context_budget['capability_status'].encode('ascii'))])
                message={**message,'headers':headers}
            try:
                await send(message)
            except OSError:
                disconnected=True
                raise
            if message['type']=='http.response.start':
                record.downstream_headers_sent = True
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
            elif scope.get('state',{}).get('lb_overloaded'):
                outcome='overloaded'
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
            if outcome == 'completed':
                record.failure_reason = record.error_origin = 'none'
            elif scope.get('state',{}).get('lb_overloaded'):
                record.failure_reason, record.error_origin = 'local_overload', 'local'
            elif outcome in ('client_disconnected', 'cancelled', 'internal_error', 'deadline_exceeded'):
                record.failure_reason = 'request_deadline_exceeded' if outcome == 'deadline_exceeded' else outcome
                record.error_origin = 'client' if outcome in ('client_disconnected', 'cancelled') else 'local'
            elif record.failure_reason == 'none':
                record.failure_reason = failure_reason({}, status) if status and status >= 400 else 'unknown'
                record.error_origin = 'unknown'
            elapsed=time.monotonic()-record.started
            self.metrics.outcomes[(api_type,outcome)] += 1
            self.metrics.reasons[(api_type,record.failure_reason)] += 1
            self.metrics.latency_sum[(api_type,outcome)] += elapsed
            for upper in BUCKETS:
                if elapsed <= upper:
                    self.metrics.latency_buckets[(api_type,outcome,upper)] += 1
            self.metrics.active[api_type] -= 1
            context = log_context()
            context.update(timeline.snapshot())
            self.metrics.phases.record(api_type, timeline)
            CURRENT.reset(token)
            CURRENT_TIMING.reset(timing_token)
            fields={**context,'kind':'lb_request_end','lb_request_id':record.request_id,'api_type':api_type,
                    'request_id':safe_id(scope.get('state',{}).get('request_id')),
                    'operation_id':operation_id,
                    'source_tenant':scope.get('state',{}).get('lb_source_tenant'),
                    'draining_at_finish':scope.get('state',{}).get('lb_draining_at_finish'),
                    'admission_reason':scope.get('state',{}).get('lb_admission_reason'),
                    'parameter_policy':record.parameter_policy,'dropped_parameters':sorted(record.dropped_parameters),
                    'rejected_parameters':sorted(record.rejected_parameters),
                    'transformed_parameters':sorted(record.transformed_parameters), 'image_trim_policy':record.image_trim_policy,
                    'outcome':outcome,'http_status':status,'generation_outcome':record.generation,
                    'failure_reason':record.failure_reason,
                    'error_origin':record.error_origin,
                    'downstream_body_completed':body_complete,'admissions':record.admissions,
                    'upstream_sends':record.sends,'duration_seconds':round(elapsed,6)}
            log_event(fields)
