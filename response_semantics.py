"""Protocol result, failure attribution, and reported usage. Author: Zeno Ren."""
from dataclasses import dataclass, field
import json
import re
import httpx

REASONS = ('none', 'context_window_exceeded', 'invalid_input', 'rate_limited',
           'authentication', 'upstream_unavailable', 'invalid_protocol',
           'upstream_failure', 'output_limit', 'unknown', 'transport_protocol_error',
           'startup_timeout', 'read_timeout', 'write_timeout', 'connection_error',
           'pool_timeout', 'upstream_truncated', 'request_deadline_exceeded',
           'client_disconnected', 'cancelled', 'local_resource_limit',
           'local_observer_error', 'internal_error', 'local_overload', 'request_body_timeout', 'recovery_preference')


def exception_reason(exc):
    # Classification is diagnostic only; it never authorizes POST replay.
    if type(exc).__name__ == 'UpstreamStartupTimeout':
        return 'startup_timeout'
    for cls, reason in ((httpx.PoolTimeout, 'pool_timeout'),
                        (httpx.RemoteProtocolError, 'transport_protocol_error'),
                        (httpx.LocalProtocolError, 'invalid_protocol'),
                        (httpx.ReadTimeout, 'read_timeout'),
                        (httpx.WriteTimeout, 'write_timeout'),
                        (httpx.ConnectTimeout, 'connection_error'),
                        (httpx.ConnectError, 'connection_error'),
                        (httpx.ReadError, 'transport_protocol_error'),
                        (httpx.WriteError, 'transport_protocol_error')):
        if isinstance(exc, cls):
            return reason
    return 'internal_error'


def decode_json_response(response):
    class Pairs(list):pass
    if isinstance(response,httpx.Response):
        def bad_constant(value):raise ValueError('Non-finite JSON value')
        value=json.loads(response.content,object_pairs_hook=Pairs,parse_constant=bad_constant)
        if isinstance(value,Pairs):
            seen=set()
            for name,_ in value:
                if name in seen and name in ('type','status','error','usage','stop_reason','choices','id','output','incomplete_details'):
                    raise ValueError('Duplicate protocol discriminator')
                seen.add(name)
        def restore(item):
            if isinstance(item,Pairs):return {k:restore(v) for k,v in item}
            if isinstance(item,list):return [restore(v) for v in item]
            return item
        value=restore(value)
    else:value=response.json()
    # Validate before accounting or returning a response. JSONResponse cannot
    # serialize NaN or an unpaired surrogate; neither proves valid generation.
    json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
    return value


def is_html(content_type):
    return isinstance(content_type, str) and content_type.split(';', 1)[0].strip().lower() in ('text/html', 'application/xhtml+xml')


def invalid_stream_type(content_type):
    media=content_type.split(';',1)[0].strip().lower() if isinstance(content_type,str) else ''
    return is_html(content_type) or media=='application/json' or media.endswith('+json')


def failure_reason(payload, status=None):
    root = payload if isinstance(payload, dict) else {}
    error = root.get('error')
    error = error if isinstance(error, dict) else root
    code = str(error.get('code') or root.get('error_code') or '').lower()
    message = str(error.get('message') or root.get('message') or '').lower()
    if code in ('context_length_exceeded', 'context_window_exceeded', 'input_too_long') or (
        status == 400 and ('exceeds context window' in message or 'exceeds the context window' in message)
    ):
        return 'context_window_exceeded'
    if status == 429:
        return 'rate_limited'
    if status in (401, 403):
        return 'authentication'
    if code in ('invalid_prompt', 'invalid_request_error') or status in (400, 404, 413, 422):
        return 'invalid_input'
    if code in ('max_output_tokens', 'length', 'content_filter'):
        return 'output_limit'
    if status in (502, 503, 504):
        return 'upstream_unavailable'
    return 'upstream_failure'


def reported_usage(payload, api):
    """Missing counters stay unknown; never invent usage from an error envelope."""
    usage = payload.get('usage') if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {}
    keys = {'input_tokens': 'prompt_tokens' if api == 'chat' else 'input_tokens',
            'output_tokens': 'completion_tokens' if api == 'chat' else 'output_tokens',
            'cache_creation_tokens': 'cache_creation_input_tokens',
            'cache_read_tokens': 'cache_read_input_tokens'}
    result = {}
    for target, source in keys.items():
        if source in usage:
            value = usage[source]
            if type(value) is not int or not 0 <= value < 2**63:
                return {}  # An inconsistent usage object is not authoritative.
            result[target] = value
    detail = usage.get('prompt_tokens_details' if api == 'chat' else 'input_tokens_details')
    if isinstance(detail, dict):
        for source, target in (('cached_tokens', 'cache_read_tokens'), ('cache_write_tokens', 'cache_creation_tokens')):
            if source in detail:
                value = detail[source]
                if type(value) is not int or not 0 <= value < 2**63:
                    return {}
                result[target] = value
    return result


@dataclass(frozen=True)
class ResultAssessment:
    outcome: str
    reason: str = 'none'
    usage: dict = field(default_factory=dict)

    @property
    def neutral(self):
        return self.outcome in ('incomplete', 'unknown') or self.reason in ('context_window_exceeded', 'invalid_input', 'output_limit')


def assess_json(payload, api):
    usage = reported_usage(payload, api)
    if not isinstance(payload, dict):
        return ResultAssessment('failed', 'invalid_protocol')
    if ('error' in payload and payload['error'] is not None) or payload.get('status') == 'failed':
        return ResultAssessment('failed', failure_reason(payload), usage)
    if payload.get('status') == 'incomplete' or payload.get('incomplete_details'):
        return ResultAssessment('incomplete', 'output_limit', usage)
    if api == 'messages':
        reason = payload.get('stop_reason')
        if payload.get('type') == 'message' and reason in ('end_turn', 'tool_use', 'stop_sequence', 'max_tokens'):
            return ResultAssessment('incomplete' if reason == 'max_tokens' else 'completed',
                                    'output_limit' if reason == 'max_tokens' else 'none', usage)
    elif api == 'responses':
        if payload.get('status') == 'completed' and isinstance(payload.get('id'), str) and isinstance(payload.get('output'), list):
            return ResultAssessment('completed', usage=usage)
    elif api == 'chat':
        choices = payload.get('choices')
        if isinstance(choices, list) and choices and all(isinstance(c, dict) for c in choices):
            reasons = [c.get('finish_reason') for c in choices]
            if all(r in ('stop', 'tool_calls', 'function_call', 'length', 'content_filter') for r in reasons):
                incomplete = any(r in ('length', 'content_filter') for r in reasons)
                return ResultAssessment('incomplete' if incomplete else 'completed', 'output_limit' if incomplete else 'none', usage)
    return ResultAssessment('failed', 'invalid_protocol', usage)


def enrich_error(detail, *, status, payload=None, headers=None, request_id=None):
    root = payload if isinstance(payload, dict) else {}
    original = root.get('error') if isinstance(root.get('error'), dict) else root
    error = dict(detail.get('error') or {})
    if not isinstance(error.get('message'),str):error['message']='Upstream request failed.'
    error['message']=error['message'][:500]
    code = original.get('code') or root.get('error_code')
    if isinstance(code, str) and re.fullmatch(r'[\w.:-]{1,128}', code):
        error['upstream_code'] = code
    reason = 'invalid_protocol' if error.get('code') in ('upstream_html_error', 'invalid_upstream_response') else failure_reason(root, status)
    if not isinstance(error.get('code'),str) or not re.fullmatch(r'[\w.:-]{1,128}',error['code']):
        error['code']=code if isinstance(code,str) and re.fullmatch(r'[\w.:-]{1,128}',code) else 'upstream_error'
    error.update(reason=reason, upstream_status=status, retryable=status==429,
                 execution_certainty='admission_rejected' if status == 429 else 'rejected' if reason in ('context_window_exceeded','invalid_input') else 'unknown')
    retry_after = (headers or {}).get('Retry-After')
    if isinstance(retry_after, str) and len(retry_after) <= 128 and '\r' not in retry_after and '\n' not in retry_after:
        error['retry_after'] = retry_after
    if request_id:
        error['lb_request_id'] = request_id
    return {'error': error}
