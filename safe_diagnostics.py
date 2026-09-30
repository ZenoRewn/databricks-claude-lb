"""Bounded, allowlisted diagnostics; never a request/response archive. Author: Zeno Ren."""
from collections import Counter
import copy
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time

SCHEMA_VERSION = 1
IDENTIFIERS = set('kind lb_request_id request_id req operation_id upstream_attempt_id provider api_type '
                  'requested_model forwarded_model resolved_model model endpoint endpoint_alias connection_id '
                  'source_tenant outcome generation_outcome failure_reason reason error_origin result action '
                  'parameter_policy admission_reason selection_reason execution_certainty first_event last_event '
                  'error_type exc_type source classification http_version upstream_code_class deadline_phase '
                  'body_size_bucket client_class capability_status estimate_method estimate_confidence context_status '
                  'budget_mode route image_trim_policy previous_state circuit_state transition_reason'.split())
NUMBERS = set('schema_version upstream_status upstream_http_status http_status duration_seconds attempt admissions '
              'upstream_sends input_bytes body_size_bytes input_tokens output_tokens cache_read_tokens chunks '
              'chunks_yielded decoded_bytes decoded_deliveries frames pending_eof_bytes peak_pending_bytes '
              'eligible_count tried_count untried_count remaining_loop_attempts remaining_seconds retry_after_seconds '
              'pool_saturated upstream_stall httpx_active httpx_total max_attempts attempts error_count '
              'startup_budget_seconds total_budget_seconds diagnostic_dropped_events source_line estimated_input_tokens '
              'reserved_output_tokens input_limit context_limit output_limit age_seconds text_bytes image_count started_at_unix '
              'circuit_generation consecutive_errors'.split())
BOOLEANS = set('retry retry_allowed retryable retry_after_present downstream_headers_sent downstream_content_started '
               'downstream_body_completed draining_at_finish saw_completion terminal_seen terminal_valid has_image '
               'sent_any_chunk account_neutral read_timeout probe_ok httpx_pool_observed_full upstream_headers_received '
               'stream upstream_stream estimate_complete enforcement_allowed'.split())
LIST_FIELDS = {'parameters', 'dropped_parameters', 'rejected_parameters', 'transformed_parameters', 'removed_params', 'unknown_components'}
POOL_KEYS = {'total', 'active', 'idle', 'closing', 'requests_waiting', 'max_connections', 'max_keepalive_connections'}
PROBE_KEYS = {'ok', 'cached', 'dns_ms', 'tcp_ms'}
DROP_REASONS = ('queue_full', 'sink_error', 'event_error', 'closed')
DIAGNOSTIC_DROPS = Counter()
QUEUE_CAPACITY = int(os.getenv('LB_DIAGNOSTIC_QUEUE_CAPACITY', '4096'))
if not 1 <= QUEUE_CAPACITY <= 65536:
    raise ValueError('LB_DIAGNOSTIC_QUEUE_CAPACITY must be between 1 and 65536')
REASON_VALUES = set('none unknown context_window_exceeded invalid_input rate_limited authentication '
    'upstream_unavailable invalid_protocol upstream_failure output_limit transport_protocol_error startup_timeout '
    'read_timeout write_timeout connection_error pool_timeout upstream_truncated request_deadline_exceeded '
    'client_disconnected cancelled local_resource_limit local_observer_error internal_error local_overload request_body_timeout retry_429 '
    'upstream_cooldown attempt_budget_exhausted status_not_retryable pool_acquire_timeout'.split())
EVENT_NAMES = set('error ping message_start message_delta message_stop content_block_start content_block_delta '
    'content_block_stop response.created response.in_progress response.completed response.failed response.incomplete '
    'response.output_item.added response.output_item.done response.content_part.added response.content_part.done '
    'response.output_text.delta response.output_text.done response.output_text.annotation.added '
    'response.function_call_arguments.delta response.function_call_arguments.done response.refusal.delta '
    'response.refusal.done response.reasoning_summary_part.added response.reasoning_summary_part.done '
    'response.reasoning_summary_text.delta response.reasoning_summary_text.done -'.split())
REQUEST_FUNCTIONS = {'stream_generator', 'stream_with_heartbeat', '_stream_response', '_stream_request',
    '_normal_request', '_proxy_request', 'proxy_request', 'proxy_responses', 'proxy_chat', '_proxy',
    '_log_openai_request', '_route_openai', '_route_openai_responses', '_route_chat_via_responses',
    'messages', 'responses', 'chat_completions', '_periodic_flush', '_read_token_from_file',
    'background_refresh_loop', '_connection_monitor_loop', 'release_abandoned_request'}
_STANDARD_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__) | {'message', 'asctime'}


def safe_identifier(value, default='unknown'):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,128}', value):
        return default
    if '://' in value or value.startswith(('ghp_', 'gho_', 'ghu_', 'github_pat_', 'sk-')):
        return default
    return value


def safe_number(value):
    if type(value) is int:
        return value if abs(value) < 2**63 else None
    return value if type(value) is float and math.isfinite(value) else None


def safe_fields(fields):
    """No arbitrary recursion, repr(), exception messages, UA, body, schema or headers."""
    result = {}
    for key in IDENTIFIERS & fields.keys():
        result[key] = safe_identifier(fields[key], None if fields[key] is None else 'unknown')
        allowed = REASON_VALUES if key in ('reason', 'failure_reason', 'upstream_code_class') else EVENT_NAMES if key in ('first_event', 'last_event') else None
        if allowed is not None and result[key] is not None and result[key] not in allowed:
            result[key] = 'unknown'
    for key in NUMBERS & fields.keys():
        result[key] = safe_number(fields[key])
    for key in BOOLEANS & fields.keys():
        result[key] = fields[key] if type(fields[key]) is bool else None
    for key in LIST_FIELDS & fields.keys():
        value = fields[key]
        result[key] = [safe_identifier(item) for item in value[:32]] if isinstance(value, (list, tuple)) else []
    elapsed = fields.get('elapsed')
    if isinstance(elapsed, str) and re.fullmatch(r'[0-9]{1,12}\.[0-9]{1,6}s', elapsed):
        result['elapsed'] = elapsed
    for name, keys in (('httpx_pool', POOL_KEYS), ('upstream_probe', PROBE_KEYS)):
        value = fields.get(name)
        if isinstance(value, dict):
            result[name] = {k: v if type(v) is bool else safe_number(v) for k, v in value.items() if k in keys}
    for name in ('phase_seconds', 'phase_offsets_seconds'):
        value = fields.get(name)
        if isinstance(value, dict):
            result[name] = {safe_identifier(k): safe_number(v) for k, v in list(value.items())[:32]}
    return result


class DiagnosticFilter(logging.Filter):
    """Protect before any handler, including synchronous third-party capture handlers.

    Legacy unstructured inference messages lose their free text. Their source and
    severity remain; supported structured events retain allowlisted fields. This
    avoids trying to redact arbitrary prompts after formatting an exception.
    """
    def __init__(self, context=lambda: {}):
        super().__init__()
        self.context = context

    def filter(self, record):
        try:
            context = self.context()
            kind = getattr(record, 'kind', None)
            if not kind and not context and record.funcName not in REQUEST_FUNCTIONS:
                return True
            fields = safe_fields({**context, **{k: v for k, v in record.__dict__.items() if k not in _STANDARD_RECORD_KEYS}})
            fields.setdefault('kind', 'lb_legacy_diagnostic')
            if not kind:
                fields.update(source=safe_identifier(record.funcName), source_line=record.lineno)
            fields['schema_version'] = SCHEMA_VERSION
            for key in tuple(record.__dict__):
                if key not in _STANDARD_RECORD_KEYS:
                    del record.__dict__[key]
            record.__dict__.update(fields)
            if fields['kind'] in ('copilot_stream_end', 'copilot_request_end'):
                label = 'stream_end' if fields['kind'] == 'copilot_stream_end' else 'request_end'
                record.msg = '[Copilot ' + label + '] ' + ' '.join(f'{k}={v}' for k, v in fields.items() if k != 'kind')
            elif fields['kind'] == 'copilot_token_exchange_error':
                record.msg = fields['kind'] + ' ' + fields.get('error_type', 'unknown')
            else:
                record.msg = fields['kind']
            record.args = ()
            record.exc_info = record.exc_text = record.stack_info = None
        except Exception:
            DIAGNOSTIC_DROPS['event_error'] += 1
            return False
        return True


class BoundedLogHandler(logging.Handler):
    """Default process sink: enqueue without waiting for I/O; bounded shutdown.

    A stuck sink owns one daemon thread, not an inference task. Shutdown does not
    promise lossless diagnostics, and never blocks request cleanup waiting for it.
    """
    def __init__(self, sink, *, capacity=4096, drops=None):
        if type(capacity) is not int or not 1 <= capacity <= 65536:
            raise ValueError('Diagnostic queue capacity must be an integer from 1 to 65536')
        super().__init__()
        self.sink = sink
        self.queue = queue.Queue(capacity)
        self.dropped = Counter() if drops is None else drops
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._work, name='lb-diagnostic-sink', daemon=True)
        self.worker.start()

    def emit(self, record):
        if self.stopping.is_set():
            self.dropped['closed'] += 1
            return
        try:
            item = copy.copy(record)
            # Only non-inference administrative records retain a bounded message.
            # Safe inference records have already been filtered before handlers.
            item.msg = record.getMessage()[:16384]
            item.args = ()
            item.exc_info = item.exc_text = item.stack_info = None
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped['queue_full'] += 1
        except Exception:
            self.dropped['event_error'] += 1

    def _work(self):
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                self.sink.handle(item)
            except Exception:
                self.dropped['sink_error'] += 1
            finally:
                self.queue.task_done()

    def drain(self, timeout=1):
        deadline = time.monotonic() + timeout
        with self.queue.all_tasks_done:
            while self.queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.queue.all_tasks_done.wait(remaining)
        return True

    def close(self):
        self.stopping.set()
        if self.worker is not threading.current_thread():
            self.worker.join(timeout=.2)
        super().close()


class DiagnosticTextFormatter(logging.Formatter):
    """Retain the same safe fields as JSON mode in the default text output."""
    def format(self, record):
        if getattr(record, 'kind', None):
            fields = safe_fields(record.__dict__)
            if fields.get('kind') in ('copilot_stream_end', 'copilot_request_end'):
                label = 'stream_end' if fields['kind'] == 'copilot_stream_end' else 'request_end'
                payload = '[Copilot ' + label + '] ' + ' '.join(f'{k}={v}' for k, v in fields.items() if k != 'kind')
            else:
                payload = json.dumps(fields, ensure_ascii=False, separators=(',', ':'))
            return f'{record.levelname}:{record.name}:{payload}'
        return super().format(record)


class DiagnosticStreamHandler:
    """Single-consumer stream sink owned by the bounded handler's daemon.

    This is deliberately not a logging.Handler: registering a second handler
    would let logging.shutdown acquire its blocked I/O lock or flush it again
    after the queue owner's bounded close returned. Only the worker writes and
    flushes this sink; failures propagate to its observable drop counter.
    """
    def __init__(self, stream=None):
        self.stream = sys.stderr if stream is None else stream
        self.formatter = logging.Formatter()

    def setFormatter(self, formatter):
        self.formatter = formatter

    def handle(self, record):
        self.stream.write(self.formatter.format(record) + '\n')
        self.stream.flush()


def default_handler(sink):
    return BoundedLogHandler(sink, capacity=QUEUE_CAPACITY,
                             drops=DIAGNOSTIC_DROPS)


def render_metrics():
    name = 'lb_diagnostic_dropped_events_total'
    return '\n'.join([f'# HELP {name} Diagnostic loss, independent of inference results',
                      f'# TYPE {name} counter',
                      *[f'{name}{{reason="{reason}"}} {DIAGNOSTIC_DROPS[reason]}' for reason in DROP_REASONS]]) + '\n'
