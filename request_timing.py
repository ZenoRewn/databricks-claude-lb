"""Monotonic observations, never inferred network phases. Author: Zeno Ren."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import time

CURRENT = ContextVar('lb_request_timing', default=None)
PHASES = ('body_read', 'admission_wait', 'auth_prepare', 'send_to_headers', 'send_call',
          'first_event_from_ingress', 'first_content_from_ingress', 'stream', 'cleanup')
UNOBSERVED = ('dns', 'tcp', 'tls', 'pool')
BUCKETS = (.01, .1, .5, 1, 2, 5, 10, 30, 60, 120, 180, 300, 600, 1800)


@dataclass
class AttemptTiming:
    started: float
    startup_budget: float
    header_seconds: float | None = None
    send_seconds: float | None = None
    stream_started: float | None = None


class Timeline:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.seconds = {}
        self.offsets = {}
        self.depths = Counter()
        self.phase_starts = {}
        self.active_attempt = None
        self.total_budget = None
        self.deadline_phase = None

    @contextmanager
    def measure(self, phase):
        if phase not in PHASES:
            raise ValueError('Unknown measurement phase')
        began = self.clock()
        if self.depths[phase] == 0:
            self.phase_starts[phase] = began
        self.depths[phase] += 1
        self.offsets.setdefault(phase + '_started', max(0, began - self.started))
        try:
            yield
        finally:
            self.depths[phase] -= 1
            if self.depths[phase] == 0:
                self.seconds[phase] = self.seconds.get(phase, 0) + max(0, self.clock() - self.phase_starts.pop(phase))
                self.offsets[phase + '_finished'] = max(0, self.clock() - self.started)

    def begin_attempt(self, startup_budget):
        self.active_attempt = AttemptTiming(self.clock(), startup_budget)
        return self.active_attempt

    def headers_received(self):
        attempt = self.active_attempt
        if attempt is not None and attempt.header_seconds is None:
            attempt.header_seconds = max(0, self.clock() - attempt.started)
            self.seconds['send_to_headers'] = self.seconds.get('send_to_headers', 0) + attempt.header_seconds

    def end_send(self, attempt):
        attempt.send_seconds = max(0, self.clock() - attempt.started)
        self.seconds['send_call'] = self.seconds.get('send_call', 0) + attempt.send_seconds

    def begin_stream(self):
        if self.active_attempt is not None and self.active_attempt.stream_started is None:
            self.active_attempt.stream_started = self.clock()

    def end_stream(self):
        attempt = self.active_attempt
        if attempt is not None and attempt.stream_started is not None:
            self.seconds['stream'] = self.seconds.get('stream', 0) + max(0, self.clock() - attempt.stream_started)
            attempt.stream_started = None

    def event(self, *, content=False):
        phase = 'first_content_from_ingress' if content else 'first_event_from_ingress'
        self.seconds.setdefault(phase, max(0, self.clock() - self.started))

    def current_phase(self):
        for name in ('cleanup', 'auth_prepare', 'body_read', 'admission_wait'):
            if self.depths[name]:
                return name
        attempt = self.active_attempt
        if attempt is not None:
            if attempt.send_seconds is None:
                return 'upstream_headers' if attempt.header_seconds is None else 'upstream_body'
            if attempt.stream_started is not None:
                return 'stream'
        return None

    def capture_deadline(self):
        self.deadline_phase = self.current_phase()

    def snapshot(self):
        return {'phase_seconds': {phase: self.seconds.get(phase) for phase in PHASES + UNOBSERVED},
                'phase_offsets_seconds': dict(self.offsets),
                'total_budget_seconds': self.total_budget,
                'startup_budget_seconds': self.active_attempt.startup_budget if self.active_attempt else None,
                'deadline_phase': self.deadline_phase}


def observed_phase(name):
    def decorate(function):
        @wraps(function)
        async def measured(*args, **kwargs):
            timeline = CURRENT.get()
            if timeline is None:
                return await function(*args, **kwargs)
            with timeline.measure(name):
                return await function(*args, **kwargs)
        return measured
    return decorate


def begin_stream():
    timeline = CURRENT.get()
    if timeline is not None:
        timeline.begin_stream()


def end_stream():
    timeline = CURRENT.get()
    if timeline is not None:
        timeline.end_stream()


def observe_event(*, content=False):
    timeline = CURRENT.get()
    if timeline is not None:
        timeline.event(content=content)


class PhaseMetrics:
    def __init__(self):
        self.counts, self.seconds, self.buckets = Counter(), Counter(), Counter()

    def record(self, api_type, timeline):
        for phase, value in timeline.seconds.items():
            if phase not in PHASES:
                continue
            self.counts[(api_type, phase)] += 1
            self.seconds[(api_type, phase)] += value
            for upper in BUCKETS:
                if value <= upper:
                    self.buckets[(api_type, phase, upper)] += 1

    def render(self):
        name = 'lb_request_phase_duration_seconds'
        lines = [f'# HELP {name} Observed per-request phase durations including failures; nested phases are not additive',
                 f'# TYPE {name} histogram']
        for api in ('messages', 'responses', 'chat'):
            for phase in PHASES:
                labels = f'api_type="{api}",phase="{phase}"'
                for upper in BUCKETS:
                    lines.append(f'{name}_bucket{{{labels},le="{upper}"}} {self.buckets[(api, phase, upper)]}')
                count = self.counts[(api, phase)]
                lines.extend([f'{name}_bucket{{{labels},le="+Inf"}} {count}',
                              f'{name}_count{{{labels}}} {count}',
                              f'{name}_sum{{{labels}}} {self.seconds[(api, phase)]}'])
        return '\n'.join(lines) + '\n'
