"""Offline matched-cohort evidence, not an automatic rollout. Author: Zeno Ren."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

OUTCOMES = {'completed', 'failed', 'incomplete', 'unknown', 'http_error', 'rejected', 'overloaded',
            'cancelled', 'client_disconnected', 'internal_error', 'deadline_exceeded'}
COHORT_FIELDS = ('provider', 'requested_model', 'forwarded_model', 'api_type', 'stream', 'body_size_bucket', 'source_tenant')
RESOURCES = ('rss_bytes', 'admission_active')
SAFETY_EVENTS = ('replay_after_content', 'duplicate_usage', 'privacy_leak')


def number(value):
    return type(value) in (int, float) and 0 <= value < 2**63 and math.isfinite(value)


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Window timestamps require a timezone')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Window timestamps require a timezone')
    return result.timestamp()


def validate_plan(plan):
    keys = {'schema_version', 'timezone', 'hours', 'min_samples_per_arm', 'min_completion_gain',
            'max_p95_ratio', 'max_disconnect_increase', 'max_resource_ratio', 'max_send_amplification_increase'}
    if not isinstance(plan, dict) or set(plan) != keys or plan['schema_version'] != 1:
        raise ValueError('An explicit, complete experiment plan is required')
    if type(plan['min_samples_per_arm']) is not int or plan['min_samples_per_arm'] < 1:
        raise ValueError('Minimum sample size must be positive')
    if not isinstance(plan['hours'], list) or not plan['hours'] or any(type(h) is not int or not 0 <= h < 24 for h in plan['hours']):
        raise ValueError('Explicit comparison hours are required')
    for name in keys - {'schema_version', 'timezone', 'hours', 'min_samples_per_arm'}:
        if not number(plan[name]):
            raise ValueError('Experiment thresholds must be finite nonnegative numbers')
    if any(plan[k] > 1 for k in ('min_completion_gain', 'max_disconnect_increase')) or any(
            plan[k] <= 0 for k in ('max_p95_ratio', 'max_resource_ratio')):
        raise ValueError('Invalid experiment rate or ratio threshold')
    return ZoneInfo(plan['timezone'])


def cohorts(dataset, zone, hours):
    if dataset.get('schema_version') != 1 or not isinstance(dataset.get('requests'), list):
        raise ValueError('Invalid experiment dataset')
    rows, pending = dataset['requests'], dataset.get('pending_requests')
    if type(pending) is not int or pending < 0 or type(dataset.get('cohort_started')) is not int or len(rows) + pending != dataset['cohort_started']:
        raise ValueError('Request rows and pending count do not match the started cohort denominator')
    left, right = timestamp(dataset.get('window_start')), timestamp(dataset.get('window_end'))
    if left >= right:
        raise ValueError('Invalid cohort window')
    grouped, seen = defaultdict(list), set()
    for row in rows:
        identifier = row.get('lb_request_id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Missing or duplicate server request identity')
        seen.add(identifier)
        if (row.get('outcome') not in OUTCOMES or not number(row.get('duration_seconds'))
                or not number(row.get('started_at_unix')) or type(row.get('upstream_sends')) is not int or row['upstream_sends'] < 0):
            raise ValueError('Invalid request outcome, phase or send count')
        start = row['started_at_unix']
        if not left <= start < right:
            raise ValueError('Request start falls outside its declared cohort window')
        values = []
        for key in COHORT_FIELDS:
            value = row.get(key)
            if key == 'stream':
                if type(value) is not bool: raise ValueError('Streaming mode must be explicit')
            elif not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,128}', value) or '://' in value:
                raise ValueError('Cohort labels must be bounded safe identifiers')
            values.append(value)
        hour = datetime.fromtimestamp(start, zone).hour
        if hour in hours:
            grouped[tuple(values) + (hour,)].append(row)
    return grouped


def wilson(completed, count):
    # Descriptive 95% binomial interval; not a causal or independence guarantee.
    if count == 0: return [0, 1]
    z = 1.959963984540054
    fraction = completed / count
    denominator = 1 + z*z/count
    center = (fraction + z*z/(2*count)) / denominator
    radius = z * math.sqrt(fraction*(1-fraction)/count + z*z/(4*count*count)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def summary(rows):
    count = len(rows)
    outcomes = Counter(row['outcome'] for row in rows)
    durations = sorted(row['duration_seconds'] for row in rows)
    return {'requests': count, 'outcomes': {key: outcomes[key] for key in sorted(OUTCOMES)},
            'completion_rate': outcomes['completed']/count if count else None,
            'completion_interval_95': wilson(outcomes['completed'], count),
            'disconnect_rate': outcomes['client_disconnected']/count if count else None,
            'send_amplification': sum(row['upstream_sends'] for row in rows)/count if count else None,
            'p95_seconds': durations[math.ceil(.95*count)-1] if count else None}


def compare(baseline, candidate, plan):
    zone = validate_plan(plan)
    left, right = (cohorts(data, zone, plan['hours']) for data in (baseline, candidate))
    reasons, stop = [], []
    resources = {}
    scopes = {data.get('reported_scope') for data in (baseline, candidate)}
    if not scopes <= {'synthetic', 'reported_production', 'unknown'}:
        raise ValueError('Dataset scope must be explicit')
    for arm, data in (('baseline', baseline), ('candidate', candidate)):
        identity = (data.get('identity') or {}).get('image_digest')
        if not isinstance(identity, str) or not re.fullmatch(r'sha256:[a-f0-9]{64}', identity):
            reasons.append(arm + '_identity_unknown')
        if data.get('coverage') != 'complete': reasons.append(arm + '_coverage_incomplete')
        if data.get('pending_requests'): reasons.append(arm + '_requests_pending')
        if data.get('diagnostic_dropped_events') != 0: reasons.append(arm + '_diagnostic_loss_unknown_or_positive')
        for name in SAFETY_EVENTS:
            value = (data.get('safety_events') or {}).get(name)
            if type(value) is not int or value < 0: reasons.append(arm + '_' + name + '_unknown')
            elif value > 0: stop.append(arm + '_' + name)
    for name in RESOURCES:
        a, b = ((data.get('resource_peaks') or {}).get(name) for data in (baseline, candidate))
        if not number(a) or not number(b) or a == 0:
            reasons.append(name + '_comparison_unknown')
        else:
            resources[name + '_ratio'] = b / a
            if b / a > plan['max_resource_ratio']: stop.append(name + '_regressed')
    compared = []
    for key in sorted(set(left) | set(right)):
        a, b = summary(left.get(key, [])), summary(right.get(key, []))
        enough = min(a['requests'], b['requests']) >= plan['min_samples_per_arm']
        status = 'inconclusive'
        if enough:
            improvement = b['completion_interval_95'][0] - a['completion_interval_95'][1]
            latency_ok = b['p95_seconds'] <= a['p95_seconds'] * plan['max_p95_ratio']
            disconnect_ok = b['disconnect_rate'] <= a['disconnect_rate'] + plan['max_disconnect_increase']
            amplification_ok = b['send_amplification'] <= a['send_amplification'] + plan['max_send_amplification_increase']
            status = 'eligible_for_review' if improvement >= plan['min_completion_gain'] and latency_ok and disconnect_ok and amplification_ok else 'not_supported'
        compared.append({'cohort': dict(zip(COHORT_FIELDS + ('hour',), key)), 'baseline': a, 'candidate': b, 'status': status})
    if not compared: reasons.append('no_matching_window_samples')
    status = ('stop' if stop else 'inconclusive' if reasons or any(row['status'] == 'inconclusive' for row in compared)
              else 'not_supported' if any(row['status'] == 'not_supported' for row in compared) else 'eligible_for_review')
    return {'schema_version': 1, 'author': 'Zeno Ren', 'status': status, 'production_acceptance': False,
            'reported_scopes': sorted(scopes), 'reasons': reasons, 'stop_reasons': stop,
            'resource_comparison': resources, 'cohorts': compared,
            'statistical_limit': 'Wilson intervals describe samples; traffic is not proven independent or randomized.',
            'next_step': 'Human and client/protocol review; this tool never changes settings or calls a model.'}


def read(path):
    with Path(path).open('rb') as source:
        raw = source.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024: raise ValueError('Experiment input exceeds 64 MiB')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'plan', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    inputs = {name: read(getattr(args, name)) for name in ('baseline', 'candidate', 'plan')}
    result = compare(*(inputs[name][0] for name in ('baseline', 'candidate', 'plan')))
    result['input_sha256'] = {name: item[1] for name, item in inputs.items()}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False); output.write('\n')
        output.flush(); os.fsync(output.fileno())
    print(json.dumps({'status': result['status'], 'production_acceptance': False, 'cohorts': len(result['cohorts'])}))
