"""Versioned Prometheus contract and strict label-preserving parser. Author: Zeno Ren."""
import json
import math
import re


def definition(kind, unit, labels=()):
    return {'type':kind,'unit':unit,'labels':list(labels)}


FAMILIES = {
    'lb_requests_started_total':definition('counter','requests',('api_type',)),
    'lb_requests_active':definition('gauge','requests',('api_type',)),
    'lb_requests_finished_total':definition('counter','requests',('api_type','outcome')),
    'lb_request_reasons_total':definition('counter','requests',('api_type','reason')),
    'lb_endpoint_admissions_total':definition('counter','admissions',('api_type',)),
    'lb_upstream_send_started_total':definition('counter','sends',('provider','api_type')),
    'lb_upstream_send_finished_total':definition('counter','sends',('provider','api_type','result')),
    'lb_retry_decisions_total':definition('counter','decisions',('reason',)),
    'lb_parameter_policy_requests_total':definition('counter','requests',('action','parameter')),
    'lb_request_duration_seconds':definition('histogram','seconds',('api_type','outcome')),
    'lb_admission_active':definition('gauge','requests'),
    'lb_admission_queued':definition('gauge','requests'),
    'lb_admission_body_bytes':definition('gauge','bytes'),
    'lb_admission_draining':definition('gauge','boolean'),
    'lb_admission_rejected_total':definition('counter','requests',('reason',)),
    'lb_admission_queue_wait_seconds':definition('summary','seconds'),
    'lb_usage_pending_events':definition('gauge','events'),
    'lb_usage_pending_groups':definition('gauge','groups'),
    'lb_usage_inflight_batch':definition('gauge','batches'),
    'lb_usage_flush_failures_total':definition('counter','batches'),
    'lb_usage_flush_successes_total':definition('counter','batches'),
    'lb_usage_rejected_events_total':definition('counter','events'),
    'lb_usage_last_success_timestamp_seconds':definition('gauge','unix_seconds'),
    'lb_usage_backend_ready':definition('gauge','boolean'),
    'lb_upstream_error_body_reads_total':definition('counter','reads',('result',)),
    'lb_cleanup_active':definition('gauge','owners'),
    'lb_cleanup_oldest_age_seconds':definition('gauge','seconds'),
    'lb_cleanup_finished_total':definition('counter','owners',('result',)),
    'lb_cleanup_duration_seconds':definition('summary','seconds'),
}
CONTRACT = {'schema_version':'lb-metrics-v2','author':'Zeno Ren','families':FAMILIES,
            'identifiers_in_labels':False,'unknown_or_missing':'unknown, never zero',
            'replica_identity':['pod_uid','container_start_time']}
FAMILIES_V3 = {**FAMILIES,
    'lb_request_phase_duration_seconds': definition('histogram', 'seconds', ('api_type', 'phase')),
    'lb_diagnostic_dropped_events_total': definition('counter', 'events', ('reason',)),
    'lb_usage_accepted_events_total': definition('counter', 'events'),
    'lb_usage_persisted_events_total': definition('counter', 'events'),
    'lb_usage_oldest_pending_age_seconds': definition('gauge', 'seconds'),
    'lb_usage_volatile_buffer': definition('gauge', 'boolean')}
CONTRACT_V3 = {**CONTRACT, 'schema_version': 'lb-metrics-v3', 'families': FAMILIES_V3,
               'request_path': '/metrics?schema=lb-metrics-v3', 'legacy_default': 'lb-metrics-v2'}
SAMPLE = re.compile(r'^(lb_[a-zA-Z0-9_:]+)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:[^"\\]|\\[\\"n])*")(?:,|$)')


def parse_exposition(text, *, schema_version='lb-metrics-v2'):
    if schema_version not in ('lb-metrics-v2', 'lb-metrics-v3'):
        raise ValueError('Unsupported metrics contract version')
    families = FAMILIES if schema_version == 'lb-metrics-v2' else FAMILIES_V3
    if len(text.encode())>4*1024*1024:
        raise ValueError('Metrics exposition exceeds 4 MiB')
    result=[];seen=set()
    for line in text.splitlines():
        if line.startswith('# TYPE lb_'):
            _,_,name,kind=line.split()
            if name not in families or families[name]['type']!=kind:
                raise ValueError('Metric type does not match the versioned contract')
        if not line.startswith('lb_'):
            continue
        matched=SAMPLE.fullmatch(line)
        if not matched:
            raise ValueError('Malformed metric sample')
        name,raw_labels,value=matched.groups();family=name;component=None
        if family not in families:
            for suffix in ('_bucket','_count','_sum'):
                if name.endswith(suffix) and name[:-len(suffix)] in families:
                    family=name[:-len(suffix)];component=suffix[1:];break
        if family not in families:
            raise ValueError('Unknown lb metric; update the contract before collecting')
        spec=families[family];labels={};position=0;raw_labels=raw_labels or ''
        while position<len(raw_labels):
            label=LABEL.match(raw_labels,position)
            if label is None:
                raise ValueError('Malformed labels')
            key,encoded=label.groups()
            if key in labels:
                raise ValueError('Duplicate metric label')
            labels[key]=json.loads(encoded)
            if len(labels[key])>128:
                raise ValueError('Unbounded metric label')
            position=label.end()
        expected=set(spec['labels'])|({'le'} if component=='bucket' else set())
        if set(labels)!=expected:
            raise ValueError('Unexpected or missing metric label')
        if spec['type'] in ('histogram','summary') and component not in ('bucket','count','sum'):
            raise ValueError('Aggregate metric requires a supported component')
        if component=='bucket' and spec['type']!='histogram':
            raise ValueError('Only histograms have buckets')
        number=float(value);number=number if math.isfinite(number) else None
        identity=(name,tuple(sorted(labels.items())))
        if identity in seen:
            raise ValueError('Duplicate series')
        seen.add(identity)
        result.append({'name':name,'family':family,'type':spec['type'],'unit':'observations' if component in ('bucket','count') else spec['unit'],
                       'component':component,'labels':labels,'value':number})
    return result


if __name__=='__main__':
    print(json.dumps(CONTRACT,indent=2,sort_keys=True))
