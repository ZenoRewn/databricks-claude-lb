"""Versioned counter windows and immutable report receipts. Author: Zeno Ren.

This module never invokes a model, edits a scheduler or sends a message. The
watcher adapter supplies canonical snapshots; delivery receipts come from its
existing sender. See docs/SERVICE_RELIABILITY.md for the input contract.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import uuid

POLICY_VERSION='counter-window-v1'


def timestamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:
        raise ValueError('Snapshot/window timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def counter(value):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0


def calculate_window(samples, start, end, metric_units, *, anchor_policy='in_window', max_gap_seconds=4500):
    """One target's ordered snapshots; never interpolate or silently bridge gaps.

    Concurrent replicas must be calculated separately and aggregated explicitly;
    a single sequence must not alternate arbitrary Service-selected replicas.
    """
    left,right=timestamp(start),timestamp(end)
    if left >= right or not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError('Invalid window or maximum sample gap')
    if anchor_policy not in ('in_window','include_previous') or not metric_units:
        raise ValueError('An explicit supported anchor policy and metric units are required')
    ordered=sorted(((timestamp(s['timestamp']),s) for s in samples), key=lambda pair:pair[0])
    if len({t for t,_ in ordered}) != len(ordered):
        raise ValueError('Duplicate timestamp: deduplicate or split target/replica series first')
    selected=[pair for pair in ordered if left <= pair[0] <= right]
    if anchor_policy=='include_previous' and (not selected or selected[0][0]>left):
        earlier=[pair for pair in ordered if pair[0]<left]
        if earlier:selected.insert(0,earlier[-1])
    first=selected[0][0] if selected else None
    last=selected[-1][0] if selected else None
    nominal=(right-left).total_seconds()
    result={'author':'Zeno Ren','policy_version':POLICY_VERSION,'anchor_policy':anchor_policy,
            'max_gap_seconds':max_gap_seconds,'input_sample_count':len(ordered),
            'nominal_start':left.isoformat(),'nominal_end':right.isoformat(),
            'actual_first_sample':first.isoformat() if first else None,'actual_last_sample':last.isoformat() if last else None,
            'sample_count':len(selected),'head_gap_seconds':max(0,(first-left).total_seconds()) if first else nominal,
            'tail_gap_seconds':max(0,(right-last).total_seconds()) if last else nominal,
            'anchor_extension_seconds':max(0,(left-first).total_seconds()) if first else 0,
            'metrics':{}}
    for name,unit in sorted(metric_units.items()):
        total=0; valid=0; covered=0; excluded=[]
        for (a_time,a),(b_time,b) in zip(selected,selected[1:]):
            reason=None
            if a.get('collection_status')!='ok' or b.get('collection_status')!='ok':
                reason='collection_failed_or_unknown'
            elif not all(s.get('pod_uid') and s.get('container_start_time') for s in (a,b)):
                reason='lifecycle_unknown'
            elif (a['pod_uid'],a['container_start_time']) != (b['pod_uid'],b['container_start_time']):
                reason='lifecycle_changed'
            elif (b_time-a_time).total_seconds()>max_gap_seconds:
                reason='sample_gap'
            av=a.get('metrics',{}).get(name);bv=b.get('metrics',{}).get(name)
            if reason is None:
                if not counter(av) or not counter(bv):reason='metric_missing_or_invalid'
                elif bv<av:reason='counter_reset'
            if reason:
                excluded.append({'start':a_time.isoformat(),'end':b_time.isoformat(),'reason':reason})
            else:
                total += bv-av; valid += 1
                covered += max(0,(min(b_time,right)-max(a_time,left)).total_seconds())
        status='unknown' if not valid else 'complete' if first==left and last==right and not excluded else 'partial'
        result['metrics'][name]={'unit':unit,'observed_delta':total if valid else None,'status':status,
                                 'valid_intervals':valid,'covered_seconds':covered,
                                 'coverage_ratio':covered/nominal,'excluded_intervals':excluded}
    return result


def encoded(data):
    return (json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path,data):
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with open(temp,'xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        directory_fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def render_report(summary):
    def cell(value):
        return html.escape(str(value)).replace('|','\\|').replace('\n',' ')
    lines=['# LB 运行窗口报告','','Author: Zeno Ren','',
           f"统计策略：`{summary['policy_version']}`；锚点策略：`{summary['anchor_policy']}`。",'',
           f"名义窗口：{summary['nominal_start']} → {summary['nominal_end']}",
           f"实际样本：{summary['actual_first_sample']} → {summary['actual_last_sample']}",
           f"样本数：{summary['sample_count']}；头部缺口：{summary['head_gap_seconds']} 秒；尾部缺口：{summary['tail_gap_seconds']} 秒。",'',
           '| 指标 | 单位 | 观察增量 | 覆盖状态 | 有效区间 |',
           '|---|---|---:|---|---:|']
    for name,m in summary['metrics'].items():
        lines.append('| '+' | '.join(cell(v) for v in (name,m['unit'],m['observed_delta'] if m['observed_delta'] is not None else 'unknown',m['status'],m['valid_intervals']))+' |')
    lines.extend(['','partial/unknown 不代表完整窗口零错误；端点计数不代表用户任务成功。',
                  f"窗外锚点扩展：{summary['anchor_extension_seconds']} 秒；若非零，增量包含该窗外边界段，未插值。",''])
    return '\n'.join(lines).encode()


def verify_archive(run):
    run=Path(run)
    receipt=json.loads((run/'receipt.json').read_text())
    if set(receipt.get('artifacts',{})) != {'summary.json','report.md'}:
        raise ValueError('Incomplete receipt: both summary and Markdown are required')
    for filename,expected in receipt['artifacts'].items():
        if filename not in ('summary.json','report.md'):
            raise ValueError('Invalid receipt artifact path')
        path=run/filename
        if not path.is_file():
            return {**receipt,'artifact_status':'missing'}
        data=path.read_bytes()
        if len(data)!=expected['bytes'] or digest(data)!=expected['sha256']:
            return {**receipt,'artifact_status':'hash_mismatch'}
    return {**receipt,'artifact_status':'verified'}


def archive_report(directory,run_id,summary):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',run_id):
        raise ValueError('Invalid run ID')
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    run=root/run_id
    content={'summary.json':encoded(summary),'report.md':render_report(summary)}
    manifest={name:{'sha256':digest(data),'bytes':len(data)} for name,data in content.items()}
    try:run.mkdir(mode=0o700)
    except FileExistsError:
        if not (run/'receipt.json').is_file():
            raise FileExistsError('Incomplete archive exists; preserve it and recover under a new run ID')
        previous=verify_archive(run)
        if previous['artifacts'] != manifest or previous['artifact_status']!='verified':
            raise FileExistsError('Existing archive differs or is incomplete; refusing to overwrite')
        return previous
    receipt={'author':'Zeno Ren','run_id':run_id,'execution_status':'succeeded',
             'artifact_status':'pending','delivery_status':'pending','message_id':None,'artifacts':manifest}
    atomic_write(run/'receipt.json',encoded(receipt))
    try:
        for name,data in content.items():atomic_write(run/name,data)
        receipt=verify_archive(run)
        if receipt['artifact_status']!='verified':raise OSError('Artifact read-back verification failed')
    except Exception:
        receipt['artifact_status']='failed'
        atomic_write(run/'receipt.json',encoded(receipt))
        raise
    atomic_write(run/'receipt.json',encoded(receipt))
    return receipt


def record_delivery(run,*,status,message_id):
    """Record an external sender's receipt; does not send or assert the user read it."""
    if status not in ('delivered','failed','unknown') or (status=='delivered' and not message_id):
        raise ValueError('Delivery requires an explicit state and a message ID for delivered')
    run=Path(run)
    with open(run/'.receipt.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=verify_archive(run)
        if receipt['artifact_status']!='verified':
            raise ValueError('Do not deliver or report saved: archive verification failed')
        if receipt['delivery_status']=='delivered' and (status!='delivered' or receipt['message_id']!=message_id):
            raise ValueError('A delivered receipt is immutable; use a separate delivery attempt record')
        receipt.update(delivery_status=status,message_id=message_id)
        atomic_write(run/'receipt.json',encoded(receipt))
        return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path,help='JSON with snapshots and metric_units')
    parser.add_argument('--start',required=True);parser.add_argument('--end',required=True)
    parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--run-id',required=True)
    parser.add_argument('--anchor-policy',choices=('in_window','include_previous'),default='in_window')
    args=parser.parse_args()
    source=args.input.read_bytes()
    data=json.loads(source)
    summary=calculate_window(data['snapshots'],args.start,args.end,data['metric_units'],anchor_policy=args.anchor_policy)
    summary['input_sha256']=digest(source)
    receipt=archive_report(args.output_dir,args.run_id,summary)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
