"""Local drain marker and bounded preStop helper. Author: Zeno Ren."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
import fcntl
import re
import tempfile
from pathlib import Path
import time
import urllib.error
import urllib.request

DRAIN_MARKER_FILE = os.getenv('LB_DRAIN_FILE','/tmp/claude-lb-draining')
MAINTENANCE_FILE = os.getenv('LB_MAINTENANCE_FILE','/tmp/claude-lb-maintenance.json')


def maintenance_state(path=None):
    path=Path(MAINTENANCE_FILE if path is None else path)
    try:
        with path.open('rb') as f: raw=f.read(8193)
    except FileNotFoundError:
        return {'version':1,'revision':0,'paused':False,'release_id':None,'epoch':0}
    if len(raw)>8192:
        raise ValueError('Maintenance state too large')
    try:
        state=json.loads(raw)
        valid=(state['version']==1 and type(state['revision']) is int and state['revision']>0
               and type(state['paused']) is bool and type(state['epoch']) is int and state['epoch']>0
               and isinstance(state['release_id'],str) and re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,62}',state['release_id']))
    except (ValueError,KeyError,TypeError):
        valid=False
    if not valid:
        raise ValueError('Invalid maintenance state')
    return state


def maintenance_command(action, release_id, epoch, expected_revision, *, path=None):
    if action not in ('pause','resume') or not isinstance(release_id,str) or not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,62}',release_id):
        raise ValueError('Invalid maintenance action or release ID')
    if type(epoch) is not int or epoch<1 or type(expected_revision) is not int or expected_revision<0:
        raise ValueError('Invalid maintenance epoch/revision')
    path=Path(MAINTENANCE_FILE if path is None else path)
    fd=os.open(str(path)+'.lock',os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(fd,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=maintenance_state(path)
        if state['revision']!=expected_revision:
            raise ValueError('Maintenance state changed; read back before retry')
        if state['release_id']==release_id and epoch<state['epoch']:
            raise ValueError('Stale maintenance epoch')
        if state['paused'] and state['release_id']!=release_id:
            raise ValueError('Maintenance belongs to another release')
        state={'version':1,'revision':state['revision']+1,'paused':action=='pause',
               'release_id':release_id,'epoch':epoch}
        temp_fd,temp_path=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
        try:
            with os.fdopen(temp_fd,'w') as f:
                json.dump(state,f);f.flush();os.fsync(f.fileno())
            os.replace(temp_path,path)
            directory=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            if os.path.exists(temp_path):os.unlink(temp_path)
        return state


def sync_maintenance(controller,path=None):
    try:
        state=maintenance_state(path)
        # Removing a previously observed control file cannot reopen admission.
        previous=getattr(controller,'maintenance_revision',0)
        if state['revision']<previous:
            controller.set_maintenance(True)
        else:
            controller.maintenance_revision=state['revision']
            controller.set_maintenance(state['paused'])
    except (ValueError,OSError):
        controller.set_maintenance(True)
    return controller.maintenance_paused


def request_drain(marker=None):
    path = Path(DRAIN_MARKER_FILE if marker is None else marker)
    try:
        fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        if not path.is_file():
            raise ValueError('Drain marker path exists but is not a file')
        return
    with os.fdopen(fd,'w') as f:
        json.dump({'requested_at':datetime.now(timezone.utc).isoformat()},f)
        f.flush();os.fsync(f.fileno())


def sync_drain(controller, marker=None):
    sync_maintenance(controller)
    path = Path(DRAIN_MARKER_FILE if marker is None else marker)
    if path.is_file() and not controller.permanent_draining:
        controller.drain()
    return controller.draining


def wait_until_drained(read_status, *, timeout=45, interval=.25, clock=time.monotonic, sleep=time.sleep):
    if not math.isfinite(timeout) or timeout < 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Drain wait must be finite and nonnegative')
    deadline = clock()+timeout
    while True:
        try:
            state = read_status()
        except (OSError,ValueError):
            state = None
        if (isinstance(state,dict) and state.get('draining') is True
                and type(state.get('active_requests')) is int and state['active_requests']==0
                and type(state.get('queued_requests')) is int and state['queued_requests']==0):
            return True
        remaining = deadline-clock()
        if remaining <= 0:
            return False
        sleep(min(interval,remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=('drain','pause','resume','status'),default='drain')
    parser.add_argument('--release-id')
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--expected-revision',type=int)
    parser.add_argument('--wait-seconds',type=float,default=45)
    parser.add_argument('--port',type=int,default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if not math.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        parser.error('wait-seconds must be finite and nonnegative')
    if args.action=='status':
        print(json.dumps({'release_control_version':2,**maintenance_state()}));return 0
    if args.action in ('pause','resume'):
        try: state=maintenance_command(args.action,args.release_id,args.epoch,args.expected_revision)
        except (ValueError,OSError) as exc:
            parser.error(str(exc))
        print(json.dumps({'release_control_version':2,**state}));return 0
    request_drain()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def status():
        url = f'http://127.0.0.1:{args.port}/health/accepting'
        try:
            response = opener.open(url,timeout=1)
        except urllib.error.HTTPError as exc:
            response = exc  # 503 is the expected draining response.
        with response:
            body = response.read(65537)
        if len(body)>65536:
            raise ValueError('Invalid local health response size')
        return json.loads(body)
    drained = wait_until_drained(status,timeout=args.wait_seconds)
    print(json.dumps({'draining_requested':True,'drained':drained,'marker':DRAIN_MARKER_FILE}))
    return 0 if drained else 2


if __name__=='__main__':
    raise SystemExit(main())
