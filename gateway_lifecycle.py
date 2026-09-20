"""Local drain marker and bounded preStop helper. Author: Zeno Ren."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

DRAIN_MARKER_FILE = os.getenv('LB_DRAIN_FILE','/tmp/claude-lb-draining')


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
    path = Path(DRAIN_MARKER_FILE if marker is None else marker)
    if path.is_file() and not controller.draining:
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
    parser.add_argument('--wait-seconds',type=float,default=45)
    parser.add_argument('--port',type=int,default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if not math.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        parser.error('wait-seconds must be finite and nonnegative')
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
