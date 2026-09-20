"""Exercise lease races against a disposable API server. Author: Zeno Ren."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import uuid

from kind_lab import lab_api, OPS
from operations.release.engine import OwnershipLost, PendingOperation
from operations.release.kube import Lease


def run(context, kubeconfig, output):
    api = lab_api(context, kubeconfig)
    receipts = []
    for _ in range(10):
        name = 'lb-test-race-' + uuid.uuid4().hex[:12]
        barrier = threading.Barrier(2)
        clients = [lab_api(context, kubeconfig) for _ in range(2)]
        leases = [Lease(client, OPS, name, str(i)) for i, client in enumerate(clients)]

        def acquire(lease):
            barrier.wait(timeout=10)
            try:
                return lease.acquire()
            except PendingOperation:
                return False

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                acquired = list(pool.map(acquire, leases))
            if acquired.count(True) != 1:
                raise AssertionError('Exactly one contender must hold the lease')
            winner = leases[acquired.index(True)]
            winner.check()
            successor = Lease(api, OPS, name, winner.identity)
            if not successor.acquire():
                raise AssertionError('Expected a new execution generation')
            try:
                winner.check()
            except OwnershipLost:
                fenced = True
            else:
                raise AssertionError('Old generation retained mutation authority')
            receipts.append({'single_winner': True, 'previous_epoch_fenced': fenced,
                             'previous_epoch': winner.epoch, 'new_epoch': successor.epoch})
        finally:
            obj = api.optional('lease', OPS, name)
            if obj:
                api.delete('lease', OPS, obj)
    destination = Path(output)
    with destination.open('x') as stream:
        json.dump({'author': 'Zeno Ren', 'scope': 'Disposable Kind API, no production',
                   'rounds': receipts, 'verified': True}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'verified': True, 'races': len(receipts)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('context', 'kubeconfig', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.context, args.kubeconfig, args.output)
