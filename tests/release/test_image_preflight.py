"""A passed image check must stay passed. Author: Zeno Ren.

`_image_preflight` used to re-derive its verdict from `pod.status.phase` on
every one-second tick, and the validation pod carries
`activeDeadlineSeconds=90`. Observed on AKS 2026-10-16 with neither image
cached on the node:

    t+66  the `-new` check passes (pull + 23 file hashes), `-old` is created
    t+81  the `-old` container exits 0 and prints OK — the check **passed**
    t+156 the 90s deadline elapses; the phase had not settled on `Succeeded`
          (the sandbox lived until t+169), so kubelet marks it DeadlineExceeded
    t+171 the next tick reads `Failed` and raises image_preflight_failed

The blast radius was bounded — it happens before the maintenance window, so
production never moved — but a release was refused for a check that succeeded.
Enlarging the deadline only lengthens the same fuse: a slower first pull burns
it again.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from operations.release.backend import Backend
from operations.release.engine import PendingOperation, UnsafeState


CONTROL = 'lb.zeno.ink/release'


def _backend(pod):
    b = object.__new__(Backend)
    b.namespace = 'lab'
    b.release_id = 'rel'
    b.epoch = 1
    b.plan = {'container': 'lb', 'image': 'img@sha256:' + 'a' * 64}
    b.check_owner = Mock()
    b.api = SimpleNamespace(optional=Mock(return_value=pod), create=Mock())
    return b


def _pod(phase, *, exit_code=None, owner='rel'):
    status = {'phase': phase}
    if exit_code is not None:
        status['containerStatuses'] = [{'state': {'terminated': {'exitCode': exit_code}}}]
    return {'metadata': {'name': 'lb-check-rel-old', 'annotations': {CONTROL: owner}},
            'status': status}


class ImagePreflightTests(unittest.TestCase):
    def test_a_clean_exit_counts_before_the_phase_settles(self):
        # The exact moment the old code lost: exited 0, phase still Running.
        b = _backend(_pod('Running', exit_code=0))
        receipts = {}
        b._image_preflight('img', 'old', None, receipts)
        self.assertEqual(receipts['old'], 'verified')

    def test_a_deadline_rewrite_after_a_recorded_success_is_ignored(self):
        receipts = {}
        b = _backend(_pod('Running', exit_code=0))
        b._image_preflight('img', 'old', None, receipts)
        # kubelet later rewrites the phase to Failed (DeadlineExceeded). The
        # verdict is already in the receipt, so this must not raise.
        b.api.optional = Mock(return_value=_pod('Failed'))
        b._image_preflight('img', 'old', None, receipts)
        self.assertEqual(receipts['old'], 'verified')

    def test_a_recorded_success_does_not_even_read_the_pod_again(self):
        b = _backend(_pod('Succeeded'))
        b._image_preflight('img', 'old', None, {'old': 'verified'})
        b.api.optional.assert_not_called()

    def test_a_genuine_failure_still_fails(self):
        b = _backend(_pod('Failed'))
        with self.assertRaisesRegex(RuntimeError, 'image_preflight_failed'):
            b._image_preflight('img', 'old', None, {})

    def test_a_nonzero_exit_is_not_success(self):
        b = _backend(_pod('Failed', exit_code=1))
        with self.assertRaisesRegex(RuntimeError, 'image_preflight_failed'):
            b._image_preflight('img', 'old', None, {})

    def test_a_still_running_container_stays_pending(self):
        b = _backend(_pod('Running'))
        with self.assertRaises(PendingOperation):
            b._image_preflight('img', 'old', None, {})

    def test_the_phase_alone_is_still_accepted(self):
        # Fast path on a warm node: phase settles before we look.
        b = _backend(_pod('Succeeded'))
        receipts = {}
        b._image_preflight('img', 'old', None, receipts)
        self.assertEqual(receipts['old'], 'verified')

    def test_a_foreign_pod_is_refused_rather_than_trusted(self):
        b = _backend(_pod('Succeeded', owner='someone-else'))
        with self.assertRaises(UnsafeState):
            b._image_preflight('img', 'old', None, {})

    def test_the_two_images_record_separate_verdicts(self):
        # `-new` passing must not short-circuit `-old`.
        receipts = {}
        b = _backend(_pod('Running', exit_code=0))
        b._image_preflight('img', 'new', {'main.py': 'x'}, receipts)
        self.assertEqual(receipts, {'new': 'verified'})
        b.api.optional = Mock(return_value=_pod('Running'))
        with self.assertRaises(PendingOperation):
            b._image_preflight('img', 'old', None, receipts)

    def test_a_missing_pod_is_created_once_and_reports_pending(self):
        b = _backend(None)
        b.snapshot = {'deployment': {'spec': {'template': {'spec': {
            'containers': [{'name': 'lb', 'image': 'old', 'volumeMounts': []}], 'volumes': []}}}}}
        with self.assertRaises(PendingOperation):
            b._image_preflight('img', 'old', None, {})
        b.api.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()
