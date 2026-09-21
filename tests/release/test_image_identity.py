"""A healthy rollback must be the inspected immutable image. Author: Zeno Ren."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from operations.release.backend import Backend
from operations.release.engine import UnsafeState
from operations.release.model import digest
from operations.release.snapshot import capture

IMAGE = 'registry.invalid/lb@sha256:' + 'a' * 64
OTHER = 'registry.invalid/lb@sha256:' + 'b' * 64


def fixture(image=IMAGE, image_id=IMAGE):
    deployment = {'metadata': {'name': 'lb', 'uid': 'deployment'}, 'spec': {
        'replicas': 1, 'selector': {'matchLabels': {'app': 'lb'}},
        'template': {'metadata': {'labels': {'app': 'lb'}},
                     'spec': {'containers': [{'name': 'lb', 'image': image}]}}}}
    pod = {'metadata': {'name': 'old', 'uid': 'old', 'labels': {'app': 'lb'}},
           'spec': {'containers': [{'name': 'lb', 'image': image}]},
           'status': {'phase': 'Running', 'containerStatuses': [
               {'name': 'lb', 'ready': True, 'imageID': image_id}]}}
    service = {'metadata': {'name': 'lb', 'uid': 'service'}, 'spec': {'selector': {'app': 'lb'}}}
    api = SimpleNamespace(get=lambda *args: copy.deepcopy(deployment),
                          list=lambda kind, *args: copy.deepcopy(
                              [pod] if kind == 'pod' else [service] if kind == 'service' else []))
    return api, deployment, pod


class ImageIdentityTests(unittest.TestCase):
    def test_planning_rejects_mutable_or_unverified_old_images(self):
        for image, image_id in [('registry.invalid/lb:latest', IMAGE), (IMAGE, OTHER), (IMAGE, '')]:
            with self.subTest(image=image, image_id=image_id):
                api, _, _ = fixture(image, image_id)
                with self.assertRaises(ValueError):capture(api, 'lab', 'lb', 'lb')

    def test_planning_accepts_runtime_digest_with_cri_prefix(self):
        api, _, _ = fixture(image_id='docker-pullable://' + IMAGE)
        snapshot = capture(api, 'lab', 'lb', 'lb')
        self.assertEqual(snapshot['deployment']['spec']['template']['spec']['containers'][0]['image'], IMAGE)

    def backend(self, image=IMAGE, image_id=IMAGE):
        _, deployment, pod = fixture(image, image_id)
        b = object.__new__(Backend)
        b.plan = {'container': 'lb', 'lab': False}; b.lab = False
        b.snapshot = {'deployment': deployment, 'old_pods': [pod]}
        b.plan['snapshot_sha256'] = digest(b.snapshot)
        b.check_owner = Mock(); b._guard_spec = Mock(); b.api = Mock()
        b._verify_references = Mock(); b._restore_routes = Mock()
        b._diagnostics = Mock(return_value={'accepting_status': 200, 'accepting': {},
                                            'metrics': {'lb_usage_backend_ready': 1}})
        return b, deployment, pod

    def test_controller_rejects_old_mutable_plan_before_any_effect(self):
        b, _, _ = self.backend('registry.invalid/lb:latest')
        with self.assertRaises(UnsafeState):b.perform('preflight', {})
        self.assertEqual(b.api.mock_calls, [])
        b._verify_references.assert_not_called()

    def test_resumed_old_pod_image_must_still_match(self):
        b, _, pod = self.backend()
        changed = copy.deepcopy(pod); changed['status']['containerStatuses'][0]['imageID'] = OTHER
        b._old_pod = Mock(return_value=changed)
        with self.assertRaises(UnsafeState):b.perform('resume_old', {})
        b._restore_routes.assert_not_called()

    def test_rollback_rejects_healthy_pod_with_wrong_image_before_opening_routes(self):
        b, deployment, pod = self.backend()
        changed = copy.deepcopy(pod); changed['metadata']['uid'] = 'replacement'
        changed['status']['containerStatuses'][0]['imageID'] = OTHER
        b.perform = Mock(); b.deployment_object = Mock(return_value=deployment)
        b._pods = Mock(return_value=[changed])
        with self.assertRaises(UnsafeState):
            b._rollback({'action_receipts': {'writer_termination': {'old': {'verified': True}}}})
        b._restore_routes.assert_not_called()
