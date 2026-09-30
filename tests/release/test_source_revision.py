"""Source annotations change with the image, including rollback. Author: Zeno Ren."""
import copy
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from operations.release.backend import Backend
from operations.release.model import APP_FILES, CONTROL, EPOCH, SCHEMA, default_target_settings, digest, target_pod_spec

SOURCE = 'lb.zeno.ink/source-revision'


def apply_patch(current, operations):
    result = copy.deepcopy(current)
    for operation in operations:
        parts = [part.replace('~1', '/').replace('~0', '~') for part in operation['path'].strip('/').split('/')]
        parent = result
        for part in parts[:-1]: parent = parent[part]
        if operation['op'] == 'test':
            if parent[parts[-1]] != operation['value']: raise ValueError('Conditional write rejected')
        else:
            parent[parts[-1]] = copy.deepcopy(operation['value'])
    return result


def rewrite(value, field):
    if field == 'source': value['spec']['template']['metadata']['annotations'][SOURCE] = 'e' * 40
    elif field == 'image': value['spec']['template']['spec']['containers'][0]['image'] = 'rewritten'
    else: value['spec']['template']['spec']['containers'][0]['livenessProbe']['timeoutSeconds'] = 99
    return value


class SourceRevisionTests(unittest.TestCase):
    def make(self, *, previous='b' * 40):
        metadata = {'labels': {'app': 'synthetic'}, 'annotations': {'prometheus.io/scrape': 'true'}}
        if previous is not None:
            metadata['annotations'][SOURCE] = previous
        old = {'metadata': {'name': 'synthetic', 'uid': 'synthetic-uid', 'resourceVersion': '1'}, 'spec': {'replicas': 1, 'template': {'metadata': metadata,
            'spec': {'containers': [{'name': 'app', 'image': 'old@sha256:' + 'b' * 64}],
                     'terminationGracePeriodSeconds': 90}}}}
        current = copy.deepcopy(old)
        plan = {'image': 'new@sha256:' + 'a' * 64, 'source_revision': 'a' * 40,
                'target_settings': default_target_settings()}
        backend = object.__new__(Backend)
        backend.snapshot = {'deployment': old}
        backend.plan = plan
        backend.deployment = 'synthetic'
        backend._owned = Mock(return_value=current)
        backend._patch = Mock(side_effect=lambda kind, obj, operations, **kwargs: apply_patch(obj, operations))
        return backend, current

    def annotations(self, backend):
        operations = backend._patch.call_args.args[2]
        matches = [op for op in operations if op['path'] == '/spec/template/metadata/annotations']
        self.assertEqual(len(matches), 1, 'Image and source annotation must share the same conditional patch')
        return matches[0]['value']

    def test_forward_sets_source_and_preserves_unrelated_metadata(self):
        backend, current = self.make()
        Backend._set_image(backend, backend.plan['image'])
        self.assertEqual(self.annotations(backend), {'prometheus.io/scrape': 'true', SOURCE: 'a' * 40})
        operations = backend._patch.call_args.args[2]
        pod = next(op['value'] for op in operations if op['path'] == '/spec/template/spec')
        self.assertEqual(pod['containers'][0]['image'], backend.plan['image'])
        self.assertEqual(current['spec']['template']['metadata']['annotations'][SOURCE], 'b' * 40)

    def test_rollback_restores_exact_previous_source_or_absence(self):
        for previous in ('b' * 40, None):
            with self.subTest(previous=previous):
                backend, current = self.make(previous=previous)
                current['spec']['template']['metadata']['annotations'][SOURCE] = 'a' * 40
                current['spec']['template']['spec']['containers'][0]['image'] = backend.plan['image']
                Backend._set_image(backend, 'old@sha256:' + 'b' * 64, rollback=True)
                expected = {'prometheus.io/scrape': 'true'}
                if previous is not None:
                    expected[SOURCE] = previous
                self.assertEqual(self.annotations(backend), expected)

    def test_ack_readback_is_noop_only_when_image_and_source_both_match(self):
        backend, current = self.make()
        current['spec']['template']['spec'] = target_pod_spec(current, backend.plan)
        Backend._set_image(backend, backend.plan['image'])
        self.assertTrue(backend._patch.called, 'A stale source annotation is not a completed update')
        current['spec']['template']['metadata']['annotations'] = self.annotations(backend)
        backend._patch.reset_mock()
        Backend._set_image(backend, backend.plan['image'])
        backend._patch.assert_not_called()

    def test_forward_handles_missing_annotation_map(self):
        backend, current = self.make(previous=None)
        current['spec']['template']['metadata'].pop('annotations')
        Backend._set_image(backend, backend.plan['image'])
        self.assertEqual(self.annotations(backend), {SOURCE: 'a' * 40})

    def test_normal_ack_cannot_silently_rewrite_the_approved_template(self):
        for field in ('source', 'image', 'probe'):
            with self.subTest(field=field):
                backend, current = self.make()
                backend._patch.side_effect = lambda kind, obj, operations, **kwargs: rewrite(apply_patch(obj, operations), field)
                with self.assertRaises(ValueError):
                    Backend._set_image(backend, backend.plan['image'])

    def dry_run_fixture(self, field):
        backend, current = self.make()
        snapshot = {'deployment': current, 'container': 'app',
                    'services': {'lb': {'metadata': {'uid': 'synthetic-service'}, 'spec': {'selector': {'app': 'synthetic'}}}}}
        profile = {**backend.plan, 'lab': True, 'schema_version': SCHEMA, 'release_id': 'synthetic',
                   'namespace': 'lb-lab', 'deployment': 'synthetic', 'container': 'app', 'services': ['lb'],
                   'source_hashes': {name: 'c' * 64 for name in APP_FILES}, 'snapshot_sha256': digest(snapshot),
                   'direct_pod_access': False, 'legacy_bootstrap': False, 'storage_compatibility': 'additive-compatible',
                   'public_urls': ['https://fixture.invalid'],
                   'business_probes': [{'api': 'messages', 'model': 'synthetic', 'stream': False, 'max_tokens': 32}]}
        api = SimpleNamespace(request=Mock(side_effect=lambda *args, **kwargs: rewrite(apply_patch(current, args[4]), field)),
                              list=Mock(side_effect=RuntimeError('Preflight continued after rewritten dry-run')))
        return backend, snapshot, profile, api

    def test_plan_rejects_server_rewriting_source_image_or_probes(self):
        cli = importlib.import_module('operations.release.__main__')
        for field in ('source', 'image', 'probe'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                _, snapshot, profile, api = self.dry_run_fixture(field)
                output = Path(directory) / 'plan'
                with patch.object(cli, 'capture', return_value=snapshot), self.assertRaises(ValueError):
                    cli.plan_release(api, profile, output, '.')
                self.assertFalse(output.exists(), 'A rejected target cannot be recorded as reviewed')

    def test_cluster_preflight_rejects_server_rewriting_source_image_or_probes(self):
        for field in ('source', 'image', 'probe'):
            with self.subTest(field=field):
                backend, snapshot, profile, api = self.dry_run_fixture(field)
                backend.plan = profile; backend.snapshot = snapshot; backend.api = api
                backend.lab = True; backend.namespace = 'lb-lab'
                backend.check_owner = Mock(); backend._guard_spec = Mock()
                backend._verified_old_image = Mock(); backend._verify_references = Mock()
                backend._verify_public_targets = Mock()
                backend.deployment_object = Mock(return_value=snapshot['deployment'])
                with self.assertRaises(ValueError):
                    Backend.perform(backend, 'preflight', {})
                api.list.assert_not_called()

    def test_per_write_dry_run_mismatch_prevents_actual_mutation(self):
        for field in ('source', 'image', 'probe'):
            with self.subTest(field=field):
                backend, current = self.make()
                backend.check_owner = Mock(); backend.release_id = 'synthetic'; backend.epoch = 1
                backend.namespace = 'lb-lab'
                current['metadata']['annotations'] = {CONTROL: 'synthetic', EPOCH: '1'}
                backend.api = SimpleNamespace(
                    request=Mock(side_effect=lambda *args, **kwargs: rewrite(apply_patch(current, args[4]), field)),
                    patch=Mock(side_effect=lambda kind, namespace, obj, operations: apply_patch(obj, operations)))
                backend._patch = Backend._patch.__get__(backend, Backend)
                with self.assertRaises(ValueError):
                    Backend._set_image(backend, backend.plan['image'])
                backend.api.patch.assert_not_called()
