"""Source provenance is explicit and runtime hashes are verified. Author: Zeno Ren."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class BuildIdentityTests(unittest.TestCase):
    def test_clean_dirty_and_untracked_runtime_sources_are_distinguished(self):
        from operations.build_identity import snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'main.py').write_text('synthetic = 1\n')
            for args in (['init', '-q'], ['add', 'main.py'], ['-c', 'user.name=Synthetic', '-c',
                'user.email=synthetic@example.invalid', 'commit', '-qm', 'synthetic']):
                subprocess.run(['git', '-C', directory, *args], check=True, capture_output=True)
            clean = snapshot(root, files=['main.py'])
            self.assertFalse(clean['source_tree_dirty'])
            self.assertFalse(clean['out_of_tree'])
            self.assertEqual(len(clean['source_revision']), 40)
            (root / 'main.py').write_text('synthetic = 2\n')
            self.assertTrue(snapshot(root, files=['main.py'])['source_tree_dirty'])
            (root / 'new.py').write_text('synthetic = 3\n')
            self.assertTrue(snapshot(root, files=['new.py'])['out_of_tree'])

    def test_build_manifest_mismatch_fails_and_unknown_is_not_attested(self):
        from build_metadata import build_info
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'main.py').write_text('synthetic = 1\n')
            with self.assertRaises(ValueError):
                build_info(root, files=['main.py'], environ={'SOURCE_MANIFEST_SHA256': '0' * 64})
            unknown = build_info(root, files=['main.py'], environ={})
            self.assertIsNone(unknown['source_revision'])
            self.assertIsNone(unknown['source_tree_dirty'])
            self.assertFalse(unknown['source_attested'])
            expected = hashlib.sha256(json.dumps(unknown['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            verified = build_info(root, files=['main.py'], environ={'SOURCE_REVISION': 'a' * 40,
                'SOURCE_TREE_DIRTY': 'false', 'SOURCE_OUT_OF_TREE': 'false', 'SOURCE_MANIFEST_SHA256': expected})
            self.assertTrue(verified['source_attested'])

    def test_runtime_tampering_is_detected_and_private_files_are_not_included(self):
        from build_metadata import build_info, runtime_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'main.py').write_text('synthetic = 1\n')
            (root / 'config.yaml').write_text('SYNTHETIC_SECRET')
            value = build_info(root, files=['main.py'], environ={})
            (root / 'build-info.json').write_text(json.dumps(value))
            self.assertTrue(runtime_identity(root)['runtime_files_match'])
            (root / 'main.py').write_text('synthetic = 2\n')
            self.assertFalse(runtime_identity(root)['runtime_files_match'])
            self.assertNotIn('SYNTHETIC_SECRET', json.dumps(runtime_identity(root)))

    def test_packaging_manifest_covers_every_runtime_module(self):
        from build_metadata import RUNTIME_FILES
        from operations.release.model import APP_FILES
        self.assertEqual(set(RUNTIME_FILES), set(APP_FILES) | {'requirements.lock', 'requirements.txt'})

    # The check above only compares the two manifests to each other, so adding a
    # module to neither keeps it green while the image would fail on import.
    # These two close that gap from both ends: what the code needs, and what the
    # image actually ships.

    def test_every_module_main_imports_is_in_the_manifest(self):
        import ast

        from build_metadata import RUNTIME_FILES
        root = Path(__file__).resolve().parent.parent
        tree = ast.parse((root / 'main.py').read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split('.')[0])
        local = {f'{name}.py' for name in imported if (root / f'{name}.py').is_file()}
        missing = sorted(local - set(RUNTIME_FILES))
        self.assertEqual(missing, [], 'main.py imports these but the runtime manifest omits them')

    def test_the_test_lock_covers_every_runtime_dependency(self):
        # requirements-test.in does `-r requirements.in`, but the lock next to it
        # is flat and hand-maintained, so adding a runtime dependency without
        # adding it here passes locally and fails only in CI on import.
        import re
        root = Path(__file__).resolve().parent.parent
        pinned = lambda name: set(re.findall(r'^([A-Za-z0-9_.-]+)==',
                                             (root / name).read_text(), re.M))
        missing = sorted(pinned('requirements.lock') - pinned('requirements-test.lock'))
        self.assertEqual(missing, [], 'runtime dependencies absent from the test lock')

    def test_dockerfile_ships_every_runtime_module(self):
        from build_metadata import RUNTIME_FILES
        root = Path(__file__).resolve().parent.parent
        dockerfile = (root / 'Dockerfile').read_text()
        # requirements are copied by their own earlier COPY layer.
        expected = set(RUNTIME_FILES) - {'requirements.lock', 'requirements.txt'}
        copied = set()
        for line in dockerfile.splitlines():
            if line.startswith('COPY '):
                copied.update(line.split()[1:-1])
        missing = sorted(expected - copied)
        self.assertEqual(missing, [], 'runtime manifest lists these but the image never copies them')
