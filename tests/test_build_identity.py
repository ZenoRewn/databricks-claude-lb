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
