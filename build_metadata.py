"""Build provenance and bounded runtime identity reads. Author: Zeno Ren."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

RUNTIME_FILES = ('main.py', 'build_metadata.py', 'chat_adapter.py', 'model_capabilities.py', 'effort_compat.py',
                 'request_telemetry.py', 'safe_diagnostics.py', 'request_timing.py', 'request_budget.py',
                 'admission.py', 'gateway_lifecycle.py', 'upstream_body.py', 'usage_store.py', 'otel_setup.py',
                 'dashboard.html', 'response_semantics.py', 'cleanup_observability.py', 'copilot_pricing.py',
                 'release_probe.py', 'requirements.lock', 'requirements.txt')


def source_hashes(root, files=RUNTIME_FILES):
    if not files or any(not isinstance(name, str) or '/' in name or '\\' in name or name in ('.', '..') for name in files):
        raise ValueError('Source manifest requires flat runtime filenames')
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in files}


def manifest_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tri_state(value):
    if value in (None, 'unknown'):
        return None
    if value not in ('true', 'false'):
        raise ValueError('Build provenance booleans must be true, false or unknown')
    return value == 'true'


def build_info(root, *, files=RUNTIME_FILES, environ=None):
    env = os.environ if environ is None else environ
    hashes = source_hashes(root, files)
    actual = manifest_hash(hashes)
    expected = env.get('SOURCE_MANIFEST_SHA256', 'unknown')
    if expected != 'unknown' and expected != actual:
        raise ValueError('Build context runtime files differ from the supplied source manifest')
    revision = env.get('SOURCE_REVISION', 'working-tree')
    if revision not in ('working-tree', 'unknown') and not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('SOURCE_REVISION must be a Git commit or explicitly unknown')
    return {'schema_version': 2, 'author': 'Zeno Ren',
            'source_revision': revision if re.fullmatch(r'[a-f0-9]{40}', revision) else None,
            'source_tree_dirty': tri_state(env.get('SOURCE_TREE_DIRTY')),
            'out_of_tree': tri_state(env.get('SOURCE_OUT_OF_TREE')),
            'source_attested': expected == actual and bool(re.fullmatch(r'[a-f0-9]{40}', revision)),
            'source_manifest_sha256': actual, 'files': hashes,
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}


def runtime_identity(root=None):
    root = Path(__file__).parent if root is None else Path(root)
    try:
        with (root / 'build-info.json').open('rb') as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError('Oversized build metadata')
        value = json.loads(raw)
        files = value.get('files')
        if (not isinstance(files, dict) or not files or len(files) > len(RUNTIME_FILES)
                or set(files) - set(RUNTIME_FILES) or any(not isinstance(v, str) or not re.fullmatch(r'[a-f0-9]{64}', v) for v in files.values())):
            raise ValueError('Invalid runtime manifest')
        matches = source_hashes(root, files) == files
        return {'available': True, 'source_revision': value.get('source_revision'),
                'source_tree_dirty': value.get('source_tree_dirty'), 'out_of_tree': value.get('out_of_tree'),
                'source_attested': value.get('source_attested') is True,
                'source_manifest_sha256': manifest_hash(files), 'runtime_files_match': matches,
                'manifest_covers_current_runtime': set(files) == set(RUNTIME_FILES)}
    except (OSError, ValueError, TypeError, AttributeError):
        return {'available': False, 'source_revision': None, 'source_attested': False,
                'runtime_files_match': None, 'manifest_covers_current_runtime': False}


if __name__ == '__main__':
    root = Path(__file__).parent
    value = build_info(root)
    value['os_packages'] = subprocess.check_output(['dpkg-query', '-W'], text=True)
    (root / 'build-info.json').write_text(json.dumps(value, sort_keys=True) + '\n')
