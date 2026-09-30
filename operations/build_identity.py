"""Prepare safe source attestations and optionally build a local image. Author: Zeno Ren."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from build_metadata import RUNTIME_FILES, manifest_hash, source_hashes


def snapshot(root, *, files=RUNTIME_FILES):
    root = Path(root).resolve()
    hashes = source_hashes(root, files)
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=False)
    head = git('rev-parse', 'HEAD')
    revision = head.stdout.decode().strip() if head.returncode == 0 else None
    dirty = False
    out_of_tree = revision is None
    for name, digest in hashes.items():
        committed = git('show', f'HEAD:{name}') if revision else None
        if committed is None or committed.returncode != 0:
            dirty = out_of_tree = True
        else:
            import hashlib
            changed = hashlib.sha256(committed.stdout).hexdigest() != digest
            dirty = dirty or changed
            out_of_tree = out_of_tree or changed
    # A changed build recipe is part of source provenance even though it is not
    # installed as an application file inside the runtime image.
    recipe = git('diff', '--quiet', 'HEAD', '--', 'Dockerfile', '.dockerignore') if revision else None
    if recipe is not None and recipe.returncode != 0:
        dirty = True
    return {'schema_version': 1, 'author': 'Zeno Ren', 'source_revision': revision,
            'source_tree_dirty': dirty if revision else None, 'out_of_tree': out_of_tree,
            'source_manifest_sha256': manifest_hash(hashes), 'files': hashes}


def build(root, identity, tag, *, platform=None, dependency_stage='dependencies-online'):
    args = ['docker', 'build', '-t', tag]
    if platform:
        args.extend(['--platform', platform])
    values = {'SOURCE_REVISION': identity['source_revision'] or 'unknown',
              'SOURCE_TREE_DIRTY': 'unknown' if identity['source_tree_dirty'] is None else str(identity['source_tree_dirty']).lower(),
              'SOURCE_OUT_OF_TREE': str(identity['out_of_tree']).lower(),
              'SOURCE_MANIFEST_SHA256': identity['source_manifest_sha256'], 'DEPENDENCY_STAGE': dependency_stage}
    for key, value in values.items():
        args.extend(['--build-arg', key + '=' + value])
    subprocess.run([*args, '.'], cwd=root, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output')
    parser.add_argument('--build-tag')
    parser.add_argument('--platform', choices=('linux/amd64', 'linux/arm64'))
    parser.add_argument('--dependency-stage', choices=('dependencies-online', 'dependencies-offline'), default='dependencies-online')
    args = parser.parse_args()
    result = snapshot(args.root)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as output:
            json.dump(result, output, indent=2); output.write('\n')
    if args.build_tag:
        build(args.root, result, args.build_tag, platform=args.platform, dependency_stage=args.dependency_stage)
    print(json.dumps(result, sort_keys=True))
