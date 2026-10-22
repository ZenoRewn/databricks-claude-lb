"""Offline reproduction of the abandoned maintenance gate. Author: Zeno Ren.

Every subprocess call is mocked. Temporary paths replace the original paths.
This validates historical failure modes, not a replacement release controller.
"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch


def reproduce(*, legacy_null=False, journal_failure=False):
    source = Path(__file__).with_name('gate-design.fixture.txt').read_text()
    state = {'gated': False, 'restore_calls': 0, 'external_commands_executed': 0}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / 'service.json'
        baseline.write_text(json.dumps({'spec': {'selector': {'app': 'claude-lb'}}}))
        source = source.replace(
            'docs/reviews/2026-09-20-service-optimization/deployment-r1', directory
        ).replace('/tmp/lb-release-20260920/service-baseline.json', str(baseline))
        if legacy_null:
            source = source.replace("(s.get('endpoints') or [])", "s.get('endpoints',[])")

        def fake_command(command, **kwargs):
            args = command[5:]
            if args[:3] == ['patch', 'service', 'claude-lb']:
                operations = json.loads(args[args.index('-p') + 1])
                if operations[-1]['op'] == 'add':
                    state['gated'] = True
                else:
                    state['gated'] = False
                    state['restore_calls'] += 1
                return 'service patched (mock)'
            if args[:2] == ['get', 'endpointslice']:
                return json.dumps({'items': [{'endpoints': None}]})
            if args[:1] == ['exec']:
                if journal_failure:
                    raise subprocess.TimeoutExpired('mock metrics read', 1)
                return json.dumps({f'endpoint_{n}': 0 for n in range(14)})
            raise RuntimeError('Unexpected mocked command')

        original_write = Path.write_text

        def write_text(path, *args, **kwargs):
            if journal_failure and path.name == 'maintenance-drain.json':
                raise OSError('mock journal unavailable')
            return original_write(path, *args, **kwargs)

        error_type = None
        with patch('subprocess.check_output', side_effect=fake_command), \
             patch('subprocess.run', side_effect=RuntimeError('External commands forbidden')), \
             patch('time.sleep'), patch.object(Path, 'write_text', write_text), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(source, 'historical-gate-design', 'exec'), {'__name__': '__main__'})
            except Exception as exc:
                error_type = type(exc).__name__
        return {**state, 'error_type': error_type}


def main():
    cases = {
        'legacy_null_endpoints': reproduce(legacy_null=True),
        'successful_drain_without_next_owner': reproduce(),
        'observation_error_and_journal_write_failure': reproduce(journal_failure=True),
    }
    expected = [
        cases['legacy_null_endpoints']['error_type'] == 'TypeError',
        cases['legacy_null_endpoints']['restore_calls'] == 1,
        cases['successful_drain_without_next_owner']['error_type'] is None,
        cases['successful_drain_without_next_owner']['gated'] is True,
        cases['successful_drain_without_next_owner']['restore_calls'] == 0,
        cases['observation_error_and_journal_write_failure']['gated'] is True,
        cases['observation_error_and_journal_write_failure']['restore_calls'] == 0,
    ]
    if not all(expected):
        raise RuntimeError('Historical behavior did not match the reproduction')
    print(json.dumps({'author': 'Zeno Ren', 'offline_only': True,
                      'historical_failure_modes_reproduced': True, 'cases': cases}, indent=2))


if __name__ == '__main__':
    main()
