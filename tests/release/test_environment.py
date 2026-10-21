"""Controlled identity settings travel atomically with the image. Author: Zeno Ren."""
import copy
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from operations.release.backend import Backend
from operations.release.engine import Engine, OwnershipLost, PendingOperation, UncertainOperation, UnsafeState, new_record
from operations.release.kube import KubeAPI
from operations.release.model import APP_FILES, CONTROL, EPOCH, SOURCE, SPEC, default_target_settings, digest, target_pod_spec, validate_plan
from test_engine import FakeBackend, MemoryStore


MODE = 'COPILOT_RESPONSES_ITEM_ID_MODE'
MODELS = 'COPILOT_RESPONSES_ITEM_ID_MODELS'
OVERRIDES = {MODE: 'stabilize', MODELS: 'gpt-6-astra,gpt-6-sol'}
OLD = 'registry.invalid/lb@sha256:' + 'b' * 64
NEW = 'registry.invalid/lb@sha256:' + 'a' * 64
ABSENT = object()


def plan(overrides=OVERRIDES):
    value = {'schema_version': 'lb-release-plan-v2', 'release_id': 'identity', 'namespace': 'lb-lab',
             'deployment': 'lb', 'container': 'app', 'services': ['lb'], 'source_revision': 'a' * 40,
             'image': NEW, 'source_hashes': {name: 'c' * 64 for name in APP_FILES},
             'snapshot_sha256': 'd' * 64, 'target_settings': default_target_settings(),
             'direct_pod_access': False, 'legacy_bootstrap': False, 'lab': True,
             'storage_compatibility': 'additive-compatible', 'public_urls': ['https://fixture.invalid'],
             'business_probes': [{'api': 'responses', 'model': 'gpt-6-astra', 'stream': True, 'max_tokens': 32}]}
    if overrides is not ABSENT:
        value['target_env'] = copy.deepcopy(overrides)
    return value


def original(env=ABSENT):
    container = {'name': 'app', 'image': OLD, 'envFrom': [{'secretRef': {'name': 'private'}}],
                 'resources': {'requests': {'cpu': '100m'}}}
    if env is not ABSENT:
        container['env'] = copy.deepcopy(env)
    return {'metadata': {'name': 'lb', 'uid': 'deployment', 'resourceVersion': '1'},
            'spec': {'replicas': 1, 'selector': {'matchLabels': {'app': 'lb'}},
                     'template': {'metadata': {'labels': {'app': 'lb'}, 'annotations': {SOURCE: 'b' * 40}},
                                  'spec': {'containers': [container], 'terminationGracePeriodSeconds': 30}}}}


def container(deployment):
    return deployment['spec']['template']['spec']['containers'][0]


def apply_json_patch(current, operations):
    result = copy.deepcopy(current)
    for operation in operations:
        parts = [part.replace('~1', '/').replace('~0', '~') for part in operation['path'].strip('/').split('/')]
        parent = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        if operation['op'] == 'test':
            if parent[key] != operation['value']:
                raise PendingOperation('conditional_write_requires_fresh_readback')
        elif operation['op'] == 'remove':
            del parent[key]
        else:
            parent[key] = copy.deepcopy(operation['value'])
    return result


class MemoryAPI:
    patch = KubeAPI.patch

    def __init__(self, current):
        self.current = copy.deepcopy(current)
        self.requests = []
        self.mutations = 0
        self.lose_ack = False
        self.admission = None

    def get(self, kind, namespace, name):
        return copy.deepcopy(self.current)

    def request(self, method, kind, namespace, name, body, query=None, patch=False):
        self.requests.append((copy.deepcopy(body), query))
        updated = apply_json_patch(self.current, body)
        if self.admission:
            self.admission(updated, bool(query))
        if query:
            return updated
        updated['metadata']['resourceVersion'] = str(int(self.current['metadata']['resourceVersion']) + 1)
        self.current = updated
        self.mutations += 1
        if self.lose_ack:
            self.lose_ack = False
            raise UncertainOperation('ACK lost after atomic image and environment update')
        return copy.deepcopy(updated)


def backend(env=ABSENT):
    saved = original(env)
    current = copy.deepcopy(saved)
    current['metadata']['annotations'] = {CONTROL: 'identity', EPOCH: '1', SPEC: digest(current['spec'])}
    value = object.__new__(Backend)
    value.plan = plan()
    value.snapshot = {'deployment': saved, 'container': 'app', 'services': {
        'lb': {'metadata': {'name': 'lb', 'uid': 'service'}, 'spec': {'selector': {'app': 'lb'}}}}}
    value.namespace = 'lb-lab'; value.deployment = 'lb'; value.release_id = 'identity'; value.epoch = 1
    value.api = MemoryAPI(current)
    value.check_owner = Mock()
    return value


@pytest.mark.parametrize('mode', ['off', 'observe', 'stabilize'])
def test_v2_accepts_fixed_modes_and_exact_model_csv(mode):
    assert validate_plan(plan({MODE: mode, MODELS: 'gpt-6-astra,gpt-6-sol'}))['target_env'][MODE] == mode


def test_plain_v1_is_unchanged_and_old_controllers_reject_env_plan_schema():
    legacy = plan(ABSENT); legacy['schema_version'] = 'lb-release-plan-v1'
    assert validate_plan(legacy)['schema_version'] == 'lb-release-plan-v1'
    assert plan()['schema_version'] != 'lb-release-plan-v1'  # Old validator accepts only this literal.
    legacy['target_env'] = OVERRIDES
    with pytest.raises(ValueError):
        validate_plan(legacy)
    assert validate_plan(plan({MODE: 'off', MODELS: ''}))['target_env'][MODELS] == ''


@pytest.mark.parametrize('overrides', [
    None, [], {MODE: True}, {MODE: 'STABILIZE'}, {MODE: ' stabilize'},
    {'PRIVATE': 'replacement'}, {MODE: {'valueFrom': {}}},
    {MODELS: '*'}, {MODELS: 'a,a'}, {MODELS: 'a, b'}, {MODELS: 'a,'},
    {MODELS: ',a'}, {MODELS: 'a\nb'}, {MODELS: 'a,,b'}, {MODELS: 'a' * 129},
])
def test_rejects_arbitrary_environment_or_ambiguous_model_values(overrides):
    with pytest.raises(ValueError):
        validate_plan(plan(overrides))


def test_forward_replaces_only_selected_keys_and_preserves_everything_else():
    old_env = [{'name': 'PRIVATE', 'valueFrom': {'secretKeyRef': {'name': 's', 'key': 'k'}}},
               {'name': MODE, 'valueFrom': {'configMapKeyRef': {'name': 'c', 'key': 'mode'}}},
               {'name': 'TAIL', 'value': 'unchanged'}]
    saved = original(old_env); before = copy.deepcopy(saved)
    result = target_pod_spec(saved, plan())['containers'][0]
    assert result['env'] == [old_env[0], {'name': MODE, 'value': 'stabilize'}, old_env[2],
                             {'name': MODELS, 'value': OVERRIDES[MODELS]}]
    assert result['envFrom'] == container(saved)['envFrom']
    assert result['resources'] == container(saved)['resources']
    assert saved == before


def test_duplicate_targeted_env_name_is_rejected_instead_of_guessing():
    saved = original([{'name': MODE, 'value': 'off'}, {'name': MODE, 'value': 'observe'}])
    with pytest.raises(ValueError):
        target_pod_spec(saved, plan())


@pytest.mark.parametrize('old_env', [ABSENT, None, [],
    [{'name': MODE, 'value': 'observe'}, {'name': 'PRIVATE', 'value': 'fixture'}],
    [{'name': MODELS, 'valueFrom': {'secretKeyRef': {'name': 's', 'key': 'models'}}}],
])
def test_atomic_forward_and_rollback_restore_full_original_env_state(old_env):
    value = backend(old_env)
    before = copy.deepcopy(container(value.snapshot['deployment']))
    value._set_image(NEW)
    current = value.api.current
    assert {e['name']: e['value'] for e in container(current)['env'] if e['name'] in OVERRIDES} == OVERRIDES
    operations = value.api.requests[-1][0]
    tests = {op['path'] for op in operations if op['op'] == 'test'}
    assert {'/metadata/uid', '/metadata/resourceVersion', '/spec/template',
            '/metadata/annotations/lb.zeno.ink~1release', '/metadata/annotations/lb.zeno.ink~1release-epoch'} <= tests
    assert container(current)['image'] == NEW
    assert current['spec']['template']['metadata']['annotations'][SOURCE] == 'a' * 40
    value._set_image(OLD, rollback=True)
    assert container(value.api.current) == before
    assert value.api.current['spec']['template']['metadata']['annotations'][SOURCE] == 'b' * 40
    assert value.api.mutations == 2


@pytest.mark.parametrize('rollback', [False, True])
def test_ack_loss_reads_back_full_image_env_pair_without_second_rollout(rollback):
    value = backend([{'name': 'PRIVATE', 'value': 'fixture'}])
    if rollback:
        value._set_image(NEW)
    value.api.lose_ack = True
    with pytest.raises(UncertainOperation):
        value._set_image(OLD if rollback else NEW, rollback=rollback)
    mutations = value.api.mutations
    # Reconstruct the backend as a new executor would, using the saved snapshot.
    takeover = backend(); takeover.api = value.api; takeover.snapshot = value.snapshot
    takeover._set_image(OLD if rollback else NEW, rollback=rollback)
    assert takeover.api.mutations == mutations
    assert container(takeover.api.current).get('env') == (
        container(value.snapshot['deployment']).get('env') if rollback else
        target_pod_spec(value.snapshot['deployment'], value.plan)['containers'][0]['env'])


@pytest.mark.parametrize('change', ['old_with_target_env', 'target_with_old_env', 'unrelated_env', 'env_from', 'target_value', 'uid'])
def test_environment_guard_is_image_tied_and_rejects_out_of_band_changes(change):
    value = backend([{'name': 'PRIVATE', 'value': 'fixture'}])
    if change not in ('old_with_target_env', 'uid'):
        value._set_image(NEW)
    current = value.api.current
    if change == 'old_with_target_env':
        container(current)['env'] = target_pod_spec(value.snapshot['deployment'], value.plan)['containers'][0]['env']
    elif change == 'target_with_old_env':
        container(current)['env'] = container(value.snapshot['deployment'])['env']
    elif change == 'unrelated_env':
        container(current)['env'][0]['value'] = 'manual'
    elif change == 'env_from':
        container(current)['envFrom'] = []
    elif change == 'target_value':
        container(current)['env'][-1]['value'] = 'unexpected-model'
    else:
        current['metadata']['uid'] = 'replacement'
    # Even forging the controlled digest cannot turn an unapproved env into a target.
    current['metadata']['annotations'][SPEC] = digest(current['spec'])
    with pytest.raises(UnsafeState):
        value._guard_spec()
    before = value.api.mutations
    with pytest.raises(UnsafeState):
        value._set_image(OLD, rollback=True)
    assert value.api.mutations == before


@pytest.mark.parametrize('when', ['dry_run', 'normal_ack'])
def test_admission_env_mutation_cannot_be_accepted(when):
    value = backend()
    def mutate(updated, dry_run):
        if dry_run == (when == 'dry_run'):
            container(updated)['env'][-1]['value'] = 'admission-model'
    value.api.admission = mutate
    with pytest.raises(ValueError):
        value._set_image(NEW)
    assert value.api.mutations == (0 if when == 'dry_run' else 1)


def test_ownership_loss_stops_environment_writes():
    value = backend(); value._set_image(NEW)
    before = copy.deepcopy(value.api.current)
    value.check_owner.side_effect = OwnershipLost('new execution epoch')
    with pytest.raises(OwnershipLost):
        value._set_image(OLD, rollback=True)
    assert value.api.current == before


@pytest.mark.parametrize('overrides,schema', [(OVERRIDES, 'lb-release-plan-v2'), ({}, 'lb-release-plan-v1'), (ABSENT, 'lb-release-plan-v1')])
def test_planner_emits_reviewable_schema_and_allowlisted_values(tmp_path, overrides, schema):
    cli = importlib.import_module('operations.release.__main__')
    value = backend(); profile = plan(overrides)
    with patch.object(cli, 'capture', return_value=value.snapshot):
        cli.plan_release(value.api, profile, tmp_path / 'plan', '.')
    saved = json.loads((tmp_path / 'plan' / 'plan.json').read_text())
    review = json.loads((tmp_path / 'plan' / 'review.json').read_text())
    assert saved['schema_version'] == schema
    assert review.get('target_env', {}) == (OVERRIDES if overrides is OVERRIDES else {})
    assert 'PRIVATE' not in json.dumps(review)
    assert value.api.mutations == 0


def test_planning_rejects_admission_environment_mutation(tmp_path):
    cli = importlib.import_module('operations.release.__main__')
    value = backend()
    value.api.admission = lambda updated, _: container(updated)['env'][-1].update(value='changed')
    with patch.object(cli, 'capture', return_value=value.snapshot), pytest.raises(ValueError):
        cli.plan_release(value.api, plan(), tmp_path / 'plan', '.')
    assert not (tmp_path / 'plan').exists()


def test_cluster_preflight_rejects_admission_environment_mutation_before_gating():
    value = backend()
    value.plan['snapshot_sha256'] = digest(value.snapshot)
    value.lab = True
    value._verified_old_image = Mock(); value._verify_references = Mock(); value._verify_public_targets = Mock()
    value.api.list = Mock(side_effect=AssertionError('Preflight continued after a modified target'))
    value.api.admission = lambda updated, _: container(updated)['env'][-1].update(value='changed')
    with pytest.raises(ValueError):
        value.perform('preflight', {'action_receipts': {}})
    value.api.list.assert_not_called()
    assert value.api.mutations == 0


@pytest.mark.parametrize('suffix', ['new', 'old'])
def test_image_preflight_uses_target_env_only_for_new_image(suffix):
    value = backend([{'name': MODE, 'value': 'off'}, {'name': 'PRIVATE', 'value': 'fixture'}])
    value.api = SimpleNamespace(optional=Mock(return_value=None), create=Mock())
    with pytest.raises(PendingOperation):
        value._image_preflight(NEW if suffix == 'new' else OLD, suffix, {'main.py': 'f'} if suffix == 'new' else None, {})
    check = value.api.create.call_args.args[2]
    expected = (target_pod_spec(value.snapshot['deployment'], value.plan)['containers'][0]
                if suffix == 'new' else container(value.snapshot['deployment']))
    assert check['spec']['containers'][0]['env'] == expected['env']
    assert check['spec']['containers'][0]['envFrom'] == expected['envFrom']
    values = {entry['name']: entry['value'] for entry in check['spec']['containers'][0]['env']}
    assert values[MODE] == ('stabilize' if suffix == 'new' else 'off')
    assert (values.get(MODELS) == OVERRIDES[MODELS]) if suffix == 'new' else MODELS not in values


@pytest.mark.parametrize('suffix', ['new', 'old'])
@pytest.mark.parametrize('mutation', ['environment', 'env_from', 'image'])
def test_preflight_pod_admission_mutation_cannot_certify_planned_environment(suffix, mutation):
    value = backend([{'name': MODE, 'value': 'off'}])
    expected = (target_pod_spec(value.snapshot['deployment'], value.plan) if suffix == 'new' else
                copy.deepcopy(value.snapshot['deployment']['spec']['template']['spec']))
    image = NEW if suffix == 'new' else OLD
    pod = {'metadata': {'annotations': {CONTROL: value.release_id}}, 'spec': expected,
           'status': {'phase': 'Succeeded', 'containerStatuses': [{'state': {'terminated': {'exitCode': 0}}}]}}
    app = pod['spec']['containers'][0]
    if mutation == 'environment':
        app['env'][0]['value'] = 'admission-value'
    elif mutation == 'env_from':
        app['envFrom'] = []
    else:
        app['image'] = 'unexpected-image'
    value.api = SimpleNamespace(optional=Mock(return_value=pod))
    receipts = {}
    with pytest.raises(UnsafeState):
        value._image_preflight(image, suffix, {'main.py': 'f'} if suffix == 'new' else None, receipts)
    assert receipts == {}


@pytest.mark.parametrize('target', ['new', 'rollback'])
@pytest.mark.parametrize('drift', [False, True])
def test_live_pod_environment_must_match_its_reviewed_image(target, drift):
    value = backend([{'name': MODE, 'value': 'observe'}])
    old_spec = value.snapshot['deployment']['spec']['template']['spec']
    old_pod = {'metadata': {'name': 'old', 'uid': 'old'}, 'spec': copy.deepcopy(old_spec),
               'status': {'containerStatuses': [{'name': 'app', 'imageID': OLD}]}}
    value.snapshot['old_pods'] = [old_pod]
    pod = {'metadata': {'name': 'new', 'uid': 'new'},
           'spec': target_pod_spec(value.snapshot['deployment'], value.plan) if target == 'new' else copy.deepcopy(old_spec),
           'status': {'containerStatuses': [{'name': 'app', 'imageID': NEW if target == 'new' else OLD}]}}
    if drift:
        pod['spec']['containers'][0]['env'][-1]['value'] = 'manual'
    value._pods = Mock(return_value=[pod])
    invoke = value._new_pod if target == 'new' else lambda: value._verified_old_image(pod)
    if drift:
        with pytest.raises(UnsafeState):
            invoke()
    else:
        invoke()


@pytest.mark.parametrize('guard', ['uid', 'resourceVersion', 'owner', 'epoch', 'template'])
def test_each_write_guard_prevents_concurrent_change_from_being_overwritten(guard):
    value = backend()
    request = value.api.request
    def race(method, kind, namespace, name, body, query=None, patch=False):
        if not query:
            current = value.api.current
            if guard in ('uid', 'resourceVersion'):
                current['metadata'][guard] = 'concurrent'
            elif guard in ('owner', 'epoch'):
                current['metadata']['annotations'][CONTROL if guard == 'owner' else EPOCH] = 'concurrent'
            else:
                container(current)['env'] = [{'name': 'MANUAL', 'value': 'preserved'}]
        return request(method, kind, namespace, name, body, query=query, patch=patch)
    value.api.request = race
    with pytest.raises(PendingOperation):
        value._set_image(NEW)
    assert value.api.mutations == 0
    assert container(value.api.current)['image'] == OLD
    if guard == 'template':
        assert container(value.api.current)['env'] == [{'name': 'MANUAL', 'value': 'preserved'}]


@pytest.mark.parametrize('slices', [[], [{}], [{'endpoints': None}], [{'endpoints': []}],
    [{'endpoints': [{'conditions': {}}]}], [{'endpoints': [{'conditions': {'ready': False}}]}],
    [{'endpoints': [{'conditions': {'ready': True, 'terminating': True}}]}],
])
def test_route_restore_never_accepts_missing_or_unusable_endpoints(slices):
    value = backend(); value._set_image(NEW)
    value._routes = Mock(return_value=True); value._verify_references = Mock(); value._public_health = Mock()
    service = value.snapshot['services']['lb']; value._owned = Mock(return_value=service)
    value.api.list = Mock(return_value=slices)
    with pytest.raises(PendingOperation):
        value._restore_routes()
    value._public_health.assert_not_called()


class EnvironmentEngineBackend(FakeBackend):
    """Real template/CAS code inside the state machine's synthetic lifecycle."""
    def __init__(self, effects):
        super().__init__(); self.effects = effects

    def observe(self, record):
        self.effects._guard_spec()
        return super().observe(record)

    def perform(self, action, record):
        self.check_owner()
        if action == 'start_new' and self.fault != action:
            self.effects._set_image(NEW)
        elif action == 'rollback' and self.healthy and self.fault != action:
            self.effects._set_image(OLD, rollback=True)
        return super().perform(action, record)


def run_engine(stop_after=None, fail_at=None, ack=False, journal=False, rollback_failure=False, lose_owner=False, journal_after_start=False):
    effects = backend([{'name': MODE, 'value': 'observe'}, {'name': 'PRIVATE', 'value': 'fixture'}])
    lifecycle = EnvironmentEngineBackend(effects)
    store = MemoryStore(new_record(validate_plan(plan())))
    if journal_after_start:
        save = store.save
        def fail_once(record):
            nonlocal journal_after_start
            if journal_after_start and 'start_new' in record['action_receipts']:
                journal_after_start = False
                raise OSError('journal lost after image and environment write')
            save(record)
        store.save = fail_once
    clock = [100.0]
    engine = Engine(store, lifecycle, clock=lambda: clock[0])
    for count in range(40):
        phase = store.read()['phase']
        if count == stop_after:
            # New executor shares only durable API/snapshot and journal.
            fresh = backend(); fresh.api = effects.api; fresh.snapshot = effects.snapshot
            lifecycle.effects = fresh
            engine = Engine(store, lifecycle, clock=lambda: clock[0])
        if phase == 'starting' and ack:
            effects.api.lose_ack = True; ack = False
        if phase == fail_at:
            lifecycle.fault = 'verify_backend' if phase == 'verifying_backend' else 'start_new'
            if rollback_failure:
                lifecycle.healthy = False
        if phase == 'draining' and journal:
            store.fail = True; engine.tick(); store.fail = False; journal = False
        if phase == 'starting' and lose_owner:
            lifecycle.owned = False
            before = copy.deepcopy(effects.api.current)
            result = engine.tick()
            assert effects.api.current == before
            return result, effects, lifecycle
        result = engine.tick(); clock[0] += 1
        if result['phase'] in ('succeeded', 'rolled_back', 'cancelled', 'needs_attention'):
            return result, effects, lifecycle
    raise AssertionError('release did not converge')


@pytest.mark.parametrize('stop_after', range(1, 11))
def test_env_release_executor_restart_in_each_phase_uses_durable_pair(stop_after):
    result, effects, _ = run_engine(stop_after=stop_after)
    assert result['phase'] == 'succeeded'
    assert container(effects.api.current)['env'] == target_pod_spec(effects.snapshot['deployment'], effects.plan)['containers'][0]['env']


def test_env_release_lost_ack_journal_failure_rollback_failure_and_owner_loss():
    result, effects, lifecycle = run_engine(ack=True)
    assert result['phase'] == 'succeeded' and effects.api.mutations == 1
    assert 'rollback' not in lifecycle.calls
    for scenario in ({'journal': True}, {'fail_at': 'verifying_backend'}):
        result, effects, lifecycle = run_engine(**scenario)
        assert result['phase'] == 'rolled_back' and lifecycle.routes
        assert container(effects.api.current)['env'] == container(effects.snapshot['deployment'])['env']
    result, _, lifecycle = run_engine(fail_at='verifying_backend', rollback_failure=True)
    assert result['phase'] == 'needs_attention' and not lifecycle.routes
    result, effects, _ = run_engine(lose_owner=True)
    assert result['phase'] != 'succeeded' and effects.api.mutations == 0


def test_journal_failure_after_environment_switch_recovers_original_pair(capsys):
    result, effects, lifecycle = run_engine(journal_after_start=True)
    assert result['phase'] == 'rolled_back' and lifecycle.routes
    assert effects.api.mutations == 2
    assert container(effects.api.current)['image'] == OLD
    assert container(effects.api.current)['env'] == container(effects.snapshot['deployment'])['env']
    assert 'release_emergency_recovery_unverified' in capsys.readouterr().err
