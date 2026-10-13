"""Operator-entered upstream credentials, sealed before they are stored. Author: Zeno Ren.

This module is a posture change: the gateway previously never stored upstream
credentials, only read them from config, env or a mounted Secret. It now accepts
them and persists them, because the Secret mount is read-only tmpfs, config.yaml
is read-only, and the app's ServiceAccount has no secret write permission — the
usage database is the only writable durable store available.

Sealing is therefore mandatory, not optional. AES-256-GCM with a key supplied
out of band (env, so it lives in the Secret rather than the database), a fresh
nonce per record, and the record identity as additional authenticated data so a
ciphertext cannot be moved to another row. Without a key the feature refuses to
operate rather than quietly writing plaintext tokens next to the usage ledger.

What this does not solve: the running process holds plaintext in memory, so
anyone who can exec in the pod can still read it, and losing the key makes
stored credentials unrecoverable.
"""
import asyncio
import base64
import json
import os
import unittest
from unittest.mock import patch

import provider_credentials as pc

KEY = base64.b64encode(b'k' * 32).decode()
OTHER_KEY = base64.b64encode(b'j' * 32).decode()
SECRET = 'dapi-synthetic-databricks-token-0123456789'


def _store(key=KEY):
    env = {} if key is None else {'LB_CREDENTIAL_KEY': key}
    with patch.dict(os.environ, env, clear=False):
        if key is None:
            os.environ.pop('LB_CREDENTIAL_KEY', None)
        store = pc.CredentialStore(None)
        asyncio.run(store.start())
    return store


class SealingTests(unittest.TestCase):
    def test_round_trip(self):
        sealed = pc.seal(KEY, 'databricks', 'ws-1', SECRET)
        self.assertEqual(pc.unseal(KEY, 'databricks', 'ws-1', sealed), SECRET)

    def test_ciphertext_does_not_contain_the_plaintext(self):
        sealed = pc.seal(KEY, 'databricks', 'ws-1', SECRET)
        self.assertNotIn(SECRET, sealed)
        self.assertNotIn(SECRET.encode(), base64.b64decode(sealed))

    def test_each_seal_differs_even_for_the_same_input(self):
        first = pc.seal(KEY, 'databricks', 'ws-1', SECRET)
        second = pc.seal(KEY, 'databricks', 'ws-1', SECRET)
        self.assertNotEqual(first, second, 'a fixed nonce would leak equality of secrets')

    def test_identity_is_authenticated_so_a_ciphertext_cannot_be_relocated(self):
        sealed = pc.seal(KEY, 'databricks', 'ws-1', SECRET)
        for provider, account in (('databricks', 'ws-2'), ('copilot', 'ws-1'), ('azure_openai', 'ws-1')):
            with self.subTest(provider=provider, account=account):
                with self.assertRaises(pc.CredentialSealError):
                    pc.unseal(KEY, provider, account, sealed)

    def test_a_different_key_cannot_open_it(self):
        sealed = pc.seal(KEY, 'databricks', 'ws-1', SECRET)
        with self.assertRaises(pc.CredentialSealError):
            pc.unseal(OTHER_KEY, 'databricks', 'ws-1', sealed)

    def test_tampering_is_detected(self):
        sealed = pc.seal(KEY, 'databricks', 'ws-1', SECRET)
        raw = bytearray(base64.b64decode(sealed))
        raw[-1] ^= 0x01
        with self.assertRaises(pc.CredentialSealError):
            pc.unseal(KEY, 'databricks', 'ws-1', base64.b64encode(bytes(raw)).decode())

    def test_key_must_be_32_bytes(self):
        for bad in (base64.b64encode(b'short').decode(), 'not-base64!!', ''):
            with self.subTest(key=bad):
                with self.assertRaises(pc.CredentialKeyError):
                    pc.seal(bad, 'databricks', 'ws-1', SECRET)

    def test_key_fingerprint_identifies_the_key_without_revealing_it(self):
        fingerprint = pc.key_fingerprint(KEY)
        self.assertEqual(len(fingerprint), 12)
        self.assertNotIn(fingerprint, KEY)
        self.assertNotEqual(fingerprint, pc.key_fingerprint(OTHER_KEY))


class StoreStateTests(unittest.TestCase):
    def test_store_without_a_key_reports_it_and_refuses_writes(self):
        store = _store(key=None)
        self.assertFalse(store.sealing_available)
        self.assertEqual(store.status()['key'], 'missing')
        with self.assertRaises(pc.CredentialKeyError):
            asyncio.run(store.put('databricks', 'ws-1', SECRET, {'api_base': 'https://x.invalid'}))

    def test_store_with_a_key_is_ready(self):
        store = _store()
        self.assertTrue(store.sealing_available)
        self.assertEqual(store.status()['key'], 'present')
        self.assertEqual(store.status()['key_fingerprint'], pc.key_fingerprint(KEY))

    def test_put_then_list_exposes_identity_only(self):
        store = _store()
        asyncio.run(store.put('databricks', 'ws-1', SECRET, {'api_base': 'https://x.invalid'}))
        listed = store.listing()
        self.assertEqual(len(listed), 1)
        entry = listed[0]
        self.assertEqual((entry['provider'], entry['account']), ('databricks', 'ws-1'))
        self.assertEqual(entry['secret_fingerprint'], pc.secret_fingerprint(SECRET))
        self.assertEqual(entry['meta']['api_base'], 'https://x.invalid')
        blob = json.dumps(listed)
        self.assertNotIn(SECRET, blob)
        self.assertNotIn('sealed', blob)

    def test_resolve_returns_the_plaintext_only_to_internal_callers(self):
        store = _store()
        asyncio.run(store.put('databricks', 'ws-1', SECRET, {'api_base': 'https://x.invalid'}))
        self.assertEqual(store.resolve('databricks', 'ws-1'), SECRET)

    def test_a_record_sealed_with_another_key_is_reported_not_crashed(self):
        store = _store()
        asyncio.run(store.put('databricks', 'ws-1', SECRET, {}))
        rotated = _store(key=OTHER_KEY)
        rotated._records = dict(store._records)
        entry = rotated.listing()[0]
        self.assertEqual(entry['key_state'], 'sealed_with_other_key')
        self.assertIsNone(rotated.resolve('databricks', 'ws-1'))

    def test_delete_removes_the_record(self):
        store = _store()
        asyncio.run(store.put('databricks', 'ws-1', SECRET, {}))
        asyncio.run(store.delete('databricks', 'ws-1'))
        self.assertEqual(store.listing(), [])

    def test_unknown_provider_is_rejected(self):
        store = _store()
        with self.assertRaises(ValueError):
            asyncio.run(store.put('gcp', 'ws-1', SECRET, {}))

    def test_empty_secret_is_rejected(self):
        store = _store()
        for bad in ('', '   ', None):
            with self.subTest(secret=bad):
                with self.assertRaises(ValueError):
                    asyncio.run(store.put('databricks', 'ws-1', bad, {}))

    def test_write_failure_does_not_leave_a_half_applied_record(self):
        store = _store()
        with patch.object(store, '_backend_write', side_effect=OSError('db down')):
            with self.assertRaises(OSError):
                asyncio.run(store.put('databricks', 'ws-1', SECRET, {}))
        self.assertEqual(store.listing(), [])

    def test_load_failure_leaves_the_store_usable_and_visible(self):
        with patch.dict(os.environ, {'LB_CREDENTIAL_KEY': KEY}):
            store = pc.CredentialStore({'type': 'mysql'})
            with patch.object(store, '_backend_load', side_effect=OSError('db down')):
                asyncio.run(store.start())
        self.assertEqual(store.status()['state'], 'unavailable')
        self.assertEqual(store.listing(), [])


class EndpointMaterialisationTests(unittest.TestCase):
    """Stored credentials become endpoints at startup, never mid-flight."""

    def test_databricks_record_yields_an_endpoint_spec(self):
        store = _store()
        asyncio.run(store.put('databricks', 'ws-9', SECRET,
                              {'api_base': 'https://adb-9.invalid/serving-endpoints', 'weight': 2}))
        specs = store.endpoint_specs('databricks')
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]['name'], 'ws-9')
        self.assertEqual(specs[0]['token'], SECRET)
        self.assertEqual(specs[0]['api_base'], 'https://adb-9.invalid/serving-endpoints')
        self.assertEqual(specs[0]['weight'], 2)

    def test_a_record_that_cannot_be_unsealed_is_skipped_not_guessed(self):
        store = _store()
        asyncio.run(store.put('databricks', 'ws-9', SECRET, {'api_base': 'https://x.invalid'}))
        rotated = _store(key=OTHER_KEY)
        rotated._records = dict(store._records)
        self.assertEqual(rotated.endpoint_specs('databricks'), [])

    def test_azure_record_carries_deployments(self):
        store = _store()
        asyncio.run(store.put('azure_openai', 'az-1', 'azure-key-value',
                              {'api_base': 'https://r.openai.azure.com', 'deployments': ['gpt-5.4']}))
        spec = store.endpoint_specs('azure_openai')[0]
        self.assertEqual(spec['deployments'], ['gpt-5.4'])
        self.assertEqual(spec['api_key'], 'azure-key-value')

    def test_copilot_record_needs_no_endpoint_url(self):
        store = _store()
        asyncio.run(store.put('copilot', 'gh-9', 'ghu_synthetic_token_value', {}))
        spec = store.endpoint_specs('copilot')[0]
        self.assertEqual(spec['github_token'], 'ghu_synthetic_token_value')
        self.assertEqual(spec['name'], 'gh-9')


class DeviceFlowTests(unittest.TestCase):
    def test_a_started_flow_never_echoes_the_device_code(self):
        flows = pc.DeviceFlowRegistry()
        handle = flows.start({'device_code': 'secret-device-code', 'user_code': 'ABCD-1234',
                              'verification_uri': 'https://github.com/login/device',
                              'expires_in': 900, 'interval': 5})
        view = flows.view(handle)
        blob = json.dumps(view)
        self.assertNotIn('secret-device-code', blob)
        self.assertEqual(view['user_code'], 'ABCD-1234')
        self.assertEqual(flows.device_code(handle), 'secret-device-code')

    def test_an_expired_flow_fails_clearly_rather_than_hanging(self):
        flows = pc.DeviceFlowRegistry()
        handle = flows.start({'device_code': 'c', 'user_code': 'U', 'verification_uri': 'https://x',
                              'expires_in': -1, 'interval': 5})
        with self.assertRaises(pc.DeviceFlowExpired):
            flows.device_code(handle)

    def test_an_unknown_handle_is_rejected(self):
        with self.assertRaises(KeyError):
            pc.DeviceFlowRegistry().device_code('nope')

    def test_completing_a_flow_forgets_the_device_code(self):
        flows = pc.DeviceFlowRegistry()
        handle = flows.start({'device_code': 'c', 'user_code': 'U', 'verification_uri': 'https://x',
                              'expires_in': 900, 'interval': 5})
        flows.finish(handle)
        with self.assertRaises(KeyError):
            flows.device_code(handle)


class AdminEndpointTests(unittest.TestCase):
    ADMIN = 'lb-cred-admin-key-0123456'

    def setUp(self):
        from starlette.testclient import TestClient

        import main
        self.main = main

        class _Stub:
            api_key = AdminEndpointTests.ADMIN

            def verify_api_key(self, key):
                return bool(key) and key == self.api_key

        self.store = _store()
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[self.ADMIN] = 'default'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        self.databricks = _Stub()
        self.databricks.load_balancer = type('LB', (), {'endpoints': []})()
        for attr, value in (('proxy', self.databricks), ('copilot_proxy', _Stub()),
                            ('credential_store', self.store), ('key_policy_store', None)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        main.CREDENTIAL_SOURCED_ACCOUNTS.clear()
        self.addCleanup(main.CREDENTIAL_SOURCED_ACCOUNTS.clear)
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.H = {'x-api-key': self.ADMIN}

    def _ok_validation(self, *_args, **_kwargs):
        async def fake(provider, account, secret, meta):
            return {'ok': True, 'models': ['gpt-4o'], 'model_count': 1}
        return patch.object(self.main, '_validate_credential', fake)

    def _rejected_validation(self):
        async def fake(provider, account, secret, meta):
            return {'ok': False, 'status': 401, 'reason': 'rejected_by_workspace'}
        return patch.object(self.main, '_validate_credential', fake)

    def test_every_endpoint_requires_a_valid_key(self):
        cases = [('get', '/admin/credentials', None),
                 ('put', '/admin/credentials/databricks/ws-1', {'secret': SECRET}),
                 ('delete', '/admin/credentials/databricks/ws-1', None),
                 ('post', '/admin/credentials/databricks/ws-1/validate', None),
                 ('post', '/admin/credentials/copilot/device/start', None),
                 ('post', '/admin/credentials/copilot/device/poll', {'handle': 'x', 'account': 'y'})]
        for method, path, body in cases:
            with self.subTest(path=path, method=method):
                kwargs = {'json': body} if body is not None else {}
                self.assertEqual(getattr(self.client, method)(path, **kwargs).status_code, 401)

    def test_entry_validates_before_storing_and_never_echoes_the_secret(self):
        with self._ok_validation():
            r = self.client.put('/admin/credentials/databricks/ws-1', headers=self.H,
                                json={'secret': SECRET, 'meta': {'api_base': 'https://x.invalid'}})
        self.assertEqual(r.status_code, 200)
        blob = json.dumps(r.json())
        self.assertNotIn(SECRET, blob)
        self.assertNotIn('sealed', blob)
        self.assertEqual(r.json()['activation'], 'next rolling restart')

    def test_a_rejected_credential_is_not_stored(self):
        with self._rejected_validation():
            r = self.client.put('/admin/credentials/databricks/ws-1', headers=self.H,
                                json={'secret': SECRET, 'meta': {'api_base': 'https://x.invalid'}})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'credential_rejected_upstream')
        self.assertEqual(self.store.listing(), [])

    def test_an_account_declared_in_config_is_refused(self):
        self.databricks.load_balancer.endpoints = [
            main_endpoint := type('E', (), {'name': 'ws-config'})()]
        self.assertTrue(main_endpoint)
        with self._ok_validation():
            r = self.client.put('/admin/credentials/databricks/ws-config', headers=self.H,
                                json={'secret': SECRET, 'meta': {'api_base': 'https://x.invalid'}})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()['detail']['error']['code'], 'account_declared_in_config')

    def test_listing_never_returns_plaintext_or_ciphertext(self):
        with self._ok_validation():
            self.client.put('/admin/credentials/databricks/ws-1', headers=self.H,
                            json={'secret': SECRET, 'meta': {'api_base': 'https://x.invalid'}})
        blob = json.dumps(self.client.get('/admin/credentials', headers=self.H).json())
        self.assertNotIn(SECRET, blob)
        self.assertNotIn('sealed', blob)

    def test_listing_marks_inactive_until_restart(self):
        with self._ok_validation():
            self.client.put('/admin/credentials/databricks/ws-1', headers=self.H,
                            json={'secret': SECRET, 'meta': {'api_base': 'https://x.invalid'}})
        entry = self.client.get('/admin/credentials', headers=self.H).json()['credentials'][0]
        self.assertFalse(entry['active'], 'a fresh credential must not claim to be live')

    def test_unknown_provider_is_refused(self):
        r = self.client.put('/admin/credentials/gcp/x', headers=self.H, json={'secret': SECRET})
        self.assertEqual(r.status_code, 400)

    def test_missing_secret_is_refused(self):
        r = self.client.put('/admin/credentials/databricks/ws-1', headers=self.H, json={})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'secret_required')

    def test_entry_is_disabled_without_a_sealing_key(self):
        with patch.object(self.main, 'credential_store', _store(key=None)):
            r = self.client.put('/admin/credentials/databricks/ws-1', headers=self.H,
                                json={'secret': SECRET})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()['detail']['error']['code'], 'credential_key_missing')

    def test_device_poll_reports_pending_without_treating_it_as_failure(self):
        handle = self.main.DEVICE_FLOWS.start({'device_code': 'dc', 'user_code': 'U',
                                               'verification_uri': 'https://x',
                                               'expires_in': 900, 'interval': 5})
        self.addCleanup(self.main.DEVICE_FLOWS.finish, handle)

        class _Resp:
            status_code = 200
            headers = {'content-type': 'application/json'}
            text = '{"error":"authorization_pending"}'

            def json(self):
                return {'error': 'authorization_pending'}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return _Resp()

        with patch.object(self.main.httpx, 'AsyncClient', lambda **k: _Client()):
            r = self.client.post('/admin/credentials/copilot/device/poll', headers=self.H,
                                 json={'handle': handle, 'account': 'gh-new'})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()['pending'])
        self.assertIsNone(r.json()['error'])

    def test_device_poll_on_an_unknown_handle_is_404(self):
        r = self.client.post('/admin/credentials/copilot/device/poll', headers=self.H,
                             json={'handle': 'nope', 'account': 'gh-new'})
        self.assertEqual(r.status_code, 404)

    def test_device_poll_on_an_expired_flow_is_410(self):
        handle = self.main.DEVICE_FLOWS.start({'device_code': 'dc', 'user_code': 'U',
                                               'verification_uri': 'https://x',
                                               'expires_in': -1, 'interval': 5})
        r = self.client.post('/admin/credentials/copilot/device/poll', headers=self.H,
                             json={'handle': handle, 'account': 'gh-new'})
        self.assertEqual(r.status_code, 410)

    def test_device_start_response_has_no_device_code(self):
        class _Resp:
            status_code = 200
            headers = {'content-type': 'application/json'}
            text = '{}'

            def json(self):
                return {'device_code': 'super-secret-code', 'user_code': 'ABCD-1234',
                        'verification_uri': 'https://github.com/login/device',
                        'expires_in': 900, 'interval': 5}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return _Resp()

        with patch.object(self.main.httpx, 'AsyncClient', lambda **k: _Client()):
            r = self.client.post('/admin/credentials/copilot/device/start', headers=self.H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertNotIn('super-secret-code', json.dumps(body))
        self.assertEqual(body['user_code'], 'ABCD-1234')
        self.main.DEVICE_FLOWS.finish(body['handle'])


if __name__ == '__main__':
    unittest.main()
