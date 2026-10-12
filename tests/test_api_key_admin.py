"""Key listing and source-driven rotation. Author: Zeno Ren.

The gateway is not a credential issuer: it never generates or stores client API
keys. Rotation follows the model already used by /admin/copilot/reload — rotate
at the source (Secret / env / file), then ask the gateway to re-read.

Listing returns identity only: tenant name, masked outer characters and a
sha256 prefix. Plaintext is never returned.

Re-reading must be atomic. The previous registration helper cleared the map
before parsing, so a malformed config would have locked every caller out.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

import key_policy
import main

KEY = 'lb-admin-key-0123456789'


class _Stub:
    api_key = KEY

    def verify_api_key(self, key):
        return bool(key) and key == self.api_key


class ApiKeyAdminTests(unittest.TestCase):
    def setUp(self):
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[KEY] = 'default'
        main.API_KEY_TO_TENANT['second-tenant-key-abcdef'] = 'team-a'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        main.TENANT_USAGE.clear()
        self.addCleanup(main.TENANT_USAGE.clear)
        for attr in ('proxy', 'copilot_proxy'):
            p = patch.object(main, attr, _Stub())
            p.start()
            self.addCleanup(p.stop)
        self.store = key_policy.KeyPolicyStore(None)
        import asyncio
        asyncio.run(self.store.start())
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)

    # -- listing ---------------------------------------------------------

    def test_listing_requires_a_valid_key(self):
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)
        self.assertEqual(self.client.get('/admin/api-keys',
                                         headers={'x-api-key': 'wrong'}).status_code, 401)

    def test_listing_never_returns_plaintext_keys(self):
        r = self.client.get('/admin/api-keys', headers={'x-api-key': KEY})
        self.assertEqual(r.status_code, 200)
        blob = json.dumps(r.json())
        for secret in main.API_KEY_TO_TENANT:
            with self.subTest(secret=secret):
                self.assertNotIn(secret, blob)

    def test_listing_reports_identity_and_policy_for_every_tenant(self):
        r = self.client.get('/admin/api-keys', headers={'x-api-key': KEY}).json()
        tenants = {item['tenant']: item for item in r['keys']}
        self.assertEqual(set(tenants), {'default', 'team-a'})
        entry = tenants['default']
        self.assertEqual(entry['fingerprint'], key_policy.fingerprint(KEY))
        self.assertEqual(entry['masked'], key_policy.mask(KEY))
        self.assertEqual(set(entry['providers']), set(key_policy.PROVIDER_IDS))
        self.assertFalse(entry['restricted'])
        self.assertEqual(r['providers'], list(key_policy.PROVIDER_IDS))
        self.assertEqual(r['policy']['state'], 'loaded')

    def test_in_process_counters_are_labelled_as_process_local(self):
        main._note_tenant_request('default')
        main._note_tenant_request('default')
        r = self.client.get('/admin/api-keys', headers={'x-api-key': KEY}).json()
        entry = next(i for i in r['keys'] if i['tenant'] == 'default')
        self.assertEqual(entry['requests'], 2)
        self.assertIsNotNone(entry['last_used_at'])
        self.assertTrue(r['counters_are_process_local'])

    def test_untouched_tenant_has_no_last_used_rather_than_a_fake_zero_time(self):
        r = self.client.get('/admin/api-keys', headers={'x-api-key': KEY}).json()
        entry = next(i for i in r['keys'] if i['tenant'] == 'team-a')
        self.assertEqual(entry['requests'], 0)
        self.assertIsNone(entry['last_used_at'])

    # -- policy write ----------------------------------------------------

    def test_setting_providers_requires_a_valid_key(self):
        r = self.client.put('/admin/api-keys/team-a/providers', json={'providers': ['copilot']})
        self.assertEqual(r.status_code, 401)

    def test_setting_providers_takes_effect_immediately(self):
        r = self.client.put('/admin/api-keys/team-a/providers',
                            headers={'x-api-key': KEY}, json={'providers': ['copilot']})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['providers'], ['copilot'])
        self.assertEqual(self.store.allowed('team-a'), frozenset({'copilot'}))

    def test_unknown_provider_is_rejected_with_400(self):
        r = self.client.put('/admin/api-keys/team-a/providers',
                            headers={'x-api-key': KEY}, json={'providers': ['gcp']})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.store.allowed('team-a'), key_policy.ALL_PROVIDERS)

    def test_empty_provider_list_is_rejected_rather_than_locking_a_tenant_out(self):
        r = self.client.put('/admin/api-keys/team-a/providers',
                            headers={'x-api-key': KEY}, json={'providers': []})
        self.assertEqual(r.status_code, 400)

    def test_unknown_tenant_is_rejected(self):
        r = self.client.put('/admin/api-keys/nobody/providers',
                            headers={'x-api-key': KEY}, json={'providers': ['copilot']})
        self.assertEqual(r.status_code, 404)

    # -- rotation --------------------------------------------------------

    def test_reload_requires_a_valid_key(self):
        self.assertEqual(self.client.post('/admin/api-keys/reload').status_code, 401)

    def test_reload_picks_up_a_rotated_source_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('auth:\n  api_keys:\n    default: rotated-key-value-9876\n')
            with patch.dict(os.environ, {'CONFIG_PATH': str(path)}):
                r = self.client.post('/admin/api-keys/reload', headers={'x-api-key': KEY})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(main.API_KEY_TO_TENANT.get('rotated-key-value-9876'), 'default')
        self.assertNotIn(KEY, main.API_KEY_TO_TENANT)
        self.assertEqual(r.json()['tenants'], ['default'])

    def test_a_malformed_source_does_not_lock_everyone_out(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('auth:\n  api_keys:\n    "": broken\n')
            with patch.dict(os.environ, {'CONFIG_PATH': str(path)}):
                r = self.client.post('/admin/api-keys/reload', headers={'x-api-key': KEY})
        self.assertEqual(r.status_code, 400)
        # The pre-existing registration must survive a rejected reload.
        self.assertEqual(main.API_KEY_TO_TENANT.get(KEY), 'default')
        self.assertEqual(main.API_KEY_TO_TENANT.get('second-tenant-key-abcdef'), 'team-a')

    def test_an_unreadable_source_does_not_lock_everyone_out(self):
        with patch.dict(os.environ, {'CONFIG_PATH': '/nonexistent/config.yaml'}):
            r = self.client.post('/admin/api-keys/reload', headers={'x-api-key': KEY})
        self.assertIn(r.status_code, (400, 503))
        self.assertEqual(main.API_KEY_TO_TENANT.get(KEY), 'default')

    def test_a_source_with_no_keys_is_refused_rather_than_emptying_the_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('auth: {}\n')
            with patch.dict(os.environ, {'CONFIG_PATH': str(path)}):
                r = self.client.post('/admin/api-keys/reload', headers={'x-api-key': KEY})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(main.API_KEY_TO_TENANT.get(KEY), 'default')

    def test_reload_reports_that_env_sourced_values_still_need_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('auth:\n  api_keys:\n    default: ${LB_ROTATION_PROBE_KEY}\n')
            with patch.dict(os.environ, {'CONFIG_PATH': str(path),
                                         'LB_ROTATION_PROBE_KEY': 'env-key-value-13579'}):
                r = self.client.post('/admin/api-keys/reload', headers={'x-api-key': KEY})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body['env_interpolated'])
        self.assertIn('restart', body['note'].lower())

    def test_reload_never_echoes_key_material(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('auth:\n  api_keys:\n    default: rotated-key-value-9876\n')
            with patch.dict(os.environ, {'CONFIG_PATH': str(path)}):
                r = self.client.post('/admin/api-keys/reload', headers={'x-api-key': KEY})
        self.assertNotIn('rotated-key-value-9876', json.dumps(r.json()))


if __name__ == '__main__':
    unittest.main()
