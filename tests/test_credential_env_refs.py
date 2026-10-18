"""An unresolved `${ENV}` credential reference must not load. Author: Zeno Ren.

`expand_env_vars` substitutes a missing variable with the empty string, and
nothing downstream rejects an empty token: the endpoint is built, logged as
«Loaded», and counted in `/config/effective`. The only symptom is 401s from the
upstream, with no line in the config or the logs pointing at the typo.

That was tolerable while every credential sat inline in the ConfigMap, because
no `${}` reference existed to get wrong. Moving them into a Secret makes one
misspelled variable name the expected failure mode, so the gateway has to refuse
to start instead of serving with a blank credential — a blank one is never
correct, and «loud and early» is the only version of this an operator can act on.

Two deliberate exceptions keep their existing behaviour:

- **The Copilot token path.** It falls back to the mounted credential cache and
  already warns that the rotated Secret is *not* in use. That fallback is
  resilience, not an oversight, so it stays — see `resolve_github_token`.
- **A literal empty value.** `auth.api_key` is absent in multi-tenant configs
  and resolves to `""` with no reference to fail on. Only a `${}` that resolves
  to nothing is an error; this is what keeps today's all-inline production
  config loading byte-for-byte unchanged.
"""
import io
import os
import unittest
from unittest.mock import patch

import main


def _load(config: str):
    with patch('builtins.open', return_value=io.StringIO(config)):
        return main.load_config('synthetic.yaml')


ADB = ('endpoints:\n'
       '  - name: synthetic-adb\n'
       '    api_base: https://fixture.invalid\n'
       '    token: {token}\n')


class UnresolvedReferencesAreFatalTests(unittest.TestCase):
    def setUp(self):
        for name in ('LB_TEST_PRESENT', 'LB_TEST_BLANK', 'LB_TEST_MISSING'):
            os.environ.pop(name, None)
        os.environ['LB_TEST_PRESENT'] = 'synthetic-value'
        os.environ['LB_TEST_BLANK'] = '   '
        self.addCleanup(lambda: [os.environ.pop(n, None)
                                 for n in ('LB_TEST_PRESENT', 'LB_TEST_BLANK')])

    def test_a_missing_variable_refuses_to_load(self):
        with self.assertRaises(ValueError) as caught:
            _load(ADB.format(token='${LB_TEST_MISSING}'))
        message = str(caught.exception)
        self.assertIn('LB_TEST_MISSING', message, 'the operator needs the variable name')
        self.assertIn('synthetic-adb', message, 'and which endpoint it belongs to')

    def test_a_blank_variable_refuses_to_load(self):
        # Set-but-empty is the silent-substitution case: the reference looks
        # wired up and the credential is not there.
        with self.assertRaises(ValueError):
            _load(ADB.format(token='${LB_TEST_BLANK}'))

    def test_a_resolved_variable_loads(self):
        proxy, _, _, _ = _load(ADB.format(token='${LB_TEST_PRESENT}'))
        self.assertEqual([e.token for e in proxy.load_balancer.endpoints], ['synthetic-value'])

    def test_an_inline_credential_is_untouched(self):
        # Today's production config is entirely inline. This migration must not
        # change how any of it loads.
        proxy, _, _, _ = _load(ADB.format(token='synthetic-inline'))
        self.assertEqual([e.token for e in proxy.load_balancer.endpoints], ['synthetic-inline'])

    def test_a_partially_resolved_reference_refuses_to_load(self):
        # `prefix-${MISSING}` yields a truncated credential, which reaches the
        # upstream as a 401 with no clue attached.
        with self.assertRaises(ValueError):
            _load(ADB.format(token='prefix-${LB_TEST_MISSING}'))

    def test_the_foundry_key_is_covered(self):
        config = ('azure_anthropic:\n  endpoints:\n    - name: synthetic-foundry\n'
                  '      api_base: https://fixture.invalid\n'
                  '      api_key: ${LB_TEST_MISSING}\n'
                  '      deployments: [claude-opus-5-5]\n')
        with self.assertRaisesRegex(ValueError, 'LB_TEST_MISSING'):
            _load(config)

    def test_the_azure_openai_key_is_covered(self):
        config = ('azure_openai:\n  endpoints:\n    - name: synthetic-azure\n'
                  '      endpoint: https://fixture.invalid\n'
                  '      api_key: ${LB_TEST_MISSING}\n'
                  '      deployments: [gpt-5.4]\n')
        with self.assertRaisesRegex(ValueError, 'LB_TEST_MISSING'):
            _load(config)

    def test_the_client_api_key_is_covered(self):
        with self.assertRaisesRegex(ValueError, 'LB_TEST_MISSING'):
            _load('auth:\n  api_key: ${LB_TEST_MISSING}\n')

    def test_a_tenant_api_key_is_covered(self):
        with self.assertRaisesRegex(ValueError, 'LB_TEST_MISSING'):
            _load('auth:\n  api_keys:\n    monitor: ${LB_TEST_MISSING}\n')

    def test_the_usage_storage_password_is_covered(self):
        config = ('usage_storage:\n  type: mysql\n  host: fixture.invalid\n'
                  '  password: ${LB_TEST_MISSING}\n')
        with self.assertRaisesRegex(ValueError, 'LB_TEST_MISSING'):
            _load(config)

    def test_an_absent_client_key_is_not_a_reference(self):
        # No `${}` to fail on, so multi-tenant-only configs keep working.
        proxy, _, _, _ = _load('auth:\n  api_keys:\n    monitor: ${LB_TEST_PRESENT}\n')
        self.assertIsNotNone(proxy)

    def test_the_message_never_contains_a_resolved_value(self):
        config = (ADB.format(token='${LB_TEST_PRESENT}')
                  + 'azure_openai:\n  endpoints:\n    - name: synthetic-azure\n'
                    '      endpoint: https://fixture.invalid\n'
                    '      api_key: ${LB_TEST_MISSING}\n')
        with self.assertRaises(ValueError) as caught:
            _load(config)
        self.assertNotIn('synthetic-value', str(caught.exception),
                         'a diagnostic must not print a credential it did resolve')


class TheCopilotFallbackIsPreservedTests(unittest.TestCase):
    """`resolve_github_token` keeps its cache fallback and its warning.

    It is the one path with somewhere else to get a credential from, and it
    already says out loud that the rotated Secret is not in use. Making it fatal
    would remove resilience that was added on purpose.
    """

    def test_an_unresolved_copilot_token_falls_back_instead_of_raising(self):
        os.environ.pop('LB_TEST_MISSING', None)
        config = ('github_copilot:\n  endpoints:\n    - name: synthetic-gh\n'
                  '      github_token: ${LB_TEST_MISSING}\n'
                  '      api_types: [responses]\n')
        with patch.object(main, 'resolve_github_token',
                          return_value=('synthetic-cached', {'type': 'file', 'path': 'x'})):
            _, _, copilot, _ = _load(config)
        self.assertEqual([e.github_token for e in copilot.load_balancer.endpoints],
                         ['synthetic-cached'])


if __name__ == '__main__':
    unittest.main()
