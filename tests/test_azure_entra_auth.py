"""Azure OpenAI endpoints can authenticate with Entra ID. Author: Zeno Ren.

A governance policy disabled key auth (`disableLocalAuth`) on every Azure AI
account behind the `azure_openai` pool, so every stored key now answers 401.
`auth: aad` on an endpoint swaps the `api-key` header for a workload-identity
bearer token; endpoints without it keep the key path byte-for-byte.
"""
import asyncio
import os
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock, patch

import main


class AzureAuthHeaderTests(unittest.TestCase):
    def test_key_endpoint_keeps_api_key_header(self):
        ep = main.AzureOpenAIEndpoint('k', 'https://k.invalid', 'secret')
        self.assertEqual(asyncio.run(main.azure_auth_headers(ep)), {'api-key': 'secret', 'Content-Type': 'application/json'})

    def test_aad_endpoint_sends_bearer_token_and_no_key(self):
        ep = main.AzureOpenAIEndpoint('a', 'https://a.invalid', '', auth='aad')
        cred = MagicMock()
        cred.get_token.return_value.token = 'tok-aad'
        with patch.object(main, '_azure_credential', return_value=cred):
            headers = asyncio.run(main.azure_auth_headers(ep))
        self.assertEqual(headers, {'Authorization': 'Bearer tok-aad', 'Content-Type': 'application/json'})
        cred.get_token.assert_called_once_with('https://cognitiveservices.azure.com/.default')


class AzureAuthConfigTests(unittest.TestCase):
    def _load(self, endpoint_yaml):
        cfg = textwrap.dedent('''\
            endpoints: []
            azure_openai:
              endpoints:
            ''') + textwrap.indent(textwrap.dedent(endpoint_yaml), '    ')
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.write(cfg)
        try:
            return main.load_config(f.name)
        finally:
            os.unlink(f.name)

    def _azure_endpoint(self, loaded):
        for item in loaded:
            lb = getattr(item, 'load_balancer', None)
            if lb is not None and lb.endpoints and isinstance(lb.endpoints[0], main.AzureOpenAIEndpoint):
                return lb.endpoints[0]
        self.fail('azure proxy not built')

    def test_aad_endpoint_loads_without_api_key(self):
        ep = self._azure_endpoint(self._load('''\
            - name: e
              endpoint: https://e.invalid
              auth: aad
              deployments: [gpt-5.4]
            '''))
        self.assertEqual((ep.auth, ep.api_key), ('aad', ''))

    def test_unknown_auth_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._load('''\
                - name: e
                  endpoint: https://e.invalid
                  auth: magic
                ''')


if __name__ == '__main__':
    unittest.main()
