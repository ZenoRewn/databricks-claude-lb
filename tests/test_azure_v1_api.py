"""Azure Foundry requests use the v1 GA surface. Author: Zeno Ren.

Per Microsoft's api-version lifecycle doc (checked 2026-10-13): the v1 GA API
drops the dated `api-version` parameter entirely and takes the deployment name
as `model` in the request body rather than in the URL path. The Responses path
already used /openai/v1/responses; chat completions still carried
?api-version=2024-10-21 with the deployment in the path.

Not verified against the live resource: all four configured Azure endpoints
currently reject their stored keys with 401 and have served zero requests, so
there is nothing live to confirm against. These tests pin the URL shape the
documentation specifies; they do not prove the upstream accepts it.
"""
import re
import unittest
from pathlib import Path

import main

SOURCE = (Path(__file__).resolve().parent.parent / 'main.py').read_text()


def _azure_urls():
    """Every Azure request path, from the single helper the call sites share."""
    return list(main.AZURE_V1_PATHS.values())


class AzureUrlShapeTests(unittest.TestCase):
    def test_every_azure_url_is_under_the_v1_prefix(self):
        urls = _azure_urls()
        self.assertTrue(urls, 'no Azure URL construction found — did the pattern change?')
        for url in urls:
            with self.subTest(url=url):
                self.assertTrue(url.startswith('/openai/v1/'), url)

    def test_no_dated_or_preview_api_version_anywhere(self):
        # A dated or preview version creeping back is the regression this guards.
        offenders = re.findall(r'api-version=[0-9A-Za-z._-]+', SOURCE)
        self.assertEqual(offenders, [], f'dated/preview api-version still present: {offenders}')

    def test_chat_completions_uses_the_v1_path(self):
        self.assertIn('/openai/v1/chat/completions', SOURCE)

    def test_responses_still_uses_the_v1_path(self):
        self.assertIn('/openai/v1/responses', SOURCE)

    def test_the_deployment_name_is_not_in_the_url_path(self):
        # v1 takes the deployment as `model` in the body; a path-embedded
        # deployment means the legacy surface is still in use.
        for url in _azure_urls():
            with self.subTest(url=url):
                self.assertNotIn('/deployments/', url)
                self.assertNotIn('{model}', url)


class AzureUrlBuilderTests(unittest.TestCase):
    """The three call sites share one builder, so version drift cannot be partial."""

    def _endpoint(self, host='https://probe.openai.azure.com'):
        return main.AzureOpenAIEndpoint(name='probe', endpoint=host,
                                        api_key='synthetic-key', deployments=['gpt-5.4'])

    def test_chat_url(self):
        self.assertEqual(main.azure_request_url(self._endpoint(), 'chat'),
                         'https://probe.openai.azure.com/openai/v1/chat/completions')

    def test_responses_url(self):
        self.assertEqual(main.azure_request_url(self._endpoint(), 'responses'),
                         'https://probe.openai.azure.com/openai/v1/responses')

    def test_a_trailing_slash_does_not_double_up(self):
        self.assertEqual(main.azure_request_url(self._endpoint('https://probe.openai.azure.com/'), 'chat'),
                         'https://probe.openai.azure.com/openai/v1/chat/completions')

    def test_the_services_ai_host_form_works_too(self):
        # Microsoft documents both openai.azure.com and services.ai.azure.com;
        # the live config uses both forms.
        url = main.azure_request_url(self._endpoint('https://r.services.ai.azure.com'), 'responses')
        self.assertEqual(url, 'https://r.services.ai.azure.com/openai/v1/responses')

    def test_an_unknown_api_type_is_rejected_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            main.azure_request_url(self._endpoint(), 'embeddings')

    def test_no_url_carries_a_query_string(self):
        for api_type in ('chat', 'responses'):
            with self.subTest(api_type=api_type):
                self.assertNotIn('?', main.azure_request_url(self._endpoint(), api_type))

    def test_every_call_site_uses_the_builder(self):
        # Guards against a fourth site being added with an inline f-string.
        inline = re.findall(r'\{(?:current_)?endpoint\.endpoint\}/openai', SOURCE)
        self.assertEqual(inline, [], 'an Azure URL is still built inline instead of via the builder')


if __name__ == '__main__':
    unittest.main()
