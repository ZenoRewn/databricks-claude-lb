"""Failover across the three Anthropic Messages providers. Author: Zeno Ren.

Databricks, Copilot and Foundry all serve Claude over this protocol, so
`/v1/messages` can fall back between them — but only where falling back is
*safe*. The no-replay invariant is unchanged: a provider may be retried only
when this provider could not serve the request at all, and never once content
may have been produced.

Concretely safe to fail over:
  * every endpoint of that provider is circuit-open / none configured (503)
  * the provider has no endpoint permitting the model (404 unsupported_model)
Concretely NOT safe:
  * any 4xx the upstream produced from the request itself — the next provider
    would reject it identically, and a 400 is an answer, not an outage
  * anything after a streaming response has been handed downstream, which the
    orchestrator cannot even observe: by then the generator owns the request

Cross-provider failover is sound for this surface because Anthropic Messages
carries its whole conversation in `messages`. There is no `previous_response_id`
to strand, unlike the OpenAI-style surface where opaque state pins a request to
one account. Prompt-cache breakpoints do not survive the hop, which costs money
but cannot corrupt an answer.
"""
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import main


def endpoint(name='synthetic', **kwargs):
    return main.WorkspaceEndpoint(name, 'https://fixture.invalid', 'token', **kwargs)


class AuthHeaderTests(unittest.TestCase):
    def test_databricks_endpoints_keep_bearer(self):
        headers = main._messages_auth_headers(endpoint())
        self.assertEqual(headers['Authorization'], 'Bearer token')
        self.assertEqual(headers['Content-Type'], 'application/json')
        self.assertNotIn('anthropic-version', headers)

    def test_foundry_endpoints_send_x_api_key_not_api_key(self):
        # Verified 2026-10-17: Foundry answers `api-key` with 401 even though
        # its documentation lists that header as acceptable.
        headers = main._messages_auth_headers(
            endpoint(provider='azure_anthropic', auth_style='x-api-key',
                     anthropic_version='2023-06-01'))
        self.assertEqual(headers['x-api-key'], 'token')
        self.assertNotIn('api-key', headers)
        self.assertNotIn('Authorization', headers)
        self.assertEqual(headers['anthropic-version'], '2023-06-01')

    def test_an_unsupported_auth_style_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            endpoint(auth_style='basic')


class FailoverPredicateTests(unittest.TestCase):
    def test_provider_wide_unavailability_may_fail_over(self):
        for status, detail in ((503, {'error': {'code': 'lb_unavailable'}}),
                               (404, {'error': {'code': 'unsupported_model'}})):
            with self.subTest(status=status):
                self.assertTrue(main._messages_failover_allowed(
                    HTTPException(status_code=status, detail=detail)))

    def test_a_request_level_rejection_must_not_fail_over(self):
        # The next provider would reject it the same way, and a 400 is an
        # answer. Retrying would turn one honest error into three.
        for status in (400, 401, 403, 413, 422):
            with self.subTest(status=status):
                self.assertFalse(main._messages_failover_allowed(
                    HTTPException(status_code=status, detail={'error': {'message': 'nope'}})))

    def test_an_upstream_failure_that_may_have_executed_must_not_fail_over(self):
        # 502/504 mean the upstream was reached. Execution state is unknown,
        # which is exactly what the no-replay invariant guards.
        for status in (500, 502, 504):
            with self.subTest(status=status):
                self.assertFalse(main._messages_failover_allowed(
                    HTTPException(status_code=status, detail={'error': {'message': 'x'}})))

    def test_a_503_that_names_an_upstream_is_not_provider_wide(self):
        # «this endpoint is unhealthy» already has per-endpoint retry inside
        # the pool. Only «no candidate at all» justifies leaving the provider.
        self.assertFalse(main._messages_failover_allowed(HTTPException(
            status_code=503, detail={'error': {'code': 'upstream_unavailable',
                                               'message': 'upstream returned 503'}})))


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.preference = ()
        store = type('Store', (), {
            'model_preference': lambda _s, model: self.preference,
            'model_accounts_exclusion_reason': lambda *a: None,
        })()
        patcher = patch.object(main, 'key_policy_store', store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def pool(self, provider, outcome):
        """A stand-in Anthropic Messages pool recording what it was asked."""
        calls = self.calls

        class Pool:
            def __init__(self):
                self.provider = provider

            async def proxy_request(self, body, stream=False, request_id=None):
                calls.append((provider, body['model']))
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome
        return Pool()

    async def test_the_first_provider_that_answers_wins(self):
        pools = {'databricks': self.pool('databricks', 'adb-result')}
        result = await main._serve_messages_across_providers(
            {'model': 'claude-opus-5-5'}, pools, stream=False, request_id='r')
        self.assertEqual(result, 'adb-result')
        self.assertEqual(self.calls, [('databricks', 'databricks-claude-opus-5-5')])

    async def test_an_unavailable_provider_hands_over_with_the_model_rewritten(self):
        # The whole point: Foundry rejects `claude-opus-5.5` with 404, so the
        # hop has to respell the model for its new host.
        pools = {
            'databricks': self.pool('databricks', HTTPException(
                status_code=503, detail={'error': {'code': 'lb_unavailable'}})),
            'azure_anthropic': self.pool('azure_anthropic', 'foundry-result'),
        }
        result = await main._serve_messages_across_providers(
            {'model': 'claude-opus-5-5'}, pools, stream=False, request_id='r')
        self.assertEqual(result, 'foundry-result')
        self.assertEqual(self.calls, [('databricks', 'databricks-claude-opus-5-5'),
                                      ('azure_anthropic', 'claude-opus-5-5')])

    async def test_a_request_level_rejection_stops_the_chain(self):
        rejection = HTTPException(status_code=400, detail={'error': {'message': 'bad'}})
        pools = {'databricks': self.pool('databricks', rejection),
                 'azure_anthropic': self.pool('azure_anthropic', 'never')}
        with self.assertRaises(HTTPException) as caught:
            await main._serve_messages_across_providers(
                {'model': 'claude-opus-5-5'}, pools, stream=False, request_id='r')
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(len(self.calls), 1, 'the second provider must not be asked')

    async def test_the_configured_order_is_honoured(self):
        self.preference = ('azure_anthropic',)
        pools = {'databricks': self.pool('databricks', 'adb'),
                 'azure_anthropic': self.pool('azure_anthropic', 'foundry')}
        result = await main._serve_messages_across_providers(
            {'model': 'claude-opus-5-5'}, pools, stream=False, request_id='r')
        self.assertEqual(result, 'foundry')
        self.assertEqual(self.calls[0][0], 'azure_anthropic')

    async def test_a_provider_without_a_pool_is_skipped_silently(self):
        pools = {'azure_anthropic': self.pool('azure_anthropic', 'foundry')}
        result = await main._serve_messages_across_providers(
            {'model': 'claude-opus-5-5'}, pools, stream=False, request_id='r')
        self.assertEqual(result, 'foundry')

    async def test_the_last_failure_surfaces_when_every_provider_declines(self):
        unavailable = HTTPException(status_code=503, detail={'error': {'code': 'lb_unavailable'}})
        pools = {'databricks': self.pool('databricks', unavailable),
                 'azure_anthropic': self.pool('azure_anthropic', unavailable)}
        with self.assertRaises(HTTPException) as caught:
            await main._serve_messages_across_providers(
                {'model': 'claude-opus-5-5'}, pools, stream=False, request_id='r')
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(len(self.calls), 2)

    async def test_a_non_claude_model_is_not_respelled(self):
        pools = {'databricks': self.pool('databricks', 'ok')}
        await main._serve_messages_across_providers(
            {'model': 'databricks-meta-llama-3'}, pools, stream=False, request_id='r')
        self.assertEqual(self.calls, [('databricks', 'databricks-meta-llama-3')])

    async def test_each_provider_sees_its_own_body_copy(self):
        """One pool's edits must not reach the next pool.

        Each pool normalises the payload for its own upstream, so they must not
        share a dict — otherwise the second provider inherits the first's
        reduction and the decision stops being per-provider. The field used
        here is one nobody strips; which fields get stripped is a separate
        question, settled per upstream by evidence.
        """
        seen = {}

        class Mutating:
            provider = 'databricks'

            async def proxy_request(self, body, stream=False, request_id=None):
                body.pop('metadata', None)
                seen['databricks'] = dict(body)
                raise HTTPException(status_code=503, detail={'error': {'code': 'lb_unavailable'}})

        class Recording:
            provider = 'azure_anthropic'

            async def proxy_request(self, body, stream=False, request_id=None):
                seen['azure_anthropic'] = dict(body)
                return 'ok'

        await main._serve_messages_across_providers(
            {'model': 'claude-opus-5-5', 'metadata': {'user_id': 'u'}},
            {'databricks': Mutating(), 'azure_anthropic': Recording()},
            stream=False, request_id='r')
        self.assertNotIn('metadata', seen['databricks'])
        self.assertIn('metadata', seen['azure_anthropic'],
                      'one pool\'s edits must not leak into the next')


class NormalisationScopeTests(unittest.IsolatedAsyncioTestCase):
    """Which providers get the extra-field strip, and why.

    I first assumed Foundry needed none of it, being an Anthropic-native
    surface. A live call on 2026-10-17 disproved that: with
    `context_management` present Foundry answered
    `400 invalid_request_error: context_management: Extra inputs are not
    permitted`. So it is not a superset of Databricks, and the conservative
    choice is to strip for both until a field is verified acceptable.
    """

    def test_foundry_is_normalised_too(self):
        self.assertIn('azure_anthropic', main._NORMALISING_MESSAGES_PROVIDERS)
        self.assertIn('databricks', main._NORMALISING_MESSAGES_PROVIDERS)

    def test_context_management_is_stripped_for_both(self):
        for provider in ('databricks', 'azure_anthropic'):
            with self.subTest(provider=provider):
                pool = main.ClaudeProxy.__new__(main.ClaudeProxy)
                pool.provider = provider
                body = {'model': 'claude-opus-5-5', 'context_management': {'edits': []},
                        'messages': [{'role': 'user', 'content': 'hi'}]}
                pool._normalize_for_databricks(body)
                self.assertNotIn('context_management', body)

    def test_copilot_is_not_in_this_set(self):
        # Copilot's /v1/messages is reached through its own client (session
        # token, editor headers); its accepted schema is a separate question.
        self.assertNotIn('copilot', main._NORMALISING_MESSAGES_PROVIDERS)


if __name__ == '__main__':
    unittest.main()
