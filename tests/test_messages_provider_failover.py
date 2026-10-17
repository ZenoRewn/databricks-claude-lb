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
import asyncio
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

    def test_all_three_providers_are_normalised(self):
        # Each verified by a live call on 2026-10-17: Foundry and Copilot both
        # answered `400 invalid_request_error: context_management: Extra inputs
        # are not permitted`. Anthropic-native does not imply a superset of
        # Databricks.
        for provider in ('databricks', 'azure_anthropic', 'copilot'):
            with self.subTest(provider=provider):
                self.assertIn(provider, main._NORMALISING_MESSAGES_PROVIDERS)

    def test_context_management_is_stripped_for_both(self):
        for provider in ('databricks', 'azure_anthropic'):
            with self.subTest(provider=provider):
                pool = main.ClaudeProxy.__new__(main.ClaudeProxy)
                pool.provider = provider
                body = {'model': 'claude-opus-5-5', 'context_management': {'edits': []},
                        'messages': [{'role': 'user', 'content': 'hi'}]}
                pool._normalize_for_databricks(body)
                self.assertNotIn('context_management', body)

    def test_context_management_is_stripped_for_copilot_too(self):
        pool = main.ClaudeProxy.__new__(main.ClaudeProxy)
        pool.provider = 'copilot'
        body = {'model': 'claude-opus-5.5', 'context_management': {'edits': []},
                'messages': [{'role': 'user', 'content': 'hi'}]}
        pool._normalize_for_databricks(body)
        self.assertNotIn('context_management', body)


if __name__ == '__main__':
    unittest.main()


class CopilotMessagesPoolTests(unittest.IsolatedAsyncioTestCase):
    """Copilot as the third Anthropic Messages upstream.

    Verified live on 2026-10-17 against a business-seat account:
    `POST {session_base_url}/v1/messages` returns a standard Anthropic body,
    and `stream: true` returns standard Anthropic SSE. Only the target differs
    from Databricks, so only `_upstream_target` is overridden.
    """

    def pool(self):
        owner = main.CopilotProxy.__new__(main.CopilotProxy)
        owner.load_balancer = main.LoadBalancer([
            main.CopilotEndpoint('biz', '', models=['claude-opus-5.5'],
                                 session_token='sess', session_token_expires_at=2 ** 31)])
        owner.api_key = 'local'
        owner.client = object()
        owner.get_session_token = lambda endpoint, force=False: _resolved('session-token')
        return main.CopilotMessagesProxy(owner), owner

    async def test_the_path_has_no_anthropic_prefix(self):
        # Databricks and Foundry serve /anthropic/v1/messages; Copilot serves
        # /v1/messages. Getting this wrong is a 404 on every request.
        pool, owner = self.pool()
        endpoint = owner.load_balancer.endpoints[0]
        endpoint.session_base_url = 'https://api.business.githubcopilot.com'
        url, _ = await pool._upstream_target(endpoint)
        self.assertEqual(url, 'https://api.business.githubcopilot.com/v1/messages')
        self.assertNotIn('/anthropic/', url)

    async def test_the_session_token_and_editor_headers_are_sent(self):
        # Without the editor headers GitHub answers 403 «Please only use
        # approved clients for Copilot».
        pool, owner = self.pool()
        _, headers = await pool._upstream_target(owner.load_balancer.endpoints[0])
        self.assertEqual(headers['Authorization'], 'Bearer session-token')
        for header in ('Editor-Version', 'Editor-Plugin-Version',
                       'Copilot-Integration-Id', 'User-Agent'):
            with self.subTest(header=header):
                self.assertIn(header, headers)

    async def test_the_vision_header_follows_the_shared_detector(self):
        pool, owner = self.pool()
        endpoint = owner.load_balancer.endpoints[0]
        image = {'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'data': 'x'}}]}]}
        token = main._MESSAGES_BODY.set(image)
        self.addCleanup(main._MESSAGES_BODY.reset, token)
        _, headers = await pool._upstream_target(endpoint)
        self.assertEqual(headers.get('Copilot-Vision-Request'), 'true')

    async def test_no_vision_header_without_an_image(self):
        pool, owner = self.pool()
        token = main._MESSAGES_BODY.set({'messages': [{'role': 'user', 'content': 'hi'}]})
        self.addCleanup(main._MESSAGES_BODY.reset, token)
        _, headers = await pool._upstream_target(owner.load_balancer.endpoints[0])
        self.assertNotIn('Copilot-Vision-Request', headers)

    async def test_concurrent_requests_do_not_see_each_other_bodies(self):
        """The reason this is a ContextVar and not an attribute.

        One pool instance serves concurrent requests. With an attribute, two
        of them clobber each other and one request can be sent with the vision
        header derived from the other's payload.
        """
        pool, owner = self.pool()
        endpoint = owner.load_balancer.endpoints[0]
        image = {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {}}]}]}
        plain = {'messages': [{'role': 'user', 'content': 'hi'}]}

        async def target(body):
            token = main._MESSAGES_BODY.set(body)
            try:
                await asyncio.sleep(0)
                _, headers = await pool._upstream_target(endpoint)
                return 'Copilot-Vision-Request' in headers
            finally:
                main._MESSAGES_BODY.reset(token)

        with_image, without = await asyncio.gather(target(image), target(plain))
        self.assertTrue(with_image)
        self.assertFalse(without)

    def test_the_pool_shares_the_owners_balancer_and_client(self):
        # Two views of one upstream: circuit state and h2 retirement must be
        # the same objects, not parallel copies.
        pool, owner = self.pool()
        self.assertIs(pool.load_balancer, owner.load_balancer)
        self.assertIs(pool.client, owner.client)
        self.assertEqual(pool.provider, 'copilot')

    def test_the_owner_exposes_one_cached_pool(self):
        owner = main.CopilotProxy.__new__(main.CopilotProxy)
        owner.load_balancer = main.LoadBalancer([])
        owner.api_key = 'local'
        owner.client = object()
        self.assertIs(owner.messages_pool, owner.messages_pool)


def _resolved(value):
    async def _coro():
        return value
    return _coro()


class CredentialReprTests(unittest.TestCase):
    """A credential must not be reachable through `repr()`.

    Found the hard way on 2026-10-17: a read-only diagnostic printed a
    `CopilotEndpoint` and its dataclass repr carried the long-lived OAuth token
    in plain text. Any `logger.info(f"{endpoint}")`, any exception whose
    traceback renders locals, any `print` in a debugging session would have
    done the same — against a repository whose stated invariant is that
    credentials never enter diagnostics.
    """

    def test_a_databricks_token_is_not_in_the_repr(self):
        text = repr(main.WorkspaceEndpoint('n', 'https://fixture.invalid', 'SECRET-DBX'))
        self.assertNotIn('SECRET-DBX', text)
        self.assertIn('n', text, 'the endpoint must still be identifiable')

    def test_a_copilot_token_is_not_in_the_repr(self):
        self.assertNotIn('SECRET-GH', repr(main.CopilotEndpoint('n', 'SECRET-GH')))

    def test_a_foundry_key_is_not_in_the_repr(self):
        self.assertNotIn('SECRET-AZ', repr(main.WorkspaceEndpoint(
            'f', 'https://fixture.invalid', 'SECRET-AZ',
            provider='azure_anthropic', auth_style='x-api-key')))

    def test_the_credential_is_still_usable_despite_being_hidden(self):
        # repr=False hides it from diagnostics, it does not remove the field.
        endpoint = main.WorkspaceEndpoint('n', 'https://fixture.invalid', 'SECRET-DBX')
        self.assertEqual(main._messages_auth_headers(endpoint)['Authorization'],
                         'Bearer SECRET-DBX')


class DatabricksModelConversionScopeTests(unittest.TestCase):
    """Only Databricks renames the model during normalisation.

    Live-verified bug, 2026-10-17: the shared normalisation began with
    `get_databricks_model()`, so a Copilot request for `claude-opus-5.5` was
    rewritten into Databricks' spelling and then matched no Copilot endpoint —
    surfacing as 404 `unsupported_model` against an account that serves the
    model perfectly well. Cross-provider respelling belongs to the
    orchestrator, which already does it via `claude_model_identity`.
    """

    def normalise(self, provider, model):
        pool = main.ClaudeProxy.__new__(main.ClaudeProxy)
        pool.provider = provider
        body = {'model': model, 'messages': [{'role': 'user', 'content': 'hi'}]}
        pool._normalize_for_databricks(body)
        return body['model']

    def test_copilot_keeps_its_own_spelling(self):
        self.assertEqual(self.normalise('copilot', 'claude-opus-5.5'), 'claude-opus-5.5')

    def test_foundry_keeps_its_own_spelling(self):
        self.assertEqual(self.normalise('azure_anthropic', 'claude-opus-5-5'), 'claude-opus-5-5')

    def test_databricks_still_converts(self):
        # Whatever get_databricks_model does for this alias, it must still run.
        self.assertEqual(self.normalise('databricks', 'claude-opus-5-5'),
                         main.get_databricks_model('claude-opus-5-5'))
