"""The Anthropic Messages surface must obey the account layers. Author: Zeno Ren.

Found in production on 2026-10-19 while taking evidence for the GHCP messages
hop. A request forced onto `copilot` landed on **`zenoren-msft`** — an account
the operator left inert (`enabled: false` in config, never explicitly enabled)
— while `/admin/model-routing/claude-opus-5-5` reported that hop as
`serving=1/5` with `zenoren-icloud` as the only candidate. The diagnostic and
the runtime disagreed, which CLAUDE.md names as worse than having no
diagnostic.

Root cause: the three Messages pools are all `ClaudeProxy`, and it selects via
`LoadBalancer.select_endpoint()`, whose only compatibility test is the
**config-declared** `models` list. Copilot accounts declare `models: []`
(= all), so for that pool the predicate is vacuously true and all four
account-level layers are skipped:

    account enablement · allowed_models · per-model account selection · capability

Those live in `CopilotEndpoint.supports()` / `_copilot_support_reason()`, which
`select_endpoint` never calls. CLAUDE.md: «筛选只能加在 supports()：那是所有
候选判断的唯一收口».

The same walk surfaced a second defect in the same line of code: `inference_call`
is handed the literal `'databricks'`, so every Foundry and GHCP-messages attempt
is labelled `provider="databricks"` in the diagnostic events **and** in
`lb_upstream_send_started_total`. The usage ledger is unaffected — that record
point was fixed on 2026-10-18 and uses `self.provider` — so this is the same
invariant failing at a second record point: the one an operator reads when
asking «which path served this request». The event that gave it away carried
`provider: databricks` next to `endpoint_alias: zenoren-msft`.
"""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

import key_policy
import main
import request_telemetry
from request_telemetry import TELEMETRY


def _lb(*endpoints):
    return main.LoadBalancer(list(endpoints))


class AccountLayersApplyToTheMessagesSurfaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        for name in ('gh-live', 'gh-inert'):
            await self.store.set_account_capability(name, ('claude-opus-5.5',))
        patcher = patch.object(main, 'key_policy_store', self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.live = main.CopilotEndpoint(name='gh-live', github_token='synthetic')
        self.inert = main.CopilotEndpoint(name='gh-inert', github_token='synthetic')
        # «Deployed but not in service», exactly the production shape.
        self.inert.enabled_in_config = False

    async def test_a_never_enabled_account_is_not_selected(self):
        await self.store.set_account('gh-live', enabled=True)
        picked = {_lb(self.inert, self.live).select_endpoint(model='claude-opus-5.5').name
                  for _ in range(20)}
        self.assertEqual(picked, {'gh-live'},
                         'an inert account must not serve the messages surface')

    async def test_an_account_outside_the_per_model_selection_is_not_selected(self):
        await self.store.set_account('gh-live', enabled=True)
        await self.store.set_account('gh-inert', enabled=True)
        self.inert.enabled_in_config = True
        await self.store.set_model_accounts('claude-opus-5.5', ['gh-live'])
        picked = {_lb(self.inert, self.live).select_endpoint(model='claude-opus-5.5').name
                  for _ in range(20)}
        self.assertEqual(picked, {'gh-live'})

    async def test_an_account_whose_allowed_models_excludes_it_is_not_selected(self):
        await self.store.set_account('gh-live', enabled=True)
        await self.store.set_account('gh-inert', enabled=True, allowed_models=['claude-opus-5.5'])
        self.inert.enabled_in_config = True
        await self.store.set_account('gh-live', enabled=True, allowed_models=['claude-opus-5.5'])
        # Now exclude the model from one account only.
        await self.store.set_account_capability('gh-inert', ('claude-opus-5.5', 'claude-haiku-5.5'))
        await self.store.set_account('gh-inert', allowed_models=['claude-haiku-5.5'])
        picked = {_lb(self.inert, self.live).select_endpoint(model='claude-opus-5.5').name
                  for _ in range(20)}
        self.assertEqual(picked, {'gh-live'})

    async def test_an_account_without_the_capability_is_not_selected(self):
        # The production case: the enterprise seat serves zero Claude models.
        await self.store.set_account('gh-live', enabled=True)
        await self.store.set_account('gh-inert', enabled=True)
        self.inert.enabled_in_config = True
        await self.store.set_account_capability('gh-inert', ('gpt-5.4',))
        picked = {_lb(self.inert, self.live).select_endpoint(model='claude-opus-5.5').name
                  for _ in range(20)}
        self.assertEqual(picked, {'gh-live'})

    async def test_no_eligible_account_leaves_no_candidate(self):
        # Must be «none permit this model», which is the condition
        # `_messages_failover_allowed` lets through to the next provider.
        self.live.enabled_in_config = False
        self.assertIsNone(_lb(self.inert, self.live).select_endpoint(model='claude-opus-5.5'))

    async def test_a_databricks_endpoint_keeps_the_config_models_behaviour(self):
        # WorkspaceEndpoint has no supports(); the existing predicate must stay.
        wide = main.WorkspaceEndpoint('adb-any', 'https://fixture.invalid', 'synthetic')
        narrow = main.WorkspaceEndpoint('adb-narrow', 'https://fixture.invalid', 'synthetic',
                                        models=['databricks-claude-opus-5'])
        lb = _lb(wide, narrow)
        self.assertEqual({lb.select_endpoint(model='databricks-claude-opus-5-5').name
                          for _ in range(20)}, {'adb-any'})
        self.assertEqual({lb.select_endpoint(model='databricks-claude-opus-5').name
                          for _ in range(20)}, {'adb-any', 'adb-narrow'})


class NoEligibleAccountFallsOverTests(unittest.IsolatedAsyncioTestCase):
    """An empty Copilot candidate set must hand the request to the next provider.

    `_messages_failover_allowed` admits 404 `unsupported_model`, so the pool has
    to raise that rather than let the client see it — a parked account set is not
    an answer about the request.
    """

    async def asyncSetUp(self):
        self.store = key_policy.KeyPolicyStore(None)
        await self.store.start()
        patcher = patch.object(main, 'key_policy_store', self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_the_pool_raises_a_failover_eligible_error(self):
        inert = main.CopilotEndpoint(name='gh-inert', github_token='synthetic')
        inert.enabled_in_config = False
        pool = main.ClaudeProxy(_lb(inert), 'synthetic', provider='copilot')
        self.addCleanup(lambda: asyncio.get_event_loop().create_task(pool.close()))
        with self.assertRaises(main.HTTPException) as caught:
            await pool.proxy_request({'model': 'claude-opus-5.5', 'messages': []}, stream=False)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertTrue(main._messages_failover_allowed(caught.exception),
                        'a parked account set must not end the request')


class AttemptsAreAttributedToTheirOwnProviderTests(unittest.IsolatedAsyncioTestCase):
    """`inference_call` was handed the literal 'databricks' by every pool."""

    async def _send(self, provider):
        ep = main.WorkspaceEndpoint('fixture', 'https://fixture.invalid', 'synthetic')
        proxy = main.ClaudeProxy(_lb(ep), 'synthetic', provider=provider)
        await proxy.client.aclose()
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'type': 'message',
                                                      'stop_reason': 'end_turn', 'usage': {}})))
        before = TELEMETRY.sends[(provider, 'messages')]
        try:
            with patch.object(main, 'usage_store', None):
                await proxy.proxy_request({'model': 'claude-opus-5.5', 'messages': []}, stream=False)
        finally:
            await proxy.close()
        return before, TELEMETRY.sends[(provider, 'messages')]

    async def test_foundry_is_attributed_to_azure_anthropic(self):
        before, after = await self._send('azure_anthropic')
        self.assertEqual(after, before + 1,
                         'a Foundry attempt counted as databricks hides the path that served it')

    async def test_copilot_messages_are_attributed_to_copilot(self):
        before, after = await self._send('copilot')
        self.assertEqual(after, before + 1)

    async def test_databricks_stays_databricks(self):
        before, after = await self._send('databricks')
        self.assertEqual(after, before + 1)

    async def test_the_event_names_the_provider_that_served_it(self):
        events = []
        ep = main.WorkspaceEndpoint('eastus2-foundry', 'https://fixture.invalid', 'synthetic')
        proxy = main.ClaudeProxy(_lb(ep), 'synthetic', provider='azure_anthropic')
        await proxy.client.aclose()
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'type': 'message',
                                                      'stop_reason': 'end_turn', 'usage': {}})))
        try:
            # `log_event` is emitted from request_telemetry, not from main —
            # patching the wrong module silently observes nothing.
            with patch.object(main, 'usage_store', None), \
                 patch.object(request_telemetry, 'log_event',
                              side_effect=lambda e: events.append(e)):
                await proxy.proxy_request({'model': 'claude-opus-5-5', 'messages': []}, stream=False)
        finally:
            await proxy.close()
        sends = [e for e in events if e.get('kind') == 'lb_upstream_send_start']
        self.assertTrue(sends, 'the attempt must emit a send_start event')
        self.assertEqual({e['provider'] for e in sends}, {'azure_anthropic'},
                         'the event must not contradict its own endpoint_alias')


class TheRetractedClaimStaysRetractedTests(unittest.TestCase):
    """«/v1/messages 仍然只走 Databricks» was disproved on 2026-10-17.

    It survived in CLAUDE.md and docs/CONFIGURATION.md for two more rounds while
    the same files carried the three-paths invariant a few lines later — one of
    the two had to be wrong, and an agent reading the stale one would treat a
    shipped capability as nonexistent.

    The phrase may still appear **as a quotation inside a retraction** (that is
    how this repository records what it withdrew and why); what it must not do
    is appear as a current claim. So each occurrence has to sit on a line that
    also says it was withdrawn.
    """

    RETRACTION = ('推翻', '撤回', '早先', '不再成立')

    def test_no_file_asserts_it_as_current(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        for name in ('CLAUDE.md', 'docs/CONFIGURATION.md', 'README.md', 'docs/AKS.md'):
            for number, line in enumerate((root / name).read_text().splitlines(), 1):
                if '只走 Databricks' not in line and '只走 ADB' not in line:
                    continue
                with self.subTest(file=name, line=number):
                    self.assertTrue(any(word in line for word in self.RETRACTION),
                                    f'{name}:{number} states it as a current claim')


if __name__ == '__main__':
    unittest.main()
