"""Per-surface path visibility for one model. Author: Zeno Ren.

The routing workbench showed only the OpenAI-style order, so ADB never
appeared anywhere and an operator could not tell whether that was deliberate
or forgotten. Now each protocol surface reports its own ordered paths, and a
provider that cannot serve the surface says so instead of being absent.
"""
import asyncio
import unittest
from unittest.mock import patch

import main


class MessagesPathViewTests(unittest.TestCase):
    def setUp(self):
        store = type('Store', (), {
            'model_preference': lambda _s, m: (),
            'model_accounts': lambda _s, m: (),
            'model_accounts_exclusion_reason': lambda *a: None,
        })()
        patcher = patch.object(main, 'key_policy_store', store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def paths(self, model='claude-opus-5-5'):
        return {p['provider']: p for p in main._messages_path_views(model)}

    def test_every_messages_provider_is_listed_even_when_unconfigured(self):
        # Absence is what made ADB look forgotten. A provider that cannot serve
        # must say why, not vanish.
        paths = self.paths()
        self.assertEqual(set(paths), {'databricks', 'copilot', 'azure_anthropic'})

    def test_each_path_carries_its_operator_facing_label(self):
        labels = {p['provider']: p['label'] for p in main._messages_path_views('claude-opus-5-5')}
        self.assertEqual(labels['databricks'], 'ADB')
        self.assertEqual(labels['copilot'], 'GHCP')
        self.assertEqual(labels['azure_anthropic'], 'Foundry Claude')

    def test_an_unconfigured_provider_names_that_reason(self):
        with patch.object(main, 'proxy', None), patch.object(main, 'copilot_proxy', None):
            paths = self.paths()
        for provider in ('databricks', 'copilot', 'azure_anthropic'):
            with self.subTest(provider=provider):
                self.assertFalse(paths[provider]['configured'])
                self.assertEqual(paths[provider]['reason'], 'provider_not_configured')

    def test_the_model_spelling_is_shown_per_provider(self):
        # The operator needs to see that one model is named three ways,
        # because a Foundry deployment list must use Foundry's spelling.
        paths = self.paths('claude-opus-5-5')
        self.assertEqual(paths['databricks']['forwarded_model'], 'databricks-claude-opus-5-5')
        self.assertEqual(paths['copilot']['forwarded_model'], 'claude-opus-5.5')
        self.assertEqual(paths['azure_anthropic']['forwarded_model'], 'claude-opus-5-5')

    def test_a_non_claude_model_keeps_one_spelling(self):
        paths = self.paths('databricks-meta-llama-3')
        for provider in paths:
            with self.subTest(provider=provider):
                self.assertEqual(paths[provider]['forwarded_model'], 'databricks-meta-llama-3')

    def test_the_order_follows_the_effective_preference(self):
        ordered = [p['provider'] for p in main._messages_path_views('claude-opus-5-5')]
        self.assertEqual(ordered, ['databricks', 'copilot', 'azure_anthropic'])

    def test_a_configured_provider_with_no_permitting_endpoint_says_so(self):
        pool = type('P', (), {'load_balancer': type('L', (), {'endpoints': [
            main.WorkspaceEndpoint('adb', 'https://fixture.invalid', 't', models=['gpt-x'])]})()})()
        with patch.object(main, 'proxy', pool):
            paths = self.paths()
        self.assertTrue(paths['databricks']['configured'])
        self.assertEqual(paths['databricks']['endpoints_serving_model'], 0)
        self.assertEqual(paths['databricks']['reason'], 'no_endpoint_permits_model')

    def test_a_serving_provider_has_no_reason(self):
        pool = type('P', (), {'load_balancer': type('L', (), {'endpoints': [
            main.WorkspaceEndpoint('adb', 'https://fixture.invalid', 't', models=[])]})()})()
        with patch.object(main, 'proxy', pool):
            paths = self.paths()
        self.assertIsNone(paths['databricks']['reason'])
        self.assertEqual(paths['databricks']['endpoints_serving_model'], 1)

    def test_the_view_is_attached_to_the_routing_payload(self):
        view = main._model_routing_view('claude-opus-5-5')
        self.assertEqual([p['provider'] for p in view['messages_paths']],
                         ['databricks', 'copilot', 'azure_anthropic'])
        self.assertEqual(view['effective_messages_order'],
                         ['databricks', 'copilot', 'azure_anthropic'])

    def test_reasons_come_from_a_closed_set(self):
        for path in main._messages_path_views('claude-opus-5-5'):
            with self.subTest(provider=path['provider']):
                self.assertIn(path['reason'], main.MESSAGES_PATH_REASONS | {None})


class SurfaceOrderDataSourceTests(unittest.TestCase):
    """Both surfaces must offer their options, or one of them cannot be set.

    The workbench built its order dropdown from `default_openai_order` alone,
    so the only «Foundry» on offer was `azure_openai` — the gpt face, which
    `_messages_provider_order` filters out. Picking it therefore did nothing to
    Claude, and `azure_anthropic` could not be named at all. A surface whose
    option list is absent is a surface the operator cannot configure.
    """

    def test_the_messages_surface_publishes_its_own_default_order(self):
        payload = asyncio.run(self._routing_payload())
        self.assertEqual(payload['default_messages_order'],
                         ['databricks', 'copilot', 'azure_anthropic'])

    def test_the_two_surfaces_publish_different_option_lists(self):
        # Shared list = the bug. azure_openai cannot serve messages and
        # databricks cannot serve the OpenAI-style entries.
        payload = asyncio.run(self._routing_payload())
        self.assertNotEqual(payload['default_messages_order'],
                            payload['default_openai_order'])
        self.assertNotIn('azure_openai', payload['default_messages_order'])
        self.assertNotIn('databricks', payload['default_openai_order'])

    def test_every_published_option_has_a_label_to_render(self):
        payload = asyncio.run(self._routing_payload())
        for provider in payload['default_messages_order']:
            with self.subTest(provider=provider):
                self.assertIn(provider, payload['provider_labels'])

    async def _routing_payload(self):
        with patch.object(main, '_require_admin_access', lambda *a, **k: None), \
             patch.object(main, '_copilot_endpoints', lambda: []), \
             patch.object(main, '_collect_openai_model_ids', lambda: []):
            return await main.admin_model_routing(None, None)
