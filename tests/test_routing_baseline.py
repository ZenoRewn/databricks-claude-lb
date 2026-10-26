"""Per-model routing needs a config baseline, not only a DB overlay. Author: Zeno Ren.

Every other policy layer in this gateway degrades to a *configured* baseline:
a per-key provider allow-list falls back to «which providers config declares»,
`allowed_models` to the endpoint's `models`, account enablement to the `enabled:`
flag, the panel password to `LB_DASHBOARD_PASSWORD`. Per-model routing was the
exception — its baseline was a hard-coded default order, so the operator's rules
could live in exactly one place, the MySQL overlay.

2026-10-26 showed what that costs. `zeno-sea-mysql` was Stopped, so
`KeyPolicyStore.reload()` timed out at every pod start and the documented
degradation rule did exactly what it promises: fall back to baseline. The
baseline being hard-coded, every deploy looked like «my routing rules were
wiped», and the only repair was to re-enter them by hand — which the next deploy
would discard again. Nothing was lost and nothing was broken; the rules simply
had nowhere durable to live that a ConfigMap could reach.

So `model_routing` in config is a baseline in the same sense as the others:
the overlay wins where it has a row, the config answers where it does not, and a
store that never loaded answers from config rather than from a constant. The
shape deliberately mirrors the `PUT /admin/model-routing/{model}` body, so what
an operator reads in the Dashboard transcribes into the ConfigMap unchanged.

Two consequences are tested here because they are easy to get wrong:

- A baseline rule is matched by model **identity**, not spelling, like a stored
  one (`key_policy.rule_key`). A baseline written in the Databricks spelling has
  to apply to a request that arrives in the Copilot spelling, otherwise it would
  be invisible to exactly the failover hop that rewrites the name.
- `DELETE /admin/model-routing/{model}` now returns to the *config* baseline
  rather than to the default order. That is what «baseline» means everywhere
  else in this codebase, and the alternative — a delete that overrides config —
  would make the ConfigMap unable to express anything the Dashboard had ever
  touched.

Recovery is the other half. The store was load-once: `reload()` ran in
`start()` and never again, with no admin entry point, so a database that came
back a minute later still left the process serving from baseline until someone
rolled the pod. The usage ledger already recovers on its own
(`_ensure_backend`); this one now does too, and the task is owned by the
lifespan alongside the other background loops rather than hidden in the store.
"""
import asyncio
import io
import unittest
from unittest.mock import patch

import key_policy
import main


async def _start_unloadable(baseline=None):
    """A mysql-backed store whose initial load failed, as on 2026-10-26."""
    store = key_policy.KeyPolicyStore({'type': 'mysql'}, routing_baseline=baseline)
    with patch.object(store, '_backend_load', side_effect=OSError('db down')):
        await store.start()
    assert store.status()['state'] == 'unavailable'
    return store


async def _start_loaded(baseline=None):
    store = key_policy.KeyPolicyStore(None, routing_baseline=baseline)
    await store.start()
    return store


def _unloadable_store(baseline=None):
    return asyncio.run(_start_unloadable(baseline))


def _loaded_store(baseline=None):
    return asyncio.run(_start_loaded(baseline))


class BaselineParsingTests(unittest.TestCase):
    def test_absent_baseline_is_empty(self):
        for value in (None, {}):
            self.assertEqual(key_policy.parse_routing_baseline(value), {})

    def test_a_rule_names_providers_in_order(self):
        parsed = key_policy.parse_routing_baseline(
            {'claude-opus-5.5': {'providers': ['copilot', 'databricks']}})
        self.assertEqual(parsed, {'claude-opus-5.5': {'providers': ('copilot', 'databricks')}})

    def test_an_accounts_only_rule_is_allowed(self):
        parsed = key_policy.parse_routing_baseline(
            {'gpt-5.4': {'copilot_accounts': ['b-acct', 'a-acct']}})
        self.assertEqual(parsed, {'gpt-5.4': {'copilot_accounts': ('a-acct', 'b-acct')}})

    def test_both_segments_may_appear_together(self):
        parsed = key_policy.parse_routing_baseline(
            {'gpt-5.4': {'providers': ['copilot'], 'copilot_accounts': ['a-acct']}})
        self.assertEqual(parsed['gpt-5.4'],
                         {'providers': ('copilot',), 'copilot_accounts': ('a-acct',)})

    def test_an_unknown_provider_is_rejected(self):
        # Refusing beats loading: a typo'd provider id silently does nothing,
        # and «the rule I wrote has no effect» has no symptom to search for.
        with self.assertRaises(ValueError) as caught:
            key_policy.parse_routing_baseline({'gpt-5.4': {'providers': ['copilto']}})
        self.assertIn('copilto', str(caught.exception))

    def test_an_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            key_policy.parse_routing_baseline({'gpt-5.4': {'provider': ['copilot']}})
        self.assertIn('provider', str(caught.exception))

    def test_an_entry_declaring_nothing_is_rejected(self):
        for value in ({}, {'copilot_accounts': []}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    key_policy.parse_routing_baseline({'gpt-5.4': value})

    def test_a_bare_list_is_rejected_and_the_message_names_the_shape(self):
        # The shape mirrors the admin PUT body on purpose; accepting a second
        # shorthand shape would mean two things to document and test.
        with self.assertRaises(ValueError) as caught:
            key_policy.parse_routing_baseline({'gpt-5.4': ['copilot']})
        self.assertIn('providers', str(caught.exception))

    def test_a_non_mapping_section_is_rejected(self):
        for value in ([], 'copilot', 7):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    key_policy.parse_routing_baseline(value)

    def test_an_empty_model_name_is_rejected(self):
        with self.assertRaises(ValueError):
            key_policy.parse_routing_baseline({'  ': {'providers': ['copilot']}})

    def test_two_spellings_of_one_model_collide(self):
        # One model, one rule — the storage invariant, enforced here too. Left
        # alone, `rule_key` would resolve one spelling and the other would sit
        # in the ConfigMap doing nothing.
        with self.assertRaises(ValueError) as caught:
            key_policy.parse_routing_baseline({
                'claude-opus-5.5': {'providers': ['copilot']},
                'databricks-claude-opus-5-5': {'providers': ['databricks']},
            })
        self.assertIn('claude-opus-5.5', str(caught.exception))

    def test_unrelated_models_with_similar_names_do_not_collide(self):
        # Only Claude models have a canonical identity; inventing one for gpt
        # would apply a routing rule nobody wrote.
        parsed = key_policy.parse_routing_baseline({
            'gpt-5.4': {'providers': ['copilot']},
            'gpt-5-4': {'providers': ['azure_openai']},
        })
        self.assertEqual(len(parsed), 2)


class BaselineResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_baseline_answers_when_no_overlay_row_exists(self):
        store = await _start_loaded({'gpt-5.4': {'providers': ['azure_openai']}})
        self.assertEqual(store.model_preference('gpt-5.4'), ('azure_openai',))
        self.assertEqual(store.model_preference_resolution('gpt-5.4'),
                         (('azure_openai',), 'config'))

    async def test_an_overlay_row_wins_over_the_baseline(self):
        store = await _start_loaded({'gpt-5.4': {'providers': ['azure_openai']}})
        await store.set_model_preference('gpt-5.4', ['copilot'])
        self.assertEqual(store.model_preference_resolution('gpt-5.4'),
                         (('copilot',), 'overlay'))

    async def test_the_baseline_survives_a_store_that_never_loaded(self):
        # The 2026-10-26 case: this is the whole point of the layer.
        store = await _start_unloadable({'gpt-5.4': {'providers': ['azure_openai']}})
        self.assertEqual(store.model_preference_resolution('gpt-5.4'),
                         (('azure_openai',), 'config'))

    async def test_a_model_with_no_rule_anywhere_has_no_preference(self):
        store = await _start_loaded({'gpt-5.4': {'providers': ['azure_openai']}})
        self.assertEqual(store.model_preference_resolution('gpt-4o'), ((), None))

    async def test_clearing_an_overlay_rule_returns_to_the_config_baseline(self):
        store = await _start_loaded({'gpt-5.4': {'providers': ['azure_openai']}})
        await store.set_model_preference('gpt-5.4', ['copilot'])
        await store.clear_model_preference('gpt-5.4')
        self.assertEqual(store.model_preference_resolution('gpt-5.4'),
                         (('azure_openai',), 'config'))

    async def test_the_baseline_is_matched_by_identity_not_spelling(self):
        store = await _start_loaded({'databricks-claude-opus-5-5': {'providers': ['copilot']}})
        for spelling in ('claude-opus-5.5', 'claude-opus-5-5',
                         'databricks-claude-opus-5-5'):
            with self.subTest(spelling=spelling):
                self.assertEqual(store.model_preference(spelling), ('copilot',))

    async def test_an_overlay_row_in_another_spelling_still_wins(self):
        store = await _start_loaded({'databricks-claude-opus-5-5': {'providers': ['copilot']}})
        await store.set_model_preference('claude-opus-5.5', ['azure_anthropic'])
        self.assertEqual(store.model_preference_resolution('databricks-claude-opus-5-5'),
                         (('azure_anthropic',), 'overlay'))

    async def test_the_account_segment_follows_the_same_rules(self):
        store = await _start_loaded({'gpt-5.4': {'copilot_accounts': ['a-acct']}})
        self.assertEqual(store.model_accounts_resolution('gpt-5.4'),
                         (('a-acct',), 'config'))
        self.assertFalse(store.model_allows_account('gpt-5.4', 'b-acct'))
        await store.set_model_accounts('gpt-5.4', ['b-acct'])
        self.assertEqual(store.model_accounts_resolution('gpt-5.4'),
                         (('b-acct',), 'overlay'))
        await store.clear_model_accounts('gpt-5.4')
        self.assertEqual(store.model_accounts_resolution('gpt-5.4'),
                         (('a-acct',), 'config'))

    async def test_the_account_baseline_survives_a_store_that_never_loaded(self):
        store = await _start_unloadable({'gpt-5.4': {'copilot_accounts': ['a-acct']}})
        self.assertEqual(store.model_accounts('gpt-5.4'), ('a-acct',))
        self.assertEqual(store.model_accounts_exclusion_reason('gpt-5.4', 'b-acct'),
                         'not_selected_for_model')

    async def test_an_unavailable_store_without_a_baseline_is_unchanged(self):
        # Existing deployments declare no `model_routing`; they must behave
        # byte-for-byte as before.
        store = await _start_unloadable()
        self.assertEqual(store.model_preference('gpt-5.4'), ())
        self.assertEqual(store.model_accounts('gpt-5.4'), ())
        self.assertTrue(store.model_allows_account('gpt-5.4', 'b-acct'))

    async def test_status_reports_the_baseline_size(self):
        store = await _start_unloadable({'gpt-5.4': {'providers': ['copilot']}})
        self.assertEqual(store.status()['baseline_models'], 1)

    async def test_the_baseline_snapshot_lists_every_declared_model(self):
        store = await _start_loaded({'gpt-5.4': {'providers': ['copilot']},
                               'gpt-4o': {'copilot_accounts': ['a-acct']}})
        self.assertEqual(sorted(store.routing_baseline_snapshot()), ['gpt-4o', 'gpt-5.4'])


class SurfaceOrderTests(unittest.TestCase):
    """The baseline has to reach the four read sites, not just the store."""

    def setUp(self):
        self.store = _unloadable_store({
            'claude-opus-5.5': {'providers': ['copilot', 'azure_anthropic']},
            'gpt-5.4': {'providers': ['azure_openai']},
        })
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def test_the_messages_order_follows_the_baseline_while_the_db_is_down(self):
        self.assertEqual(main._messages_provider_order('claude-opus-5.5'),
                         ('copilot', 'azure_anthropic', 'databricks'))

    def test_the_openai_order_follows_the_baseline_while_the_db_is_down(self):
        self.assertEqual(main._openai_provider_order('gpt-5.4'),
                         ('azure_openai', 'copilot'))

    def test_a_baseline_naming_copilot_opens_claude_on_openai_entry_points(self):
        # Same opt-in the overlay grants, now expressible in a ConfigMap.
        self.assertTrue(main._model_pinned_to_openai_style_provider('claude-opus-5.5'))

    def test_a_baseline_keeps_claude_open_on_openai_entries_after_a_reset(self):
        # Clearing the stored rule drops to the baseline, which here still
        # names copilot — so the Responses/Chat entry points stay open and the
        # reset dialog must not claim they close.
        with patch.object(main, 'copilot_proxy', None):
            reset = main._model_routing_view('claude-opus-5.5')['reset_to']
        self.assertTrue(reset['opens_claude_on_openai_entry_points'])

    def test_a_model_outside_the_baseline_keeps_the_default_order(self):
        self.assertEqual(main._openai_provider_order('gpt-4o'),
                         ('copilot', 'azure_openai'))
        self.assertEqual(main._messages_provider_order('claude-haiku-5.5'),
                         ('databricks', 'copilot', 'azure_anthropic'))


class RoutingViewTests(unittest.TestCase):
    """Diagnostics must say *where* the effective rule came from.

    A baseline rule rendered as if it were saved would send an operator looking
    for a database row that does not exist — the «diagnosis that disagrees with
    the decision» this repo treats as worse than no diagnosis at all.
    """

    def setUp(self):
        self.store = _unloadable_store({'gpt-5.4': {'providers': ['azure_openai'],
                                                    'copilot_accounts': ['a-acct']}})
        for name, value in (('key_policy_store', self.store), ('copilot_proxy', None)):
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_the_view_reports_the_source_of_each_segment(self):
        view = main._model_routing_view('gpt-5.4')
        self.assertEqual(view['providers'], ['azure_openai'])
        self.assertEqual(view['providers_source'], 'config')
        self.assertEqual(view['copilot_accounts_source'], 'config')

    def test_a_model_without_a_rule_reports_no_source(self):
        view = main._model_routing_view('gpt-4o')
        self.assertIsNone(view['providers_source'])
        self.assertIsNone(view['copilot_accounts_source'])

    def test_the_reset_preview_is_the_baseline_not_the_default_order(self):
        # `DELETE` returns to the baseline, so a dialog promising «back to the
        # default order» would describe a write the gateway will not perform.
        reset = main._model_routing_view('gpt-5.4')['reset_to']
        self.assertEqual(reset['providers'], ['azure_openai'])
        self.assertEqual(reset['effective_openai_order'], ['azure_openai', 'copilot'])
        self.assertEqual(reset['copilot_accounts'], ['a-acct'])
        self.assertEqual(reset['source'], 'config')

    def test_the_reset_preview_without_a_baseline_is_the_default_order(self):
        reset = main._model_routing_view('gpt-4o')['reset_to']
        self.assertEqual(reset['providers'], [])
        self.assertEqual(reset['effective_openai_order'], ['copilot', 'azure_openai'])
        self.assertEqual(reset['effective_messages_order'],
                         ['databricks', 'copilot', 'azure_anthropic'])
        self.assertEqual(reset['copilot_accounts'], [])
        self.assertEqual(reset['source'], 'default')
        self.assertFalse(reset['opens_claude_on_openai_entry_points'])


class ConfigLoadingTests(unittest.TestCase):
    ADB = ('endpoints:\n'
           '  - name: synthetic-adb\n'
           '    api_base: https://fixture.invalid\n'
           '    token: inline-token\n')

    def setUp(self):
        p = patch.object(main, 'MODEL_ROUTING_BASELINE', {})
        p.start()
        self.addCleanup(p.stop)

    def _load(self, extra: str):
        with patch('builtins.open', return_value=io.StringIO(self.ADB + extra)):
            return main.load_config('synthetic.yaml')

    def test_the_section_is_parsed_into_the_baseline(self):
        self._load('model_routing:\n'
                   '  claude-opus-5.5:\n'
                   '    providers: [copilot, databricks]\n')
        self.assertEqual(main.MODEL_ROUTING_BASELINE,
                         {'claude-opus-5.5': {'providers': ('copilot', 'databricks')}})

    def test_no_section_leaves_an_empty_baseline(self):
        self._load('')
        self.assertEqual(main.MODEL_ROUTING_BASELINE, {})

    def test_an_invalid_rule_refuses_to_start(self):
        with self.assertRaises(ValueError):
            self._load('model_routing:\n'
                       '  claude-opus-5.5:\n'
                       '    providers: [copilto]\n')


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    """A database that comes back must not need a pod roll."""

    async def asyncSetUp(self):
        for name, value in (('RELOAD_RETRY_INITIAL', 0.01), ('RELOAD_RETRY_MAX', 0.01)):
            p = patch.object(key_policy, name, value)
            p.start()
            self.addCleanup(p.stop)

    async def _store_with_failing_load(self):
        store = key_policy.KeyPolicyStore({'type': 'mysql'})
        self.failing = patch.object(store, '_backend_load', side_effect=OSError('db down'))
        self.failing.start()
        for name in ('_backend_load_accounts', '_backend_load_preferences',
                     '_backend_load_model_accounts', '_backend_load_panel_password'):
            p = patch.object(store, name, side_effect=OSError('db down'))
            p.start()
            self.addCleanup(p.stop)
        await store.start()
        return store

    async def test_a_failed_initial_load_is_retried_until_it_succeeds(self):
        store = await self._store_with_failing_load()
        self.assertFalse(store.fully_loaded)
        task = asyncio.create_task(store.recover_until_loaded())
        self.addCleanup(task.cancel)
        await asyncio.sleep(0.05)
        self.assertFalse(store.fully_loaded)   # still failing
        self.failing.stop()
        with patch.object(store, '_backend_load', return_value={}), \
                patch.object(store, '_backend_load_accounts', return_value={}), \
                patch.object(store, '_backend_load_preferences',
                             return_value={'gpt-5.4': ('copilot',)}), \
                patch.object(store, '_backend_load_model_accounts', return_value={}), \
                patch.object(store, '_backend_load_panel_password', return_value=None):
            await asyncio.wait_for(task, timeout=2)
        self.assertTrue(store.fully_loaded)
        self.assertEqual(store.status()['state'], 'loaded')
        self.assertEqual(store.model_preference('gpt-5.4'), ('copilot',))

    async def test_a_write_during_the_outage_does_not_end_the_recovery(self):
        # A successful write flips the state to «loaded» so the just-written
        # rule takes effect, but the cache then holds exactly one row. Treating
        # that as a loaded store would strand every other stored rule.
        store = await self._store_with_failing_load()
        with patch.object(store, '_backend_write_preference', return_value=None):
            await store.set_model_preference('gpt-4o', ['copilot'])
        self.assertEqual(store.status()['state'], 'loaded')
        self.assertFalse(store.fully_loaded)

    async def test_a_store_that_loaded_cleanly_needs_no_recovery(self):
        store = await _start_loaded()
        self.assertTrue(store.fully_loaded)

    async def test_recovery_is_visible_in_the_status(self):
        store = await self._store_with_failing_load()
        self.assertTrue(store.status()['recovering'])
        self.assertFalse((await _start_loaded()).status()['recovering'])


if __name__ == '__main__':
    unittest.main()
