"""Dashboard panel password and session cookie. Author: Zeno Ren.

The keys panel needs a credential on every write, and using the LB API Key for
that meant either retyping it per save or holding it in a JS variable. A
separate password with an HttpOnly cookie removes both problems: the browser
carries the session and **page scripts hold nothing at all**, so an XSS has
nothing to steal.

The password does not replace the API Key — `curl`, scripts, and any deployment
that has not configured a password still authenticate by header. It is an
additional way in, not a different one.

Two properties this layer must not lose:

  * An absent or too-short password means the feature is **unavailable**, never
    "unlocked without a password". A restriction layer that fails open here
    would expose every admin endpoint to the public internet, because
    `/admin/*` is served by an ingress with no OAuth in front of it.
  * The password is low-entropy by nature (a human picks it), so an online
    guessing attack is the realistic threat. Hence the length floor and the
    per-client backoff.
"""
import hashlib
import logging
import os
import time
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

import main

PASSWORD = 'panel-password-for-tests'
API_KEY = 'lb-panel-session-key-0123'


class _Stub:
    def verify_api_key(self, key):
        return bool(key) and key == API_KEY


class PanelSessionTestCase(unittest.TestCase):
    """Shared harness: a configured password plus a working API key."""

    password = PASSWORD

    def setUp(self):
        main.API_KEY_TO_TENANT.clear()
        main.API_KEY_TO_TENANT[API_KEY] = 'default'
        self.addCleanup(main.API_KEY_TO_TENANT.clear)
        main._PANEL_SESSIONS.clear()
        main._PANEL_FAILURES.clear()
        self.addCleanup(main._PANEL_SESSIONS.clear)
        self.addCleanup(main._PANEL_FAILURES.clear)
        for attr, value in (('proxy', _Stub()), ('copilot_proxy', None),
                            ('key_policy_store', None)):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        env = patch.dict(os.environ, {main.PANEL_PASSWORD_ENV: self.password}
                         if self.password is not None else {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        if self.password is None:
            os.environ.pop(main.PANEL_PASSWORD_ENV, None)
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def login(self, password=None):
        return self.client.post('/admin/session',
                                json={'password': password or self.password})


class PasswordConfigurationTests(PanelSessionTestCase):
    def test_a_configured_password_is_reported_without_revealing_it(self):
        body = self.client.get('/admin/session').json()
        self.assertTrue(body['password_configured'])
        self.assertFalse(body['authenticated'])
        self.assertNotIn(PASSWORD, str(body))

    def test_the_status_endpoint_needs_no_credential(self):
        # It is the first request the page makes; requiring a credential to ask
        # "do I need to log in" would be circular.
        self.assertEqual(self.client.get('/admin/session').status_code, 200)

    def test_an_absent_password_disables_login_without_opening_anything(self):
        os.environ.pop(main.PANEL_PASSWORD_ENV, None)
        r = self.client.post('/admin/session', json={'password': 'anything'})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()['detail']['error']['code'], 'panel_password_not_configured')
        self.assertFalse(self.client.get('/admin/session').json()['password_configured'])
        # The crucial part: no password must not mean no gate.
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)

    # Deliberately not a word that appears in the refusal message itself — the
    # first version of this test used 'short' and matched "shorter than 12".
    TOO_SHORT = 'qz7w'

    def test_a_short_password_counts_as_unconfigured(self):
        with patch.dict(os.environ, {main.PANEL_PASSWORD_ENV: self.TOO_SHORT}):
            self.assertFalse(self.client.get('/admin/session').json()['password_configured'])
            r = self.client.post('/admin/session', json={'password': self.TOO_SHORT})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)

    def test_the_minimum_length_is_stated_in_the_refusal_not_the_password(self):
        with patch.dict(os.environ, {main.PANEL_PASSWORD_ENV: self.TOO_SHORT}):
            with self.assertLogs('main', level='ERROR') as logs:
                self.client.post('/admin/session', json={'password': self.TOO_SHORT})
            joined = '\n'.join(logs.output)
        self.assertIn(str(main.PANEL_PASSWORD_MIN_LENGTH), joined)
        self.assertNotIn(self.TOO_SHORT, joined, 'the configured value must never be logged')

    def test_the_password_is_not_in_the_effective_settings(self):
        # /config/effective returns the whole settings dict to any valid key.
        body = self.client.get('/config/effective', headers={'x-api-key': API_KEY}).json()
        self.assertNotIn(PASSWORD, str(body))
        self.assertNotIn(main.PANEL_PASSWORD_ENV.lower(),
                         str(body.get('settings', {})).lower())


class LoginTests(PanelSessionTestCase):
    def test_the_right_password_issues_a_scoped_httponly_cookie(self):
        r = self.login()
        self.assertEqual(r.status_code, 200)
        raw = r.headers.get('set-cookie', '')
        self.assertIn(main.PANEL_SESSION_COOKIE + '=', raw)
        # Attribute values are case-insensitive per RFC 6265; Starlette emits
        # them lowercase, so the assertion must not pin the casing.
        lowered = raw.lower()
        self.assertIn('httponly', lowered)
        self.assertIn('samesite=strict', lowered)
        self.assertIn('path=/admin', lowered)
        self.assertIn(f'max-age={main.PANEL_SESSION_TTL_SECONDS}', lowered)

    def test_the_response_body_carries_no_token_or_password(self):
        r = self.login()
        token = self.client.cookies.get(main.PANEL_SESSION_COOKIE)
        self.assertNotIn(PASSWORD, r.text)
        self.assertNotIn(token, r.text, 'the token belongs in the cookie, not the body')

    def test_the_wrong_password_is_401_and_sets_no_cookie(self):
        r = self.client.post('/admin/session', json={'password': 'wrong-but-long-enough'})
        self.assertEqual(r.status_code, 401)
        self.assertNotIn('set-cookie', {k.lower() for k in r.headers})
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)

    def test_a_missing_or_non_string_password_is_400_not_a_crash(self):
        for payload in ({}, {'password': None}, {'password': 42}, {'password': ''}):
            with self.subTest(payload=payload):
                r = self.client.post('/admin/session', json=payload)
                self.assertEqual(r.status_code, 400)

    def test_a_non_json_body_without_a_key_is_refused(self):
        # An API-key login needs no body, so an unparsable one is not itself an
        # error — the refusal has to come from having no credential at all.
        r = self.client.post('/admin/session', content=b'not json',
                             headers={'content-type': 'application/json'})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'password_required')

    def test_a_valid_api_key_also_buys_a_session(self):
        # Otherwise the fallback path would have to keep the key in page scripts,
        # and "scripts hold no credential" would only be true half the time.
        r = self.client.post('/admin/session', headers={'x-api-key': API_KEY})
        self.assertEqual(r.status_code, 200)
        self.assertIn(main.PANEL_SESSION_COOKIE + '=', r.headers.get('set-cookie', ''))
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 200)

    def test_an_invalid_api_key_without_a_password_is_refused(self):
        r = self.client.post('/admin/session', headers={'x-api-key': 'wrong'})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)

    def test_the_api_key_path_works_when_no_password_is_configured(self):
        os.environ.pop(main.PANEL_PASSWORD_ENV, None)
        r = self.client.post('/admin/session', headers={'x-api-key': API_KEY})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 200)

    def test_no_password_and_no_key_says_the_password_is_unconfigured(self):
        os.environ.pop(main.PANEL_PASSWORD_ENV, None)
        r = self.client.post('/admin/session', json={})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()['detail']['error']['code'], 'panel_password_not_configured')

    def test_secure_is_set_only_over_https(self):
        plain = self.login()
        self.assertNotIn('Secure', plain.headers.get('set-cookie', ''),
                         'a Secure cookie over plain http would never be stored')
        forwarded = self.client.post('/admin/session', json={'password': PASSWORD},
                                     headers={'x-forwarded-proto': 'https'})
        self.assertIn('Secure', forwarded.headers.get('set-cookie', ''))

    def test_the_stored_session_is_a_hash_not_the_token(self):
        self.login()
        token = self.client.cookies.get(main.PANEL_SESSION_COOKIE)
        self.assertNotIn(token, main._PANEL_SESSIONS)
        self.assertIn(hashlib.sha256(token.encode()).hexdigest(), main._PANEL_SESSIONS)


class SessionAuthorisesAdminEndpointsTests(PanelSessionTestCase):
    def test_the_cookie_replaces_the_api_key(self):
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)
        self.login()
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 200)

    def test_the_api_key_still_works_on_its_own(self):
        # Scripts and curl must keep working; the password is an addition.
        self.assertEqual(
            self.client.get('/admin/api-keys', headers={'x-api-key': API_KEY}).status_code, 200)

    def test_a_write_endpoint_accepts_the_cookie_too(self):
        self.login()
        r = self.client.put('/admin/api-keys/default/providers', json={'providers': ['copilot']})
        # 503 because no policy store is configured here — the point is that it
        # got past the gate rather than 401.
        self.assertNotEqual(r.status_code, 401)

    def test_a_forged_cookie_is_rejected(self):
        self.client.cookies.set(main.PANEL_SESSION_COOKIE, 'not-a-real-token')
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)

    def test_the_status_endpoint_reports_an_active_session(self):
        self.login()
        body = self.client.get('/admin/session').json()
        self.assertTrue(body['authenticated'])
        self.assertGreater(body['expires_in'], 0)
        self.assertLessEqual(body['expires_in'], main.PANEL_SESSION_TTL_SECONDS)


class SessionLifetimeTests(PanelSessionTestCase):
    def test_activity_slides_the_window(self):
        self.login()
        token = self.client.cookies.get(main.PANEL_SESSION_COOKIE)
        digest = hashlib.sha256(token.encode()).hexdigest()
        main._PANEL_SESSIONS[digest] = time.time() + 5
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 200)
        self.assertGreater(main._PANEL_SESSIONS[digest], time.time() + 60,
                           'a request must renew the idle window')

    def test_an_expired_session_is_rejected_and_forgotten(self):
        self.login()
        token = self.client.cookies.get(main.PANEL_SESSION_COOKIE)
        digest = hashlib.sha256(token.encode()).hexdigest()
        main._PANEL_SESSIONS[digest] = time.time() - 1
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)
        self.assertNotIn(digest, main._PANEL_SESSIONS, 'an expired entry must not linger')

    def test_logout_invalidates_the_session_immediately(self):
        self.login()
        self.assertEqual(self.client.delete('/admin/session').status_code, 200)
        self.assertEqual(main._PANEL_SESSIONS, {})
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)

    def test_logout_without_a_session_is_not_an_error(self):
        self.assertEqual(self.client.delete('/admin/session').status_code, 200)

    def test_logout_clears_the_cookie_in_the_browser_too(self):
        self.login()
        raw = self.client.delete('/admin/session').headers.get('set-cookie', '')
        self.assertIn(main.PANEL_SESSION_COOKIE + '=', raw)
        self.assertTrue('Max-Age=0' in raw or 'expires=' in raw.lower(),
                        'the browser copy has to be cleared, not just the server entry')

    def test_logging_in_twice_does_not_strand_the_first_session(self):
        self.login()
        first = self.client.cookies.get(main.PANEL_SESSION_COOKIE)
        self.login()
        second = self.client.cookies.get(main.PANEL_SESSION_COOKIE)
        self.assertNotEqual(first, second)
        self.assertEqual(len(main._PANEL_SESSIONS), 1,
                         'a fresh login replaces the old session rather than piling up')


class BruteForceBackoffTests(PanelSessionTestCase):
    def _fail(self, times):
        for _ in range(times):
            self.client.post('/admin/session', json={'password': 'wrong-but-long-enough'})

    def test_repeated_failures_lock_the_client_out(self):
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        r = self.client.post('/admin/session', json={'password': 'wrong-but-long-enough'})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()['detail']['error']['code'], 'too_many_attempts')
        self.assertIn('retry-after', {k.lower() for k in r.headers})

    def test_the_lockout_rejects_the_correct_password_too(self):
        # Otherwise the lockout is only a speed bump: an attacker who guesses
        # right on attempt six still gets in.
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        self.assertEqual(self.login().status_code, 429)
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 401)

    def test_the_lockout_does_not_say_whether_the_password_was_right(self):
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        right = self.login()
        wrong = self.client.post('/admin/session', json={'password': 'wrong-but-long-enough'})
        self.assertEqual(right.status_code, wrong.status_code)
        self.assertEqual(right.json()['detail']['error']['code'],
                         wrong.json()['detail']['error']['code'])

    def _expire_lockout(self):
        """Let the current lockout elapse without touching the failure count.

        Attempts made *during* a lockout deliberately do not extend it — if they
        did, anyone who can reach the endpoint could keep the real operator
        locked out forever by hammering it. So growing the backoff requires
        waiting out the current one and then failing again, exactly as a real
        attacker would have to.
        """
        key = list(main._PANEL_FAILURES)[0]
        count, _ = main._PANEL_FAILURES[key]
        main._PANEL_FAILURES[key] = (count, time.time() - 1)

    def test_the_backoff_grows_with_each_round(self):
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        first = int(self.client.post('/admin/session', json={'password': 'x' * 20})
                    .headers['retry-after'])
        self._expire_lockout()
        self._fail(1)
        second = int(self.client.post('/admin/session', json={'password': 'x' * 20})
                     .headers['retry-after'])
        self.assertGreater(second, first)

    def test_attempts_during_a_lockout_do_not_extend_it(self):
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        before = main._PANEL_FAILURES[list(main._PANEL_FAILURES)[0]]
        self._fail(3)
        self.assertEqual(main._PANEL_FAILURES[list(main._PANEL_FAILURES)[0]], before,
                         'hammering during a lockout must not lock the operator out further')

    def test_the_backoff_is_capped(self):
        self._fail(1)
        key = list(main._PANEL_FAILURES)[0]
        # Jump straight to a very high failure count: 2**195 seconds of backoff
        # would be an outage inflicted on the operator, not a defence.
        main._PANEL_FAILURES[key] = (200, time.time() - 1)
        self._fail(1)
        _, until = main._PANEL_FAILURES[key]
        self.assertLessEqual(until - time.time(), main.PANEL_LOCKOUT_MAX_SECONDS + 1,
                             'an unbounded backoff would be a self-inflicted outage')

    def test_a_successful_login_clears_the_failure_count(self):
        self._fail(main.PANEL_LOCKOUT_THRESHOLD - 1)
        self.assertEqual(self.login().status_code, 200)
        self.assertEqual(main._PANEL_FAILURES, {})

    def test_the_lockout_expires(self):
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        key = list(main._PANEL_FAILURES)[0]
        count, _ = main._PANEL_FAILURES[key]
        main._PANEL_FAILURES[key] = (count, time.time() - 1)
        self.assertEqual(self.login().status_code, 200)

    def test_the_api_key_path_is_not_affected_by_panel_lockout(self):
        # Locking out panel logins must not lock out the operator's scripts.
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        self.assertEqual(
            self.client.get('/admin/api-keys', headers={'x-api-key': API_KEY}).status_code, 200)

    def test_an_api_key_login_is_not_blocked_by_panel_lockout_either(self):
        # The backoff defends a low-entropy password. Letting it also bar the
        # high-entropy key would turn a guessing attempt into an outage.
        self._fail(main.PANEL_LOCKOUT_THRESHOLD)
        self.assertEqual(
            self.client.post('/admin/session', headers={'x-api-key': API_KEY}).status_code, 200)

    def test_an_api_key_login_failure_does_not_feed_the_panel_backoff(self):
        for _ in range(main.PANEL_LOCKOUT_THRESHOLD + 2):
            self.client.post('/admin/session', headers={'x-api-key': 'wrong'})
        self.assertEqual(main._PANEL_FAILURES, {})
        self.assertEqual(self.login().status_code, 200)


class NoSecretLeaksTests(PanelSessionTestCase):
    def test_no_response_anywhere_echoes_the_password(self):
        responses = [
            self.client.get('/admin/session'),
            self.client.post('/admin/session', json={'password': PASSWORD}),
            self.client.post('/admin/session', json={'password': 'wrong-but-long-enough'}),
            self.client.get('/admin/session'),
            self.client.delete('/admin/session'),
            self.client.get('/version'),
        ]
        for r in responses:
            with self.subTest(url=str(r.url), status=r.status_code):
                self.assertNotIn(PASSWORD, r.text)
                self.assertNotIn(PASSWORD, str(dict(r.headers)))

    def test_logging_in_does_not_log_the_password(self):
        with self.assertLogs('main', level='DEBUG') as logs:
            logging.getLogger('main').debug('anchor so assertLogs always has a record')
            self.login()
            self.client.post('/admin/session', json={'password': 'wrong-but-long-enough'})
        joined = '\n'.join(logs.output)
        self.assertNotIn(PASSWORD, joined)
        self.assertNotIn('wrong-but-long-enough', joined)

    def test_the_session_token_is_not_logged(self):
        with self.assertLogs('main', level='DEBUG') as logs:
            logging.getLogger('main').debug('anchor')
            self.login()
            token = self.client.cookies.get(main.PANEL_SESSION_COOKIE)
            self.client.get('/admin/api-keys')
        self.assertNotIn(token, '\n'.join(logs.output))


NEW_PASSWORD = 'a-brand-new-panel-password'


class _PasswordStoreCase(PanelSessionTestCase):
    """Harness with a real policy store, so the overlay can actually be written."""

    def setUp(self):
        import key_policy
        self.store = key_policy.KeyPolicyStore(None)
        asyncio_run = __import__('asyncio').run
        asyncio_run(self.store.start())
        super().setUp()
        p = patch.object(main, 'key_policy_store', self.store)
        p.start()
        self.addCleanup(p.stop)

    def change(self, new, current=None, **kw):
        body = {'new_password': new}
        if current is not None:
            body['current_password'] = current
        return self.client.put('/admin/panel-password', json=body, **kw)


class PasswordVerifierStorageTests(_PasswordStoreCase):
    """A panel password is only ever *verified*, so store a one-way verifier.

    Upstream credentials are sealed with AES-GCM because they must be replayed
    to the upstream. This one never leaves the gateway, so hashing it is
    strictly better: even with `LB_CREDENTIAL_KEY` in hand nobody recovers the
    password, and the feature does not depend on that key being configured.
    """

    def test_the_stored_verifier_is_not_the_password(self):
        self.login()
        self.assertEqual(self.change(NEW_PASSWORD, current=PASSWORD).status_code, 200)
        verifier = self.store.panel_password_verifier()
        self.assertIsNotNone(verifier)
        self.assertNotIn(NEW_PASSWORD, verifier)
        self.assertTrue(verifier.startswith('pbkdf2_sha256$'),
                        'the algorithm and its cost have to be recorded with the hash')

    def test_a_stored_verifier_takes_precedence_over_the_environment(self):
        # The project's single rule: config is the baseline, the store is the
        # overlay. Here the baseline is the env var.
        self.login()
        self.change(NEW_PASSWORD, current=PASSWORD)
        self.client.delete('/admin/session')
        self.assertEqual(self.login(NEW_PASSWORD).status_code, 200)
        self.assertEqual(self.login(PASSWORD).status_code, 401,
                         'the env password must stop working once overridden')

    def test_an_unloaded_store_falls_back_to_the_environment(self):
        self.login()
        self.change(NEW_PASSWORD, current=PASSWORD)
        self.client.delete('/admin/session')
        self.store._state = 'unavailable'
        self.assertEqual(self.login(PASSWORD).status_code, 200,
                         'an overlay that cannot be read falls back to the baseline')

    def test_with_neither_overlay_nor_environment_login_is_unavailable_not_open(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(main.PANEL_PASSWORD_ENV, None)
            r = self.client.post('/admin/session', json={'password': 'anything-long-enough'})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()['detail']['error']['code'],
                             'panel_password_not_configured')
            self.assertEqual(self.client.get('/admin/api-keys').status_code, 401,
                             'an unavailable password must never unlock anything')

    def test_clearing_the_overlay_returns_to_the_environment_baseline(self):
        self.login()
        self.change(NEW_PASSWORD, current=PASSWORD)
        self.client.delete('/admin/session')
        self.login(NEW_PASSWORD)
        r = self.client.request('DELETE', '/admin/panel-password',
                                json={'current_password': NEW_PASSWORD})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(self.store.panel_password_verifier())
        self.client.delete('/admin/session')
        self.assertEqual(self.login(PASSWORD).status_code, 200)

    def test_the_session_view_names_where_the_password_comes_from(self):
        self.assertEqual(self.client.get('/admin/session').json()['password_source'], 'env')
        self.login()
        self.change(NEW_PASSWORD, current=PASSWORD)
        self.assertEqual(self.client.get('/admin/session').json()['password_source'], 'store')

    def test_no_source_is_reported_when_nothing_is_configured(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(main.PANEL_PASSWORD_ENV, None)
            body = self.client.get('/admin/session').json()
            self.assertEqual(body['password_source'], 'none')
            self.assertFalse(body['password_configured'])


class PasswordChangeAuthorisationTests(_PasswordStoreCase):
    def test_a_session_alone_is_not_enough(self):
        # The session may have been obtained with an API key, so it does not
        # prove the caller knows the current password.
        self.login()
        r = self.change(NEW_PASSWORD)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'current_password_required')
        self.assertIsNone(self.store.panel_password_verifier())

    def test_a_wrong_current_password_is_refused(self):
        self.login()
        r = self.change(NEW_PASSWORD, current='wrong-but-long-enough')
        self.assertEqual(r.status_code, 401)
        self.assertIsNone(self.store.panel_password_verifier())

    def test_an_api_key_may_stand_in_for_the_current_password(self):
        # The break-glass path: whoever holds a valid LB API Key can already
        # call every admin endpoint, so this grants nothing new — and it is the
        # only way back in once a stored password is forgotten.
        self.login()
        r = self.change(NEW_PASSWORD, headers={'x-api-key': API_KEY})
        self.assertEqual(r.status_code, 200)
        self.assertIsNotNone(self.store.panel_password_verifier())

    def test_an_unauthenticated_caller_cannot_change_it(self):
        self.assertEqual(self.change(NEW_PASSWORD, current=PASSWORD).status_code, 401)
        self.assertIsNone(self.store.panel_password_verifier())

    def test_a_too_short_new_password_is_refused(self):
        self.login()
        r = self.change('short', current=PASSWORD)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['detail']['error']['code'], 'panel_password_too_short')
        self.assertIsNone(self.store.panel_password_verifier())

    def test_the_length_floor_counts_characters_not_bytes(self):
        # `kubectl describe secret` reports bytes, so a multi-byte password can
        # look long enough there and be rejected here. Twelve characters.
        self.login()
        self.assertEqual(self.change('密码密码密码密码密码密码', current=PASSWORD).status_code, 200)

    def test_a_wrong_current_password_goes_through_the_same_backoff(self):
        # Otherwise this endpoint is a way to guess the current password without
        # ever tripping the login backoff.
        self.login()
        for _ in range(main.PANEL_LOCKOUT_THRESHOLD):
            self.change(NEW_PASSWORD, current='wrong-but-long-enough')
        r = self.change(NEW_PASSWORD, current=PASSWORD)
        self.assertEqual(r.status_code, 429,
                         'the correct password must be refused while locked out too')
        self.assertIsNone(self.store.panel_password_verifier())

    def test_changing_it_invalidates_other_sessions_but_not_this_one(self):
        other = TestClient(main.app, raise_server_exceptions=False)
        other.post('/admin/session', json={'password': PASSWORD})
        self.assertEqual(other.get('/admin/api-keys').status_code, 200)
        self.login()
        self.assertEqual(self.change(NEW_PASSWORD, current=PASSWORD).status_code, 200)
        self.assertEqual(other.get('/admin/api-keys').status_code, 401,
                         'a password change must not leave older sessions alive')
        self.assertEqual(self.client.get('/admin/api-keys').status_code, 200,
                         'the caller should not have to log in again immediately')

    def test_a_write_failure_keeps_the_old_password_working(self):
        self.login()
        with patch.object(self.store, '_backend_write_panel_password',
                          side_effect=OSError('db down')):
            r = self.change(NEW_PASSWORD, current=PASSWORD)
        self.assertEqual(r.status_code, 503)
        self.client.delete('/admin/session')
        self.assertEqual(self.login(PASSWORD).status_code, 200)


class PasswordChangeLeakTests(_PasswordStoreCase):
    def test_neither_password_appears_in_any_response(self):
        self.login()
        responses = [
            self.change(NEW_PASSWORD, current=PASSWORD),
            self.client.get('/admin/session'),
            self.client.get('/config/effective', headers={'x-api-key': API_KEY}),
        ]
        for r in responses:
            with self.subTest(url=str(r.url)):
                for secret in (PASSWORD, NEW_PASSWORD):
                    self.assertNotIn(secret, r.text)
                    self.assertNotIn(secret, str(dict(r.headers)))

    def test_the_verifier_is_never_returned_either(self):
        self.login()
        self.change(NEW_PASSWORD, current=PASSWORD)
        verifier = self.store.panel_password_verifier()
        for r in (self.client.get('/admin/session'),
                  self.client.get('/config/effective', headers={'x-api-key': API_KEY})):
            with self.subTest(url=str(r.url)):
                self.assertNotIn(verifier, r.text)

    def test_changing_it_logs_neither_password_nor_verifier(self):
        self.login()
        with self.assertLogs('main', level='DEBUG') as logs:
            logging.getLogger('main').debug('anchor')
            self.change(NEW_PASSWORD, current=PASSWORD)
            self.change(NEW_PASSWORD, current='wrong-but-long-enough')
        joined = '\n'.join(logs.output)
        for secret in (PASSWORD, NEW_PASSWORD, self.store.panel_password_verifier()):
            with self.subTest(secret=secret[:6]):
                self.assertNotIn(secret, joined)

    def test_the_settings_endpoint_never_carries_the_password(self):
        # /config/effective hands the whole settings dict to any valid key, so
        # the password must not be in it — same reason LB_CREDENTIAL_KEY is not.
        body = self.client.get('/config/effective', headers={'x-api-key': API_KEY}).json()
        self.assertNotIn(main.PANEL_PASSWORD_ENV, str(body))


if __name__ == '__main__':
    unittest.main()
