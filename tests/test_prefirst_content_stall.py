"""Observable stall before the first content byte. Author: Zeno Ren.

The 2026-10-09 incident was only visible after upstream cancelled the stream at
~120s, read back from RST_STREAM(CANCEL). Surfacing the stall while it is still
in progress makes the condition observable before the upstream timer fires.
Purely diagnostic: no stream is aborted, no retry is issued, no circuit moves.
"""
import dataclasses
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class PrefirstContentStallTests(unittest.IsolatedAsyncioTestCase):
    BASE_MONOTONIC = 1_000_000.0

    def setUp(self):
        self.now = self.BASE_MONOTONIC
        self.clock = patch.object(main, 'time', SimpleNamespace(monotonic=lambda: self.now, time=time.time))
        self.clock.start()
        self.proxy = object.__new__(main.CopilotProxy)
        self.proxy._stream_connections = {}
        self.proxy.stream_prefirst_content_stall_total = 0

    def tearDown(self):
        self.clock.stop()

    def _register(self):
        cid = 'c1'
        self.proxy._stream_connections[cid] = {
            'endpoint': 'synthetic', 'task': None, 'release': None,
            'disconnect_checker': None, 'started_at': self.now,
            'last_upstream_activity_at': self.now, 'disconnected_since': None,
            'headers_at': None, 'content_at': None, 'prefirst_stall_reported': False,
        }
        return cid

    def test_no_signal_before_headers_arrive(self):
        cid = self._register()
        self.now += 600
        self.assertEqual(self.proxy._note_prefirst_content_stall(cid, 90), 0)
        self.assertEqual(self.proxy.stream_prefirst_content_stall_total, 0)

    def test_signal_once_after_threshold_without_content(self):
        cid = self._register()
        self.proxy._note_stream_headers(cid)
        self.now += 90
        self.assertEqual(self.proxy._note_prefirst_content_stall(cid, 90), 1)
        self.assertEqual(self.proxy.stream_prefirst_content_stall_total, 1)
        # Still stalled, but already reported: the signal must not repeat.
        self.now += 300
        self.assertEqual(self.proxy._note_prefirst_content_stall(cid, 90), 0)
        self.assertEqual(self.proxy.stream_prefirst_content_stall_total, 1)

    def test_no_signal_when_content_arrived(self):
        cid = self._register()
        self.proxy._note_stream_headers(cid)
        self.now += 10
        self.proxy._note_stream_content(cid)
        self.now += 600
        self.assertEqual(self.proxy._note_prefirst_content_stall(cid, 90), 0)
        self.assertEqual(self.proxy.stream_prefirst_content_stall_total, 0)

    def test_below_threshold_is_silent(self):
        cid = self._register()
        self.proxy._note_stream_headers(cid)
        self.now += 89
        self.assertEqual(self.proxy._note_prefirst_content_stall(cid, 90), 0)
        self.assertEqual(self.proxy.stream_prefirst_content_stall_total, 0)

    def test_threshold_zero_disables_the_signal(self):
        cid = self._register()
        self.proxy._note_stream_headers(cid)
        self.now += 600
        self.assertEqual(self.proxy._note_prefirst_content_stall(cid, 0), 0)
        self.assertEqual(self.proxy.stream_prefirst_content_stall_total, 0)

    def test_diagnostic_fields_survive_the_safe_field_filter(self):
        # A field absent from the allowlist is dropped silently, which would leave
        # the log saying a stall happened without saying for how long.
        import safe_diagnostics
        fields = safe_diagnostics.safe_fields({
            'kind': 'copilot_stream_prefirst_content_stall',
            'endpoint': 'synthetic', 'upstream_headers_received': True,
            'seconds_since_headers': 95.5, 'threshold_seconds': 90,
        })
        self.assertEqual(fields.get('seconds_since_headers'), 95.5)
        self.assertEqual(fields.get('threshold_seconds'), 90)

    def test_unknown_connection_is_ignored(self):
        self.assertEqual(self.proxy._note_prefirst_content_stall('missing', 90), 0)
        self.assertEqual(self.proxy._note_prefirst_content_stall(None, 90), 0)

    def test_notes_are_safe_on_unknown_connection(self):
        self.proxy._note_stream_headers('missing')
        self.proxy._note_stream_content(None)
        self.assertEqual(self.proxy._stream_connections, {})

    def test_content_note_does_not_overwrite_first_content_time(self):
        cid = self._register()
        self.proxy._note_stream_headers(cid)
        self.proxy._note_stream_content(cid)
        first = self.proxy._stream_connections[cid]['content_at']
        self.now += 50
        self.proxy._note_stream_content(cid)
        self.assertEqual(self.proxy._stream_connections[cid]['content_at'], first)

    async def test_registered_stream_carries_the_new_fields(self):
        # _register_stream captures asyncio.current_task(), so it needs a loop.
        proxy = object.__new__(main.CopilotProxy)
        proxy._stream_connections = {}
        ep = main.CopilotEndpoint('synthetic', '', models=['a'])

        async def release():
            return None

        cid = main.CopilotProxy._register_stream(proxy, ep, release)
        item = proxy._stream_connections[cid]
        for key in ('headers_at', 'content_at', 'prefirst_stall_reported'):
            self.assertIn(key, item)
        self.assertIsNone(item['headers_at'])
        self.assertIsNone(item['content_at'])
        self.assertFalse(item['prefirst_stall_reported'])

    def test_setting_is_range_validated(self):
        for bad in (-1, 3601):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                main.LBSettings(**{**self._settings_kwargs(),
                                   'copilot_stream_prefirst_content_warn_seconds': bad})

    def _settings_kwargs(self):
        current = main.LB_SETTINGS
        return {f.name: getattr(current, f.name) for f in dataclasses.fields(current)}


if __name__ == '__main__':
    unittest.main()
