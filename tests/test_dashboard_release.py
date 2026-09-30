"""Public release identity is bounded and never claims GitHub freshness. Author: Zeno Ren."""
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient
import build_metadata
import main


class DashboardReleaseTests(unittest.TestCase):
    def test_dashboard_html_is_not_cached_across_releases(self):
        result = TestClient(main.app).get('/stats/dashboard')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers.get('cache-control'), 'no-store')

    def identity(self, **changes):
        return {'available': True, 'source_revision': 'a' * 40,
                'source_tree_dirty': False, 'out_of_tree': False,
                'source_attested': True, 'runtime_files_match': True,
                'manifest_covers_current_runtime': True, **changes}

    def request(self, identity):
        with patch.object(build_metadata, 'runtime_identity', return_value=identity):
            return TestClient(main.app).get('/version')

    def test_public_identity_matches_commit_without_exposing_build_internals(self):
        result = self.request(self.identity(packages={'private': 'SYNTHETIC_SECRET'},
                                            files={'config.yaml': 'SYNTHETIC_SECRET'}))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers['cache-control'], 'no-store')
        body = result.json()
        self.assertEqual(body['version'], 'aaaaaaa')
        self.assertEqual(body['verification'], 'matched')
        self.assertEqual(body['commit_url'], 'https://github.com/ZenoRewn/databricks-claude-lb/commit/' + 'a' * 40)
        self.assertNotIn('SYNTHETIC_SECRET', result.text)
        self.assertNotIn('latest', body)
        self.assertEqual(set(body), {'version', 'source_revision', 'verification', 'commit_url', 'compare_url'})

    def test_unknown_local_build_has_no_invented_version_or_link(self):
        result = self.request({'available': False})
        self.assertEqual(result.status_code, 200)
        self.assertIsNone(result.json()['version'])
        self.assertIsNone(result.json()['commit_url'])
        self.assertEqual(result.json()['verification'], 'unknown')

    def test_modified_partial_and_unattested_sources_are_not_marked_matching(self):
        for changes, expected in [({'runtime_files_match': False}, 'modified'),
                                  ({'source_tree_dirty': True}, 'modified'),
                                  ({'out_of_tree': True}, 'modified'),
                                  ({'manifest_covers_current_runtime': False}, 'unknown'),
                                  ({'source_attested': False}, 'unknown'),
                                  ({'source_tree_dirty': None}, 'unknown')]:
            with self.subTest(changes=changes):
                self.assertEqual(self.request(self.identity(**changes)).json()['verification'], expected)

    def test_invalid_revision_is_not_reflected_into_links_or_text(self):
        for revision in ('<script>SYNTHETIC_SECRET</script>', '../main', 'a' * 41, None):
            with self.subTest(revision=revision):
                body = self.request(self.identity(source_revision=revision)).json()
                self.assertIsNone(body['source_revision'])
                self.assertIsNone(body['commit_url'])
                self.assertEqual(body['verification'], 'unknown')


if __name__ == '__main__':
    unittest.main()
