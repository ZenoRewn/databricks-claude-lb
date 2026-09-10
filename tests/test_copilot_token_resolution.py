"""`resolve_github_token` 的三级回退优先级与静默替换告警。

为什么值得单独覆盖：这条链决定**请求以哪个 GitHub 账户发出**。它此前零覆盖，而它
有一个容易踩的静默行为 —— config 里显式写了 `${ENV}` 但变量未设时，会回退到本地
缓存文件。K8s 里那个目录正是 `copilot-cache` secret 的挂载点，所以运维可能以为在用
刚 rotate 的 Secret，实际跑的是缓存里的旧 token。回退本身是有意的（提供韧性），
但必须有日志痕迹，否则就是一次无声的凭证替换。

优先级契约：
  1) config.yaml 的 `github_token`（`${ENV}` 或明文）
  2) 本项目 device-flow 缓存 `<COPILOT_AUTH_DIR>/copilot-auth-<name>.json`
  3) 兼容 copilot-lb 旧缓存 `~/.config/copilot-lb/auth.json`
  都没有 → `RuntimeError`（fail closed，不静默跑一个没凭证的 endpoint）
"""
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main

ENV_KEY = "GITHUB_COPILOT_TOKEN_TEST_ONLY"


class _IsolatedAuthDirs(unittest.TestCase):
    """把两个缓存目录指到临时目录，绝不读写用户真实凭证。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name)
        self.auth_dir = root / "dclb"
        self.auth_dir.mkdir()
        self.legacy = root / "copilot-lb" / "auth.json"
        self.legacy.parent.mkdir()
        self.addCleanup(self._tmp.cleanup)
        for p in (patch.object(main, "COPILOT_AUTH_DIR", self.auth_dir),
                  patch.object(main, "COPILOT_LEGACY_AUTH_FILE", self.legacy)):
            p.start()
            self.addCleanup(p.stop)
        self._saved_env = os.environ.pop(ENV_KEY, None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        os.environ.pop(ENV_KEY, None)
        if self._saved_env is not None:
            os.environ[ENV_KEY] = self._saved_env

    def write_local(self, name, token, key="github_token"):
        path = self.auth_dir / f"copilot-auth-{name}.json"
        path.write_text(json.dumps({key: token}))
        return path

    def write_legacy(self, token, key="github_token"):
        self.legacy.write_text(json.dumps({key: token}))


class PriorityTests(_IsolatedAuthDirs):

    def test_config_env_ref_wins_over_both_caches(self):
        os.environ[ENV_KEY] = "tok-from-env"
        self.write_local("a", "tok-from-local")
        self.write_legacy("tok-from-legacy")
        token, source = main.resolve_github_token(
            {"name": "a", "github_token": f"${{{ENV_KEY}}}"})
        self.assertEqual(token, "tok-from-env")
        self.assertEqual(source, {"type": "env", "key": ENV_KEY})

    def test_literal_config_token_is_not_rereadable(self):
        token, source = main.resolve_github_token(
            {"name": "a", "github_token": "ghu_literal"})
        self.assertEqual(token, "ghu_literal")
        self.assertEqual(source, {"type": "literal"},
                         "明文 token 无法运行时重读，source 必须如实标注")

    def test_local_cache_wins_over_legacy(self):
        self.write_local("a", "tok-from-local")
        self.write_legacy("tok-from-legacy")
        token, source = main.resolve_github_token({"name": "a"})
        self.assertEqual(token, "tok-from-local")
        self.assertEqual(source["type"], "file")
        self.assertIn("copilot-auth-a.json", source["path"])

    def test_legacy_cache_is_the_last_resort(self):
        self.write_legacy("tok-from-legacy")
        token, source = main.resolve_github_token({"name": "a"})
        self.assertEqual(token, "tok-from-legacy")
        self.assertEqual(source["path"], str(self.legacy))

    def test_per_endpoint_cache_is_name_scoped(self):
        """多账号时不能互相读到对方的缓存。"""
        self.write_local("acct-1", "tok-1")
        self.write_local("acct-2", "tok-2")
        self.assertEqual(main.resolve_github_token({"name": "acct-1"})[0], "tok-1")
        self.assertEqual(main.resolve_github_token({"name": "acct-2"})[0], "tok-2")

    def test_unsafe_endpoint_name_is_sanitised_into_the_filename(self):
        """endpoint 名进文件名前必须消毒，否则 `../` 之类能越出目录。"""
        path = main._copilot_local_auth_path("../../etc/pwn")
        self.assertEqual(path.parent, self.auth_dir)
        self.assertNotIn("/", path.name.replace("copilot-auth-", "").replace(".json", ""))

    def test_no_source_at_all_fails_loud(self):
        with self.assertRaises(RuntimeError) as cm:
            main.resolve_github_token({"name": "nobody"})
        self.assertIn("--copilot-login", str(cm.exception),
                      "报错要给出可执行的下一步")

    def test_camel_case_key_in_cache_file_is_accepted(self):
        """copilot-lb 的缓存用 githubToken 拼写，兼容读取。"""
        self.write_legacy("tok-camel", key="githubToken")
        self.assertEqual(main.resolve_github_token({"name": "a"})[0], "tok-camel")

    def test_corrupt_cache_file_falls_through_instead_of_crashing(self):
        (self.auth_dir / "copilot-auth-a.json").write_text("{not json")
        self.write_legacy("tok-from-legacy")
        self.assertEqual(main.resolve_github_token({"name": "a"})[0], "tok-from-legacy")


class SilentSubstitutionWarningTests(_IsolatedAuthDirs):
    """回退不改，但必须留下痕迹。"""

    def test_unset_env_ref_falling_back_to_cache_is_warned(self):
        self.write_local("a", "stale-cached-token")
        with self.assertLogs(main.logger, level="WARNING") as logs:
            token, source = main.resolve_github_token(
                {"name": "a", "github_token": f"${{{ENV_KEY}}}"})
        self.assertEqual(token, "stale-cached-token", "回退行为本身不变")
        self.assertEqual(source["type"], "file")
        joined = "\n".join(logs.output)
        self.assertIn(ENV_KEY, joined)
        self.assertIn("NOT in use", joined,
                      "必须明确告诉运维：刚 rotate 的 Secret 没生效")

    def test_partially_unresolved_embedded_ref_is_warned(self):
        """`ghu_${SUFFIX}` 在 SUFFIX 未设时会返回残缺 token，上游必 401 且无线索。"""
        with self.assertLogs(main.logger, level="WARNING") as logs:
            token, source = main.resolve_github_token(
                {"name": "a", "github_token": f"ghu_${{{ENV_KEY}}}"})
        self.assertEqual(token, "ghu_", "行为不变：残缺 token 仍被返回")
        self.assertEqual(source["type"], "literal")
        self.assertIn(ENV_KEY, "\n".join(logs.output))

    def test_fully_resolved_embedded_ref_does_not_warn(self):
        os.environ[ENV_KEY] = "xyz"
        with patch.object(main.logger, "warning") as warn:
            token, _ = main.resolve_github_token(
                {"name": "a", "github_token": f"ghu_${{{ENV_KEY}}}"})
        self.assertEqual(token, "ghu_xyz")
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
