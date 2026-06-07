"""Security boundary tests for larkhelm.

Covers:
  1. /cd: symlink pointing outside cwd_root is rejected.
  2. /run: stdout display is truncated to 2000 chars.
  3. redact_error: sk-* and ds-* tokens are redacted; new patterns work.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("LARKHELM_TEST_MODE", "1")


# ── 1. /cd symlink escape ─────────────────────────────────────────────


class CdSymlinkEscapeTests(unittest.TestCase):
    """_check_cwd_root must reject symlinks that resolve outside cwd_root."""

    def test_symlink_outside_root_is_rejected(self):
        import tempfile
        from larkhelm.commands import _check_cwd_root
        import larkhelm.config as _cfg

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "allowed"
            outside = Path(td) / "outside"
            root.mkdir()
            outside.mkdir()

            link = root / "escape_link"
            link.symlink_to(outside)

            original = _cfg.config.get("cwd_root")
            try:
                _cfg.config["cwd_root"] = str(root)
                self.assertFalse(
                    _check_cwd_root(link),
                    "symlink pointing outside cwd_root must be rejected",
                )
            finally:
                if original is None:
                    _cfg.config.pop("cwd_root", None)
                else:
                    _cfg.config["cwd_root"] = original

    def test_normal_path_inside_root_is_allowed(self):
        import tempfile
        from larkhelm.commands import _check_cwd_root
        import larkhelm.config as _cfg

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "allowed"
            subdir = root / "project" / "src"
            subdir.mkdir(parents=True)

            original = _cfg.config.get("cwd_root")
            try:
                _cfg.config["cwd_root"] = str(root)
                self.assertTrue(_check_cwd_root(subdir))
            finally:
                if original is None:
                    _cfg.config.pop("cwd_root", None)
                else:
                    _cfg.config["cwd_root"] = original

    def test_no_cwd_root_configured_always_allowed(self):
        import tempfile
        from larkhelm.commands import _check_cwd_root
        import larkhelm.config as _cfg

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "anywhere"
            p.mkdir()

            original = _cfg.config.get("cwd_root")
            try:
                _cfg.config.pop("cwd_root", None)
                self.assertTrue(_check_cwd_root(p))
            finally:
                if original is not None:
                    _cfg.config["cwd_root"] = original


# ── 2. /run stdout truncation ─────────────────────────────────────────


class RunOutputTruncationTests(unittest.TestCase):
    """stdout from /run is truncated to 2000 chars before display."""

    def test_stdout_truncated_to_2000(self):
        long_output = "x" * 5000
        self.assertEqual(len(long_output.strip()[:2000]), 2000)

    def test_short_stdout_not_truncated(self):
        short_output = "hello world\n"
        self.assertEqual(short_output.strip()[:2000], "hello world")

    def test_exactly_2000_chars_not_truncated(self):
        exact_output = "a" * 2000
        self.assertEqual(len(exact_output.strip()[:2000]), 2000)

    def test_cmd_run_body_contains_truncated_stdout(self):
        from larkhelm import commands

        long_stdout = "Z" * 4000
        captured_bodies: list[str] = []

        with (
            patch.object(commands, "_run_shell", lambda chat_id, cmd: (long_stdout, "", 0)),
            patch.object(commands, "send_card_reply", lambda *a, **kw: "mid_1"),
            patch.object(commands, "reply_card", lambda chat_id, mid, title, body, **kw: captured_bodies.append(body)),
            patch.object(commands, "log_entry", lambda *a, **kw: None),
        ):
            commands._cmd_run("oc_test", "echo test", msg_id="m1")

        self.assertTrue(captured_bodies, "reply_card must have been called")
        body = captured_bodies[0]
        self.assertNotIn("Z" * 2001, body, "body must not contain >2000 Zs")
        self.assertIn("Z" * 100, body, "body must contain the (truncated) stdout")


# ── 3. redact_error patterns ─────────────────────────────────────────


class RedactErrorPatternTests(unittest.TestCase):
    """Verify existing and new redact_error patterns."""

    def setUp(self):
        from larkhelm.log import redact_error
        self.redact = redact_error

    def test_sk_token_redacted(self):
        raw = "auth failed: sk-1234567890abcdefghijklmnopqrstuvwxyz extra"
        out = self.redact(raw)
        self.assertNotIn("sk-1234567890abcdefghijklmnopqrstuvwxyz", out)
        self.assertIn("sk-***", out)

    def test_ds_token_redacted(self):
        ds_token = "ds-" + "a" * 40
        out = self.redact(f"deepseek error: {ds_token} not authorized")
        self.assertNotIn(ds_token, out)
        self.assertIn("ds-***", out)

    def test_short_ds_not_redacted(self):
        raw = "debug: ds-short something"
        self.assertIn("ds-short", self.redact(raw))

    def test_app_secret_json_form_redacted(self):
        raw = '{"APP_SECRET": "cli_abcdef1234567890longvalue", "other": "x"}'
        out = self.redact(raw)
        self.assertNotIn("cli_abcdef1234567890longvalue", out)
        self.assertIn("***", out)

    def test_app_id_json_form_redacted(self):
        raw = '{"APP_ID": "cli_xxxxxxxxxxxxxxxx", "key": "val"}'
        out = self.redact(raw)
        self.assertNotIn("cli_xxxxxxxxxxxxxxxx", out)
        self.assertIn("***", out)

    def test_app_secret_short_value_not_redacted(self):
        raw = '{"APP_SECRET": "short"}'
        self.assertIn("short", self.redact(raw))

    def test_unrelated_json_key_not_redacted(self):
        raw = '{"OTHER_KEY": "someverylongvalue123456"}'
        self.assertIn("someverylongvalue123456", self.redact(raw))

    def test_token_eq_redacted(self):
        token_val = "t-" + "b" * 30
        out = self.redact(f"request failed token={token_val} at endpoint")
        self.assertNotIn(token_val, out)
        self.assertIn("***", out)

    def test_token_colon_redacted(self):
        token_val = "Bearer_" + "c" * 30
        out = self.redact(f"auth error: token: {token_val} rejected")
        self.assertNotIn(token_val, out)
        self.assertIn("***", out)

    def test_short_token_value_not_redacted(self):
        self.assertIn("shortval", self.redact("token=shortval info"))

    def test_does_not_redact_normal_prose(self):
        raw = "received valid token from server"
        self.assertEqual(self.redact(raw), raw)

    def test_redact_idempotent(self):
        raw = '{"APP_SECRET": "cli_longvalue12345678"}'
        once = self.redact(raw)
        self.assertEqual(once, self.redact(once))


if __name__ == "__main__":
    unittest.main()
