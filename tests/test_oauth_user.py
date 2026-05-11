"""
Tests for larkhelm.oauth_user — token persistence + refresh + status.

These tests deliberately avoid the loopback HTTP server flow (``cli_login``);
that's interactive and is covered by manual verification instead. The lazy
in-process consumers (``is_token_valid`` / ``get_user_token``) are the
critical paths because every ``create_doc`` call funnels through them.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import larkhelm.config as _cfg
import larkhelm.oauth_user as ou


def _reset_module(token_path: Path) -> None:
    """Point oauth_user at a clean tempdir and flush the in-process cache."""
    _cfg.USER_TOKEN_PATH = token_path
    _cfg.APP_ID = "cli_test_app_id"
    _cfg.APP_SECRET = "cli_test_app_secret"
    _cfg.LOGGED_IN_OPEN_ID = ""
    ou._clear_cache()


def _write_token_file(token_path: Path, **overrides) -> dict:
    """Write a synthetic token file with sensible defaults; return the dict."""
    now = int(time.time())
    data = {
        "access_token":         "u-access-XXX",
        "token_type":           "Bearer",
        "refresh_token":        "ur-refresh-YYY",
        "expires_at":           now + 7200,         # 2h ahead
        "refresh_expires_at":   now + 2_592_000,    # 30d ahead
        "open_id":              "ou_synthetic_xxx",
        "scope":                "docx:document docx:document:create",
        "saved_at":             now,
    }
    data.update(overrides)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


class OAuthUserTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="larkhelm_oauth_"))
        self.token_path = self.tmpdir / "user_token.json"
        _reset_module(self.token_path)

    def tearDown(self):
        ou._clear_cache()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── _save_token / _load_token ───────────────────────────────────────

    def test_save_token_writes_atomically_and_caches(self):
        data = {"access_token": "u-x", "expires_at": 1, "open_id": "ou_x"}
        ou._save_token(data)
        self.assertTrue(self.token_path.exists())
        loaded = json.loads(self.token_path.read_text())
        self.assertEqual(loaded["access_token"], "u-x")
        # cache should also be primed
        self.assertEqual(ou._token_cache, data)

    def test_save_token_chmod_0600(self):
        """The persisted token must not be world-readable."""
        if os.name != "posix":
            self.skipTest("POSIX-only chmod check")
        ou._save_token({"access_token": "u-x", "expires_at": 1, "open_id": "ou_x"})
        mode = stat.S_IMODE(self.token_path.stat().st_mode)
        self.assertEqual(mode & 0o077, 0,
                         f"token file has loose perms: 0o{mode:o}")

    def test_load_token_missing_file_returns_none(self):
        self.assertIsNone(ou._load_token())

    def test_load_token_corrupted_returns_none(self):
        self.token_path.write_text("{not json")
        self.assertIsNone(ou._load_token())

    def test_load_token_missing_required_field_returns_none(self):
        self.token_path.write_text(json.dumps({"only": "junk"}))
        self.assertIsNone(ou._load_token())

    def test_load_token_caches_after_first_read(self):
        _write_token_file(self.token_path)
        first = ou._load_token()
        # Delete the underlying file — cached reads must still work.
        self.token_path.unlink()
        second = ou._load_token()
        self.assertIs(first, second)

    # ── is_token_valid ──────────────────────────────────────────────────

    def test_is_token_valid_returns_false_when_no_file(self):
        self.assertFalse(ou.is_token_valid())

    def test_is_token_valid_returns_true_for_fresh_token(self):
        _write_token_file(self.token_path)
        self.assertTrue(ou.is_token_valid())

    def test_is_token_valid_returns_false_for_expired(self):
        _write_token_file(self.token_path, expires_at=int(time.time()) - 60)
        self.assertFalse(ou.is_token_valid())

    def test_is_token_valid_does_not_hit_network(self):
        """is_token_valid must be cheap — verified by patching urlopen to error."""
        _write_token_file(self.token_path)
        with patch("urllib.request.urlopen",
                   side_effect=AssertionError("network hit forbidden")):
            self.assertTrue(ou.is_token_valid())

    # ── get_user_token: fast path + refresh + expiry ────────────────────

    def test_get_user_token_returns_access_when_fresh(self):
        _write_token_file(self.token_path)
        self.assertEqual(ou.get_user_token(), "u-access-XXX")

    def test_get_user_token_returns_none_when_no_file(self):
        self.assertIsNone(ou.get_user_token())

    def test_get_user_token_refreshes_when_near_expiry(self):
        """Within REFRESH_MARGIN — must call refresh_token endpoint, then return new token."""
        now = int(time.time())
        _write_token_file(self.token_path,
                          access_token="u-OLD",
                          expires_at=now + 60)  # 60s left, < 600s margin

        def fake_app_token():
            return "app-fake"

        def fake_refresh(refresh_token):
            self.assertEqual(refresh_token, "ur-refresh-YYY")
            return {
                "access_token":       "u-NEW",
                "token_type":         "Bearer",
                "refresh_token":      "ur-refresh-NEW",
                "expires_at":         int(time.time()) + 7200,
                "refresh_expires_at": int(time.time()) + 2_592_000,
                "open_id":            "ou_synthetic_xxx",
                "scope":              "docx:document",
                "saved_at":           int(time.time()),
            }

        with patch.object(ou, "_get_app_access_token", side_effect=fake_app_token), \
             patch.object(ou, "_refresh_token_call",   side_effect=fake_refresh):
            tok = ou.get_user_token()
        self.assertEqual(tok, "u-NEW")
        # File must be updated too — the next call should NOT trigger refresh.
        with patch.object(ou, "_refresh_token_call",
                          side_effect=AssertionError("should not be called")):
            self.assertEqual(ou.get_user_token(), "u-NEW")

    def test_get_user_token_deletes_file_when_refresh_token_dead(self):
        """If refresh_token is also expired we must wipe + return None."""
        now = int(time.time())
        _write_token_file(self.token_path,
                          expires_at=now - 60,
                          refresh_expires_at=now - 30)
        with patch.object(ou, "_refresh_token_call",
                          side_effect=AssertionError("must not be called")):
            self.assertIsNone(ou.get_user_token())
        self.assertFalse(self.token_path.exists(),
                         "dead token file should be removed so caller falls back cleanly")

    def test_get_user_token_returns_none_on_refresh_failure(self):
        """Network error during refresh must NOT raise — caller fallback depends on it."""
        now = int(time.time())
        _write_token_file(self.token_path, expires_at=now + 60)
        with patch.object(ou, "_get_app_access_token", return_value="app-fake"), \
             patch.object(ou, "_refresh_token_call",
                          side_effect=RuntimeError("simulated 500")):
            self.assertIsNone(ou.get_user_token())
        # File preserved — we don't punish transient failures by forcing re-login.
        self.assertTrue(self.token_path.exists())

    # ── clear_token + get_status ────────────────────────────────────────

    def test_clear_token_removes_file_and_cache(self):
        _write_token_file(self.token_path)
        ou._load_token()
        self.assertIsNotNone(ou._token_cache)
        ou.clear_token()
        self.assertFalse(self.token_path.exists())
        self.assertIsNone(ou._token_cache)

    def test_clear_token_is_idempotent(self):
        # No file → no error
        ou.clear_token()
        self.assertFalse(self.token_path.exists())

    def test_get_status_logged_out(self):
        self.assertEqual(ou.get_status(), {"logged_in": False})

    def test_get_status_logged_in_shape(self):
        _write_token_file(self.token_path)
        st = ou.get_status()
        self.assertTrue(st["logged_in"])
        self.assertEqual(st["open_id"], "ou_synthetic_xxx")
        self.assertIn("docx:document", st["scope"])
        self.assertGreater(st["expires_in_sec"], 0)
        self.assertGreater(st["refresh_expires_in_sec"], 0)

    # ── _normalize_token_response ───────────────────────────────────────

    def test_normalize_token_response_converts_durations(self):
        before = int(time.time())
        out = ou._normalize_token_response({
            "code": 0,
            "data": {
                "access_token":          "u-fresh",
                "token_type":            "Bearer",
                "refresh_token":         "ur-fresh",
                "expires_in":            7200,
                "refresh_expires_in":    2_592_000,
                "open_id":               "ou_a",
                "scope":                 "docx:document",
            }
        })
        self.assertEqual(out["access_token"], "u-fresh")
        self.assertEqual(out["open_id"], "ou_a")
        # Within tolerance — clock between the two calls.
        self.assertTrue(out["expires_at"] >= before + 7200)
        self.assertTrue(out["refresh_expires_at"] >= before + 2_592_000)

    # ── _build_authorize_url ────────────────────────────────────────────

    def test_build_authorize_url_contains_required_params(self):
        url = ou._build_authorize_url(
            "http://127.0.0.1:12345/callback", "state-abc")
        self.assertIn(ou.AUTHORIZE_URL, url)
        self.assertIn("app_id=cli_test_app_id", url)
        self.assertIn("redirect_uri=http%3A%2F%2F127.0.0.1%3A12345%2Fcallback", url)
        self.assertIn("state=state-abc", url)
        self.assertIn("response_type=code", url)
        # Scope is space-joined; argparse-urlencoded as ``+``.
        self.assertIn("scope=docx", url)

    # ── _validate_callback (#23 — covers all 4 exit branches) ───────────

    def test_validate_callback_success_returns_none(self):
        self.assertIsNone(
            ou._validate_callback("ok-code", "S", "", "S"))

    def test_validate_callback_error_param_returns_4(self):
        verdict = ou._validate_callback("", "S", "access_denied", "S")
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict[0], 4)
        self.assertIn("access_denied", verdict[1])

    def test_validate_callback_state_mismatch_returns_5(self):
        verdict = ou._validate_callback("ok-code", "OTHER", "", "EXPECTED")
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict[0], 5)
        self.assertIn("CSRF", verdict[1])

    def test_validate_callback_missing_code_returns_6(self):
        verdict = ou._validate_callback("", "S", "", "S")
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict[0], 6)

    def test_validate_callback_state_mismatch_takes_priority_over_missing_code(self):
        """If state is wrong, abort before looking at code — strict CSRF semantics."""
        verdict = ou._validate_callback("", "OTHER", "", "EXPECTED")
        self.assertEqual(verdict[0], 5)

    def test_validate_callback_err_param_takes_priority(self):
        """Error reported by Feishu wins over any other anomaly."""
        verdict = ou._validate_callback("", "OTHER", "user_cancelled", "EXPECTED")
        self.assertEqual(verdict[0], 4)

    # ── _exchange_code (#23 — mock both HTTP layers) ────────────────────

    def test_exchange_code_success_returns_normalized_dict(self):
        fake_response = {
            "code": 0,
            "data": {
                "access_token":       "u-fresh-XXX",
                "token_type":         "Bearer",
                "refresh_token":      "ur-fresh-YYY",
                "expires_in":         7200,
                "refresh_expires_in": 2_592_000,
                "open_id":            "ou_exchange_test",
                "scope":              "docx:document",
            }
        }
        with patch.object(ou, "_get_app_access_token", return_value="app-fake"), \
             patch.object(ou, "_http_post_json", return_value=fake_response):
            out = ou._exchange_code("AUTHCODE", "http://127.0.0.1:1/callback")
        self.assertEqual(out["access_token"], "u-fresh-XXX")
        self.assertEqual(out["open_id"], "ou_exchange_test")
        self.assertGreater(out["expires_at"], int(time.time()) + 7000)

    def test_exchange_code_raises_when_feishu_returns_error_code(self):
        with patch.object(ou, "_get_app_access_token", return_value="app-fake"), \
             patch.object(ou, "_http_post_json",
                          return_value={"code": 99999, "msg": "bad code"}):
            with self.assertRaises(RuntimeError) as ctx:
                ou._exchange_code("BADCODE", "http://127.0.0.1:1/callback")
        # Error message must be sanitized — must NOT include the response dict
        # repr (which on a non-error path could carry a refresh_token).
        self.assertIn("99999", str(ctx.exception))
        self.assertNotIn("'refresh_token'", str(ctx.exception))

    # ── _safe_resp_summary (#3 — token redaction) ───────────────────────

    def test_safe_resp_summary_drops_token_fields(self):
        resp = {
            "code": 0,
            "msg":  "success",
            "data": {"access_token": "u-LEAK", "refresh_token": "ur-LEAK"},
        }
        summary = ou._safe_resp_summary(resp)
        self.assertIn("code=0", summary)
        self.assertIn("success", summary)
        self.assertNotIn("u-LEAK",  summary)
        self.assertNotIn("ur-LEAK", summary)

    def test_safe_resp_summary_handles_non_dict(self):
        self.assertIn("non-dict", ou._safe_resp_summary("a string"))  # type: ignore[arg-type]

    # ── mtime cache invalidation (#10) ──────────────────────────────────

    def test_load_token_picks_up_external_rewrite(self):
        """Another process overwrites the file → fast-path must re-read it."""
        _write_token_file(self.token_path, access_token="u-OLD")
        first = ou._load_token()
        self.assertEqual(first["access_token"], "u-OLD")
        # Simulate an external rewrite. ``time.sleep`` isn't required because we
        # explicitly stamp a future mtime.
        new_data = dict(first)
        new_data["access_token"] = "u-NEW"
        self.token_path.write_text(json.dumps(new_data, indent=2))
        future = time.time() + 10
        os.utime(self.token_path, (future, future))
        second = ou._load_token()
        self.assertEqual(second["access_token"], "u-NEW",
                         "mtime change must invalidate cache")

    def test_load_token_returns_cached_when_file_deleted_mid_flight(self):
        """If the file vanishes between two reads, the in-process cache is
        retained so concurrent threads mid-refresh aren't suddenly logged out
        on a transient FS hiccup."""
        _write_token_file(self.token_path)
        first = ou._load_token()
        self.token_path.unlink()
        second = ou._load_token()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
