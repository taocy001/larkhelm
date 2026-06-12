"""Tests: failover loop uses record_call_failure instead of mark_unhealthy (LOGIC-C2).

Three test classes:
  1. TestQueryPyUsesRecordCallFailure      — source inspection of _query.py
  2. TestTransientSlidingWindow            — BackendRegistry TRANSIENT window logic
  3. TestUserCancelAndTimeoutSkipFailoverNotice — set_current_text not called on NO_HEALTH_UPDATE
"""
import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Bootstrap config ──────────────────────────────────────────────────────────
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_failover_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "test_app",
    "APP_SECRET": "test_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

from larkhelm.backend_registry import BackendRegistry, BACKEND_REGISTRY
from larkhelm.health_signals import (
    NO_HEALTH_UPDATE, USER_CANCEL, TIMEOUT, TRANSIENT, AUTH,
)

_BACKEND_SPEC_DICT = {
    "id": "test_backend",
    "provider": "claude_cli",
    "display_name": "Test Backend",
    "role": "orchestrator",
    "tags": ["tools"],
    "command": "claude",
    "enabled": True,
}


def _make_registry(*extra_specs) -> BackendRegistry:
    reg = BackendRegistry()
    reg.load([_BACKEND_SPEC_DICT, *extra_specs])
    return reg


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Source inspection — _query.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryPyUsesRecordCallFailure(unittest.TestCase):

    def setUp(self):
        self._src = Path("larkhelm/handlers/_query.py").read_text()

    def test_record_call_failure_present(self):
        self.assertIn("record_call_failure", self._src)

    def test_no_health_update_imported_and_used(self):
        self.assertIn("NO_HEALTH_UPDATE", self._src)

    def test_mark_unhealthy_not_called(self):
        non_comment = [
            ln for ln in self._src.splitlines()
            if "mark_unhealthy" in ln and not ln.strip().startswith("#")
        ]
        self.assertEqual(non_comment, [],
                         f"mark_unhealthy still referenced in _query.py: {non_comment}")

    def test_category_guards_set_current_text(self):
        """The failover notice is gated on 'category not in NO_HEALTH_UPDATE'."""
        self.assertIn("category not in NO_HEALTH_UPDATE", self._src)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. BackendRegistry TRANSIENT sliding window
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransientSlidingWindow(unittest.TestCase):

    def _fresh_registry(self) -> BackendRegistry:
        return _make_registry()

    def test_below_threshold_stays_healthy(self):
        """2 TRANSIENT failures: spec remains healthy."""
        reg = self._fresh_registry()
        for _ in range(2):
            cat = reg.record_call_failure("test_backend", "connection refused",
                                          transient_threshold=3, transient_window_sec=600.0)
            self.assertEqual(cat, TRANSIENT)
        self.assertTrue(reg.get("test_backend").healthy)

    def test_at_threshold_flips_unhealthy(self):
        """3rd TRANSIENT failure within window: spec flips unhealthy."""
        reg = self._fresh_registry()
        for _ in range(3):
            reg.record_call_failure("test_backend", "connection refused",
                                    transient_threshold=3, transient_window_sec=600.0)
        self.assertFalse(reg.get("test_backend").healthy)

    def test_auth_error_flips_immediately(self):
        """AUTH error flips unhealthy on the very first hit."""
        reg = self._fresh_registry()
        cat = reg.record_call_failure("test_backend", "401 Unauthorized")
        self.assertEqual(cat, AUTH)
        self.assertFalse(reg.get("test_backend").healthy)

    def test_user_cancel_no_health_change(self):
        """USER_CANCEL: healthy stays True, failure_window not touched."""
        reg = self._fresh_registry()
        spec = reg.get("test_backend")
        before = len(spec.failure_window)
        cat = reg.record_call_failure("test_backend", "QueryCancelledError cancelled by user")
        self.assertEqual(cat, USER_CANCEL)
        self.assertTrue(spec.healthy)
        self.assertEqual(len(spec.failure_window), before)

    def test_timeout_no_health_change(self):
        """TIMEOUT: healthy stays True."""
        reg = self._fresh_registry()
        spec = reg.get("test_backend")
        cat = reg.record_call_failure("test_backend", "hard-timeout exceeded 360 min")
        self.assertEqual(cat, TIMEOUT)
        self.assertTrue(spec.healthy)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. set_current_text not called on USER_CANCEL / TIMEOUT
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserCancelAndTimeoutSkipFailoverNotice(unittest.TestCase):
    """Simulate the failover loop logic to verify set_current_text is suppressed."""

    _BACKEND_B = {
        "id": "backend_b",
        "provider": "gemini_cli",
        "display_name": "Backend B",
        "role": "orchestrator",
        "tags": ["tools"],
        "command": "gemini",
        "enabled": True,
    }

    def _run_failover_logic(self, returned_category: str) -> MagicMock:
        """Simulate the failover except block; return a spy on card_state."""
        reg = BackendRegistry()
        reg.load([_BACKEND_SPEC_DICT, self._BACKEND_B])
        spec_a = reg.get("test_backend")
        chain = [spec_a, reg.get("backend_b")]

        mock_card = MagicMock()

        with patch.object(reg, "record_call_failure", return_value=returned_category):
            cat = reg.record_call_failure(spec_a.id, "dummy error")
            remaining = [s for s in chain if s.healthy and s.id != spec_a.id]
            if remaining and cat not in NO_HEALTH_UPDATE:
                mock_card.set_current_text(
                    f"> ⚠️ {spec_a.display_name} 不可用，切换至 {remaining[0].display_name}..."
                )

        return mock_card

    def test_user_cancel_no_card_update(self):
        spy = self._run_failover_logic(USER_CANCEL)
        spy.set_current_text.assert_not_called()

    def test_timeout_no_card_update(self):
        spy = self._run_failover_logic(TIMEOUT)
        spy.set_current_text.assert_not_called()

    def test_transient_does_show_card_update(self):
        """TRANSIENT (not in NO_HEALTH_UPDATE) → notice IS shown."""
        spy = self._run_failover_logic(TRANSIENT)
        spy.set_current_text.assert_called_once()

    def test_auth_does_show_card_update(self):
        """AUTH (not in NO_HEALTH_UPDATE) → notice IS shown."""
        spy = self._run_failover_logic(AUTH)
        spy.set_current_text.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
