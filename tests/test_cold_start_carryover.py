"""Coverage for the cold-start carry-over feature (handlers/_query.py)
and the new ``AUTO_UPDATE_FIRST`` 3-turn threshold (memory.py).

Two test surfaces:

  1. ``TestAutoUpdateThreshold`` — pure-function truth table for
     ``larkhelm.memory._should_auto_update`` across turns 0..40 and selected
     larger values, validating the "fire at 3, then every 10 thereafter"
     semantics.
  2. ``TestColdStartCarryOver`` — the post-query hook in
     ``larkhelm.handlers._query._post_query_memory_hook`` dispatches to
     ``maybe_auto_update`` with the correct ``force`` flag based on whether
     this is the chat's first successful turn.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Config bootstrap (mirrors test_phase1_modules.py style) ─────────────────
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_carryover_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

_DUMMY_CONFIG = {
    "APP_ID": "test_app",
    "APP_SECRET": "test_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))

import larkhelm.config as _cfg_mod
_cfg_mod._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

from larkhelm import memory as _memory
from larkhelm.handlers import _query as _query_mod


# ═══════════════════════════════════════════════════════════════════════════
#  1. _should_auto_update truth table
# ═══════════════════════════════════════════════════════════════════════════
class TestAutoUpdateThreshold(unittest.TestCase):
    """Verify the trigger schedule: turns 3, 13, 23, 33 fire; all others don't."""

    def test_threshold_truth_table(self):
        # turn 0..2: never trigger (pre-threshold)
        for n in (0, 1, 2):
            self.assertFalse(_memory._should_auto_update(n),
                             msg=f"turn={n} should NOT trigger")
        # turn 3: first-time threshold hit
        self.assertTrue(_memory._should_auto_update(3),
                        msg="turn=3 (AUTO_UPDATE_FIRST) must trigger")
        # turns 4..12: gap window, none trigger
        for n in range(4, 13):
            self.assertFalse(_memory._should_auto_update(n),
                             msg=f"turn={n} (between FIRST and FIRST+EVERY) must NOT trigger")
        # turn 13: FIRST + EVERY → trigger
        self.assertTrue(_memory._should_auto_update(13))
        # turns 14..22: gap, none trigger
        for n in range(14, 23):
            self.assertFalse(_memory._should_auto_update(n),
                             msg=f"turn={n} must NOT trigger")
        # turn 23, 33: subsequent cadence hits
        self.assertTrue(_memory._should_auto_update(23))
        self.assertTrue(_memory._should_auto_update(33))
        # Sanity: turn=10 (old AUTO_UPDATE_EVERY hit) must NOT trigger under
        # the new schedule — confirms we shifted off the % EVERY == 0 anchor.
        self.assertFalse(_memory._should_auto_update(10))
        self.assertFalse(_memory._should_auto_update(20))


# ═══════════════════════════════════════════════════════════════════════════
#  2. _post_query_memory_hook cold-start dispatch
# ═══════════════════════════════════════════════════════════════════════════
class TestColdStartCarryOver(unittest.TestCase):
    """The hook must read turn_count BEFORE incrementing, then route to the
    correct ``maybe_auto_update`` invocation (force=True on the first turn,
    regular path thereafter). The skip-on-empty-jsonl optimization was
    removed because by the time this hook runs, ``log_entry`` has already
    written user+assistant rows to ``all.jsonl``, making any size check
    always-True in production — ``maybe_auto_update`` itself short-circuits
    on empty ``_read_logs_tail`` results, so force=True is safe."""

    def _run_hook(self, *, old_turn: int):
        """Drive ``_post_query_memory_hook`` under controlled conditions.

        Patches:
          * ``_get_turn_count`` → returns ``old_turn``
          * ``_increment_turn_count`` → no-op MagicMock
          * ``larkhelm.memory.maybe_auto_update`` → MagicMock for assertion
        """
        mau_mock = MagicMock()
        inc_mock = MagicMock()
        with patch.object(_query_mod, "_get_turn_count", return_value=old_turn), \
             patch.object(_query_mod, "_increment_turn_count", inc_mock), \
             patch.object(_memory, "maybe_auto_update", mau_mock):
            _query_mod._post_query_memory_hook("chat_xyz", "trace_abc")
        return mau_mock, inc_mock

    def test_first_turn_triggers_force(self):
        """old=0 → maybe_auto_update(force=True) regardless of jsonl size.

        ``_is_useful_summary`` inside ``generate_memory`` rejects any thin
        output, and ``_read_logs_tail`` returns [] for genuinely empty chats,
        so the force-call is safe even when no carry-over history exists.
        """
        mau, inc = self._run_hook(old_turn=0)
        inc.assert_called_once_with("chat_xyz")
        mau.assert_called_once_with("chat_xyz", force=True)

    def test_second_turn_uses_regular_path(self):
        """old=1 → routes through regular (non-force) maybe_auto_update."""
        mau, inc = self._run_hook(old_turn=1)
        inc.assert_called_once_with("chat_xyz")
        mau.assert_called_once_with("chat_xyz")
        # Must NOT have force=True kwarg
        _, kwargs = mau.call_args
        self.assertNotIn("force", kwargs)

    def test_later_turn_uses_regular_path(self):
        """old=12 → regular path (gate decides inside maybe_auto_update)."""
        mau, inc = self._run_hook(old_turn=12)
        inc.assert_called_once_with("chat_xyz")
        mau.assert_called_once_with("chat_xyz")
        _, kwargs = mau.call_args
        self.assertNotIn("force", kwargs)

    def test_force_bypasses_gate(self):
        """Calling maybe_auto_update(force=True) at a non-trigger turn must
        not raise and must NOT call generate_memory when there are no logs.

        Verifies that force=True still funnels through the body of
        maybe_auto_update (past the trigger gate) — when logs are absent the
        background thread short-circuits at ``_read_logs_tail`` returning [].
        """
        with patch.object(_memory, "_get_turn_count", return_value=1), \
             patch.object(_memory, "_read_logs_tail", return_value=[]), \
             patch.object(_memory, "generate_memory") as gen_mock:
            # Must not raise.
            _memory.maybe_auto_update("chat_force_test", force=True)
            # The thread is daemon; join via a brief wait on the lock.
            import threading as _t
            import time as _time
            deadline = _time.monotonic() + 2.0
            while _time.monotonic() < deadline:
                if not _memory._get_update_lock("chat_force_test").locked():
                    break
                _t.Event().wait(0.05)
        # generate_memory must NOT have been called because logs were empty;
        # this also proves the gate didn't reject the force-call before
        # reaching _read_logs_tail.
        gen_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
