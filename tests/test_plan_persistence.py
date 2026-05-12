"""Tests for ``larkhelm.plan_persistence`` (U17 — bridge-restart recovery).

Background
----------
A ``/plan`` that's running when the bridge dies (cgroup OOM-kill, crash,
SIGKILL) loses its in-memory state. The Feishu progress card becomes a
permanent "⏳ 思考中" ghost — the user has no way to know the plan died.
U17 fixes the user-perception half by persisting plan state to disk on
every step transition, then scanning + sending one notification card per
interrupted plan on bridge startup.

Scope of this test module:
  * save / list / delete round-trip with realistic ``MultiPlanState``
  * schema-version mismatch is silently skipped on load
  * corrupted / unreadable files don't crash the startup scanner
  * notification card content reflects the persisted step the plan
    died on (active step description, count of N/M, age)
  * the ``plan_persist_clear`` button callback deletes the file and
    is idempotent
"""
from __future__ import annotations

import atexit
import dataclasses
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Config bootstrap so DATA_DIR exists for plan_persistence reads ─────
_TMP = tempfile.mkdtemp(prefix="larkhelm_planpersist_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

# We import the module under test AFTER bootstrap so it sees a real
# DATA_DIR. Note _DIRNAME is constant — each test below uses
# ``_state_dir()`` to find/clean the live dir.
from larkhelm import plan_persistence as pp
from larkhelm.cmd_plan import MultiPlanState, PlanStep


def _make_state(plan_id="test_plan_001", chat_id="oc_test",
                title="MyPlan", steps=None, phase="running",
                current_idx=0) -> MultiPlanState:
    """Build a realistic ``MultiPlanState`` for round-trip tests."""
    if steps is None:
        steps = [
            PlanStep(idx=0, type="dev", desc="实现 X", status="done"),
            PlanStep(idx=1, type="review", desc="审查 X", status="running"),
            PlanStep(idx=2, type="fix", desc="修问题", status="pending"),
        ]
    return MultiPlanState(
        plan_id=plan_id, chat_id=chat_id, title=title,
        steps=steps, phase=phase, current_idx=current_idx,
    )


def _clean_state_dir():
    """Wipe the persistence dir before/after each test for isolation.

    Production uses a single shared ``DATA_DIR/_active_plans/`` —
    tests inherit that path via the bootstrap config. Each test must
    start from an empty dir to avoid cross-contamination from a
    previous test's pending state.
    """
    try:
        d = pp._state_dir()
        if d.exists():
            for f in d.iterdir():
                f.unlink(missing_ok=True)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════
#  Round-trip + schema handling
# ════════════════════════════════════════════════════════════════════════

class SaveListDeleteTests(unittest.TestCase):

    def setUp(self):
        _clean_state_dir()

    def tearDown(self):
        _clean_state_dir()

    def test_save_then_list_returns_serialised_record(self):
        state = _make_state()
        pp.save_plan_state(state)
        records = pp.list_pending_plan_states()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["plan_id"], "test_plan_001")
        self.assertEqual(rec["chat_id"], "oc_test")
        self.assertEqual(rec["title"],   "MyPlan")
        self.assertEqual(rec["phase"],   "running")
        self.assertEqual(len(rec["steps"]), 3)
        self.assertEqual(rec["steps"][0]["status"], "done")
        self.assertEqual(rec["steps"][1]["desc"],   "审查 X")
        self.assertEqual(rec["schema_version"], pp._STATE_SCHEMA_VERSION)

    def test_save_is_atomic_via_tmp_then_replace(self):
        """A reader observing the directory should never see a half-written
        ``<plan_id>.json``. Verified by checking the implementation writes
        to ``.json.tmp`` then atomically renames."""
        state = _make_state()
        pp.save_plan_state(state)
        # tmp file should be gone after a successful write
        d = pp._state_dir()
        self.assertEqual(list(d.glob("*.json.tmp")), [])

    def test_save_overwrites_previous_record_idempotently(self):
        state = _make_state(phase="running")
        pp.save_plan_state(state)
        state.phase = "done"
        state.current_idx = 2
        pp.save_plan_state(state)
        records = pp.list_pending_plan_states()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["phase"], "done")
        self.assertEqual(records[0]["current_idx"], 2)

    def test_delete_removes_file(self):
        pp.save_plan_state(_make_state("p_aaa"))
        self.assertEqual(len(pp.list_pending_plan_states()), 1)
        pp.delete_plan_state("p_aaa")
        self.assertEqual(pp.list_pending_plan_states(), [])

    def test_delete_is_idempotent_when_file_missing(self):
        # No file exists; must not raise
        pp.delete_plan_state("never_existed")
        # Sanity: still empty
        self.assertEqual(pp.list_pending_plan_states(), [])

    def test_save_strips_non_serialisable_threading_state(self):
        """``MultiPlanState`` contains ``threading.Event`` / ``Lock`` that
        can't be JSON-serialised. The save path must not try to include
        them — verified by reading the on-disk file and asserting only
        the schema-allowed keys are present.

        ``notify_count`` / ``last_notified_at`` were added in the
        round-2 follow-up (#11) as forward-compatible fields for
        startup-notification throttling. ``save_plan_state`` initialises
        them to 0 / 0.0 on every write since the plan thread itself never
        notifies — only the startup notifier does.
        """
        state = _make_state()
        pp.save_plan_state(state)
        raw = (pp._state_dir() / f"{state.plan_id}.json").read_text()
        data = json.loads(raw)
        allowed = {"schema_version", "plan_id", "chat_id", "title", "phase",
                   "current_idx", "start_time", "saved_at", "steps",
                   "notify_count", "last_notified_at"}
        self.assertEqual(set(data.keys()), allowed,
            f"on-disk schema must be exactly {allowed!r}, got {set(data.keys())!r}")
        self.assertEqual(data["notify_count"], 0)
        self.assertEqual(data["last_notified_at"], 0.0)

    def test_error_field_truncated_to_200_chars(self):
        """Long error strings (e.g. tool-output dumps) would bloat the
        state file. Each step's ``error`` is capped at 200 chars during
        serialise — guard the regression."""
        step = PlanStep(idx=0, type="dev", desc="x", status="failed",
                        error="X" * 500)
        state = _make_state(steps=[step])
        pp.save_plan_state(state)
        rec = pp.list_pending_plan_states()[0]
        self.assertEqual(len(rec["steps"][0]["error"]), 200)


# ════════════════════════════════════════════════════════════════════════
#  Schema version + corruption handling
# ════════════════════════════════════════════════════════════════════════

class SchemaToleranceTests(unittest.TestCase):

    def setUp(self):
        _clean_state_dir()

    def tearDown(self):
        _clean_state_dir()

    def test_wrong_schema_version_silently_skipped(self):
        """An older bridge build wrote schema v0; current scanner must skip
        rather than crash. Future schema bumps rely on this behaviour."""
        d = pp._state_dir()
        (d / "future.json").write_text(json.dumps({
            "schema_version": 99,
            "plan_id":   "future",
            "chat_id":   "oc_x",
            "title":     "from-the-future",
            "phase":     "running",
            "current_idx": 0,
            "saved_at":   time.time(),
            "steps":     [],
        }), encoding="utf-8")
        # Plus a valid record so the scanner returns SOMETHING
        pp.save_plan_state(_make_state("good_one"))
        records = pp.list_pending_plan_states()
        ids = [r["plan_id"] for r in records]
        self.assertIn("good_one", ids)
        self.assertNotIn("future", ids)

    def test_corrupted_json_silently_skipped(self):
        d = pp._state_dir()
        (d / "broken.json").write_text("{not even json", encoding="utf-8")
        pp.save_plan_state(_make_state("good_one"))
        records = pp.list_pending_plan_states()
        self.assertEqual([r["plan_id"] for r in records], ["good_one"])

    def test_non_dict_top_level_silently_skipped(self):
        """``json.loads([1, 2, 3])`` succeeds but isn't a record dict."""
        d = pp._state_dir()
        (d / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
        records = pp.list_pending_plan_states()
        self.assertEqual(records, [])

    def test_empty_state_dir_returns_empty_list(self):
        # Defense in depth — fresh DATA_DIR with no plans
        self.assertEqual(pp.list_pending_plan_states(), [])


# ════════════════════════════════════════════════════════════════════════
#  resume_interrupted_plans (the startup notifier)
# ════════════════════════════════════════════════════════════════════════

class ResumeNotifierTests(unittest.TestCase):

    def setUp(self):
        _clean_state_dir()

    def tearDown(self):
        _clean_state_dir()

    def test_no_pending_plans_sends_no_card(self):
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 0)
        card.assert_not_called()

    def test_one_pending_plan_sends_one_card_with_step_info(self):
        state = _make_state(plan_id="p_one", chat_id="oc_X",
                            title="OAuth Setup", current_idx=1)
        pp.save_plan_state(state)
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 1)
        card.assert_called_once()
        args, kwargs = card.call_args
        chat_id, title, body = args[:3]
        self.assertEqual(chat_id, "oc_X")
        self.assertIn("Plan 被中断", title)
        self.assertIn("OAuth Setup", body)
        # The active step at current_idx=1 was "审查 X"
        self.assertIn("审查 X", body)
        # N / M counter
        self.assertIn("2 / 3", body)
        # Active step type label
        self.assertIn("[review]", body)
        # Button payload format must match handlers/_card_action.py
        self.assertIn("buttons", kwargs)
        button_value = kwargs["buttons"][0][1]
        self.assertEqual(button_value, "plan_persist_clear:p_one")

    def test_multiple_pending_plans_each_get_their_own_card(self):
        for i in range(3):
            pp.save_plan_state(_make_state(plan_id=f"p_{i}",
                                           chat_id=f"oc_{i}"))
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 3)
        self.assertEqual(card.call_count, 3)
        # Each card went to a different chat
        chat_ids = {c.args[0] for c in card.call_args_list}
        self.assertEqual(chat_ids, {"oc_0", "oc_1", "oc_2"})

    def test_one_bad_record_does_not_block_others(self):
        """One Feishu-side failure must not block the remaining notifications."""
        for i in range(3):
            pp.save_plan_state(_make_state(plan_id=f"p_{i}", chat_id=f"oc_{i}"))

        # Make the first send_card raise; subsequent must still fire
        call_count = {"n": 0}
        def _flaky_send(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("network blip")
        with patch("larkhelm.lark_client.send_card", side_effect=_flaky_send):
            sent = pp.resume_interrupted_plans()
        # 2 succeeded out of 3 attempted (3 call_count, but only 2 returned successfully)
        self.assertEqual(sent, 2)
        self.assertEqual(call_count["n"], 3)

    def test_card_body_age_humanises_minutes_and_hours(self):
        """The body shows "X 分钟前" or "X 小时前" depending on age. Patch
        the saved_at to known offsets and check both branches render."""
        state = _make_state(plan_id="p_minute_age")
        pp.save_plan_state(state)
        # Force saved_at to 12 minutes ago
        path = pp._state_dir() / "p_minute_age.json"
        rec = json.loads(path.read_text())
        rec["saved_at"] = time.time() - 12 * 60
        path.write_text(json.dumps(rec))
        with patch("larkhelm.lark_client.send_card") as card:
            pp.resume_interrupted_plans()
        body = card.call_args.args[2]
        self.assertIn("12 分钟前", body)
        _clean_state_dir()

        state2 = _make_state(plan_id="p_hour_age")
        pp.save_plan_state(state2)
        path2 = pp._state_dir() / "p_hour_age.json"
        rec2 = json.loads(path2.read_text())
        rec2["saved_at"] = time.time() - 3 * 3600 - 30
        path2.write_text(json.dumps(rec2))
        with patch("larkhelm.lark_client.send_card") as card2:
            pp.resume_interrupted_plans()
        body2 = card2.call_args.args[2]
        self.assertIn("3 小时前", body2)

    def test_card_body_handles_missing_chat_id(self):
        """A pathological record without chat_id is silently skipped —
        sending a card to ``""`` would hit a Feishu API error."""
        d = pp._state_dir()
        (d / "no_chat.json").write_text(json.dumps({
            "schema_version": pp._STATE_SCHEMA_VERSION,
            "plan_id":   "no_chat",
            "chat_id":   "",          # malformed
            "title":     "??",
            "phase":     "running",
            "current_idx": 0,
            "saved_at":   time.time(),
            "steps":     [],
        }), encoding="utf-8")
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 0)
        card.assert_not_called()


# ════════════════════════════════════════════════════════════════════════
#  clear_plan_state_button — the card callback
# ════════════════════════════════════════════════════════════════════════

class TerminalPhaseSkipTests(unittest.TestCase):
    """Round-2 follow-up #15: a state file whose ``phase`` is ``done`` /
    ``failed`` / ``cancelled`` means the plan thread reached its
    completion-card emission and either (a) the finally-block delete
    raced with SIGKILL or (b) ``finalize_workspace`` itself blocked
    long enough to take a SIGKILL during it. Either way, the user
    already saw the proper terminal card — surfacing this as
    "interrupted" would be a false alarm. Skip + auto-delete.
    """

    def setUp(self):
        _clean_state_dir()

    def tearDown(self):
        _clean_state_dir()

    def _save_phase(self, plan_id: str, phase: str) -> None:
        """Write a record directly with the given phase (since
        ``save_plan_state`` always pulls phase off a live MultiPlanState)."""
        state = _make_state(plan_id=plan_id, phase=phase)
        pp.save_plan_state(state)

    def test_done_phase_skipped_and_file_removed(self):
        self._save_phase("p_done", "done")
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 0)
        card.assert_not_called()
        # File must be auto-removed so subsequent scans don't re-evaluate
        self.assertFalse((pp._state_dir() / "p_done.json").exists(),
            "terminal-phase file must be GC'd on first scan")

    def test_failed_phase_skipped_and_file_removed(self):
        self._save_phase("p_failed", "failed")
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 0)
        card.assert_not_called()
        self.assertFalse((pp._state_dir() / "p_failed.json").exists())

    def test_cancelled_phase_skipped_and_file_removed(self):
        self._save_phase("p_cancelled", "cancelled")
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 0)
        card.assert_not_called()
        self.assertFalse((pp._state_dir() / "p_cancelled.json").exists())

    def test_running_phase_still_notified(self):
        """Sanity: the existing happy path still works — only terminal
        phases are skipped, ``running`` (and ``waiting``) still notify."""
        self._save_phase("p_running", "running")
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 1)
        card.assert_called_once()
        # And the file is RETAINED so the user's "🗑️ 清除提示" button works
        self.assertTrue((pp._state_dir() / "p_running.json").exists())

    def test_terminal_skip_takes_priority_over_throttle(self):
        """A terminal-phase record bypasses notification regardless of
        notify_count / cooldown — we want it gone, not delayed."""
        self._save_phase("p_done_throttled", "done")
        # Force throttle state that would otherwise allow notify
        path = pp._state_dir() / "p_done_throttled.json"
        rec = json.loads(path.read_text())
        rec["notify_count"] = 0
        rec["last_notified_at"] = 0.0
        path.write_text(json.dumps(rec))
        with patch("larkhelm.lark_client.send_card") as card:
            pp.resume_interrupted_plans()
        card.assert_not_called()
        self.assertFalse(path.exists())


class FloodingThrottleTests(unittest.TestCase):
    """Round-2 follow-up #11: bridge crash-loop must not spam the user.

    Throttle rules (see ``_FLOOD_THROTTLE_SEC`` / ``_MAX_NOTIFY_COUNT``):
      * After a successful notification, ``notify_count`` increments
        and ``last_notified_at`` updates.
      * Within ``_FLOOD_THROTTLE_SEC`` of the last send, subsequent
        scans see the record but skip the send (file retained).
      * After ``_MAX_NOTIFY_COUNT`` total notifications, the file is
        auto-deleted — the user clearly isn't going to triage.
    """

    def setUp(self):
        _clean_state_dir()

    def tearDown(self):
        _clean_state_dir()

    def _save_with_throttle_state(self, plan_id: str, notify_count: int,
                                  last_notified_at: float) -> None:
        """Write a running-phase record with prescribed throttle fields."""
        state = _make_state(plan_id=plan_id, phase="running")
        pp.save_plan_state(state)
        path = pp._state_dir() / f"{plan_id}.json"
        rec = json.loads(path.read_text())
        rec["notify_count"]     = notify_count
        rec["last_notified_at"] = last_notified_at
        path.write_text(json.dumps(rec))

    def test_first_notify_succeeds_and_writes_count_1(self):
        pp.save_plan_state(_make_state(plan_id="p_first"))
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 1)
        # Throttle state updated
        rec = json.loads((pp._state_dir() / "p_first.json").read_text())
        self.assertEqual(rec["notify_count"], 1)
        self.assertGreater(rec["last_notified_at"], 0)
        card.assert_called_once()

    def test_recent_notification_is_throttled_within_window(self):
        """Last sent 60s ago, window is 30 min → skip."""
        self._save_with_throttle_state(
            "p_recent", notify_count=1, last_notified_at=time.time() - 60)
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 0)
        card.assert_not_called()
        # File retained so next restart can re-evaluate
        self.assertTrue((pp._state_dir() / "p_recent.json").exists())

    def test_old_notification_allows_resend_after_window(self):
        """Last sent over 30 min ago — throttle window elapsed, allow."""
        self._save_with_throttle_state(
            "p_aged", notify_count=1,
            last_notified_at=time.time() - (pp._FLOOD_THROTTLE_SEC + 60))
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 1)
        card.assert_called_once()
        # Updated count
        rec = json.loads((pp._state_dir() / "p_aged.json").read_text())
        self.assertEqual(rec["notify_count"], 2)

    def test_max_notify_count_triggers_auto_gc(self):
        """After _MAX_NOTIFY_COUNT (3) notifications, the file is
        auto-deleted regardless of cooldown — user clearly isn't acting."""
        self._save_with_throttle_state(
            "p_maxed", notify_count=pp._MAX_NOTIFY_COUNT,
            last_notified_at=time.time() - (pp._FLOOD_THROTTLE_SEC + 60))
        with patch("larkhelm.lark_client.send_card") as card:
            sent = pp.resume_interrupted_plans()
        self.assertEqual(sent, 0)
        card.assert_not_called()
        # GC'd
        self.assertFalse((pp._state_dir() / "p_maxed.json").exists())

    def test_throttle_persists_across_multiple_scans(self):
        """Three rapid bridge restarts must produce **at most** one card
        (the very first), the next two are throttled."""
        pp.save_plan_state(_make_state(plan_id="p_loop"))
        # First restart — should notify
        with patch("larkhelm.lark_client.send_card") as card1:
            pp.resume_interrupted_plans()
        self.assertEqual(card1.call_count, 1)
        # Immediately re-scan (simulating crash + auto-restart within
        # seconds): must be throttled.
        with patch("larkhelm.lark_client.send_card") as card2:
            pp.resume_interrupted_plans()
        self.assertEqual(card2.call_count, 0)
        with patch("larkhelm.lark_client.send_card") as card3:
            pp.resume_interrupted_plans()
        self.assertEqual(card3.call_count, 0)
        # File still around, awaiting cooldown OR user action
        self.assertTrue((pp._state_dir() / "p_loop.json").exists())

    def test_notify_count_lost_on_save_plan_state_resave(self):
        """If a plan thread is still alive somehow during throttle window
        and ``save_plan_state`` runs again, ``_serialise`` resets
        notify_count to 0 — by design, since save_plan_state means the
        plan is running again and any prior "interrupted" notification
        was for a now-superseded process. Document this for future
        readers."""
        # Simulate first notification
        pp.save_plan_state(_make_state(plan_id="p_resave"))
        with patch("larkhelm.lark_client.send_card"):
            pp.resume_interrupted_plans()
        rec1 = json.loads((pp._state_dir() / "p_resave.json").read_text())
        self.assertEqual(rec1["notify_count"], 1)
        # Plan thread re-saves (e.g. step transition after a phantom
        # resume): _serialise resets the throttle fields
        pp.save_plan_state(_make_state(plan_id="p_resave"))
        rec2 = json.loads((pp._state_dir() / "p_resave.json").read_text())
        self.assertEqual(rec2["notify_count"], 0)
        self.assertEqual(rec2["last_notified_at"], 0.0)

    def test_persist_notify_state_is_no_op_when_file_vanishes(self):
        """User races the button: file is deleted between send_card and
        the throttle-write. ``_persist_notify_state`` must respect the
        deletion (don't re-create the file)."""
        # Pretend a record exists, capture the path, then delete it
        pp.save_plan_state(_make_state(plan_id="p_race"))
        path = pp._state_dir() / "p_race.json"
        path.unlink()
        # Must not raise + must not re-create the file
        pp._persist_notify_state("p_race", notify_count=1,
                                 last_notified_at=time.time())
        self.assertFalse(path.exists())


class ClearButtonTests(unittest.TestCase):

    def setUp(self):
        _clean_state_dir()

    def tearDown(self):
        _clean_state_dir()

    def test_clear_returns_true_when_file_existed(self):
        pp.save_plan_state(_make_state("p_clear_yes"))
        result = pp.clear_plan_state_button("p_clear_yes")
        self.assertTrue(result)
        self.assertEqual(pp.list_pending_plan_states(), [])

    def test_clear_returns_false_when_no_file(self):
        result = pp.clear_plan_state_button("never_existed")
        self.assertFalse(result)

    def test_clear_is_idempotent(self):
        pp.save_plan_state(_make_state("p_clear_dup"))
        pp.clear_plan_state_button("p_clear_dup")
        # Second call must not raise
        pp.clear_plan_state_button("p_clear_dup")


if __name__ == "__main__":
    unittest.main()
