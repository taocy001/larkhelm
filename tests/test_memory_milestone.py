"""Coverage for the milestone-driven memory auto-update path.

Why this exists: ``maybe_auto_update`` only fires from ``_do_query`` (normal
chat queries) when ``turn_count % 10 == 0``. ``/dev``, ``/crew``, ``/plan``
complete WITHOUT going through ``_do_query``, so important architecture
decisions made during those tasks would otherwise wait up to 10 chat turns
to be captured into memory. ``record_milestone`` closes that gap by:

1. Logging the milestone with ``role="milestone"`` so the next memory
   regenerate (or the immediate force-trigger below) sees it.
2. Force-triggering ``maybe_auto_update`` (debounced).

These tests exercise the helper directly + the role-filter inclusion in
``maybe_auto_update``.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from larkhelm import memory


def _seed_jsonl_logs(path: Path, records: list[dict]) -> None:
    """Write fake all.jsonl entries the same way ``log_entry`` does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class TestRecordMilestone(unittest.TestCase):
    """``record_milestone`` writes a log entry + force-triggers update."""

    def setUp(self):
        # Reset debounce state so each test starts clean.
        memory._last_milestone_ts.clear()

    def test_log_entry_called_with_milestone_role(self):
        with patch("larkhelm.log.log_entry") as le, \
             patch.object(memory, "maybe_auto_update") as mau:
            memory.record_milestone("oc_test", "dev", summary="实现 X 功能")
        le.assert_called_once()
        # Positional: chat_id, role, content. Kwargs: model="milestone".
        args, kwargs = le.call_args
        # log_entry signature: (chat_id, role, content, model=..., trace_id=...)
        self.assertEqual(args[0], "oc_test")
        self.assertEqual(args[1], "milestone")
        self.assertIn("[Milestone] dev", args[2])
        self.assertIn("实现 X 功能", args[2])
        self.assertEqual(kwargs.get("model"), "milestone")
        mau.assert_called_once_with("oc_test", force=True)

    def test_summary_truncated_to_200(self):
        long_summary = "x" * 500
        with patch("larkhelm.log.log_entry") as le, \
             patch.object(memory, "maybe_auto_update"):
            memory.record_milestone("oc_test", "crew", summary=long_summary)
        content = le.call_args.args[2]
        # Has the [Milestone] prefix + ": " + at most 200 chars of summary.
        self.assertLessEqual(len(content), len("[Milestone] crew: ") + 200 + 5)
        self.assertIn("xxx", content)  # not entirely empty

    def test_no_summary_renders_kind_only(self):
        with patch("larkhelm.log.log_entry") as le, \
             patch.object(memory, "maybe_auto_update"):
            memory.record_milestone("oc_test", "plan", summary="")
        content = le.call_args.args[2]
        self.assertEqual(content, "[Milestone] plan")

    def test_log_entry_failure_does_not_block_trigger(self):
        """Even if the log write fails, the auto-update trigger must still fire."""
        with patch("larkhelm.log.log_entry", side_effect=OSError("disk full")), \
             patch.object(memory, "maybe_auto_update") as mau:
            memory.record_milestone("oc_test", "dev")
        mau.assert_called_once()

    def test_trigger_failure_swallowed(self):
        """maybe_auto_update raising must NOT propagate (the milestone task
        itself must keep running through its own finally block)."""
        with patch("larkhelm.log.log_entry"), \
             patch.object(memory, "maybe_auto_update",
                          side_effect=RuntimeError("memory module dead")):
            try:
                memory.record_milestone("oc_test", "dev")
            except Exception as e:
                self.fail(f"record_milestone propagated {e!r}")


class TestMilestoneDebounce(unittest.TestCase):
    """Two milestones < ``_MILESTONE_DEBOUNCE_SEC`` apart must trigger
    only one ``maybe_auto_update``. Log entry still written each time."""

    def setUp(self):
        memory._last_milestone_ts.clear()

    def test_second_call_within_debounce_window_skipped(self):
        with patch("larkhelm.log.log_entry") as le, \
             patch.object(memory, "maybe_auto_update") as mau:
            memory.record_milestone("oc_test", "dev")
            memory.record_milestone("oc_test", "dev")  # immediate second
        # Both events logged …
        self.assertEqual(le.call_count, 2)
        # … but only the first one triggered the LLM regenerate.
        self.assertEqual(mau.call_count, 1)

    def test_second_call_after_debounce_window_runs(self):
        with patch("larkhelm.log.log_entry"), \
             patch.object(memory, "maybe_auto_update") as mau, \
             patch.object(memory, "_MILESTONE_DEBOUNCE_SEC", 0):
            # Window=0 means every call passes the debounce gate.
            memory.record_milestone("oc_test", "dev")
            memory.record_milestone("oc_test", "dev")
        self.assertEqual(mau.call_count, 2)

    def test_debounce_is_per_chat(self):
        """Two different chats must NOT debounce against each other."""
        with patch("larkhelm.log.log_entry"), \
             patch.object(memory, "maybe_auto_update") as mau:
            memory.record_milestone("oc_a", "dev")
            memory.record_milestone("oc_b", "dev")
        self.assertEqual(mau.call_count, 2)


class TestMaybeAutoUpdateFilter(unittest.TestCase):
    """``maybe_auto_update`` must include ``role="milestone"`` entries in
    the log_text it sends to the LLM summarizer. Previously only
    user/assistant were whitelisted, so milestone records would be
    invisible even though they were in the JSONL file.

    This is a white-box test of the filter inside ``maybe_auto_update``.
    ``maybe_auto_update`` runs the LLM in a daemon thread → another inner
    thread; instead of trying to synchronize against that, we extract the
    same comprehension here and verify both roles + model exclusions match
    the production code byte-for-byte. The grep below pins the production
    expression so a future edit must update both."""

    def _filter_records(self, records: list[dict]) -> str:
        """Mirror the comprehension in ``maybe_auto_update._run`` exactly."""
        return "\n".join(
            f"[{r['ts']}] {r['role']}: {r['content'][:600]}"
            for r in records
            if r["role"] in ("user", "assistant", "milestone")
            and r.get("model") not in ("crew", "shell")
        )

    def test_filter_keeps_milestone_drops_crew_and_shell(self):
        ts = "2026-05-09T22:00:00"
        records = [
            {"ts": ts, "role": "user", "content": "hi", "model": "claude"},
            {"ts": ts, "role": "assistant", "content": "hello", "model": "claude"},
            # Crew task — should be filtered (model=crew in skiplist).
            {"ts": ts, "role": "user", "content": "/dev X", "model": "crew"},
            # Shell — both role and model excluded.
            {"ts": ts, "role": "shell", "content": "ls", "model": "claude"},
            # Milestone — must NOT be filtered (this is the new behavior).
            {"ts": ts, "role": "milestone",
             "content": "[Milestone] dev: 实现 X", "model": "milestone"},
            # System — not whitelisted.
            {"ts": ts, "role": "system", "content": "boot", "model": "claude"},
        ]
        log_text = self._filter_records(records)
        self.assertIn("user: hi", log_text)
        self.assertIn("assistant: hello", log_text)
        self.assertIn("[Milestone] dev", log_text)
        self.assertNotIn("/dev X", log_text)         # model=crew excluded
        self.assertNotIn("shell:", log_text)         # role not whitelisted
        self.assertNotIn("system: boot", log_text)   # role not whitelisted

    def test_production_filter_signature_pinned(self):
        """Pin the in-source filter expression so a future edit that drops
        ``"milestone"`` from the role whitelist (or removes the model
        skiplist) trips this test, forcing the author to revisit
        ``record_milestone`` semantics."""
        import inspect
        src = inspect.getsource(memory.maybe_auto_update)
        # Both the role whitelist and model exclusion must be present.
        self.assertIn('("user", "assistant", "milestone")', src,
                      "milestone role missing from maybe_auto_update filter")
        self.assertIn('not in ("crew", "shell")', src,
                      "crew/shell skiplist drifted; review semantics first")


class TestRoundtripWithRealLogEntry(unittest.TestCase):
    """End-to-end (no mocks): record_milestone → real log_entry → real
    _read_logs → milestone record visible in the same JSONL bag the
    summarizer would consume."""

    def test_milestone_visible_in_read_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "feishu_logs"
            with patch.object(memory._cfg, "LOG_DIR", log_dir, create=True), \
                 patch.object(memory, "maybe_auto_update"):
                memory.record_milestone("oc_real", "dev", summary="real e2e")
                # _read_logs reads the same all.jsonl that log_entry writes.
                from larkhelm.log import _read_logs
                with patch("larkhelm.log._cfg.LOG_DIR", log_dir, create=True):
                    rows = _read_logs("oc_real")
            self.assertGreaterEqual(len(rows), 1)
            milestone_rows = [r for r in rows if r.get("role") == "milestone"]
            self.assertEqual(len(milestone_rows), 1)
            self.assertIn("[Milestone] dev", milestone_rows[0]["content"])
            self.assertIn("real e2e", milestone_rows[0]["content"])


if __name__ == "__main__":
    unittest.main()
