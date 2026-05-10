"""Coverage for the user-explicit ``/memory gc`` cleanup path
(``memory.gc_project_memory`` + the ``commands._cmd_memory`` dispatcher).

Spec source: ``.crew_workspace/memory_review_final.md`` §4 P1-2:
> ``/memory gc [days]``：用户显式触发清理 N 天未更新的 project 文件，
> 不做后台自动 GC。

Design contract verified here:

  * Dry-run by default; ``apply=True`` required to actually delete.
  * Age threshold is mtime-based with ``_GC_DEFAULT_DAYS = 30``.
  * ``threshold_days < 1`` raises ``ValueError`` (defense against
    catastrophic typos that would clear all project memory).
  * Stale conditions: age > N OR stored ``cwd`` no longer exists.
  * NEVER touches ``session_*.md`` or ``global_*.md``.
  * Per-file write lock honored so a concurrent cascade-extract
    write can't race the unlink.
"""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import memory


def _seed_project_file(tmp_dir: Path, name: str, *,
                       cwd: str = "/tmp/fake_project",
                       body: str = "## Test\nproject memory body",
                       age_days: float = 0) -> Path:
    """Create a minimal project_*.md under tmp_dir backdated by age_days."""
    p = tmp_dir / name
    fm = f'---\nupdated_at: "2026-05-09T00:00:00"\ncwd: "{cwd}"\n---\n\n'
    p.write_text(fm + body, encoding="utf-8")
    if age_days > 0:
        old = time.time() - age_days * 86400
        os.utime(p, (old, old))
    return p


class _GCBaseTestCase(unittest.TestCase):
    """Shared MEMORY_HOME_DIR isolation."""

    def setUp(self):
        self._tmp = __import__("tempfile").TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self._home_patch = patch.object(memory, "MEMORY_HOME_DIR", self.tmp_dir)
        self._home_patch.start()
        # Also clear the file-write-lock pool so leftover locks from other
        # tests don't leak in. Lock pool is keyed by full path string, but
        # tmp dirs have unique paths so this is mostly defensive.
        memory._file_write_locks.clear()

    def tearDown(self):
        self._home_patch.stop()
        self._tmp.cleanup()


# ── 1. gc_project_memory helper ─────────────────────────────────────────


class TestGcProjectMemoryDryRun(_GCBaseTestCase):

    def test_default_threshold_is_30_days(self):
        self.assertEqual(memory._GC_DEFAULT_DAYS, 30)

    def test_threshold_days_zero_raises(self):
        with self.assertRaises(ValueError):
            memory.gc_project_memory(threshold_days=0)

    def test_threshold_days_negative_raises(self):
        with self.assertRaises(ValueError):
            memory.gc_project_memory(threshold_days=-1)

    def test_no_files_returns_zero_scanned(self):
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        self.assertEqual(report["scanned"], 0)
        self.assertEqual(report["candidates"], [])
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["apply"])

    def test_fresh_files_not_flagged(self):
        # Create a project file pointing at /tmp (which exists) with age 0.
        _seed_project_file(self.tmp_dir, "project_fresh.md",
                           cwd="/tmp", age_days=0)
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        self.assertEqual(report["scanned"], 1)
        self.assertEqual(report["candidates"], [],
                         "fresh file with valid cwd must NOT be flagged")

    def test_stale_age_flagged_with_cwd_alive(self):
        _seed_project_file(self.tmp_dir, "project_stale.md",
                           cwd="/tmp", age_days=45)
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        self.assertEqual(report["scanned"], 1)
        self.assertEqual(len(report["candidates"]), 1)
        c = report["candidates"][0]
        self.assertEqual(c["name"], "project_stale.md")
        self.assertEqual(c["reason"], "stale_age")
        self.assertGreaterEqual(c["age_days"], 44)
        self.assertFalse(c["deleted"], "dry-run must not delete")

    def test_dead_cwd_flagged_even_when_fresh(self):
        _seed_project_file(self.tmp_dir, "project_orphan.md",
                           cwd="/tmp/this_path_definitely_does_not_exist_xyz_42",
                           age_days=0)
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        self.assertEqual(report["scanned"], 1)
        self.assertEqual(len(report["candidates"]), 1)
        self.assertEqual(report["candidates"][0]["reason"], "cwd_gone")

    def test_both_reasons_combined(self):
        _seed_project_file(
            self.tmp_dir, "project_old_orphan.md",
            cwd="/tmp/no_such_dir_for_gc_test_xyz", age_days=99,
        )
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        self.assertEqual(report["candidates"][0]["reason"], "stale_age+cwd_gone")

    def test_threshold_boundary(self):
        """A file mtimed exactly threshold_days ago should NOT be flagged
        (strict >, not ≥). 35 days old vs 30-day threshold IS flagged."""
        _seed_project_file(self.tmp_dir, "project_29d.md",
                           cwd="/tmp", age_days=29)
        _seed_project_file(self.tmp_dir, "project_31d.md",
                           cwd="/tmp", age_days=31)
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        flagged = {c["name"] for c in report["candidates"]}
        self.assertNotIn("project_29d.md", flagged)
        self.assertIn("project_31d.md", flagged)

    def test_session_and_global_files_never_scanned(self):
        # Seed all three layer types with old mtimes.
        (self.tmp_dir / "session_oc_xxx.md").write_text("body", encoding="utf-8")
        (self.tmp_dir / "global_user1.md").write_text("body", encoding="utf-8")
        _seed_project_file(self.tmp_dir, "project_only.md",
                           cwd="/tmp", age_days=99)
        old = time.time() - 99 * 86400
        os.utime(self.tmp_dir / "session_oc_xxx.md", (old, old))
        os.utime(self.tmp_dir / "global_user1.md", (old, old))
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        # Only the project_*.md is scanned; session/global never enter the loop.
        self.assertEqual(report["scanned"], 1)
        flagged = {c["name"] for c in report["candidates"]}
        self.assertEqual(flagged, {"project_only.md"})

    def test_corrupted_frontmatter_does_not_break_scan(self):
        bad = self.tmp_dir / "project_corrupt.md"
        bad.write_text("not even YAML here\nrandom bytes", encoding="utf-8")
        old = time.time() - 99 * 86400
        os.utime(bad, (old, old))
        # Plus one good stale file to make sure the loop continues.
        _seed_project_file(self.tmp_dir, "project_good_stale.md",
                           cwd="/tmp", age_days=45)
        report = memory.gc_project_memory(threshold_days=30, apply=False)
        # The corrupt file is still flagged on age (no need for valid frontmatter).
        names = {c["name"] for c in report["candidates"]}
        self.assertIn("project_corrupt.md", names)
        self.assertIn("project_good_stale.md", names)


class TestGcProjectMemoryApply(_GCBaseTestCase):

    def test_apply_actually_deletes_stale_files(self):
        a = _seed_project_file(self.tmp_dir, "project_old1.md",
                               cwd="/tmp", age_days=45)
        b = _seed_project_file(self.tmp_dir, "project_old2.md",
                               cwd="/tmp", age_days=99)
        c = _seed_project_file(self.tmp_dir, "project_fresh.md",
                               cwd="/tmp", age_days=1)
        report = memory.gc_project_memory(threshold_days=30, apply=True)
        self.assertEqual(report["apply"], True)
        # Two old files deleted, fresh kept.
        self.assertFalse(a.exists())
        self.assertFalse(b.exists())
        self.assertTrue(c.exists())
        n_deleted = sum(1 for cand in report["candidates"] if cand["deleted"])
        self.assertEqual(n_deleted, 2)

    def test_apply_skips_when_write_lock_held(self):
        """A concurrent ``_save_md`` for the same path holds the per-file
        write lock; gc must skip (not block, not double-delete) and report
        the contention as an error entry."""
        path = _seed_project_file(self.tmp_dir, "project_locked.md",
                                  cwd="/tmp", age_days=99)
        lock = memory._get_file_write_lock(path)
        lock.acquire()  # simulate in-flight write
        try:
            report = memory.gc_project_memory(threshold_days=30, apply=True)
        finally:
            lock.release()
        # The candidate is reported but NOT deleted.
        self.assertTrue(path.exists(), "locked file must not be deleted")
        cand = next(c for c in report["candidates"] if c["name"] == "project_locked.md")
        self.assertFalse(cand["deleted"])
        self.assertTrue(any("write lock busy" in e["err"] for e in report["errors"]))

    def test_apply_with_unlink_failure_is_reported(self):
        """If the unlink itself raises (e.g. permission denied), the error
        is recorded, deleted=False, and the function does NOT propagate."""
        _seed_project_file(self.tmp_dir, "project_protected.md",
                           cwd="/tmp", age_days=99)
        # Patch Path.unlink to raise.
        with patch.object(Path, "unlink",
                          side_effect=PermissionError("read-only fs")):
            report = memory.gc_project_memory(threshold_days=30, apply=True)
        self.assertEqual(len(report["errors"]), 1)
        self.assertFalse(report["candidates"][0]["deleted"])

    def test_dry_run_never_deletes_even_when_files_match(self):
        path = _seed_project_file(self.tmp_dir, "project_should_survive.md",
                                  cwd="/tmp", age_days=99)
        memory.gc_project_memory(threshold_days=30, apply=False)
        self.assertTrue(path.exists())


# ── 2. /memory gc command dispatcher ────────────────────────────────────


class TestMemoryGcDispatcher(unittest.TestCase):
    """Argument parsing of the ``/memory gc`` subcommand. Stubs the
    underlying ``gc_project_memory`` so we test only the parsing layer."""

    def _invoke(self, args: str, gc_return: dict | None = None,
                gc_raises: Exception | None = None):
        """Call ``_cmd_memory`` with ``gc <args>`` and capture send_card_reply."""
        from larkhelm import commands
        captured: list = []

        def _fake_send_card_reply(chat_id, msg_id, title, body, color="blue", **kw):
            captured.append({"title": title, "body": body, "color": color})

        gc_default = gc_return if gc_return is not None else {
            "threshold_days": 30, "apply": False,
            "scanned": 0, "candidates": [], "errors": [],
        }

        def _fake_gc(*a, **kw):
            if gc_raises is not None:
                raise gc_raises
            return gc_default

        with patch.object(commands, "send_card_reply", _fake_send_card_reply), \
             patch.object(memory, "gc_project_memory", side_effect=_fake_gc) as gc_mock, \
             patch("larkhelm.chat_state._get_cwd", return_value="/tmp"):
            commands._cmd_memory("oc_gc_test", args)
        return captured, gc_mock

    def test_no_args_dry_run_default(self):
        # Empty-candidates branch shows "无可清理项" with the plain GC title;
        # the "预演" wording only appears when candidates exist (see
        # ``test_dry_run_renders_apply_hint`` for that path).
        captured, gc_mock = self._invoke("gc")
        gc_mock.assert_called_once()
        kwargs = gc_mock.call_args.kwargs
        self.assertEqual(kwargs["threshold_days"], 30)
        self.assertEqual(kwargs["apply"], False)
        self.assertIn("项目记忆 GC", captured[0]["title"])

    def test_threshold_only(self):
        _, gc_mock = self._invoke("gc 60")
        kwargs = gc_mock.call_args.kwargs
        self.assertEqual(kwargs["threshold_days"], 60)
        self.assertEqual(kwargs["apply"], False)

    def test_apply_only_uses_default_days(self):
        _, gc_mock = self._invoke("gc apply")
        kwargs = gc_mock.call_args.kwargs
        self.assertEqual(kwargs["threshold_days"], 30)
        self.assertEqual(kwargs["apply"], True)

    def test_threshold_then_apply(self):
        _, gc_mock = self._invoke("gc 7 apply")
        kwargs = gc_mock.call_args.kwargs
        self.assertEqual(kwargs["threshold_days"], 7)
        self.assertEqual(kwargs["apply"], True)

    def test_apply_then_threshold_also_works(self):
        """Order-insensitive parsing — users shouldn't need to memorize."""
        _, gc_mock = self._invoke("gc apply 14")
        kwargs = gc_mock.call_args.kwargs
        self.assertEqual(kwargs["threshold_days"], 14)
        self.assertEqual(kwargs["apply"], True)

    def test_unknown_token_shows_usage(self):
        captured, gc_mock = self._invoke("gc bogus")
        gc_mock.assert_not_called()
        self.assertEqual(captured[0]["color"], "orange")
        self.assertIn("用法", captured[0]["title"])

    def test_zero_days_rejected_with_friendly_message(self):
        captured, gc_mock = self._invoke("gc 0")
        gc_mock.assert_not_called()
        self.assertEqual(captured[0]["color"], "orange")
        self.assertIn("阈值", captured[0]["title"])

    def test_helper_exception_surfaces_as_error_card(self):
        captured, _ = self._invoke("gc 30 apply", gc_raises=RuntimeError("disk gone"))
        self.assertEqual(captured[0]["color"], "red")
        self.assertIn("失败", captured[0]["title"])
        self.assertIn("disk gone", captured[0]["body"])

    def test_no_candidates_shows_clean_message(self):
        captured, _ = self._invoke("gc")
        self.assertIn("无可清理项", captured[0]["body"])
        self.assertEqual(captured[0]["color"], "green")

    def test_dry_run_renders_apply_hint(self):
        captured, _ = self._invoke("gc 45", gc_return={
            "threshold_days": 45, "apply": False,
            "scanned": 5,
            "candidates": [
                {"name": "project_a.md", "path": "/x/a",
                 "cwd": "/tmp/old_proj", "age_days": 60,
                 "reason": "stale_age", "deleted": False},
            ],
            "errors": [],
        })
        body = captured[0]["body"]
        self.assertIn("/memory gc 45 apply", body)
        self.assertIn("project_a.md", body)
        self.assertIn("60d", body)

    def test_apply_renders_deleted_count(self):
        captured, _ = self._invoke("gc 30 apply", gc_return={
            "threshold_days": 30, "apply": True,
            "scanned": 3,
            "candidates": [
                {"name": "p1.md", "path": "/x/p1", "cwd": "/tmp/x",
                 "age_days": 99, "reason": "stale_age", "deleted": True},
                {"name": "p2.md", "path": "/x/p2", "cwd": "/tmp/y",
                 "age_days": 55, "reason": "cwd_gone", "deleted": True},
            ],
            "errors": [],
        })
        body = captured[0]["body"]
        self.assertIn("已执行", captured[0]["title"])
        self.assertIn("**2**", body)  # n_deleted


if __name__ == "__main__":
    unittest.main()
