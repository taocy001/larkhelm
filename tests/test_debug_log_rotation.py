"""Coverage for the DEBUG_LOG rotation introduced in the unified-logging
Phase 3 cleanup. Mirrors the established ``rotate_jsonl_if_needed`` pattern.

The runtime path is: every ``_DEBUG_ROTATE_CHECK_EVERY`` writes, ``_debug_log``
calls ``rotate_debug_log_if_needed``; if DEBUG_LOG > ``_MAX_DEBUG_LOG_BYTES``
it gets renamed to ``<name>.1`` (single backup, oldest dropped).
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import log as larkhelm_log


class TestRotateDebugLogIfNeeded(unittest.TestCase):
    """Direct test of the rotation function (independent of the size probe)."""

    def test_rotates_when_oversize(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            # Pre-seed file beyond the threshold.
            log_path.write_bytes(b"x" * (5 * 1024 * 1024))  # 5 MB
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_MAX_DEBUG_LOG_BYTES", 1024 * 1024):  # 1 MB
                larkhelm_log.rotate_debug_log_if_needed()
            backup = log_path.with_name(log_path.name + ".1")
            # The backup must exist with the original 5 MB content (proves rename
            # ran). The primary file may be re-created small by the post-rotation
            # _debug_log note — that's expected and acceptable; what matters is
            # that the live file is no longer the bloated one.
            self.assertTrue(backup.exists(),
                            "backup file should exist after rotation")
            self.assertEqual(backup.stat().st_size, 5 * 1024 * 1024)
            primary_size = log_path.stat().st_size if log_path.exists() else 0
            self.assertLess(primary_size, 1024,
                            "primary file should be small (rotation note only)")

    def test_no_rotate_when_under_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            log_path.write_bytes(b"x" * 1024)  # 1 KB
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_MAX_DEBUG_LOG_BYTES", 1024 * 1024):
                larkhelm_log.rotate_debug_log_if_needed()
            self.assertTrue(log_path.exists(),
                            "primary file must stay when below threshold")
            self.assertFalse(log_path.with_name(log_path.name + ".1").exists())

    def test_existing_backup_replaced(self):
        """A second rotation must drop the older .1 backup (single-backup policy)."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            backup = log_path.with_name(log_path.name + ".1")
            backup.write_bytes(b"OLD" * 100)  # stale backup
            log_path.write_bytes(b"x" * (5 * 1024 * 1024))
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_MAX_DEBUG_LOG_BYTES", 1024 * 1024):
                larkhelm_log.rotate_debug_log_if_needed()
            self.assertTrue(backup.exists())
            # Old backup contents must have been replaced by current size.
            self.assertEqual(backup.stat().st_size, 5 * 1024 * 1024)

    def test_missing_file_is_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"  # never created
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True):
                # Should silently no-op, never raise.
                larkhelm_log.rotate_debug_log_if_needed()
            self.assertFalse(log_path.exists())

    def test_rotation_failure_swallowed(self):
        """If the rename op fails the function must NOT propagate — stderr
        fallback only. Otherwise a permission error in /var/log would abort
        the next _debug_log caller."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            log_path.write_bytes(b"x" * (5 * 1024 * 1024))
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_MAX_DEBUG_LOG_BYTES", 1024 * 1024), \
                 patch.object(Path, "rename",
                              side_effect=PermissionError("read-only")):
                # Must not raise.
                larkhelm_log.rotate_debug_log_if_needed()
            # Original file remains because rename failed.
            self.assertTrue(log_path.exists())


class TestDebugLogTriggersRotation(unittest.TestCase):
    """End-to-end: enough _debug_log writes plus an oversized file → rotation."""

    def setUp(self):
        # Reset the module-level write counter so each test is independent.
        larkhelm_log._debug_write_count = 0

    def test_probe_threshold_triggers_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            # Pre-seed with > threshold so the probe trips on first chance.
            log_path.write_bytes(b"x" * (3 * 1024 * 1024))
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_MAX_DEBUG_LOG_BYTES", 1024 * 1024), \
                 patch.object(larkhelm_log, "_DEBUG_ROTATE_CHECK_EVERY", 5):
                # 5 writes ⇒ count reaches the probe boundary
                for i in range(5):
                    larkhelm_log._debug_log(f"line {i}")
            backup = log_path.with_name(log_path.name + ".1")
            self.assertTrue(backup.exists(), "rotation should have fired")

    def test_below_probe_threshold_no_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            log_path.write_bytes(b"x" * (3 * 1024 * 1024))
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_MAX_DEBUG_LOG_BYTES", 1024 * 1024), \
                 patch.object(larkhelm_log, "_DEBUG_ROTATE_CHECK_EVERY", 100):
                for i in range(10):
                    larkhelm_log._debug_log(f"line {i}")
            # Probe boundary not reached; file must still exist (now larger).
            self.assertTrue(log_path.exists())
            self.assertFalse(log_path.with_name(log_path.name + ".1").exists())


class TestSafeAndLazyHelpers(unittest.TestCase):
    """Phase 1 helpers: safe_log / lazy_debug_log must never raise even when
    the underlying log path is broken."""

    def test_safe_log_swallows_debug_log_failure(self):
        with patch.object(larkhelm_log, "_debug_log",
                          side_effect=RuntimeError("disk full")):
            # Must not raise.
            larkhelm_log.safe_log("[Test] should be silent")

    def test_lazy_debug_log_swallows_debug_log_failure(self):
        with patch.object(larkhelm_log, "_debug_log",
                          side_effect=RuntimeError("disk full")):
            larkhelm_log.lazy_debug_log("[Test] should be silent")

    def test_safe_log_calls_debug_log_on_success(self):
        with patch.object(larkhelm_log, "_debug_log") as dbg:
            larkhelm_log.safe_log("[Test] hello")
        dbg.assert_called_once_with("[Test] hello")

    def test_lazy_debug_log_calls_debug_log_on_success(self):
        with patch.object(larkhelm_log, "_debug_log") as dbg:
            larkhelm_log.lazy_debug_log("[Test] hello")
        dbg.assert_called_once_with("[Test] hello")

    def test_agent_hub_safe_log_aliases_centralized(self):
        """All four agent_hub modules must now reference the same callable
        as ``larkhelm.log.safe_log`` (no local copies left)."""
        from larkhelm.agent_hub import (
            agent_dispatcher, agent_audit, intent_feedback, plugin_loader,
        )
        canonical = larkhelm_log.safe_log
        self.assertIs(agent_dispatcher._safe_log, canonical)
        self.assertIs(agent_audit._safe_log, canonical)
        self.assertIs(intent_feedback._safe_log, canonical)
        self.assertIs(plugin_loader._safe_log, canonical)


class TestRotationLockingDoesNotDeadlock(unittest.TestCase):
    """Sanity: rotation must not deadlock with concurrent _debug_log calls.
    The original concern is that ``rotate_debug_log_if_needed`` acquires
    ``_rotation_lock`` while ``_debug_log`` holds ``_log_lock``; calling
    rotation from within the lock would deadlock when it later re-enters
    ``_debug_log`` to record the rotation event. The fix is to release
    ``_log_lock`` before invoking rotation."""

    def test_concurrent_writes_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            log_path.write_bytes(b"x" * (3 * 1024 * 1024))
            larkhelm_log._debug_write_count = 0
            errors: list[BaseException] = []

            def _writer():
                try:
                    for i in range(50):
                        larkhelm_log._debug_log(f"thread line {i}")
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_MAX_DEBUG_LOG_BYTES", 1024 * 1024), \
                 patch.object(larkhelm_log, "_DEBUG_ROTATE_CHECK_EVERY", 5):
                threads = [threading.Thread(target=_writer) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10.0)
                    self.assertFalse(t.is_alive(),
                                     "writer thread deadlocked during rotation")

            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
