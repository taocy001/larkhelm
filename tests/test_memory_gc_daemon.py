"""AC-08 — P3 REQ-08 MemoryGC daemon enhancements.

Targets the new interval_hours plumbing + stop() event + the checkpoint
GC composition. The audit-rotation work is already covered by
``tests/test_memory_retriever_audit_rotate.py``; this file pins the
*integration* added in P3.
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from larkhelm.memory_gc import MemoryGCRunner, attach_checkpoint_gc


class TestMemoryGCRunnerP3(unittest.TestCase):

    def test_interval_hours_overrides_interval_sec(self) -> None:
        runner = MemoryGCRunner(interval_hours=2.0)
        # 2 hours == 7200s
        self.assertEqual(runner.interval_sec, 7200)
        self.assertAlmostEqual(runner.interval_hours, 2.0)

    def test_stop_event_set_by_stop(self) -> None:
        runner = MemoryGCRunner()
        self.assertFalse(runner._stop_event.is_set())
        runner.stop()
        self.assertTrue(runner._stop_event.is_set())

    def test_attach_checkpoint_gc_invokes_scan_once(self) -> None:
        runner = MemoryGCRunner()
        fake_gc = MagicMock()
        fake_gc.scan_once.return_value = 3
        runner.attach_checkpoint_gc(fake_gc)
        runner._tick()
        fake_gc.scan_once.assert_called_once()

    def test_tick_handles_ckpt_gc_failure(self) -> None:
        runner = MemoryGCRunner()
        fake_gc = MagicMock()
        fake_gc.scan_once.side_effect = RuntimeError("nope")
        runner.attach_checkpoint_gc(fake_gc)
        # _tick must not propagate the exception.
        runner._tick()

    def test_run_once_no_home_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch MEMORY_HOME_DIR via lazy import path.
            import larkhelm.memory as memory_mod
            saved = memory_mod.MEMORY_HOME_DIR
            try:
                memory_mod.MEMORY_HOME_DIR = Path(tmpdir) / "missing"
                runner = MemoryGCRunner()
                scanned, deleted = runner.run_once()
                self.assertEqual((scanned, deleted), (0, 0))
            finally:
                memory_mod.MEMORY_HOME_DIR = saved


if __name__ == "__main__":
    unittest.main()
