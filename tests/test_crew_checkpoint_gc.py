"""AC-09 — P3 REQ-09 ``crew/_checkpoint_gc.CheckpointGC``."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from larkhelm.crew._checkpoint_gc import CheckpointGC


class TestCheckpointGC(unittest.TestCase):

    def _make_ckpt(self, root: Path, name: str, mtime: float) -> Path:
        workspace = root / name
        workspace.mkdir(parents=True, exist_ok=True)
        ckpt = workspace / "crew_checkpoint.json"
        ckpt.write_text('{"phase": "done"}')
        os.utime(ckpt, (mtime, mtime))
        return ckpt

    def test_removes_orphan_older_than_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old = self._make_ckpt(root, "abc", time.time() - 8 * 86400)
            fresh = self._make_ckpt(root, "def", time.time() - 1 * 86400)
            gc = CheckpointGC(root, ttl_days=7.0)
            removed = gc.scan_once()
            self.assertEqual(removed, 1)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())

    def test_returns_zero_for_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "missing"
            gc = CheckpointGC(root, ttl_days=7.0)
            self.assertEqual(gc.scan_once(), 0)

    def test_skips_active_crew_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace_name = "abc"
            ckpt = self._make_ckpt(root, workspace_name, time.time() - 30 * 86400)
            workspace_dir = (root / workspace_name).resolve()

            # Inject a fake active crew state pointing at this workspace.
            from larkhelm.crew._state import _active_crew_lock, _active_crew_states

            class _FakeState:
                pass

            fake_state = _FakeState()
            fake_state.workspace_dir = str(workspace_dir)
            with _active_crew_lock:
                _active_crew_states["chat_xxx"] = fake_state
            try:
                gc = CheckpointGC(root, ttl_days=7.0)
                removed = gc.scan_once()
            finally:
                with _active_crew_lock:
                    _active_crew_states.pop("chat_xxx", None)
            self.assertEqual(removed, 0)
            self.assertTrue(ckpt.exists())

    def test_never_raises_on_unlink_error(self) -> None:
        """A read-only file should be reported as failed-to-unlink, not raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = self._make_ckpt(root, "abc", time.time() - 8 * 86400)
            # Drop write perms on the parent dir → unlink fails.
            os.chmod(ckpt.parent, 0o555)
            try:
                gc = CheckpointGC(root, ttl_days=7.0)
                # Must not raise even if unlink fails.
                gc.scan_once()
            finally:
                os.chmod(ckpt.parent, 0o755)


if __name__ == "__main__":
    unittest.main()
