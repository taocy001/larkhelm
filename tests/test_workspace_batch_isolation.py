"""End-to-end batch isolation tests for /dev.

Verifies that two /dev invocations with different tasks produce different
batch subdirectories, while repeated invocations for the same incomplete
task resume the same batch directory.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from larkhelm.crew._commands import (
    _find_resumable_batch,
    _make_batch_id,
    _workspace_dir,
    _write_workspace_meta,
)
from larkhelm.crew._commands import _task_hash


class BatchIsolationTests(unittest.TestCase):
    """Two /dev calls with different task hashes must land in different dirs."""

    def setUp(self) -> None:
        self._td = tempfile.mkdtemp(prefix="larkhelm_batch_iso_")

    def tearDown(self) -> None:
        shutil.rmtree(self._td, ignore_errors=True)

    def test_different_tasks_produce_different_batch_dirs(self) -> None:
        """Different requirement → different task_hash → different batch dirs."""
        hash_a = _task_hash("build a login page")
        hash_b = _task_hash("write unit tests")
        self.assertNotEqual(hash_a, hash_b)

        path_a, _ = _find_resumable_batch(self._td, hash_a, "chat_001")
        path_b, _ = _find_resumable_batch(self._td, hash_b, "chat_001")

        self.assertNotEqual(path_a, path_b, "Different tasks must produce different batch dirs")
        # Both should be under the same .crew_workspace base
        base = Path(self._td) / ".crew_workspace"
        self.assertEqual(path_a.parent, base)
        self.assertEqual(path_b.parent, base)

    def test_same_incomplete_task_resumes_same_batch(self) -> None:
        """Same requirement + not completed → second call resumes existing batch."""
        task_h = _task_hash("add dark mode")
        # Simulate first batch being created and partially done
        path_a, _ = _find_resumable_batch(self._td, task_h, "chat_001")
        path_a.mkdir(parents=True, exist_ok=True)
        _write_workspace_meta(path_a, task_hash=task_h, chat_id="chat_001", completed=False)

        path_b, meta_b = _find_resumable_batch(self._td, task_h, "chat_001")
        self.assertEqual(path_a, path_b, "Incomplete same-task batch must be resumed")
        self.assertEqual(meta_b.get("task_hash"), task_h)
        self.assertFalse(meta_b.get("completed"))

    def test_completed_task_gets_new_batch(self) -> None:
        """Completed batch must not be resumed — returned meta must be empty."""
        task_h = _task_hash("refactor auth module")
        # Create a completed batch under a fixed name so there's no ambiguity
        fixed_batch_dir = Path(self._td) / ".crew_workspace" / "batch_abcd_1000000"
        fixed_batch_dir.mkdir(parents=True, exist_ok=True)
        _write_workspace_meta(fixed_batch_dir, task_hash=task_h, chat_id="chat_001", completed=True)

        _path_b, meta_b = _find_resumable_batch(self._td, task_h, "chat_001")
        # The completed batch must not be resumed; returned meta is empty
        self.assertEqual(meta_b, {}, "Completed batch must not be resumed (meta should be empty)")

    def test_workspace_dir_name_contains_batch_id(self) -> None:
        """_workspace_dir path must include 'batch_' prefix."""
        batch_id = _make_batch_id("abcd1234", 1_700_000_000.5)
        ws = _workspace_dir(self._td, batch_id)
        self.assertIn("batch_", ws.name)
        self.assertEqual(ws.parent, Path(self._td) / ".crew_workspace")

    def test_path_sanitize_filters_dotdot(self) -> None:
        """_sanitize_file_changes must strip entries with '..' or absolute paths."""
        from larkhelm.workspace_finalize import _sanitize_file_changes
        files = [
            "safe/file.py",
            "../secret.txt",
            "/absolute/path.py",
            "also/safe.go",
            "../../etc/passwd",
        ]
        result = _sanitize_file_changes(files)
        self.assertNotIn("../secret.txt", result)
        self.assertNotIn("/absolute/path.py", result)
        self.assertNotIn("../../etc/passwd", result)
        self.assertIn("safe/file.py", result)
        self.assertIn("also/safe.go", result)
