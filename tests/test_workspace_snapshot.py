"""Tests for generate_workspace_snapshot (AC-05)."""
import json
import tempfile
import unittest
from pathlib import Path


class TestGenerateWorkspaceSnapshot(unittest.TestCase):
    def _write(self, ws: Path, filename: str, data) -> None:
        (ws / filename).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    def test_full_snapshot(self):
        from larkhelm.workspace_finalize import generate_workspace_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            meta = {
                "batch_id": "batch_001",
                "task_hash": "abc123",
                "completed": True,
                "finalized_at": 1748700000.0,
                "agent_results": [{"agent": "pm", "status": "done"}],
            }
            self._write(ws, "workspace_meta.json", meta)
            fc = {"files": [{"path": "larkhelm/foo.py", "action": "modify"}]}
            self._write(ws, "file_changes.json", fc)
            (ws / "prd.md").write_text("# My Plan Title\n\nContent", encoding="utf-8")

            snap = generate_workspace_snapshot(ws)

        self.assertEqual(snap["batch_id"], "batch_001")
        self.assertEqual(snap["task_hash"], "abc123")
        self.assertTrue(snap["completed"])
        self.assertEqual(snap["plan_title"], "My Plan Title")
        self.assertEqual(len(snap["agent_results"]), 1)
        self.assertEqual(len(snap["file_changes"]), 1)
        self.assertAlmostEqual(snap["created_at"], 1748700000.0, places=0)
        self.assertGreater(snap["snapshot_at"], 0)
        self.assertNotIn("error", snap)

    def test_missing_workspace_meta(self):
        from larkhelm.workspace_finalize import generate_workspace_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            snap = generate_workspace_snapshot(ws)

        self.assertIn("error", snap)
        self.assertIn("workspace_meta", snap["error"])

    def test_no_file_changes_json(self):
        from larkhelm.workspace_finalize import generate_workspace_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            meta = {"batch_id": "b2", "task_hash": "xyz", "completed": False}
            self._write(ws, "workspace_meta.json", meta)

            snap = generate_workspace_snapshot(ws)

        self.assertEqual(snap["batch_id"], "b2")
        self.assertEqual(snap["file_changes"], [])
        self.assertNotIn("error", snap)

    def test_no_prd_md(self):
        from larkhelm.workspace_finalize import generate_workspace_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            self._write(ws, "workspace_meta.json", {"batch_id": "b3"})
            snap = generate_workspace_snapshot(ws)

        self.assertEqual(snap["plan_title"], "")
        self.assertNotIn("error", snap)

    def test_snapshot_at_is_current_time(self):
        import time
        from larkhelm.workspace_finalize import generate_workspace_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            self._write(ws, "workspace_meta.json", {"batch_id": "b4"})
            before = time.time()
            snap = generate_workspace_snapshot(ws)
            after = time.time()

        self.assertGreaterEqual(snap["snapshot_at"], before)
        self.assertLessEqual(snap["snapshot_at"], after)


if __name__ == "__main__":
    unittest.main()
