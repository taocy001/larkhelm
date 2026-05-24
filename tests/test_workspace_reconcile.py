"""Tests for ``larkhelm.crew._workspace_reconcile`` (B4).

The reconciler runs after implementer / fixer and appends any
``git status --porcelain`` files that aren't already in
``file_changes.json``. Schema-version stamping for ``tasks.json`` is
also covered.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Bootstrap config so larkhelm.log works.
_TMP = tempfile.mkdtemp(prefix="larkhelm_reconcile_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.crew import _workspace_reconcile as rec_mod


def _make_repo() -> Path:
    """Create a fresh tempdir + .crew_workspace inside it; return cwd."""
    td = Path(tempfile.mkdtemp(prefix="larkhelm_reconcile_"))
    (td / ".crew_workspace").mkdir()
    return td


def _fake_git_status_proc(porcelain_stdout: str):
    """Return a MagicMock that mimics subprocess.run for git status."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = porcelain_stdout
    m.stderr = ""
    return m


class ReconcileFileChangesTests(unittest.TestCase):

    def test_appends_drift_with_auto_added_flag(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "file_changes.json").write_text(json.dumps({
            "files": [{"path": "a.py", "action": "modify",
                       "desc": "declared upfront"}]
        }))
        with patch(
            "larkhelm.crew._workspace_reconcile.subprocess.run",
            return_value=_fake_git_status_proc(" M a.py\n?? new.py\n M extra.py\n"),
        ):
            result = rec_mod.reconcile_file_changes(str(cwd))
        self.assertFalse(result["noop"])
        self.assertCountEqual(result["added"], ["new.py", "extra.py"])
        # Re-read the file and inspect the new entries.
        data = json.loads((ws / "file_changes.json").read_text())
        paths = [e["path"] for e in data["files"]]
        # ``a.py`` retained, two new entries appended
        self.assertIn("a.py", paths)
        self.assertIn("new.py", paths)
        self.assertIn("extra.py", paths)
        for entry in data["files"]:
            if entry["path"] in ("new.py", "extra.py"):
                self.assertTrue(entry.get("auto_added"))
                self.assertEqual(entry["action"], "modify")

    def test_stamps_schema_version_on_first_drift_write(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "file_changes.json").write_text(json.dumps({
            "files": [{"path": "a.py", "action": "modify"}]
        }))
        with patch(
            "larkhelm.crew._workspace_reconcile.subprocess.run",
            return_value=_fake_git_status_proc("?? drift.py\n"),
        ):
            rec_mod.reconcile_file_changes(str(cwd))
        data = json.loads((ws / "file_changes.json").read_text())
        self.assertEqual(data.get("schema_version"), rec_mod.SCHEMA_VERSION)

    def test_idempotent_when_no_drift(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        before = {"files": [{"path": "a.py", "action": "modify"}]}
        (ws / "file_changes.json").write_text(json.dumps(before))
        with patch(
            "larkhelm.crew._workspace_reconcile.subprocess.run",
            return_value=_fake_git_status_proc(" M a.py\n"),
        ):
            result = rec_mod.reconcile_file_changes(str(cwd))
        self.assertTrue(result["noop"])
        # Nothing got rewritten — schema_version not stamped either.
        data = json.loads((ws / "file_changes.json").read_text())
        self.assertNotIn("schema_version", data)
        self.assertEqual(data["files"], before["files"])

    def test_handles_rename_destination(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "file_changes.json").write_text(json.dumps({"files": []}))
        with patch(
            "larkhelm.crew._workspace_reconcile.subprocess.run",
            return_value=_fake_git_status_proc("R  old.py -> new_destination.py\n"),
        ):
            result = rec_mod.reconcile_file_changes(str(cwd))
        self.assertIn("new_destination.py", result["added"])

    def test_missing_file_returns_noop(self):
        cwd = _make_repo()
        # No file_changes.json
        with patch(
            "larkhelm.crew._workspace_reconcile.subprocess.run",
            return_value=_fake_git_status_proc("?? new.py\n"),
        ):
            result = rec_mod.reconcile_file_changes(str(cwd))
        self.assertTrue(result["noop"])
        self.assertEqual(result["added"], [])

    def test_garbage_json_returns_noop(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "file_changes.json").write_text("{not json")
        result = rec_mod.reconcile_file_changes(str(cwd))
        self.assertTrue(result["noop"])

    def test_git_failure_returns_noop(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "file_changes.json").write_text(json.dumps({"files": []}))
        # git status with non-zero exit code
        bad_proc = MagicMock()
        bad_proc.returncode = 128
        bad_proc.stdout = ""
        bad_proc.stderr = "fatal: not a git repository"
        with patch(
            "larkhelm.crew._workspace_reconcile.subprocess.run",
            return_value=bad_proc,
        ):
            result = rec_mod.reconcile_file_changes(str(cwd))
        self.assertTrue(result["noop"])

    def test_legacy_no_auto_added_entries_preserved(self):
        """Entries written before B4 don't carry auto_added — they must
        be left untouched, not retroactively marked."""
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "file_changes.json").write_text(json.dumps({
            "files": [{"path": "a.py", "action": "modify"}]
        }))
        with patch(
            "larkhelm.crew._workspace_reconcile.subprocess.run",
            return_value=_fake_git_status_proc("?? drift.py\n"),
        ):
            rec_mod.reconcile_file_changes(str(cwd))
        data = json.loads((ws / "file_changes.json").read_text())
        legacy = next(e for e in data["files"] if e["path"] == "a.py")
        self.assertNotIn("auto_added", legacy)


class StampSchemaVersionOnTasksTests(unittest.TestCase):

    def test_stamps_v2_when_anchors_present(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "tasks.json").write_text(json.dumps({
            "logic_analysis": [["foo.py", "modify", {
                "anchors": [{"snippet": "def foo():", "purpose": "p"}]
            }]],
            "task_list": ["foo.py"],
        }))
        wrote = rec_mod.stamp_schema_version_on_tasks(str(cwd))
        self.assertTrue(wrote)
        data = json.loads((ws / "tasks.json").read_text())
        self.assertEqual(data["schema_version"], rec_mod.SCHEMA_VERSION)

    def test_no_stamp_for_legacy_2tuple_logic_analysis(self):
        """v1 schema (no anchors) must NOT be stamped — that would mark a
        legacy file as v2 and confuse downstream tooling that uses
        schema_version to decide which fields to expect."""
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "tasks.json").write_text(json.dumps({
            "logic_analysis": [["foo.py", "just a description"]],
            "task_list": ["foo.py"],
        }))
        wrote = rec_mod.stamp_schema_version_on_tasks(str(cwd))
        self.assertFalse(wrote)
        data = json.loads((ws / "tasks.json").read_text())
        self.assertNotIn("schema_version", data)

    def test_idempotent_when_already_stamped(self):
        cwd = _make_repo()
        ws = cwd / ".crew_workspace"
        (ws / "tasks.json").write_text(json.dumps({
            "schema_version": rec_mod.SCHEMA_VERSION,
            "logic_analysis": [["foo.py", "modify", {"anchors": [
                {"snippet": "x", "purpose": "p"}
            ]}]],
            "task_list": ["foo.py"],
        }))
        wrote = rec_mod.stamp_schema_version_on_tasks(str(cwd))
        self.assertFalse(wrote)


if __name__ == "__main__":
    unittest.main()
