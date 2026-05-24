"""Tests for ``larkhelm.crew._prd_selfcheck`` (B1).

The self-check gate runs after the ``architect`` agent in the /dev /
/plan pipeline. It enforces three contracts on the workspace artifacts:

  (a) every ``logic_analysis[i][2].anchors[].snippet`` hits 1-5 lines
      under ``grep -F``
  (b) every ``prd_criteria.json criteria[].how_to_verify`` has no
      unfilled placeholders (``<VAR>`` / ``{var}`` / ``$VAR``)
  (c) every ``file_changes.json files[i]`` action/path pair is
      consistent (``create`` ⇒ not exists, others ⇒ exists)
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from larkhelm.crew._prd_selfcheck import run_prd_selfcheck


def _make_workspace(cwd: Path, *,
                    tasks: dict | None = None,
                    prd_c: dict | None = None,
                    file_changes: dict | None = None) -> None:
    """Drop the three workspace JSON files into ``<cwd>/.crew_workspace/``."""
    ws = cwd / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    if tasks is not None:
        (ws / "tasks.json").write_text(json.dumps(tasks))
    if prd_c is not None:
        (ws / "prd_criteria.json").write_text(json.dumps(prd_c))
    if file_changes is not None:
        (ws / "file_changes.json").write_text(json.dumps(file_changes))


class LegacyCompatTests(unittest.TestCase):

    def test_legacy_2tuple_logic_analysis_passes(self):
        """B1 must not break old checkpoints — entries with only path+desc
        (no anchors slot) skip the anchor check entirely."""
        with TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "foo.py").write_text("def existing():\n    return 1\n")
            _make_workspace(
                cwd,
                tasks={"logic_analysis": [["foo.py", "modify the existing fn"]]},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "foo.py", "action": "modify"}]},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertTrue(passed, f"legacy schema should pass, got report:\n{report}")
            self.assertIn("✅ PASS", report)

    def test_missing_artifacts_pass(self):
        """All three artifacts missing — degrade open, don't fail."""
        with TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / ".crew_workspace").mkdir()
            passed, _ = run_prd_selfcheck(cwd)
            self.assertTrue(passed)


class AnchorChecks(unittest.TestCase):

    def test_anchor_hits_once_passes(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "foo.py").write_text(
                "def some_uniquely_named_function_so_grep_works(arg):\n"
                "    return arg + 1\n"
            )
            _make_workspace(
                cwd,
                tasks={"logic_analysis": [[
                    "foo.py", "modify the function",
                    {"anchors": [{
                        "snippet": "def some_uniquely_named_function_so_grep_works(arg):",
                        "purpose": "edit signature here",
                    }]},
                ]]},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "foo.py", "action": "modify"}]},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertTrue(passed, report)

    def test_anchor_hits_zero_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "foo.py").write_text("nothing matches this anchor\n")
            _make_workspace(
                cwd,
                tasks={"logic_analysis": [[
                    "foo.py", "modify",
                    {"anchors": [{
                        "snippet": "this_line_definitely_does_not_appear_anywhere_in_foo",
                        "purpose": "p",
                    }]},
                ]]},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "foo.py", "action": "modify"}]},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertFalse(passed)
            self.assertIn("not found", report)

    def test_anchor_hits_too_many_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            line = "    return foo_bar_baz_quux_repeated_thirty_chars_or_more\n"
            (cwd / "foo.py").write_text(line * 6)  # 6 > _ANCHOR_HIT_MAX=5
            _make_workspace(
                cwd,
                tasks={"logic_analysis": [[
                    "foo.py", "modify",
                    {"anchors": [{
                        "snippet": "    return foo_bar_baz_quux_repeated_thirty_chars_or_more",
                        "purpose": "p",
                    }]},
                ]]},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "foo.py", "action": "modify"}]},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertFalse(passed)
            self.assertIn("ambiguous", report)

    def test_short_snippet_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "foo.py").write_text("x = 1\n")
            _make_workspace(
                cwd,
                tasks={"logic_analysis": [[
                    "foo.py", "modify",
                    {"anchors": [{"snippet": "x = 1", "purpose": "p"}]},
                ]]},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "foo.py", "action": "modify"}]},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertFalse(passed)
            self.assertIn("too generic", report)

    def test_anchor_in_create_action_target_skipped(self):
        """Action=create paths legitimately don't exist yet at PRD time —
        the anchor check should silently skip them (the file_changes
        check covers the path/action contract instead)."""
        with TemporaryDirectory() as td:
            cwd = Path(td)
            _make_workspace(
                cwd,
                tasks={"logic_analysis": [[
                    "future.py", "create new module",
                    {"anchors": [{
                        "snippet": "def placeholder_function_in_the_future_file():",
                        "purpose": "p",
                    }]},
                ]]},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "future.py", "action": "create"}]},
            )
            passed, _ = run_prd_selfcheck(cwd)
            self.assertTrue(passed)


class HowToVerifyChecks(unittest.TestCase):

    def test_placeholder_var_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": [
                    {"id": "AC-01", "how_to_verify": "run pytest -k <NAME>"},
                ]},
                file_changes={"files": []},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertFalse(passed)
            self.assertIn("placeholder", report)

    def test_python_brace_placeholder_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": [
                    {"id": "AC-01", "how_to_verify": "grep {pattern} foo.py"},
                ]},
                file_changes={"files": []},
            )
            passed, _ = run_prd_selfcheck(cwd)
            self.assertFalse(passed)

    def test_empty_how_to_verify_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": [{"id": "AC-01", "how_to_verify": ""}]},
                file_changes={"files": []},
            )
            passed, _ = run_prd_selfcheck(cwd)
            self.assertFalse(passed)

    def test_filled_how_to_verify_passes(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": [
                    {"id": "AC-01", "how_to_verify": "pytest tests/test_foo.py"},
                ]},
                file_changes={"files": []},
            )
            passed, _ = run_prd_selfcheck(cwd)
            self.assertTrue(passed)


class FileChangesConsistencyChecks(unittest.TestCase):

    def test_create_on_existing_path_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "already.py").write_text("# pre-existing\n")
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "already.py", "action": "create"}]},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertFalse(passed)
            self.assertIn("already exists", report)

    def test_modify_on_missing_path_fails(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": []},
                file_changes={"files": [{"path": "ghost.py", "action": "modify"}]},
            )
            passed, report = run_prd_selfcheck(cwd)
            self.assertFalse(passed)
            self.assertIn("does not exist", report)

    def test_consistent_actions_pass(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            (cwd / "mod_me.py").write_text("# existing\n")
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": []},
                file_changes={"files": [
                    {"path": "mod_me.py", "action": "modify"},
                    {"path": "new_one.py", "action": "create"},
                ]},
            )
            passed, _ = run_prd_selfcheck(cwd)
            self.assertTrue(passed)


class ReportSideEffectTests(unittest.TestCase):

    def test_report_written_to_workspace(self):
        with TemporaryDirectory() as td:
            cwd = Path(td)
            _make_workspace(
                cwd,
                tasks={"logic_analysis": []},
                prd_c={"criteria": []},
                file_changes={"files": []},
            )
            passed, _ = run_prd_selfcheck(cwd)
            report_path = cwd / ".crew_workspace" / "prd_selfcheck.md"
            self.assertTrue(report_path.exists())
            self.assertIn("PRD Self-Check Report", report_path.read_text())

    def test_internal_failures_never_raise(self):
        """Self-check is meant to be a soft gate; whatever happens
        internally, ``run_prd_selfcheck`` must return a tuple, not raise."""
        with TemporaryDirectory() as td:
            cwd = Path(td)
            # Garbage JSON should be tolerated.
            ws = cwd / ".crew_workspace"
            ws.mkdir()
            (ws / "tasks.json").write_text("{not valid json")
            (ws / "prd_criteria.json").write_text("[also garbage")
            (ws / "file_changes.json").write_text("totally broken")
            # Must return cleanly.
            passed, report = run_prd_selfcheck(cwd)
            self.assertIsInstance(passed, bool)
            self.assertIsInstance(report, str)


if __name__ == "__main__":
    unittest.main()
