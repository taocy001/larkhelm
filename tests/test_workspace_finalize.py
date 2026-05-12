"""Tests for ``workspace_finalize.finalize_workspace`` and its helpers.

Background
----------
``/dev`` and ``/plan`` both end with the same workspace post-processing
needs: flip ``workspace_meta.json`` to ``completed=true`` when the run's
``review.md`` ends with ``APPROVED``, and surface a Feishu summary card
with a copy-paste-able ``git add`` / ``git commit`` hint covering all
files the run touched.

The hook used to live in ``cmd_plan.py`` (P1 follow-up of commit 83d9312);
it was extracted into ``larkhelm/workspace_finalize.py`` so ``/dev`` could
share the exact same implementation. ``/dev`` previously did only the
meta-flip half (``crew/_commands.py:903-905``) and was missing the
git-add hint card.

This module guards both behaviours plus the ``kind="dev"`` / ``kind="plan"``
title prefix differentiation on the summary card.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Minimal config bootstrap so larkhelm.log / chat_state work ─────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_planfin_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm import workspace_finalize as wf_mod


# ════════════════════════════════════════════════════════════════════════
#  _format_workspace_summary — pure rendering, no I/O
# ════════════════════════════════════════════════════════════════════════

class FormatWorkspaceSummaryTests(unittest.TestCase):

    def test_approved_marks_meta_flipped(self):
        files = {"from_file_changes": ["a.py"],
                 "tracked_modified": [], "untracked": []}
        body, color = wf_mod._format_workspace_summary(files, True, "MyPlan")
        self.assertIn("APPROVED", body)
        self.assertIn("completed=true", body)
        self.assertEqual(color, "green")

    def test_not_approved_keeps_meta_false(self):
        files = {"from_file_changes": [], "tracked_modified": [], "untracked": []}
        body, color = wf_mod._format_workspace_summary(files, False, "MyPlan")
        self.assertIn("completed=false", body)
        self.assertEqual(color, "blue")

    def test_empty_lists_yield_no_section_headers(self):
        """No file-list sections (and no git-add hint) when nothing changed."""
        files = {"from_file_changes": [], "tracked_modified": [], "untracked": []}
        body, _ = wf_mod._format_workspace_summary(files, True, "MyPlan")
        self.assertNotIn("file_changes.json 声明", body)
        self.assertNotIn("工作树 modified", body)
        self.assertNotIn("工作树 untracked", body)
        self.assertNotIn("git add", body)

    def test_list_truncation_caps_at_12_with_overflow_note(self):
        many = [f"path/file{i}.py" for i in range(15)]
        files = {"from_file_changes": many, "tracked_modified": [], "untracked": []}
        body, _ = wf_mod._format_workspace_summary(files, True, "P")
        # First 12 shown
        self.assertIn("`path/file0.py`", body)
        self.assertIn("`path/file11.py`", body)
        self.assertNotIn("`path/file12.py`", body)
        self.assertIn("余 3 个略", body)

    def test_git_add_hint_quotes_spaces(self):
        files = {"from_file_changes": ["a b/c.py"], "tracked_modified": [],
                 "untracked": []}
        body, _ = wf_mod._format_workspace_summary(files, True, "P")
        self.assertIn("git add 'a b/c.py'", body)
        # title quoted via shlex.quote — simple alnum string returns as-is
        self.assertIn("git commit -m P", body)

    def test_git_commit_hint_quotes_title_with_special_chars(self):
        """Audit P2: plan titles can contain ``"`` / ``$(...)`` / backticks
        that would shell-inject or syntax-error the bare ``-m "..."`` form.
        Verify ``shlex.quote`` is applied to the title."""
        files = {"from_file_changes": ["a.py"], "tracked_modified": [],
                 "untracked": []}
        # Title with double-quote + command substitution
        body, _ = wf_mod._format_workspace_summary(
            files, True, 'fix "bug" $(rm -rf /)')
        # shlex.quote wraps in single quotes and escapes inner single quotes
        # — verify NO unquoted ``$(`` reaches the rendered command line
        commit_line = next(
            (ln for ln in body.splitlines() if ln.startswith("git commit")),
            "")
        self.assertNotIn('git commit -m "fix', commit_line,
            "raw double-quoted form must NOT be used — shell would expand $(...)")
        self.assertIn("'fix", commit_line,
            "title must be wrapped in single quotes via shlex.quote")

    def test_git_add_dedups_across_three_sources(self):
        """A path in both file_changes.json and untracked appears once."""
        files = {
            "from_file_changes": ["new.py", "modified.py"],
            "tracked_modified":  ["modified.py"],   # dup of intent
            "untracked":         ["new.py"],         # dup of intent
        }
        body, _ = wf_mod._format_workspace_summary(files, True, "P")
        # ``git add`` line should contain each path once
        for p in ("new.py", "modified.py"):
            self.assertEqual(body.count(f" {p}"), 1,
                f"{p!r} should be de-duped across sources in git-add hint")

    def test_git_add_truncates_at_20_targets(self):
        many = [f"f{i}.py" for i in range(25)]
        files = {"from_file_changes": many, "tracked_modified": [], "untracked": []}
        body, _ = wf_mod._format_workspace_summary(files, True, "P")
        self.assertIn("# +5 more — adjust as needed", body)


# ════════════════════════════════════════════════════════════════════════
#  _collect_plan_artifacts — file_changes.json + git status merge
# ════════════════════════════════════════════════════════════════════════

class CollectPlanArtifactsTests(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="larkhelm_collect_"))
        self.ws = self.workdir / ".crew_workspace"
        self.ws.mkdir()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _write_file_changes(self, files: list[str]) -> None:
        (self.ws / "file_changes.json").write_text(json.dumps({
            "files": [{"path": p, "action": "create", "desc": ""} for p in files]
        }), encoding="utf-8")

    def _patch_git_status(self, status_lines: list[str]):
        """Return a context manager that fakes ``git status --porcelain``."""
        fake_proc = MagicMock(returncode=0, stdout="\n".join(status_lines))
        return patch("subprocess.run", return_value=fake_proc)

    def test_reads_file_changes_json(self):
        self._write_file_changes(["new.py", "scripts/x.sh"])
        with self._patch_git_status([]):
            r = wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        self.assertEqual(r["from_file_changes"], ["new.py", "scripts/x.sh"])

    def test_missing_file_changes_json_returns_empty_intent(self):
        with self._patch_git_status([]):
            r = wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        self.assertEqual(r["from_file_changes"], [])

    def test_corrupted_file_changes_json_does_not_raise(self):
        (self.ws / "file_changes.json").write_text("{not json")
        with self._patch_git_status([]):
            r = wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        # Falls through with empty intent — never raises
        self.assertEqual(r["from_file_changes"], [])

    def test_git_status_split_into_modified_and_untracked(self):
        self._write_file_changes([])
        lines = [
            " M larkhelm/log.py",
            "?? scripts/new.py",
            "?? tests/test_x.py",
            "M  README.md",
        ]
        with self._patch_git_status(lines):
            r = wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        self.assertIn("larkhelm/log.py",  r["tracked_modified"])
        self.assertIn("README.md",        r["tracked_modified"])
        self.assertIn("scripts/new.py",   r["untracked"])
        self.assertIn("tests/test_x.py",  r["untracked"])

    def test_git_status_failure_returns_partial_result(self):
        """If ``git status`` errors (non-zero rc, timeout, etc), intent is
        still returned but tracked / untracked stay empty — never raises."""
        self._write_file_changes(["intent.py"])
        with patch("subprocess.run", side_effect=OSError("no git")):
            r = wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        self.assertEqual(r["from_file_changes"], ["intent.py"])
        self.assertEqual(r["tracked_modified"], [])
        self.assertEqual(r["untracked"], [])

    def test_status_lines_shorter_than_4_chars_skipped(self):
        self._write_file_changes([])
        with self._patch_git_status(["", "M", " M"]):
            r = wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        self.assertEqual(r["tracked_modified"], [])
        self.assertEqual(r["untracked"], [])

    def test_rename_status_takes_destination_path(self):
        """Audit finding #5: ``R  old.py -> new.py`` must yield ``new.py``,
        not the literal arrow expression that would break ``git add``."""
        self._write_file_changes([])
        with self._patch_git_status([
            "R  old.py -> new.py",
            "C  src.py -> copy.py",
        ]):
            r = wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        self.assertIn("new.py", r["tracked_modified"])
        self.assertIn("copy.py", r["tracked_modified"])
        # Old paths must NOT leak through — those files no longer exist on disk
        for bad in ("old.py", "src.py", "old.py -> new.py", "src.py -> copy.py"):
            self.assertNotIn(bad, r["tracked_modified"],
                f"renamed source / arrow form must not be added: {bad!r}")

    def test_quotepath_false_flag_passed_to_git(self):
        """Audit finding #5 (companion): ``core.quotePath=false`` must be in
        the git command so non-ASCII filenames come back as literal UTF-8,
        not octal-escaped + double-quoted (e.g. ``"\\344\\270\\255\\346\\226\\207.py"``)."""
        self._write_file_changes([])
        captured: dict = {}
        def _capture(args, **kw):
            captured["args"] = args
            return MagicMock(returncode=0, stdout="")
        with patch("subprocess.run", side_effect=_capture):
            wf_mod._collect_plan_artifacts(self.ws, str(self.workdir))
        self.assertIn("-c", captured["args"])
        self.assertIn("core.quotePath=false", captured["args"],
            "git invocation must disable octal escaping for non-ASCII paths")


# ════════════════════════════════════════════════════════════════════════
#  finalize_workspace — end-to-end glue
# ════════════════════════════════════════════════════════════════════════

class FinalizePlanWorkspaceTests(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="larkhelm_fin_"))
        self.ws = self.workdir / ".crew_workspace"
        self.ws.mkdir()
        (self.ws / "workspace_meta.json").write_text(json.dumps({
            "task_hash": "hash123abc", "completed": False,
        }))
        (self.ws / "file_changes.json").write_text(json.dumps({
            "files": [{"path": "a.py", "action": "modify", "desc": ""}]
        }))

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _write_review(self, body: str) -> None:
        (self.ws / "review.md").write_text(body, encoding="utf-8")

    def _patch_chat_state_and_lark(self):
        """Common patch stack: ``_get_cwd`` → our workdir, ``send_card`` → mock."""
        return (
            patch("larkhelm.chat_state._get_cwd", return_value=str(self.workdir)),
            patch("larkhelm.lark_client.send_card"),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="")),
        )

    def test_approved_review_flips_meta_to_true(self):
        self._write_review("...review body...\n\nAPPROVED\n")
        cwd_p, card_p, run_p = self._patch_chat_state_and_lark()
        with cwd_p, card_p, run_p:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertTrue(meta["completed"],
            "APPROVED review must flip workspace_meta.completed → true")
        self.assertEqual(meta["task_hash"], "hash123abc",
            "task_hash must be preserved across flip")

    def test_rejected_review_keeps_meta_false(self):
        self._write_review("...body...\n\nREJECTED\n")
        cwd_p, card_p, run_p = self._patch_chat_state_and_lark()
        with cwd_p, card_p, run_p:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertFalse(meta["completed"])

    def test_missing_review_md_keeps_meta_false(self):
        # No review.md created
        cwd_p, card_p, run_p = self._patch_chat_state_and_lark()
        with cwd_p, card_p, run_p:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertFalse(meta["completed"])

    def test_no_cwd_short_circuits(self):
        """No registered cwd for the chat → bail without touching meta."""
        self._write_review("APPROVED")
        with patch("larkhelm.chat_state._get_cwd", return_value=""), \
             patch("larkhelm.lark_client.send_card") as card:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertFalse(meta["completed"])
        card.assert_not_called()

    def test_missing_workspace_dir_short_circuits(self):
        empty = Path(tempfile.mkdtemp(prefix="larkhelm_empty_"))
        try:
            with patch("larkhelm.chat_state._get_cwd", return_value=str(empty)), \
                 patch("larkhelm.lark_client.send_card") as card:
                wf_mod.finalize_workspace("chat1", "MyPlan")
            card.assert_not_called()
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_approved_emits_summary_card(self):
        self._write_review("...\nAPPROVED")
        cwd_p, _, run_p = self._patch_chat_state_and_lark()
        with cwd_p, run_p, \
             patch("larkhelm.lark_client.send_card") as card:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        # Card was sent — verify the title + that body mentions APPROVED + filename.
        self.assertEqual(card.call_count, 1)
        chat_id, title, body = card.call_args.args[:3]
        self.assertEqual(chat_id, "chat1")
        self.assertIn("收尾", title)
        self.assertIn("APPROVED", body)
        self.assertIn("a.py", body)

    def test_card_send_failure_does_not_raise(self):
        """A flaky Feishu API must not bubble up into the plan finally block."""
        self._write_review("APPROVED")
        cwd_p, _, run_p = self._patch_chat_state_and_lark()
        with cwd_p, run_p, \
             patch("larkhelm.lark_client.send_card",
                   side_effect=RuntimeError("network")):
            # Must NOT raise
            wf_mod.finalize_workspace("chat1", "MyPlan")
        # Meta flip should still have succeeded — that step ran before the card.
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertTrue(meta["completed"])

    def test_approved_with_trailing_whitespace_still_detected(self):
        """``APPROVED`` followed by whitespace lines must be recognised."""
        self._write_review("body\n\nAPPROVED\n   \n\n")
        cwd_p, card_p, run_p = self._patch_chat_state_and_lark()
        with cwd_p, card_p, run_p:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertTrue(meta["completed"])

    def test_empty_artefacts_skips_summary_card(self):
        """Audit finding #13: when no buckets have entries, don't send a
        useless ``📦 收尾 · 改动文件`` card with empty body. (The previous
        ``if not files: return`` was dead — a dict with all empty lists is
        still truthy.)"""
        self._write_review("APPROVED")
        # Remove the file_changes.json fixture so all 3 buckets are empty
        (self.ws / "file_changes.json").unlink()
        cwd_p, _, run_p = self._patch_chat_state_and_lark()
        with cwd_p, run_p, \
             patch("larkhelm.lark_client.send_card") as card:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        card.assert_not_called()
        # Meta flip should still have happened — that's independent of the card.
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertTrue(meta["completed"])

    def test_approved_in_middle_not_at_end_is_rejected(self):
        """The contract (same as /dev) requires last meaningful line == APPROVED."""
        self._write_review("APPROVED was the previous run\n\nREJECTED")
        cwd_p, card_p, run_p = self._patch_chat_state_and_lark()
        with cwd_p, card_p, run_p:
            wf_mod.finalize_workspace("chat1", "MyPlan")
        meta = json.loads((self.ws / "workspace_meta.json").read_text())
        self.assertFalse(meta["completed"])


# ════════════════════════════════════════════════════════════════════════
#  kind="dev" — title prefix differentiation
#
#  /dev calls finalize_workspace(..., kind="dev") and expects the summary
#  card title to read "📦 Dev 收尾 · 改动文件" (vs the default "📦 Plan 收尾").
#  This lets the user tell at a glance which flow produced the card when
#  both /plan and /dev are used in the same chat.
# ════════════════════════════════════════════════════════════════════════

class FinalizeWorkspaceKindTests(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="larkhelm_finkind_"))
        self.ws = self.workdir / ".crew_workspace"
        self.ws.mkdir()
        (self.ws / "workspace_meta.json").write_text(json.dumps({
            "task_hash": "hashabcdef", "completed": False,
        }))
        (self.ws / "file_changes.json").write_text(json.dumps({
            "files": [{"path": "src/feature.py", "action": "create", "desc": ""}]
        }))
        (self.ws / "review.md").write_text("...\nAPPROVED", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _patches(self):
        return (
            patch("larkhelm.chat_state._get_cwd", return_value=str(self.workdir)),
            patch("subprocess.run",
                  return_value=MagicMock(returncode=0, stdout="")),
        )

    def test_dev_kind_title_says_dev(self):
        """``kind='dev'`` → card title contains the literal word ``Dev``,
        NOT the default ``Plan`` prefix."""
        cwd_p, run_p = self._patches()
        with cwd_p, run_p, \
             patch("larkhelm.lark_client.send_card") as card:
            wf_mod.finalize_workspace("chat1", "implement OAuth2 flow", kind="dev")
        self.assertEqual(card.call_count, 1)
        _, title, _ = card.call_args.args[:3]
        self.assertIn("Dev", title,
            f"kind='dev' should yield a card title containing 'Dev', got {title!r}")
        self.assertNotIn("Plan", title,
            f"kind='dev' should NOT contain 'Plan' in title, got {title!r}")
        self.assertIn("收尾", title)
        self.assertIn("改动文件", title)

    def test_plan_kind_default_title_says_plan(self):
        """Default ``kind='plan'`` → card title contains ``Plan`` (regression
        guard so the extracted module preserves the original /plan behaviour)."""
        cwd_p, run_p = self._patches()
        with cwd_p, run_p, \
             patch("larkhelm.lark_client.send_card") as card:
            wf_mod.finalize_workspace("chat1", "MyPlan")  # default kind
        self.assertEqual(card.call_count, 1)
        _, title, _ = card.call_args.args[:3]
        self.assertIn("Plan", title,
            f"default kind should yield 'Plan' in title, got {title!r}")
        self.assertNotIn("Dev", title)


# ════════════════════════════════════════════════════════════════════════
#  /plan → /dev step delegation: suppress_finalize=True must be propagated
#
#  When /plan runs a [dev] step it invokes ``_run_dev_crew_inner`` directly.
#  Without ``suppress_finalize=True`` the user would see ONE summary card
#  per [dev] step PLUS a final one from /plan — cosmetic noise. The plan
#  pipeline owns the final card; the per-step /dev hook must stay quiet.
# ════════════════════════════════════════════════════════════════════════

class PlanDevStepSuppressesFinalizeTests(unittest.TestCase):
    """Regression guard for the double-card UX bug spotted in commit-time
    review. ``/plan`` calls ``_run_dev_crew_inner`` once per [dev] step
    and must pass ``suppress_finalize=True`` so the inner /dev flow skips
    its own finalize card; /plan emits the consolidated one at the end."""

    def test_run_dev_step_passes_suppress_finalize_true(self):
        from larkhelm import cmd_plan as plan_mod

        # Build a minimal MultiPlanState shell — only the fields _run_dev_step
        # touches (chat_id, cancel_ev, plus the step it runs).
        state = MagicMock()
        state.chat_id = "chat1"
        state.cancel_ev = MagicMock()
        state.cancel_ev.is_set = MagicMock(return_value=False)
        step = MagicMock()
        step.desc = "implement login"
        step.idx = 0

        with patch("larkhelm.crew._commands._run_dev_crew_inner") as inner:
            plan_mod._run_dev_step(state, step, crew_id="crew-xyz")

        inner.assert_called_once()
        kwargs = inner.call_args.kwargs
        self.assertTrue(kwargs.get("suppress_finalize"),
            "/plan [dev] step MUST pass suppress_finalize=True so the inner "
            "/dev flow doesn't double-emit a workspace summary card per step "
            "(/plan emits the consolidated card after all steps complete). "
            f"Got kwargs={kwargs!r}")
        # Sanity: the existing suppress_done_signal contract is still in place
        self.assertTrue(kwargs.get("suppress_done_signal"),
            "suppress_done_signal=True is the existing contract — must remain")


if __name__ == "__main__":
    unittest.main()
