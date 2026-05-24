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


# ════════════════════════════════════════════════════════════════════════
#  U15 — "📊 本次摘要" block on the summary card
# ════════════════════════════════════════════════════════════════════════

class CollectRunMetricsTests(unittest.TestCase):
    """``_collect_run_metrics`` reads post-run signals (review, tests, diff,
    feishu doc URLs) into a metrics dict that ``_format_workspace_summary``
    renders. Each signal is fail-soft — absent signals leave keys as
    ``None`` / ``[]`` rather than raise.
    """

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="larkhelm_metrics_"))
        self.ws = self.workdir / ".crew_workspace"
        self.ws.mkdir()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _write_changes(self, body: str) -> None:
        (self.ws / "changes.md").write_text(body, encoding="utf-8")

    def _patch_git_shortstat(self, stdout: str):
        return patch("subprocess.run",
                     return_value=MagicMock(returncode=0, stdout=stdout))

    def _empty_files(self) -> dict:
        return {"from_file_changes": [], "tracked_modified": [], "untracked": []}

    def test_file_count_dedups_across_three_buckets(self):
        files = {
            "from_file_changes": ["a.py", "b.py"],
            "tracked_modified":  ["b.py", "c.py"],   # b.py dup'd
            "untracked":         ["c.py", "d.py"],   # c.py dup'd
        }
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir), files)
        self.assertEqual(m["file_count"], 4,
            "a.py / b.py / c.py / d.py should count once across buckets")

    def test_test_count_extracts_pytest_summary_line(self):
        self._write_changes("Tests: 29 passed, 10 warnings in 5.48s.\n")
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertEqual(m["tests"], "29 passed")

    def test_test_count_with_failures_renders_combined(self):
        self._write_changes("3 passed, 2 failed in 0.21s")
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertEqual(m["tests"], "3 passed, 2 failed")

    def test_test_count_takes_last_match_when_multiple_runs_logged(self):
        """``changes.md`` may include pre/post-fix pytest output. The LAST
        line wins — that's the run that produced the final accepted state."""
        self._write_changes(
            "Initial run: 5 passed\n"
            "After fix:   12 passed\n"
            "Final:       29 passed in 5.4s\n"
        )
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertEqual(m["tests"], "29 passed")

    def test_test_count_absent_returns_none(self):
        self._write_changes("plain prose with no pytest output")
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertIsNone(m["tests"])

    def test_feishu_doc_urls_extracted_and_deduped(self):
        self._write_changes(
            "上传了文档 https://feishu.cn/docx/AAA1111 和 "
            "https://my.feishu.cn/docx/BBB2222\n"
            "重复链接: https://feishu.cn/docx/AAA1111\n"
            "wiki 也算: https://open.feishu.cn/wiki/CCC3333"
        )
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        # Order preserved by first appearance, AAA seen twice but dedup'd
        self.assertEqual(m["docx_urls"], [
            "https://feishu.cn/docx/AAA1111",
            "https://my.feishu.cn/docx/BBB2222",
            "https://open.feishu.cn/wiki/CCC3333",
        ])

    def test_feishu_url_with_trailing_punctuation_not_captured(self):
        """A docx URL at end of a sentence (``…链接: https://…feishu.cn/docx/X.``)
        must not keep the trailing period. The regex character class is
        scoped to URL-safe chars only, so this comes for free — guard
        against regression."""
        self._write_changes("查看链接：https://feishu.cn/docx/TOKEN_X.")
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertEqual(m["docx_urls"], ["https://feishu.cn/docx/TOKEN_X"])

    def test_diff_stats_from_git_shortstat(self):
        with self._patch_git_shortstat(
            " 3 files changed, 120 insertions(+), 5 deletions(-)\n"
        ):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertEqual(m["diff_stats"],
            "3 files changed, 120 insertions(+), 5 deletions(-)")

    def test_diff_stats_empty_when_no_changes(self):
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertIsNone(m["diff_stats"])

    def test_missing_changes_md_leaves_signals_empty(self):
        with self._patch_git_shortstat(""):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertIsNone(m["tests"])
        self.assertEqual(m["docx_urls"], [])

    def test_git_failure_leaves_diff_stats_none(self):
        """git binary missing / not a repo → never raise; just no diff_stats."""
        with patch("subprocess.run",
                   side_effect=OSError("git not found")):
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        self.assertIsNone(m["diff_stats"])

    def test_corrupted_changes_md_does_not_raise(self):
        """Binary garbage in changes.md (a UTF-8 decode error) must not bubble."""
        (self.ws / "changes.md").write_bytes(b"\xff\xfe not utf-8")
        with self._patch_git_shortstat(""):
            # Must not raise
            m = wf_mod._collect_run_metrics(self.ws, str(self.workdir),
                                            self._empty_files())
        # Tests / urls unrecoverable, but function still returns sane shape
        self.assertIsNone(m["tests"])
        self.assertEqual(m["docx_urls"], [])


class FormatWorkspaceSummaryWithMetricsTests(unittest.TestCase):
    """``_format_workspace_summary`` U15 rendering: the metrics block is shown
    only when non-empty, individual rows omitted when their signal is None."""

    def _files(self) -> dict:
        return {"from_file_changes": ["x.py"],
                "tracked_modified": [], "untracked": []}

    def test_metrics_block_renders_all_present_fields(self):
        metrics = {
            "tests":      "29 passed",
            "diff_stats": "3 files changed, 120 insertions(+), 5 deletions(-)",
            "file_count": 4,
            "docx_urls":  ["https://feishu.cn/docx/A", "https://feishu.cn/docx/B"],
        }
        body, _ = wf_mod._format_workspace_summary(self._files(), True, "P", metrics)
        self.assertIn("📊 本次摘要", body)
        self.assertIn("测试: `29 passed`", body)
        self.assertIn("Diff: `3 files changed", body)
        self.assertIn("改动文件: 4 个", body)
        self.assertIn("飞书文档产出: 2 个", body)
        self.assertIn("https://feishu.cn/docx/A", body)

    def test_metrics_block_omitted_when_metrics_none(self):
        body, _ = wf_mod._format_workspace_summary(self._files(), True, "P", None)
        self.assertNotIn("📊 本次摘要", body)

    def test_metrics_block_omitted_when_all_signals_empty(self):
        metrics = {"tests": None, "diff_stats": None, "file_count": 0, "docx_urls": []}
        body, _ = wf_mod._format_workspace_summary(self._files(), True, "P", metrics)
        self.assertNotIn("📊 本次摘要", body)

    def test_metrics_partial_renders_only_present_rows(self):
        """When only ``tests`` is present, other rows must be suppressed
        (no ``"Diff: None"`` / ``"飞书文档产出: 0 个"`` clutter)."""
        metrics = {"tests": "12 passed", "diff_stats": None,
                   "file_count": 0, "docx_urls": []}
        body, _ = wf_mod._format_workspace_summary(self._files(), True, "P", metrics)
        self.assertIn("📊 本次摘要", body)
        self.assertIn("12 passed", body)
        self.assertNotIn("Diff:", body)
        self.assertNotIn("飞书文档产出:", body)
        self.assertNotIn("改动文件:", body)

    def test_metrics_doc_urls_truncate_at_3_with_overflow_note(self):
        urls = [f"https://feishu.cn/docx/T{i}" for i in range(5)]
        metrics = {"tests": None, "diff_stats": None, "file_count": 0,
                   "docx_urls": urls}
        body, _ = wf_mod._format_workspace_summary(self._files(), True, "P", metrics)
        self.assertIn("飞书文档产出: 5 个", body)
        self.assertIn("https://feishu.cn/docx/T0", body)
        self.assertIn("https://feishu.cn/docx/T2", body)
        self.assertNotIn("https://feishu.cn/docx/T3", body)
        self.assertIn("余 2 个略", body)

    def test_metrics_block_appears_between_review_line_and_file_lists(self):
        """Ordering invariant: review status first, then summary metrics,
        then file lists. Scannability depends on this order."""
        metrics = {"tests": "5 passed", "diff_stats": None,
                   "file_count": 1, "docx_urls": []}
        body, _ = wf_mod._format_workspace_summary(self._files(), True, "P", metrics)
        review_pos = body.find("**Review**:")
        metrics_pos = body.find("📊 本次摘要")
        files_pos = body.find("file_changes.json 声明")
        self.assertLess(review_pos, metrics_pos)
        self.assertLess(metrics_pos, files_pos)


# ════════════════════════════════════════════════════════════════════════
#  B2 — finalize auto-commit + drift detection
# ════════════════════════════════════════════════════════════════════════


class ComputeDriftTests(unittest.TestCase):
    """``_compute_drift`` is a pure function — easy to pin exhaustively."""

    def test_zero_drift_all_in_both(self):
        d = wf_mod._compute_drift(["a.py", "b.py"], ["a.py", "b.py"])
        self.assertEqual(d["in_both"], ["a.py", "b.py"])
        self.assertEqual(d["drift"], [])
        self.assertEqual(d["missing"], [])
        self.assertEqual(d["all_to_add"], ["a.py", "b.py"])

    def test_drift_unique_to_actual(self):
        d = wf_mod._compute_drift(["a.py"], ["a.py", "x.py", "y.py"])
        self.assertEqual(d["in_both"], ["a.py"])
        self.assertEqual(d["drift"], ["x.py", "y.py"])
        self.assertEqual(d["missing"], [])
        # all_to_add preserves declared-first, then drift order
        self.assertEqual(d["all_to_add"], ["a.py", "x.py", "y.py"])

    def test_missing_declared_but_unmodified(self):
        d = wf_mod._compute_drift(["a.py", "b.py"], ["a.py"])
        self.assertEqual(d["in_both"], ["a.py"])
        self.assertEqual(d["drift"], [])
        self.assertEqual(d["missing"], ["b.py"])
        self.assertEqual(d["all_to_add"], ["a.py"])

    def test_dedup_within_all_to_add(self):
        # Pathological: declared and actual both list "a.py" twice.
        d = wf_mod._compute_drift(["a.py", "a.py"], ["a.py", "a.py", "b.py"])
        self.assertEqual(d["all_to_add"], ["a.py", "b.py"])

    def test_empty_inputs(self):
        d = wf_mod._compute_drift([], [])
        self.assertEqual(d["in_both"], [])
        self.assertEqual(d["drift"], [])
        self.assertEqual(d["all_to_add"], [])


class MaybeAutoCommitFinaleTests(unittest.TestCase):

    def _files(self, declared=(), modified=(), untracked=()):
        return {
            "from_file_changes": list(declared),
            "tracked_modified":  list(modified),
            "untracked":         list(untracked),
        }

    def test_disabled_flag_skips_commit(self):
        with patch.dict(_cfg.config, {"dev_auto_commit": False}, clear=False):
            info = wf_mod._maybe_auto_commit_finale(
                "/tmp/x", self._files(declared=["a.py"], modified=["a.py"]),
                "title",
            )
        self.assertEqual(info["commit_sha"], "")
        self.assertEqual(info["skipped_reason"], "dev_auto_commit=false")

    def test_no_dirty_changes_skips_commit(self):
        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False):
            info = wf_mod._maybe_auto_commit_finale(
                "/tmp/x", self._files(declared=["a.py"]),
                "title",
            )
        self.assertEqual(info["commit_sha"], "")
        self.assertIn("no dirty", info["skipped_reason"])

    def test_drift_above_threshold_skips_commit(self):
        # 3 ≥ _DRIFT_THRESHOLD (3) — must skip.
        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False):
            info = wf_mod._maybe_auto_commit_finale(
                "/tmp/x",
                self._files(declared=["a.py"],
                            modified=["a.py", "x.py"],
                            untracked=["y.py", "z.py"]),
                "title",
            )
        self.assertEqual(info["commit_sha"], "")
        self.assertEqual(info["drift_count"], 3)
        self.assertCountEqual(info["drift_paths"], ["x.py", "y.py", "z.py"])

    def test_drift_below_threshold_calls_git(self):
        """drift_count < 3 must invoke _git_auto_commit with explicit
        add_targets covering declared ∪ drift."""
        captured = {}

        def fake_git_auto_commit(cwd, label, *, add_targets=None,
                                 commit_message=None):
            captured["cwd"] = cwd
            captured["label"] = label
            captured["add_targets"] = list(add_targets or [])
            captured["commit_message"] = commit_message
            return "abc1234"

        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False), \
             patch("larkhelm.crew._state._git_auto_commit", fake_git_auto_commit):
            info = wf_mod._maybe_auto_commit_finale(
                "/tmp/x",
                self._files(declared=["a.py"],
                            modified=["a.py"],
                            untracked=["x.py"]),  # 1 drift, below threshold
                "fix login",
            )
        self.assertEqual(info["commit_sha"], "abc1234")
        self.assertEqual(captured["label"], "finalize")
        self.assertEqual(captured["add_targets"], ["a.py", "x.py"])
        # Subject embeds the title
        self.assertIn("fix login", captured["commit_message"])
        # Body lists the drift section
        self.assertIn("Drift, auto-included", captured["commit_message"])
        self.assertIn("x.py", captured["commit_message"])

    def test_git_failure_returns_empty_sha(self):
        def fake_git_auto_commit(cwd, label, **kwargs):
            return ""   # git returned empty (e.g. exception inside)

        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False), \
             patch("larkhelm.crew._state._git_auto_commit", fake_git_auto_commit):
            info = wf_mod._maybe_auto_commit_finale(
                "/tmp/x",
                self._files(declared=["a.py"], modified=["a.py"]),
                "title",
            )
        self.assertEqual(info["commit_sha"], "")
        self.assertIn("git error", info["skipped_reason"])

    def test_zero_declared_full_drift_below_threshold_commits(self):
        """When file_changes.json is empty but only 1-2 files are dirty,
        we still want a commit (drift < threshold)."""
        captured = {}

        def fake_git_auto_commit(cwd, label, *, add_targets=None,
                                 commit_message=None):
            captured["add_targets"] = list(add_targets or [])
            return "def5678"

        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False), \
             patch("larkhelm.crew._state._git_auto_commit", fake_git_auto_commit):
            info = wf_mod._maybe_auto_commit_finale(
                "/tmp/x",
                self._files(declared=[], modified=["only.py"]),
                "title",
            )
        self.assertEqual(info["commit_sha"], "def5678")
        self.assertEqual(info["drift_count"], 1)
        self.assertEqual(captured["add_targets"], ["only.py"])


class FormatSummaryWithCommitInfoTests(unittest.TestCase):

    def _files(self):
        return {"from_file_changes": ["a.py"],
                "tracked_modified": [], "untracked": []}

    def test_commit_sha_rendered_when_present(self):
        info = {"commit_sha": "abc1234", "drift_count": 0,
                "drift_paths": [], "skipped_reason": ""}
        body, color = wf_mod._format_workspace_summary(
            self._files(), True, "MyPlan", commit_info=info)
        self.assertIn("自动提交", body)
        self.assertIn("`abc1234`", body)
        self.assertEqual(color, "green")

    def test_drift_warning_when_skipped(self):
        info = {"commit_sha": "", "drift_count": 4,
                "drift_paths": ["x.py", "y.py", "z.py", "w.py"],
                "skipped_reason": "drift_count 4 ≥ 3"}
        body, _ = wf_mod._format_workspace_summary(
            self._files(), True, "P", commit_info=info)
        self.assertIn("已跳过", body)
        self.assertIn("漂移 4 个文件", body)
        # All four drift paths visible (< 5 cap)
        for p in ("x.py", "y.py", "z.py", "w.py"):
            self.assertIn(p, body)
        self.assertIn("手动执行", body)

    def test_drift_warning_caps_path_preview_at_5(self):
        info = {"commit_sha": "", "drift_count": 8,
                "drift_paths": [f"p{i}.py" for i in range(8)],
                "skipped_reason": "drift_count 8 ≥ 3"}
        body, _ = wf_mod._format_workspace_summary(
            self._files(), True, "P", commit_info=info)
        self.assertIn("p0.py", body)
        self.assertIn("p4.py", body)
        self.assertNotIn("p5.py", body)
        self.assertIn("余 3 个", body)

    def test_disabled_reason_renders_nothing(self):
        """When dev_auto_commit=false the rendered body should NOT show a
        '⚠️ 已跳过' line — that would confuse users who never opted in."""
        info = {"commit_sha": "", "drift_count": 0, "drift_paths": [],
                "skipped_reason": "dev_auto_commit=false"}
        body, _ = wf_mod._format_workspace_summary(
            self._files(), True, "P", commit_info=info)
        self.assertNotIn("自动提交", body)

    def test_no_commit_info_argument_keeps_legacy_body(self):
        """Pre-B2 callers that don't pass commit_info must see no
        commit-related section at all."""
        body, _ = wf_mod._format_workspace_summary(
            self._files(), True, "MyPlan")
        self.assertNotIn("自动提交", body)


# ════════════════════════════════════════════════════════════════════════
#  B2 — _git_auto_commit(add_targets=...)
# ════════════════════════════════════════════════════════════════════════


class GitAutoCommitAddTargetsTests(unittest.TestCase):
    """Verify the new white-list mode of ``_git_auto_commit`` (B2)."""

    def setUp(self):
        # Each call to subprocess.run gets recorded so we can introspect.
        self.calls: list[list[str]] = []

        def _fake_run(args, **kwargs):
            self.calls.append(list(args))
            ret = MagicMock()
            ret.returncode = 0
            ret.stdout = "M file.py\n" if args[:2] == ["git", "status"] else ""
            ret.stderr = ""
            return ret
        self._fake_run = _fake_run

    def _import_target(self):
        from larkhelm.crew._state import _git_auto_commit
        return _git_auto_commit

    def test_add_targets_none_uses_full_add(self):
        _git_auto_commit = self._import_target()
        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False), \
             patch("larkhelm.crew._state.subprocess.run", side_effect=self._fake_run), \
             patch("larkhelm.crew._state._git_head", return_value="aaaa111"):
            sha = _git_auto_commit("/tmp/x", "implementer")
        self.assertEqual(sha, "aaaa111")
        # One of the calls must be ``git add -A``
        add_calls = [c for c in self.calls if c[:2] == ["git", "add"]]
        self.assertTrue(any(c == ["git", "add", "-A"] for c in add_calls),
                        f"expected `git add -A`, got {add_calls!r}")

    def test_add_targets_list_uses_explicit_paths(self):
        _git_auto_commit = self._import_target()
        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False), \
             patch("larkhelm.crew._state.subprocess.run", side_effect=self._fake_run), \
             patch("larkhelm.crew._state._git_head", return_value="bbbb222"):
            sha = _git_auto_commit(
                "/tmp/x", "finalize",
                add_targets=["a.py", "b/c.py"],
                commit_message="[finalize] Custom",
            )
        self.assertEqual(sha, "bbbb222")
        add_calls = [c for c in self.calls if c[:2] == ["git", "add"]]
        self.assertTrue(
            any(c == ["git", "add", "--", "a.py", "b/c.py"] for c in add_calls),
            f"expected explicit `git add --`, got {add_calls!r}")
        # Commit message override propagated
        commit_calls = [c for c in self.calls if c[:2] == ["git", "commit"]]
        self.assertTrue(any("[finalize] Custom" in (c[-1] if len(c) > 1 else "")
                            for c in commit_calls))

    def test_empty_add_targets_short_circuits(self):
        _git_auto_commit = self._import_target()
        with patch.dict(_cfg.config, {"dev_auto_commit": True}, clear=False), \
             patch("larkhelm.crew._state.subprocess.run", side_effect=self._fake_run):
            sha = _git_auto_commit(
                "/tmp/x", "finalize",
                add_targets=[],   # explicit empty white-list
            )
        self.assertEqual(sha, "")
        # No ``git add`` / ``git commit`` should have run
        self.assertFalse(any(c[:2] == ["git", "add"] for c in self.calls),
                         f"expected no add call, got {self.calls!r}")
        self.assertFalse(any(c[:2] == ["git", "commit"] for c in self.calls))

    def test_disabled_flag_skips_everything(self):
        _git_auto_commit = self._import_target()
        with patch.dict(_cfg.config, {"dev_auto_commit": False}, clear=False), \
             patch("larkhelm.crew._state.subprocess.run", side_effect=self._fake_run):
            sha = _git_auto_commit(
                "/tmp/x", "finalize", add_targets=["a.py"])
        self.assertEqual(sha, "")
        # No git invocation at all when flag is off
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
