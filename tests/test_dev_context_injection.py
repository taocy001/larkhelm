"""Coverage for the /dev context-injection + stale-TTL changes.

Two related bugs the cleanup addresses:

1. **Context blindness**: ``_run_dev_crew_inner`` previously fed only the
   literal /dev arg into the PM agent's prompt. Vague references like
   "实现刚才讨论的方案" had no anchor, leading PM to either probe the
   filesystem (and risk reading stale ``.crew_workspace/`` artifacts) or
   hallucinate a task. ``_augment_requirement_with_context`` now prepends
   recent chat turns + memory snippets.

2. **Stale workspace reuse**: the original "same hash + uncompleted ⇒
   reuse design.md" resume path silently inherited weeks-old artifacts.
   A 24h TTL guard now clears workspaces older than that even when the
   hash matches.

These tests target the helpers directly (no live LLM, no real workspace).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from larkhelm.crew import _commands as cmd
from larkhelm.crew._commands import (
    _augment_requirement_with_context,
    _task_hash,
    _WORKSPACE_STALE_TTL,
    _read_workspace_meta,
    _write_workspace_meta,
)


# ── Helper: invoke the inner stale-TTL decision branch in isolation ──


def _decide_stale(meta_path: Path, meta: dict, task_hash: str) -> bool:
    """Mirror ``_run_dev_crew_inner``'s stale-detection condition without
    spinning up the full crew machinery. Returns True if the decision is
    'clear workspace'."""
    is_stale_age = False
    if meta and meta_path.exists():
        try:
            age_sec = time.time() - meta_path.stat().st_mtime
            is_stale_age = age_sec > _WORKSPACE_STALE_TTL
        except OSError:
            pass
    return bool(meta) and (
        meta.get("task_hash") != task_hash
        or meta.get("completed")
        or is_stale_age
    )


# ── 1. Stale-TTL guard ─────────────────────────────────────────────────


class TestStaleTTL(unittest.TestCase):
    """The stale-TTL decision must clear workspaces older than 24h even when
    task_hash still matches and the previous run wasn't completed."""

    def test_constant_is_24_hours(self):
        self.assertEqual(_WORKSPACE_STALE_TTL, 24 * 3600)

    def test_fresh_workspace_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            meta_file = ws / "workspace_meta.json"
            _write_workspace_meta(ws, task_hash="abc", completed=False)
            # Just-now mtime, hash matches, not completed → KEEP.
            self.assertFalse(_decide_stale(meta_file, {"task_hash": "abc", "completed": False}, "abc"))

    def test_stale_workspace_cleared_even_when_hash_matches(self):
        """Core regression: same task_hash + uncompleted + 25h old should clear."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            meta_file = ws / "workspace_meta.json"
            _write_workspace_meta(ws, task_hash="abc", completed=False)
            # Backdate to 25h ago.
            old = time.time() - 25 * 3600
            os.utime(meta_file, (old, old))
            self.assertTrue(_decide_stale(meta_file, {"task_hash": "abc", "completed": False}, "abc"))

    def test_just_under_threshold_still_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            meta_file = ws / "workspace_meta.json"
            _write_workspace_meta(ws, task_hash="abc", completed=False)
            # 23h ago — still under 24h, must keep.
            old = time.time() - 23 * 3600
            os.utime(meta_file, (old, old))
            self.assertFalse(_decide_stale(meta_file, {"task_hash": "abc", "completed": False}, "abc"))

    def test_hash_diff_clears_regardless_of_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            meta_file = ws / "workspace_meta.json"
            _write_workspace_meta(ws, task_hash="abc", completed=False)
            self.assertTrue(_decide_stale(meta_file, {"task_hash": "abc", "completed": False}, "different_hash"))

    def test_completed_clears_regardless_of_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            meta_file = ws / "workspace_meta.json"
            _write_workspace_meta(ws, task_hash="abc", completed=True)
            # Completed previous run = different task expected next.
            self.assertTrue(_decide_stale(meta_file, {"task_hash": "abc", "completed": True}, "abc"))

    def test_no_meta_does_not_trigger_stale_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            # Empty meta dict → first run; nothing to clear.
            self.assertFalse(_decide_stale(ws / "workspace_meta.json", {}, "abc"))


# ── 2. Context injection (passthrough + content + caps) ──────────────


class TestContextAugmentation(unittest.TestCase):

    def test_passthrough_when_no_context_available(self):
        """With both helpers returning empty, requirement must be unchanged."""
        with patch("larkhelm.log._get_recent_turns", return_value=""), \
             patch("larkhelm.memory.get_memory_context_v2", return_value=("", [])):
            out = _augment_requirement_with_context("实现 X", "oc_test", "/tmp")
        self.assertEqual(out, "实现 X")

    def test_chat_only_injection(self):
        with patch("larkhelm.log._get_recent_turns",
                   return_value="User: 我们要做 Y\nAssistant: 好的"), \
             patch("larkhelm.memory.get_memory_context_v2", return_value=("", [])):
            out = _augment_requirement_with_context("实现 X", "oc_test", "/tmp")
        self.assertTrue(out.startswith("实现 X"),
                        "user-typed requirement must come FIRST so PM/title both see it")
        self.assertIn("最近对话", out)
        self.assertIn("我们要做 Y", out)
        self.assertNotIn("长期记忆", out)

    def test_memory_only_injection(self):
        with patch("larkhelm.log._get_recent_turns", return_value=""), \
             patch("larkhelm.memory.get_memory_context_v2",
                   return_value=("[GLOBAL MEMORY]\n用户偏好中文输出\n[/GLOBAL MEMORY]", [])):
            out = _augment_requirement_with_context("实现 X", "oc_test", "/tmp")
        self.assertTrue(out.startswith("实现 X"))
        self.assertIn("长期记忆", out)
        self.assertIn("用户偏好中文输出", out)
        self.assertNotIn("最近对话", out)

    def test_both_injected(self):
        with patch("larkhelm.log._get_recent_turns",
                   return_value="User: 我们要做 Y"), \
             patch("larkhelm.memory.get_memory_context_v2",
                   return_value=("[PROJECT MEMORY]\n用 PascalCase\n[/PROJECT MEMORY]", [])):
            out = _augment_requirement_with_context("实现 X", "oc_test", "/tmp")
        self.assertIn("长期记忆", out)
        self.assertIn("最近对话", out)
        # Both must appear AFTER the user requirement.
        self.assertLess(out.index("实现 X"), out.index("长期记忆"))
        self.assertLess(out.index("长期记忆"), out.index("最近对话"))

    def test_memory_truncated_when_oversized(self):
        big_memory = "x" * 10_000  # > _DEV_CTX_MEMORY_CHARS (2000)
        with patch("larkhelm.log._get_recent_turns", return_value=""), \
             patch("larkhelm.memory.get_memory_context_v2", return_value=(big_memory, [])):
            out = _augment_requirement_with_context("实现 X", "oc_test", "/tmp")
        self.assertIn("(truncated)", out)
        # Memory body capped; the rest of the augmented prompt must still fit.
        self.assertLess(len(out), 10_000)

    def test_helper_failures_swallowed(self):
        """When EITHER helper raises, augmentation must still return a
        usable string (degraded, not crashed)."""
        with patch("larkhelm.log._get_recent_turns",
                   side_effect=RuntimeError("log dir missing")), \
             patch("larkhelm.memory.get_memory_context_v2",
                   side_effect=OSError("no permission")):
            out = _augment_requirement_with_context("实现 X", "oc_test", "/tmp")
        # Both helpers failed → context empty → requirement unchanged.
        self.assertEqual(out, "实现 X")

    def test_disclaimer_present_when_context_attached(self):
        """The augmented prompt must remind the LLM that the trailing context
        is BACKGROUND only, not the new requirement."""
        with patch("larkhelm.log._get_recent_turns", return_value="User: hi"), \
             patch("larkhelm.memory.get_memory_context_v2", return_value=("", [])):
            out = _augment_requirement_with_context("实现 X", "oc_test", "/tmp")
        self.assertIn("仅供 PM 阶段理解需求", out)
        self.assertIn("以需求为准", out)


# ── 3. Hash stability — task_key still hashes the literal ──────────────


class TestTaskKeyHashStability(unittest.TestCase):
    """Critical invariant: ``task_hash`` must NOT shift just because chat
    history grew. Otherwise the resume / stale-detect logic chains fall
    apart (every /dev re-run would compute a new hash and clear workspace)."""

    def test_hash_stable_across_chat_growth(self):
        requirement = "实现 X 功能"
        h_baseline = _task_hash(requirement)

        # Simulate two different chat-history snapshots.
        for chat_text in ("", "user A turn", "much longer\nmulti\nline\nhistory"):
            with patch("larkhelm.log._get_recent_turns", return_value=chat_text), \
                 patch("larkhelm.memory.get_memory_context_v2", return_value=("", [])):
                aug = _augment_requirement_with_context(requirement, "oc_test", "/tmp")
            # The CALLER hashes the ORIGINAL literal, not the augmented form —
            # this test pins that contract.
            self.assertEqual(_task_hash(requirement), h_baseline,
                             "literal hash must be deterministic regardless of chat ctx")
            # And the augmented form must START with the literal so any
            # accidental hashing of `aug[:N]` still picks up the user's
            # original wording (defensive belt-and-suspenders).
            self.assertTrue(aug.startswith(requirement))

    def test_title_truncation_stays_correct(self):
        """``_make_dev_pipeline`` builds title from the first line of
        ``requirement``. Augmentation appends background AFTER a newline
        separator, so title shows the user's literal — not separator text."""
        from larkhelm.crew._pipeline import _make_dev_pipeline
        long_user_req = "实现 LarkHelm 统一日志方案 Phase 1+2+3"
        with patch("larkhelm.log._get_recent_turns", return_value="User: t"), \
             patch("larkhelm.memory.get_memory_context_v2", return_value=("", [])):
            aug = _augment_requirement_with_context(long_user_req, "oc_test", "/tmp")
        # ``_make_dev_pipeline`` reads ``_cfg.RESPONSE_TIMEOUT`` which only
        # exists after ``_init_runtime`` runs; stub it for this offline test.
        with patch("larkhelm.config.RESPONSE_TIMEOUT", 60, create=True):
            plan = _make_dev_pipeline(aug, "/tmp", no_confirm=True)
        self.assertEqual(plan.title, f"软件开发：{long_user_req[:30]}")

    def test_short_requirement_does_not_leak_separator_into_title(self):
        """Regression: previously ``aug[:30]`` for a 4-char requirement
        leaked the ``\\n\\n---\\n\\n## 任务背景上下文…`` separator into the
        card title. The first-line-only fix in ``_make_dev_pipeline`` keeps
        the title clean even when augmentation attaches background text."""
        from larkhelm.crew._pipeline import _make_dev_pipeline
        short_req = "X 实现"  # 4 chars; aug[:30] would have leaked separator
        with patch("larkhelm.log._get_recent_turns", return_value="User: t"), \
             patch("larkhelm.memory.get_memory_context_v2", return_value=("", [])):
            aug = _augment_requirement_with_context(short_req, "oc_test", "/tmp")
        with patch("larkhelm.config.RESPONSE_TIMEOUT", 60, create=True):
            plan = _make_dev_pipeline(aug, "/tmp", no_confirm=True)
        self.assertEqual(plan.title, f"软件开发：{short_req}")
        self.assertNotIn("---", plan.title)
        self.assertNotIn("任务背景", plan.title)


# ── 4. Workspace meta helpers ──────────────────────────────────────────


class TestWorkspaceMetaIO(unittest.TestCase):

    def test_read_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_read_workspace_meta(Path(tmp)), {})

    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _write_workspace_meta(ws, task_hash="deadbeef", completed=True)
            meta = _read_workspace_meta(ws)
            # B3 extended schema: core fields always present; new fields default to safe values
            self.assertEqual(meta["task_hash"], "deadbeef")
            self.assertTrue(meta["completed"])
            self.assertEqual(meta.get("commit_sha", ""), "")
            self.assertEqual(meta.get("finalized_at", 0.0), 0.0)
            self.assertEqual(meta.get("chat_id", ""), "")
            self.assertEqual(meta.get("plan_id", ""), "")

    def test_corrupted_meta_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "workspace_meta.json").write_text("not valid json")
            # Must not raise; treats as no-meta state.
            self.assertEqual(_read_workspace_meta(ws), {})


if __name__ == "__main__":
    unittest.main()
