"""Coverage for the P0 memory-system hardening (memory_review_final.md §4):

  1. ``_is_useful_summary`` validator: empty / refusal-prefix / too-short
     LLM outputs must be rejected so they don't pollute ``existing_memory``
     in the next round.
  2. ``_save_md`` explicit ``chmod 0600`` on tmp file before atomic replace
     so memory files never inherit a broader umask (aligns with audit /
     feedback JSONL files which already enforce 0600).
  3. ``[memory]`` → ``[Memory]`` prefix consistency: every ``_debug_log`` /
     ``safe_log`` / ``lazy_debug_log`` call in memory.py must use PascalCase.
"""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import memory


# ── 1. _is_useful_summary validator ──────────────────────────────────────


class TestIsUsefulSummary(unittest.TestCase):

    def test_none_rejected(self):
        self.assertFalse(memory._is_useful_summary(None))

    def test_empty_rejected(self):
        self.assertFalse(memory._is_useful_summary(""))
        self.assertFalse(memory._is_useful_summary("   "))
        self.assertFalse(memory._is_useful_summary("\n\n\t"))

    def test_too_short_rejected(self):
        # < 50 chars after strip
        short = "x" * (memory._MIN_USEFUL_SUMMARY_CHARS - 1)
        self.assertFalse(memory._is_useful_summary(short))
        # exactly at threshold passes
        ok = "x" * memory._MIN_USEFUL_SUMMARY_CHARS
        self.assertTrue(memory._is_useful_summary(ok))

    def test_english_refusal_prefixes_rejected(self):
        bad = [
            "I cannot fulfill this request because the content violates policy.",
            "I can't help with that, please consult an expert in your area.",
            "I'm sorry, but I can't generate the requested summary today.",
            "I am sorry, this request is outside my scope of assistance.",
            "As an AI language model, I do not retain memory between sessions.",
            "As a language model, I cannot remember our previous conversation.",
            "Sorry, but the system encountered an unrecoverable error here.",
        ]
        for b in bad:
            self.assertFalse(memory._is_useful_summary(b),
                             msg=f"should reject: {b[:40]!r}")

    def test_chinese_refusal_prefixes_rejected(self):
        bad = [
            "抱歉，我无法完成这个请求，请尝试更具体的指令再调用一次。",
            "对不起，由于内容限制我无法生成本次记忆摘要内容。",
            "我无法回答这个问题，因为它超出了我能处理的领域范围。",
            "作为一个AI语言模型，我并没有记忆能力来保留之前的对话。",
            "作为 AI 助手我没有访问持久存储的能力，所以无法记住。",
        ]
        for b in bad:
            self.assertFalse(memory._is_useful_summary(b),
                             msg=f"should reject: {b[:40]!r}")

    def test_useful_summary_accepted(self):
        good = (
            "## Work Context\nProject: larkhelm. Currently refactoring "
            "memory system with milestone-driven auto-update.\n\n"
            "## Key Decisions\n- record_milestone debounced 60s"
        )
        self.assertTrue(memory._is_useful_summary(good))

    def test_refusal_prefix_only_matches_at_head(self):
        """An apology somewhere mid-string should NOT trigger the gate —
        only the first 80 chars are scanned."""
        ok = (
            "## Work Context\nUser asked the assistant about working hours, "
            "but later admitted: I cannot work on weekends. So we agreed."
        )
        self.assertTrue(memory._is_useful_summary(ok),
                        "apology mid-text must not trip the head-only check")


class TestGenerateMemoryRejectsNonUsefulOutput(unittest.TestCase):
    """``generate_memory`` must raise ``ValueError`` (not silently return)
    when the LLM output fails the validator. The caller relies on the
    exception to skip ``save_memory`` and preserve the previous summary."""

    def test_valueerror_on_empty_llm_output(self):
        with patch.object(memory, "_run_one_shot", return_value=""):
            with self.assertRaises(ValueError):
                memory.generate_memory("chat_x", "some logs", existing_memory=None)

    def test_valueerror_on_refusal_prefix(self):
        with patch.object(memory, "_run_one_shot",
                          return_value="I cannot help with that, sorry."):
            with self.assertRaises(ValueError):
                memory.generate_memory("chat_x", "some logs")

    def test_valueerror_on_too_short(self):
        with patch.object(memory, "_run_one_shot", return_value="ok\n"):
            with self.assertRaises(ValueError):
                memory.generate_memory("chat_x", "some logs")

    def test_returns_normally_on_useful_output(self):
        good = "## Work Context\n" + ("This is a real summary. " * 5)
        with patch.object(memory, "_run_one_shot", return_value=good):
            out = memory.generate_memory("chat_x", "some logs")
        self.assertEqual(out, good[:memory.SESSION_MAX_CHARS])

    def test_session_memory_preserved_on_rejection(self):
        """Integration: ``maybe_auto_update``'s ``_gen`` thread captures the
        ValueError, the outer except path logs + notifies, ``save_memory``
        is NEVER called → old session memory file is preserved on disk.

        Implementation note: we patch ``load_memory`` to return a sentinel
        previous summary, capture the ``on_done`` callback to synchronize
        with the background thread (rather than polling), and assert the
        error code is the rejection ``ValueError``. Earlier iteration of
        this test relied on ``_cfg.DATA_DIR`` being set; if it wasn't,
        ``load_memory`` raised before ``_gen`` ran and ``save_memory`` was
        skipped for the wrong reason — the test passed vacuously. This
        version isolates the failure path under test.
        """
        ts = "2026-05-09T22:00:00"
        records = [
            {"ts": ts, "chat_id": "c_p0", "role": "user", "content": "hi", "model": "claude"},
            {"ts": ts, "chat_id": "c_p0", "role": "assistant", "content": "hello", "model": "claude"},
        ]
        prior_session = (
            "## Work Context\nProject: larkhelm. Current task: real previous "
            "memory that must NOT be overwritten by a refusal output."
        )
        done_evt = __import__("threading").Event()
        captured: dict = {}

        def _on_done(ok, content, err):
            captured["ok"], captured["content"], captured["err"] = ok, content, err
            done_evt.set()

        with patch.object(memory, "_read_logs", return_value=records), \
             patch.object(memory, "_get_turn_count", return_value=10), \
             patch.object(memory, "load_memory", return_value=prior_session), \
             patch.object(memory, "_run_one_shot",
                          return_value="I cannot help with that, sorry."), \
             patch.object(memory, "save_memory") as save_m, \
             patch.object(memory, "_cascade_extract"):
            memory.maybe_auto_update("c_p0", force=True, on_done=_on_done)
            # Wait for the background ``_run`` thread to actually call on_done,
            # not for a side-effect that may never come.
            self.assertTrue(done_evt.wait(timeout=5.0),
                            "maybe_auto_update background thread did not finish")
        # The chain ran the ValueError path — not some unrelated AttributeError —
        # so save_memory must NOT have been called and the failure code must
        # mention the validator rejection.
        save_m.assert_not_called()
        self.assertFalse(captured["ok"])
        self.assertIsNotNone(captured["err"])
        self.assertIn("non-useful summary", captured["err"])


# ── 2. _save_md chmod 0600 ───────────────────────────────────────────────


class TestSaveMd0600(unittest.TestCase):
    """``_save_md`` must produce a 0600 file regardless of process umask."""

    def test_perm_is_0600_under_default_umask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session_x.md"
            memory._save_md(path, "## Test\n" + "x" * 100, max_chars=500)
            self.assertTrue(path.exists())
            mode = os.stat(path).st_mode & 0o777
            self.assertEqual(mode, 0o600,
                             f"expected 0600, got {oct(mode)}")

    def test_perm_is_0600_under_lax_umask(self):
        """Even with a 022 umask (which would normally yield 0644), the
        explicit chmod must close the gap."""
        old_umask = os.umask(0o022)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "project_x.md"
                memory._save_md(path, "## Test\n" + "x" * 100, max_chars=500)
                mode = os.stat(path).st_mode & 0o777
                self.assertEqual(mode, 0o600)
        finally:
            os.umask(old_umask)

    def test_chmod_failure_does_not_abort_write(self):
        """If chmod itself fails (e.g. on a CIFS mount), the write must
        still complete — we log and move on rather than lose the memory."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "global_x.md"
            with patch("os.chmod", side_effect=OSError("operation not supported")):
                memory._save_md(path, "## Hi\n" + "x" * 100, max_chars=500)
            self.assertTrue(path.exists())  # write completed

    def test_none_path_is_noop(self):
        # Should not raise.
        memory._save_md(None, "anything", max_chars=10)

    def test_atomic_replace_no_tmp_leftover(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session_y.md"
            memory._save_md(path, "## Test\n" + "x" * 100, max_chars=500)
            # No leftover ``.md.tmp`` file.
            self.assertFalse(path.with_suffix(".md.tmp").exists())


# ── 3. [memory] → [Memory] log prefix consistency ────────────────────────


class TestMemoryLogPrefixConsistency(unittest.TestCase):
    """Every ``_debug_log`` / ``safe_log`` / ``lazy_debug_log`` line in
    ``memory.py`` must use ``[Memory]`` PascalCase, no surviving lowercase
    ``[memory]`` entries (per CLAUDE.md PascalCase rule)."""

    def test_no_lowercase_memory_prefix_remains(self):
        src = Path("larkhelm/memory.py").read_text(encoding="utf-8")
        # Strip docstrings/comments would be over-engineering; just look at
        # f-string call patterns where the prefix actually matters.
        lowercase_hits = re.findall(
            r'_debug_log\(f?"\[memory\]', src
        )
        self.assertEqual(
            lowercase_hits, [],
            f"found {len(lowercase_hits)} surviving [memory] (lowercase) call(s); "
            "rename to [Memory] per CLAUDE.md prefix convention"
        )

    def test_memory_log_calls_are_pascalcase(self):
        """Spot-check: at least 10 [Memory] PascalCase log call sites
        (the file is verbose; this confirms the rename actually fired)."""
        src = Path("larkhelm/memory.py").read_text(encoding="utf-8")
        pascal_hits = re.findall(r'_debug_log\(f?"\[Memory\]', src)
        self.assertGreater(
            len(pascal_hits), 10,
            f"expected >10 [Memory] log calls, found {len(pascal_hits)}"
        )


if __name__ == "__main__":
    unittest.main()
