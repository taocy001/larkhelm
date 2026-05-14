"""Tests for ``larkhelm.memory_context`` (S49–S52, Phase B).

Covered:
  * ``should_include_global``  — keyword presence + force + empty fail-open.
  * ``should_include_project`` — cwd gate + keyword + cwd-less rejection +
    /dev|/crew|/plan force-include.
  * ``split_session_slots``    — three-section parse + fallback to raw.
  * ``dedup_recent_turns``     — overlap removal + dedup-flag respect.
  * ``MemoryContextBuilder.build`` — default args byte-equivalent to legacy
    ``get_memory_context``; layered slicing kicks in when query is non-empty
    and config flag is on.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Bootstrap config (shared) ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_memctx_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm import memory_context as mc  # noqa: E402
from larkhelm import memory as mem  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
#  should_include_global / should_include_project
# ════════════════════════════════════════════════════════════════════════


class GlobalGatingTests(unittest.TestCase):

    def test_keyword_match_includes(self):
        self.assertTrue(mc.should_include_global("讨论我的代码风格"))
        self.assertTrue(mc.should_include_global("change my preference"))
        self.assertTrue(mc.should_include_global("set language to English"))

    def test_no_keyword_excludes(self):
        self.assertFalse(mc.should_include_global("今天北京天气"))
        self.assertFalse(mc.should_include_global("multiply 3 by 7"))

    def test_empty_query_fails_open(self):
        self.assertTrue(mc.should_include_global(""))
        self.assertTrue(mc.should_include_global("   "))

    def test_force_overrides(self):
        self.assertTrue(mc.should_include_global("无关键词", force=True))

    def test_lazy_global_disabled_always_includes(self):
        with patch.dict(_cfg.config, {"memory_lazy_global": False}, clear=False):
            self.assertTrue(mc.should_include_global("无关键词"))


class ProjectGatingTests(unittest.TestCase):

    def test_no_cwd_excludes(self):
        self.assertFalse(mc.should_include_project("修复 bug", None, False))
        self.assertFalse(mc.should_include_project("修复 bug", "", False))

    def test_keyword_match_includes(self):
        self.assertTrue(mc.should_include_project("修复 bug", "/x", False))
        self.assertTrue(mc.should_include_project("write a test", "/x", False))
        self.assertTrue(mc.should_include_project("refactor the module", "/x", False))

    def test_no_keyword_excludes(self):
        self.assertFalse(mc.should_include_project("讲个笑话", "/x", False))

    def test_force_overrides(self):
        self.assertTrue(mc.should_include_project("讲个笑话", "/x", False, force=True))

    def test_dev_prefix_force_includes(self):
        self.assertTrue(mc.should_include_project("/dev fix login", "/x", False))
        self.assertTrue(mc.should_include_project("/crew plan refactor", "/x", False))
        self.assertTrue(mc.should_include_project("/plan ship release", "/x", False))

    def test_doc_url_includes(self):
        self.assertTrue(mc.should_include_project("see this", "/x", True))

    def test_path_fragment_triggers(self):
        self.assertTrue(mc.should_include_project("look at handlers/_query.py", "/x", False))
        self.assertTrue(mc.should_include_project("inspect /etc/larkhelm/config.json", "/x", False))

    def test_code_fence_triggers(self):
        self.assertTrue(mc.should_include_project("```python\nprint(1)\n```", "/x", False))

    def test_empty_query_fails_open(self):
        self.assertTrue(mc.should_include_project("", "/x", False))


# ════════════════════════════════════════════════════════════════════════
#  split_session_slots
# ════════════════════════════════════════════════════════════════════════


class SplitSessionSlotsTests(unittest.TestCase):

    def test_three_section_parse(self):
        body = (
            "## Work Context\n"
            "Working on fix Y in larkhelm.\n\n"
            "## Key Decisions & Facts\n"
            "Picked X over Z.\n\n"
            "## Next Steps\n"
            "Continue refactor.\n"
        )
        slots = mc.split_session_slots(body)
        self.assertTrue(slots.parsed)
        self.assertIn("Working on fix Y", slots.work_context)
        self.assertIn("Picked X over Z", slots.decisions)
        self.assertIn("Continue refactor", slots.history)

    def test_fallback_when_no_sections(self):
        slots = mc.split_session_slots("无结构 raw text")
        self.assertFalse(slots.parsed)
        self.assertEqual(slots.work_context, "无结构 raw text")
        self.assertEqual(slots.raw, "无结构 raw text")

    def test_empty_returns_empty_slots(self):
        slots = mc.split_session_slots("")
        self.assertFalse(slots.parsed)
        self.assertEqual(slots.work_context, "")

    def test_chinese_headers_three_section_parse(self):
        # The summariser tells the LLM to emit H2 sections "in the SAME
        # LANGUAGE as the conversation". Chinese sessions therefore
        # produce 工作上下文 / 关键决策 / 后续步骤 (or 下一步). All three
        # must classify into the right slot — previously `## 后续步骤`
        # fell through to None because the `history` branch's Chinese
        # vocabulary only covered "进展".
        body = (
            "## 工作上下文\n"
            "在 larkhelm 修 bug Y。\n\n"
            "## 关键决策\n"
            "选择 X 而不是 Z。\n\n"
            "## 后续步骤\n"
            "继续重构。\n"
        )
        slots = mc.split_session_slots(body)
        self.assertTrue(slots.parsed)
        self.assertIn("修 bug Y", slots.work_context)
        self.assertIn("选择 X", slots.decisions)
        self.assertIn("继续重构", slots.history)

    def test_chinese_alternative_next_step_synonym(self):
        # `## 下一步` and `## 步骤` should also map to history.
        for header in ("## 下一步", "## 步骤", "## 进展"):
            with self.subTest(header=header):
                body = f"{header}\n要写测试\n"
                slots = mc.split_session_slots(body)
                self.assertTrue(slots.parsed, f"failed to parse {header!r}")
                self.assertIn("要写测试", slots.history)


# ════════════════════════════════════════════════════════════════════════
#  dedup_recent_turns
# ════════════════════════════════════════════════════════════════════════


class DedupRecentTurnsTests(unittest.TestCase):

    def test_drops_overlapping_entry(self):
        recent = [
            "[12:00] user: A first long message about refactoring login",
            "[12:01] user: B unique question about deploys",
        ]
        session_body = "## Work Context\nA first long message about refactoring login earlier today."
        out = mc.dedup_recent_turns(recent, session_body)
        # The "A first..." line should be dropped; "B" survives.
        self.assertEqual(len(out), 1)
        self.assertIn("B unique", out[0])

    def test_no_overlap_keeps_all(self):
        recent = ["[12:00] user: nothing related"]
        session_body = "## Work Context\nUnrelated summary."
        out = mc.dedup_recent_turns(recent, session_body)
        self.assertEqual(out, recent)

    def test_empty_session_keeps_all(self):
        recent = ["[12:00] user: anything"]
        out = mc.dedup_recent_turns(recent, "")
        self.assertEqual(out, recent)

    def test_dedup_flag_disabled_keeps_all(self):
        recent = ["[12:00] user: A first long message about refactoring login"]
        session_body = "A first long message about refactoring login"
        with patch.dict(_cfg.config, {"memory_recent_turns_dedup": False}, clear=False):
            out = mc.dedup_recent_turns(recent, session_body)
        self.assertEqual(out, recent)


# ════════════════════════════════════════════════════════════════════════
#  MemoryContextBuilder
# ════════════════════════════════════════════════════════════════════════


class MemoryContextBuilderTests(unittest.TestCase):

    def test_default_args_equivalent_to_legacy(self):
        """``MemoryContextBuilder(chat, cwd).build()`` with no other args
        must produce the same string as the legacy ``get_memory_context``."""
        with patch.object(mem, "load_global_memory", return_value="GLOBAL_X"), \
             patch.object(mem, "load_project_memory", return_value="PROJECT_Y"), \
             patch.object(mem, "load_memory", return_value="SESSION_Z"):
            new_out = mc.MemoryContextBuilder("chat1", "/x").build()
            legacy_out = mem.get_memory_context("chat1", "/x")
        self.assertEqual(new_out, legacy_out)

    def test_global_excluded_when_query_lacks_keywords(self):
        with patch.object(mem, "load_global_memory", return_value="GLOBAL_X"), \
             patch.object(mem, "load_project_memory", return_value=None), \
             patch.object(mem, "load_memory", return_value="SESSION_Z"):
            out = mc.MemoryContextBuilder(
                "chat1", "/x", query="今天北京天气如何",
            ).build()
        self.assertNotIn("GLOBAL_X", out)
        self.assertIn("SESSION_Z", out)

    def test_project_excluded_when_query_lacks_keywords(self):
        with patch.object(mem, "load_global_memory", return_value=None), \
             patch.object(mem, "load_project_memory", return_value="PROJECT_Y"), \
             patch.object(mem, "load_memory", return_value="SESSION_Z"):
            out = mc.MemoryContextBuilder(
                "chat1", "/x", query="讲个笑话",
            ).build()
        self.assertNotIn("PROJECT_Y", out)

    def test_layered_session_when_query_present(self):
        body = (
            "## Work Context\nWORK_BODY\n\n"
            "## Key Decisions & Facts\nDECISION_BODY\n\n"
            "## Next Steps\nNEXT_BODY\n"
        )
        with patch.object(mem, "load_global_memory", return_value=None), \
             patch.object(mem, "load_project_memory", return_value=None), \
             patch.object(mem, "load_memory", return_value=body):
            out = mc.MemoryContextBuilder(
                "chat1", "/x", query="quick code question",
            ).build()
        self.assertIn("WORK_BODY", out)

    def test_lazy_global_disabled_always_includes(self):
        with patch.dict(_cfg.config, {"memory_lazy_global": False}, clear=False), \
             patch.object(mem, "load_global_memory", return_value="GLOBAL_X"), \
             patch.object(mem, "load_project_memory", return_value=None), \
             patch.object(mem, "load_memory", return_value=None):
            out = mc.MemoryContextBuilder(
                "chat1", "/x", query="今天天气",
            ).build()
        self.assertIn("GLOBAL_X", out)

    def test_build_for_crew_emits_project_and_session_only(self):
        with patch.object(mem, "load_global_memory", return_value="GLOBAL_X"), \
             patch.object(mem, "load_project_memory", return_value="PROJECT_Y"), \
             patch.object(mem, "load_memory", return_value="SESSION_Z"):
            out = mc.MemoryContextBuilder(
                "chat1", "/x", force_project=True,
            ).build_for_crew()
        self.assertIn("PROJECT_Y", out)
        self.assertIn("SESSION_Z", out)
        self.assertNotIn("GLOBAL_X", out)


if __name__ == "__main__":
    unittest.main()
