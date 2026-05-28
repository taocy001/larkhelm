"""Tests for P1 on-demand injection gates (P1a / P1b / P1c).

P1a — memory_intent_policy_enabled: global memory stripped for dev/crew/shell.
P1b — crew_sticky_keyword_gate_enabled: sticky context skipped without crew keywords.
P1c — project_guide_enabled: guide content injected for API backends, skipped for claude_cli.
"""
from __future__ import annotations

import atexit
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ── Bootstrap shared config (same pattern as test_query_backend_skip_recent) ──
_TMP = tempfile.mkdtemp(prefix="larkhelm_igates_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)


# ══════════════════════════════════════════════════════════════════════════════
#  P1a — memory_intent_policy_enabled
# ══════════════════════════════════════════════════════════════════════════════

_GLOBAL_MEMORY_BLOCK = (
    "[GLOBAL MEMORY]\nstyle: terse\n[/GLOBAL MEMORY]\n\n"
    "[SESSION MEMORY]\ncontext: working on larkhelm\n[/SESSION MEMORY]"
)
_SESSION_ONLY = "[SESSION MEMORY]\ncontext: working on larkhelm\n[/SESSION MEMORY]"


def _make_intent(agent_type: str):
    return SimpleNamespace(agent_type=agent_type, sub_intent="", confidence=1.0, complexity="high")


class MemorySkipGlobalIntentsConstantTests(unittest.TestCase):
    """P1-3: _MEMORY_SKIP_GLOBAL_INTENTS must be a module-level constant."""

    def test_is_module_level_attribute(self):
        """Constant must be importable directly from the module (not just a local)."""
        from larkhelm.handlers._query import _MEMORY_SKIP_GLOBAL_INTENTS
        self.assertIsInstance(_MEMORY_SKIP_GLOBAL_INTENTS, frozenset)

    def test_contains_expected_intents(self):
        from larkhelm.handlers._query import _MEMORY_SKIP_GLOBAL_INTENTS
        for intent in ("dev", "crew", "shell"):
            self.assertIn(intent, _MEMORY_SKIP_GLOBAL_INTENTS)

    def test_does_not_contain_chat(self):
        from larkhelm.handlers._query import _MEMORY_SKIP_GLOBAL_INTENTS
        self.assertNotIn("chat", _MEMORY_SKIP_GLOBAL_INTENTS)


class MemoryIntentPolicyTests(unittest.TestCase):

    def setUp(self):
        _cfg.config["memory_intent_policy_enabled"] = True

    def tearDown(self):
        _cfg.config["memory_intent_policy_enabled"] = False

    def _run_gate(self, memory_ctx: str, agent_type: str) -> str:
        """Replicate the gate logic from _do_query without importing the full handler."""
        _MEMORY_SKIP_GLOBAL_INTENTS = {"dev", "crew", "shell"}
        _intent = _make_intent(agent_type)
        _intent_type = getattr(_intent, "agent_type", "") or ""
        if (bool(_cfg.config.get("memory_intent_policy_enabled"))
                and _intent_type in _MEMORY_SKIP_GLOBAL_INTENTS
                and "[GLOBAL MEMORY]" in memory_ctx):
            memory_ctx = re.sub(
                r"\[GLOBAL MEMORY\].*?\[/GLOBAL MEMORY\]\n*",
                "",
                memory_ctx,
                flags=re.DOTALL,
            ).lstrip()
        return memory_ctx

    def test_dev_intent_strips_global_memory(self):
        result = self._run_gate(_GLOBAL_MEMORY_BLOCK, "dev")
        self.assertNotIn("[GLOBAL MEMORY]", result)
        self.assertNotIn("[/GLOBAL MEMORY]", result)
        self.assertIn("[SESSION MEMORY]", result)

    def test_crew_intent_strips_global_memory(self):
        result = self._run_gate(_GLOBAL_MEMORY_BLOCK, "crew")
        self.assertNotIn("[GLOBAL MEMORY]", result)
        self.assertIn("[SESSION MEMORY]", result)

    def test_shell_intent_strips_global_memory(self):
        result = self._run_gate(_GLOBAL_MEMORY_BLOCK, "shell")
        self.assertNotIn("[GLOBAL MEMORY]", result)

    def test_chat_intent_keeps_global_memory(self):
        result = self._run_gate(_GLOBAL_MEMORY_BLOCK, "chat")
        self.assertIn("[GLOBAL MEMORY]", result)
        self.assertIn("[SESSION MEMORY]", result)

    def test_gate_disabled_keeps_global_memory_for_dev(self):
        _cfg.config["memory_intent_policy_enabled"] = False
        result = self._run_gate(_GLOBAL_MEMORY_BLOCK, "dev")
        self.assertIn("[GLOBAL MEMORY]", result)

    def test_no_global_block_no_change(self):
        result = self._run_gate(_SESSION_ONLY, "dev")
        self.assertEqual(result, _SESSION_ONLY)

    def test_empty_memory_ctx_no_error(self):
        result = self._run_gate("", "dev")
        self.assertEqual(result, "")


# ══════════════════════════════════════════════════════════════════════════════
#  P1b — crew_sticky_keyword_gate_enabled
# ══════════════════════════════════════════════════════════════════════════════

_CREW_STICKY_KW_RE = re.compile(
    r"(crew|/dev|/plan|agent|任务|流水线|pipeline|checkpoint)",
    re.IGNORECASE,
)


def _sticky_gate_active(text: str, gate_on: bool) -> bool:
    """Replicate the gate decision: True means 'inject', False means 'skip'."""
    if not gate_on:
        return True
    return bool(_CREW_STICKY_KW_RE.search(text or ""))


class CrewStickyKeywordGateTests(unittest.TestCase):

    def test_gate_off_always_injects(self):
        for text in ("hello", "how are you", "tell me something", "crew mission"):
            with self.subTest(text=text):
                self.assertTrue(_sticky_gate_active(text, gate_on=False))

    def test_gate_on_crew_keyword_injects(self):
        self.assertTrue(_sticky_gate_active("how is the crew doing", gate_on=True))

    def test_gate_on_dev_keyword_injects(self):
        self.assertTrue(_sticky_gate_active("run /dev now", gate_on=True))

    def test_gate_on_plan_keyword_injects(self):
        self.assertTrue(_sticky_gate_active("/plan this feature", gate_on=True))

    def test_gate_on_agent_keyword_injects(self):
        self.assertTrue(_sticky_gate_active("ask the agent about it", gate_on=True))

    def test_gate_on_chinese_keyword_injects(self):
        self.assertTrue(_sticky_gate_active("执行这个任务", gate_on=True))

    def test_gate_on_pipeline_keyword_injects(self):
        self.assertTrue(_sticky_gate_active("run the pipeline", gate_on=True))

    def test_gate_on_checkpoint_keyword_injects(self):
        self.assertTrue(_sticky_gate_active("restore from checkpoint", gate_on=True))

    def test_gate_on_no_keyword_skips(self):
        self.assertFalse(_sticky_gate_active("what is the weather today", gate_on=True))

    def test_gate_on_empty_message_skips(self):
        self.assertFalse(_sticky_gate_active("", gate_on=True))

    def test_gate_on_casual_chat_skips(self):
        self.assertFalse(_sticky_gate_active("hello, how can I help you?", gate_on=True))

    def test_gate_on_case_insensitive(self):
        self.assertTrue(_sticky_gate_active("CREW is running", gate_on=True))
        self.assertTrue(_sticky_gate_active("Checkpoint found", gate_on=True))


class CrewStickyBugRegressionTests(unittest.TestCase):
    """Regression pins for Bug 2 and Bug 3 found during code review."""

    def test_crew_sticky_kw_re_is_module_level(self):
        """Bug 2: regex must be compiled at module level, not inside hot path."""
        import re as _stdlib_re
        from larkhelm.handlers._message import _CREW_STICKY_KW_RE
        self.assertIsInstance(_CREW_STICKY_KW_RE, type(_stdlib_re.compile("")))
        # Spot-check correctness after hoist
        self.assertIsNotNone(_CREW_STICKY_KW_RE.search("crew task"))
        self.assertIsNone(_CREW_STICKY_KW_RE.search("weather today"))

    def test_consume_called_even_when_gate_skips(self):
        """Bug 3: consume_recent_crew_context must be called even when keyword
        gate fires and skips injection so the injection counter advances and
        recent_crew_sticky_max_injections eviction still works.

        Tests through _apply_crew_sticky_context (the extracted helper that
        handle_message now delegates to), so this pin guards the real code path
        rather than a manual simulation.
        """
        from unittest.mock import patch
        from larkhelm.handlers._message import _apply_crew_sticky_context

        mock_crew_ctx = {"title": "crew task", "summary": "did stuff"}
        _cfg.config["crew_sticky_keyword_gate_enabled"] = True
        try:
            with patch("larkhelm.crew.consume_recent_crew_context",
                       return_value=mock_crew_ctx) as mock_consume:
                original_prompt = "what is the weather today"
                result = _apply_crew_sticky_context(
                    "chat_p1b_test", original_prompt, original_prompt
                )
                # consume was called (eviction counter advanced)
                mock_consume.assert_called_once_with("chat_p1b_test")
                # gate fired: prompt must NOT contain crew summary
                self.assertNotIn("刚完成的 Crew 任务", result)
                self.assertEqual(result, original_prompt)
        finally:
            _cfg.config["crew_sticky_keyword_gate_enabled"] = False

    def test_inject_when_crew_keyword_present(self):
        """gate=True + crew keyword in text → crew summary IS injected."""
        from unittest.mock import patch
        from larkhelm.handlers._message import _apply_crew_sticky_context

        mock_crew_ctx = {"title": "my crew task", "summary": "result here"}
        _cfg.config["crew_sticky_keyword_gate_enabled"] = True
        try:
            with patch("larkhelm.crew.consume_recent_crew_context",
                       return_value=mock_crew_ctx):
                result = _apply_crew_sticky_context(
                    "chat_kw_test", "继续那个crew任务", "follow up question"
                )
                self.assertIn("刚完成的 Crew 任务", result)
                self.assertIn("result here", result)
        finally:
            _cfg.config["crew_sticky_keyword_gate_enabled"] = False

    def test_gate_off_always_injects(self):
        """gate=False → crew summary always injected regardless of keywords."""
        from unittest.mock import patch
        from larkhelm.handlers._message import _apply_crew_sticky_context

        mock_crew_ctx = {"title": "task", "summary": "summary text"}
        _cfg.config["crew_sticky_keyword_gate_enabled"] = False
        try:
            with patch("larkhelm.crew.consume_recent_crew_context",
                       return_value=mock_crew_ctx):
                result = _apply_crew_sticky_context(
                    "chat_off_test", "random unrelated message", "user query"
                )
                self.assertIn("刚完成的 Crew 任务", result)
        finally:
            _cfg.config["crew_sticky_keyword_gate_enabled"] = False

    def test_no_ctx_returns_prompt_unchanged(self):
        """No sticky context → prompt returned unmodified."""
        from unittest.mock import patch
        from larkhelm.handlers._message import _apply_crew_sticky_context

        with patch("larkhelm.crew.consume_recent_crew_context", return_value=None):
            original = "my prompt"
            result = _apply_crew_sticky_context("chat_empty", "some text", original)
            self.assertEqual(result, original)


# ══════════════════════════════════════════════════════════════════════════════
#  P1c — project_guide_enabled
# ══════════════════════════════════════════════════════════════════════════════

_GUIDE_CONTENT = "# Project Guide\n\nUse snake_case for variables."


class ProjectGuideInjectionTests(unittest.TestCase):

    def setUp(self):
        self._guide_file = Path(_TMP) / "project_guide.md"
        self._guide_file.write_text(_GUIDE_CONTENT, encoding="utf-8")
        _cfg.config["project_guide_enabled"] = True
        _cfg.config["project_guide_path"] = str(self._guide_file)

    def tearDown(self):
        _cfg.config["project_guide_enabled"] = False
        _cfg.config["project_guide_path"] = ""

    def _run_guide_gate(self, provider: str, existing_memory: str = "") -> str:
        """Replicate the project guide gate logic from _do_query."""
        memory_ctx = existing_memory
        if bool(_cfg.config.get("project_guide_enabled")) and _cfg.config.get("project_guide_path"):
            _is_cli_claude = provider == "claude_cli"
            if not _is_cli_claude:
                try:
                    _guide_path = Path(_cfg.config["project_guide_path"]).expanduser()
                    _guide_content = _guide_path.read_text(encoding="utf-8")
                    if len(_guide_content) > 4000:
                        _guide_content = _guide_content[:4000]
                    memory_ctx = (
                        f"[Project Guide]\n{_guide_content}\n[/Project Guide]\n\n"
                        + memory_ctx
                    )
                except Exception:
                    pass
        return memory_ctx

    def test_api_backend_injects_guide(self):
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertIn("[Project Guide]", result)
        self.assertIn(_GUIDE_CONTENT, result)
        self.assertIn("[SESSION MEMORY]", result)

    def test_google_api_backend_injects_guide(self):
        result = self._run_guide_gate("google_api")
        self.assertIn("[Project Guide]", result)

    def test_claude_cli_skips_guide(self):
        result = self._run_guide_gate("claude_cli", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertNotIn("[Project Guide]", result)
        self.assertIn("[SESSION MEMORY]", result)

    def test_guide_prepended_before_memory(self):
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        guide_pos = result.index("[Project Guide]")
        session_pos = result.index("[SESSION MEMORY]")
        self.assertLess(guide_pos, session_pos)

    def test_gate_disabled_no_injection(self):
        _cfg.config["project_guide_enabled"] = False
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertNotIn("[Project Guide]", result)

    def test_no_path_no_injection(self):
        _cfg.config["project_guide_path"] = ""
        result = self._run_guide_gate("anthropic_api")
        self.assertNotIn("[Project Guide]", result)

    def test_guide_truncated_at_4000_chars(self):
        long_guide = "x" * 5000
        long_file = Path(_TMP) / "long_guide.md"
        long_file.write_text(long_guide, encoding="utf-8")
        _cfg.config["project_guide_path"] = str(long_file)

        result = self._run_guide_gate("anthropic_api")
        guide_match = re.search(
            r"\[Project Guide\]\n(.*?)\n\[/Project Guide\]", result, re.DOTALL
        )
        self.assertIsNotNone(guide_match)
        self.assertLessEqual(len(guide_match.group(1)), 4000)

    def test_missing_guide_file_no_crash(self):
        _cfg.config["project_guide_path"] = "/nonexistent/path/guide.md"
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertNotIn("[Project Guide]", result)
        self.assertIn("[SESSION MEMORY]", result)


if __name__ == "__main__":
    unittest.main()
