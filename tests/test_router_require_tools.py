"""Bug 2 — chat-path ``resolve_backend`` must reject tool-incapable
backends when the query implicitly needs tools (Approach C).

Pre-fix: ``router.resolve_backend`` Rule 3 cheap-routing picked
``get_by_tag(["cheap", "fast"])`` for any short message. DeepSeek
(``tags=["cheap", "fast"]``, no ``"tools"``) would win and produce
confident wrong answers about files it never read.

Post-fix: the router computes ``require_tools = has_doc_urls or
_likely_needs_tools(message)`` and skips cheap routing entirely when
True. User-preference and default_backend fall-throughs also check
the ``"tools"`` tag before returning a tool-incapable backend.

This file pins the new behaviour with both positive and negative
cases. It complements ``test_router_task_aware.py`` which covers
the crew-side rank_for_task path.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="larkhelm_router_tools_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.backend_registry import BackendRegistry, BackendSpec
from larkhelm.router import _likely_needs_tools, resolve_backend


# ─────────────────────────────────────────────────────────────────────────
# 1. _likely_needs_tools heuristic
# ─────────────────────────────────────────────────────────────────────────


class LikelyNeedsToolsTests(unittest.TestCase):
    """Detector must catch the patterns that DID land in DeepSeek's
    inbox in production, AND must NOT fire on plain chat queries."""

    POSITIVE = [
        # File extensions
        "看一下 cmd_doc.py",
        "test_memory.py 跑一下",
        "fix the bug in config.json",
        "@bot 看看 README.md 第二章",
        # Filesystem paths
        "grep foo /etc/larkhelm",
        "cat ~/.config/larkhelm/config.json",
        # Commands
        "grep -n bar in repo",
        "pytest tests/ 跑下",
        "git log --oneline -5",
        "ls /tmp",
        "find . -name '*.py'",
        "head -20 README.md",
        "systemctl status larkhelm",
        # Code fence
        "怎么看下面这段代码\n```python\nprint(1)\n```",
    ]

    NEGATIVE = [
        # Pure chat
        "今天几号",
        "把这段中文翻译成英文",
        "你好",
        "讲个笑话",
        "我喜欢用 PascalCase 命名",
        "明天上午开会",
        # 边缘但合理：用户在闲聊"使用 Python"——不引用具体文件
        "推荐一个 Python 学习路径",
        # Empty input
        "",
        "   \t  ",
    ]

    def test_positive_signals(self):
        for msg in self.POSITIVE:
            with self.subTest(msg=msg):
                self.assertTrue(_likely_needs_tools(msg),
                                f"should detect tools-need: {msg!r}")

    def test_negative_signals(self):
        for msg in self.NEGATIVE:
            with self.subTest(msg=msg):
                self.assertFalse(_likely_needs_tools(msg),
                                 f"should NOT flag plain chat: {msg!r}")

    def test_case_insensitive_command_detection(self):
        # "GREP" and "Grep" should also trigger.
        self.assertTrue(_likely_needs_tools("GREP foo bar"))
        self.assertTrue(_likely_needs_tools("Run Pytest please"))


# ─────────────────────────────────────────────────────────────────────────
# 2. resolve_backend end-to-end with require_tools enforced
# ─────────────────────────────────────────────────────────────────────────


def _make_registry() -> BackendRegistry:
    """A registry with three specs:
      * deepseek (cheap, fast, no tools) — provider="deepseek_api"
      * claude   (vision, tools, orchestrator)
      * kimi     (vision, tools, worker)
    Mirrors the default config.py layout.
    """
    reg = BackendRegistry()
    specs = [
        BackendSpec(
            id="deepseek-chat", provider="deepseek_api",
            display_name="DeepSeek", role="worker",
            tags=["cheap", "fast"], api_key="x", base_url="https://x", model="deepseek-chat",
            healthy=True, enabled=True,
        ),
        BackendSpec(
            id="claude", provider="claude_cli", display_name="Claude",
            role="orchestrator", tags=["vision", "tools"], command="claude",
            healthy=True, enabled=True,
        ),
        BackendSpec(
            id="kimi", provider="kimi_cli", display_name="Kimi",
            role="worker", tags=["vision", "tools"], command="kimi",
            healthy=True, enabled=True,
        ),
    ]
    with reg._lock:
        for s in specs:
            reg._specs[s.id] = s
    return reg


class ResolveBackendRequireToolsTests(unittest.TestCase):
    """The bug: short queries needing tools routed to DeepSeek. The
    fix: router refuses to give those queries to a non-tools backend."""

    def setUp(self):
        # Enable cheap routing — without it, Rule 3 is dormant and the
        # bug couldn't have triggered. We want to verify the fix even
        # when cheap routing is ON.
        self._patches = [
            patch.dict(_cfg.config, {"enable_cheap_routing": True}, clear=False),
            patch("larkhelm.router.BACKEND_REGISTRY", _make_registry()),
        ]
        for p in self._patches:
            p.start()
        for p in self._patches:
            self.addCleanup(p.stop)

    def test_short_tools_query_does_NOT_go_to_deepseek(self):
        """Reproduces the original production bug: '看一下 cmd_doc.py'
        is < 100 chars and would have triggered Rule 3 cheap routing."""
        spec = resolve_backend(
            chat_id="oc_test",
            message="看一下 cmd_doc.py",
            has_images=False, has_doc_urls=False,
        )
        self.assertNotEqual(spec.id, "deepseek-chat",
                            "cheap routing must skip DeepSeek when query implies tools")
        self.assertIn("tools", spec.tags,
                      "selected backend must have tools tag for tools-implying query")

    def test_short_plain_chat_still_goes_to_deepseek(self):
        """Don't over-correct: pure chat queries SHOULD still be eligible
        for cheap routing (this is the whole point of cheap routing)."""
        spec = resolve_backend(
            chat_id="oc_test",
            message="今天几号",
            has_images=False, has_doc_urls=False,
        )
        self.assertEqual(spec.id, "deepseek-chat",
                         "pure chat should still go to cheap backend when enabled")

    def test_doc_url_query_avoids_deepseek_via_rule_2(self):
        """has_doc_urls=True triggers Rule 2 (tools-tagged). Already
        in pre-fix behaviour but pin it under the new require_tools
        bookkeeping so a future refactor can't accidentally bypass
        Rule 2 for has_doc_urls."""
        spec = resolve_backend(
            chat_id="oc_test",
            message="总结一下",  # short, no file path
            has_images=False, has_doc_urls=True,
        )
        self.assertIn("tools", spec.tags)
        self.assertNotEqual(spec.id, "deepseek-chat")

    def test_user_preference_deepseek_falls_through_when_query_needs_tools(self):
        """User pinned /model deepseek but then asks 'grep foo' — the
        pin must be IGNORED (falling through to default/orchestrator)
        rather than handing the tools-query to DeepSeek anyway."""
        # Simulate user-preference via the chat_state.
        from larkhelm.chat_state import _set_chat_field
        _set_chat_field("oc_pref_test", "backend_id", "deepseek-chat")

        spec = resolve_backend(
            chat_id="oc_pref_test",
            message="grep -n foo larkhelm/",
            has_images=False, has_doc_urls=False,
        )
        self.assertNotEqual(spec.id, "deepseek-chat",
                            "user pref to non-tools backend must fall through for tools queries")
        self.assertIn("tools", spec.tags)

    def test_long_message_with_tools_signal_also_filters(self):
        """The require_tools gate isn't just for Rule 3 (short) — it's
        global. A long code-flavoured message must also not land on
        DeepSeek via fall-through (Rule 5 default_backend etc)."""
        # Drop user-pref + force fall-through to default_backend.
        long_msg = (
            "I'd like to refactor cmd_doc.py to split the read/write "
            "paths cleanly. Could you look at the existing structure "
            "and propose a refactor plan? Don't make changes yet."
        )
        self.assertGreater(len(long_msg), 100)  # past short threshold
        with patch.dict(_cfg.config,
                        {"default_backend": "deepseek-chat"}, clear=False):
            spec = resolve_backend(
                chat_id="oc_long_test",
                message=long_msg,
                has_images=False, has_doc_urls=False,
            )
        # Default_backend pointed at deepseek but query needs tools →
        # router falls through to orchestrator chain.
        self.assertNotEqual(spec.id, "deepseek-chat")
        self.assertIn("tools", spec.tags)


if __name__ == "__main__":
    unittest.main()
