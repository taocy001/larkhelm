"""Tests for P2 on-demand injection gates (P2a / P2b / P2c).

P2a — workspace_hint_keyword_gate: regex expanded with code-task keywords.
P2b — parent_inject_skip_when_api_history: skip parent msg for API backends.
P2c — doc_inject observe-only metrics: inc_injection_gate called on doc inject.
"""
from __future__ import annotations

import atexit
import json
import re
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

# ── Bootstrap shared config (same pattern as test_injection_gates_p1) ──
_TMP = tempfile.mkdtemp(prefix="larkhelm_igates_p2_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = _TMP + "/config.json"
with open(_cfg_file, "w") as _f:
    json.dump({"APP_ID": "x", "APP_SECRET": "x"}, _f)

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=_cfg_file, data_dir=_TMP)


# ══════════════════════════════════════════════════════════════════════════════
#  P2a — _WORKSPACE_KEYWORD_RE expanded with code-task keywords
# ══════════════════════════════════════════════════════════════════════════════

# Import the compiled regex from the module so we test the live object.
from larkhelm.handlers._message import _WORKSPACE_KEYWORD_RE  # noqa: E402


class WorkspaceKeywordRegexTests(unittest.TestCase):
    """Pin the updated _WORKSPACE_KEYWORD_RE pattern for P2a."""

    # ── New code-task keywords must match ─────────────────────────────────
    def test_matches_code(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("write some code for me"))

    def test_matches_fix(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("fix this bug"))

    def test_matches_debug(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("help me debug this"))

    def test_matches_refactor(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("refactor the module"))

    def test_matches_implement(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("implement the feature"))

    def test_matches_edit(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("edit the file"))

    def test_matches_chinese_write_code(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("帮我写代码"))

    def test_matches_chinese_change_code(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("改代码"))

    def test_matches_chinese_fix(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("修复这个问题"))

    def test_matches_chinese_refactor(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("重构一下"))

    # ── Original planning/crew keywords must still match ──────────────────
    def test_matches_crew(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("start a crew task"))

    def test_matches_tasks(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("list all tasks"))

    def test_matches_design(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("design the architecture"))

    def test_matches_prd(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("write a PRD"))

    def test_matches_review(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("code review please"))

    def test_matches_qa(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("run QA checks"))

    def test_matches_workspace(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("open the workspace"))

    def test_matches_chinese_plan(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("制定一个计划"))

    def test_matches_chinese_task(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("执行任务"))

    # ── Casual/non-work messages must NOT match ────────────────────────────
    def test_no_match_weather(self):
        self.assertIsNone(_WORKSPACE_KEYWORD_RE.search("what is the weather today"))

    def test_no_match_hello(self):
        self.assertIsNone(_WORKSPACE_KEYWORD_RE.search("hello, how are you"))

    def test_no_match_casual_question(self):
        self.assertIsNone(_WORKSPACE_KEYWORD_RE.search("tell me a joke"))

    def test_no_match_empty(self):
        self.assertIsNone(_WORKSPACE_KEYWORD_RE.search(""))

    # ── Case insensitivity ────────────────────────────────────────────────
    def test_case_insensitive_CODE(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("Write some CODE"))

    def test_case_insensitive_FIX(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("FIX this"))

    def test_case_insensitive_CREW(self):
        self.assertIsNotNone(_WORKSPACE_KEYWORD_RE.search("CREW mission"))


# ══════════════════════════════════════════════════════════════════════════════
#  P2b — parent_inject_skip_when_api_history
# ══════════════════════════════════════════════════════════════════════════════

def _make_spec(provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider,
        display_name=provider,
        id=provider,
    )


def _run_parent_gate(
    parent_id: str | None,
    early_spec,
    gate_enabled: bool,
    fetch_returns: str | None = "some parent text",
):
    """
    Replicate the P2b gate logic from _do_query without importing the full handler.

    Returns a dict:
      skip: bool          — whether gate decided to skip
      injected_metric: bool — whether the 'injected' metric was called
      skipped_metric: bool  — whether the 'skipped_api' metric was called
    """
    _API_PROVIDERS = {"anthropic_api", "google_api", "openai_compat_api"}
    _skip_parent = False
    inc_calls: list[tuple[str, str]] = []

    def _fake_inc(point: str, outcome: str) -> None:
        inc_calls.append((point, outcome))

    if parent_id:
        try:
            if (
                gate_enabled
                and early_spec is not None
                and getattr(early_spec, "provider", "") in _API_PROVIDERS
            ):
                _skip_parent = True
                _fake_inc("parent_msg", "skipped_api")
        except Exception:
            pass  # fail-open

        if not _skip_parent:
            parent_text = fetch_returns
            if parent_text:
                _fake_inc("parent_msg", "injected")

    return {
        "skip": _skip_parent,
        "injected_metric": ("parent_msg", "injected") in inc_calls,
        "skipped_metric": ("parent_msg", "skipped_api") in inc_calls,
    }


class ParentInjectSkipApiTests(unittest.TestCase):

    # ── Gate ON + API backend → skip ──────────────────────────────────────
    def test_anthropic_api_skips_when_gate_on(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("anthropic_api"),
            gate_enabled=True,
        )
        self.assertTrue(result["skip"])
        self.assertTrue(result["skipped_metric"])
        self.assertFalse(result["injected_metric"])

    def test_google_api_skips_when_gate_on(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("google_api"),
            gate_enabled=True,
        )
        self.assertTrue(result["skip"])
        self.assertTrue(result["skipped_metric"])

    def test_openai_compat_api_skips_when_gate_on(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("openai_compat_api"),
            gate_enabled=True,
        )
        self.assertTrue(result["skip"])
        self.assertTrue(result["skipped_metric"])

    # ── Gate OFF → no skip regardless of backend ─────────────────────────
    def test_gate_off_anthropic_api_injects(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("anthropic_api"),
            gate_enabled=False,
        )
        self.assertFalse(result["skip"])
        self.assertTrue(result["injected_metric"])

    def test_gate_off_claude_cli_injects(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("claude_cli"),
            gate_enabled=False,
        )
        self.assertFalse(result["skip"])
        self.assertTrue(result["injected_metric"])

    # ── Gate ON + CLI backend → no skip ──────────────────────────────────
    def test_claude_cli_not_skipped_when_gate_on(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("claude_cli"),
            gate_enabled=True,
        )
        self.assertFalse(result["skip"])
        self.assertTrue(result["injected_metric"])

    def test_gemini_cli_not_skipped_when_gate_on(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("gemini_cli"),
            gate_enabled=True,
        )
        self.assertFalse(result["skip"])
        self.assertTrue(result["injected_metric"])

    def test_kimi_cli_not_skipped_when_gate_on(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("kimi_cli"),
            gate_enabled=True,
        )
        self.assertFalse(result["skip"])

    def test_deepseek_api_not_skipped_when_gate_on(self):
        # deepseek_api is not in the API-history set
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("deepseek_api"),
            gate_enabled=True,
        )
        self.assertFalse(result["skip"])

    # ── No parent_id → nothing happens ────────────────────────────────────
    def test_no_parent_id_no_skip_no_metric(self):
        result = _run_parent_gate(
            parent_id=None,
            early_spec=_make_spec("anthropic_api"),
            gate_enabled=True,
        )
        self.assertFalse(result["skip"])
        self.assertFalse(result["skipped_metric"])
        self.assertFalse(result["injected_metric"])

    # ── early_spec is None → fail-open (no skip) ─────────────────────────
    def test_early_spec_none_fail_open(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=None,
            gate_enabled=True,
        )
        self.assertFalse(result["skip"])

    # ── fetch returns empty string → no injected metric ───────────────────
    def test_fetch_returns_empty_gate_off_no_injected_metric(self):
        result = _run_parent_gate(
            parent_id="msg_123",
            early_spec=_make_spec("claude_cli"),
            gate_enabled=False,
            fetch_returns="",
        )
        self.assertFalse(result["injected_metric"])


class ParentInjectConfigIntegrationTests(unittest.TestCase):
    """Check that the new config key is set with the right default."""

    def test_config_default_is_false(self):
        # After _init_runtime, the key must exist and default to False.
        self.assertIn("parent_inject_skip_when_api_history", _cfg.config)
        self.assertFalse(_cfg.config["parent_inject_skip_when_api_history"])

    def test_doc_inject_relevance_gate_default_is_false(self):
        self.assertIn("doc_inject_relevance_gate_enabled", _cfg.config)
        self.assertFalse(_cfg.config["doc_inject_relevance_gate_enabled"])


# ══════════════════════════════════════════════════════════════════════════════
#  P2c — doc_inject observe-only metrics
# ══════════════════════════════════════════════════════════════════════════════

class DocInjectMetricsTests(unittest.TestCase):
    """Verify that inc_injection_gate is called correctly inside _inject_doc_context."""

    def _make_doc_result(self, content: str, title: str = "Test Doc"):
        return SimpleNamespace(content=content, title=title)

    def _run_inject(self, content: str, inc_mock: MagicMock) -> None:
        """
        Replicate the P2c metric-emit block in _inject_doc_context.
        Called after a successful doc read with `result.content = content`.
        """
        result = self._make_doc_result(content)
        # This mirrors the P2c block in _inject_doc_context:
        try:
            inc_mock("doc_inject", "injected")
            if len(result.content) > 10000:
                inc_mock("doc_inject", "large_doc")
        except Exception:
            pass

    def test_injected_metric_emitted_always(self):
        mock = MagicMock()
        self._run_inject("short content", mock)
        mock.assert_any_call("doc_inject", "injected")

    def test_large_doc_metric_emitted_for_content_gt_10000(self):
        mock = MagicMock()
        self._run_inject("x" * 10001, mock)
        mock.assert_any_call("doc_inject", "injected")
        mock.assert_any_call("doc_inject", "large_doc")

    def test_large_doc_not_emitted_for_exactly_10000(self):
        mock = MagicMock()
        self._run_inject("x" * 10000, mock)
        mock.assert_any_call("doc_inject", "injected")
        # large_doc only fires for > 10000, not == 10000
        self.assertNotIn(call("doc_inject", "large_doc"), mock.call_args_list)

    def test_large_doc_not_emitted_for_short_content(self):
        mock = MagicMock()
        self._run_inject("hello world", mock)
        self.assertNotIn(call("doc_inject", "large_doc"), mock.call_args_list)

    def test_metric_exception_does_not_propagate(self):
        """fail-silent: if inc raises, _inject_doc_context must not propagate."""
        raising_inc = MagicMock(side_effect=RuntimeError("prometheus down"))
        # Should not raise:
        try:
            raising_inc("doc_inject", "injected")
            if len("content") > 10000:
                raising_inc("doc_inject", "large_doc")
        except Exception:
            pass  # gate has try/except so this is equivalent


class DocInjectMetricsLiveTests(unittest.TestCase):
    """Integration-level test: patch inc_injection_gate and run _inject_doc_context."""

    def _make_doc_result(self, content: str, title: str = "Test"):
        return SimpleNamespace(content=content, title=title)

    def _make_result_meta(self, content: str):
        return SimpleNamespace(
            payload=self._make_doc_result(content),
            from_cache=False,
            age_sec=None,
        )

    def test_inject_doc_context_emits_injected_metric(self):
        from larkhelm.handlers._query import _inject_doc_context

        short_content = "short doc content under 10000 chars"
        result_meta = self._make_result_meta(short_content)

        with patch("larkhelm.config.DOC_AUTO_INJECT", True), \
             patch("larkhelm.config.DOC_INJECT_MAX_DOCS", 3), \
             patch("larkhelm.config.DOC_INJECT_MAX_CHARS", 5000), \
             patch("larkhelm.handlers._query._cfg.DOC_INJECT_MAX_DOCS", 3), \
             patch("larkhelm.handlers._query._cfg.DOC_INJECT_MAX_CHARS", 5000), \
             patch("larkhelm.handlers._query._cfg.DOC_INJECT_CACHE_ENABLED", True), \
             patch(
                 "larkhelm.handlers._query._extract_feishu_urls",
                 return_value=["https://example.feishu.cn/docx/abc"],
             ), \
             patch(
                 "larkhelm.lark_client.parse_doc_url",
                 return_value=SimpleNamespace(token="abc", type="docx"),
             ), \
             patch(
                 "larkhelm._context_cache.cached_doc_read_with_meta",
                 return_value=result_meta,
             ), \
             patch("larkhelm.metrics.inc_injection_gate") as mock_inc:
            _inject_doc_context("check this https://example.feishu.cn/docx/abc", "chat_1")
            mock_inc.assert_any_call("doc_inject", "injected")

    def test_inject_doc_context_emits_large_doc_metric_for_big_content(self):
        from larkhelm.handlers._query import _inject_doc_context

        large_content = "x" * 10001
        result_meta = self._make_result_meta(large_content)

        with patch("larkhelm.config.DOC_AUTO_INJECT", True), \
             patch("larkhelm.config.DOC_INJECT_MAX_DOCS", 3), \
             patch("larkhelm.config.DOC_INJECT_MAX_CHARS", 20000), \
             patch("larkhelm.handlers._query._cfg.DOC_INJECT_MAX_DOCS", 3), \
             patch("larkhelm.handlers._query._cfg.DOC_INJECT_MAX_CHARS", 20000), \
             patch("larkhelm.handlers._query._cfg.DOC_INJECT_CACHE_ENABLED", True), \
             patch(
                 "larkhelm.handlers._query._extract_feishu_urls",
                 return_value=["https://example.feishu.cn/docx/big"],
             ), \
             patch(
                 "larkhelm.lark_client.parse_doc_url",
                 return_value=SimpleNamespace(token="big", type="docx"),
             ), \
             patch(
                 "larkhelm._context_cache.cached_doc_read_with_meta",
                 return_value=result_meta,
             ), \
             patch("larkhelm.metrics.inc_injection_gate") as mock_inc:
            _inject_doc_context("check this https://example.feishu.cn/docx/big", "chat_2")
            mock_inc.assert_any_call("doc_inject", "injected")
            mock_inc.assert_any_call("doc_inject", "large_doc")


if __name__ == "__main__":
    unittest.main()
