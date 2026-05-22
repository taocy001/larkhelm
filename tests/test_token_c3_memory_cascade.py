"""tests/test_token_c3_memory_cascade.py — TOKEN-C3 integration tests.

Covers:
  AC-03  memory._try_extract_project uses f"{chat_id}__memory_project" (no _proj_ fallback)
  AC-04  BaseProcessRunner._record_tokens strips __memory_ prefix via resolve_record_chat_id
  AC-05  backend_api_streaming._run_streaming_api strips __memory_ prefix
  AC-06  resolve_record_chat_id decision tree
  AC-07  get_token_stats_persistent aggregates memory-cascade tokens under orig chat_id
"""
from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── AC-06: resolve_record_chat_id unit tests ─────────────────────────────────


def test_resolve_no_prefix():
    from larkhelm.token_stats import resolve_record_chat_id
    assert resolve_record_chat_id("chat123") == "chat123"


def test_resolve_record_under_wins():
    from larkhelm.token_stats import resolve_record_chat_id
    assert resolve_record_chat_id("chat123__memory_session", "explicit") == "explicit"


def test_resolve_crew_prefix():
    from larkhelm.token_stats import resolve_record_chat_id
    assert resolve_record_chat_id("abc__crew_agent1") == "abc"


def test_resolve_memory_session():
    from larkhelm.token_stats import resolve_record_chat_id
    assert resolve_record_chat_id("abc__memory_session") == "abc"


def test_resolve_memory_project():
    from larkhelm.token_stats import resolve_record_chat_id
    assert resolve_record_chat_id("abc__memory_project") == "abc"


def test_resolve_memory_global():
    from larkhelm.token_stats import resolve_record_chat_id
    assert resolve_record_chat_id("abc__memory_global") == "abc"


def test_resolve_crew_takes_precedence_over_memory():
    # A namespace that somehow contains both prefixes → crew wins (first check)
    from larkhelm.token_stats import resolve_record_chat_id
    ns = "abc__crew_x__memory_session"
    assert resolve_record_chat_id(ns) == "abc"


def test_resolve_record_under_none_still_strips():
    from larkhelm.token_stats import resolve_record_chat_id
    assert resolve_record_chat_id("abc__memory_global", None) == "abc"


# ── AC-03: memory._try_extract_project namespace format ──────────────────────


def test_extract_project_ns_format():
    """Verify that _try_extract_project always builds ns as {chat_id}__memory_project."""
    captured_ns: list[str] = []

    def fake_run_one_shot_with_backoff(prompt, ns, cancel_ev=None):
        captured_ns.append(ns)
        return "UNCHANGED"

    with patch("larkhelm.memory._run_one_shot_with_backoff", side_effect=fake_run_one_shot_with_backoff):
        with patch("larkhelm.memory.load_project_memory", return_value="(empty)"):
            with patch("larkhelm.memory._project_memory_file", return_value=MagicMock(exists=lambda: False)):
                with patch("larkhelm.memory._load_md_frontmatter", return_value={}):
                    from larkhelm.memory import _try_extract_project
                    _try_extract_project("some session content", "/some/cwd", "chatXYZ")

    assert len(captured_ns) == 1
    assert captured_ns[0] == "chatXYZ__memory_project"


def test_extract_project_ns_no_proj_fallback():
    """Even when chat_id is empty, no _proj_ fallback is used."""
    captured_ns: list[str] = []

    def fake_run_one_shot_with_backoff(prompt, ns, cancel_ev=None):
        captured_ns.append(ns)
        return "UNCHANGED"

    with patch("larkhelm.memory._run_one_shot_with_backoff", side_effect=fake_run_one_shot_with_backoff):
        with patch("larkhelm.memory.load_project_memory", return_value="(empty)"):
            with patch("larkhelm.memory._project_memory_file", return_value=MagicMock(exists=lambda: False)):
                with patch("larkhelm.memory._load_md_frontmatter", return_value={}):
                    from larkhelm.memory import _try_extract_project
                    _try_extract_project("some session content", "/some/cwd", "")

    assert len(captured_ns) == 1
    # Must be the unified format, never "_proj_*"
    assert not captured_ns[0].startswith("_proj_")
    assert captured_ns[0] == "__memory_project"


# ── AC-04: BaseProcessRunner._record_tokens strips __memory_ ─────────────────


def test_runner_base_record_tokens_strips_memory_prefix(tmp_path):
    """_record_tokens on a memory-ns chat_id records under the original chat_id."""
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path
    _cfg.DATA_DIR = tmp_path

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        # Import after patching so the module-level import chain is clean.
        from larkhelm.runner_base import BaseProcessRunner

        class _FakeRunner(BaseProcessRunner):
            def build_args(self): return []
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return True
            def cleanup_extra(self): pass

        runner = _FakeRunner(
            backend_name="test",
            chat_id="origchat__memory_session",
            message="hello",
            sid=None,
            cwd=str(tmp_path),
        )
        runner._record_tokens("claude", {"input_tokens": 10, "output_tokens": 5}, 0.01)

    assert len(recorded) == 1
    assert recorded[0][0] == "origchat", f"Expected 'origchat', got {recorded[0][0]!r}"


def test_runner_base_record_tokens_crew_still_works(tmp_path):
    """__crew_ prefix still strips correctly after refactor."""
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path
    _cfg.DATA_DIR = tmp_path

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        from larkhelm.runner_base import BaseProcessRunner

        class _FakeRunner(BaseProcessRunner):
            def build_args(self): return []
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return True
            def cleanup_extra(self): pass

        runner = _FakeRunner(
            backend_name="test",
            chat_id="origchat__crew_agent42",
            message="hello",
            sid=None,
            cwd=str(tmp_path),
        )
        runner._record_tokens("claude", {"input_tokens": 10, "output_tokens": 5}, 0.01)

    assert len(recorded) == 1
    assert recorded[0][0] == "origchat"


def test_runner_base_record_under_overrides(tmp_path):
    """record_under takes highest priority over all prefix stripping."""
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path
    _cfg.DATA_DIR = tmp_path

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        from larkhelm.runner_base import BaseProcessRunner

        class _FakeRunner(BaseProcessRunner):
            def build_args(self): return []
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return True
            def cleanup_extra(self): pass

        runner = _FakeRunner(
            backend_name="test",
            chat_id="origchat__memory_global",
            message="hello",
            sid=None,
            cwd=str(tmp_path),
            record_under="explicit_target",
        )
        runner._record_tokens("claude", {"input_tokens": 1, "output_tokens": 1}, 0.0)

    assert recorded[0][0] == "explicit_target"


# ── DeepSeekRunner._record_tokens ─────────────────────────────────────────────


def test_deepseek_runner_record_tokens_strips_memory_prefix(tmp_path):
    """DeepSeekRunner._record_tokens also uses resolve_record_chat_id."""
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path
    _cfg.DATA_DIR = tmp_path
    _cfg.DEEPSEEK_API_KEY = "testkey"
    _cfg.DEEPSEEK_MODEL = "deepseek-chat"
    _cfg.DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        from larkhelm.runner_deepseek import DeepSeekRunner
        runner = DeepSeekRunner(
            chat_id="origchat__memory_project",
            message="hello",
            sid=None,
            cwd=str(tmp_path),
        )
        runner._record_tokens({"input_tokens": 7, "output_tokens": 3}, cost=0.0)

    assert len(recorded) == 1
    assert recorded[0][0] == "origchat"


def test_deepseek_runner_record_tokens_crew(tmp_path):
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path
    _cfg.DATA_DIR = tmp_path
    _cfg.DEEPSEEK_API_KEY = "testkey"
    _cfg.DEEPSEEK_MODEL = "deepseek-chat"
    _cfg.DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        from larkhelm.runner_deepseek import DeepSeekRunner
        runner = DeepSeekRunner(
            chat_id="origchat__crew_planner",
            message="hello",
            sid=None,
            cwd=str(tmp_path),
        )
        runner._record_tokens({"input_tokens": 7, "output_tokens": 3}, cost=0.0)

    assert len(recorded) == 1
    assert recorded[0][0] == "origchat"


def test_deepseek_runner_record_under_overrides(tmp_path):
    """DeepSeekRunner.record_under takes highest priority over prefix stripping."""
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path
    _cfg.DATA_DIR = tmp_path
    _cfg.DEEPSEEK_API_KEY = "testkey"
    _cfg.DEEPSEEK_MODEL = "deepseek-chat"
    _cfg.DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        from larkhelm.runner_deepseek import DeepSeekRunner
        runner = DeepSeekRunner(
            chat_id="origchat__memory_session",
            message="hello",
            sid=None,
            cwd=str(tmp_path),
            record_under="explicit_target",
        )
        runner._record_tokens({"input_tokens": 7, "output_tokens": 3}, cost=0.0)

    assert len(recorded) == 1
    assert recorded[0][0] == "explicit_target"


# ── AC-05: backend_api_streaming._run_streaming_api strips __memory_ ─────────


def test_streaming_api_strips_memory_prefix():
    """_run_streaming_api records tokens under the orig chat_id, not the memory ns."""
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    # Build a minimal stub adapter that returns a single chunk and usage.
    class _StubAdapter:
        provider_label = "anthropic_api"

        def build_client(self, spec):
            return MagicMock()

        def prepare_request(self, spec, history, message, extra_system):
            return {}

        def iter_text_chunks(self, client, request):
            yield "hello"

        def extract_usage(self):
            return {"input_tokens": 5, "output_tokens": 2, "cache_read": 0, "cache_create": 0}

        def format_history(self, history, message, response_text):
            return []

    spec = MagicMock()
    spec.id = "test-spec"
    spec.model = "claude-test"

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        with patch("larkhelm.backend_api_streaming.BACKEND_REGISTRY"):
            from larkhelm.backend_api_streaming import _run_streaming_api
            _run_streaming_api(
                adapter=_StubAdapter(),
                spec=spec,
                chat_id="origchat__memory_session",
                message="hi",
                history=[],
            )

    assert len(recorded) == 1, f"Expected 1 record, got {len(recorded)}: {recorded}"
    assert recorded[0][0] == "origchat", f"Expected 'origchat', got {recorded[0][0]!r}"


def test_streaming_api_plain_chat_id_unchanged():
    """Plain chat_id (no prefix) passes through unchanged."""
    recorded: list[tuple] = []

    def fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    class _StubAdapter:
        provider_label = "anthropic_api"

        def build_client(self, spec): return MagicMock()
        def prepare_request(self, spec, history, message, extra_system): return {}
        def iter_text_chunks(self, client, request):
            yield "world"
        def extract_usage(self):
            return {"input_tokens": 3, "output_tokens": 1, "cache_read": 0, "cache_create": 0}
        def format_history(self, history, message, response_text): return []

    spec = MagicMock()
    spec.id = "test-spec"
    spec.model = "claude-test"

    with patch("larkhelm.token_stats.record_token_usage", side_effect=fake_record):
        with patch("larkhelm.backend_api_streaming.BACKEND_REGISTRY"):
            from larkhelm.backend_api_streaming import _run_streaming_api
            _run_streaming_api(
                adapter=_StubAdapter(),
                spec=spec,
                chat_id="plainid",
                message="hello",
                history=[],
            )

    assert len(recorded) == 1
    assert recorded[0][0] == "plainid"


# ── AC-07: get_token_stats_persistent aggregation ─────────────────────────────


def test_get_token_stats_persistent_aggregates_memory_tokens(tmp_path):
    """Memory-cascade tokens written under orig chat_id are found by persistent stats."""
    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path

    # Write a fake all.jsonl with a memory-cascade token record (chat_id already stripped)
    jsonl = tmp_path / "all.jsonl"
    record = {
        "ts":            "2026-05-21T10:00:00",
        "chat_id":       "origchat",
        "role":          "token",
        "model":         "deepseek-chat",
        "input_tokens":  100,
        "output_tokens": 50,
        "cache_read":    0,
        "cache_create":  0,
        "cost_usd":      0.001,
    }
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    from larkhelm.token_stats import get_token_stats_persistent
    result = get_token_stats_persistent("origchat")

    assert "deepseek-chat" in result
    assert result["deepseek-chat"]["input_tokens"] == 100
    assert result["deepseek-chat"]["output_tokens"] == 50


def test_get_token_stats_persistent_ignores_namespace_entries(tmp_path):
    """If a namespace accidentally ended up in the JSONL, it should NOT match orig chat_id."""
    import larkhelm.config as _cfg
    _cfg.LOG_DIR = tmp_path

    jsonl = tmp_path / "all.jsonl"
    # A badly-recorded namespace entry (this should not happen after TOKEN-C3)
    record = {
        "ts":            "2026-05-21T10:00:00",
        "chat_id":       "origchat__memory_session",  # bad: namespace leaked
        "role":          "token",
        "model":         "claude-sonnet",
        "input_tokens":  999,
        "output_tokens": 111,
        "cache_read":    0,
        "cache_create":  0,
        "cost_usd":      0.1,
    }
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    from larkhelm.token_stats import get_token_stats_persistent
    result = get_token_stats_persistent("origchat")

    # Strict exact-match in persistent stats means bad namespace entry is NOT aggregated
    assert result == {}


# ── Three-path integration: verify resolve is used in all three paths ─────────


def test_all_three_paths_resolve_memory_prefix(tmp_path):
    """Smoke-test that all three token-recording paths correctly strip __memory_ prefix."""
    from larkhelm.token_stats import resolve_record_chat_id

    # CLI path (BaseProcessRunner)
    assert resolve_record_chat_id("chat1__memory_session") == "chat1"
    # HTTP path (DeepSeekRunner)
    assert resolve_record_chat_id("chat2__memory_project") == "chat2"
    # SDK path (backend_api_streaming)
    assert resolve_record_chat_id("chat3__memory_global") == "chat3"
    # No prefix — unchanged
    assert resolve_record_chat_id("chat4") == "chat4"
    # crew prefix — unchanged from existing behaviour
    assert resolve_record_chat_id("chat5__crew_agent") == "chat5"
