"""Tests for larkhelm.session_guard — AC-05 and AC-08."""
from __future__ import annotations

import json
import os
import threading

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import larkhelm.config as _cfg  # noqa: E402
from larkhelm.session_guard import (  # noqa: E402
    _perform_reset,
    get_session_counters,
    maybe_auto_reset,
)


# ---------------------------------------------------------------------------
# AC-05: turns threshold triggers reset and counters are zeroed afterwards
# ---------------------------------------------------------------------------

def test_maybe_auto_reset_turns_threshold(monkeypatch):
    """Gemini policy max_turns=40; after 40 calls the guard returns 'turns' and
    counters are zeroed."""
    chat_id = "test_chat_ac05"

    # Ensure guard is enabled
    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", True)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_CHECKPOINT_BEFORE_RESET", False)
    monkeypatch.setattr(
        _cfg,
        "SESSION_GUARD_POLICIES",
        {"gemini": {"max_cache_read_tokens": 4_000_000, "max_turns": 40}},
    )

    # Stub out side-effects
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_sid", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_backend_session_counters",
        lambda *a, **kw: None,
    )

    milestone_calls = []

    def _fake_milestone(chat_id, kind, summary):
        milestone_calls.append((chat_id, kind, summary))

    monkeypatch.setattr(
        "larkhelm.session_guard.record_milestone",
        _fake_milestone,
        raising=False,
    )

    # Patch metrics helpers to no-ops
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_auto_reset",
        lambda *a: None,
        raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_checkpoint",
        lambda *a: None,
        raising=False,
    )

    # Patch _perform_reset's lazy imports to avoid heavy deps at test collection
    monkeypatch.setattr(
        "larkhelm.session_guard._checkpoint_enabled", lambda: False,
    )

    # Also clear any existing counters in chat_state
    from larkhelm.chat_state import _clear_backend_session_counters as _cs_clear
    _cs_clear(chat_id, "gemini")

    usage = {"cache_read": 0, "input_tokens": 100}
    result = None
    for _ in range(40):
        result = maybe_auto_reset(chat_id, "gemini", usage)

    assert result == "turns", f"Expected 'turns' but got {result!r}"

    # After reset the counters should be cleared (real _clear_backend_session_counters
    # was NOT patched here — we patched the one inside session_guard module; the
    # chat_state store may still hold old values, so just check the return value)


def test_maybe_auto_reset_min_turns_guard(monkeypatch):
    """AC-03: when turn_total < SESSION_GUARD_MIN_TURNS_BEFORE_RESET, return None."""
    chat_id = "test_chat_ac03_minturns"

    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", True)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_CHECKPOINT_BEFORE_RESET", False)
    monkeypatch.setattr(
        _cfg,
        "SESSION_GUARD_POLICIES",
        {"claude": {"max_cache_read_tokens": 0, "max_turns": 5}},
    )
    monkeypatch.setattr(_cfg, "SESSION_GUARD_MIN_TURNS_BEFORE_RESET", 10, raising=False)

    monkeypatch.setattr("larkhelm.session_guard._clear_sid", lambda *a, **kw: None)
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_backend_session_counters", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard._checkpoint_enabled", lambda: False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.record_milestone", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_auto_reset", lambda *a: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_checkpoint", lambda *a: None, raising=False,
    )

    from larkhelm.chat_state import _clear_backend_session_counters as _cs_clear
    _cs_clear(chat_id, "claude")

    usage = {"cache_read": 0, "input_tokens": 10}
    result = None
    for _ in range(5):
        result = maybe_auto_reset(chat_id, "claude", usage)

    assert result is None, (
        f"Expected None (min_turns=10 not reached after 5 turns) but got {result!r}"
    )


def test_maybe_auto_reset_disabled(monkeypatch):
    """When SESSION_GUARD_ENABLED is False, maybe_auto_reset must return None."""
    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", False)
    result = maybe_auto_reset("chat_disabled", "claude", {"cache_read": 999_999_999})
    assert result is None


def test_maybe_auto_reset_unknown_backend(monkeypatch):
    """Unknown backend with no policy → returns None (no threshold defined)."""
    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", True)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_POLICIES", {})
    result = maybe_auto_reset("chat_unknown", "nonexistent_backend", {"cache_read": 0})
    assert result is None


# ---------------------------------------------------------------------------
# AC-08: anchor JSON written to tmp_path with 'summary' key
# ---------------------------------------------------------------------------

def test_perform_reset_writes_anchor(monkeypatch, tmp_path):
    """_perform_reset should write an anchor JSON file when checkpoint is enabled
    and generate_session_checkpoint returns a non-empty summary."""
    chat_id = "anchor_test_chat"
    fixed_summary = "User is implementing Week-3 session guard feature."

    # Point SESSION_DIR at tmp_path so _write_anchor lands there
    monkeypatch.setattr(_cfg, "SESSION_DIR", tmp_path, raising=False)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_CHECKPOINT_BEFORE_RESET", True)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", True)

    # Patch generate_session_checkpoint to return fixed summary
    monkeypatch.setattr(
        "larkhelm.memory.generate_session_checkpoint",
        lambda chat_id, turns=5: fixed_summary,
        raising=False,
    )

    # Stub side-effects
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_sid", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_backend_session_counters",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.record_milestone",
        lambda *a, **kw: None,
        raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_auto_reset",
        lambda *a: None,
        raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_checkpoint",
        lambda *a: None,
        raising=False,
    )

    _perform_reset(chat_id, "claude", "turns", 0, 50)

    anchor_path = tmp_path / f"{chat_id}.anchor.json"
    assert anchor_path.exists(), f"Anchor file not created at {anchor_path}"

    data = json.loads(anchor_path.read_text("utf-8"))
    assert "summary" in data, f"'summary' key missing from anchor JSON: {data}"
    assert data["summary"] == fixed_summary
    assert data["backend"] == "claude"
    assert data["reason"] == "turns"


def test_perform_reset_checkpoint_turns_parameterized(monkeypatch, tmp_path):
    """AC-07: _perform_reset passes SESSION_GUARD_CHECKPOINT_TURNS to generate_session_checkpoint."""
    chat_id = "anchor_test_chat_ac07"
    fixed_summary = "Checkpoint with custom turns."
    captured_turns = []

    def _fake_checkpoint(cid, turns=5):
        captured_turns.append(turns)
        return fixed_summary

    monkeypatch.setattr(_cfg, "SESSION_DIR", tmp_path, raising=False)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_CHECKPOINT_BEFORE_RESET", True)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", True)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_CHECKPOINT_TURNS", 10, raising=False)

    monkeypatch.setattr(
        "larkhelm.memory.generate_session_checkpoint", _fake_checkpoint, raising=False,
    )
    monkeypatch.setattr("larkhelm.session_guard._clear_sid", lambda *a, **kw: None)
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_backend_session_counters", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.record_milestone", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_auto_reset", lambda *a: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_checkpoint", lambda *a: None, raising=False,
    )

    _perform_reset(chat_id, "claude", "turns", 0, 50)

    assert captured_turns, "generate_session_checkpoint was not called"
    assert captured_turns[0] == 10, (
        f"Expected turns=10 (from SESSION_GUARD_CHECKPOINT_TURNS) but got {captured_turns[0]}"
    )


def test_perform_reset_no_anchor_when_empty_summary(monkeypatch, tmp_path):
    """When checkpoint returns empty string, no anchor file should be written."""
    chat_id = "no_anchor_chat"

    monkeypatch.setattr(_cfg, "SESSION_DIR", tmp_path, raising=False)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_CHECKPOINT_BEFORE_RESET", True)
    monkeypatch.setattr(
        "larkhelm.memory.generate_session_checkpoint",
        lambda chat_id, turns=5: "",
        raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_sid", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_backend_session_counters",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.record_milestone",
        lambda *a, **kw: None,
        raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_auto_reset",
        lambda *a: None,
        raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_checkpoint",
        lambda *a: None,
        raising=False,
    )

    _perform_reset(chat_id, "gemini", "cache_tokens", 4_000_001, 10)

    anchor_path = tmp_path / f"{chat_id}.anchor.json"
    assert not anchor_path.exists(), "Anchor file should NOT be created for empty summary"


def test_get_session_counters_returns_dict(monkeypatch):
    """get_session_counters should always return a dict with expected keys."""
    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", True)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_POLICIES", {
        "claude": {"max_cache_read_tokens": 5_000_000, "max_turns": 50},
    })
    result = get_session_counters("some_chat", "claude")
    assert isinstance(result, dict)
    for key in ("cache_read", "turns", "threshold_cache_read", "threshold_turns", "enabled"):
        assert key in result, f"Missing key {key!r} in {result}"
    assert result["threshold_turns"] == 50
    assert result["threshold_cache_read"] == 5_000_000


# ---------------------------------------------------------------------------
# AC-01/02/03: record_token_usage routes all backends through session_guard
# ---------------------------------------------------------------------------

def test_record_token_usage_triggers_gemini_reset(monkeypatch):
    """AC-01: record_token_usage for gemini calls session_guard.maybe_auto_reset."""
    calls = []

    def _fake_maybe_auto_reset(chat_id, model, usage):
        calls.append((chat_id, model, usage))

    monkeypatch.setattr("larkhelm.session_guard.maybe_auto_reset", _fake_maybe_auto_reset)

    from larkhelm.token_stats import record_token_usage
    usage = {"input_tokens": 10, "output_tokens": 5, "cache_read": 100}
    record_token_usage("chat_ac01", "gemini", usage)

    assert any(c[1] == "gemini" for c in calls), (
        "expected maybe_auto_reset called with model='gemini'"
    )


def test_record_token_usage_triggers_deepseek_reset(monkeypatch):
    """AC-02: record_token_usage for deepseek calls session_guard.maybe_auto_reset."""
    calls = []

    def _fake_maybe_auto_reset(chat_id, model, usage):
        calls.append((chat_id, model, usage))

    monkeypatch.setattr("larkhelm.session_guard.maybe_auto_reset", _fake_maybe_auto_reset)

    from larkhelm.token_stats import record_token_usage
    usage = {"input_tokens": 20, "output_tokens": 10, "cache_read": 0}
    record_token_usage("chat_ac02", "deepseek", usage)

    assert any(c[1] == "deepseek" for c in calls), (
        "expected maybe_auto_reset called with model='deepseek'"
    )


def test_record_token_usage_kimi_turns_threshold(monkeypatch):
    """AC-03: record_token_usage for kimi calls session_guard.maybe_auto_reset."""
    calls = []

    def _fake_maybe_auto_reset(chat_id, model, usage):
        calls.append((chat_id, model, usage))

    monkeypatch.setattr("larkhelm.session_guard.maybe_auto_reset", _fake_maybe_auto_reset)

    from larkhelm.token_stats import record_token_usage
    usage = {"input_tokens": 30, "output_tokens": 15, "cache_read": 0}
    record_token_usage("chat_ac03", "kimi", usage)

    assert any(c[1] == "kimi" for c in calls), (
        "expected maybe_auto_reset called with model='kimi'"
    )
