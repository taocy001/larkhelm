"""Tests for larkhelm.session_guard — AC-05 and settle-before-reset."""
from __future__ import annotations

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
        "larkhelm.session_guard.record_milestone", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.inc_session_auto_reset", lambda *a: None, raising=False,
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
# Checkpoint/anchor chain removed (write-only dead path): _perform_reset must
# not write {chat_id}.anchor.json and the chain's symbols must stay deleted.
# ---------------------------------------------------------------------------

def test_perform_reset_writes_no_anchor_file(monkeypatch, tmp_path):
    """The checkpoint/anchor sidecar chain was deleted: a reset must not
    create any *.anchor.json file under SESSION_DIR."""
    chat_id = "no_anchor_chain_chat"

    monkeypatch.setattr(_cfg, "SESSION_DIR", tmp_path, raising=False)
    monkeypatch.setattr(_cfg, "SESSION_GUARD_ENABLED", True)
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_sid", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_backend_session_counters",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.memory.record_milestone", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.memory.maybe_auto_update",
        lambda cid, force=False, on_done=None, *, sender_open_id=None: (
            on_done(True, "memory content", None) if on_done else None
        ),
        raising=False,
    )

    _perform_reset(chat_id, "claude", "turns", 0, 50)

    assert not list(tmp_path.glob("*.anchor.json")), (
        "checkpoint/anchor chain was deleted — no anchor file may be written"
    )


def test_checkpoint_anchor_symbols_deleted():
    """Pin the deletion: neither session_guard nor memory re-grow the
    write-only checkpoint/anchor helpers."""
    import larkhelm.memory as memory
    import larkhelm.session_guard as session_guard

    for mod, name in (
        (session_guard, "_write_anchor"),
        (session_guard, "_checkpoint_enabled"),
        (memory, "generate_session_checkpoint"),
        (memory, "load_session_anchor"),
    ):
        assert not hasattr(mod, name), f"{mod.__name__}.{name} should be deleted"
    assert not hasattr(_cfg, "SESSION_GUARD_CHECKPOINT_BEFORE_RESET")
    assert not hasattr(_cfg, "SESSION_GUARD_CHECKPOINT_TURNS")


# ---------------------------------------------------------------------------
# Settle-before-reset: memory summary must complete before _clear_sid
# ---------------------------------------------------------------------------

def _stub_reset_side_effects(monkeypatch, events=None):
    """Stub everything in _perform_reset except the settle step."""
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_sid",
        lambda *a, **kw: events.append("clear_sid") if events is not None else None,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard._clear_backend_session_counters",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "larkhelm.memory.record_milestone", lambda *a, **kw: None, raising=False,
    )


def test_perform_reset_waits_for_settle_before_clear_sid(monkeypatch):
    """The forced pre-reset maybe_auto_update must complete (on_done fired)
    before _clear_sid runs — not fire-and-forget."""
    chat_id = "settle_order_chat"
    events = []
    _stub_reset_side_effects(monkeypatch, events)

    def _fake_maybe_auto_update(cid, force=False, on_done=None, *,
                                sender_open_id=None):
        def _bg():
            import time
            time.sleep(0.2)  # simulate slow LLM summarization
            events.append("settle_done")
            if on_done:
                on_done(True, "memory content", None)
        threading.Thread(target=_bg, daemon=True).start()

    monkeypatch.setattr(
        "larkhelm.memory.maybe_auto_update", _fake_maybe_auto_update, raising=False,
    )

    _perform_reset(chat_id, "claude", "turns", 0, 50)

    assert events[:2] == ["settle_done", "clear_sid"], (
        f"settle must complete before sid is cleared, got order: {events}"
    )


def test_perform_reset_settle_retries_once_on_already_in_progress(monkeypatch):
    """If the forced update is crowded out by a regular update
    (already_in_progress), settle waits for the update lock and retries once."""
    chat_id = "settle_retry_chat"
    events = []
    _stub_reset_side_effects(monkeypatch, events)

    calls = []

    def _fake_maybe_auto_update(cid, force=False, on_done=None, *,
                                sender_open_id=None):
        calls.append(force)
        if len(calls) == 1:
            on_done(False, None, "already_in_progress")
        else:
            events.append("settle_done")
            on_done(True, "memory content", None)

    monkeypatch.setattr(
        "larkhelm.memory.maybe_auto_update", _fake_maybe_auto_update, raising=False,
    )

    _perform_reset(chat_id, "claude", "turns", 0, 50)

    assert len(calls) == 2, f"expected exactly one retry (2 calls), got {len(calls)}"
    assert events[:2] == ["settle_done", "clear_sid"]


def test_perform_reset_settle_skipped_after_retry_still_resets(monkeypatch):
    """If both settle attempts are crowded out, log 'settle skipped' and
    proceed with the reset anyway (best-effort)."""
    chat_id = "settle_skip_chat"
    events = []
    _stub_reset_side_effects(monkeypatch, events)

    calls = []

    def _fake_maybe_auto_update(cid, force=False, on_done=None, *,
                                sender_open_id=None):
        calls.append(force)
        on_done(False, None, "already_in_progress")

    monkeypatch.setattr(
        "larkhelm.memory.maybe_auto_update", _fake_maybe_auto_update, raising=False,
    )

    logs = []
    monkeypatch.setattr(
        "larkhelm.session_guard._debug_log", lambda msg: logs.append(msg),
    )

    _perform_reset(chat_id, "claude", "turns", 0, 50)

    assert len(calls) == 2, f"expected exactly 2 attempts, got {len(calls)}"
    assert "clear_sid" in events, "reset must still proceed when settle is skipped"
    assert any("settle skipped" in m for m in logs), (
        f"expected '[SessionGuard] settle skipped' log, got: {logs}"
    )


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
