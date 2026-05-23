"""P0 acceptance tests — Claude session auto-reset (design.md §7 AC-01..05).

These tests exercise ``larkhelm.claude_session_guard`` end-to-end:
  * counter accumulation via ``record_token_usage``
  * threshold trip → sid clear + counter zero + milestone + metric
  * disabled gate → byte-compat with master
  * ``/reset`` paths zero the counters

Each test pinpoints one acceptance criterion so a regression bisect lands
on the right gate.
"""
from __future__ import annotations

import os

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import pytest

import larkhelm.config as _cfg
from larkhelm import chat_state as cs
from larkhelm import claude_session_guard as guard
from larkhelm import metrics as _met
from larkhelm import token_stats as ts


@pytest.fixture
def fresh_chat(tmp_path, monkeypatch):
    """Wire a fresh chat state + LOG_DIR + SESSION_DIR + metrics registry.

    Each test starts with zero counters and a writable session dir so
    ``_clear_sid`` can unlink without ``FileNotFoundError`` surfacing.
    """
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_cfg, "SESSION_DIR", session_dir, raising=False)
    monkeypatch.setattr(_cfg, "LOG_DIR", log_dir, raising=False)
    monkeypatch.setattr(_cfg, "STATE_FILE", tmp_path / "state.json", raising=False)
    # Default thresholds (config-level — tests can override)
    monkeypatch.setattr(_cfg, "CLAUDE_SESSION_AUTO_RESET_ENABLED", True, raising=False)
    monkeypatch.setattr(_cfg, "CLAUDE_SESSION_RESET_CACHE_TOKENS", 5_000_000, raising=False)
    monkeypatch.setattr(_cfg, "CLAUDE_SESSION_RESET_TURNS", 50, raising=False)
    # Stub out memory.record_milestone so tests don't trigger maybe_auto_update.
    import larkhelm.memory as _memory
    monkeypatch.setattr(_memory, "record_milestone", lambda *a, **kw: None,
                        raising=False)
    # Wipe the in-memory state so prior tests can't bleed counters.
    with cs._state_lock:
        cs._chat_state_store.clear()
    _met._reset_for_tests()
    chat_id = "test_chat_p0"
    yield chat_id
    _met._reset_for_tests()
    with cs._state_lock:
        cs._chat_state_store.clear()


def _seed_sid(chat_id: str) -> str:
    """Write a sid file so we can later assert it gets unlinked."""
    sid = "fake-sid-abcdef"
    cs._save_sid(chat_id, sid, "claude")
    return sid


# ── AC-01: cache_read accumulator threshold ───────────────────────────────

def test_cache_read_threshold(fresh_chat):
    """≥ 5_000_000 cache_read tokens → reset triggered with reason=cache_tokens."""
    chat_id = fresh_chat
    _seed_sid(chat_id)
    # Just below threshold — no reset
    reason = guard.maybe_auto_reset_session(
        chat_id, "claude", {"cache_read": 4_900_000},
    )
    assert reason is None
    assert cs._load_sid(chat_id, "claude") is not None
    cache, turns = cs._get_claude_session_counters(chat_id)
    assert cache == 4_900_000
    assert turns == 1

    # Crossing 5_000_000 triggers reset
    reason = guard.maybe_auto_reset_session(
        chat_id, "claude", {"cache_read": 200_000},
    )
    assert reason == "cache_tokens"
    assert cs._load_sid(chat_id, "claude") is None, "sid should be unlinked"
    cache, turns = cs._get_claude_session_counters(chat_id)
    assert cache == 0
    assert turns == 0

    # Metric verification (only when prometheus_client is installed).
    try:
        import prometheus_client  # noqa: F401
        body = _met.render_exposition()
        assert (
            'larkhelm_session_auto_reset_total{reason="cache_tokens"} 1' in body
        )
    except ImportError:
        pass


# ── AC-02: turn-count threshold via record_token_usage hook ───────────────

def test_turn_threshold(fresh_chat):
    """50 record_token_usage(model='claude') calls → 51st sees sid=None."""
    chat_id = fresh_chat
    _seed_sid(chat_id)
    # 50 cheap calls, none crosses cache threshold (tiny cache_read).
    for _ in range(50):
        ts.record_token_usage(chat_id, "claude", {
            "input_tokens": 1, "output_tokens": 1,
            "cache_read":   100, "cache_create": 0, "cost_usd": 0.0,
        })
    # After the 50th call the threshold has been crossed → sid cleared.
    assert cs._load_sid(chat_id, "claude") is None, (
        "sid must be cleared once turn counter reaches 50"
    )
    cache, turns = cs._get_claude_session_counters(chat_id)
    assert turns == 0, "counters zeroed after reset"

    try:
        import prometheus_client  # noqa: F401
        body = _met.render_exposition()
        assert 'larkhelm_session_auto_reset_total{reason="turns"} 1' in body
    except ImportError:
        pass


# ── AC-03: get_session_counters return shape + values ─────────────────────

def test_get_session_counters_fields(fresh_chat):
    chat_id = fresh_chat
    snap = guard.get_session_counters(chat_id)
    assert set(snap.keys()) == {
        "cache_read", "turns", "threshold_cache_read", "threshold_turns",
        "enabled",
    }
    assert snap["cache_read"] == 0
    assert snap["turns"] == 0
    assert snap["threshold_cache_read"] == 5_000_000
    assert snap["threshold_turns"] == 50
    assert snap["enabled"] is True

    # Accumulate one call worth of cache and verify the snapshot reflects it.
    guard.maybe_auto_reset_session(chat_id, "claude", {"cache_read": 1234})
    snap = guard.get_session_counters(chat_id)
    assert snap["cache_read"] == 1234
    assert snap["turns"] == 1


# ── AC-04: disabled gate → no_op ──────────────────────────────────────────

def test_disabled_no_op(fresh_chat, monkeypatch):
    """When CLAUDE_SESSION_AUTO_RESET_ENABLED=False, no reset ever fires."""
    chat_id = fresh_chat
    monkeypatch.setattr(_cfg, "CLAUDE_SESSION_AUTO_RESET_ENABLED", False,
                        raising=False)
    sid = _seed_sid(chat_id)
    for _ in range(200):
        ts.record_token_usage(chat_id, "claude", {
            "input_tokens": 1, "output_tokens": 1,
            "cache_read":   100_000, "cache_create": 0, "cost_usd": 0.0,
        })
    # sid unchanged, counters still zero (guard didn't run).
    assert cs._load_sid(chat_id, "claude") == sid
    cache, turns = cs._get_claude_session_counters(chat_id)
    assert cache == 0 and turns == 0
    try:
        import prometheus_client  # noqa: F401
        body = _met.render_exposition()
        # No labeled values means the counter never fired — HELP/TYPE lines
        # may still be emitted by the registry, but no value rows exist.
        assert 'larkhelm_session_auto_reset_total{reason=' not in body
    except ImportError:
        pass


# ── AC-05: clear_session_counters zeroes both fields ──────────────────────

def test_clear_session_counters(fresh_chat):
    chat_id = fresh_chat
    # Pre-populate
    guard.maybe_auto_reset_session(chat_id, "claude", {"cache_read": 100})
    cache, turns = cs._get_claude_session_counters(chat_id)
    assert (cache, turns) == (100, 1)
    guard.clear_session_counters(chat_id)
    cache, turns = cs._get_claude_session_counters(chat_id)
    assert cache == 0
    assert turns == 0


# ── AC-04b: non-claude models are ignored ─────────────────────────────────

def test_non_claude_models_ignored(fresh_chat):
    chat_id = fresh_chat
    for model in ("gemini", "kimi", "deepseek"):
        reason = guard.maybe_auto_reset_session(
            chat_id, model, {"cache_read": 10_000_000},
        )
        assert reason is None
        cache, turns = cs._get_claude_session_counters(chat_id)
        assert cache == 0 and turns == 0


# ── never-raise: usage missing fields or garbage ──────────────────────────

def test_guard_never_raises_on_bad_usage(fresh_chat):
    chat_id = fresh_chat
    # None / missing keys / wrong types must all be safely swallowed.
    for bad in (None, {}, {"cache_read": "junk"}, {"cache_read": -1}):
        guard.maybe_auto_reset_session(chat_id, "claude", bad)  # no raise
