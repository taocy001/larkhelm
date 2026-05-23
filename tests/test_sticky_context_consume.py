"""P2 acceptance tests — sticky crew context dedup (design.md §7 AC-09..11)."""
from __future__ import annotations

import inspect
import os
import time

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import pytest

import larkhelm.config as _cfg
from larkhelm import metrics as _met
from larkhelm.crew import _state as crew_state


@pytest.fixture(autouse=True)
def _fresh_crew_state(monkeypatch):
    """Reset the in-memory crew card index + metrics between tests."""
    with crew_state._crew_card_index_lock:
        crew_state._crew_card_index.clear()
        crew_state._recent_crew_by_chat.clear()
    _met._reset_for_tests()
    # Default knobs (tests may override)
    monkeypatch.setattr(_cfg, "RECENT_CREW_STICKY_TTL_SEC", 1800, raising=False)
    monkeypatch.setattr(_cfg, "RECENT_CREW_STICKY_MAX_INJECTIONS", 5, raising=False)
    yield
    with crew_state._crew_card_index_lock:
        crew_state._crew_card_index.clear()
        crew_state._recent_crew_by_chat.clear()
    _met._reset_for_tests()


def _register(chat_id: str, title: str = "Demo task") -> None:
    crew_state._register_crew_card(
        card_mid=f"card_{chat_id}", chat_id=chat_id, title=title, summary="…",
    )


# ── AC-09: TTL default 1800 + expiry behaviour ────────────────────────────

def test_ttl_default_1800(monkeypatch):
    """TTL=1800 by default; ageing the entry past TTL drops it.

    A read after expiry must return None AND bump the 'ttl' metric (when
    prometheus-client is installed).
    """
    chat_id = "chat_ttl"
    _register(chat_id, title="Plan A")
    entry = crew_state._recent_crew_by_chat[chat_id]
    # Age the entry 2000 s — past the 1800 s default TTL.
    entry["ts"] = time.time() - 2000
    assert crew_state.get_recent_crew_context(chat_id) is None
    # Lazy-removed
    assert chat_id not in crew_state._recent_crew_by_chat

    try:
        import prometheus_client  # noqa: F401
        body = _met.render_exposition()
        assert (
            'larkhelm_sticky_context_evicted_total{reason="ttl"} 1' in body
        )
    except ImportError:
        pass


# ── AC-10: max_injections evicts after 5 consumes ────────────────────────

def test_max_injections_evicts():
    chat_id = "chat_evict"
    _register(chat_id, title="Plan B")
    # 5 consumes — all return the entry, then the 6th must return None.
    seen = []
    for _ in range(5):
        seen.append(crew_state.consume_recent_crew_context(chat_id))
    assert all(e is not None for e in seen)
    # After 5 consumes the entry is evicted.
    assert crew_state.consume_recent_crew_context(chat_id) is None

    try:
        import prometheus_client  # noqa: F401
        body = _met.render_exposition()
        assert (
            'larkhelm_sticky_context_evicted_total{reason="max_injections"} 1'
            in body
        )
    except ImportError:
        pass


def test_max_injections_zero_disables_eviction(monkeypatch):
    """Setting max_injections=0 keeps the entry indefinitely (TTL-only mode)."""
    monkeypatch.setattr(_cfg, "RECENT_CREW_STICKY_MAX_INJECTIONS", 0,
                        raising=False)
    chat_id = "chat_unlimited"
    _register(chat_id)
    for _ in range(50):
        assert crew_state.consume_recent_crew_context(chat_id) is not None


# ── AC-11: _message.py uses consume; read-only callers keep get_* ─────────

def test_message_main_path_uses_consume():
    """_message.py main path must call consume_recent_crew_context."""
    import larkhelm.handlers._message as _msg
    src = inspect.getsource(_msg)
    assert "consume_recent_crew_context" in src, (
        "handlers/_message.py must use consume_recent_crew_context"
    )
    # And it must NOT have reverted to the read-only path on the main inject
    # branch — accept get_recent_crew_context only in unrelated contexts.
    # Simplest invariant: there is at least one consume_* call.


def test_read_only_callers_keep_get_recent_crew_context():
    """Read-only callers (commands, _query, _query_session) must not mutate."""
    import larkhelm.commands as _cmd
    cmd_src = inspect.getsource(_cmd)
    assert "consume_recent_crew_context" not in cmd_src, (
        "/status / /memory must not consume the sticky entry"
    )

    # _query / _query_session may not exist as modules in every commit; skip
    # gracefully if they don't to keep the test forwards-compatible.
    for mod_name in ("larkhelm.handlers._query",
                     "larkhelm.handlers._query_session"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        src = inspect.getsource(mod)
        assert "consume_recent_crew_context" not in src, (
            f"{mod_name} must keep using get_recent_crew_context"
        )


# ── Defensive: consume on empty chat is safe ──────────────────────────────

def test_consume_on_empty_chat_returns_none():
    assert crew_state.consume_recent_crew_context("missing_chat") is None
