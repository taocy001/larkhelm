"""MEM-C1 tests — AC-01/02/03/06/09.

Verifies that global memory file resolution uses the 3-level priority:
  explicit sender_open_id > ContextVar > chat_state fallback.
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest


# ─── helpers ──────────────────────────────────────────────────────────────────

def _file(memory_dir: Path, open_id: str) -> Path:
    return memory_dir / f"global_{open_id}.md"


def _patch_mem(monkeypatch, mem, tmp_path):
    """Patch MEMORY_HOME_DIR and stub _ensure_dir to use tmp_path."""
    monkeypatch.setattr(mem, "MEMORY_HOME_DIR", tmp_path)
    monkeypatch.setattr(mem, "_ensure_dir", lambda: None)


# ─── AC-01 explicit param takes priority over ContextVar ──────────────────────

def test_explicit_param_wins_over_contextvar(tmp_path, monkeypatch):
    """_global_memory_file returns file for explicit open_id even when ContextVar has a different value."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)

    # Set ContextVar to a different open_id
    token = mem._query_sender_open_id.set("contextvar_user")
    try:
        result = mem._global_memory_file("chat1", sender_open_id="explicit_user")
        assert result == tmp_path / "global_explicit_user.md"
    finally:
        mem._query_sender_open_id.reset(token)


# ─── AC-02 ContextVar takes priority over chat_state ──────────────────────────

def test_contextvar_wins_over_chat_state(tmp_path, monkeypatch):
    """When no explicit param, ContextVar is used before chat_state."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)

    # Patch chat_state to return a different open_id
    with patch("larkhelm.memory._get_chat_state", return_value={"sender_open_id": "state_user"}):
        token = mem._query_sender_open_id.set("contextvar_user")
        try:
            result = mem._global_memory_file("chat1")
            assert result == tmp_path / "global_contextvar_user.md"
        finally:
            mem._query_sender_open_id.reset(token)


# ─── AC-03 chat_state fallback when no explicit or ContextVar ─────────────────

def test_chat_state_fallback(tmp_path, monkeypatch):
    """Falls back to chat_state["sender_open_id"] when no explicit param and ContextVar is empty."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)

    # Ensure ContextVar is empty
    token = mem._query_sender_open_id.set("")
    try:
        with patch("larkhelm.memory._get_chat_state", return_value={"sender_open_id": "state_user"}):
            result = mem._global_memory_file("chat1")
            assert result == tmp_path / "global_state_user.md"
    finally:
        mem._query_sender_open_id.reset(token)


# ─── AC-06 no open_id at all → returns None ───────────────────────────────────

def test_returns_none_when_no_open_id(tmp_path, monkeypatch):
    """Returns None when no open_id can be resolved from any source."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)

    token = mem._query_sender_open_id.set("")
    try:
        with patch("larkhelm.memory._get_chat_state", return_value={}):
            result = mem._global_memory_file("chat1")
            assert result is None
    finally:
        mem._query_sender_open_id.reset(token)


# ─── AC-09 thread isolation via ContextVar ────────────────────────────────────

def test_contextvar_is_thread_isolated(tmp_path, monkeypatch):
    """Two concurrent threads set different ContextVar values and get different file paths."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)

    results = {}
    errors = []

    def worker(name: str, open_id: str):
        try:
            token = mem._query_sender_open_id.set(open_id)
            try:
                with patch("larkhelm.memory._get_chat_state", return_value={}):
                    result = mem._global_memory_file(f"chat_{name}")
                results[name] = result
            finally:
                mem._query_sender_open_id.reset(token)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("alice", "user_alice"))
    t2 = threading.Thread(target=worker, args=("bob", "user_bob"))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not errors, f"Thread errors: {errors}"
    assert results["alice"] == tmp_path / "global_user_alice.md"
    assert results["bob"] == tmp_path / "global_user_bob.md"
    assert results["alice"] != results["bob"]


# ─── MEM-C1 cascade write path: explicit sender_open_id pass-through ──────────
#
# ContextVar does NOT cross threads (CPython 3.14 default build:
# sys.flags.thread_inherit_context=0; ≤3.13 has no inheritance at all), so
# the cascade write path must carry sender_open_id as an explicit parameter
# all the way down — otherwise group chats fall back to chat_state's
# last-writer-wins value and pollute another user's global memory.

def _reset_cascade_coordinator(mem):
    with mem._active_cancels_lock:
        mem._active_cascade_cancels.clear()
    with mem._CASCADE_SEM_LOCK:
        mem._CASCADE_SEM = None


def test_cascade_passes_explicit_sender_to_global_extract(tmp_path, monkeypatch):
    """_cascade_extract must hand the EXPLICIT sender_open_id to
    _try_extract_global — not the chat_state fallback value."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)
    _reset_cascade_coordinator(mem)

    received = {}
    done = threading.Event()

    def fake_global(session_content, chat_id, cancel_ev=None, *,
                    sender_open_id=None):
        received["sender_open_id"] = sender_open_id
        done.set()

    monkeypatch.setattr(mem, "_try_extract_global", fake_global)
    monkeypatch.setattr(mem, "_try_extract_project", lambda *a, **k: None)
    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda chat_id: None)
    # Group-chat scenario: chat_state holds ANOTHER user (last-writer-wins).
    monkeypatch.setattr(mem, "_get_chat_state",
                        lambda chat_id: {"sender_open_id": "ou_bob"})

    mem._cascade_extract("session payload", "chat_grp",
                         sender_open_id="ou_alice")
    assert done.wait(timeout=5), "global extract worker did not run"
    assert received["sender_open_id"] == "ou_alice"


def test_cascade_default_sender_is_none(tmp_path, monkeypatch):
    """Legacy callers (no sender) keep the old fallback chain: the cascade
    forwards sender_open_id=None untouched."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)
    _reset_cascade_coordinator(mem)

    received = {}
    done = threading.Event()

    def fake_global(session_content, chat_id, cancel_ev=None, *,
                    sender_open_id="SENTINEL"):
        received["sender_open_id"] = sender_open_id
        done.set()

    monkeypatch.setattr(mem, "_try_extract_global", fake_global)
    monkeypatch.setattr(mem, "_try_extract_project", lambda *a, **k: None)
    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda chat_id: None)

    mem._cascade_extract("session payload", "chat_legacy")
    assert done.wait(timeout=5), "global extract worker did not run"
    assert received["sender_open_id"] is None


def test_maybe_auto_update_forwards_sender_to_cascade(tmp_path, monkeypatch):
    """maybe_auto_update(sender_open_id=...) must reach _cascade_extract."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)

    received = {}
    done = threading.Event()

    def fake_cascade(session_content, chat_id, *, sender_open_id=None):
        received["sender_open_id"] = sender_open_id
        done.set()

    monkeypatch.setattr(mem, "_cascade_extract", fake_cascade)
    monkeypatch.setattr(mem, "_get_turn_count", lambda chat_id: 5)
    monkeypatch.setattr(mem, "_read_logs_tail", lambda chat_id: [
        {"ts": "2026-06-12T00:00:00", "role": "user",
         "content": "hi", "model": "claude"},
        {"ts": "2026-06-12T00:00:01", "role": "assistant",
         "content": "hello", "model": "claude"},
    ])
    monkeypatch.setattr(mem, "load_memory", lambda chat_id: None)
    monkeypatch.setattr(mem, "generate_memory",
                        lambda *a, **k: "## Summary\nsubstantial content here")
    monkeypatch.setattr(mem, "save_memory", lambda *a, **k: True)

    mem.maybe_auto_update("chat_grp", force=True, sender_open_id="ou_alice")
    assert done.wait(timeout=5), "cascade was not invoked"
    assert received["sender_open_id"] == "ou_alice"


# ─── load/save round-trip with explicit param ─────────────────────────────────

def test_load_save_roundtrip_with_explicit_param(tmp_path, monkeypatch):
    """load_global_memory and save_global_memory respect explicit sender_open_id."""
    import larkhelm.memory as mem
    _patch_mem(monkeypatch, mem, tmp_path)

    token = mem._query_sender_open_id.set("")
    try:
        with patch("larkhelm.memory._get_chat_state", return_value={}):
            mem.save_global_memory("hello world", "chat1", sender_open_id="user_x")
            loaded = mem.load_global_memory("chat1", sender_open_id="user_x")
            assert loaded == "hello world"
            # Different open_id returns nothing (None or empty)
            loaded_other = mem.load_global_memory("chat1", sender_open_id="user_y")
            assert not loaded_other
    finally:
        mem._query_sender_open_id.reset(token)
