"""P0 fixes for /btw and /compact.

1. /btw — memory block is injected ONLY into brand-new CLI sessions
   (sid is None). A --resume'd session already carries the first-turn
   memory in its transcript; re-injecting on every /btw accumulates
   duplicate copies (CTX-C1 recurrence at the /btw entry point).
   Also pins the session_key plumbing: the runner must save a new sid
   under the same key /btw loaded it from (btw_<spec.id> when the main
   lock is busy), instead of overwriting the main-session sid.

2. /compact — _do_compact must wait for maybe_auto_update's on_done
   callback: clear sid + green card only on success; warning card and
   NO sid clear on failure / timeout / already_in_progress.
"""
from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import larkhelm.commands as commands  # noqa: E402


# ═══════════════════════════════════════════════════
#  /btw helpers
# ═══════════════════════════════════════════════════


class _SyncThread:
    """threading.Thread stand-in that runs target synchronously on start()."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        pass


class _FakeThreadingModule:
    """Delegates to the real threading module except Thread → _SyncThread."""

    Thread = _SyncThread

    def __getattr__(self, name):
        return getattr(threading, name)


def _claude_spec():
    return SimpleNamespace(
        id="claude", provider="claude_cli", healthy=True, enabled=True,
        command="claude", model=None, extra_args=None,
    )


def _run_btw(monkeypatch, *, sid, main_busy=False, mem_ctx="MEMBLOCK",
             question="what is x?"):
    """Drive _cmd_btw synchronously against a fully-mocked claude_cli backend.

    Returns the kwargs captured from backend_cli.run_claude plus the
    sid_key passed to _load_sid.
    """
    import larkhelm.backend_registry as breg
    import larkhelm.backend_cli as bcli
    import larkhelm.memory as mem

    captured: dict = {}

    spec = _claude_spec()
    fake_reg = SimpleNamespace(
        get=lambda pid: spec,
        get_orchestrator=lambda: spec,
    )

    chat_lock = threading.Lock()
    if main_busy:
        chat_lock.acquire()

    def fake_load_sid(chat_id, key):
        captured["load_sid_key"] = key
        return sid

    def fake_run_claude(**kw):
        captured["run_claude"] = kw
        return "answer"

    monkeypatch.setattr(commands, "threading", _FakeThreadingModule())
    monkeypatch.setattr(commands, "_get_lang", lambda c: "zh")
    monkeypatch.setattr(commands, "_get_chat_lock", lambda c: chat_lock)
    monkeypatch.setattr(commands, "_get_btw_lock", lambda c: threading.Lock())
    monkeypatch.setattr(commands, "_get_cwd", lambda c: "/tmp")
    monkeypatch.setattr(commands, "_get_chat_state",
                        lambda c: {"backend_id": "claude"})
    monkeypatch.setattr(commands, "_load_sid", fake_load_sid)
    monkeypatch.setattr(commands, "react_to_message", lambda *a, **k: None)
    monkeypatch.setattr(commands, "delete_reaction", lambda *a, **k: None)
    monkeypatch.setattr(commands, "_reply_card_raw", lambda *a, **k: "mid1")
    monkeypatch.setattr(commands, "_patch_card_raw", lambda *a, **k: None)
    monkeypatch.setattr(commands, "_make_card", lambda *a, **k: {})
    monkeypatch.setattr(commands, "_register_btw_msg", lambda *a, **k: None)
    monkeypatch.setattr(breg, "BACKEND_REGISTRY", fake_reg)
    monkeypatch.setattr(mem, "get_memory_context_v2",
                        lambda *a, **k: (mem_ctx, None))
    monkeypatch.setattr(bcli, "run_claude", fake_run_claude)

    commands._cmd_btw("chat_btw_1", question, "umsg1", sender_open_id="ou_x")
    return captured


# ═══════════════════════════════════════════════════
#  /btw — memory injection guard
# ═══════════════════════════════════════════════════


def test_btw_new_session_injects_memory(monkeypatch):
    """sid=None (new CLI session) → memory block IS injected."""
    cap = _run_btw(monkeypatch, sid=None)
    msg = cap["run_claude"]["message"]
    assert msg.startswith("[System]\nMEMBLOCK")
    assert "[User Query]\nwhat is x?" in msg


def test_btw_resumed_session_skips_memory(monkeypatch):
    """sid set (--resume'd session) → message is the bare question."""
    cap = _run_btw(monkeypatch, sid="sid-existing")
    assert cap["run_claude"]["message"] == "what is x?"
    assert cap["run_claude"]["sid"] == "sid-existing"


def test_btw_no_memory_no_prefix(monkeypatch):
    """Empty memory context never produces a [System] prefix."""
    cap = _run_btw(monkeypatch, sid=None, mem_ctx="")
    assert cap["run_claude"]["message"] == "what is x?"


# ═══════════════════════════════════════════════════
#  /btw — session_key isolation
# ═══════════════════════════════════════════════════


def test_btw_main_free_uses_main_session_key(monkeypatch):
    cap = _run_btw(monkeypatch, sid="sid-main", main_busy=False)
    assert cap["load_sid_key"] == "claude"
    assert cap["run_claude"]["session_key"] == "claude"


def test_btw_main_busy_uses_btw_session_key(monkeypatch):
    """Main lock busy → /btw loads AND saves under btw_<spec.id>, so a
    new sid from the runner cannot overwrite the main-session sid."""
    cap = _run_btw(monkeypatch, sid=None, main_busy=True)
    assert cap["load_sid_key"] == "btw_claude"
    assert cap["run_claude"]["session_key"] == "btw_claude"


# ═══════════════════════════════════════════════════
#  /compact — on_done gating
# ═══════════════════════════════════════════════════


def _run_compact(monkeypatch, fake_auto_update, wait_sec=None):
    """Drive _do_compact with a stubbed maybe_auto_update.

    Returns (cards, cleared) where cards is the list of
    (title, body, color) sent via send_card_reply and cleared is the
    list of _clear_sid(chat_id, model) calls.
    """
    import larkhelm.memory as mem
    import larkhelm.chat_state as cstate
    import larkhelm.log as logmod

    cards: list = []
    cleared: list = []

    def fake_send_card_reply(chat_id, msg_id, title, body, color=None, **kw):
        cards.append((title, body, color))
        return "mid"

    monkeypatch.setattr(mem, "maybe_auto_update", fake_auto_update)
    monkeypatch.setattr(cstate, "_get_chat_model", lambda c: "claude")
    monkeypatch.setattr(cstate, "_load_sid", lambda c, m: "sid123")
    monkeypatch.setattr(cstate, "_clear_sid",
                        lambda c, m: cleared.append((c, m)))
    monkeypatch.setattr(logmod, "_read_logs_tail",
                        lambda c: [{"role": "user", "content": "hi"}])
    monkeypatch.setattr(commands, "send_card_reply", fake_send_card_reply)
    if wait_sec is not None:
        monkeypatch.setattr(commands, "_COMPACT_WAIT_SEC", wait_sec)

    commands._do_compact("chat_cp_1", "msg1", "zh")
    return cards, cleared


def test_compact_success_clears_sid_and_reports_green(monkeypatch):
    def fake_auto_update(chat_id, force=False, on_done=None):
        on_done(True, "summary text", None)

    cards, cleared = _run_compact(monkeypatch, fake_auto_update)
    assert cleared == [("chat_cp_1", "claude")]
    title, _body, color = cards[-1]
    assert "压缩完成" in title
    assert color == "green"


def test_compact_failure_keeps_sid_and_warns(monkeypatch):
    def fake_auto_update(chat_id, force=False, on_done=None):
        on_done(False, None, "already_in_progress")

    cards, cleared = _run_compact(monkeypatch, fake_auto_update)
    assert cleared == []  # sid must NOT be cleared on failure
    title, body, color = cards[-1]
    assert "压缩未完成" in title
    assert color == "orange"
    assert "未重置" in body


def test_compact_timeout_keeps_sid_and_warns(monkeypatch):
    def fake_auto_update(chat_id, force=False, on_done=None):
        pass  # never calls on_done → wait times out

    cards, cleared = _run_compact(monkeypatch, fake_auto_update,
                                  wait_sec=0.05)
    assert cleared == []
    title, body, color = cards[-1]
    assert "压缩未完成" in title
    assert color == "orange"
    assert "超时" in body


def test_compact_generation_error_keeps_sid(monkeypatch):
    def fake_auto_update(chat_id, force=False, on_done=None):
        on_done(False, None, "boom: llm exploded")

    cards, cleared = _run_compact(monkeypatch, fake_auto_update)
    assert cleared == []
    title, body, color = cards[-1]
    assert color == "orange"
    assert "boom" in body
