"""P1-1b: cron scheduler failure observability tests.

Covers REQ-01 through REQ-04 + AC-08 by driving
``larkhelm.bridge._process_cron_tick`` directly with synthetic cron
entries — never starts the daemon thread, so behaviour is deterministic.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from unittest import mock

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import larkhelm.config as _cfg  # noqa: E402

if not getattr(_cfg, "_runtime", None):
    import json as _json
    import pathlib as _pl
    _tmp = _pl.Path("/tmp") / "larkhelm-test-cron-obs"
    _tmp.mkdir(parents=True, exist_ok=True)
    _cfg_path = _tmp / "config.json"
    _cfg_path.write_text(_json.dumps({
        "APP_ID": "X", "APP_SECRET": "Y",
        "response_timeout": 30, "hard_timeout": 120,
    }))
    _cfg._init_runtime(str(_cfg_path), str(_tmp))


def _make_cron_entry(cron_id: str = "abc12345", query: str = "总结今日 git log") -> dict:
    """Build a cron entry whose ``expr`` always matches "just now"."""
    return {
        "id": cron_id,
        "expr": "* * * * *",  # every minute → diff < 65s for any tick
        "query": query,
        "model": "claude",
        "created_at": "2026-05-24T09:00:00",
    }


def _seed_chat(chat_id: str, crons: list[dict]) -> None:
    """Install a cron list into the chat store via the public setter."""
    from larkhelm.chat_state import _chat_state_store, _state_lock, _set_chat_field
    # Reset the store entry to avoid bleed from prior tests in the same module.
    with _state_lock:
        _chat_state_store.pop(chat_id, None)
    _set_chat_field(chat_id, "crons", crons)


# ── REQ-01 / AC-08 — success path writes last_run_status="ok" ─────────


def test_success_writes_last_run_ok(monkeypatch):
    from larkhelm import bridge as _br
    from larkhelm.chat_state import _get_chat_state

    chat_id = "chat_ok_1"
    entry = _make_cron_entry("ok000001")
    _seed_chat(chat_id, [entry])

    started: list[str] = []

    class _FakeThread:
        def __init__(self, *_a, **_kw):
            pass

        def start(self):
            started.append("ok")

    monkeypatch.setattr(_br.threading, "Thread", _FakeThread)
    # Bypass the lazy `_do_query` import in _process_cron_tick by stubbing
    # the handlers module attribute.
    import larkhelm.handlers as _h
    monkeypatch.setattr(_h, "_do_query", lambda *a, **kw: None, raising=False)

    now_aware = datetime.now()
    last_fired: dict[str, float] = {}
    consec: dict[str, int] = {}
    _br._process_cron_tick(chat_id, entry, now_aware, last_fired, consec)

    crons = _get_chat_state(chat_id).get("crons", [])
    assert crons, "cron entry should still exist after persist"
    assert crons[0]["last_run_status"] == "ok"
    assert crons[0]["last_run_at"], "last_run_at must be a non-empty ISO ts"
    assert crons[0].get("last_error", "") == ""
    assert consec.get("ok000001", 0) == 0
    assert started == ["ok"]


# ── REQ-02 — failure path writes last_run_status="error" + truncates ──


def test_failure_writes_last_run_error_and_truncates(monkeypatch):
    from larkhelm import bridge as _br
    from larkhelm.chat_state import _get_chat_state

    chat_id = "chat_err_1"
    entry = _make_cron_entry("err00001")
    _seed_chat(chat_id, [entry])

    long_msg = "x" * 1000  # 1000 chars → must be truncated to ≤ 200

    class _Boom:
        def __init__(self, *_a, **_kw):
            pass

        def start(self):
            raise RuntimeError(long_msg)

    monkeypatch.setattr(_br.threading, "Thread", _Boom)

    now_aware = datetime.now()
    last_fired: dict[str, float] = {}
    consec: dict[str, int] = {}
    _br._process_cron_tick(chat_id, entry, now_aware, last_fired, consec)

    crons = _get_chat_state(chat_id).get("crons", [])
    assert crons[0]["last_run_status"] == "error"
    assert crons[0]["last_run_at"]
    assert len(crons[0]["last_error"]) <= 200
    # Counter bumped by 1 (well below the 3-strike threshold).
    assert consec.get("err00001", 0) == 1


# ── REQ-03 — emit fires exactly once per 3-failure window ─────────────


def test_three_consecutive_failures_emits_once(monkeypatch):
    from larkhelm import bridge as _br

    chat_id = "chat_emit_1"
    entry = _make_cron_entry("emit0001")
    _seed_chat(chat_id, [entry])

    class _Boom:
        def __init__(self, *_a, **_kw):
            pass

        def start(self):
            raise RuntimeError("persistent failure")

    monkeypatch.setattr(_br.threading, "Thread", _Boom)

    calls: list[tuple] = []

    def _fake_emit(category, summary, detail=""):
        calls.append((category, summary, detail))

    monkeypatch.setattr(_br, "_emit_failure_report", _fake_emit)

    base = datetime.now()
    last_fired: dict[str, float] = {}
    consec: dict[str, int] = {}
    for i in range(5):
        # Step 60s between ticks so the per-cron last-fired dedup
        # (>50s window) lets every iteration actually execute.
        _br._process_cron_tick(
            chat_id, entry, base + timedelta(seconds=i * 60),
            last_fired, consec,
        )

    # Only the 3rd failure should trigger emit; tick #4 / #5 are below the
    # threshold because the 3-strike emit resets the counter back to 0.
    assert len(calls) == 1
    cat, summary, _detail = calls[0]
    assert cat == "cron"
    assert "emit0001" in summary
    assert "RuntimeError" in summary


# ── REQ-04 — /cron list renders both old and new entries gracefully ───


def test_cron_list_renders_old_and_new_entries(monkeypatch):
    from larkhelm import commands as _cm

    chat_id = "chat_list_1"
    old_entry = {
        "id": "old00001",
        "expr": "0 9 * * *",
        "query": "老条目无新字段",
        "model": "claude",
        "created_at": "2026-04-01T09:00:00",
    }
    new_entry = {
        "id": "new00001",
        "expr": "*/5 * * * *",
        "query": "新条目带 last_run_*",
        "model": "claude",
        "created_at": "2026-05-20T09:00:00",
        "last_run_at": "2026-05-24T09:05:00",
        "last_run_status": "ok",
        "last_error": "",
    }
    _seed_chat(chat_id, [old_entry, new_entry])

    captured: list[dict] = []

    def fake_send_card_reply(chat_id, msg_id, title, body, color="blue", note="",
                              buttons=None, normalize=True):
        captured.append({"title": title, "body": body, "color": color})
        return "fake_mid"

    monkeypatch.setattr(_cm, "send_card_reply", fake_send_card_reply, raising=False)

    _cm._cmd_cron(chat_id, "list", msg_id="m1")

    assert len(captured) == 1
    body = captured[0]["body"]
    assert "从未执行" in body
    assert "✅" in body
    assert "2026-05-24T09:05:00" in body
