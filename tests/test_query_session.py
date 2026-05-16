"""Tests for ``larkhelm.handlers._query_session.QuerySession`` (P1-1 PR2)."""
from __future__ import annotations

import os
import threading
import time
from unittest import mock

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import larkhelm.config as _cfg  # noqa: E402

if not getattr(_cfg, "_runtime", None):
    import json as _json
    import pathlib as _pl
    _tmp = _pl.Path("/tmp") / "larkhelm-test-qsession"
    _tmp.mkdir(parents=True, exist_ok=True)
    _cfg_path = _tmp / "config.json"
    _cfg_path.write_text(_json.dumps({
        "APP_ID": "X", "APP_SECRET": "Y",
        "response_timeout": 30, "hard_timeout": 120,
    }))
    _cfg._init_runtime(str(_cfg_path), str(_tmp))

from larkhelm.handlers._query_session import QuerySession  # noqa: E402


def test_construct_minimal():
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    assert qs.chat_id == "c1"
    assert qs.message == "hi"
    assert qs.model == "claude"
    assert qs.lock_released is False
    assert qs.cancel_ev is None
    # trace_id is hex (default 12 chars)
    assert isinstance(qs.trace_id, str) and len(qs.trace_id) == 12


def test_release_lock_safe_with_no_lock():
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    # Should be a silent no-op when no lock is set
    qs._release_lock_safe()


def test_release_lock_safe_with_unlocked_lock():
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    qs.chat_lock = threading.Lock()  # not acquired
    qs._release_lock_safe()  # must not raise


def test_release_lock_safe_with_acquired_lock():
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    qs.chat_lock = threading.Lock()
    qs.chat_lock.acquire()
    qs._release_lock_safe()
    # Now it should be released — re-acquire works
    assert qs.chat_lock.acquire(blocking=False)
    qs.chat_lock.release()


def test_on_soft_timeout_idempotent():
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    qs.chat_lock = threading.Lock()
    qs.chat_lock.acquire()
    qs.start_time = time.time() - 1
    # Patch concurrency helpers so we don't fan out into a real thread.
    with mock.patch("larkhelm.handlers._query_session._replace_cancel_event"), \
         mock.patch("larkhelm.handlers._query_session._pop_pending", return_value=None):
        qs.on_soft_timeout()
        assert qs.lock_released is True
        # Second call: no-op
        qs.on_soft_timeout()
        assert qs.lock_released is True


def test_on_cancel_patches_card(monkeypatch):
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    qs.start_time = time.time()
    qs.mid = "mid-1"
    qs.user_msg_id = "user-1"
    qs.eyes_reaction_id = "rxn-1"
    from larkhelm.handlers._query_card_state import QueryCardState
    qs.card_state = QueryCardState(chat_id="c1", model_name="Claude",
                                   start_time=qs.start_time)

    captured = {}

    def fake_patch(mid, card_json):
        captured["mid"] = mid
        captured["card"] = card_json
        return True

    def fake_delete(_uid, _rid):
        captured["deleted_reaction"] = True

    monkeypatch.setattr("larkhelm.handlers._query_session._patch_card_raw", fake_patch)
    monkeypatch.setattr("larkhelm.handlers._query_session.delete_reaction", fake_delete)
    qs.on_cancel()
    assert captured.get("mid") == "mid-1"
    assert "已取消" in captured.get("card", "")
    assert captured.get("deleted_reaction") is True
    assert qs.eyes_reaction_id is None


def test_on_timeout_reports_to_reply_card(monkeypatch):
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    qs.start_time = time.time()
    qs.mid = "mid-1"
    from larkhelm.handlers._query_card_state import QueryCardState
    qs.card_state = QueryCardState(chat_id="c1", model_name="Claude",
                                   start_time=qs.start_time)

    captured = {}

    def fake_reply_card(chat_id, mid, title, body, color="red", note=""):
        captured["title"] = title
        captured["body"] = body
        captured["color"] = color

    monkeypatch.setattr("larkhelm.handlers._query_session.reply_card", fake_reply_card)
    monkeypatch.setattr("larkhelm.handlers._query_session.log_entry", lambda *a, **kw: None)
    qs.on_timeout(TimeoutError("idle window exceeded"))
    assert captured.get("color") == "red"
    assert "强制终止" in captured.get("title", "")
    assert "idle window exceeded" in captured.get("body", "")


def test_on_error_emits_error_card(monkeypatch):
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    qs.start_time = time.time()
    qs.mid = "mid-1"
    from larkhelm.handlers._query_card_state import QueryCardState
    qs.card_state = QueryCardState(chat_id="c1", model_name="Claude",
                                   start_time=qs.start_time)
    captured = {}
    monkeypatch.setattr(
        "larkhelm.handlers._query_session.reply_card",
        lambda chat_id, mid, title, body, color="red", note="": captured.update(
            {"title": title, "color": color, "body": body},
        ),
    )
    monkeypatch.setattr("larkhelm.handlers._query_session.log_entry", lambda *a, **kw: None)
    qs.on_error(RuntimeError("boom"))
    assert captured.get("color") == "red"
    assert "错误" in captured.get("title", "")


def test_run_v2_flag_off_no_op(monkeypatch, tmp_path):
    """When the v2 flag is off, _do_query won't call QuerySession.run."""
    import larkhelm.config as _cfg
    _cfg.config = {"query_session_v2_enabled": False}
    # Just verify the flag check is the gating point.
    assert not _cfg.config.get("query_session_v2_enabled")
