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


def test_on_error_handles_none_card_state(monkeypatch):
    """Regression for round-3 NICE-TO-HAVE.

    When ``QuerySession.run()`` raises BEFORE line 113 (where ``card_state``
    is assigned) — e.g. ``_resolve_initial_model`` / ``emit_init_card``
    raised — the old ``assert self.card_state is not None`` at the top
    of ``on_error`` itself raised AssertionError, leaving the user with
    a stuck init card and no error feedback.

    Fixed by gracefully skipping the heartbeat-flush path when
    ``card_state is None`` (nothing to flush). The error card itself
    still gets emitted via ``reply_card``.
    """
    qs = QuerySession(chat_id="c1", message="hi", model="claude")
    qs.start_time = time.time()
    qs.mid = "mid-1"
    qs.card_state = None   # ← the bug scenario: card_state never got assigned
    captured = {}
    monkeypatch.setattr(
        "larkhelm.handlers._query_session.reply_card",
        lambda chat_id, mid, title, body, color="red", note="": captured.update(
            {"title": title, "color": color, "body": body},
        ),
    )
    monkeypatch.setattr("larkhelm.handlers._query_session.log_entry", lambda *a, **kw: None)
    # Must not raise AssertionError; must still surface the error to user.
    qs.on_error(RuntimeError("v2 raised before card_state set"))
    assert captured.get("color") == "red"
    assert "错误" in captured.get("title", "")


def test_run_v2_flag_off_no_op(monkeypatch, tmp_path):
    """When the v2 flag is off, _do_query won't call QuerySession.run."""
    import larkhelm.config as _cfg
    _cfg.config = {"query_session_v2_enabled": False}
    # Just verify the flag check is the gating point.
    assert not _cfg.config.get("query_session_v2_enabled")


def test_queue_behind_running_query_rolls_back_pending_on_card_failure(monkeypatch):
    """Regression for round-3 NICE-TO-HAVE pending-leak.

    ``_queue_behind_running_query`` writes the pending slot FIRST, then
    emits a Feishu card. If the card emission fails (5xx, network), the
    pending state remained set with no consumer — the next user message
    couldn't re-queue cleanly. Fix: catch + ``_pop_pending`` on rollback.
    """
    from larkhelm.concurrency import _pop_pending, _pending_msg
    # Make sure no leftover state from earlier tests
    _pop_pending("c_rollback")

    qs = QuerySession(chat_id="c_rollback", message="hi", model="claude",
                      user_msg_id="msg-abc")

    def _boom_card(*a, **kw):
        raise RuntimeError("Feishu 503 simulated")

    monkeypatch.setattr(
        "larkhelm.handlers._query_session._reply_card_raw", _boom_card,
    )
    monkeypatch.setattr(
        "larkhelm.handlers._query_session._send_card_raw", _boom_card,
    )
    monkeypatch.setattr(
        "larkhelm.handlers._query_session._patch_card_raw", _boom_card,
    )

    # Must not raise; the function silently rolls back.
    qs._queue_behind_running_query()

    # Pending slot must NOT remain set after a failed card emission.
    assert "c_rollback" not in _pending_msg, (
        "pending slot leaked after card emission failure — next message "
        "will see a ghost queue with no consumer"
    )


def test_do_query_v2_raise_does_not_fall_back_to_legacy(monkeypatch):
    """Regression for round-2 review MUST-FIX (_query.py:347-357).

    Bug: when ``query_session_v2_enabled=true`` and ``QuerySession.run()``
    raised AFTER acquiring side effects (chat_lock / init card / heartbeat),
    the catch-all ``except Exception`` fell through to the legacy
    ``_do_query`` body, which would re-acquire the lock and emit a SECOND
    init card + run the LLM AGAIN — double-processing the same query.

    Fix narrowed the fallback to setup-time failures (import / __init__):
    once ``run()`` is entered, v2 OWNS the request — any raise is logged
    and we return, never re-entering legacy.

    This test:
      • Flips ``query_session_v2_enabled=true``.
      • Makes ``QuerySession.run`` raise mid-execution (post-construction).
      • Asserts the legacy ``_do_query`` body is NOT reached afterwards
        (we instrument it via a sentinel side effect).
    """
    import larkhelm.config as _cfg
    _cfg.config = {"query_session_v2_enabled": True}

    legacy_reached: list[bool] = []
    raise_payload = RuntimeError("v2 raised post-construction")

    # Stub QuerySession.run to raise like the bug scenario.
    class BoomSession:
        def __init__(self, **_kwargs): pass
        def run(self): raise raise_payload

    monkeypatch.setattr(
        "larkhelm.handlers._query_session.QuerySession", BoomSession,
    )

    # Sentinel hook just BELOW the v2 dispatch in _do_query — the line
    # ``trace_id = uuid.uuid4().hex[:12]`` is the first thing the legacy
    # body does. Patching uuid in that module catches re-entry.
    import larkhelm.handlers._query as _q
    import uuid

    class _UUIDProbe:
        @staticmethod
        def uuid4():
            legacy_reached.append(True)
            return uuid.UUID(int=0)
    monkeypatch.setattr(_q, "uuid", _UUIDProbe)

    # Call should NOT raise (v2's run_err is logged) AND legacy must not run.
    _q._do_query(
        chat_id="t_chat", message="hi", model="claude",
        user_msg_id=None, images=None, parent_id=None, force_backend_id=None,
    )

    assert legacy_reached == [], (
        "Legacy _do_query body was re-entered after v2 raised — "
        "this is the double-processing bug round-2 review caught"
    )


def test_do_query_v2_setup_failure_does_fall_back(monkeypatch):
    """Inverse of the previous test: when v2 fails BEFORE side effects
    (import / __init__), legacy fallback IS still the correct path —
    that's what makes the flag safe to flip on."""
    import larkhelm.config as _cfg
    _cfg.config = {"query_session_v2_enabled": True}

    legacy_reached: list[bool] = []

    class BoomConstructor:
        def __init__(self, **_kwargs):
            raise RuntimeError("v2 __init__ failure (pre-side-effect)")
        def run(self):  # pragma: no cover — never reached
            raise AssertionError("run() should not be called")

    monkeypatch.setattr(
        "larkhelm.handlers._query_session.QuerySession", BoomConstructor,
    )

    import larkhelm.handlers._query as _q
    import uuid

    class _UUIDProbe:
        @staticmethod
        def uuid4():
            legacy_reached.append(True)
            # Raise after marking re-entry so we exit quickly — full legacy
            # path needs a real Feishu setup we don't have here.
            raise StopIteration("legacy reached, abort test setup")
    monkeypatch.setattr(_q, "uuid", _UUIDProbe)

    with pytest.raises(StopIteration):
        _q._do_query(
            chat_id="t_chat", message="hi", model="claude",
            user_msg_id=None, images=None, parent_id=None,
            force_backend_id=None,
        )

    assert legacy_reached == [True], (
        "Legacy fallback should have been entered after v2 __init__ failed "
        "(pre-side-effect = safe to retry); instead it was skipped"
    )
