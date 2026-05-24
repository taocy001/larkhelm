"""Tests for AC-07: breakpoint auto-cancel after CREW_BREAKPOINT_TIMEOUT_SEC."""
from __future__ import annotations

import threading
import time
import uuid


def _make_breakpoint_state(fake_agent_spec):
    """Build a CrewState whose only agent has breakpoint=True."""
    from larkhelm.crew_types import AgentState, AgentStatus, CrewPlan, CrewState
    spec = fake_agent_spec(id="pm", task_profile="planner", breakpoint=True)
    plan = CrewPlan(title="bp test", agents=[spec], synthesis_prompt="")
    agents = {"pm": AgentState(spec=spec, status=AgentStatus.DONE)}
    return CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents=agents,
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="breakpoint",
        kind="dev",
        breakpoint_agent_id="pm",
    )


def test_breakpoint_timeout_sets_cancel_ev(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """AC-07: After CREW_BREAKPOINT_TIMEOUT_SEC seconds with no user click,
    cancel_ev should be set and the function should return False."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 2)

    from larkhelm.crew._runner import _wait_for_breakpoint
    state = _make_breakpoint_state(fake_agent_spec)
    t0 = time.time()
    confirmed = _wait_for_breakpoint(state, "pm")
    elapsed = time.time() - t0
    assert confirmed is False
    assert state.cancel_ev.is_set()
    # 2s deadline + up to 2s polling slack — generous upper bound
    assert elapsed < 7, f"breakpoint wait took too long: {elapsed:.1f}s"


def test_breakpoint_timeout_emits_card(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """The orange ⏳ card should reach the recorder."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 2)
    from larkhelm.crew._runner import _wait_for_breakpoint
    state = _make_breakpoint_state(fake_agent_spec)
    fake_card_sender.clear()
    _wait_for_breakpoint(state, "pm")
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert any("等待人工确认超时" in c["title"] for c in sends), (
        f"expected timeout card; got titles={[c['title'] for c in sends]}"
    )


def test_breakpoint_timeout_marks_phase_timeout(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """P2-3a (W4/W6): breakpoint auto-cancel now writes ``phase="timeout"``."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 2)
    from larkhelm.crew._runner import _wait_for_breakpoint
    state = _make_breakpoint_state(fake_agent_spec)
    _wait_for_breakpoint(state, "pm")
    assert state.phase == "timeout"


def test_breakpoint_user_confirm_short_circuits(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """If signal_breakpoint(confirmed=True) fires before deadline, return True."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 60)

    from larkhelm.crew._runner import _wait_for_breakpoint
    from larkhelm.crew._state import signal_breakpoint
    state = _make_breakpoint_state(fake_agent_spec)

    def _click_after(delay):
        time.sleep(delay)
        signal_breakpoint(state.crew_id, True)

    threading.Thread(target=_click_after, args=(0.3,), daemon=True).start()
    confirmed = _wait_for_breakpoint(state, "pm")
    assert confirmed is True
    assert not state.cancel_ev.is_set()


def test_breakpoint_cancel_event_short_circuits(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """If state.cancel_ev fires (e.g. /cancel button) before timeout, exit
    immediately with False."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 60)
    from larkhelm.crew._runner import _wait_for_breakpoint
    state = _make_breakpoint_state(fake_agent_spec)

    def _cancel_after(delay):
        time.sleep(delay)
        state.cancel_ev.set()

    threading.Thread(target=_cancel_after, args=(0.3,), daemon=True).start()
    t0 = time.time()
    confirmed = _wait_for_breakpoint(state, "pm")
    elapsed = time.time() - t0
    assert confirmed is False
    assert elapsed < 4


# ── Checkpoint cleanup after cancel/hardfail ─────────────────────────────
#
# Regression for the "ghost crew" bug: a breakpoint timeout (or any
# user-initiated cancel) used to leave ``crew_checkpoint.json`` on disk with
# ``phase="running"``, so the next bridge restart's ``resume_interrupted_crews``
# would happily pick the cancelled crew back up and run it through to the end.
# Fix lives in ``_run_crew``'s ``QueryCancelledError`` / ``HardFailError``
# branches: clear the checkpoint unless the bridge is mid-shutdown (where
# ``cancel_all_crews`` deliberately preserves the file for crash recovery).


def _drive_run_crew_through_exception(
    chat_id, cwd, exc_to_raise, fake_agent_spec, monkeypatch,
):
    """Drive ``_run_crew`` to its exception branch with a real on-disk
    checkpoint, then return whether the file still exists after the call."""
    from pathlib import Path
    import threading as _t
    from larkhelm.crew_types import (
        AgentState, AgentStatus, CrewPlan, CrewState,
    )

    # Point chat_state's cwd at our tmp dir so ``_clear_checkpoint`` can
    # resolve the right path.
    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda cid: str(cwd))

    # Seed the checkpoint file on disk so we can prove it gets removed
    # (or kept) by the exception handler.
    ws = Path(cwd) / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    cp_path = ws / "crew_checkpoint.json"
    cp_path.write_text('{"version":1,"phase":"running","crew_id":"x"}',
                       encoding="utf-8")
    assert cp_path.exists()

    # Build a 1-agent CrewState.
    spec = fake_agent_spec(id="pm", task_profile="planner")
    plan = CrewPlan(title="cp-clear test", agents=[spec], synthesis_prompt="")
    state = CrewState(
        crew_id="cidcptest", chat_id=chat_id, plan=plan,
        agents={"pm": AgentState(spec=spec, status=AgentStatus.PENDING)},
        card_mid="mid_fake", cancel_ev=_t.Event(),
        phase="running", kind="dev",
    )

    # Make ``_execute`` immediately raise — that's the only path we care about.
    def _boom(_state, _timeout):
        raise exc_to_raise
    monkeypatch.setattr("larkhelm.crew._runner._execute", _boom)

    # Avoid touching Feishu / cards / heartbeat.
    monkeypatch.setattr("larkhelm.crew._runner._start_heartbeat",
                        lambda *a, **kw: _t.Thread(target=lambda: None))
    monkeypatch.setattr("larkhelm.crew_card._crew_update_card",
                        lambda *a, **kw: None)

    from larkhelm.crew._runner import _run_crew
    _run_crew(state, total_timeout=60)
    return cp_path.exists()


def test_query_cancelled_clears_checkpoint_when_not_shutting_down(
    init_test_config, fake_agent_spec, fake_card_sender, tmp_path,
    monkeypatch,
):
    """Regression: breakpoint-timeout (and any other QueryCancelledError
    path) must clear ``crew_checkpoint.json`` so the next bridge restart
    does NOT auto-resume the cancelled crew."""
    from larkhelm.ai_runner import QueryCancelledError
    from larkhelm.concurrency import _shutting_down  # noqa: F401  (introspection)

    # Make sure we're not flagged as shutting down (rest of the suite may
    # have flipped this true once and never reset).
    import larkhelm.concurrency as _conc
    monkeypatch.setattr(_conc, "_shutting_down", False)

    still_exists = _drive_run_crew_through_exception(
        chat_id="chat_cp_cancel",
        cwd=tmp_path,
        exc_to_raise=QueryCancelledError("user cancelled"),
        fake_agent_spec=fake_agent_spec,
        monkeypatch=monkeypatch,
    )
    assert still_exists is False, (
        "QueryCancelledError handler must clear the checkpoint when the "
        "bridge is NOT shutting down — otherwise a ghost crew gets revived "
        "by resume_interrupted_crews on the next restart"
    )


def test_query_cancelled_preserves_checkpoint_during_shutdown(
    init_test_config, fake_agent_spec, fake_card_sender, tmp_path,
    monkeypatch,
):
    """During SIGTERM, ``cancel_all_crews`` deliberately re-saves the
    checkpoint with ``phase="running"`` so resume_interrupted_crews can pick
    up after restart. The QueryCancelledError handler must NOT undo that."""
    from larkhelm.ai_runner import QueryCancelledError

    import larkhelm.concurrency as _conc
    monkeypatch.setattr(_conc, "_shutting_down", True)

    still_exists = _drive_run_crew_through_exception(
        chat_id="chat_cp_shutdown",
        cwd=tmp_path,
        exc_to_raise=QueryCancelledError("sigterm cancel"),
        fake_agent_spec=fake_agent_spec,
        monkeypatch=monkeypatch,
    )
    assert still_exists is True, (
        "During shutdown, checkpoint must be preserved so the next bridge "
        "restart can resume_interrupted_crews picks up where we left off"
    )


def test_hard_fail_clears_checkpoint(
    init_test_config, fake_agent_spec, fake_card_sender, tmp_path,
    monkeypatch,
):
    """Hard-fail is terminal — checkpoint must be cleared regardless of
    shutdown state. (We're explicitly choosing not-shutting-down here; the
    behavior is the same either way.)"""
    from larkhelm.crew_types import HardFailError

    import larkhelm.concurrency as _conc
    monkeypatch.setattr(_conc, "_shutting_down", False)

    still_exists = _drive_run_crew_through_exception(
        chat_id="chat_cp_hardfail",
        cwd=tmp_path,
        exc_to_raise=HardFailError("qa blew up"),
        fake_agent_spec=fake_agent_spec,
        monkeypatch=monkeypatch,
    )
    assert still_exists is False, (
        "HardFailError handler must clear the checkpoint — a hard fail "
        "is terminal, the crew can't meaningfully resume from it"
    )
