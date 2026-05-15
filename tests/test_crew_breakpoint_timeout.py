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


def test_breakpoint_timeout_marks_phase_cancelled(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 2)
    from larkhelm.crew._runner import _wait_for_breakpoint
    state = _make_breakpoint_state(fake_agent_spec)
    _wait_for_breakpoint(state, "pm")
    assert state.phase == "cancelled"


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
