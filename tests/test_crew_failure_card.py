"""Tests for AC-03: ⚠️ failure cards (``crew/_failure_card.py``)."""
from __future__ import annotations


# ── emit_agent_failure ──────────────────────────────────────────────

def test_emit_agent_failure_writes_redacted_error(
    init_test_config, fake_crew_state, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_agent_failure
    state = fake_crew_state(["alpha"])
    emit_agent_failure(state, "alpha", "run",
                       RuntimeError("api_key=mysecret123 went bad"))
    err = state.agents["alpha"].error
    assert "mysecret123" not in err
    assert "***" in err


def test_emit_agent_failure_marks_status_failed(
    init_test_config, fake_crew_state, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_agent_failure
    from larkhelm.crew_types import AgentStatus
    state = fake_crew_state(["alpha"])
    emit_agent_failure(state, "alpha", "run", RuntimeError("boom"))
    assert state.agents["alpha"].status == AgentStatus.FAILED


def test_emit_agent_failure_does_not_demote_done(
    init_test_config, fake_crew_state, fake_card_sender,
):
    """If the agent already finished DONE/CANCELLED, don't flip back to FAILED."""
    from larkhelm.crew._failure_card import emit_agent_failure
    from larkhelm.crew_types import AgentStatus
    state = fake_crew_state(["alpha"])
    state.agents["alpha"].status = AgentStatus.DONE
    emit_agent_failure(state, "alpha", "run", RuntimeError("late error"))
    assert state.agents["alpha"].status == AgentStatus.DONE


def test_emit_agent_failure_oom_uses_friendly_prefix(
    init_test_config, fake_crew_state, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_agent_failure
    state = fake_crew_state(["alpha"])
    emit_agent_failure(state, "alpha", "run",
                       RuntimeError("killed by OS (rc=-9)"))
    assert "内存超限" in state.agents["alpha"].error


def test_emit_agent_failure_backend_select_stage(
    init_test_config, fake_crew_state, fake_card_sender,
):
    """stage='backend_select' label appears in the error text."""
    from larkhelm.crew._failure_card import emit_agent_failure
    from larkhelm.crew_types import NoBackendAvailableError
    state = fake_crew_state(["alpha"])
    emit_agent_failure(state, "alpha", "backend_select",
                       NoBackendAvailableError("engineer", "no kimi"))
    assert "backend_select" in state.agents["alpha"].error


def test_emit_agent_failure_pushes_card_update(
    init_test_config, fake_crew_state, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_agent_failure
    state = fake_crew_state(["alpha"])
    fake_card_sender.clear()
    emit_agent_failure(state, "alpha", "run", RuntimeError("oops"))
    kinds = [c["kind"] for c in fake_card_sender]
    assert "crew_update_card" in kinds


def test_emit_agent_failure_never_raises_on_lark_error(
    init_test_config, fake_crew_state, monkeypatch,
):
    """Network failures during card push must be swallowed."""
    from larkhelm.crew._failure_card import emit_agent_failure
    import larkhelm.crew._failure_card as fc
    def _boom(state):
        raise RuntimeError("lark API down")
    monkeypatch.setattr(fc, "_crew_update_card", _boom)
    state = fake_crew_state(["alpha"])
    # Should not raise
    emit_agent_failure(state, "alpha", "run", RuntimeError("primary failure"))


# ── emit_terminal_failure ───────────────────────────────────────────

def test_emit_terminal_failure_sends_red_card(
    init_test_config, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_terminal_failure
    fake_card_sender.clear()
    emit_terminal_failure("test_chat", kind="dev",
                          reason="disk full", exc=OSError("ENOSPC"))
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends
    assert sends[0]["color"] == "red"
    assert "/dev" in sends[0]["title"]


def test_emit_terminal_failure_includes_status_hint(
    init_test_config, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_terminal_failure
    fake_card_sender.clear()
    emit_terminal_failure("test_chat", kind="crew",
                          reason="x", exc=RuntimeError("y"))
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert "/status" in sends[0]["body"]


def test_emit_terminal_failure_handles_kind_crew(
    init_test_config, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_terminal_failure
    fake_card_sender.clear()
    emit_terminal_failure("test_chat", kind="crew", reason="z")
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert "/crew" in sends[0]["title"]


def test_emit_terminal_failure_no_exc(
    init_test_config, fake_card_sender,
):
    """exc=None still produces a card."""
    from larkhelm.crew._failure_card import emit_terminal_failure
    fake_card_sender.clear()
    emit_terminal_failure("test_chat", kind="dev", reason="oh no")
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends


def test_emit_terminal_failure_redacts_exc(
    init_test_config, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_terminal_failure
    fake_card_sender.clear()
    emit_terminal_failure("test_chat", kind="dev", reason="auth failed",
                          exc=RuntimeError("Authorization: Bearer tok-secret-123"))
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    body = sends[0]["body"]
    assert "tok-secret-123" not in body
    assert "***" in body


# ── emit_breakpoint_timeout ─────────────────────────────────────────

def test_emit_breakpoint_timeout_sets_phase_cancelled(
    init_test_config, fake_crew_state, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_breakpoint_timeout
    state = fake_crew_state(["pm"])
    state.breakpoint_agent_id = "pm"
    emit_breakpoint_timeout(state)
    assert state.phase == "cancelled"


def test_emit_breakpoint_timeout_writes_agent_error(
    init_test_config, fake_crew_state, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_breakpoint_timeout
    state = fake_crew_state(["pm"])
    state.breakpoint_agent_id = "pm"
    emit_breakpoint_timeout(state)
    assert "等待人工确认超时" in state.agents["pm"].error


def test_emit_breakpoint_timeout_sends_orange_card(
    init_test_config, fake_crew_state, fake_card_sender,
):
    from larkhelm.crew._failure_card import emit_breakpoint_timeout
    state = fake_crew_state(["pm"])
    state.breakpoint_agent_id = "pm"
    fake_card_sender.clear()
    emit_breakpoint_timeout(state)
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends
    assert sends[0]["color"] == "orange"


def test_emit_breakpoint_timeout_no_breakpoint_agent(
    init_test_config, fake_crew_state, fake_card_sender,
):
    """If breakpoint_agent_id is empty, function still doesn't crash."""
    from larkhelm.crew._failure_card import emit_breakpoint_timeout
    state = fake_crew_state(["pm"])
    state.breakpoint_agent_id = ""
    emit_breakpoint_timeout(state)
    assert state.phase == "cancelled"
