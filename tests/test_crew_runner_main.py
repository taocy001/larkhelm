"""Tests for AC-02 / AC-01: ``crew/_runner._execute`` main paths."""
from __future__ import annotations

import threading
import uuid

import pytest


def _make_state(plan_specs):
    from larkhelm.crew_types import AgentState, CrewPlan, CrewState
    plan = CrewPlan(title="t", agents=plan_specs, synthesis_prompt="")
    agents = {s.id: AgentState(spec=s) for s in plan_specs}
    return CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents=agents,
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="crew",
    )


def test_execute_runs_topo_waves_in_order(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    from larkhelm.crew._runner import _execute
    specs = [
        fake_agent_spec(id="a", depends_on=[], task_profile="engineer"),
        fake_agent_spec(id="b", depends_on=["a"], task_profile="engineer"),
        fake_agent_spec(id="c", depends_on=["b"], task_profile="engineer"),
    ]
    state = _make_state(specs)
    order: list[str] = []
    def fake(s, aid):
        order.append(aid)
        return f"result {aid}"
    mock_run_agent(fake)
    _execute(state, total_timeout=60)
    assert order == ["a", "b", "c"]


def test_execute_skips_trigger_only_first_wave(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    from larkhelm.crew._runner import _execute
    from larkhelm.crew_types import AgentStatus
    specs = [
        fake_agent_spec(id="a", depends_on=[], task_profile="engineer"),
        fake_agent_spec(id="b", depends_on=["a"], trigger_only=True,
                        task_profile="engineer"),
    ]
    state = _make_state(specs)
    called: list[str] = []
    def fake(s, aid):
        called.append(aid)
        return "ok"
    mock_run_agent(fake)
    _execute(state, total_timeout=60)
    assert called == ["a"]   # b skipped (trigger_only)
    # b is marked DONE with empty result (sentinel)
    assert state.agents["b"].status == AgentStatus.DONE
    assert state.agents["b"].result == ""


def test_execute_total_timeout_fails_remaining(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """Negative remaining-time → outstanding agents flipped to FAILED."""
    from larkhelm.crew._runner import _execute
    from larkhelm.crew_types import AgentStatus
    specs = [fake_agent_spec(id="x", task_profile="engineer")]
    state = _make_state(specs)
    # No mock_run_agent install needed — we'll set deadline before any run
    def fake(s, aid):
        return "ok"
    mock_run_agent(fake)
    _execute(state, total_timeout=-1)   # already past deadline
    assert state.agents["x"].status == AgentStatus.FAILED


def test_execute_cancellation_propagates(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    from larkhelm.crew._runner import _execute
    from larkhelm.ai_runner import QueryCancelledError
    specs = [fake_agent_spec(id="x", task_profile="engineer")]
    state = _make_state(specs)
    state.cancel_ev.set()
    def fake(s, aid):
        return "ok"
    mock_run_agent(fake)
    with pytest.raises(QueryCancelledError):
        _execute(state, total_timeout=60)


def test_execute_agent_failure_emits_card(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    from larkhelm.crew._runner import _execute
    from larkhelm.crew_types import AgentStatus
    specs = [fake_agent_spec(id="boom", task_profile="engineer")]
    state = _make_state(specs)
    def fake(s, aid):
        raise RuntimeError("planned failure")
    mock_run_agent(fake)
    _execute(state, total_timeout=60)
    assert state.agents["boom"].status == AgentStatus.FAILED
    # The error string carries the run-stage prefix (set by emit_agent_failure)
    assert "planned failure" in state.agents["boom"].error or "run" in state.agents["boom"].error


def test_execute_retry_path(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """fail_marker triggered → agent re-runs once via retry_target rules."""
    from larkhelm.crew._runner import _execute
    specs = [
        fake_agent_spec(id="qa", depends_on=[], task_profile="qa",
                        exit_marker="OK", fail_marker="FAIL",
                        retry_target=[], max_retries=1),
    ]
    state = _make_state(specs)
    call_outputs = ["FAIL", "OK"]
    call_count = [0]
    def fake(s, aid):
        out = call_outputs[call_count[0] % 2]
        call_count[0] += 1
        return out
    mock_run_agent(fake)
    _execute(state, total_timeout=60)
    assert call_count[0] == 2  # initial + 1 retry


def test_execute_handles_empty_runnable_wave(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """Wave whose agents all skipped/failed should not crash _execute."""
    from larkhelm.crew._runner import _execute
    from larkhelm.crew_types import AgentStatus
    specs = [
        fake_agent_spec(id="a", depends_on=[], task_profile="engineer"),
        fake_agent_spec(id="b", depends_on=["a"], task_profile="engineer"),
    ]
    state = _make_state(specs)
    # Pre-mark a as failed so b's wave will be empty (b skipped)
    def fake(s, aid):
        if aid == "a":
            raise RuntimeError("a fails")
        return "b"
    mock_run_agent(fake)
    _execute(state, total_timeout=60)
    assert state.agents["a"].status == AgentStatus.FAILED
    # b should have been skipped / marked failed for upstream reason
    assert state.agents["b"].status == AgentStatus.FAILED
