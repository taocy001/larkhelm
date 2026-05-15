"""Tests for AC-04: ``NoBackendAvailableError`` does not crash the runner."""
from __future__ import annotations

import pytest


def test_resolver_raises_with_empty_registry(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend
    from larkhelm.crew_types import NoBackendAvailableError
    fake_backend_registry._specs.clear()
    with pytest.raises(NoBackendAvailableError):
        resolve_backend(fake_agent_spec(task_profile="qa"))


def test_run_agent_wrapper_emits_failure_card(
    init_test_config, fake_crew_state, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """When _run_agent raises NoBackendAvailableError, _run_agent_wrapper
    should emit a ⚠️ card with stage='backend_select'."""
    from larkhelm.crew._runner import _run_agent_wrapper
    from larkhelm.crew_types import NoBackendAvailableError, AgentStatus

    def fake(state, agent_id):
        raise NoBackendAvailableError("engineer", "no kimi available")

    mock_run_agent(fake)
    state = fake_crew_state(["alpha"], task_profile="engineer")
    fake_card_sender.clear()
    _run_agent_wrapper(state, "alpha")

    assert state.agents["alpha"].status == AgentStatus.FAILED
    err = state.agents["alpha"].error
    # Stage tag is in the error string
    assert "backend_select" in err
    # Card update fired
    assert any(c["kind"] == "crew_update_card" for c in fake_card_sender)


def test_no_backend_error_carries_metadata():
    from larkhelm.crew_types import NoBackendAvailableError
    e = NoBackendAvailableError("planner", "registry empty")
    assert e.task_profile == "planner"
    assert e.reason == "registry empty"
    assert "planner" in str(e)


def test_run_agent_wrapper_does_not_retry_no_backend(
    init_test_config, fake_crew_state, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """NoBackendAvailableError is non-retryable — _run_agent should be called
    exactly once, not twice."""
    from larkhelm.crew._runner import _run_agent_wrapper
    from larkhelm.crew_types import NoBackendAvailableError

    call_count = [0]
    def fake(state, agent_id):
        call_count[0] += 1
        raise NoBackendAvailableError("engineer", "none")

    mock_run_agent(fake)
    state = fake_crew_state(["alpha"], task_profile="engineer")
    _run_agent_wrapper(state, "alpha")
    assert call_count[0] == 1
