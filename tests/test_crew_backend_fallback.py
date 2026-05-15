"""Tests for AC-06: claude fully disabled → /dev still completes via fallback."""
from __future__ import annotations


def _disable_claude(reg):
    """Disable every claude variant in the registry so the resolver must
    pick another backend."""
    for sid, spec in list(reg._specs.items()):
        if "claude" in sid.lower() or spec.provider == "claude_cli":
            spec.enabled = False


def test_with_claude_disabled_resolver_picks_kimi(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend
    _disable_claude(fake_backend_registry)
    spec = fake_agent_spec(task_profile="engineer")
    out = resolve_backend(spec)
    assert out.id != "claude"
    # kimi has next-best coding (0.85) and tools tag
    assert out.id == "kimi"


def test_with_claude_disabled_qa_falls_back(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """qa profile requires tools — only claude + kimi qualify; claude disabled
    → kimi wins."""
    from larkhelm.crew._backend_resolver import resolve_backend
    _disable_claude(fake_backend_registry)
    spec = fake_agent_spec(task_profile="qa")
    out = resolve_backend(spec)
    assert out.id == "kimi"


def test_with_claude_disabled_chat_picks_deepseek_or_kimi(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """chat profile has no require_tools — deepseek (chat=0.85) ties with kimi."""
    from larkhelm.crew._backend_resolver import resolve_backend
    _disable_claude(fake_backend_registry)
    spec = fake_agent_spec(task_profile="chat")
    out = resolve_backend(spec)
    assert out.id in {"deepseek", "kimi"}


def test_pipeline_runs_through_with_claude_disabled(
    init_test_config, fake_crew_state, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """AC-06: The full /dev pipeline (6 agents) should be runnable when claude
    is disabled — every _run_agent call resolves to a non-claude backend and
    no NoBackendAvailableError surfaces."""
    from larkhelm.crew._runner import _run_agent_wrapper
    from larkhelm.crew._pipeline import _make_dev_pipeline
    from larkhelm.crew_types import (
        AgentState, AgentStatus, CrewState,
    )
    import threading
    import uuid as _uuid

    _disable_claude(fake_backend_registry)
    plan = _make_dev_pipeline("requirement", "/tmp/cwd", no_confirm=True,
                              skip_planning=False)
    # Skip the implementer's huge timeout — we just need it not to block
    state = CrewState(
        crew_id=_uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents={s.id: AgentState(spec=s) for s in plan.agents},
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="dev",
    )

    routed_backends = []
    def fake(state_, agent_id):
        from larkhelm.crew._backend_resolver import resolve_backend
        spec = state_.agents[agent_id].spec
        chosen = resolve_backend(spec)
        routed_backends.append(chosen.id)
        return f"output of {agent_id}"

    mock_run_agent(fake)

    # Run wrappers serially (not threaded — we just want resolution coverage)
    for spec in plan.agents:
        _run_agent_wrapper(state, spec.id)

    assert len(routed_backends) == 6
    # No backend should be claude
    assert all(bid != "claude" for bid in routed_backends), (
        f"unexpected claude routing: {routed_backends}"
    )
    # All 6 agents should be DONE
    for ag in state.agents.values():
        assert ag.status == AgentStatus.DONE, (
            f"agent {ag.spec.id} status={ag.status}, error={ag.error}"
        )


def test_resolver_never_returns_disabled_spec(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend
    fake_backend_registry._specs["kimi"].enabled = False
    fake_backend_registry._specs["deepseek"].enabled = False
    spec = fake_agent_spec(task_profile="engineer")
    out = resolve_backend(spec)
    assert out.enabled is True
