"""Tests for AC-02: cover ``crew/_runner._run_agent`` dispatch branches,
``_synthesize``, ``_execute_from``, ``_run_crew`` and ``_write_hermes_summary``.

These tests stub out the backend functions at module level so each branch
of ``_run_agent`` is exercised end-to-end without spawning real subprocesses.
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_state(plan_specs, *, chat_id: str = "test_chat",
                kind: str = "dev", synthesis_prompt: str = ""):
    from larkhelm.crew_types import AgentState, CrewPlan, CrewState
    plan = CrewPlan(title="t", agents=plan_specs,
                    synthesis_prompt=synthesis_prompt)
    agents = {s.id: AgentState(spec=s) for s in plan_specs}
    return CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id=chat_id,
        plan=plan,
        agents=agents,
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind=kind,
    )


def _stub_perm_and_memory(monkeypatch):
    """Silence permission grant/revoke and memory context — _run_agent
    calls these unconditionally; tests don't care about their side effects."""
    import larkhelm.perm as _perm
    monkeypatch.setattr(_perm, "grant_yolo", lambda ns: None, raising=False)
    monkeypatch.setattr(_perm, "revoke_yolo", lambda ns: None, raising=False)
    import larkhelm.memory as _mem
    monkeypatch.setattr(_mem, "get_memory_context_v2",
                        lambda *a, **k: ("", {}), raising=False)


def _force_resolved_backend(monkeypatch, *, provider: str, sid: str = "fake"):
    """Patch ``crew._runner.resolve_backend`` to return a synthetic
    BackendSpec whose ``provider`` drives the dispatch branch under test."""
    from larkhelm.backend_registry import BackendSpec
    spec = BackendSpec(
        id=sid, provider=provider, display_name=f"fake-{sid}",
        role="orchestrator", tags=["tools"], command="fake",
        healthy=True, enabled=True,
    )
    import larkhelm.crew._runner as _rn
    monkeypatch.setattr(_rn, "resolve_backend", lambda s, **kw: spec)
    return spec


# ─────────────────────────────────────────────────────────────────────────
#  _run_agent dispatch branches
# ─────────────────────────────────────────────────────────────────────────

def test_run_agent_gemini_branch(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """Resolver returns provider=gemini_cli → ``query_gemini`` is called."""
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="gemini_cli", sid="gemini")

    captured = {}
    def fake_gemini(**kwargs):
        captured.update(kwargs)
        return "gemini output"
    import larkhelm.ai_runner as _ar
    monkeypatch.setattr(_ar, "query_gemini", fake_gemini)

    spec = fake_agent_spec(id="g", task_profile="planner")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    out = _run_agent(state, "g")
    assert out == "gemini output"
    assert captured["chat_id"].startswith("test_chat__crew_")
    assert captured["use_session"] is False


def test_run_agent_kimi_branch(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="kimi_cli", sid="kimi")

    import larkhelm.ai_runner as _ar
    monkeypatch.setattr(_ar, "query_kimi", lambda **kw: "kimi output")

    spec = fake_agent_spec(id="k", task_profile="engineer")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    assert _run_agent(state, "k") == "kimi output"


def test_run_agent_deepseek_branch(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="deepseek_api", sid="deepseek")

    import larkhelm.ai_runner as _ar
    monkeypatch.setattr(_ar, "query_deepseek", lambda **kw: "deepseek output",
                        raising=False)

    spec = fake_agent_spec(id="d", task_profile="chat")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    assert _run_agent(state, "d") == "deepseek output"


def test_run_agent_hermes_branch(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """provider=hermes synthetic spec → ``_run_hermes_orchestrator`` is called."""
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="hermes", sid="hermes_race")

    captured = {}
    def fake_hermes(**kwargs):
        captured.update(kwargs)
        return "hermes output"
    import larkhelm.crew._hermes_orchestrator as _ho
    monkeypatch.setattr(_ho, "_run_hermes_orchestrator", fake_hermes,
                        raising=False)

    spec = fake_agent_spec(id="h", model="hermes_race", task_profile="")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    out = _run_agent(state, "h")
    assert out == "hermes output"
    assert captured["agent_id"] == "h"
    # Hermes summary markdown file written alongside result
    import larkhelm.config as _cfg
    summary = _cfg.SESSION_DIR / state.chat_id / f"crew_{state.crew_id}" / "h_summary.md"
    assert summary.exists()
    body = summary.read_text(encoding="utf-8")
    assert "hermes output" in body and "race" in body


def test_run_agent_orchestrator_claude_cli_branch(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """provider=claude_cli falls into the else-branch backend_cli.run_claude."""
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="claude_cli", sid="claude")

    captured = {}
    def fake_claude(**kwargs):
        captured.update(kwargs)
        kwargs.get("on_start", lambda: None)()  # exercise the semaphore callback
        return "claude output"
    import larkhelm.backend_cli as _bc
    monkeypatch.setattr(_bc, "run_claude", fake_claude)

    spec = fake_agent_spec(id="c", task_profile="reviewer")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    out = _run_agent(state, "c")
    assert out == "claude output"
    assert captured["allow_retry"] is False


def test_run_agent_orchestrator_gemini_cli_branch(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """resolved.provider=gemini_cli at the else-branch (no profile match)."""
    _stub_perm_and_memory(monkeypatch)
    # Note _disp_kind already routes gemini_cli to "gemini" earlier; cover the
    # else-branch by simulating an unknown provider that still routes through
    # _spec.provider == "gemini_cli" mapping inside the else clause. Use the
    # legacy model path: model="" + profile="" → resolver-default fallback.
    from larkhelm.backend_registry import BackendSpec
    fake_spec = BackendSpec(
        id="x", provider="gemini_cli", display_name="fake-g",
        role="orchestrator", tags=["tools"], command="gemini",
        healthy=True, enabled=True,
    )
    # Force _disp_kind="orchestrator" by patching resolve_backend to return a
    # provider not matched by the earlier `if/elif` chain. Easiest: set
    # provider to something exotic so _disp_kind falls to "orchestrator", then
    # let the inner else dispatch check provider directly. But the only path
    # that goes there is provider not in {hermes, gemini_cli, kimi_cli,
    # deepseek_api}. So switch to anthropic_api covering the API branch.
    fake_spec.provider = "anthropic_api"
    import larkhelm.crew._runner as _rn
    monkeypatch.setattr(_rn, "resolve_backend", lambda s, **kw: fake_spec)

    import larkhelm.backend_api as _bapi
    monkeypatch.setattr(_bapi, "run_anthropic",
                        lambda **kw: ("anthropic output", {"in": 1}))

    spec = fake_agent_spec(id="a", task_profile="planner")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    out = _run_agent(state, "a")
    assert out == "anthropic output"


def test_run_agent_writes_result_file(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """Successful run persists ``{agent_id}_result.txt`` in the crew workspace."""
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="kimi_cli", sid="kimi")
    import larkhelm.ai_runner as _ar
    monkeypatch.setattr(_ar, "query_kimi", lambda **kw: "  payload  ")
    spec = fake_agent_spec(id="rf", task_profile="engineer")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    out = _run_agent(state, "rf")
    assert out == "payload"
    import larkhelm.config as _cfg
    ws = _cfg.SESSION_DIR / state.chat_id / f"crew_{state.crew_id}"
    assert (ws / "rf_result.txt").read_text(encoding="utf-8") == "payload"


def test_run_agent_empty_output_falls_back_to_sentinel(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """``output.strip() or "（无输出）"`` branch when backend returns empty."""
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="kimi_cli", sid="kimi")
    import larkhelm.ai_runner as _ar
    monkeypatch.setattr(_ar, "query_kimi", lambda **kw: "")
    spec = fake_agent_spec(id="empty", task_profile="engineer")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    assert _run_agent(state, "empty") == "（无输出）"


def test_run_agent_propagates_no_backend(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """NoBackendAvailableError raised by resolver propagates out (no retry)."""
    from larkhelm.crew_types import NoBackendAvailableError
    _stub_perm_and_memory(monkeypatch)

    def _raise(spec, **kw):
        raise NoBackendAvailableError(spec.task_profile, "all disabled")
    import larkhelm.crew._runner as _rn
    monkeypatch.setattr(_rn, "resolve_backend", _raise)

    spec = fake_agent_spec(id="nb", task_profile="engineer")
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    with pytest.raises(NoBackendAvailableError):
        _run_agent(state, "nb")


def test_run_agent_per_agent_timeout_raises_runtime_error(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """When the backend raises QueryCancelledError without crew-level cancel,
    _run_agent re-raises as RuntimeError(timed out)."""
    from larkhelm.ai_runner import QueryCancelledError
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="kimi_cli", sid="kimi")

    def _fail(**kw):
        raise QueryCancelledError("timeout")
    import larkhelm.ai_runner as _ar
    monkeypatch.setattr(_ar, "query_kimi", _fail)

    spec = fake_agent_spec(id="t", task_profile="engineer", timeout=1)
    state = _make_state([spec])
    from larkhelm.crew._runner import _run_agent
    with pytest.raises(RuntimeError, match="timed out"):
        _run_agent(state, "t")


def test_run_agent_propagates_cancellation(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """When crew-level cancel is set, QueryCancelledError propagates as-is."""
    from larkhelm.ai_runner import QueryCancelledError
    _stub_perm_and_memory(monkeypatch)
    _force_resolved_backend(monkeypatch, provider="kimi_cli", sid="kimi")

    spec = fake_agent_spec(id="c", task_profile="engineer", timeout=1)
    state = _make_state([spec])
    state.cancel_ev.set()

    def _fail(**kw):
        raise QueryCancelledError("cancelled")
    import larkhelm.ai_runner as _ar
    monkeypatch.setattr(_ar, "query_kimi", _fail)

    from larkhelm.crew._runner import _run_agent
    with pytest.raises(QueryCancelledError):
        _run_agent(state, "c")


# ─────────────────────────────────────────────────────────────────────────
#  _synthesize branches
# ─────────────────────────────────────────────────────────────────────────

def test_synthesize_single_agent_no_prompt_returns_directly(
    init_test_config, fake_agent_spec, fake_card_sender,
):
    """No synthesis_prompt + exactly one DONE agent → return its result."""
    from larkhelm.crew_types import AgentStatus
    spec = fake_agent_spec(id="solo", task_profile="engineer")
    state = _make_state([spec], synthesis_prompt="")
    state.agents["solo"].status = AgentStatus.DONE
    state.agents["solo"].result = "the only result"
    from larkhelm.crew._runner import _synthesize
    assert _synthesize(state) == "the only result"


def test_synthesize_single_agent_failed_returns_error(
    init_test_config, fake_agent_spec, fake_card_sender,
):
    from larkhelm.crew_types import AgentStatus
    spec = fake_agent_spec(id="solo", task_profile="engineer")
    state = _make_state([spec], synthesis_prompt="")
    state.agents["solo"].status = AgentStatus.FAILED
    state.agents["solo"].error = "boom"
    from larkhelm.crew._runner import _synthesize
    assert _synthesize(state) == "boom"


def test_synthesize_multi_agent_uses_orchestrator(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, monkeypatch,
):
    """Multi-agent synth runs the orchestrator (claude_cli branch)."""
    from larkhelm.crew_types import AgentStatus
    specs = [
        fake_agent_spec(id="a", task_profile="engineer"),
        fake_agent_spec(id="b", task_profile="engineer"),
    ]
    state = _make_state(specs, synthesis_prompt="Synthesize:")
    state.agents["a"].status = AgentStatus.DONE
    state.agents["a"].result = "result a"
    state.agents["b"].status = AgentStatus.DONE
    state.agents["b"].result = "result b"

    captured = {}
    def fake_claude(**kwargs):
        captured.update(kwargs)
        return "synthesized"
    import larkhelm.backend_cli as _bc
    monkeypatch.setattr(_bc, "run_claude", fake_claude)
    import larkhelm.perm as _perm
    monkeypatch.setattr(_perm, "grant_yolo", lambda ns: None, raising=False)
    monkeypatch.setattr(_perm, "revoke_yolo", lambda ns: None, raising=False)

    from larkhelm.crew._runner import _synthesize
    assert _synthesize(state) == "synthesized"
    # Both agents' previews appear in the prompt body
    msg = captured["message"]
    assert "result a" in msg and "result b" in msg
    assert "Synthesize:" in msg


def test_synthesize_includes_failed_partial_output(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, monkeypatch,
):
    """FAILED agents with partial result land in the synthesis prompt with
    the ``Incomplete execution`` annotation."""
    from larkhelm.crew_types import AgentStatus
    specs = [
        fake_agent_spec(id="ok", task_profile="engineer"),
        fake_agent_spec(id="bad", task_profile="engineer"),
    ]
    state = _make_state(specs, synthesis_prompt="Summarize:")
    state.agents["ok"].status = AgentStatus.DONE
    state.agents["ok"].result = "good stuff"
    state.agents["bad"].status = AgentStatus.FAILED
    state.agents["bad"].result = "partial output before crash"

    captured = {}
    def fake_claude(**kwargs):
        captured.update(kwargs)
        return "ok"
    import larkhelm.backend_cli as _bc
    monkeypatch.setattr(_bc, "run_claude", fake_claude)
    import larkhelm.perm as _perm
    monkeypatch.setattr(_perm, "grant_yolo", lambda ns: None, raising=False)
    monkeypatch.setattr(_perm, "revoke_yolo", lambda ns: None, raising=False)

    from larkhelm.crew._runner import _synthesize
    _synthesize(state)
    assert "Incomplete execution" in captured["message"]
    assert "partial output before crash" in captured["message"]


def test_synthesize_no_orchestrator_raises(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry,
):
    """When no orchestrator backend is available, raise RuntimeError."""
    # Disable every orchestrator in the registry
    for spec in fake_backend_registry._specs.values():
        if spec.role == "orchestrator":
            spec.enabled = False

    from larkhelm.crew_types import AgentStatus
    specs = [
        fake_agent_spec(id="a", task_profile="engineer"),
        fake_agent_spec(id="b", task_profile="engineer"),
    ]
    state = _make_state(specs, synthesis_prompt="Synthesize:")
    for aid in ("a", "b"):
        state.agents[aid].status = AgentStatus.DONE
        state.agents[aid].result = "x"

    from larkhelm.crew._runner import _synthesize
    with pytest.raises(RuntimeError, match="No orchestrator"):
        _synthesize(state)


def test_synthesize_returns_fallback_when_no_parts(
    init_test_config, fake_agent_spec, fake_card_sender,
):
    """All agents still PENDING → no parts → return canned fallback string."""
    spec = fake_agent_spec(id="solo", task_profile="engineer")
    state = _make_state([spec], synthesis_prompt="Synthesize me")
    # status stays PENDING → parts is empty → fallback path
    from larkhelm.crew._runner import _synthesize
    assert _synthesize(state) == "All agents produced no results."


# ─────────────────────────────────────────────────────────────────────────
#  _execute_from (resume path)
# ─────────────────────────────────────────────────────────────────────────

def test_execute_from_skips_already_done_agents(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """skip_ids agents already in DONE state are not re-run."""
    from larkhelm.crew_types import AgentStatus
    specs = [
        fake_agent_spec(id="done1", depends_on=[], task_profile="engineer"),
        fake_agent_spec(id="todo", depends_on=["done1"], task_profile="engineer"),
    ]
    state = _make_state(specs)
    state.agents["done1"].status = AgentStatus.DONE
    state.agents["done1"].result = "preserved"

    called: list[str] = []
    def fake(s, aid):
        called.append(aid)
        return f"new {aid}"
    mock_run_agent(fake)

    from larkhelm.crew._runner import _execute_from
    _execute_from(state, total_timeout=60, skip_ids={"done1"})
    assert called == ["todo"]
    # Preserved DONE state was not overwritten
    assert state.agents["done1"].result == "preserved"


def test_execute_from_respects_deadline(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    """Negative timeout breaks out of the wave loop without running anything."""
    specs = [fake_agent_spec(id="x", task_profile="engineer")]
    state = _make_state(specs)
    called: list[str] = []
    def fake(s, aid):
        called.append(aid)
        return "ok"
    mock_run_agent(fake)
    from larkhelm.crew._runner import _execute_from
    _execute_from(state, total_timeout=-1, skip_ids=set())
    assert called == []


def test_execute_from_propagates_cancellation(
    init_test_config, fake_agent_spec, fake_card_sender,
    fake_backend_registry, mock_run_agent,
):
    from larkhelm.ai_runner import QueryCancelledError
    specs = [fake_agent_spec(id="x", task_profile="engineer")]
    state = _make_state(specs)
    state.cancel_ev.set()
    mock_run_agent(lambda s, aid: "ok")
    from larkhelm.crew._runner import _execute_from
    with pytest.raises(QueryCancelledError):
        _execute_from(state, total_timeout=60, skip_ids=set())


# ─────────────────────────────────────────────────────────────────────────
#  _run_crew end-to-end (with _execute / _synthesize mocked)
# ─────────────────────────────────────────────────────────────────────────

def _stub_run_crew_io(monkeypatch):
    """Stub external IO _run_crew touches: heartbeat, register, checkpoint."""
    import larkhelm.crew_card as _cc
    # Make _start_heartbeat return a benign joined thread immediately
    def _fake_hb(state, stop_ev):
        stop_ev.set()
        t = threading.Thread(target=lambda: None)
        t.start()
        return t
    monkeypatch.setattr(_cc, "_start_heartbeat", _fake_hb)
    import larkhelm.crew._runner as _rn
    monkeypatch.setattr(_rn, "_start_heartbeat", _fake_hb, raising=False)
    import larkhelm.crew._state as _st
    monkeypatch.setattr(_st, "_register_crew_card",
                        lambda *a, **kw: None, raising=False)
    import larkhelm.crew._checkpoint as _ckpt
    monkeypatch.setattr(_ckpt, "_clear_checkpoint",
                        lambda *a, **kw: None, raising=False)


def test_run_crew_happy_path(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """_run_crew → executes → synthesizes → marks phase=done."""
    _stub_run_crew_io(monkeypatch)
    import larkhelm.crew._runner as _rn
    monkeypatch.setattr(_rn, "_execute", lambda s, t: None)
    monkeypatch.setattr(_rn, "_synthesize", lambda s: "final result")

    specs = [fake_agent_spec(id="a", task_profile="engineer")]
    state = _make_state(specs, synthesis_prompt="Synth")

    _rn._run_crew(state, total_timeout=60)
    assert state.phase == "done"
    assert "final result" in state.final_output


def test_run_crew_cancelled(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    from larkhelm.ai_runner import QueryCancelledError
    _stub_run_crew_io(monkeypatch)
    import larkhelm.crew._runner as _rn
    def _cancel(state, total_timeout):
        raise QueryCancelledError("user cancelled")
    monkeypatch.setattr(_rn, "_execute", _cancel)

    specs = [fake_agent_spec(id="a", task_profile="engineer")]
    state = _make_state(specs)

    _rn._run_crew(state, total_timeout=60)
    assert state.phase == "cancelled"


def test_run_crew_hard_fail(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    from larkhelm.crew_types import HardFailError
    _stub_run_crew_io(monkeypatch)
    import larkhelm.crew._runner as _rn
    def _hard(state, total_timeout):
        raise HardFailError("qa final failure")
    monkeypatch.setattr(_rn, "_execute", _hard)

    specs = [fake_agent_spec(id="a", task_profile="engineer")]
    state = _make_state(specs)

    _rn._run_crew(state, total_timeout=60)
    assert state.phase == "failed"
    assert "hard failure" in state.final_output.lower() or "qa" in state.final_output


def test_run_crew_cancel_between_execute_and_synthesize(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """cancel_ev set after _execute returns → skip synthesis, phase=cancelled."""
    _stub_run_crew_io(monkeypatch)
    import larkhelm.crew._runner as _rn

    def _exec(state, total_timeout):
        state.cancel_ev.set()
    monkeypatch.setattr(_rn, "_execute", _exec)

    synth_called = []
    monkeypatch.setattr(_rn, "_synthesize",
                        lambda s: synth_called.append(1) or "x")

    specs = [fake_agent_spec(id="a", task_profile="engineer")]
    state = _make_state(specs)

    _rn._run_crew(state, total_timeout=60)
    assert state.phase == "cancelled"
    assert synth_called == []


def test_run_crew_synthesize_failure_fallback(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """synthesis exception → fall back to concatenated agent results."""
    from larkhelm.crew_types import AgentStatus
    _stub_run_crew_io(monkeypatch)
    import larkhelm.crew._runner as _rn
    monkeypatch.setattr(_rn, "_execute", lambda s, t: None)
    def _boom(s):
        raise RuntimeError("orchestrator down")
    monkeypatch.setattr(_rn, "_synthesize", _boom)

    specs = [fake_agent_spec(id="a", task_profile="engineer")]
    state = _make_state(specs)
    state.agents["a"].status = AgentStatus.DONE
    state.agents["a"].result = "agent a output"

    _rn._run_crew(state, total_timeout=60)
    assert state.phase == "done"
    assert "agent a output" in state.final_output


# ─────────────────────────────────────────────────────────────────────────
#  _write_hermes_summary
# ─────────────────────────────────────────────────────────────────────────

def test_write_hermes_summary_creates_markdown(
    init_test_config, fake_agent_spec, fake_card_sender, tmp_path,
):
    spec = fake_agent_spec(id="h", model="hermes_split", task_profile="")
    state = _make_state([spec])
    from larkhelm.crew._runner import _write_hermes_summary
    _write_hermes_summary(state, "h", "the result", tmp_path)
    out = (tmp_path / "h_summary.md").read_text(encoding="utf-8")
    assert "split" in out
    assert "the result" in out
    assert state.plan.title in out


# ─────────────────────────────────────────────────────────────────────────
#  _run_agent_wrapper happy retry path (review §6 gap)
# ─────────────────────────────────────────────────────────────────────────

def test_run_agent_wrapper_retries_then_succeeds(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """Review §6 noted that retry coverage was OOM-biased — the happy
    path (first attempt raises a transient exception, second attempt
    succeeds) was only exercised indirectly. Pin it: agent ends in
    DONE state, result captured from the second call, no ⚠️ failure
    card emitted.

    Also asserts the inter-attempt backoff is the *short* 1s path
    (non-OOM-class error) rather than the 8s OOM backoff — otherwise
    a sleep-misclassification would silently double test runtime.
    """
    from larkhelm.crew_types import AgentStatus
    from larkhelm.crew import _runner as cr

    specs = [fake_agent_spec(id="impl", depends_on=[], task_profile="engineer")]
    state = _make_state(specs)

    # First call raises a generic (non-OOM) RuntimeError; second returns "OK".
    call_log: list[str] = []
    def fake_run_agent(s, aid):
        call_log.append(aid)
        if len(call_log) == 1:
            raise RuntimeError("transient HTTP read error")
        return "OK"
    monkeypatch.setattr(cr, "_run_agent", fake_run_agent)

    # _sync_output_file pokes the filesystem; stub to a no-op so tests
    # stay hermetic.
    monkeypatch.setattr(cr, "_sync_output_file", lambda st, aid: "")

    # Track whether emit_agent_failure ever fires (it must NOT on the
    # happy retry path). The fixture's fake_card_sender already mocks
    # the card send paths so we just import and assert call count.
    import larkhelm.crew._failure_card as _fc
    fail_calls: list[tuple] = []
    monkeypatch.setattr(
        _fc, "emit_agent_failure",
        lambda *a, **k: fail_calls.append((a, k)),
    )

    # Watch the actual backoff to confirm we hit the short (1s) branch.
    sleeps: list[float] = []
    monkeypatch.setattr(cr.time, "sleep", lambda s: sleeps.append(s))

    cr._run_agent_wrapper(state, "impl")

    assert call_log == ["impl", "impl"], (
        f"expected exactly 2 _run_agent calls (initial + retry), got {call_log}"
    )
    assert state.agents["impl"].status == AgentStatus.DONE
    assert state.agents["impl"].result == "OK"
    assert fail_calls == [], (
        "emit_agent_failure must not fire on the happy retry path — the "
        "user shouldn't see a ⚠️ card when retry succeeded"
    )
    assert sleeps == [1], (
        f"expected single 1s short-backoff sleep, got {sleeps} — an 8s "
        "value would indicate _is_likely_oom_error misclassified a "
        "non-OOM transient error and unfairly slowed retry"
    )


def test_run_agent_wrapper_oom_first_attempt_uses_8s_backoff(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch,
):
    """Counterpart to the happy retry test: when the first attempt
    raises an OOM-shaped error, the wrapper must back off 8s (not 1s)
    so the cgroup has a chance to reclaim memory before retry. The
    second attempt then succeeds → DONE without a ⚠️ card."""
    from larkhelm.crew_types import AgentStatus
    from larkhelm.crew import _runner as cr

    specs = [fake_agent_spec(id="impl", depends_on=[], task_profile="engineer")]
    state = _make_state(specs)

    call_log: list[str] = []
    def fake_run_agent(s, aid):
        call_log.append(aid)
        if len(call_log) == 1:
            raise RuntimeError("claude killed by OS (rc=-9, likely cgroup OOM)")
        return "OK"
    monkeypatch.setattr(cr, "_run_agent", fake_run_agent)
    monkeypatch.setattr(cr, "_sync_output_file", lambda st, aid: "")
    # _log_oom_diagnostics pokes /sys/fs/cgroup; stub to a no-op.
    monkeypatch.setattr(cr, "_log_oom_diagnostics", lambda aid: None)

    sleeps: list[float] = []
    monkeypatch.setattr(cr.time, "sleep", lambda s: sleeps.append(s))

    cr._run_agent_wrapper(state, "impl")

    assert state.agents["impl"].status == AgentStatus.DONE
    assert state.agents["impl"].result == "OK"
    assert sleeps == [8], (
        f"OOM-class error must trigger the 8s backoff, got {sleeps}"
    )
