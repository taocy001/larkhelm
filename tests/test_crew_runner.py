"""
Pure-function unit tests for Week3-P1: DAG semantics + plan preflight.

No subprocess spawning, no Feishu credentials required.
"""
from __future__ import annotations

import threading
import uuid

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_spec(agent_id: str, depends_on: list[str] | None = None, **kw):
    from larkhelm.crew_types import AgentSpec
    return AgentSpec(
        id=agent_id, role=agent_id, model="", system="", prompt="",
        depends_on=depends_on or [], timeout=60,
        **kw,
    )


def _make_state(specs):
    from larkhelm.crew_types import AgentState, CrewPlan, CrewState
    plan = CrewPlan(title="test", agents=specs, synthesis_prompt="")
    agents = {s.id: AgentState(spec=s) for s in specs}
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


# ── AC-04: AgentStatus.COMPLETED alias ───────────────────────────────────


def test_agent_status_completed_exists():
    """AgentStatus.COMPLETED must exist and have value 'completed'."""
    from larkhelm.crew_types import AgentStatus
    assert AgentStatus.COMPLETED is not None
    assert AgentStatus.COMPLETED.value == "completed"


def test_agent_status_done_is_completed():
    """AgentStatus.DONE must be an alias for AgentStatus.COMPLETED."""
    from larkhelm.crew_types import AgentStatus
    assert AgentStatus.DONE is AgentStatus.COMPLETED


def test_agent_status_missing_done_maps_to_completed():
    """AgentStatus('done') via _missing_ must return COMPLETED."""
    from larkhelm.crew_types import AgentStatus
    result = AgentStatus("done")
    assert result is AgentStatus.COMPLETED


def test_agent_status_missing_unknown_returns_none():
    """AgentStatus with unknown value must raise ValueError (not return garbage)."""
    from larkhelm.crew_types import AgentStatus
    with pytest.raises(ValueError):
        AgentStatus("nonexistent_value_xyz")


# ── _toposort_agents: linear chain ───────────────────────────────────────


def test_toposort_linear_chain():
    """A → B → C: C has no deps, so order should be [C, B, A]."""
    from larkhelm.crew._runner import _toposort_agents
    specs = [
        _make_spec("a", depends_on=["b"]),
        _make_spec("b", depends_on=["c"]),
        _make_spec("c", depends_on=[]),
    ]
    order = _toposort_agents(specs)
    ids = [s.id for s in order]
    assert ids.index("c") < ids.index("b") < ids.index("a"), (
        f"Expected c < b < a, got {ids}"
    )


# ── _toposort_agents: diamond ─────────────────────────────────────────────


def test_toposort_diamond():
    """Diamond: A depends on B and C; both depend on D. D must come first, A last."""
    from larkhelm.crew._runner import _toposort_agents
    specs = [
        _make_spec("a", depends_on=["b", "c"]),
        _make_spec("b", depends_on=["d"]),
        _make_spec("c", depends_on=["d"]),
        _make_spec("d", depends_on=[]),
    ]
    order = _toposort_agents(specs)
    ids = [s.id for s in order]
    assert ids.index("d") < ids.index("b"), f"Expected d < b, got {ids}"
    assert ids.index("d") < ids.index("c"), f"Expected d < c, got {ids}"
    assert ids.index("b") < ids.index("a"), f"Expected b < a, got {ids}"
    assert ids.index("c") < ids.index("a"), f"Expected c < a, got {ids}"


# ── _toposort_agents: cycle detection ────────────────────────────────────


def test_toposort_cycle_raises_value_error():
    """A → B → A creates a cycle; must raise ValueError containing 'cycle'."""
    from larkhelm.crew._runner import _toposort_agents
    specs = [
        _make_spec("a", depends_on=["b"]),
        _make_spec("b", depends_on=["a"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        _toposort_agents(specs)


def test_toposort_self_loop_raises():
    """A → A is a trivial cycle."""
    from larkhelm.crew._runner import _toposort_agents
    specs = [_make_spec("a", depends_on=["a"])]
    with pytest.raises(ValueError, match="cycle"):
        _toposort_agents(specs)


# ── _migrate_v1_to_v2 ────────────────────────────────────────────────────


def _v1_checkpoint_dict() -> dict:
    return {
        "version": 1,
        "crew_id": "abc12345",
        "chat_id": "test_chat",
        "card_mid": "card_x",
        "start_time": 0.0,
        "phase": "running",
        "kind": "dev",
        "git_head_before": "",
        "phase_commits": {},
        "plan": {
            "title": "test",
            "synthesis_prompt": "",
            "agents": [
                {
                    "id": "pm", "role": "PM", "model": "claude",
                    "system": "s", "prompt": "p",
                    "depends_on": [], "timeout": 60,
                    "exit_marker": "", "fail_marker": "",
                    "retry_target": [], "max_retries": 0,
                    "is_gatekeeper": False, "breakpoint": False,
                    "trigger_only": False, "hard_fail_on_exhaust": False,
                    "retry_system": "", "retry_prompt": "",
                    "output_file": "prd.md",
                    "require_arch": "", "require_docker_image": "",
                    "task_profile": "",
                },
            ],
        },
        "agents": {
            "pm": {"status": "done", "result": "prd ok",
                   "error": "", "retry_count": 0, "round_label": ""},
        },
        "completed_wave_ids": ["pm"],
    }


def test_migrate_v1_to_v2_schema_version():
    """After migration, data must have schema_version=2 and no 'version' key."""
    from larkhelm.crew._checkpoint import _migrate_v1_to_v2
    data = _v1_checkpoint_dict()
    result = _migrate_v1_to_v2(data)
    assert result["schema_version"] == 2
    assert "version" not in result


def test_migrate_v1_to_v2_status_done_to_completed():
    """Agent snapshot 'done' status must be migrated to 'completed'."""
    from larkhelm.crew._checkpoint import _migrate_v1_to_v2
    data = _v1_checkpoint_dict()
    result = _migrate_v1_to_v2(data)
    assert result["agents"]["pm"]["status"] == "completed"


def test_migrate_v1_to_v2_adds_fallback_agent_id():
    """Each agent spec must have fallback_agent_id='' after migration."""
    from larkhelm.crew._checkpoint import _migrate_v1_to_v2
    data = _v1_checkpoint_dict()
    result = _migrate_v1_to_v2(data)
    for spec in result["plan"]["agents"]:
        assert "fallback_agent_id" in spec
        assert spec["fallback_agent_id"] == ""


# ── AC-06: _load_checkpoint accepts v1 format ────────────────────────────


def test_load_checkpoint_accepts_v1(tmp_path, init_test_config, monkeypatch):
    """AC-06: _load_checkpoint must accept old 'version': 1 checkpoint format."""
    import json
    from larkhelm.crew._checkpoint import _load_checkpoint

    # Write a v1 checkpoint to tmp_path
    cp_dir = tmp_path / ".crew_workspace"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_path = cp_dir / "crew_checkpoint.json"
    cp_path.write_text(json.dumps(_v1_checkpoint_dict()), encoding="utf-8")

    # Patch _get_cwd to return tmp_path
    import larkhelm.chat_state as _cs
    monkeypatch.setattr(_cs, "_get_cwd", lambda chat_id: str(tmp_path))

    data = _load_checkpoint("test_chat")
    assert data is not None, "_load_checkpoint returned None for v1 checkpoint"
    assert data.get("schema_version") == 2, "v1 checkpoint should be migrated to v2"
    assert data["agents"]["pm"]["status"] == "completed"


def test_load_checkpoint_accepts_v2(tmp_path, init_test_config, monkeypatch):
    """_load_checkpoint must also accept schema_version: 2 format."""
    import json
    from larkhelm.crew._checkpoint import _load_checkpoint

    cp_dir = tmp_path / ".crew_workspace"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_path = cp_dir / "crew_checkpoint.json"
    v2_data = _v1_checkpoint_dict()
    v2_data.pop("version", None)
    v2_data["schema_version"] = 2
    v2_data["plan"]["agents"][0]["fallback_agent_id"] = ""
    v2_data["agents"]["pm"]["status"] = "completed"
    cp_path.write_text(json.dumps(v2_data), encoding="utf-8")

    import larkhelm.chat_state as _cs
    monkeypatch.setattr(_cs, "_get_cwd", lambda chat_id: str(tmp_path))

    data = _load_checkpoint("test_chat")
    assert data is not None
    assert data.get("schema_version") == 2


def test_load_checkpoint_rejects_unknown_format(tmp_path, init_test_config, monkeypatch):
    """_load_checkpoint must return None for unknown formats."""
    import json
    from larkhelm.crew._checkpoint import _load_checkpoint

    cp_dir = tmp_path / ".crew_workspace"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_path = cp_dir / "crew_checkpoint.json"
    cp_path.write_text(json.dumps({"some": "garbage"}), encoding="utf-8")

    import larkhelm.chat_state as _cs
    monkeypatch.setattr(_cs, "_get_cwd", lambda chat_id: str(tmp_path))

    data = _load_checkpoint("test_chat")
    assert data is None


# ── SKIPPED status in dep-failure ────────────────────────────────────────


def test_dep_failure_sets_skipped(
    init_test_config, fake_card_sender, fake_backend_registry, mock_run_agent,
):
    """When upstream agent is FAILED, downstream must be SKIPPED (not FAILED)."""
    from larkhelm.crew._runner import _execute
    from larkhelm.crew_types import AgentStatus

    specs = [
        _make_spec("a", depends_on=[]),
        _make_spec("b", depends_on=["a"]),
    ]
    state = _make_state(specs)

    # Make agent "a" always fail
    def fake_run(s, agent_id):
        if agent_id == "a":
            raise RuntimeError("simulated failure")
        return "ok"

    mock_run_agent(fake_run)

    _execute(state, total_timeout=30)

    # "a" should be FAILED (ran but errored)
    assert state.agents["a"].status == AgentStatus.FAILED, (
        f"Expected a=FAILED, got {state.agents['a'].status}"
    )
    # "b" should be SKIPPED (dep failed — intentionally not run)
    assert state.agents["b"].status == AgentStatus.SKIPPED, (
        f"Expected b=SKIPPED, got {state.agents['b'].status}"
    )


# ── fallback_agent_id field exists on AgentSpec ──────────────────────────


def test_agent_spec_fallback_agent_id_default():
    """AgentSpec must have fallback_agent_id field defaulting to empty string."""
    spec = _make_spec("x")
    assert hasattr(spec, "fallback_agent_id")
    assert spec.fallback_agent_id == ""


def test_agent_spec_fallback_agent_id_settable():
    """fallback_agent_id must be settable on AgentSpec."""
    spec = _make_spec("x", fallback_agent_id="y")
    assert spec.fallback_agent_id == "y"
