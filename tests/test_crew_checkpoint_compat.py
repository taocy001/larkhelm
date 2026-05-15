"""Tests for AC-08: legacy checkpoint compatibility (no task_profile field)."""
from __future__ import annotations


# ── Build a synthetic legacy checkpoint dict ──────────────────────────

def _legacy_checkpoint_dict() -> dict:
    """Return a checkpoint dict shaped like Phase B serialization (no
    task_profile field on AgentSpec)."""
    return {
        "version": 1,
        "crew_id": "abc12345",
        "chat_id": "test_chat",
        "card_mid": "card_xxx",
        "start_time": 0.0,
        "phase": "running",
        "kind": "dev",
        "git_head_before": "",
        "phase_commits": {},
        "plan": {
            "title": "legacy test plan",
            "synthesis_prompt": "synth",
            "agents": [
                # Notably NO ``task_profile`` field below
                {
                    "id": "pm", "role": "PM", "model": "claude",
                    "system": "sys", "prompt": "do pm",
                    "depends_on": [], "timeout": 60,
                    "exit_marker": "", "fail_marker": "",
                    "retry_target": [], "max_retries": 0,
                    "is_gatekeeper": False, "breakpoint": False,
                    "trigger_only": False, "hard_fail_on_exhaust": False,
                    "retry_system": "", "retry_prompt": "",
                    "output_file": "prd.md",
                },
                {
                    "id": "engineer", "role": "Engineer", "model": "kimi",
                    "system": "sys", "prompt": "code",
                    "depends_on": ["pm"], "timeout": 60,
                    "exit_marker": "", "fail_marker": "",
                    "retry_target": [], "max_retries": 0,
                    "is_gatekeeper": False, "breakpoint": False,
                    "trigger_only": False, "hard_fail_on_exhaust": False,
                    "retry_system": "", "retry_prompt": "",
                    "output_file": "changes.md",
                },
            ],
        },
        "agents": {
            "pm": {"status": "done", "result": "prd ok",
                   "error": "", "retry_count": 0, "round_label": ""},
        },
        "completed_wave_ids": ["pm"],
    }


def test_rebuild_legacy_checkpoint_does_not_raise(
    init_test_config, fake_card_sender,
):
    """AC-08: ``_rebuild_state_from_checkpoint`` must accept legacy schemas."""
    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    state = _rebuild_state_from_checkpoint(_legacy_checkpoint_dict())
    assert state is not None
    assert state.crew_id == "abc12345"


def test_rebuild_legacy_specs_default_empty_task_profile(
    init_test_config, fake_card_sender,
):
    """AgentSpec defaults task_profile="" so legacy snaps round-trip cleanly."""
    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    state = _rebuild_state_from_checkpoint(_legacy_checkpoint_dict())
    for spec in state.plan.agents:
        assert spec.task_profile == "", (
            f"agent {spec.id!r} should default task_profile='' on legacy load"
        )


def test_rebuild_preserves_legacy_model_field(
    init_test_config, fake_card_sender,
):
    """``model`` field stays intact so the resolver fallback path works."""
    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    state = _rebuild_state_from_checkpoint(_legacy_checkpoint_dict())
    by_id = {s.id: s for s in state.plan.agents}
    assert by_id["pm"].model == "claude"
    assert by_id["engineer"].model == "kimi"


def test_rebuild_preserves_completed_agent_state(
    init_test_config, fake_card_sender,
):
    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    from larkhelm.crew_types import AgentStatus
    state = _rebuild_state_from_checkpoint(_legacy_checkpoint_dict())
    pm = state.agents["pm"]
    assert pm.status == AgentStatus.DONE
    assert pm.result == "prd ok"


def test_modern_checkpoint_with_task_profile_round_trips(
    init_test_config, fake_card_sender,
):
    """A modern checkpoint with explicit task_profile should also rebuild correctly."""
    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    data = _legacy_checkpoint_dict()
    for snap in data["plan"]["agents"]:
        snap["task_profile"] = "engineer"
    state = _rebuild_state_from_checkpoint(data)
    for spec in state.plan.agents:
        assert spec.task_profile == "engineer"


def test_rebuild_handles_unknown_field_gracefully(
    init_test_config, fake_card_sender,
):
    """Extra unknown fields in the snap dict are NOT a current contract — but
    if they appear (e.g. forward-compat experiments), rebuild should fail
    cleanly (return None) rather than crash the resume thread."""
    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    data = _legacy_checkpoint_dict()
    data["plan"]["agents"][0]["nonsense_field_xyz"] = True
    state = _rebuild_state_from_checkpoint(data)
    assert state is None
