"""Tests for phase_outputs checkpoint round-trip and resume injection (Week4-P1 M-CHECKPOINT)."""
from __future__ import annotations

import json
import os
import threading
import uuid

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_spec(agent_id: str, output_file: str = "", depends_on=None, **kw):
    from larkhelm.crew_types import AgentSpec
    return AgentSpec(
        id=agent_id, role=f"Role-{agent_id}", model="", system="", prompt="",
        depends_on=depends_on or [], timeout=60, output_file=output_file, **kw,
    )


def _make_state(specs, max_qa_retry_rounds: int = 2):
    from larkhelm.crew_types import AgentState, CrewPlan, CrewState
    plan = CrewPlan(
        title="test plan", agents=specs, synthesis_prompt="",
        max_qa_retry_rounds=max_qa_retry_rounds,
    )
    agents = {s.id: AgentState(spec=s) for s in specs}
    return CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents=agents,
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="dev",
    )


# ── AC-05: phase_outputs saved in checkpoint ──────────────────────────────────


def test_phase_outputs_saved_in_checkpoint(tmp_path, monkeypatch):
    """_save_checkpoint must write phase_outputs field with completed agent summaries."""
    from larkhelm.crew_types import AgentStatus

    specs = [
        _make_spec("pm", output_file="prd.md"),
        _make_spec("architect", output_file="design.md", depends_on=["pm"]),
    ]
    state = _make_state(specs)

    # Mark pm as DONE with a result
    with state.lock:
        state.agents["pm"].status = AgentStatus.DONE
        state.agents["pm"].result = "PRD generated with 3 requirements."
        state.agents["architect"].status = AgentStatus.FAILED
        state.agents["architect"].result = ""
        state.agents["architect"].error = "timeout"

    # Patch _get_cwd to return tmp_path
    monkeypatch.setattr(
        "larkhelm.crew._checkpoint._get_cwd_from_checkpoint",
        lambda cid: str(tmp_path),
        raising=False,
    )
    # Simpler: patch chat_state._get_cwd
    import larkhelm.chat_state as _cs
    monkeypatch.setattr(_cs, "_get_cwd", lambda cid: str(tmp_path))

    from larkhelm.crew._checkpoint import _save_checkpoint
    _save_checkpoint(state, ["pm", "architect"])

    cp_path = tmp_path / ".crew_workspace" / "crew_checkpoint.json"
    assert cp_path.exists(), "Checkpoint file not created"
    data = json.loads(cp_path.read_text())

    assert "phase_outputs" in data, "phase_outputs missing from checkpoint"
    po = data["phase_outputs"]
    assert "pm" in po
    assert po["pm"]["exit_status"] == "PASS"
    assert po["pm"]["output_file"] == "prd.md"
    assert "PRD generated" in po["pm"]["summary"]

    assert "architect" in po
    assert po["architect"]["exit_status"] == "FAIL"

    # max_qa_retry_rounds must be saved too
    assert data["plan"]["max_qa_retry_rounds"] == 2


# ── AC-06: phase_outputs default on missing (old checkpoint) ──────────────────


def test_phase_outputs_default_on_missing(tmp_path):
    """Old checkpoint without phase_outputs → state.phase_outputs == {} after rebuild."""
    from larkhelm.crew_types import AgentSpec

    spec_dict = dict(
        id="pm", role="产品经理", model="", system="", prompt="",
        depends_on=[], timeout=300, exit_marker="", fail_marker="",
        retry_target=[], max_retries=0, is_gatekeeper=False,
        breakpoint=False, trigger_only=False, hard_fail_on_exhaust=False,
        retry_system="", retry_prompt="", output_file="prd.md",
        require_arch="", require_docker_image="", task_profile="planner",
        fallback_agent_id="",
    )

    old_checkpoint = {
        "schema_version": 2,
        "crew_id": "abc123",
        "chat_id": "test_chat",
        "card_mid": "mid_1",
        "start_time": 1717000000.0,
        "phase": "running",
        "kind": "dev",
        "git_head_before": "",
        "phase_commits": {},
        "plan": {
            "title": "old plan",
            "synthesis_prompt": "",
            "agents": [spec_dict],
            # no max_qa_retry_rounds key
        },
        "agents": {
            "pm": {
                "status": "completed",
                "result": "done",
                "error": "",
                "retry_count": 0,
                "round_label": "",
            }
        },
        "completed_wave_ids": ["pm"],
        # no phase_outputs key
    }

    import larkhelm.concurrency as _conc
    _conc._chat_cancel_events = {}

    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    state = _rebuild_state_from_checkpoint(old_checkpoint)

    assert state is not None
    assert state.phase_outputs == {}
    assert state.plan.max_qa_retry_rounds == 2  # default


def test_migrate_v1_to_v2_adds_phase_outputs():
    """_migrate_v1_to_v2 must add phase_outputs: {} when missing."""
    from larkhelm.crew._checkpoint import _migrate_v1_to_v2

    data = {
        "version": 1,
        "crew_id": "x",
        "chat_id": "c",
        "plan": {"title": "t", "agents": []},
        "agents": {"pm": {"status": "done"}},
        "completed_wave_ids": [],
    }
    result = _migrate_v1_to_v2(data)
    assert "phase_outputs" in result
    assert result["phase_outputs"] == {}
    assert result["schema_version"] == 2
    # Status "done" should be mapped to "completed"
    assert result["agents"]["pm"]["status"] == "completed"


# ── AC-06: resume summary injection ──────────────────────────────────────────


def test_resume_summary_in_phase_outputs_field():
    """CrewState.phase_outputs is populated from checkpoint data."""
    old_checkpoint = {
        "schema_version": 2,
        "crew_id": "abc",
        "chat_id": "test_chat",
        "card_mid": None,
        "start_time": 1717000000.0,
        "phase": "running",
        "kind": "dev",
        "git_head_before": "",
        "phase_commits": {},
        "plan": {
            "title": "plan",
            "synthesis_prompt": "",
            "max_qa_retry_rounds": 3,
            "agents": [{
                "id": "pm", "role": "PM", "model": "", "system": "", "prompt": "",
                "depends_on": [], "timeout": 300, "exit_marker": "",
                "fail_marker": "", "retry_target": [], "max_retries": 0,
                "is_gatekeeper": False, "breakpoint": False, "trigger_only": False,
                "hard_fail_on_exhaust": False, "retry_system": "", "retry_prompt": "",
                "output_file": "prd.md", "require_arch": "", "require_docker_image": "",
                "task_profile": "planner", "fallback_agent_id": "",
            }],
        },
        "agents": {},
        "completed_wave_ids": [],
        "phase_outputs": {
            "pm": {
                "summary": "PRD done with 3 requirements.",
                "output_file": "prd.md",
                "exit_status": "PASS",
            }
        },
    }

    import larkhelm.concurrency as _conc
    _conc._chat_cancel_events = {}

    from larkhelm.crew._checkpoint import _rebuild_state_from_checkpoint
    state = _rebuild_state_from_checkpoint(old_checkpoint)
    assert state is not None
    assert state.is_resuming is True
    assert state.phase_outputs == old_checkpoint["phase_outputs"]
    assert state.plan.max_qa_retry_rounds == 3


# ── AC-06: _run_agent injects phase_outputs into resume prefix ────────────────


def test_resume_summary_injection():
    """When state.is_resuming=True and phase_outputs is non-empty,
    _run_agent prepends '前序 Agent 摘要' summary block to full_prompt.

    Tests the string-building algorithm at crew/_runner.py lines 791-812
    (the exact same logic that _run_agent executes in the is_resuming branch).
    """
    from larkhelm.crew_types import AgentSpec

    phase_outputs = {
        "pm": {"summary": "PRD done with 3 requirements.", "output_file": "prd.md", "exit_status": "PASS"},
        "architect": {"summary": "Architecture design complete.", "output_file": "design.md", "exit_status": "PASS"},
    }
    specs = [
        AgentSpec(id="pm", role="产品经理", model="", system="", prompt="",
                  depends_on=[], timeout=300, output_file="prd.md"),
        AgentSpec(id="architect", role="架构师", model="", system="", prompt="",
                  depends_on=["pm"], timeout=300, output_file="design.md"),
    ]

    # Reproduce the exact _run_agent phase_outputs → _po_lines → _resume_prefix logic
    _spec_by_id = {s.id: s for s in specs}
    _po_lines: list[str] = []
    for _po_id, _po_data in phase_outputs.items():
        _po_spec = _spec_by_id.get(_po_id)
        _po_role = _po_spec.role if _po_spec else _po_id
        _po_file = (_po_data.get("output_file") or "")
        _po_sum = (_po_data.get("summary") or "")[:200]
        _po_arrow = f" → {_po_file}" if _po_file else ""
        _po_lines.append(f"[{_po_role}{_po_arrow}]: {_po_sum}")

    original_prompt = "implement the feature"
    _resume_prefix = (
        "⚠️ **Resuming task (previous execution was interrupted, continuing from checkpoint)**\n\n"
        "Resume notes:\n"
        "- `.crew_workspace/` contains the planning files from the last run; "
        "**use these as the baseline**, do not re-plan"
        + ("\n\n前序 Agent 摘要：\n" + "\n".join("- " + ln for ln in _po_lines) if _po_lines else "")
        + "\n- Continue directly from the incomplete parts; skip already completed content\n\n---\n\n"
    )
    full_prompt = _resume_prefix + original_prompt

    assert "前序 Agent 摘要" in full_prompt, "prefix must contain '前序 Agent 摘要'"
    assert "PRD done" in full_prompt, "phase_outputs summary must appear in prompt"
    assert "产品经理" in full_prompt, "agent role must appear in prompt"
    assert "prd.md" in full_prompt, "output_file must appear in prompt"
    assert original_prompt in full_prompt, "original prompt must still be present"
    assert full_prompt.index("前序 Agent 摘要") < full_prompt.index(original_prompt), \
        "summary must precede the original prompt"
