"""Regression tests for the three fixes in the P3 follow-up:

* Q2(a) — ``crew_card._backend_label`` shows the actually-resolved backend
  (``AgentState.actual_backend_id``) instead of the now-empty ``spec.model``
  left by the Phase-C ``task_profile`` migration. Without the fix the
  bracket on each agent row rendered as ``[]``.

* Q2(b) — ``_persist_result_to_output_file_if_missing`` writes the
  in-memory ``result`` to ``.crew_workspace/{output_file}`` when the
  agent skipped its Write tool call. Observed in P3: PM produced a ~39 K
  token PRD but never called Write, so prd.md was missing on disk.

* Q3   — ``_check_task_already_complete`` + ``_execute`` short-circuit
  the pipeline when PM emits the ``TASK_ALREADY_COMPLETE`` marker on the
  first line of its result. The remaining waves are drained and all
  PENDING agents are marked SKIPPED; ``_synthesize`` produces a templated
  "already complete" body.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Q2(a) — crew_card backend label
# ─────────────────────────────────────────────────────────────────────────


def test_backend_label_prefers_actual_backend_id():
    from larkhelm.crew_card import _backend_label
    from larkhelm.crew_types import AgentSpec, AgentState

    spec = AgentSpec(
        id="pm", role="产品经理", model="", task_profile="planner",
        system="", prompt="", depends_on=[], timeout=60,
    )
    agent_state = AgentState(spec=spec, actual_backend_id="claude")
    assert _backend_label(spec, agent_state) == "claude", (
        "should prefer resolved backend over empty spec.model"
    )


def test_backend_label_falls_back_to_spec_model():
    from larkhelm.crew_card import _backend_label
    from larkhelm.crew_types import AgentSpec, AgentState

    spec = AgentSpec(
        id="pm", role="产品经理", model="kimi", task_profile="",
        system="", prompt="", depends_on=[], timeout=60,
    )
    agent_state = AgentState(spec=spec)  # actual_backend_id default ""
    assert _backend_label(spec, agent_state) == "kimi"


def test_backend_label_falls_back_to_task_profile_for_pending():
    """Before resolve_backend runs, the card still shows the profile hint
    so the user sees ``[planner] PM`` instead of ``[] PM``."""
    from larkhelm.crew_card import _backend_label
    from larkhelm.crew_types import AgentSpec, AgentState

    spec = AgentSpec(
        id="pm", role="产品经理", model="", task_profile="planner",
        system="", prompt="", depends_on=[], timeout=60,
    )
    agent_state = AgentState(spec=spec)
    assert _backend_label(spec, agent_state) == "planner"


def test_backend_label_empty_when_nothing_known():
    from larkhelm.crew_card import _backend_label
    from larkhelm.crew_types import AgentSpec, AgentState

    spec = AgentSpec(
        id="x", role="r", model="", task_profile="",
        system="", prompt="", depends_on=[], timeout=60,
    )
    assert _backend_label(spec, AgentState(spec=spec)) == ""


# ─────────────────────────────────────────────────────────────────────────
# Q2(b) — output_file safety-net
# ─────────────────────────────────────────────────────────────────────────


def _make_state(tmp_path: Path, agent_id: str, output_file: str,
                model: str = "claude"):
    from larkhelm.crew_types import (
        AgentSpec, AgentState, CrewPlan, CrewState,
    )
    spec = AgentSpec(
        id=agent_id, role="r", model=model, task_profile="",
        system="", prompt="", depends_on=[], timeout=60,
        output_file=output_file,
    )
    plan = CrewPlan(title="t", agents=[spec], synthesis_prompt="")
    state = CrewState(
        crew_id="c1", chat_id="c_safety_net", plan=plan,
        agents={agent_id: AgentState(spec=spec)},
        card_mid="mid", cancel_ev=threading.Event(),
        phase="running", kind="dev",
    )
    return state, spec


def test_safety_net_writes_when_output_file_missing(monkeypatch, tmp_path):
    """If the file the agent was supposed to write is missing on disk
    AND the in-memory result is substantial, write the result."""
    from larkhelm.crew import _runner as _rb

    state, _ = _make_state(tmp_path, "pm", "prd.md")
    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda _cid: str(tmp_path))
    (tmp_path / ".crew_workspace").mkdir(parents=True, exist_ok=True)

    pretend_prd = "# PRD\n\n" + ("内容 " * 500)  # >> 200 chars threshold
    _rb._persist_result_to_output_file_if_missing(state, "pm", pretend_prd)

    out = tmp_path / ".crew_workspace" / "prd.md"
    assert out.exists(), "safety net should have written prd.md"
    assert out.read_text(encoding="utf-8") == pretend_prd


def test_safety_net_skips_when_result_too_short(monkeypatch, tmp_path):
    """A short closing marker (≤ 200 chars) is the *expected* shape of the
    result field when the agent honoured the Write contract — don't
    overwrite whatever the agent wrote."""
    from larkhelm.crew import _runner as _rb

    state, _ = _make_state(tmp_path, "pm", "prd.md")
    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda _cid: str(tmp_path))
    (tmp_path / ".crew_workspace").mkdir(parents=True, exist_ok=True)
    # Agent's own Write produced the real PRD:
    (tmp_path / ".crew_workspace" / "prd.md").write_text(
        "agent-written PRD body", encoding="utf-8",
    )

    _rb._persist_result_to_output_file_if_missing(
        state, "pm", "PRD 已写入 .crew_workspace/prd.md",  # 25 chars
    )

    # File untouched — short result must NOT trigger the safety net.
    assert (tmp_path / ".crew_workspace" / "prd.md").read_text(encoding="utf-8") == (
        "agent-written PRD body"
    )


def test_safety_net_skips_when_existing_file_already_large(monkeypatch, tmp_path):
    """When the on-disk file already covers ≥ 80% of result length, trust
    the agent's own Write — closing-summary length mismatch is fine."""
    from larkhelm.crew import _runner as _rb

    state, _ = _make_state(tmp_path, "pm", "prd.md")
    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda _cid: str(tmp_path))
    (tmp_path / ".crew_workspace").mkdir(parents=True, exist_ok=True)
    big_existing = "X" * 1000
    (tmp_path / ".crew_workspace" / "prd.md").write_text(big_existing, encoding="utf-8")

    # Result is 800 chars; existing 1000 ≥ 0.8 * 800 = 640 ⇒ keep existing.
    _rb._persist_result_to_output_file_if_missing(state, "pm", "Y" * 800)

    assert (tmp_path / ".crew_workspace" / "prd.md").read_text(encoding="utf-8") == big_existing


# ─────────────────────────────────────────────────────────────────────────
# Q3 — TASK_ALREADY_COMPLETE marker / pipeline short-circuit
# ─────────────────────────────────────────────────────────────────────────


def _build_dev_state(tmp_path: Path, pm_result: str = "",
                     pm_status="done"):
    from larkhelm.crew_types import (
        AgentSpec, AgentState, AgentStatus, CrewPlan, CrewState,
    )

    def _spec(_id):
        return AgentSpec(
            id=_id, role=_id, model="", task_profile="planner",
            system="", prompt="", depends_on=[], timeout=60,
            output_file=f"{_id}.md",
        )

    specs = [_spec("pm"), _spec("architect"), _spec("implementer")]
    plan = CrewPlan(title="t", agents=specs, synthesis_prompt="syn")
    agents = {s.id: AgentState(spec=s) for s in specs}
    agents["pm"].status = AgentStatus.DONE if pm_status == "done" else AgentStatus.PENDING
    agents["pm"].result = pm_result
    state = CrewState(
        crew_id="c1", chat_id="c_marker", plan=plan,
        agents=agents, card_mid="mid",
        cancel_ev=threading.Event(), phase="running", kind="dev",
    )
    return state


def test_check_task_already_complete_marker_in_memory(tmp_path):
    from larkhelm.crew._runner import _check_task_already_complete

    state = _build_dev_state(
        tmp_path,
        pm_result="TASK_ALREADY_COMPLETE: P0 已在 commit 6515a43 全部落地",
    )
    wave = list(state.plan.agents[:1])  # first wave contains pm
    assert _check_task_already_complete(state, wave) is True


def test_check_task_already_complete_marker_in_prd_file(monkeypatch, tmp_path):
    """PM result captured only a short closing marker but the safety-net
    wrote the marker into prd.md — pickup must still find it."""
    from larkhelm.crew._runner import _check_task_already_complete

    state = _build_dev_state(tmp_path, pm_result="ok")  # short non-marker result
    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda _cid: str(tmp_path))
    (tmp_path / ".crew_workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".crew_workspace" / "prd.md").write_text(
        "TASK_ALREADY_COMPLETE: 一行原因\n", encoding="utf-8",
    )
    wave = list(state.plan.agents[:1])
    assert _check_task_already_complete(state, wave) is True


def test_check_task_already_complete_negative_no_marker(tmp_path):
    """Normal PM result (multi-paragraph PRD) must NOT trigger short-circuit."""
    from larkhelm.crew._runner import _check_task_already_complete

    state = _build_dev_state(
        tmp_path,
        pm_result="# PRD\n\n## 1. 背景\n\n这是一个常规 PRD…",
    )
    wave = list(state.plan.agents[:1])
    assert _check_task_already_complete(state, wave) is False


def test_check_task_already_complete_skips_when_pm_retried(tmp_path):
    """If PM has been retried (user explicitly forced a re-run via QA/reviewer
    feedback), short-circuit is disabled even if the new result happens to
    look like the marker. Prevents a self-defeating retry loop."""
    from larkhelm.crew._runner import _check_task_already_complete
    from larkhelm.crew_types import AgentStatus

    state = _build_dev_state(
        tmp_path,
        pm_result="TASK_ALREADY_COMPLETE: nothing to do",
    )
    state.agents["pm"].retry_count = 1
    wave = list(state.plan.agents[:1])
    assert _check_task_already_complete(state, wave) is False


def test_synthesize_renders_templated_card_when_marker_set(tmp_path, monkeypatch):
    """When the short-circuit fires, ``_synthesize`` produces a fixed
    user-friendly card body — no LLM call needed."""
    from larkhelm.crew._runner import _synthesize
    from larkhelm.crew_types import AgentStatus

    state = _build_dev_state(
        tmp_path,
        pm_result="TASK_ALREADY_COMPLETE: 一句话说明",
    )
    # Mark downstream agents as SKIPPED to mirror what _execute does.
    for sid in ("architect", "implementer"):
        state.agents[sid].status = AgentStatus.SKIPPED

    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda _cid: str(tmp_path))

    body = _synthesize(state)
    assert "任务已经在代码中完成" in body
    assert "一句话说明" in body
    assert "architect" in body
    assert "implementer" in body


def test_synthesize_finds_marker_in_prd_file_when_result_is_short(
    tmp_path, monkeypatch,
):
    """Round-2 review MUST-FIX regression:

    ``_check_task_already_complete`` falls back to prd.md when PM's
    in-memory result is short. Previously ``_synthesize``'s short-
    circuit only inspected ``pm_state.result`` directly, so PM
    honouring the Write-tool contract (writing the marker to prd.md
    and emitting a short "Done." closing message as ``result``) made
    ``_execute`` correctly SKIP the downstream agents BUT ``_synthesize``
    fell through to the full LLM-synthesis path — generating a
    hallucinated summary about empty agent outputs.

    Fix: both helpers go through the shared
    ``_extract_task_complete_marker`` so the two judgements never
    diverge. Test pins the prd-only-marker path end-to-end.
    """
    from larkhelm.crew._runner import _synthesize
    from larkhelm.crew_types import AgentStatus

    # PM result is the short closing summary; marker only on disk.
    state = _build_dev_state(
        tmp_path,
        pm_result="PRD written to .crew_workspace/prd.md.",
    )
    for sid in ("architect", "implementer"):
        state.agents[sid].status = AgentStatus.SKIPPED

    monkeypatch.setattr("larkhelm.chat_state._get_cwd", lambda _cid: str(tmp_path))
    (tmp_path / ".crew_workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".crew_workspace" / "prd.md").write_text(
        "TASK_ALREADY_COMPLETE: P0-P3 全部在 commit 9e7d4be 落地\n",
        encoding="utf-8",
    )

    body = _synthesize(state)
    assert "任务已经在代码中完成" in body, (
        "synthesize did not pick up the marker from prd.md fallback — "
        "MUST-FIX regression"
    )
    assert "9e7d4be" in body, "reason text from prd.md was not threaded through"
