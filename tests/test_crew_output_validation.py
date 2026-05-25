"""Tests for ``_validate_output_artifact``: tool-call sentinel + JSON
contract checks for crew agent output files.

Regression context (2026-05-22 dev failure):
  A planner backend emitted ``<｜｜DSML｜｜tool_call...`` tokens as plain text
  into ``prd.md`` / ``design.md``. The runner persisted the bytes verbatim,
  marked the agent DONE, and let architect / fixer / QA burn full agents'
  worth of tokens on an unusable contract. The validator catches the leak
  at the producer so the crew circuit-breaks before downstream cost.
"""
from __future__ import annotations

import json
from pathlib import Path


def _patch_cwd(monkeypatch, tmp_path: Path) -> None:
    """Make ``_get_cwd`` return ``tmp_path`` for the test chat."""
    from larkhelm import chat_state
    monkeypatch.setattr(chat_state, "_get_cwd", lambda chat_id: str(tmp_path))


def _write_workspace_file(tmp_path: Path, name: str, content: str) -> Path:
    ws = tmp_path / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    p = ws / name
    p.write_text(content, encoding="utf-8")
    return p


# ── sentinel detection on disk ─────────────────────────────────────

def test_validate_detects_dsml_sentinel_in_md(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "prd.md",
        "我需要先读取 PRD\n\n<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name=\"Read\">\n",
    )
    issue = _validate_output_artifact(state, "pm", result="ack")
    assert issue
    assert "sentinel" in issue
    assert "prd.md" in issue


def test_validate_detects_openai_style_sentinel(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(tmp_path, "design.md", "thinking...\n<|tool_call name=\"Read\">\n")
    issue = _validate_output_artifact(state, "arch", result="ack")
    assert issue
    assert "<|tool_call" in issue


# ── sentinel detection via result fallback (no file written) ───────

def test_validate_falls_back_to_result_when_no_file(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    # No file on disk → scan the in-memory result instead.
    leak = "ok\n<｜｜DSML｜｜tool_calls>\n…"
    issue = _validate_output_artifact(state, "pm", result=leak)
    assert issue
    assert "sentinel" in issue
    assert "result" in issue


# ── JSON contract ──────────────────────────────────────────────────

def test_validate_rejects_malformed_json_output_file(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="tasks.json")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(tmp_path, "tasks.json", "{not valid json")
    issue = _validate_output_artifact(state, "pm", result="ack")
    assert issue
    assert "not valid JSON" in issue
    assert "tasks.json" in issue


def test_validate_accepts_valid_json_output_file(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="tasks.json")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "tasks.json",
        json.dumps({"task_list": [{"id": 1, "title": "x"}], "required_packages": []}),
    )
    assert _validate_output_artifact(state, "pm", result="ack") == ""


# ── happy path: clean markdown ─────────────────────────────────────

def test_validate_accepts_clean_markdown(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "prd.md",
        "# PRD\n\n## 需求 1\n用户应能 X。\n\n## 需求 2\n系统应保证 Y。\n",
    )
    assert _validate_output_artifact(state, "pm", result="PRD written.") == ""


# ── no output_file declared (e.g. synthesis agent) ─────────────────

def test_validate_skips_when_no_output_file_declared(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Agents with no ``output_file`` only have their result scanned;
    clean result returns empty.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["synth"])
    state.agents["synth"].spec = fake_agent_spec(id="synth", output_file="")
    _patch_cwd(monkeypatch, tmp_path)
    assert _validate_output_artifact(state, "synth", result="all good.") == ""


# ── empty input ────────────────────────────────────────────────────

def test_validate_empty_result_and_no_file_is_ok(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """An agent that produced nothing is not a validation failure — it's a
    different problem caught elsewhere (fail markers, retry loop, etc.).
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    assert _validate_output_artifact(state, "pm", result="") == ""


# ── sentinel buried past 8 KiB head is NOT scanned (deliberate cap) ──

def test_validate_full_scan_catches_leak_past_8kib(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Updated for F2 (2026-05-25): the 8 KiB cap was removed because it
    created a false sense of safety — the SecurityExpert false positive
    sat at line 12 (well inside the cap), and a real tail-leak past
    the cap was silently let through. Post-F2 the scan reads the full
    artifact; a raw (un-quoted) sentinel anywhere in the file trips.

    To exempt a legitimate inline citation of the token, wrap it in
    inline backticks ``` `<｜｜DSML｜｜tool_call>` ``` or a fenced
    block — both are scrubbed by ``_strip_code_evidence`` before the
    sentinel scan (see ``test_validate_strips_*_citation`` in
    ``test_crew_failure_hardening.py``).
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    padding = "clean line\n" * 1000   # ~ 11 KiB
    _write_workspace_file(
        tmp_path, "prd.md",
        padding + "\n附录：演示 <｜｜DSML｜｜tool_call> 这种 token 长这样。\n",
    )
    # Pre-F2: returned "" (capped at 8 KiB, leak past it unseen).
    # Post-F2: returns sentinel violation — bare-quoted token in prose.
    issue = _validate_output_artifact(state, "pm", result="")
    assert issue, "F2 full-scan should catch the tail leak"
    assert "sentinel" in issue


def test_validate_inline_backtick_citation_past_8kib_passes(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Companion to the test above: when the deep-body sentinel mention
    is wrapped in inline backticks (legitimate citation), the
    ``_strip_code_evidence`` scrubber removes it BEFORE sentinel scan,
    so the full-scan does NOT trip. Pins the F1 inline-strip combined
    with F2 full-scan working together.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    padding = "clean line\n" * 1000
    _write_workspace_file(
        tmp_path, "prd.md",
        padding + "\n附录：演示 `<｜｜DSML｜｜tool_call>` 这种 token 长这样。\n",
    )
    assert _validate_output_artifact(state, "pm", result="") == ""


# ── fenced code block exemption (legitimate quoted evidence) ──────

def test_validate_ignores_sentinel_inside_fenced_code_block(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Regression for 2026-05-22 false-positive: the implementer wrote
    ``changes.md`` explaining why it could not proceed and quoted the
    upstream broken DSML block inside a ``` fence as evidence. The
    validator must NOT flag fenced-quoted sentinels — otherwise every
    QA / bail-out report that documents a failure trips the check.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["impl"])
    state.agents["impl"].spec = fake_agent_spec(id="impl", output_file="changes.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "changes.md",
        "# Implementer 阶段：阻塞\n\n"
        "上游 design.md 完整内容：\n\n"
        "```\n"
        "我需要先读取 PRD\n\n"
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"Read\">...\n"
        "</｜｜DSML｜｜tool_calls>\n"
        "```\n\n"
        "## 文件改动清单\n\n无。\n",
    )
    assert _validate_output_artifact(state, "impl", result="") == ""


def test_validate_still_catches_sentinel_outside_fence(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Companion to the fence-exemption test: if the leak is in the
    narrative prose (not inside a fence), the validator must still fire.
    Otherwise the fence-strip would be a free bypass for real corruption.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "prd.md",
        "# PRD\n\n"
        "```python\nprint('this is a real code example')\n```\n\n"
        "## 需求\n<｜｜DSML｜｜tool_call name=\"Read\">  ← leak in prose, not fence\n",
    )
    issue = _validate_output_artifact(state, "pm", result="")
    assert issue
    assert "sentinel" in issue


def test_validate_handles_unclosed_fence(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Defensive: a content with an unclosed ``` opens a fence that
    swallows everything to EOF. The strip function should not crash; the
    practical effect is that a sentinel buried after an unclosed fence
    gets ignored (acceptable — those docs are already malformed enough
    to need human review).
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "prd.md",
        "# PRD\n\n```\nunclosed fence ...\n<｜｜DSML｜｜tool_call>\n",
    )
    # Should not raise; sentinel inside unclosed fence is ignored.
    assert _validate_output_artifact(state, "pm", result="") == ""


# ── quarantine of invalid output_file on validation failure ───────

def test_wrapper_quarantines_invalid_output_on_validate_failure(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch, tmp_path,
):
    """When validation fails twice, the wrapper must rename the corrupt
    ``output_file`` to ``<name>.invalid`` so the scheduler's
    partial-delivery rule does NOT let downstream agents cascade-read the
    garbage (regression 2026-05-22: PM/architect/implementer all ran on
    the same DSML leak because each FAILED upstream still had a
    non-empty output_file present).
    """
    import threading
    import uuid
    from larkhelm.crew import _runner as cr
    from larkhelm.crew_types import (
        AgentState, AgentStatus, CrewPlan, CrewState,
    )

    spec = fake_agent_spec(id="pm", output_file="prd.md", task_profile="planner")
    plan = CrewPlan(title="t", agents=[spec], synthesis_prompt="")
    state = CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents={"pm": AgentState(spec=spec)},
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="dev",
    )
    from larkhelm import chat_state
    monkeypatch.setattr(chat_state, "_get_cwd", lambda chat_id: str(tmp_path))

    # Seed a pre-existing corrupt prd.md (e.g. left over from an earlier
    # failed run — this is exactly the on-disk state observed 2026-05-22).
    ws = tmp_path / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    leak = (
        "我需要先读取 PRD\n"
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"Read\">...\n"
        "</｜｜DSML｜｜tool_calls>\n"
    ) * 20
    (ws / "prd.md").write_text(leak, encoding="utf-8")

    def fake_run_agent(s, aid):
        return leak  # both attempts produce the same corrupt output
    monkeypatch.setattr(cr, "_run_agent", fake_run_agent)
    monkeypatch.setattr(cr, "_sync_output_file", lambda st, aid: "")
    monkeypatch.setattr(cr, "emit_agent_failure", lambda *a, **k: None)
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)

    cr._run_agent_wrapper(state, "pm")

    # The corrupt file should now be quarantined under .invalid suffix.
    assert not (ws / "prd.md").exists(), (
        "prd.md must be moved aside so the scheduler's partial-delivery "
        "rule does NOT see a 'delivered' file"
    )
    assert (ws / "prd.md.invalid").exists(), (
        "the original bytes must be preserved under .invalid for user "
        "inspection rather than silently deleted"
    )


def test_wrapper_does_not_quarantine_on_clean_output(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch, tmp_path,
):
    """Companion: clean runs must NOT produce a .invalid file. Pinned to
    catch a regression where the quarantine fires unconditionally.
    """
    import threading
    import uuid
    from larkhelm.crew import _runner as cr
    from larkhelm.crew_types import (
        AgentState, AgentStatus, CrewPlan, CrewState,
    )

    spec = fake_agent_spec(id="pm", output_file="prd.md", task_profile="planner")
    plan = CrewPlan(title="t", agents=[spec], synthesis_prompt="")
    state = CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents={"pm": AgentState(spec=spec)},
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="dev",
    )
    from larkhelm import chat_state
    monkeypatch.setattr(chat_state, "_get_cwd", lambda chat_id: str(tmp_path))

    ws = tmp_path / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    clean = "# PRD\n\n## 需求 1\n用户应能 X。\n" * 20
    (ws / "prd.md").write_text(clean, encoding="utf-8")

    monkeypatch.setattr(cr, "_run_agent", lambda s, aid: clean)
    monkeypatch.setattr(cr, "_sync_output_file", lambda st, aid: "")
    monkeypatch.setattr(cr, "emit_agent_failure", lambda *a, **k: None)

    cr._run_agent_wrapper(state, "pm")

    assert (ws / "prd.md").exists()
    assert not (ws / "prd.md.invalid").exists()


# ── wrapper integration: tainted output retries once, then circuit-breaks ──

def test_wrapper_marks_failed_on_persistent_sentinel_leak(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch, tmp_path,
):
    """End-to-end: when ``_run_agent`` keeps returning a DSML-tainted result,
    the wrapper should retry once and then mark the agent FAILED with
    ``stage="validate"`` (not silently DONE). This is the production
    regression that motivated the validator.
    """
    import threading
    import uuid
    from larkhelm.crew import _runner as cr
    from larkhelm.crew_types import (
        AgentState, AgentStatus, CrewPlan, CrewState,
    )

    spec = fake_agent_spec(id="pm", output_file="prd.md",
                           task_profile="planner")
    plan = CrewPlan(title="t", agents=[spec], synthesis_prompt="")
    state = CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents={"pm": AgentState(spec=spec)},
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="dev",
    )

    # _get_cwd reads from chat_state; redirect to tmp_path for the validator.
    from larkhelm import chat_state
    monkeypatch.setattr(chat_state, "_get_cwd", lambda chat_id: str(tmp_path))

    leak = (
        "我需要先读取 PRD 和了解现有代码结构才能完成设计。\n\n"
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"Read\">...\n"
        "</｜｜DSML｜｜tool_calls>\n"
    ) * 20  # make it ≥ 200 chars so the persist safety-net writes the file

    call_log: list[str] = []
    def fake_run_agent(s, aid):
        call_log.append(aid)
        return leak
    monkeypatch.setattr(cr, "_run_agent", fake_run_agent)
    monkeypatch.setattr(cr, "_sync_output_file", lambda st, aid: "")

    # Capture the emit_agent_failure stage tag.
    import larkhelm.crew._failure_card as _fc
    fail_calls: list[tuple] = []
    monkeypatch.setattr(
        _fc, "emit_agent_failure",
        lambda *a, **k: fail_calls.append((a, k)),
    )
    # Also stub it on the crew runner side (it imports the symbol directly).
    monkeypatch.setattr(cr, "emit_agent_failure",
                        lambda *a, **k: fail_calls.append((a, k)))
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)

    cr._run_agent_wrapper(state, "pm")

    # Pinned behaviors:
    assert call_log == ["pm", "pm"], (
        f"expected 1 initial + 1 retry, got {call_log}"
    )
    assert len(fail_calls) == 1, (
        f"emit_agent_failure should fire exactly once, got {len(fail_calls)}"
    )
    args, kwargs = fail_calls[0]
    # signature: (state, agent_id, stage, exc)
    assert args[2] == "validate", f"expected stage='validate', got {args[2]!r}"
    assert "contract violation" in str(args[3]).lower()


def test_wrapper_passes_clean_output_through(
    init_test_config, fake_agent_spec, fake_card_sender, monkeypatch, tmp_path,
):
    """Companion to above: clean markdown output reaches DONE on first try,
    no retry, no failure card. Ensures the validator does not regress the
    happy path.
    """
    import threading
    import uuid
    from larkhelm.crew import _runner as cr
    from larkhelm.crew_types import (
        AgentState, AgentStatus, CrewPlan, CrewState,
    )

    spec = fake_agent_spec(id="pm", output_file="prd.md",
                           task_profile="planner")
    plan = CrewPlan(title="t", agents=[spec], synthesis_prompt="")
    state = CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents={"pm": AgentState(spec=spec)},
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="dev",
    )
    from larkhelm import chat_state
    monkeypatch.setattr(chat_state, "_get_cwd", lambda chat_id: str(tmp_path))

    # Pre-create a clean prd.md so persist_safety_net sees a healthy file
    # (writes only if file is missing / much smaller than result).
    ws = tmp_path / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    clean = "# PRD\n\n## 需求 1\n用户应能 X。\n\n## 需求 2\n系统应保证 Y。\n" * 20
    (ws / "prd.md").write_text(clean, encoding="utf-8")

    call_log: list[str] = []
    def fake_run_agent(s, aid):
        call_log.append(aid)
        return clean
    monkeypatch.setattr(cr, "_run_agent", fake_run_agent)
    monkeypatch.setattr(cr, "_sync_output_file", lambda st, aid: "")

    fail_calls: list[tuple] = []
    monkeypatch.setattr(cr, "emit_agent_failure",
                        lambda *a, **k: fail_calls.append((a, k)))

    cr._run_agent_wrapper(state, "pm")

    assert call_log == ["pm"]
    assert state.agents["pm"].status == AgentStatus.DONE
    assert fail_calls == []
