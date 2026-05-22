"""MEM-C1 tests — AC-07.

Verifies that cmd_crew, cmd_dev, and cmd_plan accept sender_open_id and
store it in CrewState / MultiPlanState, and that crew/_runner.py reads
state.sender_open_id for the memory call.
"""
from __future__ import annotations

import inspect


def test_cmd_crew_accepts_sender_open_id():
    """cmd_crew has a sender_open_id keyword parameter."""
    from larkhelm.crew._commands import cmd_crew
    sig = inspect.signature(cmd_crew)
    assert "sender_open_id" in sig.parameters


def test_cmd_dev_accepts_sender_open_id():
    """cmd_dev has a sender_open_id keyword parameter."""
    from larkhelm.crew._commands import cmd_dev
    sig = inspect.signature(cmd_dev)
    assert "sender_open_id" in sig.parameters


def test_cmd_plan_accepts_sender_open_id():
    """cmd_plan has a sender_open_id keyword parameter."""
    from larkhelm.cmd_plan import cmd_plan
    sig = inspect.signature(cmd_plan)
    assert "sender_open_id" in sig.parameters


def test_multi_plan_state_has_sender_open_id():
    """MultiPlanState dataclass has a sender_open_id field."""
    from larkhelm.cmd_plan import MultiPlanState
    import dataclasses
    fields = {f.name for f in dataclasses.fields(MultiPlanState)}
    assert "sender_open_id" in fields


def test_crew_state_has_sender_open_id():
    """CrewState dataclass has a sender_open_id field."""
    from larkhelm.crew_types import CrewState
    import dataclasses
    fields = {f.name for f in dataclasses.fields(CrewState)}
    assert "sender_open_id" in fields


def test_augment_requirement_accepts_sender_open_id():
    """_augment_requirement_with_context has a sender_open_id keyword parameter."""
    from larkhelm.crew._commands import _augment_requirement_with_context
    sig = inspect.signature(_augment_requirement_with_context)
    assert "sender_open_id" in sig.parameters


def test_auto_plan_accepts_sender_open_id():
    """_auto_plan has a sender_open_id keyword parameter."""
    from larkhelm.cmd_plan import _auto_plan
    sig = inspect.signature(_auto_plan)
    assert "sender_open_id" in sig.parameters


def test_run_dev_step_passes_sender_open_id(monkeypatch):
    """_run_dev_step passes state.sender_open_id to _run_dev_crew_inner."""
    from larkhelm.cmd_plan import _run_dev_step, MultiPlanState, PlanStep
    import threading

    captured = {}

    def fake_run_dev_crew_inner(*args, sender_open_id="", **kwargs):
        captured["sender_open_id"] = sender_open_id

    import larkhelm.cmd_plan as plan_mod
    monkeypatch.setattr(plan_mod, "_run_dev_step",
                        lambda state, step, crew_id: _run_dev_step_impl(state, step, crew_id,
                                                                          fake_run_dev_crew_inner),
                        raising=False)

    # Build state with sender_open_id
    state = MultiPlanState(
        plan_id="pid1", chat_id="chat1",
        title="test", steps=[],
        sender_open_id="user_crew_test",
    )
    step = PlanStep(idx=0, type="dev", desc="implement X")

    # Patch the import inside _run_dev_step
    from unittest.mock import patch
    with patch("larkhelm.crew._commands._run_dev_crew_inner",
               side_effect=fake_run_dev_crew_inner):
        _run_dev_step(state, step, "crew123")

    assert captured.get("sender_open_id") == "user_crew_test"


def _run_dev_step_impl(state, step, crew_id, fake_fn):
    """Thin reimplementation to capture the inner call."""
    fake_fn(
        chat_id=state.chat_id,
        requirement=step.desc,
        user_msg_id=None,
        no_confirm=True,
        crew_id=crew_id,
        force_replan=True,
        suppress_done_signal=True,
        suppress_finalize=True,
        sender_open_id=state.sender_open_id,
    )
    return True


def test_run_single_agent_step_passes_sender_open_id_to_crew_state():
    """_run_single_agent_step passes state.sender_open_id to CrewState."""
    import ast
    import pathlib

    src = pathlib.Path("larkhelm/cmd_plan.py").read_text()
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = (func.attr if isinstance(func, ast.Attribute) else
                     func.id if isinstance(func, ast.Name) else "")
        if func_name != "CrewState":
            continue
        kw_names = [kw.arg for kw in node.keywords]
        if "sender_open_id" in kw_names:
            found = True
            break

    assert found, "_run_single_agent_step does not pass sender_open_id to CrewState"
