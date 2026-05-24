"""Tests for ``larkhelm.crew._pipeline._make_dev_pipeline`` (Phase C: AC-05).

Key assertion: no agent in the dev pipeline carries ``model="claude"``;
they all use ``task_profile`` for backend selection instead.
"""
from __future__ import annotations

import pytest


def test_make_dev_pipeline_returns_6_agents(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("test req", "/tmp/cwd", no_confirm=True)
    assert len(plan.agents) == 6


def test_no_agent_uses_model_claude(init_test_config):
    """AC-05: task_profile-driven dispatch means no agent should hardcode
    model='claude' (which would short-circuit the resolver)."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("test req", "/tmp/cwd", no_confirm=True)
    for spec in plan.agents:
        assert spec.model != "claude", (
            f"agent {spec.id!r} still hardcoded model='claude' — "
            "should use task_profile instead"
        )


def test_every_agent_has_task_profile(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("test req", "/tmp/cwd", no_confirm=True)
    for spec in plan.agents:
        assert spec.task_profile != "", (
            f"agent {spec.id!r} has empty task_profile — "
            "Phase C requires explicit profile assignment"
        )


def test_task_profile_values_are_known(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    from larkhelm.crew._backend_resolver import TASK_PROFILES
    plan = _make_dev_pipeline("test req", "/tmp/cwd", no_confirm=True)
    valid = set(TASK_PROFILES)
    for spec in plan.agents:
        assert spec.task_profile in valid, (
            f"agent {spec.id!r} has unknown task_profile={spec.task_profile!r}"
        )


def test_pipeline_specific_profile_assignments(init_test_config):
    """Pin each role to its expected profile (regression guard)."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("test req", "/tmp/cwd", no_confirm=True)
    by_id = {s.id: s for s in plan.agents}
    assert by_id["pm"].task_profile == "planner"
    assert by_id["architect"].task_profile == "planner"
    assert by_id["implementer"].task_profile == "engineer"
    assert by_id["fixer"].task_profile == "engineer"
    assert by_id["qa"].task_profile == "qa"
    assert by_id["reviewer"].task_profile == "reviewer"


def test_skip_planning_drops_pm_and_architect(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True, skip_planning=True)
    ids = [s.id for s in plan.agents]
    assert "pm" not in ids
    assert "architect" not in ids
    assert "implementer" in ids


def test_skip_planning_implementer_depends_empty(init_test_config):
    """When pm/architect are dropped, implementer's dependency list shouldn't
    contain orphaned references."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True, skip_planning=True)
    impl = next(s for s in plan.agents if s.id == "implementer")
    # depends_on may be empty or contain only agents still present
    remaining = {s.id for s in plan.agents}
    for d in impl.depends_on:
        assert d in remaining


def test_breakpoint_set_when_no_confirm_false(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=False)
    pm = next(s for s in plan.agents if s.id == "pm")
    assert pm.breakpoint is True


def test_breakpoint_clear_when_no_confirm_true(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True)
    pm = next(s for s in plan.agents if s.id == "pm")
    assert pm.breakpoint is False


# ════════════════════════════════════════════════════════════════════════
#  B1 — prompt contract pins: anchor rules, ban on line numbers,
#  §6 Release Gate, architect self-check retry slot
# ════════════════════════════════════════════════════════════════════════

def test_b1_architect_has_self_retry_slot(init_test_config):
    """architect spec must carry ``retry_target=['architect']`` /
    ``max_retries=1`` so the PRD self-check gate can rerun it once."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True)
    architect = next(s for s in plan.agents if s.id == "architect")
    assert architect.retry_target == ["architect"], (
        f"architect.retry_target must be ['architect'] for B1 self-check, "
        f"got {architect.retry_target!r}"
    )
    assert architect.max_retries == 1, (
        f"architect.max_retries must be 1 for B1 self-check, "
        f"got {architect.max_retries!r}"
    )


def test_b1_architect_prompt_carries_anchor_rules(init_test_config):
    """architect system prompt must require anchors[].snippet and
    forbid line numbers (CLAUDE.md §文件清单规则)."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True)
    architect = next(s for s in plan.agents if s.id == "architect")
    sys = architect.system
    # Must demand anchors metadata
    assert "anchors" in sys
    assert "snippet" in sys
    # Must forbid line numbers explicitly
    assert "严禁" in sys and "行号" in sys
    # Must reference the grep -F locator path so downstream agents pick up
    assert "grep -F" in sys or "grep `-F`" in sys


def test_b1_implementer_and_fixer_prompt_use_grep_anchors(init_test_config):
    """implementer + fixer must instruct using ``Grep -F snippet``
    instead of line-number-based navigation."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True)
    for agent_id in ("implementer", "fixer"):
        spec = next(s for s in plan.agents if s.id == agent_id)
        sys = spec.system
        assert "anchors" in sys or "anchor" in sys, (
            f"{agent_id} prompt should reference anchors"
        )
        assert "Grep" in sys and "-F" in sys, (
            f"{agent_id} prompt should use Grep -F"
        )
        assert "行号" in sys, (
            f"{agent_id} prompt should mention 行号 (forbidden)"
        )


def test_b1_pm_prompt_has_release_gate_section(init_test_config):
    """pm system prompt must require §6 Release Gate in the PRD."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True)
    pm = next(s for s in plan.agents if s.id == "pm")
    assert "Release Gate" in pm.system
    # Mention the "pytest --collect-only" rule (B1 review口径)
    assert "--collect-only" in pm.system or "collect-only" in pm.system
    # Mention the out-of-scope ticket section
    assert "范围外问题" in pm.system


def test_b1_qa_prompt_has_release_gate_scope_split(init_test_config):
    """qa spec must distinguish task_list-inside vs outside failures."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True)
    qa = next(s for s in plan.agents if s.id == "qa")
    assert "Release Gate" in qa.system
    assert "范围外问题" in qa.system
    # Mention the mypy 传染 rule
    assert "传染" in qa.system


def test_b1_reviewer_8th_check_binds_release_gate(init_test_config):
    """reviewer's 8th item must reference Release Gate."""
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("req", "/tmp/cwd", no_confirm=True)
    reviewer = next(s for s in plan.agents if s.id == "reviewer")
    assert "Release Gate" in reviewer.system


def test_pipeline_title_uses_first_line(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    multi = "main requirement\n\n## context\nbackground stuff"
    plan = _make_dev_pipeline(multi, "/tmp/cwd", no_confirm=True)
    # First-line truncation; should not contain the second line
    assert "## context" not in plan.title
    assert "main requirement" in plan.title
