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


def test_pipeline_title_uses_first_line(init_test_config):
    from larkhelm.crew._pipeline import _make_dev_pipeline
    multi = "main requirement\n\n## context\nbackground stuff"
    plan = _make_dev_pipeline(multi, "/tmp/cwd", no_confirm=True)
    # First-line truncation; should not contain the second line
    assert "## context" not in plan.title
    assert "main requirement" in plan.title
