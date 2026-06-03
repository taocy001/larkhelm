"""Tests for crew_card._build_card rendering SKIPPED agents (AC-08)."""
import pytest
from unittest.mock import patch


def _make_state(skip_reason: str = "需要 linux/amd64，当前 darwin/arm64"):
    from larkhelm.crew_types import (
        AgentSpec, AgentState, AgentStatus, CrewState, CrewPlan,
    )

    spec = AgentSpec(
        id="eng", role="工程师", model="", system="", prompt="",
        depends_on=[], timeout=60, require_arch="linux/amd64",
    )
    plan = CrewPlan(title="Test Plan", agents=[spec])
    agent_state = AgentState(spec=spec, status=AgentStatus.SKIPPED)
    agent_state.skip_reason = skip_reason
    state = CrewState(
        crew_id="c1", chat_id="ch1", plan=plan,
        agents={"eng": agent_state},
        phase="running",
    )
    return state


def test_skip_reason_in_card_json(init_test_config):
    from larkhelm.crew_card import _build_card

    state = _make_state()
    with patch("larkhelm.crew_card._get_cwd", return_value="/tmp/fake"):
        card_json = _build_card(state)

    assert "darwin/arm64" in card_json, "skip_reason text not found in rendered card JSON"
    assert "linux/amd64" in card_json


def test_skip_reason_truncated_at_60(init_test_config):
    from larkhelm.crew_card import _build_card

    long_reason = "A" * 80
    state = _make_state(skip_reason=long_reason)
    with patch("larkhelm.crew_card._get_cwd", return_value="/tmp/fake"):
        card_json = _build_card(state)

    assert "A" * 60 in card_json
    assert "A" * 61 not in card_json


def test_empty_skip_reason_still_renders(init_test_config):
    from larkhelm.crew_card import _build_card

    state = _make_state(skip_reason="")
    with patch("larkhelm.crew_card._get_cwd", return_value="/tmp/fake"):
        card_json = _build_card(state)

    assert isinstance(card_json, str)
    assert len(card_json) > 0
