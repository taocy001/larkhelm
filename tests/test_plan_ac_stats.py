"""Tests for plan AC stats in _pipeline.py prompts (AC-09)."""
import unittest


def _get_agent_system(agent_id: str) -> str:
    from larkhelm.crew._pipeline import _make_dev_pipeline
    plan = _make_dev_pipeline("test requirement", "/tmp", no_confirm=True)
    for spec in plan.agents:
        if spec.id == agent_id:
            return spec.system
    raise AssertionError(f"Agent '{agent_id}' not found in pipeline")


def test_qa_prompt_contains_ac_stats_section(init_test_config):
    system = _get_agent_system("qa")
    assert "## AC 统计" in system, "qa agent system prompt must contain '## AC 统计' section"


def test_qa_prompt_contains_gate_type(init_test_config):
    system = _get_agent_system("qa")
    assert "gate_type" in system, "qa agent system prompt must mention 'gate_type'"


def test_qa_prompt_contains_env_blocked(init_test_config):
    system = _get_agent_system("qa")
    assert "ENV_BLOCKED" in system, "qa agent system prompt must contain 'ENV_BLOCKED'"


def test_qa_prompt_contains_skip_enum(init_test_config):
    system = _get_agent_system("qa")
    assert "SKIP" in system, "qa agent system prompt must contain 'SKIP' outcome"


def test_pm_prompt_contains_gate_type(init_test_config):
    system = _get_agent_system("pm")
    assert "gate_type" in system, "pm agent system prompt must mention 'gate_type'"
