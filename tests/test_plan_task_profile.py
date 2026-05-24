"""Regression tests for C3 #8: plan single-agent step specs use task_profile.

Pre-C3 ``_run_single_agent_step`` built each spec with a hardcoded
``model="claude"`` or ``model="gemini"``, short-circuiting
``resolve_backend``'s task_profile path (per design.md §6.1) and
diverging from ``crew/_pipeline.py`` which had already migrated
every agent to ``model="" + task_profile=<role>``. This locked plan
sub-steps to a single CLI backend regardless of health / availability.

These tests pin:
  • each plan step type maps to the expected task_profile
  • model field is empty so the resolver's path-1 branch wins
  • the chosen profile values are all valid registry entries
"""
from __future__ import annotations

import unittest
from unittest import mock

import pytest

from larkhelm.cmd_plan import MultiPlanState, PlanStep
from larkhelm.crew._backend_resolver import TASK_PROFILES


@pytest.fixture(autouse=True)
def _bootstrap_cfg(init_test_config):
    """Ensure config module attributes (``RESPONSE_TIMEOUT`` etc.) exist
    before ``_run_single_agent_step`` reads them; ``init_test_config`` is
    declared in ``conftest.py`` with autouse=False so we opt in here."""
    yield


_EXPECTED = {
    "review": ("reviewer", "reviewer"),  # (spec.id, expected_task_profile)
    "fix":    ("fixer",    "engineer"),
    "test":   ("qa",       "qa"),
}


def _make_state() -> MultiPlanState:
    return MultiPlanState(
        plan_id="p_tp", chat_id="oc_t", title="t",
        steps=[PlanStep(idx=0, type="dev", desc="x")],
    )


class TestPlanSingleAgentSpecTaskProfile(unittest.TestCase):
    """Drive ``_run_single_agent_step`` with the inner ``_run_crew`` mocked
    so we can capture the constructed ``AgentSpec`` and assert its fields
    without standing up a real backend.
    """

    def _build_spec(self, step_type: str):
        """Run ``_run_single_agent_step`` with all side effects stubbed and
        return the AgentSpec it constructed (read from the captured CrewState)."""
        import larkhelm.cmd_plan as plan_mod

        captured = {}

        def _fake_run_crew(crew_state, _timeout):
            captured["crew_state"] = crew_state

        state = _make_state()
        step = PlanStep(idx=0, type=step_type, desc="example desc")

        with mock.patch("larkhelm.crew._runner._run_crew", side_effect=_fake_run_crew), \
             mock.patch("larkhelm.lark_client._send_card_raw", return_value="m1"), \
             mock.patch("larkhelm.lark_client._pin_task_card"), \
             mock.patch("larkhelm.chat_state._get_cwd", return_value="/tmp"):
            plan_mod._run_single_agent_step(state, step)

        self.assertIn("crew_state", captured,
                      f"_run_single_agent_step({step_type}) never invoked _run_crew")
        agents = captured["crew_state"].plan.agents
        self.assertEqual(len(agents), 1,
                         f"expected 1 spec for {step_type}, got {len(agents)}")
        return agents[0]

    def test_review_step_uses_reviewer_profile(self):
        spec = self._build_spec("review")
        expected_id, expected_profile = _EXPECTED["review"]
        self.assertEqual(spec.id, expected_id)
        self.assertEqual(spec.task_profile, expected_profile)
        self.assertEqual(spec.model, "",
                         "model must be empty so resolver path-1 picks task_profile")

    def test_fix_step_uses_engineer_profile(self):
        spec = self._build_spec("fix")
        expected_id, expected_profile = _EXPECTED["fix"]
        self.assertEqual(spec.id, expected_id)
        self.assertEqual(spec.task_profile, expected_profile)
        self.assertEqual(spec.model, "")

    def test_test_step_uses_qa_profile(self):
        spec = self._build_spec("test")
        expected_id, expected_profile = _EXPECTED["test"]
        self.assertEqual(spec.id, expected_id)
        self.assertEqual(spec.task_profile, expected_profile)
        self.assertEqual(spec.model, "")

    def test_no_plan_step_hardcodes_legacy_claude_or_gemini(self):
        """AC-05 mirror (crew/_pipeline.py): legacy direct-dispatch models
        must not leak back into plan specs after the migration."""
        for step_type in _EXPECTED:
            spec = self._build_spec(step_type)
            self.assertNotIn(
                spec.model, {"claude", "gemini", "kimi", "deepseek"},
                f"plan {step_type} step still hardcoded legacy model={spec.model!r}",
            )

    def test_chosen_profiles_are_registered(self):
        """Defence against typos like task_profile='reviewers' that would
        silently fall through to the orchestrator path."""
        valid = set(TASK_PROFILES)
        for step_type, (_id, profile) in _EXPECTED.items():
            self.assertIn(
                profile, valid,
                f"plan {step_type} step uses unknown task_profile={profile!r}",
            )


if __name__ == "__main__":
    unittest.main()
