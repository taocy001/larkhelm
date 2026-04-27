"""
P3 — crew/_scheduler.py unit tests

Coverage:
  - _detect_cycle            cycle dependency detection
  - _topo_waves              BFS topological layering
  - _topo_waves_subset       subset topological layering
  - _get_failed_dep          failed dependency propagation
  - _resolve_prompt          placeholder substitution
"""
import atexit
import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# ── Initialize config ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_sched_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "default_cwd": _TMP,
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.crew._scheduler import (
    _detect_cycle, _topo_waves, _topo_waves_subset, _get_failed_dep, _resolve_prompt,
)
from larkhelm.crew_types import (
    AgentSpec, AgentState, AgentStatus, CrewPlan, CrewState,
)


# ── Helper: quickly build an AgentSpec ───────────────────────────

def _spec(id: str, depends_on: list[str] = None) -> AgentSpec:
    return AgentSpec(
        id=id, role=f"role_{id}", model="claude",
        system="", prompt=f"prompt for {id}",
        depends_on=depends_on or [],
        timeout=300,
    )


def _state(spec: AgentSpec, status: AgentStatus = AgentStatus.PENDING,
           result: str = "", error: str = "") -> AgentState:
    return AgentState(spec=spec, status=status, result=result, error=error)


def _make_crew_state(agents: list[AgentSpec],
                     agent_states: dict[str, AgentState],
                     chat_id: str = "test_chat",
                     crew_id: str = "crew_001") -> CrewState:
    plan = CrewPlan(title="Test Plan", agents=agents)
    return CrewState(
        crew_id=crew_id, chat_id=chat_id, plan=plan, agents=agent_states,
    )


# ═══════════════════════════════════════════════════════════════════
#  _detect_cycle
# ═══════════════════════════════════════════════════════════════════

class TestDetectCycle(unittest.TestCase):
    def test_no_deps_no_cycle(self):
        agents = [{"id": "a1", "depends_on": []}, {"id": "a2", "depends_on": []}]
        self.assertIsNone(_detect_cycle(agents))

    def test_linear_chain_no_cycle(self):
        agents = [
            {"id": "a1", "depends_on": []},
            {"id": "a2", "depends_on": ["a1"]},
            {"id": "a3", "depends_on": ["a2"]},
        ]
        self.assertIsNone(_detect_cycle(agents))

    def test_diamond_no_cycle(self):
        agents = [
            {"id": "a1", "depends_on": []},
            {"id": "a2", "depends_on": ["a1"]},
            {"id": "a3", "depends_on": ["a1"]},
            {"id": "a4", "depends_on": ["a2", "a3"]},
        ]
        self.assertIsNone(_detect_cycle(agents))

    def test_self_loop_detected(self):
        agents = [{"id": "a1", "depends_on": ["a1"]}]
        result = _detect_cycle(agents)
        self.assertIsNotNone(result)
        self.assertIn("a1", result)

    def test_two_node_cycle(self):
        agents = [
            {"id": "a1", "depends_on": ["a2"]},
            {"id": "a2", "depends_on": ["a1"]},
        ]
        result = _detect_cycle(agents)
        self.assertIsNotNone(result)

    def test_three_node_cycle(self):
        agents = [
            {"id": "a1", "depends_on": ["a3"]},
            {"id": "a2", "depends_on": ["a1"]},
            {"id": "a3", "depends_on": ["a2"]},
        ]
        result = _detect_cycle(agents)
        self.assertIsNotNone(result)

    def test_cycle_returns_path(self):
        agents = [
            {"id": "a1", "depends_on": ["a2"]},
            {"id": "a2", "depends_on": ["a1"]},
        ]
        path = _detect_cycle(agents)
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)

    def test_empty_agents(self):
        self.assertIsNone(_detect_cycle([]))


# ═══════════════════════════════════════════════════════════════════
#  _topo_waves
# ═══════════════════════════════════════════════════════════════════

class TestTopoWaves(unittest.TestCase):
    def test_single_agent_one_wave(self):
        agents = [_spec("a1")]
        waves = _topo_waves(agents)
        self.assertEqual(len(waves), 1)
        self.assertEqual(waves[0][0].id, "a1")

    def test_two_independent_agents_one_wave(self):
        agents = [_spec("a1"), _spec("a2")]
        waves = _topo_waves(agents)
        self.assertEqual(len(waves), 1)
        ids = {a.id for a in waves[0]}
        self.assertEqual(ids, {"a1", "a2"})

    def test_linear_chain_sequential_waves(self):
        agents = [_spec("a1"), _spec("a2", ["a1"]), _spec("a3", ["a2"])]
        waves = _topo_waves(agents)
        self.assertEqual(len(waves), 3)
        self.assertEqual(waves[0][0].id, "a1")
        self.assertEqual(waves[1][0].id, "a2")
        self.assertEqual(waves[2][0].id, "a3")

    def test_diamond_three_waves(self):
        # a1 → a2, a3 → a4
        agents = [_spec("a1"), _spec("a2", ["a1"]), _spec("a3", ["a1"]), _spec("a4", ["a2", "a3"])]
        waves = _topo_waves(agents)
        self.assertEqual(len(waves), 3)
        self.assertEqual(waves[0][0].id, "a1")
        self.assertEqual({a.id for a in waves[1]}, {"a2", "a3"})
        self.assertEqual(waves[2][0].id, "a4")

    def test_all_agents_covered(self):
        agents = [_spec("a1"), _spec("a2", ["a1"]), _spec("a3"), _spec("a4", ["a2", "a3"])]
        waves = _topo_waves(agents)
        all_ids = {a.id for wave in waves for a in wave}
        self.assertEqual(all_ids, {"a1", "a2", "a3", "a4"})

    def test_empty_agents_no_waves(self):
        self.assertEqual(_topo_waves([]), [])


# ═══════════════════════════════════════════════════════════════════
#  _topo_waves_subset
# ═══════════════════════════════════════════════════════════════════

class TestTopoWavesSubset(unittest.TestCase):
    def test_subset_excludes_external_dep(self):
        # Full chain a1, a2, a3; only processing {a2, a3}, a1 is considered already satisfied
        agents = [_spec("a1"), _spec("a2", ["a1"]), _spec("a3", ["a2"])]
        waves = _topo_waves_subset(agents, {"a2", "a3"})
        # a1 is not in the subset, so a2 should be in the first wave
        self.assertEqual(waves[0][0].id, "a2")
        self.assertEqual(waves[1][0].id, "a3")

    def test_subset_single_agent(self):
        agents = [_spec("a1"), _spec("a2", ["a1"])]
        waves = _topo_waves_subset(agents, {"a2"})
        self.assertEqual(len(waves), 1)
        self.assertEqual(waves[0][0].id, "a2")

    def test_empty_subset(self):
        agents = [_spec("a1"), _spec("a2")]
        self.assertEqual(_topo_waves_subset(agents, set()), [])

    def test_subset_with_internal_dep(self):
        # Dependencies within the subset still take effect
        agents = [_spec("a1"), _spec("a2", ["a1"]), _spec("a3", ["a2"])]
        waves = _topo_waves_subset(agents, {"a1", "a2", "a3"})
        self.assertEqual(len(waves), 3)


# ═══════════════════════════════════════════════════════════════════
#  _get_failed_dep
# ═══════════════════════════════════════════════════════════════════

class TestGetFailedDep(unittest.TestCase):
    def _make(self, specs_deps, statuses):
        """specs_deps: list of (id, [deps]); statuses: dict of {id: AgentStatus}"""
        specs = [_spec(id_, deps) for id_, deps in specs_deps]
        states = {
            id_: _state(_spec(id_, deps), status=statuses.get(id_, AgentStatus.PENDING))
            for id_, deps in specs_deps
        }
        return _make_crew_state(specs, states)

    def test_no_deps_returns_none(self):
        state = self._make([("a1", [])], {})
        spec_a1 = state.plan.agents[0]
        self.assertIsNone(_get_failed_dep(state, spec_a1))

    def test_pending_dep_not_failed(self):
        state = self._make([("a1", []), ("a2", ["a1"])],
                           {"a1": AgentStatus.PENDING})
        spec_a2 = next(s for s in state.plan.agents if s.id == "a2")
        self.assertIsNone(_get_failed_dep(state, spec_a2))

    def test_done_dep_not_failed(self):
        state = self._make([("a1", []), ("a2", ["a1"])],
                           {"a1": AgentStatus.DONE})
        spec_a2 = next(s for s in state.plan.agents if s.id == "a2")
        self.assertIsNone(_get_failed_dep(state, spec_a2))

    def test_failed_dep_returns_id(self):
        state = self._make([("a1", []), ("a2", ["a1"])],
                           {"a1": AgentStatus.FAILED})
        spec_a2 = next(s for s in state.plan.agents if s.id == "a2")
        result = _get_failed_dep(state, spec_a2)
        self.assertEqual(result, "a1")

    def test_cancelled_dep_returns_id(self):
        state = self._make([("a1", []), ("a2", ["a1"])],
                           {"a1": AgentStatus.CANCELLED})
        spec_a2 = next(s for s in state.plan.agents if s.id == "a2")
        result = _get_failed_dep(state, spec_a2)
        self.assertEqual(result, "a1")

    def test_transitive_failed_dep(self):
        # a1 failed → a2 depends on a1 → a3 depends on a2: _get_failed_dep(a3) should return a1
        state = self._make(
            [("a1", []), ("a2", ["a1"]), ("a3", ["a2"])],
            {"a1": AgentStatus.FAILED, "a2": AgentStatus.PENDING},
        )
        spec_a3 = next(s for s in state.plan.agents if s.id == "a3")
        result = _get_failed_dep(state, spec_a3)
        self.assertIsNotNone(result)

    def test_needs_retry_flag_exempts_failure(self):
        state = self._make([("a1", []), ("a2", ["a1"])],
                           {"a1": AgentStatus.FAILED})
        # Set a1's needs_retry to True → should not block downstream agents
        state.agents["a1"].needs_retry = True
        spec_a2 = next(s for s in state.plan.agents if s.id == "a2")
        result = _get_failed_dep(state, spec_a2)
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════
#  _resolve_prompt
# ═══════════════════════════════════════════════════════════════════

class TestResolvePrompt(unittest.TestCase):
    def _make_done_state(self, agent_id: str, result: str) -> AgentState:
        spec = _spec(agent_id)
        st = _state(spec, status=AgentStatus.DONE, result=result)
        return st

    def _crew(self, agents_done: dict[str, str]) -> CrewState:
        """agents_done: dict of {agent_id: result_text}"""
        specs = [_spec(aid) for aid in agents_done]
        states = {aid: self._make_done_state(aid, res) for aid, res in agents_done.items()}
        return _make_crew_state(specs, states)

    def test_no_placeholder_unchanged(self):
        state = self._crew({"agent_1": "result1"})
        template = "No placeholder here"
        result = _resolve_prompt(template, state)
        self.assertEqual(result, template)

    def test_placeholder_replaced_with_result(self):
        state = self._crew({"agent_1": "summary output"})
        template = "Use this: {agent_1_result}"
        result = _resolve_prompt(template, state)
        self.assertIn("summary output", result)
        self.assertNotIn("{agent_1_result}", result)

    def test_multiple_placeholders(self):
        state = self._crew({"agent_1": "out1", "agent_2": "out2"})
        template = "{agent_1_result} and {agent_2_result}"
        result = _resolve_prompt(template, state)
        self.assertIn("out1", result)
        self.assertIn("out2", result)

    def test_nonexistent_agent_placeholder(self):
        state = self._crew({"agent_1": "out"})
        template = "{agent_99_result}"
        result = _resolve_prompt(template, state)
        self.assertIn("does not exist", result)

    def test_failed_agent_placeholder(self):
        specs = [_spec("agent_1")]
        states = {"agent_1": _state(_spec("agent_1"), status=AgentStatus.FAILED, error="crash")}
        state = _make_crew_state(specs, states)
        result = _resolve_prompt("{agent_1_result}", state)
        self.assertIn("执行失败", result)
        self.assertIn("crash", result)

    def test_pending_agent_placeholder(self):
        specs = [_spec("agent_1")]
        states = {"agent_1": _state(_spec("agent_1"), status=AgentStatus.PENDING)}
        state = _make_crew_state(specs, states)
        result = _resolve_prompt("{agent_1_result}", state)
        self.assertIn("未就绪", result)

    def test_long_result_preview_truncated(self):
        from larkhelm.crew_types import CREW_RESULT_PREVIEW
        long_result = "x" * (CREW_RESULT_PREVIEW + 100)
        state = self._crew({"agent_1": long_result})
        result = _resolve_prompt("{agent_1_result}", state)
        self.assertIn("截断", result)


if __name__ == "__main__":
    unittest.main()
