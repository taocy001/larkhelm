"""AC-07: rank_for_task routes simple complexity → cheap, complex → worker."""
import unittest

from larkhelm.agent_hub.intent_types import TaskProfile
from larkhelm.agent_hub.model_selector import resolve_backend_for_task
from larkhelm.backend_registry import BackendRegistry


def _make_registry():
    registry = BackendRegistry()
    registry.load([
        {"id": "cheap-bot", "provider": "gemini_cli", "display_name": "Cheap",
         "role": "worker", "tags": ["cheap", "fast", "tools"],
         "capability_scores": {"instant": 0.9, "code": 0.4, "reasoning": 0.3},
         "latency_tier": "fast",
         "cost_per_1k_input": 0.05, "cost_per_1k_output": 0.20},
        {"id": "worker-bot", "provider": "claude_cli", "display_name": "Worker",
         "role": "worker", "tags": ["tools"],
         "capability_scores": {"code": 0.95, "reasoning": 0.95, "instant": 0.0},
         "latency_tier": "slow",
         "cost_per_1k_input": 3.0, "cost_per_1k_output": 15.0},
    ])
    return registry


class TestTaskAwareRouting(unittest.TestCase):

    def test_simple_complexity_routes_to_cheap(self):
        registry = _make_registry()
        simple = TaskProfile(
            complexity="simple",
            required_capabilities={"instant": 1.0},
            latency_pref="fast",
        )
        ranked = registry.rank_for_task(simple)
        self.assertEqual(ranked[0].id, "cheap-bot")

    def test_complex_complexity_routes_to_worker(self):
        registry = _make_registry()
        complex_p = TaskProfile(
            complexity="complex",
            required_capabilities={"code": 1.0, "reasoning": 0.9},
            latency_pref="medium",
        )
        ranked = registry.rank_for_task(complex_p)
        self.assertEqual(ranked[0].id, "worker-bot")

    def test_resolve_backend_for_task_via_singleton(self):
        # End-to-end through model_selector — we patch the singleton.
        import larkhelm.backend_registry as br_mod
        registry = _make_registry()
        original = br_mod.BACKEND_REGISTRY
        br_mod.BACKEND_REGISTRY = registry
        try:
            simple = TaskProfile(
                complexity="simple",
                required_capabilities={"instant": 1.0},
                latency_pref="fast",
            )
            spec = resolve_backend_for_task("chat-x", simple)
            self.assertEqual(spec.id, "cheap-bot")
        finally:
            br_mod.BACKEND_REGISTRY = original

    def test_force_backend_id_wins(self):
        import larkhelm.backend_registry as br_mod
        registry = _make_registry()
        original = br_mod.BACKEND_REGISTRY
        br_mod.BACKEND_REGISTRY = registry
        try:
            spec = resolve_backend_for_task(
                "chat-x",
                TaskProfile(complexity="simple"),
                force_backend_id="worker-bot",
            )
            self.assertEqual(spec.id, "worker-bot")
        finally:
            br_mod.BACKEND_REGISTRY = original


if __name__ == "__main__":
    unittest.main()
