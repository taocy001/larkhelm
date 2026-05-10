import unittest
import threading
from unittest.mock import patch, MagicMock

class TestAC16MultiHopDelegation(unittest.TestCase):
    def test_multi_hop_limited_to_2(self):
        """Hop>=2 guard short-circuits to direct orchestrator answer (no delegation parsing).

        Phase 3 synthesis recursion was deliberately removed in commit 3f04fe6
        ("specialist output returned directly"); the `if hop >= 2` guard is retained
        as a defensive depth limit. This test exercises that guard directly by
        invoking _do_query_with_delegation with hop=2 and verifying it bypasses
        delegation, calling the orchestrator exactly once with the original message.
        """
        from larkhelm.handlers._query import _do_query_with_delegation
        from larkhelm.backend_registry import BackendSpec

        orch_spec = BackendSpec(
            id="orch", provider="claude_cli", display_name="Orchestrator",
            role="orchestrator", tags=["tools"], command="claude",
        )
        worker_spec = BackendSpec(
            id="specialist", provider="kimi_cli", display_name="Specialist",
            role="worker", tags=["tools"], command="kimi-code",
            healthy=True, enabled=True,
        )

        orch_calls = []
        spec_calls = []

        def fake_run(spec, chat_id, message, cwd, cancel_ev, on_text, on_tool, on_tool_result, on_soft_timeout, images=None, **kwargs):
            if spec.id == "orch":
                orch_calls.append(message)
                return "Final direct answer at hop>=2"
            elif spec.id == "specialist":
                spec_calls.append(message)
                return "Specialist result (should not appear)"
            return ""

        def fake_on_tool(name, desc, tool_id=""):
            pass

        def fake_on_tool_result(tool_id, result, is_error, elapsed):
            pass

        cancel_ev = threading.Event()

        with patch("larkhelm.handlers._query._run_backend_single", side_effect=fake_run), \
             patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.all_enabled.return_value = [worker_spec]
            with patch("larkhelm.orchestration.build_orchestrator_system_prompt", return_value="System Prompt"):
                result = _do_query_with_delegation(
                    "chat1", "Go deep", orch_spec,
                    {"specialist": worker_spec}, "/tmp", cancel_ev,
                    lambda t, s="typing": None,
                    fake_on_tool, fake_on_tool_result, lambda: None,
                    hop=2,
                )

        # Hop>=2 guard: orchestrator called once directly, specialist never invoked.
        self.assertEqual(len(orch_calls), 1)
        self.assertEqual(len(spec_calls), 0)
        self.assertEqual(result, "Final direct answer at hop>=2")

    def test_single_hop_delegation_returns_specialist_output(self):
        """Current single-hop design: orchestrator delegates → specialist runs → output returned directly.

        Verifies the post-3f04fe6 behavior: no synthesis recursion, the specialist's
        output is the final answer.
        """
        from larkhelm.handlers._query import _do_query_with_delegation
        from larkhelm.backend_registry import BackendSpec

        orch_spec = BackendSpec(
            id="orch", provider="claude_cli", display_name="Orchestrator",
            role="orchestrator", tags=["tools"], command="claude",
        )
        worker_spec = BackendSpec(
            id="specialist", provider="kimi_cli", display_name="Specialist",
            role="worker", tags=["tools"], command="kimi-code",
            healthy=True, enabled=True,
        )

        orch_calls = []
        spec_calls = []

        def fake_run(spec, chat_id, message, cwd, cancel_ev, on_text, on_tool, on_tool_result, on_soft_timeout, images=None, **kwargs):
            if spec.id == "orch":
                orch_calls.append(message)
                return "DELEGATE specialist\nSubquery body\nEND_DELEGATE"
            elif spec.id == "specialist":
                spec_calls.append(message)
                return "Specialist final result"
            return ""

        cancel_ev = threading.Event()

        with patch("larkhelm.handlers._query._run_backend_single", side_effect=fake_run), \
             patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.all_enabled.return_value = [worker_spec]
            with patch("larkhelm.orchestration.build_orchestrator_system_prompt", return_value="System Prompt"):
                result = _do_query_with_delegation(
                    "chat1", "Go deep", orch_spec,
                    {"specialist": worker_spec}, "/tmp", cancel_ev,
                    lambda t, s="typing": None,
                    lambda *a, **k: None,
                    lambda *a, **k: None,
                    lambda: None,
                    hop=0,
                )

        self.assertEqual(len(orch_calls), 1)
        self.assertEqual(len(spec_calls), 1)
        self.assertIn("Specialist final result", result)


if __name__ == "__main__":
    unittest.main()
