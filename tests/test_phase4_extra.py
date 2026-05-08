import unittest
import threading
from unittest.mock import patch, MagicMock

class TestAC16MultiHopDelegation(unittest.TestCase):
    def test_multi_hop_limited_to_2(self):
        """Orchestrator can delegate twice, then must answer directly (3rd hop fallback)."""
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

        def fake_run(spec, chat_id, message, cwd, cancel_ev, on_text, on_tool, on_tool_result, on_soft_timeout, images=None, **kwargs):
            if spec.id == "orch":
                current_hop = len(orch_calls)
                orch_calls.append(current_hop)
                if current_hop < 2:
                    # Delegate again
                    return f"DELEGATE specialist\nHop {current_hop} subquery\nEND_DELEGATE"
                return f"Final answer after {current_hop} hops"
            elif spec.id == "specialist":
                return "Specialist result"
            return ""

        def fake_on_tool(name, desc, tool_id=""):
            pass

        def fake_on_tool_result(tool_id, result, is_error, elapsed):
            pass

        cancel_ev = threading.Event()

        with patch("larkhelm.handlers._query._run_backend_single", side_effect=fake_run), \
             patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.all_enabled.return_value = [worker_spec]
            
            # Mock build_orchestrator_system_prompt in the orchestration module
            with patch("larkhelm.orchestration.build_orchestrator_system_prompt", return_value="System Prompt"):
                result = _do_query_with_delegation(
                    "chat1", "Go deep", orch_spec,
                    {"specialist": worker_spec}, "/tmp", cancel_ev,
                    lambda t, s="typing": None,
                    fake_on_tool, fake_on_tool_result, lambda: None,
                    hop=0
                )

        # Expected flow:
        # 1. orch_calls=[0], returns DELEGATE -> specialist runs -> calls _do_query_with_delegation(hop=1)
        # 2. orch_calls=[0, 1], returns DELEGATE -> specialist runs -> calls _do_query_with_delegation(hop=2)
        # 3. _do_query_with_delegation(hop=2) hits hop >= 2, calls _run_backend_single(orch)
        # 4. orch_calls=[0, 1, 2], returns "Final answer after 2 hops"

        self.assertEqual(orch_calls, [0, 1, 2])
        self.assertIn("Final answer after 2 hops", result)

if __name__ == "__main__":
    unittest.main()
