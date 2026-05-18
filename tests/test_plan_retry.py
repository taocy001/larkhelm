"""AC-06 — P3 REQ-06 plan retry engine."""
from __future__ import annotations

import unittest

from larkhelm.plan_retry import PlanRetryEngine, RetryDecision


class TestPlanRetryEngine(unittest.TestCase):

    def test_now_strategy_allows_retries_until_exhausted(self) -> None:
        engine = PlanRetryEngine("now")
        state = {"retry_count": 0, "max_retries": 2}

        decision = engine.evaluate(state)
        self.assertIsInstance(decision, RetryDecision)
        self.assertTrue(decision.should_retry)
        engine.mark_retry_attempted(state)
        self.assertEqual(state["retry_count"], 1)

        decision = engine.evaluate(state)
        self.assertTrue(decision.should_retry)
        engine.mark_retry_attempted(state)
        self.assertEqual(state["retry_count"], 2)

        decision = engine.evaluate(state)
        self.assertFalse(decision.should_retry)
        self.assertEqual(decision.reason, "retries_exhausted")

    def test_off_strategy_never_retries(self) -> None:
        engine = PlanRetryEngine("off")
        state = {"retry_count": 0, "max_retries": 5}
        decision = engine.evaluate(state)
        self.assertFalse(decision.should_retry)
        self.assertEqual(decision.reason, "disabled")

    def test_manual_strategy_returns_manual_required(self) -> None:
        engine = PlanRetryEngine("manual")
        state = {"retry_count": 0, "max_retries": 1}
        decision = engine.evaluate(state)
        self.assertTrue(decision.should_retry)
        self.assertEqual(decision.reason, "manual_required")
        self.assertEqual(decision.next_retry_at, 0.0)

    def test_unknown_strategy_collapses_to_off(self) -> None:
        engine = PlanRetryEngine("hurry")
        self.assertEqual(engine.strategy, "off")
        decision = engine.evaluate({"retry_count": 0, "max_retries": 5})
        self.assertFalse(decision.should_retry)

    def test_evaluate_does_not_mutate_state(self) -> None:
        engine = PlanRetryEngine("now")
        state = {"retry_count": 1, "max_retries": 3}
        engine.evaluate(state)
        self.assertEqual(state, {"retry_count": 1, "max_retries": 3})


if __name__ == "__main__":
    unittest.main()
