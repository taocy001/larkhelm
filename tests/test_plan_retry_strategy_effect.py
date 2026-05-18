"""AC-05 — P3 REQ-04 decide_retry_action routing.

The /plan failure branch was previously hardcoded to consult the
``auto_retried`` counter regardless of ``plan_retry_strategy``. After
the REQ-04 fix the strategy actually shapes the routing:

* ``"off"``    → counter-driven auto-retry (status quo).
* ``"now"``    → PlanRetryEngine decides per-step.
* ``"manual"`` → always defer to the user card.
"""
from __future__ import annotations

import unittest

from larkhelm.cmd_plan import decide_retry_action


class TestDecideRetryAction(unittest.TestCase):

    def test_off_within_budget_auto_retries(self) -> None:
        self.assertEqual(
            decide_retry_action("off", 0, 0, 2),
            ("auto_retry", "below_threshold"),
        )
        self.assertEqual(
            decide_retry_action("off", 0, 1, 2),
            ("auto_retry", "below_threshold"),
        )
        self.assertEqual(
            decide_retry_action("off", 0, 2, 2),
            ("user_prompt", "retries_exhausted"),
        )

    def test_now_uses_step_retry_count(self) -> None:
        # ``auto_retried`` is irrelevant under "now" — the engine reads
        # ``step.retry_count`` instead.
        action, reason = decide_retry_action("now", 0, 99, 2)
        self.assertEqual(action, "auto_retry")
        self.assertEqual(reason, "below_threshold")

        self.assertEqual(
            decide_retry_action("now", 2, 0, 2),
            ("user_prompt", "retries_exhausted"),
        )
        self.assertEqual(
            decide_retry_action("now", 5, 0, 2),
            ("user_prompt", "retries_exhausted"),
        )

    def test_manual_skips_auto(self) -> None:
        # Manual always defers to the user card, regardless of counters.
        self.assertEqual(
            decide_retry_action("manual", 0, 0, 99),
            ("user_prompt", "manual_required"),
        )
        self.assertEqual(
            decide_retry_action("manual", 10, 10, 1),
            ("user_prompt", "manual_required"),
        )

    def test_unknown_strategy_collapses_to_off(self) -> None:
        self.assertEqual(
            decide_retry_action("hurry", 0, 0, 1),
            ("auto_retry", "below_threshold"),
        )
        self.assertEqual(
            decide_retry_action("", 0, 1, 1),
            ("user_prompt", "retries_exhausted"),
        )


if __name__ == "__main__":
    unittest.main()
