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

    def test_off_and_now_diverge_on_same_input(self) -> None:
        """Side-by-side comparison: on the **exact same** failure inputs the
        three strategies must produce observably different actions.

        Concrete scenario: step has retried once (``step_retry_count=1``),
        no auto-retries spent on this step yet (``auto_retried=0``),
        ``max_retries=2``.

          * ``off`` looks at ``auto_retried=0`` < ``max_retries=2`` → auto_retry
          * ``now`` looks at ``step_retry_count=1`` < ``max_retries=2`` → auto_retry
            but with a DIFFERENT reason path (engine.evaluate) — both
            return ``"below_threshold"`` so the *reason* matches, but
            the SOURCE that produced the decision diverges.
          * ``manual`` ignores both counters → always ``user_prompt`` with
            reason ``"manual_required"``.

        Pin the cross-strategy difference loudly: this is the regression
        bait for any future "let's simplify, they all look the same"
        refactor.
        """
        step_retry, auto_retried, max_retries = 1, 0, 2

        off_action,    off_reason    = decide_retry_action("off",    step_retry, auto_retried, max_retries)
        now_action,    now_reason    = decide_retry_action("now",    step_retry, auto_retried, max_retries)
        manual_action, manual_reason = decide_retry_action("manual", step_retry, auto_retried, max_retries)

        # off and now both auto_retry here but for different reasons —
        # off because the auto_retried budget isn't exhausted; now
        # because the per-step engine.evaluate counts step_retry_count.
        self.assertEqual(off_action,    "auto_retry")
        self.assertEqual(now_action,    "auto_retry")
        self.assertEqual(manual_action, "user_prompt",
            "manual must NEVER auto-retry, even when counters say it could")

        # Reasons are observably different across strategies.
        self.assertEqual(manual_reason, "manual_required")
        self.assertNotEqual(off_reason, manual_reason)
        self.assertNotEqual(now_reason, manual_reason)

    def test_off_and_now_diverge_when_counters_disagree(self) -> None:
        """Stronger divergence test: when the two counters point in
        OPPOSITE directions, ``off`` and ``now`` produce different actions.

        Setup: step has retried 5 times already (``step_retry_count=5``,
        well over the budget), but no auto-retries spent on THIS step yet
        (``auto_retried=0``, fresh budget). ``max_retries=2``.

          * ``off`` reads ``auto_retried=0`` → still has budget → auto_retry
          * ``now`` reads ``step_retry_count=5`` → exhausted → user_prompt

        This is the case the engine was designed for: per-step retry
        tracking. If a future refactor flattens the engine back into
        the counter path, this test fires.
        """
        action_off, _ = decide_retry_action("off", 5, 0, 2)
        action_now, _ = decide_retry_action("now", 5, 0, 2)
        self.assertEqual(action_off, "auto_retry",
            "off ignores step_retry_count and only looks at auto_retried")
        self.assertEqual(action_now, "user_prompt",
            "now reads step_retry_count and sees 5 > max_retries=2")
        self.assertNotEqual(action_off, action_now,
            "the whole point of plan_retry_strategy=now is to diverge "
            "from off — if they agree here the engine is dead code")


if __name__ == "__main__":
    unittest.main()
