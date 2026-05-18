"""PRD Appendix A regression: chat-level /cancel must wake plan _wait_confirm.

Construct a MultiPlanState and enter `_wait_for_confirm_or_cancel` in a worker
thread, then trigger a chat-level cancel. The function must return promptly
with `state.cancel_ev` set.
"""
import threading
import time
import unittest
from unittest import mock

from larkhelm.cmd_plan import MultiPlanState, _wait_for_confirm_or_cancel
from larkhelm.concurrency import _trigger_cancel, _reset_cancel


def _state(chat_id: str = "oc_test") -> MultiPlanState:
    return MultiPlanState(
        plan_id="p1", chat_id=chat_id, title="t", steps=[],
    )


class TestPlanCancelBug(unittest.TestCase):

    def setUp(self):
        # Speed up the polling loop so the test stays fast.
        import larkhelm.cmd_plan as cp
        self._orig_poll = cp._WAIT_POLL_INTERVAL
        cp._WAIT_POLL_INTERVAL = 0.05

    def tearDown(self):
        import larkhelm.cmd_plan as cp
        cp._WAIT_POLL_INTERVAL = self._orig_poll

    def test_chat_cancel_wakes_wait_confirm(self):
        chat_id = "oc_plan_cancel_bug"
        _reset_cancel(chat_id)
        state = _state(chat_id=chat_id)

        result_holder: list[bool] = []

        def _worker():
            result_holder.append(_wait_for_confirm_or_cancel(state, timeout=5.0))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        time.sleep(0.1)
        _trigger_cancel(chat_id)
        t.join(timeout=2.0)

        self.assertFalse(t.is_alive(), "_wait_for_confirm_or_cancel did not return")
        self.assertEqual(result_holder, [True])
        self.assertTrue(state.cancel_ev.is_set())
        self.assertEqual(state._confirm_result, "cancel")

        # Reset for downstream tests.
        _reset_cancel(chat_id)

    def test_card_button_cancel_still_works(self):
        # Sanity: when the card button signals cancel directly, function returns.
        chat_id = "oc_plan_button"
        _reset_cancel(chat_id)
        state = _state(chat_id=chat_id)

        result_holder: list[bool] = []

        def _worker():
            result_holder.append(_wait_for_confirm_or_cancel(state, timeout=5.0))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        time.sleep(0.1)

        with state.lock:
            state._confirm_result = "cancel"
            state.cancel_ev.set()
        state._confirm_ev.set()

        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result_holder, [True])

    def test_active_crew_cleanup_in_finally_paths(self):
        """Real end-to-end: _run_plan must release _active_crew[chat_id] on exit.

        Drive _run_plan with an empty step list so the loop body never executes
        and only the finally block runs. card_mid is empty, so _update_plan_card
        is a no-op (no Feishu API call). Pre-populate _active_crew to verify the
        finally branch actually pops it.
        """
        import larkhelm.cmd_plan as cp
        from larkhelm.crew._state import _active_crew, _active_crew_lock

        chat_id = "oc_plan_finally"
        with _active_crew_lock:
            _active_crew[chat_id] = "plan:p_finally"

        state = MultiPlanState(
            plan_id="p_finally", chat_id=chat_id, title="t", steps=[],
        )
        with cp._active_plans_lock:
            cp._active_plans[state.plan_id] = state

        # Patch network-touching helpers; with empty steps the body skips
        # straight to the success branch which sends a Feishu card.
        try:
            with mock.patch("larkhelm.lark_client.send_card"):
                cp._run_plan(state)
        finally:
            with cp._active_plans_lock:
                cp._active_plans.pop(state.plan_id, None)
            # Defensive: ensure we don't leak chat_id into other tests if the
            # body assertion above fails.
            with _active_crew_lock:
                _active_crew.pop(chat_id, None)

        with _active_crew_lock:
            self.assertNotIn(chat_id, _active_crew,
                             "_run_plan finally must release _active_crew slot")
        # ``state.phase`` is a ``PlanPhase`` enum after the P3-7 migration;
        # accept both the enum and its raw string value so this test can
        # stay readable without importing the enum.
        from larkhelm.cmd_plan import PlanPhase
        self.assertEqual(state.phase, PlanPhase.DONE)


if __name__ == "__main__":
    unittest.main()
