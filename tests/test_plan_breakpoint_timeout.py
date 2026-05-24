"""Regression tests for C2: plan wait_confirm bounded by
``CREW_BREAKPOINT_TIMEOUT_SEC``.

Pre-C2 ``_wait_for_confirm_or_cancel`` defaulted to 86400s (24h), so a
user who walked away from a [step→step] confirmation card kept the plan
in ``_active_plans`` for a full day, blocking the per-chat ``_active_crew``
slot. C2 routes both wait points through ``_resolve_breakpoint_timeout_sec``
(default 1800s, mirrors /crew), sets ``state.cancel_ev`` on timeout so the
``_run_plan`` main loop tears down cleanly, and emits an orange notification
card so the user sees a clear end-of-plan signal.

These tests pin:
  • ``_resolve_breakpoint_timeout_sec`` reads config + floors at 60s
  • timeout path in ``_wait_confirm`` returns "cancel" + sets cancel_ev
  • timeout path triggers the orange notification card exactly once
  • a normal button press still wins the race (no false-positive timeout)
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from larkhelm.cmd_plan import (
    MultiPlanState,
    PlanStep,
    _resolve_breakpoint_timeout_sec,
    _send_breakpoint_timeout_card,
    _wait_confirm,
)


def _state(chat_id: str = "oc_test", title: str = "t") -> MultiPlanState:
    return MultiPlanState(
        plan_id="p_bp",
        chat_id=chat_id,
        title=title,
        steps=[PlanStep(idx=0, type="dev", desc="x")],
    )


class TestResolveBreakpointTimeout(unittest.TestCase):
    def test_reads_config_value(self):
        with mock.patch("larkhelm.config.CREW_BREAKPOINT_TIMEOUT_SEC", 2400, create=True):
            self.assertEqual(_resolve_breakpoint_timeout_sec(), 2400)

    def test_floors_at_60s(self):
        with mock.patch("larkhelm.config.CREW_BREAKPOINT_TIMEOUT_SEC", 10, create=True):
            self.assertEqual(_resolve_breakpoint_timeout_sec(), 60)

    def test_zero_falls_back_to_1800(self):
        with mock.patch("larkhelm.config.CREW_BREAKPOINT_TIMEOUT_SEC", 0, create=True):
            self.assertEqual(_resolve_breakpoint_timeout_sec(), 1800)


class TestPlanWaitConfirmTimeout(unittest.TestCase):

    def setUp(self):
        import larkhelm.cmd_plan as cp
        self._orig_poll = cp._WAIT_POLL_INTERVAL
        cp._WAIT_POLL_INTERVAL = 0.02

    def tearDown(self):
        import larkhelm.cmd_plan as cp
        cp._WAIT_POLL_INTERVAL = self._orig_poll

    def test_timeout_returns_cancel_and_sets_cancel_ev(self):
        state = _state(chat_id="oc_plan_bp_timeout")
        # Force a tiny bp timeout via the resolver hook.
        with mock.patch(
            "larkhelm.cmd_plan._resolve_breakpoint_timeout_sec",
            return_value=60,  # min floor; actual wait below uses tiny value
        ), mock.patch(
            "larkhelm.cmd_plan._wait_for_confirm_or_cancel",
            return_value=False,  # simulate timeout
        ) as m_wait, mock.patch(
            "larkhelm.cmd_plan._update_plan_card"
        ), mock.patch(
            "larkhelm.cmd_plan._send_breakpoint_timeout_card"
        ) as m_card:
            result = _wait_confirm(state)

        self.assertEqual(result, "cancel")
        self.assertTrue(state.cancel_ev.is_set())
        m_wait.assert_called_once()
        # The timeout value passed to the inner wait function must come
        # from the resolver, not the legacy 24h dataclass default.
        self.assertEqual(m_wait.call_args.kwargs.get("timeout"), 60.0)
        m_card.assert_called_once()
        # Phase hint must distinguish between the two wait sites.
        self.assertEqual(m_card.call_args.kwargs.get("phase_hint"), "步骤间确认")

    def test_button_press_wins_no_timeout_card(self):
        """When the user actually clicks a button before the timeout, the
        orange notification card must NOT be sent — that would falsely
        suggest the plan was cancelled when it wasn't."""
        state = _state(chat_id="oc_plan_bp_normal")

        def _fake_wait(s, timeout):
            # Simulate the card-button path: signal_plan('continue') would
            # set _confirm_result + _confirm_ev, then return True.
            s._confirm_result = "continue"
            s._confirm_ev.set()
            return True

        with mock.patch(
            "larkhelm.cmd_plan._resolve_breakpoint_timeout_sec",
            return_value=60,
        ), mock.patch(
            "larkhelm.cmd_plan._wait_for_confirm_or_cancel",
            side_effect=_fake_wait,
        ), mock.patch(
            "larkhelm.cmd_plan._update_plan_card"
        ), mock.patch(
            "larkhelm.cmd_plan._send_breakpoint_timeout_card"
        ) as m_card:
            result = _wait_confirm(state)

        self.assertEqual(result, "continue")
        self.assertFalse(state.cancel_ev.is_set())
        m_card.assert_not_called()


class TestSendBreakpointTimeoutCard(unittest.TestCase):
    def test_send_card_fail_soft(self):
        """Card-send failure must not propagate — surrounding cancel flow
        depends on _wait_confirm returning cleanly."""
        state = _state()
        with mock.patch(
            "larkhelm.lark_client.send_card",
            side_effect=RuntimeError("feishu down"),
        ):
            # Should not raise.
            _send_breakpoint_timeout_card(state, phase_hint="测试")

    def test_phase_hint_appears_in_title(self):
        state = _state(title="My Plan")
        captured = {}

        def _capture(chat_id, title, body, *, color=None, **_kw):
            captured["chat_id"] = chat_id
            captured["title"] = title
            captured["body"] = body
            captured["color"] = color

        with mock.patch("larkhelm.cmd_plan._resolve_breakpoint_timeout_sec",
                        return_value=1800):
            with mock.patch("larkhelm.lark_client.send_card", side_effect=_capture):
                _send_breakpoint_timeout_card(state, phase_hint="计划生成后确认")

        self.assertIn("计划生成后确认", captured["title"])
        self.assertIn("30 分钟", captured["title"])   # 1800 / 60
        self.assertIn("My Plan", captured["body"])
        self.assertEqual(captured["color"], "orange")


if __name__ == "__main__":
    unittest.main()
