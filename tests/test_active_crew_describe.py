"""Regression tests for C3 #9: describe_active_owner + conflict-card UX.

Pre-C3 both ``/plan`` and ``/crew`` / ``/dev`` conflict cards just said
"任务正在运行" with no hint about which command holds the slot — the
operator couldn't tell whether to cancel a plan or a crew run, and
``_active_crew[chat_id]`` was opaque (raw owner-token string).

C3 introduces ``describe_active_owner`` in ``crew/_state`` that decodes
the owner-token format:
  * ``"plan:<plan_id>"`` → "`/plan` 任务 (id=...)"
  * raw hex                 → "`/crew` 或 `/dev` 任务 (id=...)"

These tests pin the contract + assert both conflict-card sites pipe the
description through.
"""
from __future__ import annotations

import unittest
from unittest import mock

from larkhelm.crew._state import describe_active_owner


class TestDescribeActiveOwner(unittest.TestCase):

    def test_plan_owner_token(self):
        out = describe_active_owner("plan:abc123def456")
        self.assertIn("`/plan`", out)
        self.assertIn("abc123de", out)   # first 8 chars of plan_id

    def test_crew_dev_owner_token(self):
        out = describe_active_owner("a1b2c3d4e5f6")
        self.assertIn("`/crew`", out)
        self.assertIn("`/dev`", out)
        self.assertIn("a1b2c3d4", out)

    def test_empty_owner_token(self):
        # Defensive: returns a non-empty string so the conflict card never
        # collapses to "正在运行 " (trailing space, no info).
        out = describe_active_owner("")
        self.assertTrue(out)
        self.assertIn("未知", out)

    def test_plan_owner_with_short_id(self):
        # ``plan:`` prefix but unusually short plan_id (defensive).
        out = describe_active_owner("plan:abc")
        self.assertIn("`/plan`", out)


class TestConflictCardWiring(unittest.TestCase):
    """End-to-end-ish: drive ``cmd_plan`` and ``cmd_crew`` with a pre-occupied
    ``_active_crew`` slot and capture the card body to confirm the
    description is plumbed through.
    """

    def test_cmd_plan_conflict_card_shows_owner(self):
        from larkhelm.crew._state import _active_crew, _active_crew_lock
        import larkhelm.cmd_plan as plan_mod

        chat_id = "oc_plan_conflict_card"
        captured = {}

        def _capture(chat_id_arg, title, body, *, color=None, **_kw):
            captured.setdefault("calls", []).append(
                {"title": title, "body": body, "color": color}
            )

        with _active_crew_lock:
            _active_crew[chat_id] = "plan:deadbeef1234"
        try:
            with mock.patch("larkhelm.lark_client.send_card", side_effect=_capture), \
                 mock.patch("larkhelm.concurrency._reset_cancel"):
                plan_mod.cmd_plan(chat_id, "/plan [dev] foo")
        finally:
            with _active_crew_lock:
                _active_crew.pop(chat_id, None)

        self.assertTrue(captured.get("calls"), "no card was sent")
        conflict = next(
            (c for c in captured["calls"] if "任务冲突" in c["title"]),
            None,
        )
        self.assertIsNotNone(conflict, "expected the conflict card to fire")
        self.assertIn("`/plan`", conflict["body"])
        self.assertIn("deadbeef", conflict["body"])
        self.assertEqual(conflict["color"], "orange")

    def test_cmd_crew_conflict_card_shows_owner(self):
        from larkhelm.crew._state import _active_crew, _active_crew_lock
        from larkhelm.crew import _commands as crew_cmds

        chat_id = "oc_crew_conflict_card"
        captured = {}

        def _capture(chat_id_arg, title, body, *, color=None, **_kw):
            captured.setdefault("calls", []).append(
                {"title": title, "body": body, "color": color}
            )

        with _active_crew_lock:
            _active_crew[chat_id] = "cafebabe9999"   # raw hex = crew/dev
        try:
            with mock.patch("larkhelm.lark_client.send_card", side_effect=_capture):
                crew_cmds.cmd_crew(chat_id, "do a thing", None)
        finally:
            with _active_crew_lock:
                _active_crew.pop(chat_id, None)

        self.assertTrue(captured.get("calls"), "no card was sent")
        conflict = next(
            (c for c in captured["calls"] if "Crew 已在运行" in c["title"]),
            None,
        )
        self.assertIsNotNone(conflict, "expected the conflict card to fire")
        self.assertIn("`/crew`", conflict["body"])
        self.assertIn("`/dev`", conflict["body"])
        self.assertIn("cafebabe", conflict["body"])


if __name__ == "__main__":
    unittest.main()
