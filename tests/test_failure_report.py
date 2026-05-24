"""P1-1a — unit tests for ``larkhelm.failure_report.emit``.

Covers the five acceptance criteria from PRD §AC:

* AC-04 flag-off no-op  → ``test_emit_skipped_when_flag_disabled``
* AC-05 happy path orange card → ``test_emit_sends_orange_card_on_happy_path``
* AC-06 send_card raises → emit doesn't → ``test_emit_swallows_send_card_failure``
* AC-10 admin_chat_id empty no-op → ``test_emit_skipped_when_admin_chat_empty``
* AC-09 truncate rules → ``test_emit_truncates_category_summary_detail``
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import larkhelm.config as _cfg
from larkhelm import failure_report


class TestEmitFailureReport(unittest.TestCase):

    def setUp(self) -> None:
        self._prev_enabled = getattr(_cfg, "FAILURE_REPORT_CARD_ENABLED", False)
        self._prev_admin = getattr(_cfg, "ADMIN_CHAT_ID", "")
        _cfg.FAILURE_REPORT_CARD_ENABLED = True
        _cfg.ADMIN_CHAT_ID = "chat_admin"

    def tearDown(self) -> None:
        _cfg.FAILURE_REPORT_CARD_ENABLED = self._prev_enabled
        _cfg.ADMIN_CHAT_ID = self._prev_admin

    def test_emit_skipped_when_flag_disabled(self) -> None:
        _cfg.FAILURE_REPORT_CARD_ENABLED = False
        with patch("larkhelm.lark_client.send_card") as fake:
            failure_report.emit("backoff_exhausted", "cascade retry burned out")
        fake.assert_not_called()

    def test_emit_sends_orange_card_on_happy_path(self) -> None:
        with patch("larkhelm.lark_client.send_card") as fake:
            failure_report.emit(
                "circuit_open",
                "cheap backend tripped breaker",
                "stack: deepseek 5xx x6",
            )
        fake.assert_called_once()
        args, kwargs = fake.call_args
        chat_id = args[0] if args else kwargs.get("chat_id")
        title = args[1] if len(args) > 1 else kwargs.get("title", "")
        body = args[2] if len(args) > 2 else kwargs.get("body", "")
        color = args[3] if len(args) > 3 else kwargs.get("color", "")
        self.assertEqual(chat_id, "chat_admin")
        self.assertIn("circuit_open", title)
        self.assertTrue(title.startswith("⚠️"))
        self.assertIn("cheap backend tripped breaker", body)
        self.assertIn("deepseek 5xx x6", body)
        self.assertEqual(color, "orange")

    def test_emit_swallows_send_card_failure(self) -> None:
        with patch(
            "larkhelm.lark_client.send_card",
            side_effect=RuntimeError("net down"),
        ):
            try:
                failure_report.emit("oom_killed", "worker reaped by cgroup")
            except Exception as e:  # pragma: no cover — defensive: emit must never raise
                self.fail(f"emit() raised despite send_card failure: {e}")

    def test_emit_skipped_when_admin_chat_empty(self) -> None:
        _cfg.ADMIN_CHAT_ID = ""
        with patch("larkhelm.lark_client.send_card") as fake:
            failure_report.emit("backoff_exhausted", "won't be sent")
        fake.assert_not_called()

    def test_emit_truncates_category_summary_detail(self) -> None:
        long_cat = "x" * 100
        long_sum = "y" * 500
        long_det = "z" * 2000
        with patch("larkhelm.lark_client.send_card") as fake:
            failure_report.emit(long_cat, long_sum, long_det)
        fake.assert_called_once()
        args, kwargs = fake.call_args
        title = args[1] if len(args) > 1 else kwargs.get("title", "")
        body = args[2] if len(args) > 2 else kwargs.get("body", "")
        # Title = "⚠️ " + truncated category (≤ 32 chars with trailing "…")
        # The category portion (after the prefix) must be ≤ _CATEGORY_MAX.
        cat_part = title.split(" ", 1)[1] if " " in title else title
        self.assertLessEqual(len(cat_part), failure_report._CATEGORY_MAX)
        self.assertTrue(cat_part.endswith("…"))
        # Body must contain truncated summary (≤ 200 chars + ellipsis) and
        # truncated detail (≤ 800 chars + ellipsis). Easiest contract check:
        # neither raw input length survives in body.
        self.assertNotIn("y" * 500, body)
        self.assertNotIn("z" * 2000, body)
        self.assertIn("…", body)


if __name__ == "__main__":
    unittest.main()
