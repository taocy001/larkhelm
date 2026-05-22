"""Regression tests for the /help card layout.

The previous version of ``_cmd_help`` shipped 5 buttons in a single
``column_set`` with ``width: "auto"`` columns. Feishu's auto-width logic
splits the available card width evenly across columns, so at 5 columns
each button got ~⅕ of the card width (~60 px on mobile) — small enough
that emoji-prefixed Chinese labels collapsed to ``...`` on the actual
device. User complaint: 五个按钮全部显示 ``...``，根本不知道是干什么的.

These tests pin:
  • the button count ≤ 3 (so each button gets ≥ ⅓ width)
  • each label is short enough to fit at that width (≤ 6 visual cols,
    where ASCII counts 1 and CJK / emoji counts 2)
  • the body still mentions every top-level command (decision-tree,
    common, lock, memory, single-shot model dispatch) — proves the
    redesign didn't accidentally hide functionality
"""
from __future__ import annotations

import unicodedata
import unittest
from unittest.mock import patch


def _visual_width(text: str) -> int:
    """Approximate the rendered width of ``text`` in monospace columns.

    Wide characters (CJK / fullwidth / most emoji) count as 2, regular
    ASCII as 1. Matches the heuristic Feishu uses to layout button
    text — close enough for regression bounds.
    """
    width = 0
    for ch in text:
        eaw = unicodedata.east_asian_width(ch)
        width += 2 if eaw in ("W", "F") else 1
    return width


class TestHelpCardLayout(unittest.TestCase):

    def _capture_help_call(self):
        """Invoke ``_cmd_help`` with all I/O monkey-patched and return the
        kwargs passed to ``send_card_reply``."""
        captured: dict = {}

        def fake_send_card_reply(chat_id, msg_id, title, body,
                                 color="blue", buttons=None, normalize=True,
                                 **kw):
            captured["chat_id"] = chat_id
            captured["title"]   = title
            captured["body"]    = body
            captured["color"]   = color
            captured["buttons"] = list(buttons or [])
            captured["normalize"] = normalize

        with patch("larkhelm.commands.send_card_reply", fake_send_card_reply):
            with patch("larkhelm.commands._get_chat_model", return_value="claude"):
                from larkhelm.commands import _cmd_help
                _cmd_help("oc_test", msg_id="msg_1")

        return captured

    # ── Button regressions ───────────────────────────────────────────────

    def test_help_card_has_at_most_three_buttons(self):
        """5 buttons → ellipsis. The whole point of the layout refresh."""
        cap = self._capture_help_call()
        self.assertLessEqual(
            len(cap["buttons"]), 3,
            f"too many buttons ({len(cap['buttons'])}) — Feishu will "
            f"split the column_set into narrow slots and labels will "
            f"collapse to '...': {cap['buttons']}"
        )

    def test_help_button_labels_short_enough_to_render(self):
        """Each label must fit a ⅓-width column on a mobile card.

        Empirical bound: with 3 columns the per-button width on a
        ~340 px mobile card is ~110 px after padding ≈ 10 visual columns
        (CJK/emoji = 2, ASCII = 1). We pin ≤ 10. The longest backend
        name (``deepseek``, 8 ASCII) + ``→ `` (1 wide arrow + 1 space =
        3 cols if the arrow renders single-width, 4 if double-width;
        budgeting 4 to be safe) → 12 max. Reserve a hard floor of 10
        and let the ``→ deepseek`` corner case live at the edge.
        """
        cap = self._capture_help_call()
        for label, cmd in cap["buttons"]:
            w = _visual_width(label)
            self.assertLessEqual(
                w, 11,
                f"button label too wide for ⅓-column rendering: "
                f"{label!r} = {w} visual cols (cmd={cmd!r})"
            )

    def test_help_buttons_map_to_valid_commands(self):
        """Every button cmd must look like a real /command (defensive — a
        typo in the cmd string would leave the user clicking a no-op
        button)."""
        cap = self._capture_help_call()
        for label, cmd in cap["buttons"]:
            self.assertTrue(
                cmd.startswith("/"),
                f"button cmd must start with /: {label!r} → {cmd!r}"
            )

    # ── Body regressions ─────────────────────────────────────────────────

    def test_help_body_still_covers_top_level_commands(self):
        """The redesign deduped a lot of repetition but every public
        command family must still be discoverable from /help. If the
        dedupe accidentally hides one, the user can't find it."""
        cap = self._capture_help_call()
        body = cap["body"]

        # Top-level user commands — each must appear at least once.
        for required in (
            "/dev",       # software engineering pipeline
            "/crew",      # dynamic multi-agent
            "/plan",      # multi-stage orchestration
            "/reset",     # session reset
            "/cancel",    # cancel current query
            "/lock",      # backend select / switch
            "/cd",        # cwd switch
            "/run",       # shell exec
            "/memory",    # memory subsystem
            "/doc",       # Feishu doc ops
            "/cron",      # scheduled tasks
            "/status",    # runtime status
            "/history",   # conversation log
            "/stats",     # token / usage stats
        ):
            self.assertIn(
                required, body,
                f"top-level command {required} disappeared from /help body — "
                f"users can't discover it anymore"
            )

    def test_help_body_shows_model_section_and_intro(self):
        """The card body has an intro line and a per-model shortcut section."""
        cap = self._capture_help_call()
        # Intro line must be present so users know how to start
        self.assertIn("发消息直接提问", cap["body"])
        # Per-model shortcut section must be discoverable
        self.assertIn("与指定模型对话", cap["body"])

    def test_help_card_title_and_color(self):
        cap = self._capture_help_call()
        self.assertEqual(cap["title"], "📖 帮助")
        self.assertEqual(cap["color"], "blue")
        self.assertFalse(
            cap["normalize"],
            "normalize=False — help body uses inline emoji that the "
            "card-builder normaliser would otherwise rewrap"
        )


if __name__ == "__main__":
    unittest.main()
