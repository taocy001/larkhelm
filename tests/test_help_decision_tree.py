"""
Lightweight string-assertion tests for the `/help` task-complexity decision tree.

Guards against regression of the U1 docs change in `_cmd_help`:
- the "任务复杂度决策树" section exists and lists its entries
- body fits within `max_card_len` so a single Feishu card still suffices
- the full-width `｜` separator (U+FF5C) is used instead of the GFM table pipe `|`
"""
import unittest
from unittest.mock import patch


class TestHelpDecisionTree(unittest.TestCase):
    """Capture the body string passed to `send_card_reply` and assert on it."""

    def _capture_help_body(self) -> str:
        captured = {}

        def _fake_send(chat_id, msg_id, title, body, **kwargs):
            captured["body"] = body

        with patch("larkhelm.commands.send_card_reply", side_effect=_fake_send), \
             patch("larkhelm.commands._get_chat_model", return_value="claude"):
            from larkhelm.commands import _cmd_help
            _cmd_help("chat_help_test", "msg_help_test")

        self.assertIn("body", captured, "send_card_reply was not invoked by _cmd_help")
        return captured["body"]

    # NOTE (2026-05-19 help-card refresh):
    #   • Section heading renamed from "任务复杂度决策树" to "任务怎么选" —
    #     more conversational, same intent. Test asserts on the new heading.
    #   • Per-row content trimmed from "描述 ｜ 产物 ｜ 示例" three-segment
    #     form to a single short description. Without ``｜`` separators
    #     the GFM-table-parse pitfall the old test guarded against no
    #     longer applies; the test is rewritten as a defensive "if you
    #     re-introduce row separators, use the full-width variant".
    def _decision_tree_block(self, body: str) -> str:
        """Return the substring between the decision-tree heading and
        the first ``---`` separator that follows it."""
        start = body.index("任务怎么选")
        end = body.index("---", start)
        return body[start:end]

    def test_decision_tree_section_present(self):
        body = self._capture_help_body()
        self.assertIn("任务怎么选", body)

    def test_direct_chat_entry_present(self):
        body = self._capture_help_body()
        self.assertIn("直接发消息", body)

    def test_fallback_hint_present(self):
        body = self._capture_help_body()
        self.assertIn("不确定时", body)

    def test_body_fits_single_card(self):
        body = self._capture_help_body()
        # max_card_len default is 3000 (see CLAUDE.md). Keep body well below it
        # so the decision tree never pushes /help past the single-card threshold.
        self.assertLess(len(body), 3000, f"_cmd_help body grew to {len(body)} chars")

    def test_decision_tree_avoids_gfm_pipe_separators(self):
        """Even if a future revision re-introduces in-row separators, they
        must be ``｜`` (U+FF5C) — never the GFM table pipe ``|`` which
        would render as an empty-header three-column table on Feishu."""
        block = self._decision_tree_block(self._capture_help_body())
        self.assertNotIn(
            "|", block,
            "decision-tree block contains a GFM `|` separator, which "
            "Feishu interprets as a table header row — use the full-"
            "width `｜` (U+FF5C) if you really need a row separator"
        )


if __name__ == "__main__":
    unittest.main()
