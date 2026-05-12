"""
Lightweight string-assertion tests for the `/help` task-complexity decision tree.

Guards against regression of the U1 docs change in `_cmd_help`:
- the "任务复杂度决策树" section exists and lists all four entries
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

    def test_decision_tree_section_present(self):
        body = self._capture_help_body()
        self.assertIn("任务复杂度决策树", body)

    def test_all_four_entries_present(self):
        body = self._capture_help_body()
        # 直接对话 + /plan + /dev + /crew
        self.assertIn("直接发消息", body)
        self.assertIn("/plan", body)
        self.assertIn("/dev", body)
        self.assertIn("/crew", body)

    def test_fallback_hint_present(self):
        body = self._capture_help_body()
        self.assertIn("不确定时", body)

    def test_body_fits_single_card(self):
        body = self._capture_help_body()
        # max_card_len default is 3000 (see CLAUDE.md). Keep body well below it
        # so the decision tree never pushes /help past the single-card threshold.
        self.assertLess(len(body), 3000, f"_cmd_help body grew to {len(body)} chars")

    def test_uses_fullwidth_separator_not_gfm_pipe(self):
        """Decision tree rows must use `｜` (U+FF5C) to avoid GFM table parsing."""
        body = self._capture_help_body()
        # Extract just the decision tree block (between the section heading and the
        # following `---` separator) so we don't accidentally inspect unrelated
        # `|` characters that may appear in other markdown segments.
        start = body.index("任务复杂度决策树")
        end = body.index("---", start)
        block = body[start:end]
        self.assertIn("｜", block)
        self.assertNotIn("|", block)


if __name__ == "__main__":
    unittest.main()
