"""
P1 — card_builder.py comprehensive unit tests

Coverage: _fmt_elapsed / _btn_type / _normalize_newlines / _split_md / _make_card
"""
import atexit
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# ── Initialize config ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_cbtest_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "max_card_len": 100,   # small value to make _split_md easy to test
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.card_builder import (
    _fmt_elapsed, _btn_type, _normalize_newlines, _split_md, _make_card,
)


class TestFmtElapsed(unittest.TestCase):
    def test_seconds_under_60(self):
        self.assertEqual(_fmt_elapsed(0), "0s")
        self.assertEqual(_fmt_elapsed(59.9), "60s")

    def test_exact_59(self):
        self.assertEqual(_fmt_elapsed(59), "59s")

    def test_minutes_at_60(self):
        # 60s → 1.0m
        self.assertEqual(_fmt_elapsed(60), "1.0m")

    def test_minutes_90(self):
        self.assertEqual(_fmt_elapsed(90), "1.5m")

    def test_minutes_3600(self):
        self.assertEqual(_fmt_elapsed(3600), "60.0m")


class TestBtnType(unittest.TestCase):
    def test_primary_keywords(self):
        for kw in ("允许", "确认", "✅", "同意", "Yes", "OK"):
            with self.subTest(kw=kw):
                self.assertEqual(_btn_type(f"请{kw}"), "primary")

    def test_danger_keywords(self):
        for kw in ("拒绝", "取消", "删除", "❌", "No", "Deny"):
            with self.subTest(kw=kw):
                self.assertEqual(_btn_type(f"点击{kw}"), "danger")

    def test_default_for_neutral(self):
        self.assertEqual(_btn_type("查看状态"), "default")
        self.assertEqual(_btn_type("继续"), "default")

    def test_primary_takes_precedence(self):
        # Contains both an allow and a deny keyword → primary (primary matched first)
        self.assertEqual(_btn_type("允许并拒绝"), "primary")


class TestNormalizeNewlines(unittest.TestCase):
    def test_single_newline_becomes_double(self):
        result = _normalize_newlines("line1\nline2")
        self.assertIn("\n\n", result)

    def test_already_double_newline_unchanged(self):
        result = _normalize_newlines("line1\n\nline2")
        lines = result.split("\n")
        # Should not produce three consecutive blank lines
        for i in range(len(lines) - 2):
            self.assertFalse(lines[i] == "" and lines[i+1] == "" and lines[i+2] == "",
                             "Three consecutive blank lines")

    def test_code_block_preserved(self):
        code = "```\nline1\nline2\n```"
        result = _normalize_newlines(code)
        # Newlines inside a code block should not be expanded to double newlines
        self.assertIn("line1\nline2", result)

    def test_empty_string(self):
        self.assertEqual(_normalize_newlines(""), "")

    def test_outside_code_block_gets_blank_line(self):
        text = "before\n```\ncode\n```\nafter"
        result = _normalize_newlines(text)
        # before/after outside the code block should be separated by newlines
        self.assertIn("before", result)
        self.assertIn("after", result)

    def test_multiline_code_block_internal(self):
        text = "```python\nx = 1\ny = 2\nz = 3\n```"
        result = _normalize_newlines(text)
        # No blank lines should be inserted between lines inside the block
        self.assertIn("x = 1\ny = 2\nz = 3", result)


class TestSplitMd(unittest.TestCase):
    # max_card_len = 100 (set in config above)

    def test_short_text_no_split(self):
        text = "hello"
        chunks = _split_md(text)
        self.assertEqual(chunks, [text])

    def test_exact_limit_no_split(self):
        text = "x" * 100
        chunks = _split_md(text)
        self.assertEqual(len(chunks), 1)

    def test_over_limit_splits(self):
        # Generate multiline text exceeding 3000 chars (default MAX_CARD_LEN)
        lines = ["line" + str(i) + "x" * 100 for i in range(30)]
        text = "\n".join(lines)
        self.assertGreater(len(text), 3000)
        chunks = _split_md(text)
        self.assertGreater(len(chunks), 1)
        # All content should be preserved
        combined = "\n".join(chunks)
        for line in lines:
            self.assertIn(line, combined)

    def test_does_not_split_inside_code_block(self):
        # A single oversized code block should not be split at its interior
        inner = "\n".join([f"code_line_{i}" for i in range(50)])
        text = f"```\n{inner}\n```"
        chunks = _split_md(text)
        # The entire code block should reside in the same chunk
        code_lines_found = sum(1 for c in chunks if "code_line_0" in c and "code_line_49" in c)
        self.assertGreaterEqual(code_lines_found, 1)

    def test_empty_text_returns_list_with_empty_string(self):
        chunks = _split_md("")
        self.assertEqual(chunks, [""])


class TestMakeCard(unittest.TestCase):
    # ── JSON 2.0 (no buttons) ────────────────────────────────────

    def test_no_buttons_returns_json20(self):
        card = json.loads(_make_card("Title", "Body"))
        self.assertEqual(card.get("schema"), "2.0")
        self.assertIn("body", card)

    def test_header_color_and_title(self):
        card = json.loads(_make_card("MyTitle", "Body", color="red"))
        self.assertEqual(card["header"]["template"], "red")
        self.assertEqual(card["header"]["title"]["content"], "MyTitle")

    def test_subtitle_added_to_header(self):
        card = json.loads(_make_card("T", "B", subtitle="Sub"))
        self.assertEqual(card["header"]["subtitle"]["content"], "Sub")

    def test_body_contains_markdown_element(self):
        card = json.loads(_make_card("T", "Hello world"))
        elements = card["body"]["elements"]
        md_elements = [e for e in elements if e.get("tag") == "markdown"]
        self.assertTrue(md_elements)

    def test_note_appended_to_body(self):
        card = json.loads(_make_card("T", "Body", note="footnote"))
        text = json.dumps(card)
        self.assertIn("footnote", text)

    def test_empty_body_no_markdown_element(self):
        card = json.loads(_make_card("T", ""))
        elements = card["body"]["elements"]
        # Empty body should not produce a markdown element
        self.assertFalse(any(e.get("tag") == "markdown" for e in elements))

    def test_normalize_false_preserves_single_newline(self):
        body = "line1\nline2"
        card = json.loads(_make_card("T", body, normalize=False))
        text = json.dumps(card)
        # With normalize=False, line1\nline2 is kept as-is (not expanded to double newline)
        self.assertIn("line1\\nline2", text)

    def test_tools_md_produces_collapsible_panel(self):
        card = json.loads(_make_card("T", "Body", tools_md="- tool output"))
        elements = card["body"]["elements"]
        panels = [e for e in elements if e.get("tag") == "collapsible_panel"]
        self.assertTrue(panels)

    def test_tools_list_produces_collapsible_panel(self):
        tools = [
            {"name": "Bash", "elapsed": 1.5, "desc": "ls", "is_error": False, "full_result": ""},
            {"name": "Read", "elapsed": 0.3, "desc": "foo.py", "is_error": True, "full_result": "err"},
        ]
        card = json.loads(_make_card("T", "Body", tools_list=tools))
        text = json.dumps(card)
        self.assertIn("Bash", text)
        self.assertIn("Read", text)

    def test_tools_list_long_result_truncated(self):
        long_result = "x" * 5000
        tools = [{"name": "Bash", "elapsed": 1.0, "desc": "", "is_error": False, "full_result": long_result}]
        card = json.loads(_make_card("T", "B", tools_list=tools))
        text = json.dumps(card, ensure_ascii=False)
        # Truncation notice should be present
        self.assertIn("截断", text)

    # ── JSON 2.0 buttons (post-migration; was JSON 1.0 prior to commit) ─────

    def _buttons_in(self, card: dict) -> list[dict]:
        """Walk card['body']['elements'] and return every ``tag:"button"``
        encountered, including those nested inside a ``column_set`` /
        ``column``. Provides a single helper used by every button test so
        a future schema layout change touches one place."""
        buttons: list[dict] = []
        for el in card["body"]["elements"]:
            if el.get("tag") == "button":
                buttons.append(el)
            elif el.get("tag") == "column_set":
                for col in el.get("columns", []):
                    for child in col.get("elements", []):
                        if child.get("tag") == "button":
                            buttons.append(child)
        return buttons

    def test_with_buttons_uses_json20_schema(self):
        """Post-migration: button cards have ``schema:"2.0"`` and route
        buttons through ``body.elements`` (not the legacy 1.0 top-level
        ``elements`` + ``tag:"action"`` container)."""
        card = json.loads(_make_card("T", "B", buttons=[("OK", "/ok")]))
        self.assertEqual(card.get("schema"), "2.0")
        self.assertIn("body", card)
        self.assertNotIn("actions", json.dumps(card),
                         "no JSON 1.0 'actions' container should remain")

    def test_single_button_is_bare_element(self):
        """One button → bare ``tag:"button"`` directly in ``body.elements``,
        NOT wrapped in a column_set (avoids gratuitous nesting)."""
        card = json.loads(_make_card("T", "B", buttons=[("OK", "/ok")]))
        last = card["body"]["elements"][-1]
        self.assertEqual(last["tag"], "button")
        self.assertEqual(last["text"]["content"], "OK")
        self.assertEqual(last["text"]["tag"], "plain_text")

    def test_multi_buttons_wrapped_in_column_set(self):
        """≥2 buttons → ``column_set`` with one ``width:"auto"`` column per
        button. Necessary because JSON 2.0 has no native multi-button row
        container; column_set is the canonical layout per Feishu docs."""
        card = json.loads(_make_card("T", "B",
                                     buttons=[("✅ 允许", "/a"), ("❌ 拒绝", "/d")]))
        last = card["body"]["elements"][-1]
        self.assertEqual(last["tag"], "column_set")
        self.assertEqual(len(last["columns"]), 2)
        for col in last["columns"]:
            self.assertEqual(col["tag"], "column")
            self.assertEqual(col["width"], "auto")
            self.assertEqual(len(col["elements"]), 1)
            self.assertEqual(col["elements"][0]["tag"], "button")

    def test_buttons_rendered_correctly(self):
        card = json.loads(_make_card("T", "B",
                                     buttons=[("✅ 允许", "/allow"), ("❌ 拒绝", "/deny")]))
        buttons = self._buttons_in(card)
        self.assertEqual(len(buttons), 2)
        labels = [b["text"]["content"] for b in buttons]
        self.assertIn("✅ 允许", labels)
        self.assertIn("❌ 拒绝", labels)

    def test_button_types_primary_danger(self):
        card = json.loads(_make_card("T", "B",
                                     buttons=[("✅ 允许", "/a"), ("❌ 取消", "/c")]))
        types = {b["text"]["content"]: b["type"] for b in self._buttons_in(card)}
        self.assertEqual(types["✅ 允许"], "primary")
        self.assertEqual(types["❌ 取消"], "danger")

    def test_button_cmd_routed_via_callback_behavior(self):
        """JSON 2.0 puts the cmd payload inside ``behaviors[0].value``,
        not directly on the button element. ``handlers/_card_action.py``
        receives it via ``CallBackAction.value`` regardless of schema."""
        card = json.loads(_make_card("T", "B", buttons=[("Go", "/go_cmd")]))
        btn = self._buttons_in(card)[0]
        self.assertEqual(btn["behaviors"][0]["type"], "callback")
        self.assertEqual(btn["behaviors"][0]["value"]["cmd"], "/go_cmd")

    def test_note_appears_in_button_card(self):
        card = json.loads(_make_card("T", "B",
                                     buttons=[("X", "/x")], note="my note"))
        # Note merges into a markdown element OR appends as its own.
        # Just assert the literal text shows up in the body.
        body_text = json.dumps(card["body"]["elements"], ensure_ascii=False)
        self.assertIn("my note", body_text)

    def test_wide_screen_mode_enabled(self):
        for buttons in [None, [("X", "/x")]]:
            with self.subTest(buttons=buttons):
                card = json.loads(_make_card("T", "B", buttons=buttons))
                self.assertTrue(card["config"]["wide_screen_mode"])

    def test_plain_text_label_not_lark_md(self):
        """JSON 2.0 button labels MUST use ``plain_text``; ``lark_md`` is
        a JSON 1.0 construct and adding it back would re-introduce the
        font/markdown-subset inconsistencies that prompted this migration."""
        card = json.loads(_make_card("T", "B",
                                     buttons=[("**bold label**", "/x")]))
        btn = self._buttons_in(card)[0]
        self.assertEqual(btn["text"]["tag"], "plain_text")

    def test_returns_valid_json_string(self):
        # Should not raise; must return a parseable JSON string
        result = _make_card("测试标题", "测试内容\n包含换行", color="turquoise")
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)


if __name__ == "__main__":
    unittest.main()
