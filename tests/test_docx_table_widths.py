"""
Tests for ``lark_client._md_to_descendants`` table-width emission.

Previous behavior: ``table.property`` shipped only ``row_size`` and
``column_size``; Feishu fell back to its ~80 px default column width which
clipped most real-world cells. These tests pin the new "size from content"
behavior so a regression to default widths would fail loudly.

Approach: pump markdown through ``_md_to_descendants`` (a pure helper that
needs no live Feishu client) and assert the emitted block tree carries
``column_width``, ``header_row=True``, and content-proportional sizes.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# ── Bootstrap config (mirrors test_card_builder.py) ─────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_docxtbl_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.lark_client import (
    FeishuDocClient, _estimate_col_widths_px, _visual_width,
    _COL_MIN_PX, _COL_MAX_PX,
)


def _table_block_from(content: str) -> dict:
    """Run a markdown string through the descendant builder and return its
    sole table block. Helper so each test reads cleanly."""
    client = FeishuDocClient()
    _children_ids, descendants = client._md_to_descendants(content)
    tables = [b for b in descendants if b.get("block_type") == 31]
    if len(tables) != 1:
        raise AssertionError(
            f"expected exactly 1 table block, found {len(tables)}")
    return tables[0]


class VisualWidthTests(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(_visual_width(""), 0)

    def test_ascii_one_per_char(self):
        self.assertEqual(_visual_width("hello"), 5)

    def test_cjk_two_per_char(self):
        self.assertEqual(_visual_width("中文"), 4)

    def test_mixed_ascii_cjk(self):
        self.assertEqual(_visual_width("ab中"), 4)


class EstimateColWidthsTests(unittest.TestCase):
    def test_floor_enforced(self):
        widths = _estimate_col_widths_px([["x"]], 1)
        self.assertEqual(widths, [_COL_MIN_PX])

    def test_ceiling_enforced(self):
        widths = _estimate_col_widths_px([["x" * 200]], 1)
        self.assertEqual(widths, [_COL_MAX_PX])

    def test_widest_cell_drives_width(self):
        rows = [
            ["a", "x"],
            ["m", "verylongcontentinthistallcolumn"],
        ]
        widths = _estimate_col_widths_px(rows, 2)
        self.assertLess(widths[0], widths[1])

    def test_empty_rows_falls_back_to_default(self):
        widths = _estimate_col_widths_px([], 3)
        # default=4 → 4*12+24=72 → clamped to floor 120
        self.assertEqual(widths, [_COL_MIN_PX] * 3)


class TableBlockEmissionTests(unittest.TestCase):
    """End-to-end pinning: markdown → table block → property shape."""

    MD = (
        "| Name | Description |\n"
        "|------|-------------|\n"
        "| a    | longer content that needs more room |\n"
        "| b    | c    |\n"
    )

    def test_table_block_carries_column_width(self):
        blk = _table_block_from(self.MD)
        prop = blk["table"]["property"]
        self.assertIn("column_width", prop,
            "regression: column_width is the fix for the narrow-table bug")
        self.assertEqual(len(prop["column_width"]), prop["column_size"])

    def test_table_block_sets_header_row(self):
        blk = _table_block_from(self.MD)
        self.assertTrue(blk["table"]["property"].get("header_row"),
            "first row should be marked as header so Feishu bolds it")

    def test_column_widths_are_clamped_integers(self):
        blk = _table_block_from(self.MD)
        for w in blk["table"]["property"]["column_width"]:
            self.assertIsInstance(w, int)
            self.assertGreaterEqual(w, _COL_MIN_PX)
            self.assertLessEqual(w, _COL_MAX_PX)

    def test_wide_column_gets_more_room(self):
        blk = _table_block_from(self.MD)
        widths = blk["table"]["property"]["column_width"]
        # Column 0 ("a"/"b") is narrow; column 1 has a long sentence.
        self.assertLess(widths[0], widths[1])

    def test_single_column_table_emits_one_width(self):
        md = "| Only |\n|------|\n| value |\n"
        blk = _table_block_from(md)
        self.assertEqual(len(blk["table"]["property"]["column_width"]), 1)

    def test_row_and_column_size_unchanged(self):
        """The width work must not regress the existing row/column counts."""
        blk = _table_block_from(self.MD)
        # 2 data rows + 1 header row = 3 rows (separator already filtered)
        self.assertEqual(blk["table"]["property"]["row_size"], 3)
        self.assertEqual(blk["table"]["property"]["column_size"], 2)


if __name__ == "__main__":
    unittest.main()
