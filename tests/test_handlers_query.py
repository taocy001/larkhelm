"""
P1 — handlers/_query.py unit tests

Coverage:
  - _extract_feishu_urls    Feishu document URL extraction (pure regex)
  - _inject_doc_context     automatic document context injection (mock FeishuDocClient)
"""
import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Initialize config ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_hqtest_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "doc_auto_inject": True,
    "doc_inject_max_chars": 500,
    "doc_inject_max_docs": 2,
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.handlers._query import _extract_feishu_urls, _inject_doc_context


# ═══════════════════════════════════════════════════════════════════
#  _extract_feishu_urls
# ═══════════════════════════════════════════════════════════════════

class TestExtractFeishuUrls(unittest.TestCase):
    def test_docx_url(self):
        text = "请看这个文档 https://mycompany.feishu.cn/docx/AbCdEfGhIjKl"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertIn("docx", urls[0])

    def test_wiki_url(self):
        text = "参考 https://myco.feishu.cn/wiki/SomeWikiToken 的内容"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertIn("wiki", urls[0])

    def test_sheets_url(self):
        text = "https://acme.feishu.cn/sheets/SheetToken123?sheet=abc"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)

    def test_multiple_urls(self):
        text = ("看 https://a.feishu.cn/docx/D1 和 "
                "https://b.feishu.cn/wiki/W2 这两个")
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 2)

    def test_no_feishu_url(self):
        text = "普通消息，没有文档链接，看 https://github.com/xxx"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 0)

    def test_empty_text(self):
        self.assertEqual(_extract_feishu_urls(""), [])

    def test_url_stops_at_whitespace(self):
        text = "https://x.feishu.cn/docx/ABC def"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertNotIn(" def", urls[0])

    def test_url_stops_at_chinese_bracket(self):
        text = "（https://x.feishu.cn/docx/ABC）"
        urls = _extract_feishu_urls(text)
        # URL should not include the closing bracket
        if urls:
            self.assertNotIn("）", urls[0])

    def test_docs_url(self):
        text = "https://company.feishu.cn/docs/DocToken"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)


# ═══════════════════════════════════════════════════════════════════
#  _inject_doc_context
# ═══════════════════════════════════════════════════════════════════

def _make_doc_result(title="测试文档", content="文档正文内容"):
    r = MagicMock()
    r.title = title
    r.content = content
    return r


class TestInjectDocContext(unittest.TestCase):
    def test_no_url_returns_original(self):
        text = "普通消息，没有飞书链接"
        result = _inject_doc_context(text, "chat1")
        self.assertEqual(result, text)

    # _inject_doc_context imports FeishuDocClient / parse_doc_url from larkhelm.lark_client internally,
    # so we must patch the lark_client module rather than the _query module
    _PATCH_CLIENT = "larkhelm.lark_client.FeishuDocClient"
    _PATCH_PARSE  = "larkhelm.lark_client.parse_doc_url"

    def test_url_triggers_injection(self):
        text = "请读 https://x.feishu.cn/docx/ABC"
        mock_client = MagicMock()
        mock_client.read.return_value = _make_doc_result("标题A", "内容A")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertIn("文档内容", result)
        self.assertIn("标题A", result)
        self.assertIn("内容A", result)
        self.assertIn(text, result)

    def test_permission_error_adds_placeholder(self):
        from larkhelm.lark_client import DocPermissionError
        text = "https://x.feishu.cn/docx/ABC"
        mock_client = MagicMock()
        mock_client.read.side_effect = DocPermissionError("no perm")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertIn("无读取权限", result)

    def test_doc_error_silently_skipped(self):
        from larkhelm.lark_client import DocError
        text = "https://x.feishu.cn/docx/ABC"
        mock_client = MagicMock()
        mock_client.read.side_effect = DocError("some error")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertEqual(result, text)

    def test_max_docs_limit(self):
        """Documents beyond doc_inject_max_docs=2 should not be injected."""
        text = ("https://x.feishu.cn/docx/A "
                "https://x.feishu.cn/docx/B "
                "https://x.feishu.cn/docx/C")
        call_count = []
        mock_client = MagicMock()

        def mock_read(ref, max_chars):
            call_count.append(1)
            return _make_doc_result(f"doc{len(call_count)}", f"content{len(call_count)}")

        mock_client.read.side_effect = mock_read

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            _inject_doc_context(text, "chat1")

        self.assertLessEqual(len(call_count), _cfg_module.DOC_INJECT_MAX_DOCS)

    def test_unrecognized_url_returns_original(self):
        """When parse_doc_url returns None the URL should be skipped."""
        text = "https://x.feishu.cn/some-unknown-path/token"
        mock_client = MagicMock()

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=None):
            result = _inject_doc_context(text, "chat1")

        self.assertEqual(result, text)
        mock_client.read.assert_not_called()

    def test_injected_content_prepended(self):
        """Injected content should appear before the original message."""
        text = "原始消息 https://x.feishu.cn/docx/T"
        mock_client = MagicMock()
        mock_client.read.return_value = _make_doc_result("标题", "内容")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertLess(result.index("文档内容"), result.index("原始消息"))

    def test_separator_between_injections_and_original(self):
        """There should be a --- separator between injected content and the original message."""
        text = "消息 https://x.feishu.cn/docx/T"
        mock_client = MagicMock()
        mock_client.read.return_value = _make_doc_result("T", "C")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertIn("---", result)


if __name__ == "__main__":
    unittest.main()
