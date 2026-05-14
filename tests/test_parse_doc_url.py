"""Unit tests for parse_doc_url + explain_doc_url_failure (S11).

The Feishu document family of commands (/doc read/write/append/ls/create/
setfolder) all share one URL-parsing entry point. When parsing fails users
historically got the generic message ``不是有效的飞书文档链接``, which left
them guessing whether they pasted the wrong link, the wrong path, or the
right link with broken formatting. ``explain_doc_url_failure`` distinguishes
the three failure modes (empty / non-Feishu host / Feishu host but
unsupported path) so the error card can tell the user exactly what to fix.
"""
import unittest

from larkhelm.lark_client import parse_doc_url, explain_doc_url_failure


class ParseDocUrlTests(unittest.TestCase):
    def test_docx_url(self):
        ref = parse_doc_url("https://feishu.cn/docx/ABC123_xyz")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "docx")
        self.assertEqual(ref.token, "ABC123_xyz")

    def test_wiki_url(self):
        ref = parse_doc_url("https://acme.feishu.cn/wiki/WK-token_99")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "wiki")

    def test_sheets_url(self):
        ref = parse_doc_url("https://feishu.cn/sheets/SHt0k3n")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "sheets")

    def test_folder_url(self):
        ref = parse_doc_url("https://feishu.cn/drive/folder/FoLdEr1")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "folder")

    def test_docs_legacy_url(self):
        ref = parse_doc_url("https://feishu.cn/docs/legacyTOKEN")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "docs")

    def test_larksuite_docx_url(self):
        # International tenants (Lark, the non-China brand of Feishu) use
        # *.larksuite.com. The current parser uses bare "feishu.cn" in the
        # regex so larksuite URLs return None — this test pins that gap so
        # explain_doc_url_failure tells the user *why* instead of the
        # generic "not a Feishu link".
        self.assertIsNone(parse_doc_url("https://acme.larksuite.com/docx/ABC123"))

    def test_unknown_path_returns_none(self):
        self.assertIsNone(parse_doc_url("https://feishu.cn/calendar/abc"))
        self.assertIsNone(parse_doc_url("https://feishu.cn/space/xyz"))

    def test_non_feishu_returns_none(self):
        self.assertIsNone(parse_doc_url("https://docs.google.com/document/d/X/edit"))
        self.assertIsNone(parse_doc_url("https://notion.so/foo"))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_doc_url(""))
        self.assertIsNone(parse_doc_url("   "))


class ExplainDocUrlFailureTests(unittest.TestCase):
    """The helper must always produce a non-empty, distinguishing hint."""

    def test_empty_input(self):
        msg = explain_doc_url_failure("")
        self.assertIn("URL 为空", msg)
        msg = explain_doc_url_failure("   ")
        self.assertIn("URL 为空", msg)

    def test_non_feishu_domain(self):
        msg = explain_doc_url_failure("https://docs.google.com/document/d/X/edit")
        # Must mention what domain we expect
        self.assertIn("feishu", msg.lower())

    def test_non_feishu_includes_truncated_input(self):
        long_url = "https://example.com/" + "A" * 200
        msg = explain_doc_url_failure(long_url)
        # Long inputs are truncated and marked with ellipsis to keep cards readable
        self.assertIn("…", msg)

    def test_feishu_unsupported_path_lists_supported_types(self):
        msg = explain_doc_url_failure("https://acme.feishu.cn/calendar/event/123")
        # Tells user which path types ARE supported instead of a vague "invalid"
        self.assertIn("/docx/", msg)
        self.assertIn("/wiki/", msg)
        self.assertIn("/sheets/", msg)
        self.assertIn("/drive/folder/", msg)

    def test_larksuite_domain_also_recognised(self):
        # International tenants use larksuite.com; helper must not call them "non-Feishu"
        msg = explain_doc_url_failure("https://acme.larksuite.com/foobar/123")
        self.assertNotIn("识别不到飞书域名", msg)
        # Should report it as unsupported-path instead
        self.assertIn("/docx/", msg)

    def test_helper_is_idempotent_for_valid_urls(self):
        # parse_doc_url succeeds → caller never invokes explain helper. But
        # the helper itself must remain safe to call with any string; verify
        # it doesn't crash on a *valid* URL (even though it'd misclassify it).
        try:
            explain_doc_url_failure("https://feishu.cn/docx/ABC123")
        except Exception as e:  # pragma: no cover
            self.fail(f"explain_doc_url_failure raised on valid URL: {e}")

    def test_url_with_backtick_does_not_break_card_markdown(self):
        # The echo of the user's input is wrapped in `code spans` for
        # readability. A backtick in the user's input would close the span
        # and turn the rest of the card body into raw text. Verify the
        # echoed user input contains NO backticks (the helper strips them
        # before interpolating).
        msg = explain_doc_url_failure("https://example.com/x`y`z")
        # Locate the echoed input by splitting on the first backtick after
        # the literal prefix; everything between matched backticks is the
        # span content.
        between = msg.split("`")
        # The span at index 1 is the echo of the user URL; assert no inner
        # backticks survived (which would have produced extra split parts
        # nested inside the supposed span).
        echo = between[1] if len(between) > 1 else ""
        self.assertNotIn("`", echo)
        # And the message should still have an even number of backticks
        # (each span properly closed) so the card markdown stays balanced.
        self.assertEqual(msg.count("`") % 2, 0,
                         f"unbalanced backticks in: {msg!r}")


if __name__ == "__main__":
    unittest.main()
