"""
P1 — handlers/_query.py:_maybe_doc_usage_hint tests (REQ-06, AC-05)

Covers all 4 branches:
  1. Single unrecognised URL  → send orange hint, return True
  2. Single recognised URL    → no hint, return False
  3. Multiple URLs            → no hint, return False
  4. URL with extra text      → no hint, return False
"""
import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

_TMP = tempfile.mkdtemp(prefix="larkhelm_qhint_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.handlers import _query as _q  # noqa: E402


class TestMaybeDocUsageHint(unittest.TestCase):

    def test_single_failed_url_sends_hint(self):
        url = "https://example.feishu.cn/unknown/abc123"
        with patch.object(_q, "send_card_reply") as send_reply, \
             patch("larkhelm.lark_client.parse_doc_url",
                   return_value=None) as parse:
            result = _q._maybe_doc_usage_hint(url, "chat_a", "msg_1")
            self.assertTrue(result)
            self.assertEqual(send_reply.call_count, 1)
            call_kwargs = send_reply.call_args.kwargs
            call_args = send_reply.call_args.args
            # Color check (kwarg)
            self.assertEqual(call_kwargs.get("color"), "orange")
            # Body should mention all 4 supported types
            body = call_args[3] if len(call_args) > 3 else call_kwargs.get("body", "")
            for kw in ("docx", "wiki", "sheets", "folder"):
                self.assertIn(kw, body,
                              f"hint body missing {kw!r}: {body[:200]}")
            self.assertTrue(parse.called)

    def test_single_valid_url_no_hint(self):
        url = "https://example.feishu.cn/docx/abc123"
        with patch.object(_q, "send_card_reply") as send_reply, \
             patch("larkhelm.lark_client.parse_doc_url",
                   return_value=MagicMock()):
            result = _q._maybe_doc_usage_hint(url, "chat_b", "msg_2")
            self.assertFalse(result)
            self.assertEqual(send_reply.call_count, 0)

    def test_multi_urls_no_hint(self):
        text = ("https://x.feishu.cn/docx/aaa "
                "https://x.feishu.cn/wiki/bbb")
        with patch.object(_q, "send_card_reply") as send_reply, \
             patch("larkhelm.lark_client.parse_doc_url",
                   return_value=None):
            result = _q._maybe_doc_usage_hint(text, "chat_c", "msg_3")
            self.assertFalse(result)
            self.assertEqual(send_reply.call_count, 0)

    def test_url_with_text_no_hint(self):
        text = "看下 https://x.feishu.cn/foo 这个"
        with patch.object(_q, "send_card_reply") as send_reply, \
             patch("larkhelm.lark_client.parse_doc_url",
                   return_value=None):
            result = _q._maybe_doc_usage_hint(text, "chat_d", "msg_4")
            self.assertFalse(result)
            self.assertEqual(send_reply.call_count, 0)


if __name__ == "__main__":
    unittest.main()
