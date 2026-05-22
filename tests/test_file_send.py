"""Tests for send_text_as_file (AC-07)."""
from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import json
_TMP = tempfile.mkdtemp(prefix="larkhelm_send_test_")
import atexit; atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_path = pathlib.Path(_TMP) / "config.json"
_cfg_path.write_text(json.dumps({"APP_ID": "X", "APP_SECRET": "Y"}))

import larkhelm.config as _cfg
_cfg._init_runtime(str(_cfg_path), str(_TMP))


class TestSendTextAsFile(unittest.TestCase):
    """AC-07: send_text_as_file uploads text and sends as file message."""

    def test_send_text_as_file_basic(self):
        from larkhelm.lark_client import send_text_as_file

        with patch("larkhelm.lark_client.upload_file_to_feishu",
                   return_value="file_key_123") as mock_upload, \
             patch("larkhelm.lark_client.send_file_message",
                   return_value="msg_id_abc") as mock_send:
            result = send_text_as_file("chat_id_1", "report content", "report.md")

        self.assertEqual(result, "msg_id_abc")
        mock_upload.assert_called_once()
        mock_send.assert_called_once()

        # The upload path should include the target filename
        upload_path_arg = mock_upload.call_args[0][0]
        self.assertIn("report.md", str(upload_path_arg))

    def test_send_text_as_file_with_msg_id(self):
        """msg_id is forwarded to send_file_message for thread replies."""
        from larkhelm.lark_client import send_text_as_file

        with patch("larkhelm.lark_client.upload_file_to_feishu",
                   return_value="fk_reply"), \
             patch("larkhelm.lark_client.send_file_message",
                   return_value="mid_reply") as mock_send:
            result = send_text_as_file("chat_reply", "text", "out.txt", msg_id="parent_msg")

        self.assertEqual(result, "mid_reply")
        _, kw = mock_send.call_args
        passed_msg_id = kw.get("msg_id") or mock_send.call_args[0][2]
        self.assertEqual(passed_msg_id, "parent_msg")

    def test_send_text_as_file_upload_failure_returns_none(self):
        from larkhelm.lark_client import send_text_as_file

        with patch("larkhelm.lark_client.upload_file_to_feishu", return_value=None):
            result = send_text_as_file("chat_fail", "text", "f.txt")

        self.assertIsNone(result)

    def test_send_text_as_file_no_exception_on_error(self):
        """send_text_as_file should never raise exceptions."""
        from larkhelm.lark_client import send_text_as_file

        with patch("larkhelm.lark_client.upload_file_to_feishu",
                   side_effect=RuntimeError("boom")):
            result = send_text_as_file("chat_err", "content", "f.txt")

        self.assertIsNone(result)

    def test_send_text_as_file_content_encoded_utf8(self):
        """Unicode content should be written as UTF-8."""
        from larkhelm.lark_client import send_text_as_file

        written_data = []

        real_write = os.write
        def tracking_write(fd, data):
            written_data.append(data)
            return real_write(fd, data)

        with patch("os.write", side_effect=tracking_write), \
             patch("larkhelm.lark_client.upload_file_to_feishu", return_value="fk"), \
             patch("larkhelm.lark_client.send_file_message", return_value="mid"):
            send_text_as_file("chat_utf8", "你好世界 — hello", "report.md")

        # Find the write that encoded the text content
        utf8_encoded = "你好世界 — hello".encode("utf-8")
        any_match = any(utf8_encoded in d for d in written_data)
        self.assertTrue(any_match, "Content should be encoded as UTF-8")

    def test_send_text_as_file_temp_cleanup_on_success(self):
        """Temp file is cleaned up after successful upload."""
        from larkhelm.lark_client import send_text_as_file

        seen_paths = []
        real_rename = pathlib.Path.rename
        def tracking_rename(self_p, target):
            seen_paths.append(str(target))
            real_rename(self_p, target)

        with patch.object(pathlib.Path, "rename", tracking_rename), \
             patch("larkhelm.lark_client.upload_file_to_feishu", return_value="fk"), \
             patch("larkhelm.lark_client.send_file_message", return_value="mid"):
            send_text_as_file("chat_cleanup", "data", "result.txt")

        if seen_paths:
            # Final tmp path should no longer exist
            self.assertFalse(pathlib.Path(seen_paths[-1]).exists(),
                             "Temp file should be deleted after send")


if __name__ == "__main__":
    unittest.main(verbosity=2)
