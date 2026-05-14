"""
P1 — lark_client.py multi-chunk dispatch tests (REQ-03)

Covers ``_send_chunks_compat`` indirectly via ``send_card`` /
``send_card_reply`` / ``update_card``. Mocks the raw transport functions so
no real Feishu API calls happen.
"""
import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Initialize config with small max_card_len so we can hit the split path
# without generating multi-thousand-char fixtures.
_TMP = tempfile.mkdtemp(prefix="larkhelm_lctest_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "max_card_len": 100,
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

import larkhelm.lark_client as _lc  # noqa: E402


# Pin MAX_CARD_LEN to 100 inside this module so other test files that reset
# ``_init_runtime`` (e.g. with the default 3000) can't poison split-fixture
# size assumptions. Tests set this value in ``setUp`` so order doesn't matter.
_FORCED_MAX_CARD_LEN = 100


def _make_body(total_chars: int) -> str:
    """Build a multi-line body whose total length is ``total_chars`` but where
    each line stays comfortably under MAX_CARD_LEN so ``_split_md`` flushes
    at line boundaries (not via the single-oversized-line guard)."""
    line = "x" * 50  # 50 + 1 newline = 51 chars per line, well under 100
    n = max(1, total_chars // (len(line) + 1))
    return "\n".join([line] * n)


class _BaseChunkTest(unittest.TestCase):
    def setUp(self):
        self._saved_max = _cfg_module.MAX_CARD_LEN
        _cfg_module.MAX_CARD_LEN = _FORCED_MAX_CARD_LEN

    def tearDown(self):
        _cfg_module.MAX_CARD_LEN = self._saved_max


class TestSendCardChunks(_BaseChunkTest):

    def test_send_card_long_body_calls_raw_N_times(self):
        body = _make_body(350)
        chunks = _lc._split_md(body.strip())
        expected = len(chunks)
        self.assertGreaterEqual(expected, 2,
                                "fixture must actually trigger splitting")
        with patch.object(_lc, "_send_card_raw",
                          return_value="mid_x") as raw:
            _lc.send_card("chat_long", "Title", body)
            self.assertEqual(raw.call_count, expected)

    def test_send_card_short_body_single_call(self):
        with patch.object(_lc, "_send_card_raw",
                          return_value="mid_short") as raw:
            _lc.send_card("chat_short", "Title", "tiny body")
            self.assertEqual(raw.call_count, 1)


class TestSendCardReplyChunks(_BaseChunkTest):

    def test_first_chunk_uses_reply_then_continues_via_send(self):
        body = _make_body(350)
        chunks = _lc._split_md(body.strip())
        cont = len(chunks) - 1
        self.assertGreaterEqual(cont, 1,
                                "fixture must actually trigger splitting")
        with patch.object(_lc, "_reply_card_raw",
                          return_value="mid_reply") as reply, \
             patch.object(_lc, "_send_card_raw",
                          return_value="mid_send") as send:
            _lc.send_card_reply("chat_reply", "user_mid", "Title", body)
            self.assertEqual(reply.call_count, 1)
            self.assertEqual(send.call_count, cont)


class TestUpdateCardChunks(_BaseChunkTest):

    def test_with_chat_id_sends_remainder(self):
        body = _make_body(350)
        chunks = _lc._split_md(body.strip())
        cont = len(chunks) - 1
        self.assertGreaterEqual(cont, 1,
                                "fixture must actually trigger splitting")
        with patch.object(_lc, "_patch_card_raw",
                          return_value=True) as patch_raw, \
             patch.object(_lc, "_send_card_raw",
                          return_value="mid_cont") as send_raw:
            ok = _lc.update_card(
                "msg_id_xyz", "Title", body,
                chat_id="chat_with_id",
            )
            self.assertTrue(ok)
            self.assertEqual(patch_raw.call_count, 1)
            self.assertEqual(send_raw.call_count, cont)

    def test_without_chat_id_drops_remainder(self):
        body = _make_body(350)
        with patch.object(_lc, "_patch_card_raw",
                          return_value=True) as patch_raw, \
             patch.object(_lc, "_send_card_raw",
                          return_value="mid_cont") as send_raw:
            ok = _lc.update_card("msg_id_xyz", "Title", body)
            self.assertTrue(ok)
            self.assertEqual(patch_raw.call_count, 1)
            self.assertEqual(send_raw.call_count, 0,
                             "continuation chunks must be dropped when chat_id absent")


if __name__ == "__main__":
    unittest.main()
