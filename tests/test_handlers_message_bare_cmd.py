"""
P1 — handlers/_message.py tests (REQ-04 / REQ-05, AC-03 / AC-04)

Two groups:
  - TestBareCmd:         /cd and /run with no args route to a usage card and
                         must NOT call _cmd_cd / _cmd_run / _do_query.
  - TestThreadErrorCard: _thread_error_card logs full traceback and surfaces
                         a red card containing the (truncated) exception.
"""
import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

_TMP = tempfile.mkdtemp(prefix="larkhelm_msgtest_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.handlers import _message as _m  # noqa: E402


def _build_event(text: str, chat_id: str = "chat_test",
                 msg_id: str = "msg_test") -> SimpleNamespace:
    """Build a minimal stand-in for P2ImMessageReceiveV1.

    Only the attributes that ``handle_message`` actually touches before
    reaching the routing chain need to exist; ``SimpleNamespace`` keeps the
    structure trivial.
    """
    content_json = json.dumps({"text": text})
    message = SimpleNamespace(
        message_type="text",
        content=content_json,
        chat_id=chat_id,
        message_id=msg_id,
        chat_type="p2p",
        mentions=None,
        parent_id=None,
    )
    sender = SimpleNamespace(sender_id=SimpleNamespace(open_id="user_open_id"))
    event = SimpleNamespace(message=message, sender=sender)
    header = SimpleNamespace(event_id=f"ev_{msg_id}")
    return SimpleNamespace(event=event, header=header)


class TestBareCmd(unittest.TestCase):
    """`/cd` / `/run` with no arg → orange 用法 card, no command dispatch."""

    def _run(self, cmd: str):
        data = _build_event(cmd, chat_id=f"chat_{cmd[1:]}",
                            msg_id=f"mid_{cmd[1:]}")
        with patch.object(_m, "send_card_reply") as send_reply, \
             patch("larkhelm.commands._cmd_cd") as cd_mock, \
             patch("larkhelm.commands._cmd_run") as run_mock, \
             patch("larkhelm.handlers._query._do_query") as do_query:
            _m.handle_message(data)
            return send_reply, cd_mock, run_mock, do_query

    def test_bare_cd_sends_usage_card(self):
        send_reply, cd_mock, run_mock, do_query = self._run("/cd")
        self.assertEqual(send_reply.call_count, 1)
        kwargs = send_reply.call_args.kwargs
        args = send_reply.call_args.args
        self.assertEqual(kwargs.get("color"), "orange")
        body_arg = args[3] if len(args) > 3 else kwargs.get("body", "")
        self.assertIn("用法", body_arg + kwargs.get("title", "") + (args[2] if len(args) > 2 else ""))
        self.assertEqual(cd_mock.call_count, 0)
        self.assertEqual(run_mock.call_count, 0)
        self.assertEqual(do_query.call_count, 0)

    def test_bare_run_sends_usage_card(self):
        send_reply, cd_mock, run_mock, do_query = self._run("/run")
        self.assertEqual(send_reply.call_count, 1)
        kwargs = send_reply.call_args.kwargs
        args = send_reply.call_args.args
        self.assertEqual(kwargs.get("color"), "orange")
        body_arg = args[3] if len(args) > 3 else kwargs.get("body", "")
        self.assertIn("用法", body_arg + kwargs.get("title", "") + (args[2] if len(args) > 2 else ""))
        self.assertEqual(cd_mock.call_count, 0)
        self.assertEqual(run_mock.call_count, 0)
        self.assertEqual(do_query.call_count, 0)


class TestThreadErrorCard(unittest.TestCase):
    """``_thread_error_card`` writes full traceback to debug log and sends a
    red error card carrying the truncated exception message."""

    def test_writes_traceback_and_sends_red_card(self):
        # Run the helper inside an active ``except`` block so
        # ``traceback.format_exc()`` returns the real stack trace, mirroring
        # how the real daemon-thread wrappers call it (``except Exception
        # as _e: _thread_error_card(...)``).
        with patch.object(_m, "_debug_log") as dbg, \
             patch.object(_m, "send_card") as send:
            try:
                raise ValueError("boom")
            except ValueError as exc:
                _m._thread_error_card("chat_xyz", "Plan", exc)
        # ── _debug_log must record the traceback ──────────────────────
        self.assertGreaterEqual(dbg.call_count, 1)
        logged = "\n".join(
            str(call.args[0]) if call.args else "" for call in dbg.call_args_list
        )
        # The label tag + exception summary + a traceback marker should all show up.
        self.assertIn("[Plan]", logged)
        self.assertIn("boom", logged)
        self.assertIn("Traceback", logged)
        # ── send_card called once with color=red, contains 失败 + boom ──
        self.assertEqual(send.call_count, 1)
        args = send.call_args.args
        kwargs = send.call_args.kwargs
        body = args[2] if len(args) > 2 else kwargs.get("body", "")
        self.assertEqual(kwargs.get("color"), "red")
        self.assertIn("失败", body + (args[1] if len(args) > 1 else ""))
        self.assertIn("boom", body)
        # 200-char truncation guard: body should not be ridiculously long
        self.assertLess(len(body), 400)


if __name__ == "__main__":
    unittest.main()
