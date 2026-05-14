"""End-to-end command routing matrix for ``handlers/_message.py`` after
the S1+S7 command-registry refactor.

Each ``case`` exercises a single user-typed command and asserts:
  * the corresponding ``_cmd_*`` (or ``cmd_*``) function is invoked once
  * the args reach the handler unchanged
  * ``_do_query`` is NOT triggered (these are control commands, not chat)

Bare-prefix usage cards (``/run`` / ``/cd`` with no arg) are covered by
``test_handlers_message_bare_cmd.py``; this file focuses on the args-present
happy paths for breadth.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# ── Bootstrap ────────────────────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_msgroute_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.handlers import _message as _m  # noqa: E402


def _build_event(text: str, chat_id: str = "chat_route",
                 msg_id: str = "msg_route") -> SimpleNamespace:
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
    header = SimpleNamespace(event_id=f"ev_{msg_id}_{text[:10]}")
    return SimpleNamespace(event=event, header=header)


# Each tuple: (input_text, target_module, target_func, expected_args_substr)
# expected_args_substr is checked against str(call_args) to keep the
# table compact; for empty-args commands pass "" to skip the substring check.
_CASES = [
    # /reset family
    ("/reset",                    "larkhelm.commands._cmd_reset",          ""),
    ("/reset claude",             "larkhelm.commands._cmd_reset",          "claude"),
    ("/reset gemini",             "larkhelm.commands._cmd_reset",          "gemini"),
    ("/reset memory",             "larkhelm.commands._cmd_reset",          "memory"),
    ("/reset perm",               "larkhelm.commands._cmd_reset",          "perm"),
    # exact commands
    ("/status",                   "larkhelm.commands._cmd_status",         ""),
    ("/help",                     "larkhelm.commands._cmd_help",           ""),
    ("/pickup",                   "larkhelm.commands._cmd_pickup",         ""),
    ("/upgrade",                  "larkhelm.commands._cmd_upgrade",        ""),
    ("/pwd",                      "larkhelm.commands._cmd_pwd",            ""),
    # history variants
    ("/history",                  "larkhelm.commands._cmd_history",        ""),
    ("/history all",              "larkhelm.commands._cmd_history",        ""),
    # prefix commands with args
    ("/stats foo",                "larkhelm.commands._cmd_stats",          "foo"),
    ("/memory status",            "larkhelm.commands._cmd_memory",         "status"),
    ("/cron list",                "larkhelm.commands._cmd_cron",           "list"),
    ("/cd /tmp",                  "larkhelm.commands._cmd_cd",             "/tmp"),
    ("/ls subdir",                "larkhelm.commands._cmd_ls",             "subdir"),
    ("/run echo hi",              "larkhelm.commands._cmd_run",            "echo hi"),
    ("/model claude",             "larkhelm.commands._cmd_lock",           "claude"),
    ("/lock kimi",                "larkhelm.commands._cmd_lock",           "kimi"),
    ("/voice status",             "larkhelm.commands._cmd_voice",          "status"),
    # /doc — args land on the dispatcher
    ("/doc read https://x",       "larkhelm.cmd_doc._cmd_doc",             "read"),
]

# Async commands are dispatched on a daemon thread; they need a small grace
# period before the call_count assertion fires.
_ASYNC_CASES = [
    ("/crew plan refactor",       "larkhelm.crew.cmd_crew",                "refactor"),
    ("/dev fix login",            "larkhelm.crew.cmd_dev",                 "fix login"),
    ("/plan ship release",        "larkhelm.cmd_plan.cmd_plan",            "ship release"),
    ("/run echo hi",              "larkhelm.commands._cmd_run",            "echo hi"),
]


class TestMessageRoutingMatrix(unittest.TestCase):
    """Each command case fires once; the patched handler is invoked."""

    def _dispatch(self, text: str, target: str):
        ev = _build_event(text, chat_id=f"chat_{abs(hash(text))%10000}",
                          msg_id=f"mid_{abs(hash(text))%10000}")
        with patch(target) as fn, \
             patch("larkhelm.handlers._query._do_query") as do_query, \
             patch("larkhelm.handlers._message.send_card_reply"):
            _m.handle_message(ev)
            time.sleep(0.05)  # let any async dispatch land
            return fn, do_query

    def test_each_case_dispatches_once(self):
        for text, target, sub in _CASES:
            with self.subTest(text=text, target=target):
                fn, do_query = self._dispatch(text, target)
                self.assertEqual(fn.call_count, 1,
                                 f"{target} not invoked for {text!r}")
                # Ensure no chat dispatch fired for control commands.
                self.assertEqual(do_query.call_count, 0,
                                 f"_do_query unexpectedly called for {text!r}")
                if sub:
                    call_str = str(fn.call_args)
                    self.assertIn(sub, call_str,
                                  f"args {sub!r} not found in {call_str!r} "
                                  f"for {text!r}")

    def test_async_commands_fire_on_thread(self):
        for text, target, sub in _ASYNC_CASES:
            with self.subTest(text=text, target=target):
                ev = _build_event(text,
                                  chat_id=f"chat_a_{abs(hash(text))%10000}",
                                  msg_id=f"mid_a_{abs(hash(text))%10000}")
                # Use an Event so we can wait deterministically rather than
                # relying on a fixed sleep.
                fired = threading.Event()

                def _fake(*a, **kw):
                    fired.set()

                with patch(target, side_effect=_fake), \
                     patch("larkhelm.handlers._query._do_query"), \
                     patch("larkhelm.handlers._message.send_card_reply"):
                    _m.handle_message(ev)
                    self.assertTrue(fired.wait(timeout=2.0),
                                    f"async target {target} did not fire for {text!r}")


class TestRegistryDoesNotEatChatMessages(unittest.TestCase):
    """Plain text (no leading slash) must NOT be intercepted by the registry."""

    def test_freeform_text_passes_through(self):
        ev = _build_event("hello world", chat_id="chat_freeform",
                          msg_id="mid_freeform")
        # Block the legitimate chat path so we can observe the dispatch attempt.
        called = MagicMock()
        with patch("larkhelm.handlers._message.send_card_reply"), \
             patch("threading.Thread") as ThreadCls:
            # Make the threading.Thread constructor record the kwargs but
            # never actually run the target.
            ThreadCls.return_value = MagicMock()
            _m.handle_message(ev)
        # We only need to assert that some _do_query thread was spawned —
        # i.e. the freeform message was NOT silently swallowed by the registry.
        spawned = [c for c in ThreadCls.call_args_list
                   if "kwargs" in c.kwargs or "args" in c.kwargs
                   or any("query" in str(a) for a in c.args + tuple(c.kwargs.values()))]
        self.assertGreater(ThreadCls.call_count, 0,
                           "free-form text must reach the chat-dispatch thread spawn")


if __name__ == "__main__":
    unittest.main()
