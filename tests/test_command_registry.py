"""Tests for ``larkhelm.command_registry`` (S1+S7 — Phase B).

Covered:
  * ``CommandSpec.matches`` for exact + prefix + sub_matches.
  * ``CommandSpec.extract_args`` strips the matched token and trims whitespace.
  * ``CommandRegistry.dispatch`` empty-args + usage_card path.
  * ``CommandRegistry.dispatch`` async-handler spawns a daemon thread.
  * Duplicate registration raises ValueError.
  * Unknown command returns "unhandled".
  * Aliases are honoured (``/lock`` mirrors ``/model``).
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from larkhelm.command_registry import (
    CommandRegistry, CommandSpec, DispatchContext,
)


def _ctx(text: str) -> DispatchContext:
    return DispatchContext(
        chat_id="oc_test", msg_id="m1", text=text,
        tl=text.lower().strip(),
    )


# ── CommandSpec ─────────────────────────────────────────────────────


class CommandSpecTests(unittest.TestCase):

    def test_exact_match(self):
        spec = CommandSpec(name="/status", handler=lambda c: None)
        self.assertTrue(spec.matches("/status"))
        self.assertFalse(spec.matches("/status foo"))
        self.assertFalse(spec.matches("/help"))

    def test_prefix_match(self):
        spec = CommandSpec(name="/run", handler=lambda c: None, match_kind="prefix")
        self.assertTrue(spec.matches("/run"))
        self.assertTrue(spec.matches("/run echo hi"))
        self.assertFalse(spec.matches("/running"))

    def test_sub_matches_override(self):
        spec = CommandSpec(
            name="/reset",
            handler=lambda c: None,
            sub_matches=("/reset claude", "/reset memory"),
        )
        self.assertTrue(spec.matches("/reset"))
        self.assertTrue(spec.matches("/reset claude"))
        self.assertTrue(spec.matches("/reset memory"))
        self.assertFalse(spec.matches("/reset foo"))

    def test_aliases_match(self):
        spec = CommandSpec(name="/model", aliases=("/lock",),
                           handler=lambda c: None, match_kind="prefix")
        self.assertTrue(spec.matches("/model claude"))
        self.assertTrue(spec.matches("/lock kimi"))

    def test_extract_args_prefix(self):
        spec = CommandSpec(name="/run", handler=lambda c: None, match_kind="prefix")
        self.assertEqual(spec.extract_args("/run echo hi"), "echo hi")
        self.assertEqual(spec.extract_args("/run"), "")
        self.assertEqual(spec.extract_args("/RUN echo hi"), "echo hi",
                         "case-insensitive match must preserve arg casing")

    def test_extract_args_sub_match(self):
        spec = CommandSpec(
            name="/reset",
            handler=lambda c: None,
            sub_matches=("/reset claude",),
        )
        self.assertEqual(spec.extract_args("/reset claude"), "")


# ── CommandRegistry.dispatch ───────────────────────────────────────


class DispatchTests(unittest.TestCase):

    def setUp(self):
        self.reg = CommandRegistry()

    def test_unknown_command_unhandled(self):
        called: list[str] = []
        self.reg.register(CommandSpec(
            name="/status", handler=lambda c: called.append("status"),
        ))
        result = self.reg.dispatch(_ctx("/missing"))
        self.assertEqual(result, "unhandled")
        self.assertEqual(called, [])

    def test_exact_match_dispatched(self):
        called: list[str] = []
        self.reg.register(CommandSpec(
            name="/status", handler=lambda c: called.append(c.chat_id),
        ))
        result = self.reg.dispatch(_ctx("/status"))
        self.assertEqual(result, "handled")
        self.assertEqual(called, ["oc_test"])

    def test_prefix_args_passed(self):
        captured: dict = {}
        self.reg.register(CommandSpec(
            name="/run",
            match_kind="prefix",
            handler=lambda c: captured.setdefault("args", c.raw_args),
        ))
        self.reg.dispatch(_ctx("/run echo hi"))
        self.assertEqual(captured["args"], "echo hi")

    def test_empty_args_usage_card_short_circuits(self):
        called: list[str] = []
        self.reg.register(CommandSpec(
            name="/run",
            match_kind="prefix",
            usage_card="`/run <command>`",
            handler=lambda c: called.append("ran"),
        ))
        # Registry routes usage_card via ``handlers._message.send_card_reply``
        # so existing _message-mocking tests continue to catch the call.
        with patch("larkhelm.handlers._message.send_card_reply") as send:
            result = self.reg.dispatch(_ctx("/run"))
        self.assertEqual(result, "handled")
        self.assertEqual(called, [])
        send.assert_called_once()
        # The handler MUST be skipped when usage_card fires.
        # (No "ran" entry above asserts that.)

    def test_sub_match_dispatched(self):
        captured: dict = {}
        self.reg.register(CommandSpec(
            name="/reset",
            handler=lambda c: captured.setdefault("tl", c.tl),
            sub_matches=("/reset claude",),
        ))
        self.reg.dispatch(_ctx("/reset claude"))
        self.assertEqual(captured["tl"], "/reset claude")

    def test_async_handler_spawns_thread(self):
        done = threading.Event()

        def _handler(ctx):
            done.set()

        self.reg.register(CommandSpec(
            name="/crew",
            handler=_handler,
            match_kind="prefix",
            run_async=True,
            thread_label="Crew",
        ))
        # Use a real Thread spawn (the registry uses threading.Thread directly).
        self.reg.dispatch(_ctx("/crew plan refactor"))
        # The handler runs on a daemon thread; wait briefly.
        self.assertTrue(done.wait(timeout=2.0),
                        "async handler must run on a spawned thread")

    def test_async_handler_exception_routes_to_error_card(self):
        """An exception inside an async handler should NOT crash the bridge —
        it should be funnelled through ``_thread_error_card`` (which we
        patch here so the test doesn't depend on real Feishu cards)."""
        def _boom(ctx):
            raise RuntimeError("simulated crash")

        self.reg.register(CommandSpec(
            name="/crew",
            handler=_boom,
            match_kind="prefix",
            run_async=True,
            thread_label="Crew",
        ))
        with patch("larkhelm.handlers._message._thread_error_card") as err:
            self.reg.dispatch(_ctx("/crew x"))
            time.sleep(0.2)  # let the daemon thread surface
        err.assert_called_once()
        # First arg is chat_id, second is label
        args = err.call_args.args
        self.assertEqual(args[0], "oc_test")
        self.assertEqual(args[1], "Crew")

    def test_duplicate_registration_raises(self):
        self.reg.register(CommandSpec(name="/foo", handler=lambda c: None))
        with self.assertRaises(ValueError):
            self.reg.register(CommandSpec(name="/foo", handler=lambda c: None))

    def test_aliases_dispatch_same_handler(self):
        seen: list[str] = []
        self.reg.register(CommandSpec(
            name="/model",
            handler=lambda c: seen.append(c.tl),
            match_kind="prefix",
            aliases=("/lock",),
        ))
        self.reg.dispatch(_ctx("/lock kimi"))
        self.reg.dispatch(_ctx("/model claude"))
        self.assertEqual(len(seen), 2)


class DefaultRegistrationsTests(unittest.TestCase):
    """The module-level COMMAND_REGISTRY must wire up all expected commands."""

    def test_default_registrations_present(self):
        from larkhelm.command_registry import COMMAND_REGISTRY
        names = {s.name for s in COMMAND_REGISTRY.iter_visible()}
        # A representative subset — full coverage is in test_handlers_message_routing.
        for expected in ("/status", "/help", "/reset", "/run", "/cd", "/ls",
                         "/dev", "/crew", "/plan", "/memory", "/doc", "/cron",
                         "/model", "/voice"):
            self.assertIn(expected, names, f"{expected} missing from default registry")


if __name__ == "__main__":
    unittest.main()
