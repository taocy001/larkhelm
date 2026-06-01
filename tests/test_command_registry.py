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

    def test_prefix_match_accepts_any_whitespace(self):
        """Regression: ``/plan\\n<task>`` (and tab / full-width space) used to
        fall through to model dispatch because matching required a single
        ASCII space after the command token. Any Unicode whitespace counts
        now — newline, tab, and U+3000 (full-width space) all separate the
        command from its arguments. This is the user-visible fix for
        "larkhelm 的 /plan 总是跑到 Claude 自己的 /plan"."""
        spec = CommandSpec(name="/plan", handler=lambda c: None, match_kind="prefix")
        for sep in (" ", "\n", "\t", "　", "\r\n", "  ", "\t\n"):
            self.assertTrue(spec.matches(f"/plan{sep}做点事"),
                            f"prefix match must accept separator {sep!r}")
        # Must NOT match when the next char is a non-whitespace continuation —
        # the original ``/planet`` guard is still required.
        self.assertFalse(spec.matches("/planet"))
        self.assertFalse(spec.matches("/plana"))

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

    def test_extract_args_accepts_any_whitespace_separator(self):
        """Regression mirror of ``test_prefix_match_accepts_any_whitespace`` —
        once the matcher accepts ``/plan\\n<task>``, the arg extractor must
        also strip the command token correctly. Without this fix the handler
        would see ``raw_args=""`` (matcher rejected) or the wrong slice."""
        spec = CommandSpec(name="/plan", handler=lambda c: None, match_kind="prefix")
        for sep in (" ", "\n", "\t", "　", "  ", "\n\n", "\t  ", "\r\n"):
            self.assertEqual(
                spec.extract_args(f"/plan{sep}做点事"), "做点事",
                f"extract_args must strip separator {sep!r}",
            )
        # Leading whitespace on the whole message is normalised away too.
        self.assertEqual(spec.extract_args("  /plan\n做点事"), "做点事")

    def test_sub_match_extract_args_accepts_newline(self):
        """The ``sub_matches`` loop in ``extract_args`` also relaxed its
        separator. Verify ``/reset claude\\nextra`` strips down to ``extra``
        — the sub-match must consume the full ``/reset claude`` token even
        when followed by a newline rather than a space."""
        spec = CommandSpec(
            name="/reset",
            handler=lambda c: None,
            match_kind="prefix",
            sub_matches=("/reset claude",),
        )
        self.assertEqual(spec.extract_args("/reset claude\nextra"), "extra")
        self.assertEqual(spec.extract_args("/reset claude\textra"), "extra")

    def test_alias_prefix_match_with_newline(self):
        """Aliases share the ``_names`` loop in both ``matches`` and
        ``extract_args``; without an alias-specific test the relaxed
        separator coverage is incomplete. Pin: ``/m\\nfoo`` routes via
        the ``/m`` alias of ``/model`` exactly like ``/m foo``."""
        spec = CommandSpec(
            name="/model", aliases=("/m",),
            handler=lambda c: None, match_kind="prefix",
        )
        self.assertTrue(spec.matches("/m\nfoo"))
        self.assertEqual(spec.extract_args("/m\nfoo"), "foo")

    def test_prefix_match_rejects_zero_width_space(self):
        """Pin behavior: U+200B (ZWSP) is NOT ``.isspace()``. If someone
        later argues we should accept it as a separator, that needs to be
        an explicit decision — not a silent broadening. Today, ``/plan``
        followed by ZWSP must NOT match (the matcher's separator rule is
        Unicode whitespace, period)."""
        spec = CommandSpec(name="/plan", handler=lambda c: None, match_kind="prefix")
        self.assertFalse(spec.matches("/plan​foo"))
        self.assertFalse(spec.matches("/plan﻿foo"))  # BOM, also not isspace


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

    def test_prefix_dispatch_with_newline_separator(self):
        """End-to-end regression: the registry must route ``/plan\\n<task>``
        to the registered handler (not return ``unhandled``)."""
        captured: dict = {}
        self.reg.register(CommandSpec(
            name="/plan",
            match_kind="prefix",
            handler=lambda c: captured.setdefault("args", c.raw_args),
        ))
        result = self.reg.dispatch(_ctx("/plan\n实现深色模式"))
        self.assertEqual(result, "handled")
        self.assertEqual(captured["args"], "实现深色模式")

    def test_prefix_dispatch_with_fullwidth_space(self):
        captured: dict = {}
        self.reg.register(CommandSpec(
            name="/plan",
            match_kind="prefix",
            handler=lambda c: captured.setdefault("args", c.raw_args),
        ))
        result = self.reg.dispatch(_ctx("/plan　实现深色模式"))
        self.assertEqual(result, "handled")
        self.assertEqual(captured["args"], "实现深色模式")

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

    def test_usage_card_short_circuits_with_trailing_whitespace_only(self):
        """``_ctx`` builds ``tl`` via ``text.lower().strip()``, so ``/run\\n``
        and ``/run   `` reduce to ``tl == "/run"`` and the usage_card branch
        must still fire (not call the handler). This pins the registry's
        ``ctx.tl == spec.name.lower()`` short-circuit against the relaxed
        separator change — a regression here would silently invoke handlers
        with empty args instead of showing the usage hint."""
        called: list[str] = []
        self.reg.register(CommandSpec(
            name="/run",
            match_kind="prefix",
            usage_card="`/run <command>`",
            handler=lambda c: called.append("ran"),
        ))
        for text in ("/run\n", "/run   ", "/run\t", "/run　"):
            with patch("larkhelm.handlers._message.send_card_reply") as send:
                result = self.reg.dispatch(_ctx(text))
            self.assertEqual(result, "handled", f"text={text!r}")
            send.assert_called_once()
        self.assertEqual(called, [],
                         "handler must never fire when usage_card short-circuits")

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
        # /doc was retired in commit 7c9845c (方案B) — no longer in the registry.
        for expected in ("/status", "/help", "/reset", "/run", "/cd", "/ls",
                         "/dev", "/crew", "/plan", "/memory", "/cron",
                         "/model", "/lock", "/voice"):
            self.assertIn(expected, names, f"{expected} missing from default registry")


class CommandSpecMetadataTests(unittest.TestCase):
    """P1-2a: CommandSpec.description / examples (pure metadata, no dispatch impact)."""

    def test_metadata_defaults_empty(self):
        spec = CommandSpec(name="/x", handler=lambda c: None)
        self.assertEqual(spec.description, "")
        self.assertEqual(spec.examples, ())

    def test_default_registrations_all_have_description(self):
        from larkhelm.command_registry import COMMAND_REGISTRY
        missing = [s.name for s in COMMAND_REGISTRY._ordered if not s.description]
        self.assertEqual(missing, [],
                         f"specs missing description: {missing}")

    def test_default_registrations_examples_threshold(self):
        from larkhelm.command_registry import COMMAND_REGISTRY
        n_with_examples = sum(1 for s in COMMAND_REGISTRY._ordered if s.examples)
        self.assertGreaterEqual(
            n_with_examples, 13,
            f"PRD REQ-06 requires >=13 specs with examples; got {n_with_examples}",
        )

    def test_three_commands_have_examples_metadata(self):
        from larkhelm.command_registry import COMMAND_REGISTRY
        for name in ("/dev", "/cron", "/memory"):
            spec = COMMAND_REGISTRY.lookup(name)
            self.assertIsNotNone(spec, f"{name} must be registered")
            self.assertGreaterEqual(
                len(spec.examples), 1,
                f"{name} must carry at least 1 example",
            )

    def test_dispatch_ignores_new_metadata(self):
        """Two specs identical except for description/examples must behave
        byte-identically in matches() / extract_args() / dispatch()."""
        calls_a: list[str] = []
        calls_b: list[str] = []

        def _h_a(ctx: DispatchContext) -> None:
            calls_a.append(ctx.raw_args)

        def _h_b(ctx: DispatchContext) -> None:
            calls_b.append(ctx.raw_args)

        spec_bare = CommandSpec(
            name="/probe", handler=_h_a, match_kind="prefix",
        )
        spec_meta = CommandSpec(
            name="/probe", handler=_h_b, match_kind="prefix",
            description="probe handler — should not affect dispatch",
            examples=("/probe foo", "/probe bar baz"),
        )

        # matches() parity
        for tl in ("/probe", "/probe foo", "/probe foo bar", "/probex", ""):
            self.assertEqual(spec_bare.matches(tl), spec_meta.matches(tl),
                             f"matches({tl!r}) diverged")

        # extract_args() parity
        for text in ("/probe", "/probe foo", "/probe  foo bar", "/PROBE Foo"):
            self.assertEqual(spec_bare.extract_args(text),
                             spec_meta.extract_args(text),
                             f"extract_args({text!r}) diverged")

        # dispatch() parity: two registries, one spec each.
        reg_a = CommandRegistry()
        reg_a.register(spec_bare)
        reg_b = CommandRegistry()
        reg_b.register(spec_meta)
        for text in ("/probe foo", "/probe alpha beta"):
            r_a = reg_a.dispatch(_ctx(text))
            r_b = reg_b.dispatch(_ctx(text))
            self.assertEqual(r_a, r_b, f"dispatch({text!r}) result diverged")
        self.assertEqual(calls_a, calls_b,
                         "handler invocations diverged between bare and meta specs")


if __name__ == "__main__":
    unittest.main()
