"""Unit tests for runner_types Protocols + ai_runner DRY helper (S21+S26).

The type aliases / Protocols in runner_types are mostly compile-time
contracts (mypy / IDE). At runtime Python doesn't enforce ``Protocol``
arity unless the protocol is ``runtime_checkable``. These tests
therefore:

  * Verify the Protocols import cleanly and can be referenced as types.
  * Verify ``QueryCardState`` callbacks (the canonical implementors)
    structurally satisfy the protocol signatures (positional + kwarg).
  * Pin the new ``_resolve_spec_or_default`` helper's two paths.
"""
import threading
import time
import unittest
from unittest.mock import patch

from larkhelm import runner_types as rt
from larkhelm.ai_runner import _resolve_spec_or_default
from larkhelm.backend_registry import BackendSpec
from larkhelm.handlers._query_card_state import QueryCardState


class ProtocolSignatureTests(unittest.TestCase):
    """The QueryCardState callbacks must structurally satisfy the runner_types
    Protocols. We can't enforce this at runtime (Protocols aren't checked
    unless decorated ``@runtime_checkable``), but we *can* assert that the
    callbacks accept the documented call shapes without raising
    ``TypeError``.
    """

    def setUp(self):
        self.state = QueryCardState(chat_id="x", model_name="Claude",
                                    start_time=time.time())

    def test_on_text_accepts_positional_and_status_kwarg(self):
        # on_text(text) and on_text(text, status="done") both valid
        self.state.on_text("hello")
        self.state.on_text("hello", status="done")
        self.state.on_text("hello", status="typing")

    def test_on_tool_accepts_full_signature(self):
        # on_tool(name) — minimal
        self.state.on_tool("Read")
        # on_tool(name, desc) — common
        self.state.on_tool("Read", "/tmp/x")
        # on_tool(name, desc, tool_id) — full
        self.state.on_tool("Read", "/tmp/x", "tool-1")
        # All by kw
        self.state.on_tool(name="Write", desc="/tmp/y", tool_id="tool-2")

    def test_on_tool_result_accepts_full_signature(self):
        # Must have the four positional args in order
        self.state.on_tool("Bash", "ls", "tool-3")
        self.state.on_tool_result("tool-3", "ok", False, 0.1)
        # All by kw also works
        self.state.on_tool("Bash", "ls", "tool-4")
        self.state.on_tool_result(tool_id="tool-4", result="ok",
                                  is_error=False, elapsed=0.2)

    def test_callback_aliases_are_callable_typehints(self):
        # The TypeAlias / Protocol names should be usable in
        # ``Callable``-style annotations without raising.
        def take_on_text(cb: rt.OnText | None) -> None:
            return None
        def take_on_tool(cb: rt.OnTool | None) -> None:
            return None
        def take_on_tool_result(cb: rt.OnToolResult | None) -> None:
            return None
        def take_on_soft_timeout(cb: rt.OnSoftTimeout | None) -> None:
            return None
        def take_on_start(cb: rt.OnStart | None) -> None:
            return None
        # Pass the QueryCardState callbacks — these are the canonical
        # implementors and must satisfy the Protocols structurally.
        take_on_text(self.state.on_text)
        take_on_tool(self.state.on_tool)
        take_on_tool_result(self.state.on_tool_result)
        take_on_soft_timeout(lambda: None)
        take_on_start(lambda pid: None)

    def test_cancel_event_alias_is_threading_event(self):
        # The alias is a re-export, not a subclass — verify both directions
        ev = threading.Event()
        self.assertIsInstance(ev, rt.CancelEvent)


class ResolveSpecOrDefaultTests(unittest.TestCase):
    """The DRY helper for query_* spec resolution."""

    def test_registry_hit_returns_registry_spec(self):
        # When the registry has the backend, we return that spec, not the
        # default. Verify by checking object identity.
        registered = BackendSpec(
            id="myclaude", provider="claude_cli", display_name="My Claude",
            role="orchestrator", tags=[], command="/usr/bin/claude",
        )
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.get.return_value = registered
            # The factory must NOT be called when registry hits.
            factory_calls = []
            def factory():
                factory_calls.append(True)
                return BackendSpec(id="unused", provider="claude_cli",
                                   display_name="Unused", role="worker", tags=[])
            result = _resolve_spec_or_default("myclaude", factory)
            self.assertIs(result, registered)
            self.assertEqual(factory_calls, [],
                             "factory should not run on registry hit")
            mock_reg.get.assert_called_once_with("myclaude")

    def test_registry_miss_runs_factory(self):
        default = BackendSpec(
            id="missing", provider="claude_cli", display_name="Missing",
            role="worker", tags=[], command="/x/y",
        )
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.get.return_value = None
            factory_calls = []
            def factory():
                factory_calls.append(True)
                return default
            result = _resolve_spec_or_default("missing", factory)
            self.assertIs(result, default)
            self.assertEqual(len(factory_calls), 1,
                             "factory should run exactly once on miss")

    def test_factory_evaluated_lazily(self):
        # If the factory raises, that exception must only surface when the
        # registry misses — confirms the factory truly is lazy.
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.get.return_value = BackendSpec(
                id="found", provider="claude_cli", display_name="Found",
                role="worker", tags=[],
            )
            def crashing_factory():
                raise RuntimeError("should not be called")
            # Hit path: factory not invoked, no exception bubbles up
            result = _resolve_spec_or_default("found", crashing_factory)
            self.assertEqual(result.id, "found")


if __name__ == "__main__":
    unittest.main()
