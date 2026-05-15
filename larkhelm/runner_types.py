"""larkhelm · runner callback type aliases (S21+S26).

The four spawn / query wrappers in ``ai_runner.py`` and the
``BaseProcessRunner.__init__`` accept a small set of optional callbacks
that are passed unchanged down to the subprocess streaming loop. Before
this module the parameters were typed as plain ``None`` (or implicitly
``object``), so mypy / IDE tooling could not warn when a caller passed
a callback with the wrong arity or accepted the wrong kwarg.

These TypeAliases pin the shapes; every runner module imports from here
rather than re-deriving the signatures.

Naming: the originals are ``on_text`` / ``on_tool`` / ``on_tool_result`` /
``on_soft_timeout`` / ``on_start``. Aliases use ``OnText`` / ``OnTool`` /
``OnToolResult`` / ``OnSoftTimeout`` / ``OnStart`` for PEP-8 CapWords.

The aliases all accept ``None`` because every callback in the runner
constructors defaults to ``None`` and is skipped at the call site when
not provided. Callers therefore see ``OnText | None`` etc.
"""
from __future__ import annotations

import threading
from typing import Callable, Protocol, TypeAlias


# ─────────────────────────────────────────────────────────────────────
# Cancel event — same shape everywhere.
# ─────────────────────────────────────────────────────────────────────

CancelEvent: TypeAlias = threading.Event


# ─────────────────────────────────────────────────────────────────────
# Streaming text callback
# ─────────────────────────────────────────────────────────────────────
#
# Backends invoke ``on_text(text, status="typing")`` (or ``status="done"``
# on the final frame). The status kwarg is optional — older shims
# (notably ``QueryCardState.on_text``) accept and discard it.
# ─────────────────────────────────────────────────────────────────────

class OnText(Protocol):
    """Streaming text callback signature.

    Implementations must accept the positional ``text`` and tolerate a
    ``status`` keyword used as a free-form sentinel distinguishing the
    runner's current phase. Sentinels observed across the runners:

      * ``"typing"`` — mid-stream incremental text (default)
      * ``"done"``   — final frame
      * ``"thinking"`` — DeepSeek extended-thinking phase
      * ``"init"``   — initial connection / hello frame
      * ``"error"``  — Claude CLI's error-text frame

    Implementations should treat unknown sentinels as ``"typing"``.
    """
    def __call__(self, text: str, status: str = "typing") -> None: ...


# ─────────────────────────────────────────────────────────────────────
# Tool callbacks
# ─────────────────────────────────────────────────────────────────────
#
# ``on_tool`` fires when the model starts executing a tool.
# ``on_tool_result`` fires when the tool produces a result.
# Both are required for the streaming-card UI to track in-flight tools.
# ─────────────────────────────────────────────────────────────────────

class OnTool(Protocol):
    """Tool-invocation start callback.

    Args:
      name:    Display name of the tool (e.g. ``"Read"`` / ``"Bash"``).
      desc:    Human-readable description (often the tool's primary arg).
      tool_id: Stable id assigned by the backend for matching with
               the eventual ``on_tool_result`` call. Empty string is
               legal for backends that don't expose ids.
    """
    def __call__(self, name: str, desc: str = "", tool_id: str = "") -> None: ...


class OnToolResult(Protocol):
    """Tool result callback.

    Args:
      tool_id:  Same id passed to ``on_tool``.
      result:   Stringified tool output (may be long).
      is_error: True iff the tool reported an error.
      elapsed:  Wall-clock seconds the tool ran.
    """
    def __call__(self, tool_id: str, result: str, is_error: bool, elapsed: float) -> None: ...


# ─────────────────────────────────────────────────────────────────────
# Lifecycle callbacks
# ─────────────────────────────────────────────────────────────────────

OnSoftTimeout: TypeAlias = Callable[[], None]
"""Fires when the soft-timeout (no output for ``response_timeout``
seconds) elapses but the subprocess is still alive. The query thread
releases the chat lock and promotes the task to "background" so the
heartbeat card keeps updating while the user can start a new query."""

OnStart: TypeAlias = Callable[[], None]
"""Fires once right after the runner acquires its semaphore slot and
spawns the subprocess (or, for DeepSeek, opens the HTTP request).

Takes **no arguments**. The original v1 of this alias declared
``Callable[[int], None]`` claiming it received the subprocess pid, but
every call site in runner_base / runner_deepseek / crew/_runner invokes
it with zero args — so the v1 type signature was a lie that would have
broken any caller defining ``def cb(pid): ...`` trusting the type.

Crew uses it to flip the agent's status to RUNNING and start its
timeout countdown only after the slot is held."""
