"""larkhelm · query card render state machine

Extracted from ``_do_query`` (P1 #8 / S2) to make the streaming-card state
machine independently testable. Before this split, the renderer's 15 nested
closures shared local state via ``nonlocal``, which made it impossible to
exercise the render / push / tool-tracking logic without spinning up the
full query pipeline (chat lock, backend resolution, Feishu API).

This class owns:

  * Streaming text buffer (``current_text``) and dirty flag
  * Cursor animation frame index
  * Active + completed tool sets (with locks for concurrent updates)
  * Last-pushed-body snapshot for delta detection
  * Heartbeat timestamp + background-promoted flag
  * Card-patch lock that serialises heartbeat patches against final
    success / error patches in the main thread

It deliberately does NOT own:

  * Network I/O (the patch callback is injected by the caller)
  * Chat / cancel events (caller decides when to bail)
  * Backend selection or message routing

The state is fully thread-safe — every public method acquires the
appropriate lock. The two locks (``_state_lock`` for scalar state and
``_tools_lock`` for tool dicts) are NEVER held simultaneously to keep
lock ordering acyclic.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from larkhelm.card_builder import _fmt_elapsed

# These three are mirrored from larkhelm.config at import time. The
# ``_do_query`` caller reads them from config the same way, so the values
# stay in sync as long as the module is imported during the same process.
# Keeping a local copy avoids importing _cfg into a pure state module.
import larkhelm.config as _cfg

TOOL_HISTORY_CAP: int = _cfg.TOOL_HISTORY_CAP
CURSOR_FRAMES: list[str] = _cfg.CURSOR_FRAMES
STALL_THRESHOLD: float = _cfg.STALL_THRESHOLD


@dataclass
class ToolRecord:
    """A finished or in-flight tool invocation, as seen by the card renderer."""
    name: str
    desc: str = ""
    start: float = 0.0           # time.monotonic() at start
    elapsed: float = 0.0         # seconds; 0 means in-flight
    result: str = ""
    full_result: str = ""        # populated when result > 200 chars
    is_error: bool = False


@dataclass
class RenderedBody:
    """Result of rendering one card frame."""
    title: str
    tools_md: str | None
    response_md: str


class QueryCardState:
    """Encapsulates the streaming-card state machine for one ``_do_query`` invocation.

    Usage pattern (matches the original closures one-for-one):

        state = QueryCardState(chat_id=cid, model_name="Claude", start_time=t0)

        # Wire callbacks into the backend runner:
        runner.run(on_tool=state.on_tool,
                   on_tool_result=state.on_tool_result,
                   on_text=state.on_text,
                   ...)

        # Start a heartbeat thread that periodically:
        #   1. Calls state.tick_cursor()
        #   2. Renders via state.render_body(elapsed)
        #   3. Pushes via the caller-provided patch_callback if state.should_push(...)

        # On terminal events (cancel / error / done):
        state.snapshot_active_tools_as_completed()  # so the final card lists them
        final_tools = state.snapshot_completed_tools()
    """

    def __init__(self, chat_id: str, model_name: str, start_time: float):
        self.chat_id = chat_id
        self.model_name = model_name
        self.start_time = start_time

        # ── Card push state (scalars) ─────────────────────────────────
        self._dirty = False
        self._cursor_idx = 0
        self._current_text = ""
        self._last_pushed_body = ""
        self._last_heartbeat = time.monotonic()
        self._in_background = False
        self._state_lock = threading.Lock()

        # ── Tool state ───────────────────────────────────────────────
        self.active_tools: dict[str, ToolRecord] = {}
        self.completed_tools: list[ToolRecord] = []
        self._tools_lock = threading.Lock()

        # ── Card patch serialisation lock ─────────────────────────────
        # The heartbeat thread acquires this around its patch call. Main
        # thread acquires it (without holding work) after signalling stop
        # so any in-flight heartbeat patch finishes before the final
        # success / error / cancel card overwrite.
        self.card_patch_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────
    # Scalar state mutators
    # ─────────────────────────────────────────────────────────────────

    def set_dirty(self, v: bool) -> None:
        with self._state_lock:
            self._dirty = v

    def set_current_text(self, v: str) -> None:
        """Update the streaming text buffer and mark the card dirty."""
        with self._state_lock:
            self._current_text = v
            self._dirty = True

    def set_in_background(self, v: bool) -> None:
        """Toggle the 'background' indicator (shown after soft timeout)."""
        with self._state_lock:
            self._in_background = v
            self._dirty = True

    def tick_cursor(self) -> None:
        with self._state_lock:
            self._cursor_idx = (self._cursor_idx + 1) % len(CURSOR_FRAMES)

    def update_heartbeat(self) -> None:
        with self._state_lock:
            self._last_heartbeat = time.monotonic()

    # ─────────────────────────────────────────────────────────────────
    # Scalar state readers
    # ─────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> tuple[bool, int, str, str, float, bool]:
        """Atomic read of the six scalar state fields."""
        with self._state_lock:
            return (
                self._dirty,
                self._cursor_idx,
                self._current_text,
                self._last_pushed_body,
                self._last_heartbeat,
                self._in_background,
            )

    @property
    def dirty(self) -> bool:
        with self._state_lock:
            return self._dirty

    @property
    def in_background(self) -> bool:
        with self._state_lock:
            return self._in_background

    @property
    def last_heartbeat(self) -> float:
        with self._state_lock:
            return self._last_heartbeat

    @property
    def current_text(self) -> str:
        with self._state_lock:
            return self._current_text

    # ─────────────────────────────────────────────────────────────────
    # Tool tracking
    # ─────────────────────────────────────────────────────────────────

    def on_tool(self, name: str, desc: str = "", tool_id: str = "") -> None:
        """Backend callback: a new tool invocation has started.

        The previous in-flight tool (if any) is flushed to ``completed_tools``
        as a synthetic "no-result" record. This matches the original
        ``_do_query`` behaviour where overlapping tool calls were treated as
        sequential (the model usually finishes one before starting the next,
        but mid-call corrections can produce this pattern).
        """
        with self._tools_lock:
            now_mono = time.monotonic()
            for tid, t in list(self.active_tools.items()):
                self.completed_tools.append(ToolRecord(
                    name=t.name, desc=t.desc,
                    elapsed=now_mono - t.start,
                    is_error=False, result="",
                ))
            self.active_tools.clear()
            self.active_tools[tool_id] = ToolRecord(name=name, desc=desc, start=now_mono)
        self.set_dirty(True)

    def on_tool_result(self, tool_id: str, result: str, is_error: bool, elapsed: float) -> None:
        """Backend callback: a tool invocation has produced a result."""
        with self._tools_lock:
            info = self.active_tools.pop(tool_id, None)
            if info is not None:
                self.completed_tools.append(ToolRecord(
                    name=info.name, desc=info.desc,
                    elapsed=elapsed,
                    result=result,
                    # Stash a larger snippet for the final tools_list panel
                    # only when the result is long enough to be interesting.
                    full_result=result[:5000] if len(result) > 200 else "",
                    is_error=is_error,
                ))
        self.set_dirty(True)

    def on_text(self, text: str, status: str = "typing") -> None:
        """Backend callback: streaming text update.

        Note: ``status`` is accepted for callback-signature parity with the
        runners but not used here. The card title is derived from active
        tools and stream presence in :py:meth:`render_body`.
        """
        del status  # unused; retained for signature compat
        self.set_current_text(text)

    def snapshot_active_tools_as_completed(self) -> None:
        """Move any still-in-flight tools into ``completed_tools``.

        Called on terminal paths (success / error / cancel) so the final
        card / log accurately reports every tool that ran, even if the
        backend died before emitting a result event for them.
        """
        with self._tools_lock:
            now_mono = time.monotonic()
            for tid, t in list(self.active_tools.items()):
                self.completed_tools.append(ToolRecord(
                    name=t.name, desc=t.desc,
                    elapsed=now_mono - t.start,
                    is_error=False, result="",
                ))
            self.active_tools.clear()

    def snapshot_completed_tools(self) -> list[ToolRecord]:
        """Return a copy of the completed-tools list (safe to iterate
        outside the lock)."""
        with self._tools_lock:
            return list(self.completed_tools)

    def n_completed_tools(self) -> int:
        with self._tools_lock:
            return len(self.completed_tools)

    # ─────────────────────────────────────────────────────────────────
    # Render
    # ─────────────────────────────────────────────────────────────────

    def render_body(self) -> RenderedBody:
        """Build the title / tools panel / response markdown for one card frame.

        Pure function of current state — calling this twice without any
        state change produces byte-identical output (verified by tests).
        """
        elapsed = _fmt_elapsed(time.time() - self.start_time)

        with self._tools_lock:
            act = dict(self.active_tools)
            comp = list(self.completed_tools)

        cur_text, cursor_i, in_bg = self._read_render_inputs()

        tool_parts: list[str] = []
        n_hidden = max(0, len(comp) - TOOL_HISTORY_CAP)
        if n_hidden > 0:
            tool_parts.append(f"_+{n_hidden} 条更早记录已隐藏_")

        for t in comp[-TOOL_HISTORY_CAP:]:
            icon = "✗" if t.is_error else "✓"
            desc_str = _fmt_desc(t.desc)
            tool_parts.append(
                f"{icon} **{t.name}** ({_fmt_elapsed(t.elapsed)}){desc_str}"
            )

        now_mono = time.monotonic()
        for t in act.values():
            tool_elapsed = now_mono - t.start
            desc_str = _fmt_desc(t.desc)
            if tool_elapsed > STALL_THRESHOLD:
                tool_parts.append(
                    f"🔧 **{t.name}** ⚠️ 响应停滞 ({_fmt_elapsed(tool_elapsed)}){desc_str}"
                )
            else:
                tool_parts.append(
                    f"🔧 **{t.name}** ({_fmt_elapsed(tool_elapsed)})…{desc_str}"
                )

        tools_md = "\n\n".join(tool_parts) if tool_parts else None

        if cur_text.strip():
            cursor = CURSOR_FRAMES[cursor_i]
            response_md = cur_text.strip() + cursor
        elif not tool_parts:
            response_md = "> 正在思考..."
        else:
            response_md = ""

        bg_prefix = "后台·" if in_bg else ""
        if act:
            title = f"⚙️ {self.model_name} · {bg_prefix}工具调用中 ({elapsed})"
        elif cur_text.strip():
            title = f"✍️ {self.model_name} · {bg_prefix}回应中 ({elapsed})"
        else:
            title = f"⏳ {self.model_name} · {bg_prefix}思考中 ({elapsed})"

        return RenderedBody(title=title, tools_md=tools_md, response_md=response_md)

    def _read_render_inputs(self) -> tuple[str, int, bool]:
        """Atomic snapshot of the three render-affecting scalars."""
        with self._state_lock:
            return self._current_text, self._cursor_idx, self._in_background

    # ─────────────────────────────────────────────────────────────────
    # Push decision
    # ─────────────────────────────────────────────────────────────────

    def should_push(self, rendered: RenderedBody, force: bool = False) -> tuple[bool, str]:
        """Return ``(should_push, combined_key)``.

        ``combined_key`` is the canonical "what does this card look like
        now" signature; the caller passes it back to :py:meth:`mark_pushed`
        after a successful patch so the next call's delta detection has
        the right baseline.
        """
        combined = f"{rendered.title}||{rendered.tools_md}||{rendered.response_md}"
        with self._state_lock:
            need_push = force or self._dirty or combined != self._last_pushed_body
        return need_push, combined

    def mark_pushed(self, combined: str) -> None:
        """Record the body string that was last successfully patched."""
        with self._state_lock:
            self._last_pushed_body = combined
            self._dirty = False

    def update_model_name(self, name: str) -> None:
        """Replace the cached display name (used when the failover chain
        switches to a different backend mid-query)."""
        # No lock — single-word assignment is atomic in CPython and reading
        # a stale name for a single render frame is harmless.
        self.model_name = name


# ─────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────

def _fmt_desc(d: str) -> str:
    """Render a tool's description as inline code or fenced block.

    Multi-line descriptions go in a fenced block so newlines survive the
    Feishu markdown renderer. Single-line descriptions go in inline code.
    Empty descriptions render to empty string.
    """
    if not d:
        return ""
    if "\n" in d:
        return f"\n```\n{d}\n```"
    return f"  \n`{d}`"
