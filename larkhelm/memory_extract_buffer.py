"""larkhelm · session→cascade extract buffer (P2 REQ-06 / AC-06).

Coalesces a burst of session-summary updates into a single cascade extract
call. The 60-second timer (configurable via
``memory_extract_buffer_window_sec``) trades cascade-call frequency for
freshness: during a rapid /dev sequence with five back-to-back milestones
we want one cheap-LLM extract at the end, not five.

Default behaviour: ``memory_extract_buffer_window_sec=0`` means "no
buffering" — every ``record_session_update`` call immediately fires the
underlying ``_cascade_extract``. This is the P1 byte-compat path and the
default install setting; tests must enforce it via
``test_buffer_disabled_byte_compatible``.

Threading model: single process-wide ``ExtractBuffer`` singleton owns
``{chat_id: BufferState}`` under a single ``threading.Lock``. The
threading.Timer fires on the timer thread; the flush path re-enters the
lock briefly to detach the state then calls ``_cascade_extract`` outside
the lock so the cheap-LLM call doesn't serialise across chats.

SIGTERM handling: ``bridge._handle_sigterm`` invokes
``flush_all_for_shutdown()`` so any pending updates are written before the
process exits.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import larkhelm.config as _cfg
from larkhelm.log import _debug_log


# Capacity floor: even with window>0 we want to flush when the pending
# content grows past this many chars, so a chatty user doesn't see their
# summaries withheld indefinitely.
DEFAULT_CAPACITY_FLOOR_CHARS = 8000


@dataclass
class BufferState:
    """Per-chat buffer slot.

    ``pending_content`` is *overwrite-style*: each new session update
    replaces the previous text because memory.maybe_auto_update already
    emits a self-contained summary (not a delta). update_count tracks how
    many summaries were absorbed since the last flush — useful for the
    "trigger=capacity" path and for debug logs.
    """
    chat_id: str
    pending_content: str = ""
    first_arrival: float = 0.0
    last_arrival: float = 0.0
    update_count: int = 0
    timer: threading.Timer | None = field(default=None, repr=False)


class ExtractBuffer:
    """Process-wide buffer singleton."""

    def __init__(self) -> None:
        self._states: dict[str, BufferState] = {}
        self._lock = threading.Lock()
        # Injection point for tests: the production ``_cascade_extract``
        # lives in memory.py (lazy-imported at flush time). Setting this
        # attribute lets a unit test count calls without touching the real
        # cascade machinery.
        self._cascade_fn: Callable[[str, str], None] | None = None

    # ── public API ─────────────────────────────────────────────────────

    def record_session_update(self, chat_id: str, content: str) -> None:
        """Note a fresh session summary for ``chat_id``.

        If buffering is disabled (window==0), invokes the cascade
        synchronously so P1 callers see zero behavioural drift. Otherwise
        absorbs the update into the per-chat slot, (re)arming the timer
        on first arrival; subsequent arrivals within the window update
        ``pending_content`` and bump ``update_count`` without restarting
        the timer (the first arrival anchors the deadline).
        """
        if not chat_id:
            return
        window = self._window_sec()

        if window <= 0:
            # Byte-compat path: invoke cascade immediately, do NOT touch
            # _states. Tests pin this via test_buffer_disabled_byte_compatible.
            self._invoke_cascade(chat_id, content, trigger="immediate")
            return

        now = time.monotonic()
        with self._lock:
            state = self._states.get(chat_id)
            if state is None:
                state = BufferState(chat_id=chat_id)
                self._states[chat_id] = state
            state.pending_content = content or ""
            state.last_arrival = now
            if state.first_arrival == 0.0:
                state.first_arrival = now
            state.update_count += 1

            # Capacity flush: if the absorbed content already blew past
            # the floor, don't wait for the timer.
            if len(state.pending_content) >= DEFAULT_CAPACITY_FLOOR_CHARS:
                self._cancel_timer_locked(state)
                pending = state.pending_content
                self._reset_state_locked(chat_id)
            else:
                # First arrival → arm timer. Subsequent arrivals reuse
                # the existing timer — the deadline stays anchored to
                # first_arrival.
                if state.timer is None:
                    self._arm_timer_locked(state, window)
                return

        # capacity-flush path (lock released)
        self._invoke_cascade(chat_id, pending, trigger="capacity")

    def flush(self, chat_id: str, trigger: str = "manual") -> None:
        """Flush any pending update for ``chat_id`` immediately.

        No-op when the chat has no pending state. ``trigger`` ∈
        {timer, capacity, manual, shutdown} and ends up in the
        ``larkhelm_extract_buffer_flushes_total{trigger}`` metric.
        """
        with self._lock:
            state = self._states.get(chat_id)
            if state is None or not state.pending_content:
                return
            self._cancel_timer_locked(state)
            pending = state.pending_content
            self._reset_state_locked(chat_id)
        self._invoke_cascade(chat_id, pending, trigger=trigger)

    def flush_all_for_shutdown(self, timeout_sec: float = 10.0) -> None:
        """Flush every pending slot. Called from the SIGTERM handler.

        ``timeout_sec`` bounds the total time spent: once it elapses we
        stop flushing and the remaining slots are dropped — operator
        accepts the loss to keep shutdown bounded.
        """
        deadline = time.monotonic() + timeout_sec
        with self._lock:
            chat_ids = list(self._states.keys())
        for chat_id in chat_ids:
            if time.monotonic() >= deadline:
                _debug_log(
                    f"[ExtractBuffer] shutdown flush deadline hit; "
                    f"{len(chat_ids)} remaining slots dropped"
                )
                break
            try:
                self.flush(chat_id, trigger="shutdown")
            except Exception as e:
                _debug_log(f"[ExtractBuffer] shutdown flush failed for {chat_id[:8]}: {e}")

    # ── internals ──────────────────────────────────────────────────────

    def _window_sec(self) -> int:
        """Read the current operator-configured window. Cached at the call
        site by ``record_session_update``; deliberately not memoised so an
        operator can flip the flag at runtime without restarting the bridge.
        """
        try:
            return max(0, int(getattr(_cfg, "MEMORY_EXTRACT_BUFFER_WINDOW_SEC", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _arm_timer_locked(self, state: BufferState, window: int) -> None:
        """Caller MUST hold ``self._lock``."""
        def _on_timer():
            try:
                self.flush(state.chat_id, trigger="timer")
            except Exception as e:
                _debug_log(f"[ExtractBuffer] timer flush failed for {state.chat_id[:8]}: {e}")
        state.timer = threading.Timer(window, _on_timer)
        state.timer.daemon = True
        state.timer.name = f"extract-buf-{state.chat_id[:8]}"
        state.timer.start()

    def _cancel_timer_locked(self, state: BufferState) -> None:
        """Caller MUST hold ``self._lock``."""
        t = state.timer
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
            state.timer = None

    def _reset_state_locked(self, chat_id: str) -> None:
        """Caller MUST hold ``self._lock``. Drops the slot entirely so the
        next ``record_session_update`` re-arms cleanly.
        """
        self._states.pop(chat_id, None)

    def _invoke_cascade(self, chat_id: str, content: str, trigger: str) -> None:
        """Run the underlying cascade + bump the trigger counter.

        Caller MUST NOT hold ``self._lock`` (the cascade can take seconds
        and we don't want to block ``record_session_update`` from other chats).
        """
        # Bump the buffer metric BEFORE the cascade call so a failure
        # mid-cascade is still visible in the flushes counter.
        if trigger != "immediate":
            try:
                from larkhelm import metrics as _met
                _met.inc_extract_buffer_flush(trigger)
            except Exception:
                pass

        fn = self._cascade_fn
        if fn is None:
            try:
                from larkhelm.memory import _cascade_extract as _live_cascade
                fn = _live_cascade
            except Exception as e:
                _debug_log(f"[ExtractBuffer] cannot import _cascade_extract: {e}")
                return
        self._flush_with_backoff(fn, content, chat_id)

    def _flush_with_backoff(
        self,
        fn: Callable[[str, str], None],
        content: str,
        chat_id: str,
    ) -> None:
        """REQ-05: wrap the cascade call in ExponentialBackoff.

        ``cascade_backoff_max_attempts`` controls the max retries (default
        3). Sleep sequence depends on the configured attempts: default 3
        gives ``[1.0, 2.0]``; set ``cascade_backoff_max_attempts=4`` to
        extend to ``[1.0, 2.0, 4.0]``. Each delay is capped at 30s.
        Last-failure bubbles up to a debug log; the existing cascade
        pipeline already counts ``cascade_extract_total{outcome="error"}``
        so we don't double-count.
        """
        try:
            from larkhelm.memory_circuit import BackoffConfig, ExponentialBackoff
            # ``_cfg`` is already imported at module top (line 32); the
            # earlier local re-import was redundant — flagged as STYLE-1
            # in P3 review.
            attempts = int(getattr(_cfg, "CASCADE_BACKOFF_MAX_ATTEMPTS", 3) or 3)
        except Exception:
            # If memory_circuit can't be imported (very early bootstrap),
            # fall back to the plain single-shot call.
            try:
                fn(content, chat_id)
            except Exception as e:
                _debug_log(f"[CascadeBackoff] cascade invocation failed for {chat_id[:8]}: {e}")
            return

        backoff = ExponentialBackoff(BackoffConfig(max_attempts=max(1, attempts)))
        try:
            backoff.run(lambda: fn(content, chat_id))
        except Exception as e:
            _debug_log(
                f"[CascadeBackoff] cascade gave up after {attempts} attempts "
                f"for {chat_id[:8]}: {e}"
            )
            # P1-5a (W14): expose backoff exhaustion so Grafana can see how
            # often cascade extract finally fails — the _debug_log line alone
            # was invisible to alerting.
            try:
                from larkhelm.metrics import inc_cascade_backoff_exhausted
                inc_cascade_backoff_exhausted()
            except Exception:
                pass

    # ── test hook ──────────────────────────────────────────────────────

    def set_cascade_fn_for_tests(self, fn: Callable[[str, str], None] | None) -> None:
        """Inject a fake cascade callable; pass ``None`` to restore live."""
        self._cascade_fn = fn


# ── Module-level singleton + façade ────────────────────────────────────

_buffer_singleton: ExtractBuffer | None = None
_buffer_singleton_lock = threading.Lock()


def _get_buffer() -> ExtractBuffer:
    global _buffer_singleton
    if _buffer_singleton is None:
        with _buffer_singleton_lock:
            if _buffer_singleton is None:
                _buffer_singleton = ExtractBuffer()
    return _buffer_singleton


def record_session_update(chat_id: str, content: str) -> None:
    """Public entry point — see :meth:`ExtractBuffer.record_session_update`."""
    _get_buffer().record_session_update(chat_id, content)


def flush(chat_id: str, trigger: str = "manual") -> None:
    _get_buffer().flush(chat_id, trigger=trigger)


def flush_all_for_shutdown(timeout_sec: float = 10.0) -> None:
    _get_buffer().flush_all_for_shutdown(timeout_sec=timeout_sec)


def _get_buffer_window_sec() -> int:
    return _get_buffer()._window_sec()


def _is_buffer_disabled() -> bool:
    return _get_buffer_window_sec() <= 0


def _reset_for_tests() -> None:
    """Test-only: drop the singleton + cancel any pending timer.

    Used by ``tests/test_memory_extract_buffer.py`` between cases so timer
    state from one test doesn't leak into the next. Production code MUST
    NOT call this.
    """
    global _buffer_singleton
    with _buffer_singleton_lock:
        if _buffer_singleton is not None:
            with _buffer_singleton._lock:
                for state in list(_buffer_singleton._states.values()):
                    _buffer_singleton._cancel_timer_locked(state)
                _buffer_singleton._states.clear()
        _buffer_singleton = None


__all__ = [
    "BufferState",
    "ExtractBuffer",
    "record_session_update",
    "flush",
    "flush_all_for_shutdown",
    "_get_buffer_window_sec",
    "_is_buffer_disabled",
    "_get_buffer",
]
