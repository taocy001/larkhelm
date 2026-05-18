"""larkhelm · memory_circuit — shared failure-governance primitives.

Provides two thread-safe, dependency-free building blocks used by P3
REQ-04 (memory_llm_router circuit) and REQ-05 (cascade extract backoff):

* :class:`CircuitBreaker` — closed / open / half_open state machine.
* :class:`ExponentialBackoff` — bounded retry loop with capped delay.

Both classes are pure algorithm: they know nothing about LLMs, HTTP,
or larkhelm config. Callers wire them up.

Design notes
------------
* All state mutations sit behind a single ``threading.Lock`` per breaker
  instance; reads of :pymeth:`CircuitBreaker.current_state` go through
  the same lock so observers never see torn writes.
* ``ExponentialBackoff.run`` injects ``time.sleep`` lazily so test code
  can monkey-patch the module-level ``time`` reference for fake-clock
  tests (see ``tests/test_cascade_backoff.py``).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Configuration dataclasses ─────────────────────────────────────────────


@dataclass
class CircuitConfig:
    failure_threshold: int = 5
    cool_down_sec: float = 30.0
    window_sec: float = 60.0
    half_open_max_calls: int = 1


@dataclass
class BackoffConfig:
    initial_sec: float = 1.0
    multiplier: float = 2.0
    max_sec: float = 30.0
    max_attempts: int = 3


@dataclass
class CircuitState:
    state: str = "closed"           # "closed" | "open" | "half_open"
    consecutive_failures: int = 0
    opened_at: float = 0.0
    last_attempt_at: float = 0.0
    last_failure_at: float = 0.0
    half_open_in_flight: int = field(default=0)


# ── CircuitBreaker ────────────────────────────────────────────────────────


class CircuitBreaker:
    """Simple closed/open/half_open breaker.

    State transitions::

        closed   --(N consecutive failures within window)--> open
        open     --(cool_down elapsed)-------------------->  half_open
        half_open --(success)-->                            closed
        half_open --(failure)-->                            open
    """

    def __init__(self, config: CircuitConfig) -> None:
        self.config = config
        self.state = CircuitState()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Return True iff a call should be attempted right now."""
        now = time.monotonic()
        with self._lock:
            self.state.last_attempt_at = now
            if self.state.state == "closed":
                return True
            if self.state.state == "open":
                if now - self.state.opened_at >= self.config.cool_down_sec:
                    # Transition open -> half_open and admit this probe.
                    self.state.state = "half_open"
                    self.state.half_open_in_flight = 1
                    return True
                return False
            # half_open
            if self.state.half_open_in_flight < self.config.half_open_max_calls:
                self.state.half_open_in_flight += 1
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.state.consecutive_failures = 0
            self.state.state = "closed"
            self.state.opened_at = 0.0
            self.state.half_open_in_flight = 0

    def record_failure(self) -> None:
        now = time.monotonic()
        with self._lock:
            # Reset counter when previous failure is outside the rolling window.
            # Use ``last_failure_at`` (not ``last_attempt_at``) so successful
            # calls that refresh ``last_attempt_at`` don't keep the window
            # sliding forward and prevent the reset branch from firing.
            if (
                self.state.last_failure_at > 0
                and self.state.consecutive_failures > 0
                and (now - self.state.last_failure_at) > self.config.window_sec
            ):
                self.state.consecutive_failures = 0
            self.state.consecutive_failures += 1
            self.state.last_failure_at = now
            if self.state.state == "half_open":
                self.state.state = "open"
                self.state.opened_at = now
                self.state.half_open_in_flight = 0
                return
            if self.state.consecutive_failures >= self.config.failure_threshold:
                self.state.state = "open"
                self.state.opened_at = now
                self.state.half_open_in_flight = 0

    def current_state(self) -> str:
        with self._lock:
            return self.state.state


# ── ExponentialBackoff ────────────────────────────────────────────────────


class ExponentialBackoff:
    """Capped exponential-delay retry loop.

    ``run(fn)`` invokes ``fn()`` up to ``max_attempts`` times. With
    ``max_attempts=N`` there are at most ``N-1`` sleeps; the i-th sleep
    (1-indexed) is ``min(initial * multiplier**(i-1), max_sec)``. The
    default ``max_attempts=3`` therefore yields two sleeps ``[1.0, 2.0]``;
    set ``cascade_backoff_max_attempts=4`` to extend the sequence to
    ``[1.0, 2.0, 4.0]``. If every attempt raises, the last exception
    propagates.

    Tests can monkey-patch ``larkhelm.memory_circuit.time.sleep`` to
    avoid real sleeps; ``run`` calls ``time.sleep`` through the module
    reference for exactly that reason.
    """

    def __init__(self, config: BackoffConfig) -> None:
        self.config = config

    def _delay_for(self, attempt: int) -> float:
        """Return the delay before attempt ``attempt+1`` (0-indexed)."""
        if attempt <= 0:
            return 0.0
        raw = self.config.initial_sec * (self.config.multiplier ** (attempt - 1))
        return min(raw, self.config.max_sec)

    def run(
        self,
        fn: Callable[[], Any],
        on_error: Optional[Callable[[Exception, int], None]] = None,
    ) -> Any:
        last_exc: Optional[BaseException] = None
        attempts = max(int(self.config.max_attempts), 1)
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if on_error is not None:
                    try:
                        on_error(e, attempt)
                    except Exception:
                        pass
                if attempt >= attempts:
                    raise
                time.sleep(self._delay_for(attempt))
        # Unreachable, but mypy needs a return.
        if last_exc is not None:
            raise last_exc
        return None


__all__ = [
    "BackoffConfig",
    "CircuitBreaker",
    "CircuitConfig",
    "CircuitState",
    "ExponentialBackoff",
]
