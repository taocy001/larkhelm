"""AC-04 — P3 REQ-04 LLM router circuit breaker.

* 5 consecutive failures open the circuit; the 6th ``allow()`` returns
  False without touching the backend.
* After ``cool_down_sec`` elapses, exactly one half-open probe is
  permitted; a successful probe closes the circuit.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from larkhelm.memory_circuit import CircuitBreaker, CircuitConfig


class TestCircuitBreaker(unittest.TestCase):

    def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(CircuitConfig(failure_threshold=5, cool_down_sec=30.0))
        for _ in range(5):
            self.assertTrue(cb.allow())
            cb.record_failure()
        # 6th call: circuit is open, no backend touch.
        self.assertFalse(cb.allow())
        self.assertEqual(cb.current_state(), "open")

    def test_half_open_then_close_on_success(self) -> None:
        cb = CircuitBreaker(CircuitConfig(failure_threshold=5, cool_down_sec=30.0))
        for _ in range(5):
            cb.allow()
            cb.record_failure()
        self.assertFalse(cb.allow())  # open

        # Fake-clock the cool-down by patching time.monotonic.
        future = 1_000_000.0
        with patch("larkhelm.memory_circuit.time.monotonic", return_value=future):
            self.assertTrue(cb.allow())     # admitted as half-open probe
            self.assertEqual(cb.current_state(), "half_open")
            # Second simultaneous probe is denied (half_open_max_calls=1).
            self.assertFalse(cb.allow())
            cb.record_success()
            self.assertEqual(cb.current_state(), "closed")
            # And fresh calls flow through normally.
            self.assertTrue(cb.allow())

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(CircuitConfig(failure_threshold=5, cool_down_sec=30.0))
        for _ in range(5):
            cb.allow()
            cb.record_failure()
        future = 1_000_000.0
        with patch("larkhelm.memory_circuit.time.monotonic", return_value=future):
            self.assertTrue(cb.allow())  # half_open probe
            cb.record_failure()
            self.assertEqual(cb.current_state(), "open")

    def test_window_outside_resets_failures(self) -> None:
        """Failures spaced beyond ``window_sec`` must not accumulate."""
        cb = CircuitBreaker(CircuitConfig(
            failure_threshold=5, window_sec=1.0, cool_down_sec=30.0,
        ))
        t = [0.0]
        with patch("larkhelm.memory_circuit.time.monotonic",
                   side_effect=lambda: t[0]):
            for _ in range(10):
                self.assertTrue(cb.allow())
                cb.record_failure()
                t[0] += 2.0  # each failure 2s apart > window 1.0s
            self.assertEqual(cb.current_state(), "closed")

    def test_window_inside_accumulates(self) -> None:
        """Failures within ``window_sec`` accumulate to open the circuit."""
        cb = CircuitBreaker(CircuitConfig(
            failure_threshold=5, window_sec=10.0, cool_down_sec=30.0,
        ))
        t = [0.0]
        with patch("larkhelm.memory_circuit.time.monotonic",
                   side_effect=lambda: t[0]):
            for _ in range(5):
                cb.allow()
                cb.record_failure()
                t[0] += 0.2  # 5 failures within 1s — well inside window
            self.assertEqual(cb.current_state(), "open")


class TestMemoryLLMRouterCircuit(unittest.TestCase):

    def test_circuit_state_exposed(self) -> None:
        from larkhelm import memory_llm_router as mlr
        mlr._rebuild_circuit()
        self.assertEqual(mlr.circuit_state(), "closed")


if __name__ == "__main__":
    unittest.main()
