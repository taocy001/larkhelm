"""AC-05 — P3 REQ-05 cascade ExponentialBackoff retry chain.

3 attempts with delays 1s, 2s should sum to 3s of (faked) sleep
before the 4th attempt — except we stop at ``max_attempts=3``
total, so the calling function sleeps 1s + 2s = 3s and then the
third raise propagates.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from larkhelm.memory_circuit import BackoffConfig, ExponentialBackoff


class TestExponentialBackoff(unittest.TestCase):

    def test_retries_until_success(self) -> None:
        bo = ExponentialBackoff(BackoffConfig(max_attempts=3))
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("flaky")
            return "ok"

        slept: list[float] = []
        with patch("larkhelm.memory_circuit.time.sleep", side_effect=slept.append):
            self.assertEqual(bo.run(fn), "ok")
        # Two sleeps before the third call.
        self.assertEqual(slept, [1.0, 2.0])
        self.assertEqual(calls["n"], 3)

    def test_gives_up_after_max_attempts(self) -> None:
        bo = ExponentialBackoff(BackoffConfig(max_attempts=3))

        def fn():
            raise RuntimeError("permanent")

        slept: list[float] = []
        with patch("larkhelm.memory_circuit.time.sleep", side_effect=slept.append):
            with self.assertRaises(RuntimeError):
                bo.run(fn)
        # 2 sleeps total (between attempts 1→2 and 2→3); 4th attempt never runs.
        self.assertEqual(slept, [1.0, 2.0])

    def test_delay_doubles_and_caps(self) -> None:
        bo = ExponentialBackoff(BackoffConfig(
            initial_sec=1.0, multiplier=2.0, max_sec=4.0, max_attempts=5,
        ))
        self.assertEqual(bo._delay_for(1), 1.0)
        self.assertEqual(bo._delay_for(2), 2.0)
        self.assertEqual(bo._delay_for(3), 4.0)
        # Past cap → max_sec
        self.assertEqual(bo._delay_for(4), 4.0)
        self.assertEqual(bo._delay_for(5), 4.0)


if __name__ == "__main__":
    unittest.main()
