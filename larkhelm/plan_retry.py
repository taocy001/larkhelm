"""larkhelm · plan_retry — step-level retry decision engine (P3 REQ-06).

The plan / multi-stage workflows already track ``retry_count`` and
``max_retries`` on each stage but never act on them. This module
provides the small piece that was missing: given a stage_state dict,
decide whether to rerun.

Three policies:

* ``"now"`` — immediately re-execute the stage (caller resets the
  stage state and dispatches).
* ``"manual"`` — render a "Retry" button on the failure card; only a
  human tap triggers the rerun.
* ``"off"`` — never retry (status quo P0-P2 behaviour).

The engine does NOT mutate ``stage_state`` inside :meth:`evaluate`;
:meth:`mark_retry_attempted` is the explicit mutator. Callers persist
state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RetryDecision:
    should_retry: bool
    reason: str               # "retries_exhausted" | "below_threshold" | "manual_required" | "disabled"
    next_retry_at: float = 0.0  # epoch seconds; 0 = immediate / manual


_VALID_STRATEGIES = ("now", "manual", "off")


class PlanRetryEngine:
    """Decision engine for stage-level retries."""

    def __init__(self, strategy: str) -> None:
        s = str(strategy or "off").strip().lower()
        if s not in _VALID_STRATEGIES:
            s = "off"
        self.strategy = s

    def evaluate(self, stage_state: dict) -> RetryDecision:
        """Inspect ``stage_state['retry_count']`` and ``['max_retries']``.

        Returns a :class:`RetryDecision`. Does not mutate the dict.
        """
        if self.strategy == "off":
            return RetryDecision(False, "disabled")
        try:
            retry_count = int(stage_state.get("retry_count", 0) or 0)
            max_retries = int(stage_state.get("max_retries", 0) or 0)
        except (TypeError, ValueError):
            return RetryDecision(False, "disabled")
        if max_retries <= 0:
            return RetryDecision(False, "retries_exhausted")
        if retry_count >= max_retries:
            return RetryDecision(False, "retries_exhausted")
        if self.strategy == "manual":
            return RetryDecision(
                True, "manual_required", next_retry_at=0.0,
            )
        # strategy == "now"
        return RetryDecision(True, "below_threshold", next_retry_at=time.time())

    def mark_retry_attempted(self, stage_state: dict) -> None:
        """Bump ``stage_state['retry_count']`` in place by 1.

        Caller is responsible for persisting. No-op if the stage_state
        dict is malformed (missing or non-int ``retry_count``).
        """
        if not isinstance(stage_state, dict):
            return
        try:
            current = int(stage_state.get("retry_count", 0) or 0)
        except (TypeError, ValueError):
            current = 0
        stage_state["retry_count"] = current + 1


__all__ = ["PlanRetryEngine", "RetryDecision"]
