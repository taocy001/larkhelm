"""P1-5a (W14): pin the ``cascade_backoff_exhausted_total`` Counter.

Two cascade-extract paths each call into :class:`ExponentialBackoff` and
must bump the counter exactly once when the retry loop genuinely exhausts:

  * ``larkhelm.memory_extract_buffer.ExtractBuffer._flush_with_backoff``
    — buffer-driven cascade; swallows the final exception by contract.
  * ``larkhelm.memory._run_one_shot_with_backoff``
    — direct cascade extract; re-raises the final exception.

Cancellation (``QueryCancelledError``) is NOT a backoff exhaustion and
must leave the counter untouched (AC-05 sanity).
"""
from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import metrics as _met  # noqa: E402
from larkhelm import memory_extract_buffer as _meb  # noqa: E402


pytestmark = pytest.mark.timeout(30)


@pytest.fixture(autouse=True)
def _reset_buffer_state():
    _meb._reset_for_tests()
    yield
    _meb._reset_for_tests()


def _read_counter(counter) -> float:
    """Pull the live value from a non-labelled prometheus_client Counter.

    Mirrors ``tests/test_context_cache.py:test_counters_increment_when_available``
    — ``_value.get()`` is the internal accessor for single-process mode. We
    catch and skip if the prometheus_client internals shift, rather than
    breaking the whole P1-5a regression net.
    """
    try:
        return counter._value.get()
    except Exception:
        pytest.skip("prometheus_client internal shape changed")


def _require_counter():
    pytest.importorskip("prometheus_client")
    reg = _met.get_registry()
    if not reg.available or reg.cascade_backoff_exhausted_total is None:
        pytest.skip("prometheus-client not installed in this venv")
    return reg.cascade_backoff_exhausted_total


# ── (a) _flush_with_backoff exhausted → buffer path bumps once ────────────


def test_invoke_cascade_with_backoff_exhausted_bumps_once(monkeypatch):
    """AC-04: a cascade fn that always raises must trigger exactly one bump
    on the ``cascade_backoff_exhausted_total`` counter after the retry loop
    gives up. The buffer path swallows the final exception (no re-raise).
    """
    counter = _require_counter()

    # Skip the configured sleep so the test runs in <1s rather than 3s.
    monkeypatch.setattr("larkhelm.memory_circuit.time.sleep", lambda *_a, **_kw: None)
    # Force buffer window=0 so record_session_update fires cascade synchronously
    # but the buffer still wraps the call in _flush_with_backoff.
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "MEMORY_EXTRACT_BUFFER_WINDOW_SEC", 0, raising=False)
    monkeypatch.setattr(_cfg, "CASCADE_BACKOFF_MAX_ATTEMPTS", 3, raising=False)

    def _always_fail(_content, _chat_id):
        raise RuntimeError("synthetic cascade failure")

    buf = _meb._get_buffer()
    buf.set_cascade_fn_for_tests(_always_fail)

    before = _read_counter(counter)
    # window=0 path → _invoke_cascade("immediate") → _flush_with_backoff
    _meb.record_session_update("oc_chat", "summary X")
    after = _read_counter(counter)

    assert after - before == pytest.approx(1.0), (
        f"expected exactly one bump, got delta={after - before}"
    )


# ── (b) _run_one_shot_with_backoff exhausted → bumps + re-raises ──────────


def test_one_shot_with_backoff_exhausted_bumps_once_before_raise(monkeypatch):
    """AC-05: ``_run_one_shot_with_backoff`` re-raises the last exception to
    its caller after backoff exhaustion. The bump must land before the
    re-raise so Grafana sees the give-up rate even when the caller bubbles
    the error up the stack.
    """
    counter = _require_counter()
    import larkhelm.memory as _memory
    import larkhelm.config as _cfg

    monkeypatch.setattr("larkhelm.memory_circuit.time.sleep", lambda *_a, **_kw: None)
    # Also patch the time.sleep referenced from memory.py's own retry loop
    # (the inner ``time.sleep(backoff._delay_for(attempt))`` call).
    monkeypatch.setattr(_memory.time, "sleep", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(_cfg, "CASCADE_BACKOFF_MAX_ATTEMPTS", 3, raising=False)

    call_count = {"n": 0}

    def _always_fail(*_a, **_kw):
        call_count["n"] += 1
        raise RuntimeError("synthetic one-shot failure")

    monkeypatch.setattr(_memory, "_run_one_shot", _always_fail, raising=False)

    before = _read_counter(counter)
    with pytest.raises(RuntimeError, match="synthetic one-shot failure"):
        _memory._run_one_shot_with_backoff("prompt", ns="ns", cancel_ev=None)
    after = _read_counter(counter)

    assert call_count["n"] == 3, (
        f"expected 3 retry attempts, got {call_count['n']}"
    )
    assert after - before == pytest.approx(1.0), (
        f"expected exactly one bump, got delta={after - before}"
    )


# ── (c) cancellation propagates immediately without bumping ───────────────


def test_cancellation_does_not_bump(monkeypatch):
    """``QueryCancelledError`` is not a backoff exhaustion — it's the user
    pulling the cord. The counter must stay flat so cancel storms don't
    inflate the "cascade gave up" gauge that triggers ops alerts.
    """
    counter = _require_counter()
    import larkhelm.memory as _memory
    import larkhelm.config as _cfg
    from larkhelm.ai_runner import QueryCancelledError

    monkeypatch.setattr("larkhelm.memory_circuit.time.sleep", lambda *_a, **_kw: None)
    monkeypatch.setattr(_memory.time, "sleep", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(_cfg, "CASCADE_BACKOFF_MAX_ATTEMPTS", 3, raising=False)

    def _cancel_immediately(*_a, **_kw):
        raise QueryCancelledError("cancelled mid-flight")

    monkeypatch.setattr(_memory, "_run_one_shot", _cancel_immediately, raising=False)

    cancel_ev = threading.Event()
    before = _read_counter(counter)
    with pytest.raises(QueryCancelledError):
        _memory._run_one_shot_with_backoff("prompt", ns="ns", cancel_ev=cancel_ev)
    after = _read_counter(counter)

    assert after - before == pytest.approx(0.0), (
        f"cancel path must not bump cascade_backoff_exhausted_total; delta={after - before}"
    )
