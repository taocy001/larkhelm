"""P1-5a: contract that ``inc_intent_layer`` / ``inc_intent_l2_fallback`` /
``inc_cascade_backoff_exhausted`` are best-effort and never raise.

Pins AC-10: when ``prometheus-client`` is missing OR the registry's Counter
attribute happens to be ``None`` (e.g. a partial init), the three helpers
return ``None`` silently. These three counters bump from cold paths
(``intent_router.resolve_intent`` / ``memory._run_one_shot_with_backoff`` /
``memory_extract_buffer._flush_with_backoff``) where an exception would
either corrupt classification or mask the underlying business error, so the
safety contract is load-bearing rather than cosmetic.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import metrics as _met  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_metrics_state():
    """Drop the cached singleton + prometheus_client handle between cases."""
    _met._reset_for_tests()
    yield
    _met._reset_for_tests()


# ── prometheus-client missing → helpers are no-ops ────────────────────────


def test_inc_intent_layer_when_unavailable(monkeypatch):
    monkeypatch.setattr(_met, "_resolve_prom_client", lambda: None)
    _met._reset_for_tests()
    # Two separate calls cover both label slots being passed through.
    assert _met.inc_intent_layer("l1", "hit") is None
    assert _met.inc_intent_layer("fallback", "hit") is None
    assert not _met.is_prometheus_available()


def test_inc_intent_l2_fallback_when_unavailable(monkeypatch):
    monkeypatch.setattr(_met, "_resolve_prom_client", lambda: None)
    _met._reset_for_tests()
    assert _met.inc_intent_l2_fallback() is None


def test_inc_cascade_backoff_exhausted_when_unavailable(monkeypatch):
    monkeypatch.setattr(_met, "_resolve_prom_client", lambda: None)
    _met._reset_for_tests()
    assert _met.inc_cascade_backoff_exhausted() is None


# ── registry available but Counter attr happens to be None ────────────────


def test_helpers_when_counter_attr_is_none():
    """AC-10 second leg: even if the registry constructed successfully but
    a Counter slot is ``None`` (defensive guard or partial init), each
    helper must early-return rather than dereferencing ``None.labels(...)``.
    """
    pytest.importorskip("prometheus_client")
    reg = _met.get_registry()
    if not reg.available:
        pytest.skip("prometheus-client not installed in this venv")

    # Stomp the three new Counter attributes — this simulates a registry
    # that thinks it's available but lost the slots (which is exactly the
    # path the ``or reg.<counter> is None`` early-return guards against).
    reg.intent_layer_total = None
    reg.intent_l2_fallback_total = None
    reg.cascade_backoff_exhausted_total = None

    # None of these should raise.
    assert _met.inc_intent_layer("l1", "hit") is None
    assert _met.inc_intent_layer("microlearn", "abstain") is None
    assert _met.inc_intent_l2_fallback() is None
    assert _met.inc_cascade_backoff_exhausted() is None


# ── inner labels(...).inc() raising must not propagate ────────────────────


def test_inc_intent_layer_swallows_labels_exception(monkeypatch):
    """If ``labels(...).inc()`` itself blows up (cardinality cap, bad value
    type, …), the helper must catch and ``safe_log`` rather than propagate.
    """
    pytest.importorskip("prometheus_client")
    reg = _met.get_registry()
    if not reg.available or reg.intent_layer_total is None:
        pytest.skip("prometheus-client not installed in this venv")

    class _BoomCounter:
        def labels(self, *_a, **_kw):
            raise RuntimeError("synthetic prom_client failure")

        def inc(self, *_a, **_kw):
            raise RuntimeError("should not be reached")

    monkeypatch.setattr(reg, "intent_layer_total", _BoomCounter(), raising=False)
    # The contract is "never raise" — assertion is implicit (no exception).
    _met.inc_intent_layer("l1", "hit")


def test_inc_intent_l2_fallback_swallows_inc_exception(monkeypatch):
    pytest.importorskip("prometheus_client")
    reg = _met.get_registry()
    if not reg.available or reg.intent_l2_fallback_total is None:
        pytest.skip("prometheus-client not installed in this venv")

    class _BoomCounter:
        def inc(self, *_a, **_kw):
            raise RuntimeError("synthetic prom_client failure")

    monkeypatch.setattr(reg, "intent_l2_fallback_total", _BoomCounter(), raising=False)
    _met.inc_intent_l2_fallback()


def test_inc_cascade_backoff_exhausted_swallows_inc_exception(monkeypatch):
    pytest.importorskip("prometheus_client")
    reg = _met.get_registry()
    if not reg.available or reg.cascade_backoff_exhausted_total is None:
        pytest.skip("prometheus-client not installed in this venv")

    class _BoomCounter:
        def inc(self, *_a, **_kw):
            raise RuntimeError("synthetic prom_client failure")

    monkeypatch.setattr(reg, "cascade_backoff_exhausted_total", _BoomCounter(), raising=False)
    _met.inc_cascade_backoff_exhausted()
