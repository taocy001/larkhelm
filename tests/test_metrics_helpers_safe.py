"""Pins AC-10: when ``prometheus-client`` is missing OR the registry's Counter
attribute happens to be ``None``, metric helpers return ``None`` silently.
The safety contract is load-bearing — these helpers are called from cold paths
where an exception would mask the underlying business error.
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


def test_inc_cascade_backoff_exhausted_when_unavailable(monkeypatch):
    monkeypatch.setattr(_met, "_resolve_prom_client", lambda: None)
    _met._reset_for_tests()
    assert _met.inc_cascade_backoff_exhausted() is None


# ── registry available but Counter attr happens to be None ────────────────


def test_helpers_when_counter_attr_is_none():
    """AC-10: even if registry constructed successfully but a Counter slot
    is ``None``, helpers must early-return rather than dereference None.labels().
    """
    pytest.importorskip("prometheus_client")
    reg = _met.get_registry()
    if not reg.available:
        pytest.skip("prometheus-client not installed in this venv")

    reg.cascade_backoff_exhausted_total = None
    assert _met.inc_cascade_backoff_exhausted() is None


# ── inner inc() raising must not propagate ────────────────────────────────


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
