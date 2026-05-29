"""Tests for cache token metrics (larkhelm_cache_write_tokens_total,
larkhelm_cache_read_tokens_total, larkhelm_cache_hit_ratio).

Requires prometheus_client; skipped automatically when absent.
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import metrics as _met  # noqa: E402
import larkhelm.token_stats as _ts  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset metrics singleton and _cache_totals_by_model between cases."""
    _met._reset_for_tests()
    _ts._cache_totals_by_model.clear()
    yield
    _met._reset_for_tests()
    _ts._cache_totals_by_model.clear()


def _reg():
    return _met.get_registry()


# ── 1) cache_write_tokens_total accumulates correctly ─────────────────────

def test_cache_write_tokens_counter_accumulates():
    pytest.importorskip("prometheus_client")
    reg = _reg()
    assert reg.available

    _met.inc_cache_write_tokens("claude", 100)
    _met.inc_cache_write_tokens("claude", 50)

    body = reg.render()
    assert "larkhelm_cache_write_tokens_total" in body
    assert 'model="claude"} 150' in body


# ── 2) cache_read_tokens_total accumulates correctly ──────────────────────

def test_cache_read_tokens_counter_accumulates():
    pytest.importorskip("prometheus_client")
    reg = _reg()

    _met.inc_cache_read_tokens("claude", 200)
    _met.inc_cache_read_tokens("claude", 300)

    body = reg.render()
    assert "larkhelm_cache_read_tokens_total" in body
    assert 'model="claude"} 500' in body


# ── 3) hit_ratio calculated correctly from record_token_usage ─────────────

def test_hit_ratio_via_record_token_usage(tmp_path, monkeypatch):
    """record_token_usage updates both counters and the Gauge correctly."""
    pytest.importorskip("prometheus_client")
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "LOG_DIR", tmp_path, raising=False)

    reg = _reg()

    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_create": 80,   # write
        "cache_read": 320,    # read
        "cost_usd": 0.001,
    }
    _ts.record_token_usage("chat1", "claude", usage)

    body = reg.render()
    # cache_write_tokens_total should be 80
    assert 'larkhelm_cache_write_tokens_total{model="claude"} 80.0' in body
    # cache_read_tokens_total should be 320
    assert 'larkhelm_cache_read_tokens_total{model="claude"} 320.0' in body
    # hit_ratio = 320 / (80+320) = 0.8
    assert 'larkhelm_cache_hit_ratio{model="claude"} 0.8' in body


# ── 4) hit_ratio is 0 when denominator is zero ────────────────────────────

def test_hit_ratio_zero_when_no_cache_activity():
    pytest.importorskip("prometheus_client")
    reg = _reg()

    _met.set_cache_hit_ratio("claude", 0.0)

    body = reg.render()
    assert 'larkhelm_cache_hit_ratio{model="claude"} 0.0' in body


# ── 5) set_cache_hit_ratio updates Gauge directly ─────────────────────────

def test_set_cache_hit_ratio_direct():
    pytest.importorskip("prometheus_client")
    reg = _reg()

    _met.set_cache_hit_ratio("gemini", 0.95)

    body = reg.render()
    assert 'larkhelm_cache_hit_ratio{model="gemini"} 0.95' in body


# ── 6) helpers are no-ops when prometheus_client is absent ────────────────

def test_helpers_noop_without_prometheus(monkeypatch):
    """Never raise even when registry is unavailable."""
    _met._reset_for_tests()
    # Patch _resolve_prom_client to return None (simulate absent prom)
    monkeypatch.setattr(_met, "_prom_client", None)
    monkeypatch.setattr(_met, "_prom_client_checked", True)
    _met._reset_for_tests()
    monkeypatch.setattr(_met, "_prom_client", None)
    monkeypatch.setattr(_met, "_prom_client_checked", True)

    # These should not raise
    _met.inc_cache_write_tokens("claude", 10)
    _met.inc_cache_read_tokens("claude", 10)
    _met.set_cache_hit_ratio("claude", 0.5)
