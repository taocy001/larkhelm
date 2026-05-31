"""Tests for larkhelm.token_stats.estimate_cache_savings — AC-03."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm.token_stats import estimate_cache_savings, record_token_usage  # noqa: E402
from larkhelm.log import _debug_log  # noqa: F401,E402


def test_estimate_cache_savings_claude_positive():
    """AC-03: claude with real cache_read should return positive savings."""
    savings = estimate_cache_savings(
        "claude",
        {"input_tokens": 1000, "cache_read": 800, "cache_create": 200},
    )
    assert savings > 0, f"Expected positive savings for claude, got {savings}"


def test_estimate_cache_savings_kimi_is_zero():
    """Kimi has no cache pricing → savings must be 0.0."""
    savings = estimate_cache_savings("kimi", {"cache_read": 99999})
    assert savings == 0.0


def test_estimate_cache_savings_zero_cache_read():
    """Claude with cache_read=0 → savings must be 0.0."""
    savings = estimate_cache_savings("claude", {"cache_read": 0})
    assert savings == 0.0


def test_estimate_cache_savings_unknown_model():
    """Unknown model → savings must be 0.0 (never raises)."""
    savings = estimate_cache_savings("unknown_model_xyz", {"cache_read": 500})
    assert savings == 0.0


def test_estimate_cache_savings_missing_cache_read_key():
    """usage dict with no cache_read key → 0.0, no exception."""
    savings = estimate_cache_savings("claude", {"input_tokens": 1000})
    assert savings == 0.0


def test_estimate_cache_savings_none_usage():
    """None usage → 0.0, no exception."""
    savings = estimate_cache_savings("claude", None)
    assert savings == 0.0


def test_estimate_cache_savings_deepseek_positive():
    """DeepSeek has a defined rate and should return > 0 for nonzero cache_read."""
    savings = estimate_cache_savings("deepseek", {"cache_read": 1_000_000})
    assert savings > 0


def test_estimate_cache_savings_gemini_positive():
    """Gemini has a defined rate and should return > 0 for nonzero cache_read."""
    savings = estimate_cache_savings("gemini", {"cache_read": 1_000_000})
    assert savings > 0


def test_estimate_cache_savings_large_token_count():
    """Sanity check: 1M claude cache_read at $2.70/M → ~$2.70."""
    savings = estimate_cache_savings("claude", {"cache_read": 1_000_000})
    assert abs(savings - 2.70) < 0.01, f"Expected ~2.70 USD but got {savings}"


def test_record_token_usage_low_cache_hit_rate_logs(monkeypatch):
    """AC-09: record_token_usage emits _debug_log when hit rate < threshold and cache_read > 0."""
    import larkhelm.config as _cfg

    log_calls: list[str] = []

    monkeypatch.setattr(
        "larkhelm.token_stats._debug_log",
        lambda msg: log_calls.append(msg),
        raising=False,
    )
    monkeypatch.setattr(_cfg, "CACHE_HIT_RATE_ALERT_THRESHOLD", 0.8, raising=False)

    # Stub heavy side-effects that aren't relevant to this test
    monkeypatch.setattr("larkhelm.token_stats.inc_tokens", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(
        "larkhelm.token_stats.inc_cache_write_tokens", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.token_stats.inc_cache_read_tokens", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.token_stats.observe_cache_hit_rate", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.token_stats.set_cache_hit_ratio", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.token_stats.inc_cache_savings", lambda *a, **kw: None, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.session_guard.maybe_auto_reset", lambda *a, **kw: None, raising=False,
    )

    # cache_read=100, input_tokens=900 → hit_rate ≈ 0.10 < threshold 0.8
    usage = {"input_tokens": 900, "output_tokens": 50, "cache_read": 100, "cache_create": 0}
    record_token_usage("chat_ac09", "claude", usage)

    low_hit_msgs = [m for m in log_calls if "low cache hit rate" in m]
    assert low_hit_msgs, (
        f"Expected a 'low cache hit rate' log message but got log_calls={log_calls}"
    )
