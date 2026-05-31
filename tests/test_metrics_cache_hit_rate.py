"""AC-01 / AC-02: prompt_cache_hit_rate Histogram tests."""
import pytest

pytest.importorskip("prometheus_client")

import larkhelm.metrics as _metrics


@pytest.fixture(autouse=True)
def _reset_for_tests():
    _metrics._reset_for_tests()
    yield
    _metrics._reset_for_tests()


def test_ac01_histogram_registered():
    """AC-01: prompt_cache_hit_rate Histogram is registered and exposition contains the metric name."""
    reg = _metrics.get_registry()
    assert reg.available, "prometheus_client must be installed for this test"
    assert reg.prompt_cache_hit_rate is not None, "prompt_cache_hit_rate must be registered"

    text = _metrics.render_exposition()
    assert "larkhelm_prompt_cache_hit_rate" in text, (
        f"Expected 'larkhelm_prompt_cache_hit_rate' in exposition output"
    )


def test_ac02_observe_cache_hit_rate():
    """AC-02: observe_cache_hit_rate records a sample into the Histogram."""
    reg = _metrics.get_registry()
    assert reg.available

    _metrics.observe_cache_hit_rate("claude", 0.667)

    # Collect and verify a sample was recorded
    families = list(reg.prompt_cache_hit_rate.collect())
    assert families, "No metric families collected"

    sample_count = None
    sample_sum = None
    for family in families:
        for sample in family.samples:
            if "claude" not in (sample.labels or {}).get("backend", ""):
                continue
            if sample.name.endswith("_count"):
                sample_count = sample.value
            elif sample.name.endswith("_sum"):
                sample_sum = sample.value

    assert sample_count is not None and sample_count >= 1, (
        f"Expected sample_count >= 1, got {sample_count}"
    )
    assert sample_sum is not None and abs(sample_sum - 0.667) < 0.01, (
        f"Expected sample_sum ≈ 0.667, got {sample_sum}"
    )


def test_observe_on_record(tmp_path, monkeypatch):
    """AC-02: record_token_usage → observe_cache_hit_rate integration path.

    cache_read=1000, input_tokens=500 → hit_rate = 1000/(1000+500) ≈ 0.667
    Histogram must show sample_count≥1 and sample_sum≈0.667.
    """
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "LOG_DIR", tmp_path, raising=False)

    from larkhelm.token_stats import record_token_usage

    reg = _metrics.get_registry()
    assert reg.available

    record_token_usage("test_chat", "claude", {"cache_read": 1000, "input_tokens": 500})

    families = list(reg.prompt_cache_hit_rate.collect())
    assert families, "No metric families collected"

    sample_count = None
    sample_sum = None
    for family in families:
        for sample in family.samples:
            if "claude" not in (sample.labels or {}).get("backend", ""):
                continue
            if sample.name.endswith("_count"):
                sample_count = sample.value
            elif sample.name.endswith("_sum"):
                sample_sum = sample.value

    assert sample_count is not None and sample_count >= 1, (
        f"Expected sample_count >= 1, got {sample_count}"
    )
    expected_rate = 1000 / (1000 + 500 + 1e-9)  # ≈ 0.667
    assert sample_sum is not None and abs(sample_sum - expected_rate) < 0.01, (
        f"Expected sample_sum ≈ {expected_rate:.3f}, got {sample_sum}"
    )


def test_observe_cache_hit_rate_no_raise_when_unavailable(monkeypatch):
    """observe_cache_hit_rate must not raise when prometheus is unavailable."""
    monkeypatch.setattr(_metrics, "_prom_client", None)
    monkeypatch.setattr(_metrics, "_prom_client_checked", True)
    _metrics._reset_for_tests()
    # After reset, rebuild with no prom_client
    monkeypatch.setattr(_metrics, "_prom_client_checked", False)
    monkeypatch.setattr(_metrics, "_prom_client", None)

    # Calling should be a no-op, never raise
    _metrics.observe_cache_hit_rate("claude", 0.5)
