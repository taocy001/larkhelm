"""P2 AC-01: tests for ``larkhelm.metrics`` registry + ``/metrics`` integration."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import metrics as _met  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_metrics_state():
    """Drop the cached singleton + prometheus-client handle between cases."""
    _met._reset_for_tests()
    yield
    _met._reset_for_tests()


@pytest.fixture
def _force_legacy_off(monkeypatch):
    """Ensure ``metrics_text_legacy`` is False so the prom path is taken."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "METRICS_TEXT_LEGACY", False, raising=False)
    return _cfg


# ── 1) prometheus-client path: 4 core metrics + HELP/TYPE ─────────────────

def test_render_exposition_emits_core_metrics(_force_legacy_off):
    pytest.importorskip("prometheus_client")
    body = _met.render_exposition()
    # Four core metrics from PRD §7 AC-01.
    for name in (
        "larkhelm_backend_healthy",
        "larkhelm_active_queries",
        "larkhelm_memory_rss_bytes",
        "larkhelm_cascade_extract_total",
    ):
        assert f"# HELP {name}" in body, f"{name} missing HELP"
        assert f"# TYPE {name}" in body, f"{name} missing TYPE"


# ── 2) metrics_text_legacy=True forces fallback ────────────────────────────

def test_legacy_mode_forces_fallback(monkeypatch):
    pytest.importorskip("prometheus_client")
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "METRICS_TEXT_LEGACY", True, raising=False)
    # The renderer must raise so health_server falls back to the P1 text.
    with pytest.raises(_met.PrometheusNotInstalled):
        _met.render_exposition()


# ── 3) prometheus-client missing → auto fallback ───────────────────────────

def test_missing_prometheus_client_falls_back(_force_legacy_off, monkeypatch):
    # Monkeypatch the resolver to pretend the SDK isn't installed.
    monkeypatch.setattr(_met, "_resolve_prom_client", lambda: None)
    _met._reset_for_tests()  # re-build registry with fake "missing" state
    assert not _met.is_prometheus_available()
    with pytest.raises(_met.PrometheusNotInstalled):
        _met.render_exposition()


# ── 4) update_health_gauges accepts a HealthSnapshot ──────────────────────

def test_update_health_gauges_accepts_snapshot(_force_legacy_off):
    pytest.importorskip("prometheus_client")
    from larkhelm.health_server import HealthSnapshot
    snap = HealthSnapshot(
        ws_connected=True,
        backend_healthy_count=2,
        backend_total_count=3,
        init_complete=True,
        active_queries=4,
        memory_rss_bytes=1024 * 1024,
        backend_status=(("claude", 1), ("kimi", 1), ("deepseek", 0)),
        cascade_active=1,
        cascade_dropped_total=5,
        cascade_midflight_cancelled_total=2,
    )
    _met.update_health_gauges(snap)
    body = _met.render_exposition()
    assert 'larkhelm_backend_healthy{name="claude"} 1' in body
    assert 'larkhelm_backend_healthy{name="deepseek"} 0' in body
    assert "larkhelm_active_queries 4" in body
    # Memory RSS value — prometheus_client emits floats; for 1 MiB it
    # may render as ``1048576.0`` or ``1.048576e+06`` depending on
    # version, so check for the metric name + a non-zero numeric prefix.
    assert "larkhelm_memory_rss_bytes 1" in body


# ── 5) inc_cascade_extract accumulates per (kind, outcome) ────────────────

def test_inc_cascade_extract_per_label(_force_legacy_off):
    pytest.importorskip("prometheus_client")
    _met.inc_cascade_extract("project", "success")
    _met.inc_cascade_extract("project", "success")
    _met.inc_cascade_extract("project", "unchanged")
    _met.inc_cascade_extract("global", "error")
    body = _met.render_exposition()
    # Labels appear on the metric line; values are 1 or 2 depending on
    # how many times we incremented above.
    assert 'larkhelm_cascade_extract_total{kind="project",outcome="success"} 2' in body
    assert 'larkhelm_cascade_extract_total{kind="project",outcome="unchanged"} 1' in body
    assert 'larkhelm_cascade_extract_total{kind="global",outcome="error"} 1' in body


# ── Bonus: extract_buffer_flushes counter is wired ────────────────────────

def test_inc_extract_buffer_flush(_force_legacy_off):
    pytest.importorskip("prometheus_client")
    _met.inc_extract_buffer_flush("timer")
    _met.inc_extract_buffer_flush("timer")
    _met.inc_extract_buffer_flush("capacity")
    body = _met.render_exposition()
    assert 'larkhelm_extract_buffer_flushes_total{trigger="timer"} 2' in body
    assert 'larkhelm_extract_buffer_flushes_total{trigger="capacity"} 1' in body


# ── health_server.HealthRequestHandler ↔ metrics bridge ───────────────────

def test_health_server_metrics_route_uses_registry(_force_legacy_off):
    """When prometheus-client is present, /metrics body must come from
    metrics.render_exposition (not the legacy hand-written path).
    """
    pytest.importorskip("prometheus_client")
    from larkhelm import health_server as _hs

    # Exercise the metrics-route handler against a fake socket. We use a
    # minimal stub that captures write_text calls.
    captured: dict[str, str] = {}

    class _StubHandler(_hs.HealthRequestHandler):
        def __init__(self):
            pass  # bypass BaseHTTPRequestHandler.__init__

        def _write_text(self, status, body):
            captured["status"] = status
            captured["body"] = body

    h = _StubHandler()
    h._handle_metrics()
    assert captured["status"] == 200
    # The Prometheus exposition body always opens with a "# HELP" line.
    assert "# HELP larkhelm_backend_healthy" in captured["body"]
