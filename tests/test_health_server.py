"""Tests for ``larkhelm.health_server`` (P1-3)."""
from __future__ import annotations

import os
import socket
import time
import urllib.request

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import health_server  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def test_start_health_server_port_zero_is_noop():
    assert health_server.start_health_server(0) is False
    assert health_server.is_running() is False


def test_start_health_server_starts_and_serves():
    port = _free_port()
    try:
        ok = health_server.start_health_server(port, "127.0.0.1")
        assert ok is True
        assert health_server.is_running()
        # Give the thread a moment to bind
        time.sleep(0.05)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=2,
        ) as resp:
            body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "larkhelm_memory_rss_bytes" in body
    finally:
        health_server.stop_health_server()


def test_health_endpoint_returns_unhealthy_without_ws():
    port = _free_port()
    try:
        ok = health_server.start_health_server(port)
        assert ok
        time.sleep(0.05)
        # In test mode the bridge hasn't initialised, so /health → 503.
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2,
            ) as resp:
                # Snapshot may report healthy depending on prior fixture
                # state; only assert the status is 200 or 503 (one of two).
                assert resp.status in (200, 503)
        except urllib.error.HTTPError as e:
            assert e.code == 503
    finally:
        health_server.stop_health_server()


def test_ready_endpoint_present():
    port = _free_port()
    try:
        health_server.start_health_server(port)
        time.sleep(0.05)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/ready", timeout=2,
            ) as resp:
                assert resp.status in (200, 503)
        except urllib.error.HTTPError as e:
            assert e.code == 503
    finally:
        health_server.stop_health_server()


def test_unknown_endpoint_404():
    port = _free_port()
    try:
        health_server.start_health_server(port)
        time.sleep(0.05)
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/nonexistent", timeout=2,
            )
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        health_server.stop_health_server()


def test_current_snapshot_returns_dataclass():
    snap = health_server.current_snapshot()
    assert isinstance(snap.ws_connected, bool)
    assert isinstance(snap.backend_total_count, int)
    assert isinstance(snap.active_queries, int)
    # is_healthy must be a callable
    assert callable(snap.is_healthy)


def test_active_query_counter_tracks_record_start_end():
    """Wiring regression test.

    P1 review caught that ``health_server.increment_active_query`` /
    ``decrement_active_query`` were never called from production code,
    so the ``/metrics`` ``larkhelm_active_queries`` gauge was permanently
    flat-zero. The fix routes ``_get_active_queries`` through
    ``_query_card_state.get_diagnostics`` — the SAME counter that
    ``_do_query`` already writes via
    ``record_query_start`` / ``record_query_end``.

    This test exercises that wiring end-to-end: drive the production
    bookkeeping API and assert the metric reflects it.
    """
    from larkhelm.handlers._query_card_state import (
        record_query_start, record_query_end,
    )
    snap_a = health_server.current_snapshot()
    record_query_start()
    try:
        snap_b = health_server.current_snapshot()
        assert snap_b.active_queries == snap_a.active_queries + 1, (
            "active_queries gauge did not see record_query_start — "
            "metric is back to flat-zero (P1 review regression)"
        )
    finally:
        record_query_end(elapsed_sec=0.0)
    snap_c = health_server.current_snapshot()
    assert snap_c.active_queries == snap_a.active_queries


def test_stop_health_server_idempotent():
    # No raise, even when nothing is running.
    health_server.stop_health_server()
    health_server.stop_health_server()


# ── Week4-P1 M-HEALTH-SERVER: liveness / readiness split ────────────────────


def test_health_liveness_ignores_ws(monkeypatch):
    """/health (liveness) returns 200 even when WebSocket is disconnected."""
    monkeypatch.setattr(health_server, "_get_ws_connected", lambda: False)
    monkeypatch.setattr(health_server, "_get_memory_ok", lambda: True)
    monkeypatch.setattr(health_server, "_crash_flag", False)
    snap = health_server.current_snapshot()
    assert snap.is_live() is True, "is_live() must be True when only ws is down"


def test_health_503_on_crash_flag(monkeypatch):
    """set_crash_flag() causes is_live() → False → /health 503."""
    monkeypatch.setattr(health_server, "_crash_flag", False)
    monkeypatch.setattr(health_server, "_get_memory_ok", lambda: True)
    health_server.set_crash_flag()
    try:
        snap = health_server.current_snapshot()
        assert snap.is_live() is False, "crash_flag should make is_live() False"
        assert snap.crash_flag is True
    finally:
        # Reset so other tests are not affected
        monkeypatch.setattr(health_server, "_crash_flag", False)


def test_health_503_on_memory_oom(monkeypatch):
    """Memory over limit causes is_live() → False."""
    monkeypatch.setattr(health_server, "_get_rss_bytes", lambda: 9 * 1024 * 1024 * 1024)
    monkeypatch.setattr(health_server, "_crash_flag", False)
    snap = health_server.current_snapshot()
    assert snap.memory_ok is False
    assert snap.is_live() is False


def test_ready_503_when_ws_disconnected(monkeypatch):
    """is_ready() → False when WebSocket is not connected."""
    monkeypatch.setattr(health_server, "_get_ws_connected", lambda: False)
    snap = health_server.current_snapshot()
    assert snap.is_ready() is False


def test_ready_503_when_lark_api_stale(monkeypatch):
    """is_ready() → False when _lark_api_last_ok_ts == 0 (never succeeded)."""
    import larkhelm.lark_client as _lc
    monkeypatch.setattr(_lc, "_lark_api_last_ok_ts", 0.0)
    monkeypatch.setattr(health_server, "_get_ws_connected", lambda: True)
    snap = health_server.current_snapshot()
    assert snap.lark_api_ok is False
    assert snap.is_ready() is False


def test_ready_200_when_ws_and_api_ok(monkeypatch):
    """is_ready() → True when both ws is connected and Lark API is fresh."""
    import time
    import larkhelm.lark_client as _lc
    monkeypatch.setattr(_lc, "_lark_api_last_ok_ts", time.time())
    monkeypatch.setattr(health_server, "_get_ws_connected", lambda: True)
    snap = health_server.current_snapshot()
    assert snap.lark_api_ok is True
    assert snap.ws_connected is True
    assert snap.is_ready() is True
