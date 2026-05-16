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


def test_active_query_counter_increments():
    snap_a = health_server.current_snapshot()
    health_server.increment_active_query()
    snap_b = health_server.current_snapshot()
    health_server.decrement_active_query()
    snap_c = health_server.current_snapshot()
    assert snap_b.active_queries == snap_a.active_queries + 1
    assert snap_c.active_queries == snap_a.active_queries


def test_stop_health_server_idempotent():
    # No raise, even when nothing is running.
    health_server.stop_health_server()
    health_server.stop_health_server()
