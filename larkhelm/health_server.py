"""larkhelm · HTTP health / readiness / metrics endpoint (P1-3).

Implements three text endpoints exposed via a stdlib ``ThreadingHTTPServer``:

* ``GET /health``  — 200 when the bridge looks alive (init complete +
  WebSocket connected + ≥1 healthy backend); 503 otherwise.
* ``GET /ready``   — same condition as ``/health``, separate endpoint to
  follow the k8s convention so liveness vs. readiness can be wired up
  differently later.
* ``GET /metrics`` — Prometheus exposition: ``larkhelm_backend_healthy{name=...}``,
  ``larkhelm_active_queries``, ``larkhelm_memory_rss_bytes``,
  ``larkhelm_cascade_*``.

Default: ``health_endpoint_port=0`` (disabled). Operators flip the port
to enable the endpoint. Bound to ``127.0.0.1`` by default so prod
deployments don't accidentally expose internals to the world; mount a
reverse proxy if external scraping is needed.

This module imports lazily (only when ``start_health_server`` is called)
to keep startup cheap; no third-party deps.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


@dataclass(frozen=True)
class HealthSnapshot:
    ws_connected: bool
    backend_healthy_count: int
    backend_total_count: int
    init_complete: bool
    active_queries: int
    memory_rss_bytes: int
    backend_status: tuple = field(default_factory=tuple)  # list[(name, healthy: 0/1)]
    cascade_active: int = 0
    cascade_dropped_total: int = 0
    cascade_midflight_cancelled_total: int = 0

    def is_healthy(self) -> bool:
        return (
            self.ws_connected
            and self.backend_healthy_count >= 1
            and self.init_complete
        )


# ── Active-query counter ────────────────────────────────────────────────
#
# Single source of truth: ``_query_card_state._DIAG_ACTIVE``, mutated by
# ``record_query_start`` / ``record_query_end`` which BOTH legacy
# ``_do_query`` and ``QuerySession.run`` already call (see
# ``_query.py:435/938`` and ``_query_session.py:97/140``). The earlier
# ``increment_active_query`` / ``decrement_active_query`` pair lived here
# but was never wired to production code — independent P1 review caught
# that ``/metrics``'s ``larkhelm_active_queries`` gauge was permanently
# flat-zero. Routing the read through ``get_diagnostics()`` removes the
# second counter entirely so the metric reflects reality.
def _get_active_queries() -> int:
    try:
        from larkhelm.handlers._query_card_state import get_diagnostics
        return int(get_diagnostics().get("active_queries", 0))
    except Exception:
        return 0


# ── Snapshot collector ──────────────────────────────────────────────────


def _get_rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:
            with open("/proc/self/statm") as fh:
                pages = int(fh.read().split()[1])
            return pages * 4096
        except Exception:
            return 0


def _get_ws_connected() -> bool:
    """Heuristic: ``lark_client.client`` is set after bridge ``main()`` builds it."""
    try:
        import larkhelm.lark_client as _lc
        return getattr(_lc, "client", None) is not None
    except Exception:
        return False


def _get_init_complete() -> bool:
    try:
        import larkhelm.config as _cfg
        return getattr(_cfg, "_runtime", None) is not None
    except Exception:
        return False


def _get_backend_statuses() -> tuple:
    try:
        import larkhelm.config as _cfg
        registry = getattr(_cfg, "BACKEND_REGISTRY", None)
        if registry is None:
            return tuple()
        return tuple(
            (s.id, 1 if getattr(s, "healthy", False) else 0)
            for s in registry.all_enabled()
        )
    except Exception:
        return tuple()


def _get_cascade_counts() -> tuple[int, int, int]:
    try:
        from larkhelm.memory import get_cascade_stats
        stats = get_cascade_stats() or {}
        return (
            int(stats.get("active", 0)),
            int(stats.get("dropped_total", 0)),
            int(stats.get("midflight_cancelled_total", 0)),
        )
    except Exception:
        return 0, 0, 0


def current_snapshot() -> HealthSnapshot:
    backend_status = _get_backend_statuses()
    healthy = sum(1 for _, h in backend_status if h)
    casc_active, casc_dropped, casc_mid = _get_cascade_counts()
    return HealthSnapshot(
        ws_connected=_get_ws_connected(),
        backend_healthy_count=healthy,
        backend_total_count=len(backend_status),
        init_complete=_get_init_complete(),
        active_queries=_get_active_queries(),
        memory_rss_bytes=_get_rss_bytes(),
        backend_status=backend_status,
        cascade_active=casc_active,
        cascade_dropped_total=casc_dropped,
        cascade_midflight_cancelled_total=casc_mid,
    )


# ── HTTP handler ────────────────────────────────────────────────────────


class HealthRequestHandler(BaseHTTPRequestHandler):
    server_version = "larkhelm-health/1.0"

    def do_GET(self) -> None:  # noqa: N802 — stdlib name
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._handle_health()
        elif path == "/ready":
            self._handle_ready()
        elif path == "/metrics":
            self._handle_metrics()
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found\n")

    def _write_text(self, status: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_health(self) -> None:
        snap = current_snapshot()
        status = 200 if snap.is_healthy() else 503
        body = "ok\n" if snap.is_healthy() else "unhealthy\n"
        self._write_text(status, body)

    def _handle_ready(self) -> None:
        snap = current_snapshot()
        status = 200 if snap.is_healthy() else 503
        body = "ready\n" if snap.is_healthy() else "not ready\n"
        self._write_text(status, body)

    def _handle_metrics(self) -> None:
        snap = current_snapshot()

        # P2 REQ-01 / AC-01: prefer the prometheus-client renderer; fall back
        # to the legacy hand-written text path when the SDK is missing OR the
        # operator forced ``metrics_text_legacy=true``. The fall-back path
        # below is byte-compatible with master so existing scrapers keep
        # parsing during a half-rolled-out deployment.
        try:
            from larkhelm import metrics as _met
            _met.update_health_gauges(snap)
            body = _met.render_exposition()
            self._write_text(200, body)
            return
        except Exception:
            # Either prometheus-client not installed, legacy flag set, or
            # rendering blew up — fall through to the P1 text path.
            pass

        lines: list[str] = []
        lines.append("# HELP larkhelm_backend_healthy 1 if backend is healthy else 0")
        lines.append("# TYPE larkhelm_backend_healthy gauge")
        for name, healthy in snap.backend_status:
            safe = name.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'larkhelm_backend_healthy{{name="{safe}"}} {healthy}')

        lines.append("# HELP larkhelm_active_queries Current active _do_query invocations")
        lines.append("# TYPE larkhelm_active_queries gauge")
        lines.append(f"larkhelm_active_queries {snap.active_queries}")

        lines.append("# HELP larkhelm_memory_rss_bytes Resident set size in bytes")
        lines.append("# TYPE larkhelm_memory_rss_bytes gauge")
        lines.append(f"larkhelm_memory_rss_bytes {snap.memory_rss_bytes}")

        lines.append("# HELP larkhelm_cascade_active In-flight memory cascade extracts")
        lines.append("# TYPE larkhelm_cascade_active gauge")
        lines.append(f"larkhelm_cascade_active {snap.cascade_active}")

        lines.append("# HELP larkhelm_cascade_dropped_total Cascades dropped (sem full / cancel)")
        lines.append("# TYPE larkhelm_cascade_dropped_total counter")
        lines.append(f"larkhelm_cascade_dropped_total {snap.cascade_dropped_total}")

        lines.append("# HELP larkhelm_cascade_midflight_cancelled_total Cascades cancelled mid-LLM")
        lines.append("# TYPE larkhelm_cascade_midflight_cancelled_total counter")
        lines.append(
            f"larkhelm_cascade_midflight_cancelled_total {snap.cascade_midflight_cancelled_total}"
        )

        # Diagnostics from card-state (best-effort; if not loaded, skip)
        try:
            from larkhelm.handlers._query_card_state import get_diagnostics
            diag = get_diagnostics() or {}
            avg = float(diag.get("avg_elapsed_sec", 0.0) or 0.0)
            lines.append("# HELP larkhelm_query_avg_elapsed_sec Recent average query elapsed seconds")
            lines.append("# TYPE larkhelm_query_avg_elapsed_sec gauge")
            lines.append(f"larkhelm_query_avg_elapsed_sec {avg}")
        except Exception:
            pass

        body = "\n".join(lines) + "\n"
        self._write_text(200, body)

    def log_message(self, fmt: str, *args) -> None:  # noqa: N802 — silence access log
        # Silence stderr access log. Each request would otherwise emit one
        # line and Prometheus scrapes every 15s → ~5K lines/day of noise.
        return


# ── Module-level server lifecycle ───────────────────────────────────────

_server: Optional[ThreadingHTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_server_lock = threading.Lock()


def start_health_server(port: int, bind_addr: str = "127.0.0.1") -> bool:
    """Start the HTTP health/metrics server.

    Returns ``False`` (no-op) when ``port == 0`` — that's the operator's
    way of disabling the endpoint. Returns ``True`` on successful start.
    Idempotent: a second call while a server is running logs and returns
    True without rebinding.
    """
    global _server, _server_thread
    if not port or port == 0:
        return False

    with _server_lock:
        if _server is not None:
            return True

        try:
            server = ThreadingHTTPServer((bind_addr, int(port)), HealthRequestHandler)
        except Exception as e:
            try:
                from larkhelm.log import warn
                warn(f"[HealthServer] failed to bind {bind_addr}:{port}: {e}")
            except Exception:
                pass
            return False

        thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="health-server",
        )
        thread.start()
        _server = server
        _server_thread = thread

    try:
        from larkhelm.log import info
        info(f"[HealthServer] listening on {bind_addr}:{port}")
    except Exception:
        pass
    return True


def stop_health_server(timeout: float = 2.0) -> None:
    """Idempotent shutdown."""
    global _server, _server_thread
    with _server_lock:
        srv = _server
        thr = _server_thread
        _server = None
        _server_thread = None

    if srv is not None:
        try:
            srv.shutdown()
        except Exception:
            pass
        try:
            srv.server_close()
        except Exception:
            pass
    if thr is not None:
        try:
            thr.join(timeout=timeout)
        except Exception:
            pass


def is_running() -> bool:
    with _server_lock:
        return _server is not None


__all__ = [
    "HealthSnapshot",
    "HealthRequestHandler",
    "start_health_server",
    "stop_health_server",
    "current_snapshot",
    "is_running",
]
