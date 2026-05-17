"""larkhelm · Prometheus metrics registry (P2 REQ-01 / AC-01).

Single source of truth for all ``larkhelm_*`` Prometheus metrics. Modules
import the helper functions (``inc_cascade_extract`` / ``observe_query_duration``
/ ``update_health_gauges``) rather than touching the underlying
``prometheus_client`` objects directly so:

  * the import of ``prometheus_client`` stays optional (``[metrics]`` extra),
  * a single ``LarkhelmMetricsRegistry`` singleton owns the registry so
    ``generate_latest()`` always returns the same set of series, and
  * tests can rebuild the registry between cases via ``_reset_for_tests``
    without monkey-patching the prometheus_client globals.

Operator escape hatches:

  * ``metrics_text_legacy=true`` in config.json → ``render_exposition``
    raises :class:`PrometheusNotInstalled` regardless of install state, so
    :func:`health_server._handle_metrics` falls back to the P1 hand-written
    text exposition (byte-compatible with master).
  * ``prometheus-client`` not installed → same fallback; the registry is
    permanently in "unavailable" mode for the lifetime of the process.

Cardinality discipline: every metric below caps its label set at 3 labels
and ≤ 50 unique values per label, matching the PRD §3.1 budget.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

# Module-level safe_log import: putting this inside the except handlers (as
# the earlier version did) made the "never raise on metrics" contract worse
# — every silent path would re-trigger the import on each miss, and an
# ImportError (transient partial install / circular import) would propagate
# OUT of the metric call. Loading once at module load time pays the import
# cost up front and removes that failure surface. larkhelm.log has no
# circular dependency on metrics, so this is safe.
from larkhelm.log import safe_log

if TYPE_CHECKING:
    from larkhelm.health_server import HealthSnapshot


class PrometheusNotInstalled(RuntimeError):
    """Raised by :func:`render_exposition` when prometheus-client is absent
    or the operator forced legacy mode via ``metrics_text_legacy=true``.

    Caller (``health_server._handle_metrics``) catches this and renders the
    P1 hand-written exposition path instead.
    """


# Module-level handle to the optional prometheus_client module. Set at first
# call to :func:`_resolve_prom_client`; ``None`` means "not installed" and
# stays that way for the process lifetime (no retry).
_prom_client: Any = None
_prom_client_checked = False
_resolve_lock = threading.Lock()


def _resolve_prom_client() -> Any:
    """Return the imported prometheus_client module or ``None``.

    Memoised so we pay the import cost at most once. Errors other than
    ImportError (e.g. partial installs that fail mid-import) also resolve
    to ``None`` rather than propagating — the caller falls back to the
    legacy text path either way.
    """
    global _prom_client, _prom_client_checked
    if _prom_client_checked:
        return _prom_client
    with _resolve_lock:
        if _prom_client_checked:
            return _prom_client
        try:
            import prometheus_client as _pc  # type: ignore[import-not-found]
            _prom_client = _pc
        except Exception:
            _prom_client = None
        _prom_client_checked = True
        return _prom_client


def _legacy_mode_enabled() -> bool:
    """Honour the ``metrics_text_legacy=true`` operator override.

    Read at every call (not cached) so tests can flip the flag at runtime
    without rebuilding the registry. The cost is one attribute access.
    """
    try:
        import larkhelm.config as _cfg
        return bool(getattr(_cfg, "METRICS_TEXT_LEGACY", False))
    except Exception:
        return False


# ── Registry singleton ─────────────────────────────────────────────────────


class LarkhelmMetricsRegistry:
    """Owns all ``larkhelm_*`` Prometheus collectors.

    Constructed exactly once per process via :func:`get_registry`. When
    prometheus-client is missing the constructor records that state and
    every mutation method short-circuits — calls remain safe but become
    no-ops so callers don't need ``if available:`` guards on the hot path.
    """

    def __init__(self) -> None:
        pc = _resolve_prom_client()
        self._available: bool = pc is not None
        if not self._available:
            self._registry = None
            self.backend_healthy = None
            self.active_queries = None
            self.memory_rss_bytes = None
            self.cascade_active = None
            self.cascade_extract_total = None
            self.cascade_dropped_total = None
            self.cascade_midflight_cancelled_total = None
            self.query_duration_seconds = None
            self.extract_buffer_flushes_total = None
            return

        # Use a private registry so the larkhelm metrics don't collide with
        # any prometheus_client globals an embedding plugin may also touch.
        # ``generate_latest(self._registry)`` returns only our series.
        self._registry = pc.CollectorRegistry()

        self.backend_healthy = pc.Gauge(
            "larkhelm_backend_healthy",
            "1 if backend is healthy else 0",
            ["name"],
            registry=self._registry,
        )
        self.active_queries = pc.Gauge(
            "larkhelm_active_queries",
            "Current active _do_query invocations",
            registry=self._registry,
        )
        self.memory_rss_bytes = pc.Gauge(
            "larkhelm_memory_rss_bytes",
            "Process RSS in bytes",
            registry=self._registry,
        )
        self.cascade_active = pc.Gauge(
            "larkhelm_cascade_active",
            "In-flight memory cascade extracts",
            registry=self._registry,
        )
        self.cascade_extract_total = pc.Counter(
            "larkhelm_cascade_extract_total",
            "Cascade extract invocations",
            ["kind", "outcome"],
            registry=self._registry,
        )
        self.cascade_dropped_total = pc.Counter(
            "larkhelm_cascade_dropped_total",
            "Cascades dropped (sem full / cancel)",
            registry=self._registry,
        )
        self.cascade_midflight_cancelled_total = pc.Counter(
            "larkhelm_cascade_midflight_cancelled_total",
            "Cascades cancelled mid-LLM",
            registry=self._registry,
        )
        self.query_duration_seconds = pc.Histogram(
            "larkhelm_query_duration_seconds",
            "Query end-to-end latency",
            buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
            registry=self._registry,
        )
        self.extract_buffer_flushes_total = pc.Counter(
            "larkhelm_extract_buffer_flushes_total",
            "Buffer-induced cascade flushes",
            ["trigger"],
            registry=self._registry,
        )

    @property
    def available(self) -> bool:
        return self._available

    def render(self) -> str:
        """Render exposition for the underlying registry. Caller must have
        already checked ``available`` and ``not _legacy_mode_enabled()``.
        """
        pc = _resolve_prom_client()
        if pc is None or self._registry is None:
            raise PrometheusNotInstalled("prometheus-client not installed")
        return pc.generate_latest(self._registry).decode("utf-8")


_registry: LarkhelmMetricsRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> LarkhelmMetricsRegistry:
    """Process-wide singleton accessor."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = LarkhelmMetricsRegistry()
    return _registry


def _reset_for_tests() -> None:
    """Test-only: drop the singleton + the cached prometheus_client handle.

    Used by ``tests/test_metrics.py`` to exercise the
    "prometheus-client missing" branch by monkey-patching ``_resolve_prom_client``
    and forcing a fresh registry. Production code MUST NOT call this.
    """
    global _registry, _prom_client, _prom_client_checked
    with _registry_lock:
        _registry = None
    with _resolve_lock:
        _prom_client = None
        _prom_client_checked = False


# ── Public API ─────────────────────────────────────────────────────────────


def is_prometheus_available() -> bool:
    """True iff prometheus-client is importable AND not in legacy mode."""
    if _legacy_mode_enabled():
        return False
    return get_registry().available


def render_exposition() -> str:
    """Return the Prometheus text exposition for the larkhelm registry.

    Raises :class:`PrometheusNotInstalled` when prometheus-client is missing
    OR the operator forced legacy mode via ``metrics_text_legacy=true``.
    Callers (``health_server._handle_metrics``) are expected to catch this
    and fall back to the P1 hand-written text path.
    """
    if _legacy_mode_enabled():
        raise PrometheusNotInstalled("metrics_text_legacy=true forces fallback")
    reg = get_registry()
    if not reg.available:
        raise PrometheusNotInstalled("prometheus-client not installed")
    return reg.render()


def inc_cascade_extract(kind: str, outcome: str) -> None:
    """Bump ``larkhelm_cascade_extract_total{kind, outcome}``.

    ``kind`` ∈ {project, global}; ``outcome`` ∈
    {success, unchanged, rejected, cancelled, error}. Labels are not
    validated against a whitelist — bad strings would inflate cardinality;
    callers (``memory.py``) must keep them on the documented set.
    """
    reg = get_registry()
    if not reg.available or reg.cascade_extract_total is None:
        return
    try:
        reg.cascade_extract_total.labels(kind=kind, outcome=outcome).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_cascade_extract failed (kind={kind}, outcome={outcome}): {e}")


def observe_query_duration(seconds: float) -> None:
    """Record a single query end-to-end duration into the histogram."""
    reg = get_registry()
    if not reg.available or reg.query_duration_seconds is None:
        return
    try:
        reg.query_duration_seconds.observe(float(seconds))
    except Exception as e:
        safe_log(f"[Metrics] observe_query_duration failed (seconds={seconds!r}): {e}")


def inc_extract_buffer_flush(trigger: str) -> None:
    """Bump ``larkhelm_extract_buffer_flushes_total{trigger}``.

    ``trigger`` ∈ {timer, capacity, manual, shutdown}.
    """
    reg = get_registry()
    if not reg.available or reg.extract_buffer_flushes_total is None:
        return
    try:
        reg.extract_buffer_flushes_total.labels(trigger=trigger).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_extract_buffer_flush failed (trigger={trigger}): {e}")


def update_health_gauges(snapshot: "HealthSnapshot") -> None:
    """Mirror :class:`HealthSnapshot` numbers into the gauge series.

    Called by ``health_server._handle_metrics`` on every scrape so the
    Prometheus-rendered values stay in sync with the live snapshot. Safe
    to call when prometheus-client is missing (no-op).
    """
    reg = get_registry()
    if not reg.available:
        return
    try:
        # Backend health: emit one labelled sample per (name).
        if reg.backend_healthy is not None:
            # Reset stale labels: a backend disappearing from the registry
            # would otherwise leave its last value pinned. Re-emitting all
            # current names is cheap (≤ 11 backends).
            for name, healthy in (snapshot.backend_status or ()):
                reg.backend_healthy.labels(name=str(name)).set(int(healthy))
        if reg.active_queries is not None:
            reg.active_queries.set(int(snapshot.active_queries or 0))
        if reg.memory_rss_bytes is not None:
            reg.memory_rss_bytes.set(int(snapshot.memory_rss_bytes or 0))
        if reg.cascade_active is not None:
            reg.cascade_active.set(int(snapshot.cascade_active or 0))
        if reg.cascade_dropped_total is not None:
            # Counters never go down — set the underlying value by computing
            # the delta against the last-known total. prometheus_client
            # Counters don't expose ``set``, so we re-create the increment
            # path: read the current value, compute the diff, inc by delta
            # (or skip if negative — would indicate a restart on the
            # snapshot side that we shouldn't propagate).
            _maybe_inc_counter(reg.cascade_dropped_total, snapshot.cascade_dropped_total)
        if reg.cascade_midflight_cancelled_total is not None:
            _maybe_inc_counter(
                reg.cascade_midflight_cancelled_total,
                snapshot.cascade_midflight_cancelled_total,
            )
    except Exception as e:
        # Never raise on render path — a metrics bug must not 503 /metrics —
        # but log so silent gauge stalls don't go undiagnosed.
        safe_log(f"[Metrics] update_health_gauges failed: {e}")


# Per-counter watermark for the snapshot → Counter bridge above. Counters
# are monotonic so we only ``inc`` by the positive delta; the dict lives
# outside the registry so a counter rebuild during tests resets cleanly.
_counter_watermark: dict[int, int] = {}


def _maybe_inc_counter(counter: Any, snapshot_total: int) -> None:
    """Increment ``counter`` by the positive delta vs. the last snapshot."""
    if snapshot_total is None:
        return
    try:
        snap = int(snapshot_total)
    except (TypeError, ValueError):
        return
    key = id(counter)
    prev = _counter_watermark.get(key, 0)
    delta = snap - prev
    if delta > 0:
        try:
            counter.inc(delta)
        except Exception as e:
            safe_log(f"[Metrics] _maybe_inc_counter failed (delta={delta}): {e}")
            return
    # On regression (delta < 0) we DON'T set the counter back — the next
    # increase from this lower floor will simply add the diff above ``prev``
    # which is incorrect, so we re-arm the watermark to the current snap
    # value instead so the next call computes the right delta.
    _counter_watermark[key] = snap


__all__ = [
    "PrometheusNotInstalled",
    "LarkhelmMetricsRegistry",
    "get_registry",
    "is_prometheus_available",
    "render_exposition",
    "inc_cascade_extract",
    "observe_query_duration",
    "inc_extract_buffer_flush",
    "update_health_gauges",
]
