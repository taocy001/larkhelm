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
            self.llm_router_circuit_state = None
            self.recent_turns_cache_total = None
            self.memory_layer_cache_total = None
            self.doc_inject_cache_total = None
            self.file_downloads_total = None
            self.file_extract_errors_total = None
            self.tokens_total = None
            self.session_auto_reset_total = None
            self.sticky_context_evicted_total = None
            self.workspace_hint_total = None
            self.intent_feedback_total = None
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
        # P3 REQ-04: circuit-breaker state per LLM-router backend.
        # 0 = closed, 1 = half_open, 2 = open.
        self.llm_router_circuit_state = pc.Gauge(
            "larkhelm_llm_router_circuit_state",
            "LLM router circuit-breaker state (0=closed,1=half_open,2=open)",
            ["backend"],
            registry=self._registry,
        )
        # P1 context-cache counters (REQ-01..03). Each cache reports
        # ``outcome`` ∈ {hit, miss, evict, invalidate, bypass}; the layer
        # cache adds the layer name as a second label so global / project /
        # session / global_slots / project_sections can be tuned
        # independently.
        self.recent_turns_cache_total = pc.Counter(
            "larkhelm_recent_turns_cache_total",
            "Recent-turns cache outcomes",
            ["outcome"],
            registry=self._registry,
        )
        self.memory_layer_cache_total = pc.Counter(
            "larkhelm_memory_layer_cache_total",
            "Memory layer cache outcomes",
            ["layer", "outcome"],
            registry=self._registry,
        )
        self.doc_inject_cache_total = pc.Counter(
            "larkhelm_doc_inject_cache_total",
            "Doc inject cache outcomes",
            ["outcome"],
            registry=self._registry,
        )
        self.file_downloads_total = pc.Counter(
            "larkhelm_file_downloads_total",
            "File download attempts",
            ["ext", "outcome"],
            registry=self._registry,
        )
        self.file_extract_errors_total = pc.Counter(
            "larkhelm_file_extract_errors_total",
            "File content extraction errors",
            ["ext", "error_type"],
            registry=self._registry,
        )
        self.tokens_total = pc.Counter(
            "larkhelm_tokens_total",
            "Token usage per backend per kind",
            ["backend", "kind"],
            registry=self._registry,
        )
        # P0/P2 cache-bleed counters (design.md §3.4).
        self.session_auto_reset_total = pc.Counter(
            "larkhelm_session_auto_reset_total",
            "Claude session auto-resets by trigger reason",
            ["reason"],
            registry=self._registry,
        )
        self.sticky_context_evicted_total = pc.Counter(
            "larkhelm_sticky_context_evicted_total",
            "Sticky crew context evictions by reason",
            ["reason"],
            registry=self._registry,
        )
        # P3 REQ-03: workspace-hint segment telemetry. One outcome emitted
        # per handle_message that reaches the workspace injection block.
        self.workspace_hint_total = pc.Counter(
            "larkhelm_workspace_hint_total",
            "Workspace hint segment outcomes per message",
            ["outcome"],
            registry=self._registry,
        )
        # Phase D follow-up (May 2026): intent_feedback.jsonl writes by
        # signal_type. signal_type ∈ {force_chat, cancel_after_dispatch,
        # agent_reswitch, dispatch_failed, l1_gray_zone, l2_dispatched}.
        # Lets operators see whether the extended-signal collector is
        # producing the rates a downstream L1 trainer expects (>0 across
        # all live buckets).
        self.intent_feedback_total = pc.Counter(
            "larkhelm_intent_feedback_total",
            "intent_feedback.jsonl rows written by signal_type",
            ["signal_type"],
            registry=self._registry,
        )
        # P1-5a (review_summary §3 ROI table + W8/W9/W14): observability for
        # the intent classifier layers and cascade-backoff exhaustion. These
        # were the three blind spots the operator-facing review flagged as
        # blocking the Phase 5 default-on rollout (P1-4).
        self.intent_layer_total = pc.Counter(
            "larkhelm_intent_layer_total",
            "Intent resolution outcomes per layer",
            ["layer", "outcome"],
            registry=self._registry,
        )
        self.intent_l2_fallback_total = pc.Counter(
            "larkhelm_intent_l2_fallback_total",
            "L2 classifier fallback to chat",
            registry=self._registry,
        )
        self.cascade_backoff_exhausted_total = pc.Counter(
            "larkhelm_cascade_backoff_exhausted_total",
            "Memory cascade ExponentialBackoff retries exhausted (gave up)",
            registry=self._registry,
        )
        # SEC-CRIT-4 layer-2 sentinel heuristic outcomes. Bumped exactly
        # once per artifact validation that has ≥1 raw sentinel match
        # (layer-1 already missed at this point). ``outcome`` ∈
        # {hit_drop_ratio, hit_paranoid, abstain}. ``mode`` ∈
        # {enforced, observe} — observe means traffic bucket missed or
        # crew_sentinel_layer2_enabled=false, so the artifact was NOT
        # rejected; track it anyway so operators can calibrate thresholds
        # against real crew runs before enforcing.
        self.crew_validate_layer2_total = pc.Counter(
            "larkhelm_crew_validate_layer2_total",
            "Layer-2 sentinel heuristic outcomes",
            ["outcome", "mode"],
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


_CIRCUIT_STATE_CODE = {"closed": 0, "half_open": 1, "open": 2}


def set_llm_router_circuit_state(backend: str, state: str) -> None:
    """Reflect a :class:`CircuitBreaker` state into Prometheus (REQ-04).

    Recognised states: ``closed`` (0), ``half_open`` (1), ``open`` (2).
    Unknown strings collapse to 0 so a typo never poisons the gauge.
    Safe to call when prometheus-client is missing (no-op).
    """
    reg = get_registry()
    if not reg.available or reg.llm_router_circuit_state is None:
        return
    value = _CIRCUIT_STATE_CODE.get(state, 0)
    try:
        reg.llm_router_circuit_state.labels(backend=str(backend)).set(value)
    except Exception as e:
        safe_log(
            f"[Metrics] set_llm_router_circuit_state failed "
            f"(backend={backend}, state={state}): {e}"
        )


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


def inc_recent_turns_cache(outcome: str) -> None:
    """Bump ``larkhelm_recent_turns_cache_total{outcome}``.

    ``outcome`` ∈ {hit, miss, evict, invalidate, bypass}. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.recent_turns_cache_total is None:
        return
    try:
        reg.recent_turns_cache_total.labels(outcome=outcome).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_recent_turns_cache failed (outcome={outcome}): {e}")


def inc_memory_layer_cache(layer: str, outcome: str) -> None:
    """Bump ``larkhelm_memory_layer_cache_total{layer, outcome}``.

    ``layer`` ∈ {global, project, session, global_slots, project_sections};
    ``outcome`` ∈ {hit, miss, evict, invalidate, bypass}. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.memory_layer_cache_total is None:
        return
    try:
        reg.memory_layer_cache_total.labels(layer=layer, outcome=outcome).inc()
    except Exception as e:
        safe_log(
            f"[Metrics] inc_memory_layer_cache failed "
            f"(layer={layer}, outcome={outcome}): {e}"
        )


def inc_doc_inject_cache(outcome: str) -> None:
    """Bump ``larkhelm_doc_inject_cache_total{outcome}``.

    ``outcome`` ∈ {hit, miss, evict, invalidate, bypass, hit_with_age_hint}.
    ``hit_with_age_hint`` (P4 REQ-06) is emitted by
    :func:`larkhelm._context_cache.cached_doc_read_with_meta` instead of
    plain ``hit`` to mark hits that are surfaced to the user with an age
    annotation. Grafana queries should match `outcome=~"hit|hit_with_age_hint"`
    for total hit rate. Accepts arbitrary strings (no whitelist) — callers
    must keep the documented set to preserve cardinality discipline.
    Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.doc_inject_cache_total is None:
        return
    try:
        reg.doc_inject_cache_total.labels(outcome=outcome).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_doc_inject_cache failed (outcome={outcome}): {e}")


def inc_file_download(ext: str, outcome: str) -> None:
    """Bump ``larkhelm_file_downloads_total{ext, outcome}``.

    ``outcome`` ∈ {success, rejected, failed}. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.file_downloads_total is None:
        return
    try:
        reg.file_downloads_total.labels(ext=ext, outcome=outcome).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_file_download failed (ext={ext}, outcome={outcome}): {e}")


def inc_file_extract_error(ext: str, error_type: str) -> None:
    """Bump ``larkhelm_file_extract_errors_total{ext, error_type}``.

    ``error_type`` ∈ {missing_lib, decode, io, unknown}. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.file_extract_errors_total is None:
        return
    try:
        reg.file_extract_errors_total.labels(ext=ext, error_type=error_type).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_file_extract_error failed (ext={ext}, error_type={error_type}): {e}")


_TOKEN_KIND_TO_USAGE_KEY = (
    ("input",        "input_tokens"),
    ("output",       "output_tokens"),
    ("cache_read",   "cache_read"),
    ("cache_create", "cache_create"),
)


def inc_tokens(backend: str, usage: dict) -> None:
    """Bump ``larkhelm_tokens_total{backend, kind}`` for one query.

    ``usage`` must carry the keys produced by adapters / runners:
      - ``input_tokens``  (int)
      - ``output_tokens`` (int)
      - ``cache_read``    (int)
      - ``cache_create``  (int)

    Missing / negative / non-int values coerce to 0 silently — the call
    never raises and never increments by a negative number. Safe to call
    when prometheus-client is missing (registry no-op).
    """
    reg = get_registry()
    if not reg.available or reg.tokens_total is None:
        return
    if not isinstance(usage, dict):
        return
    for kind, key in _TOKEN_KIND_TO_USAGE_KEY:
        raw = usage.get(key, 0)
        try:
            value = max(0, int(raw or 0))
        except (TypeError, ValueError):
            value = 0
        if value == 0:
            continue
        try:
            reg.tokens_total.labels(backend=str(backend), kind=kind).inc(value)
        except Exception as e:
            safe_log(
                f"[Metrics] inc_tokens failed "
                f"(backend={backend}, kind={kind}, value={value}): {e}"
            )


def inc_session_auto_reset(reason: str) -> None:
    """Bump ``larkhelm_session_auto_reset_total{reason}``.

    ``reason`` ∈ {'cache_tokens', 'turns'}. Never raises. Safe to call
    when prometheus-client is missing (registry no-op).
    """
    reg = get_registry()
    if not reg.available or reg.session_auto_reset_total is None:
        return
    try:
        reg.session_auto_reset_total.labels(reason=str(reason)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_session_auto_reset failed (reason={reason}): {e}")


def inc_sticky_context_evicted(reason: str) -> None:
    """Bump ``larkhelm_sticky_context_evicted_total{reason}``.

    ``reason`` ∈ {'max_injections', 'ttl'}. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.sticky_context_evicted_total is None:
        return
    try:
        reg.sticky_context_evicted_total.labels(reason=str(reason)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_sticky_context_evicted failed (reason={reason}): {e}")


def inc_intent_feedback(signal_type: str) -> None:
    """Bump ``larkhelm_intent_feedback_total{signal_type}``.

    Called from :func:`larkhelm.agent_hub.intent_feedback._bump_metric`
    on every JSONL row write. ``signal_type`` is currently one of
    ``force_chat`` / ``cancel_after_dispatch`` / ``agent_reswitch`` /
    ``dispatch_failed`` / ``l1_gray_zone`` / ``l2_dispatched`` but the
    label isn't whitelisted here so plugins that introduce their own
    signal types can opt into the same observability without registry
    changes. Operators should monitor cardinality if a plugin starts
    emitting per-text labels.
    """
    reg = get_registry()
    if not reg.available or reg.intent_feedback_total is None:
        return
    try:
        reg.intent_feedback_total.labels(signal_type=str(signal_type)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_intent_feedback failed (signal_type={signal_type}): {e}")


def inc_intent_layer(layer: str, outcome: str) -> None:
    """Bump ``larkhelm_intent_layer_total{layer, outcome}``.

    P1-5a: Called from ``intent_router.resolve_intent`` at every layer
    decision. ``layer`` ∈ {explicit, l1, microlearn, l2_embedding, l2_llm,
    fallback}; ``outcome`` ∈ {hit, abstain, error}. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.intent_layer_total is None:
        return
    try:
        reg.intent_layer_total.labels(layer=str(layer), outcome=str(outcome)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_intent_layer failed (layer={layer},outcome={outcome}): {e}")


def inc_intent_l2_fallback() -> None:
    """Bump ``larkhelm_intent_l2_fallback_total`` once per L2→chat fallback.

    P1-5a: Counts every time the L2 path (embedding or LLM) couldn't
    resolve and ``_fallback("chat")`` was returned. Used to gauge whether
    raising L2 confidence thresholds is safe. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.intent_l2_fallback_total is None:
        return
    try:
        reg.intent_l2_fallback_total.inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_intent_l2_fallback failed: {e}")


def inc_cascade_backoff_exhausted() -> None:
    """Bump ``larkhelm_cascade_backoff_exhausted_total`` when ``ExponentialBackoff``
    in ``memory_extract_buffer`` / ``memory.cascade_extract`` gave up.

    P1-5a / W14: previously the backoff exhaustion only landed in
    ``_debug_log`` with no metric, so cascade losses were invisible to
    Prometheus alerts. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.cascade_backoff_exhausted_total is None:
        return
    try:
        reg.cascade_backoff_exhausted_total.inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_cascade_backoff_exhausted failed: {e}")


def inc_crew_validate_layer2(outcome: str, mode: str) -> None:
    """Bump ``larkhelm_crew_validate_layer2_total{outcome, mode}``.

    Called from ``crew/_runner.py:_validate_output_artifact`` exactly once
    per artifact whose pre-scrub content contains ≥ 1 raw sentinel match
    (layer-1 missed). Never raises. Safe when prometheus-client is absent.

    Args:
        outcome: ``"hit_drop_ratio"`` / ``"hit_paranoid"`` / ``"abstain"``
        mode: ``"enforced"`` (artifact rejected) / ``"observe"``
              (gray rollout missed; metric only, no rejection)
    """
    reg = get_registry()
    if not reg.available or reg.crew_validate_layer2_total is None:
        return
    try:
        reg.crew_validate_layer2_total.labels(
            outcome=str(outcome), mode=str(mode),
        ).inc()
    except Exception as e:
        safe_log(
            f"[Metrics] inc_crew_validate_layer2 failed "
            f"(outcome={outcome}, mode={mode}): {e}"
        )


def inc_workspace_hint(outcome: str) -> None:
    """Bump ``larkhelm_workspace_hint_total{outcome}``.

    ``outcome`` ∈ {injected_passive, injected_active_legacy, skipped_by_gate,
    skipped_empty}. Never raises. Safe to call when prometheus-client is
    missing (registry no-op).

    Invariant: each call to handle_message that reaches the workspace
    injection block (after command routing + crew sticky + early returns)
    emits exactly ONE outcome — never zero, never multiple.
    """
    reg = get_registry()
    if not reg.available or reg.workspace_hint_total is None:
        return
    try:
        reg.workspace_hint_total.labels(outcome=str(outcome)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_workspace_hint failed (outcome={outcome}): {e}")


# ── Module-level Counter aliases (PRD verify-command sugar) ───────────────
# PRD verify commands (cf. ``.crew_workspace/prd_criteria.json``) sometimes
# spell counter references as module-level constants (``WORKSPACE_HINT_TOTAL``
# etc.) rather than chasing through ``get_registry()``. Expose them via PEP-562
# ``__getattr__`` so the lookup stays lazy: when ``prometheus-client`` is
# missing the registry's attribute is ``None`` and the constant resolves to
# ``None`` (verify command exits 0 with ``print(None)`` — same byte-compat
# semantics as the existing ``inc_*`` helpers' no-op contract).
_REGISTRY_ALIASES = {
    "WORKSPACE_HINT_TOTAL":          "workspace_hint_total",
    "SESSION_AUTO_RESET_TOTAL":      "session_auto_reset_total",
    "STICKY_CONTEXT_EVICTED_TOTAL":  "sticky_context_evicted_total",
    "TOKENS_TOTAL":                  "tokens_total",
    "DOC_INJECT_CACHE_TOTAL":        "doc_inject_cache_total",
}


def __getattr__(name: str):
    """Lazy module attribute lookup for ``_REGISTRY_ALIASES``.

    Raises ``AttributeError`` for any other name so import-time typos still
    surface immediately rather than silently resolving to ``None``.
    """
    if name in _REGISTRY_ALIASES:
        return getattr(get_registry(), _REGISTRY_ALIASES[name], None)
    raise AttributeError(f"module 'larkhelm.metrics' has no attribute {name!r}")


__all__ = [
    "PrometheusNotInstalled",
    "LarkhelmMetricsRegistry",
    "get_registry",
    "is_prometheus_available",
    "render_exposition",
    "inc_cascade_extract",
    "observe_query_duration",
    "inc_extract_buffer_flush",
    "set_llm_router_circuit_state",
    "update_health_gauges",
    "inc_recent_turns_cache",
    "inc_memory_layer_cache",
    "inc_doc_inject_cache",
    "inc_file_download",
    "inc_file_extract_error",
    "inc_tokens",
    "inc_session_auto_reset",
    "inc_sticky_context_evicted",
    "inc_workspace_hint",
    "inc_intent_feedback",
    # Module-level Counter aliases (resolved lazily via __getattr__):
    "WORKSPACE_HINT_TOTAL",
    "SESSION_AUTO_RESET_TOTAL",
    "STICKY_CONTEXT_EVICTED_TOTAL",
    "TOKENS_TOTAL",
    "DOC_INJECT_CACHE_TOTAL",
]
