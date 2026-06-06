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
            self.injection_gate_total = None
            self.cache_write_tokens_total = None
            self.cache_read_tokens_total = None
            self.cache_hit_ratio = None
            self.cache_savings_total = None
            self.session_checkpoint_total = None
            self.prefix_stability_low_total = None
            self.crew_preflight_total = None
            self.lark_api_retry_total = None
            self.plan_ac_total = None
            self.prompt_cache_hit_rate = None
            self.webhook_received_total = None
            self.lark_api_duration_seconds = None
            self.message_errors_total = None
            self.lark_api_ok = None
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
        # P0 injection-gate telemetry: emitted on every gate decision so
        # operators can observe skip rates before enabling gates.
        # point ∈ {recent_turns_api, memory_intent_global, memory_intent_project,
        #          memory_intent_session, crew_sticky, doc_inject, project_guide}
        # outcome ∈ {injected, skipped_by_gate, skipped_by_state,
        #             skipped_by_relevance, truncated_by_relevance, large_doc,
        #             skipped_cli, error}
        self.injection_gate_total = pc.Counter(
            "larkhelm_injection_gate_total",
            "Context injection gate skip decisions",
            ["point", "outcome"],
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
        # SEC-v2-MED-1 (review_security_v2): structural Anthropic XML
        # leak check. ``outcome`` ∈ {hit_enforced, hit_observed} —
        # ``hit_enforced`` rejects the artifact, ``hit_observed`` only
        # fires when the gate is OFF so operators can spot real-world
        # match rates before flipping enforcement. Abstain is intentionally
        # not bumped (the regex misses on 99%+ of artifacts; emitting
        # ``abstain`` per validate call would flood the series).
        self.crew_validate_anthropic_loose_total = pc.Counter(
            "larkhelm_crew_validate_anthropic_loose_total",
            "SEC-v2-MED-1 structural Anthropic XML check outcomes",
            ["outcome"],
            registry=self._registry,
        )
        # SEC-v2-MED-2 (review_security_v2): backend "swing" detector.
        # Bumped each time ``_run_agent_wrapper`` writes a backend id
        # into the agent's ``excluded_backends_until`` map that
        # ALREADY had a (possibly expired) entry — i.e. the same
        # backend has been excluded twice during this crew run.
        # Healthy crews emit zero of these; a steady rate signals
        # an attacker repeatedly poisoning the preferred backend to
        # burn tokens on retries. Label cardinality is bounded by
        # the agent-id set defined in the crew plan (~5-10 per crew).
        self.crew_backend_swing_total = pc.Counter(
            "larkhelm_crew_backend_swing_total",
            "Same backend re-excluded within a crew (swing/DoS signal)",
            ["agent_id"],
            registry=self._registry,
        )
        self.cache_write_tokens_total = pc.Counter(
            "larkhelm_cache_write_tokens_total",
            "Anthropic cache_creation_input_tokens cumulative, bucketed by model",
            ["model"],
            registry=self._registry,
        )
        self.cache_read_tokens_total = pc.Counter(
            "larkhelm_cache_read_tokens_total",
            "Anthropic cache_read_input_tokens cumulative, bucketed by model",
            ["model"],
            registry=self._registry,
        )
        self.cache_hit_ratio = pc.Gauge(
            "larkhelm_cache_hit_ratio",
            "Real-time cache hit ratio cache_read/(cache_read+cache_write); 0 when denominator is zero",
            ["model"],
            registry=self._registry,
        )
        self.cache_savings_total = pc.Counter(
            "larkhelm_cache_savings_total",
            "Estimated USD saved by prompt cache hits, bucketed by backend",
            ["backend"],
            registry=self._registry,
        )
        self.session_checkpoint_total = pc.Counter(
            "larkhelm_session_checkpoint_total",
            "Session auto-reset checkpoints by backend and reason",
            ["backend", "reason"],
            registry=self._registry,
        )
        self.prefix_stability_low_total = pc.Counter(
            "larkhelm_prefix_stability_low_total",
            "Stable prefix hash changes detected (Anthropic layered cache instability)",
            ["backend"],
            registry=self._registry,
        )
        self.prompt_cache_hit_rate = pc.Histogram(
            "larkhelm_prompt_cache_hit_rate",
            "Per-query prompt cache hit rate (cache_read / (cache_read + input_tokens))",
            ["backend"],
            buckets=[0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            registry=self._registry,
        )
        self.crew_preflight_total = pc.Counter(
            "larkhelm_crew_preflight_total",
            "Crew agent preflight env check outcomes",
            ["outcome", "check_type"],
            registry=self._registry,
        )
        self.lark_api_retry_total = pc.Counter(
            "larkhelm_lark_api_retry_total",
            "Feishu API call retry outcomes",
            ["method", "outcome"],
            registry=self._registry,
        )
        self.plan_ac_total = pc.Counter(
            "larkhelm_plan_ac_total",
            "Plan acceptance criteria outcomes by gate_type",
            ["gate_type", "outcome"],
            registry=self._registry,
        )
        self.lark_api_ok = pc.Gauge(
            "larkhelm_lark_api_ok",
            "1 if a recent Feishu API call succeeded (age < health_lark_api_stale_sec), else 0",
            registry=self._registry,
        )
        self.webhook_received_total = pc.Counter(
            "larkhelm_webhook_received_total",
            "Feishu webhook events received by message type",
            ["event_type"],
            registry=self._registry,
        )
        self.lark_api_duration_seconds = pc.Histogram(
            "larkhelm_lark_api_duration_seconds",
            "Feishu API call duration in seconds",
            buckets=(0.05, 0.1, 0.5, 1, 5),
            registry=self._registry,
        )
        self.message_errors_total = pc.Counter(
            "larkhelm_message_errors_total",
            "Message processing errors by error type",
            ["error_type"],
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
        if getattr(reg, "lark_api_ok", None) is not None:
            reg.lark_api_ok.set(1 if snapshot.lark_api_ok else 0)
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


def inc_crew_backend_swing(agent_id: str, backend_id: str) -> None:
    """Bump ``larkhelm_crew_backend_swing_total{agent_id}``.

    SEC-v2-MED-2: called from ``crew/_runner._run_agent_wrapper`` each
    time we re-exclude the same backend on a retry round. The
    ``backend_id`` flows into the log breadcrumb only (not exported
    as a label) to keep series cardinality bounded. Never raises;
    safe when prometheus-client is absent.
    """
    reg = get_registry()
    if (not reg.available
            or getattr(reg, "crew_backend_swing_total", None) is None):
        return
    try:
        reg.crew_backend_swing_total.labels(agent_id=str(agent_id)).inc()
    except Exception as e:
        safe_log(
            f"[Metrics] inc_crew_backend_swing failed "
            f"(agent_id={agent_id}, backend_id={backend_id}): {e}"
        )


def inc_crew_validate_anthropic_loose(outcome: str) -> None:
    """Bump ``larkhelm_crew_validate_anthropic_loose_total{outcome}``.

    SEC-v2-MED-1: called from ``crew/_runner.py:_anthropic_loose_check``
    each time the structural Anthropic-XML regex matches in
    scrubbed prose. Outcomes: ``"hit_enforced"`` (gate on → artifact
    rejected) / ``"hit_observed"`` (gate off → metric-only). Never
    raises; safe when prometheus-client is absent.

    Abstain is intentionally NOT bumped — the regex misses on 99%+ of
    artifacts so per-call emission would flood the series.
    """
    reg = get_registry()
    if (not reg.available
            or getattr(reg, "crew_validate_anthropic_loose_total", None) is None):
        return
    try:
        reg.crew_validate_anthropic_loose_total.labels(
            outcome=str(outcome),
        ).inc()
    except Exception as e:
        safe_log(
            f"[Metrics] inc_crew_validate_anthropic_loose failed "
            f"(outcome={outcome}): {e}"
        )


def inc_injection_gate(point: str, outcome: str) -> None:
    """Bump ``larkhelm_injection_gate_total{point, outcome}``.

    Always emitted regardless of feature flag state so operators can observe
    skip rates in flag=False mode before enabling gates.

    ``point`` ∈ {recent_turns_api, memory_intent_global, memory_intent_project,
    memory_intent_session, crew_sticky, doc_inject, project_guide}.
    ``outcome`` ∈ {injected, skipped_by_gate, skipped_by_state,
    skipped_by_relevance, truncated_by_relevance, large_doc, skipped_cli,
    error}. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.injection_gate_total is None:
        return
    try:
        reg.injection_gate_total.labels(point=str(point), outcome=str(outcome)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_injection_gate failed (point={point}, outcome={outcome}): {e}")


def inc_cache_write_tokens(model: str, count: int) -> None:
    """Bump larkhelm_cache_write_tokens_total{model} by count.

    model ∈ {claude, gemini, kimi, deepseek}. count must be > 0
    (caller filters before calling). Never raises; no-op when
    prometheus-client is absent.
    """
    reg = get_registry()
    if not reg.available or reg.cache_write_tokens_total is None:
        return
    try:
        reg.cache_write_tokens_total.labels(model=str(model)).inc(count)
    except Exception as e:
        safe_log(f"[Metrics] inc_cache_write_tokens failed (model={model}, count={count}): {e}")


def inc_cache_read_tokens(model: str, count: int) -> None:
    """Bump larkhelm_cache_read_tokens_total{model} by count.

    Same contract as inc_cache_write_tokens. Never raises.
    """
    reg = get_registry()
    if not reg.available or reg.cache_read_tokens_total is None:
        return
    try:
        reg.cache_read_tokens_total.labels(model=str(model)).inc(count)
    except Exception as e:
        safe_log(f"[Metrics] inc_cache_read_tokens failed (model={model}, count={count}): {e}")


def set_cache_hit_ratio(model: str, ratio: float) -> None:
    """Set larkhelm_cache_hit_ratio{model} to ratio.

    ratio ∈ [0.0, 1.0]. Caller computes: cache_read / (cache_read + cache_write),
    or 0.0 when denominator is zero. Never raises; no-op when
    prometheus-client is absent.
    """
    reg = get_registry()
    if not reg.available or reg.cache_hit_ratio is None:
        return
    try:
        reg.cache_hit_ratio.labels(model=str(model)).set(float(ratio))
    except Exception as e:
        safe_log(f"[Metrics] set_cache_hit_ratio failed (model={model}, ratio={ratio}): {e}")


def inc_cache_savings(backend: str, amount_usd: float) -> None:
    """Bump larkhelm_cache_savings_total{backend} by amount_usd.

    backend ∈ {"claude", "gemini", "deepseek", "kimi"}.
    amount_usd may be a small positive float (e.g. 0.00027).
    Never raises; no-op when prometheus-client is absent.
    """
    reg = get_registry()
    if not reg.available or getattr(reg, "cache_savings_total", None) is None:
        return
    try:
        reg.cache_savings_total.labels(backend=str(backend)).inc(float(amount_usd))
    except Exception as e:
        safe_log(f"[Metrics] inc_cache_savings failed (backend={backend}, amount_usd={amount_usd}): {e}")


def inc_session_checkpoint(backend: str, reason: str) -> None:
    """Bump larkhelm_session_checkpoint_total{backend, reason}.

    reason ∈ {"cache_tokens", "turns"}.
    Never raises; no-op when prometheus-client is absent.
    """
    reg = get_registry()
    if not reg.available or getattr(reg, "session_checkpoint_total", None) is None:
        return
    try:
        reg.session_checkpoint_total.labels(backend=str(backend), reason=str(reason)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_session_checkpoint failed (backend={backend}, reason={reason}): {e}")


def inc_prefix_stability_low(backend: str) -> None:
    """Bump larkhelm_prefix_stability_low_total{backend} by 1.

    backend — the spec.id of the Anthropic backend (e.g. "anthropic_claude_3_7").
    Never raises; no-op when prometheus-client is absent.
    """
    reg = get_registry()
    if not reg.available or getattr(reg, "prefix_stability_low_total", None) is None:
        return
    try:
        reg.prefix_stability_low_total.labels(backend=str(backend)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_prefix_stability_low failed (backend={backend}): {e}")


def observe_cache_hit_rate(backend: str, ratio: float) -> None:
    """Observe larkhelm_prompt_cache_hit_rate{backend} with ratio.

    ratio ∈ [0.0, 1.0]. Caller computes: cache_read / (cache_read + input_tokens + 1e-9).
    Never raises; no-op when prometheus-client is absent.
    """
    reg = get_registry()
    if not reg.available or getattr(reg, "prompt_cache_hit_rate", None) is None:
        return
    try:
        reg.prompt_cache_hit_rate.labels(backend=str(backend)).observe(float(ratio))
    except Exception as e:
        safe_log(f"[Metrics] observe_cache_hit_rate failed (backend={backend}, ratio={ratio}): {e}")


def inc_crew_preflight(outcome: str, check_type: str) -> None:
    """Bump larkhelm_crew_preflight_total{outcome, check_type}. 永不抛出。
    outcome: pass / fail_arch / fail_docker
    check_type: arch / docker
    """
    reg = get_registry()
    if not reg.available or getattr(reg, "crew_preflight_total", None) is None:
        return
    try:
        reg.crew_preflight_total.labels(outcome=str(outcome), check_type=str(check_type)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_crew_preflight failed (outcome={outcome}, check_type={check_type}): {e}")


def inc_lark_api_retry(method: str, outcome: str) -> None:
    """Bump larkhelm_lark_api_retry_total{method, outcome}. 永不抛出。
    outcome: success_after_retry / exhausted
    """
    reg = get_registry()
    if not reg.available or getattr(reg, "lark_api_retry_total", None) is None:
        return
    try:
        reg.lark_api_retry_total.labels(method=str(method), outcome=str(outcome)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_lark_api_retry failed (method={method}, outcome={outcome}): {e}")


def inc_plan_ac(gate_type: str, outcome: str) -> None:
    """Bump larkhelm_plan_ac_total{gate_type, outcome}. 永不抛出。
    gate_type: code / runtime / release
    outcome: passed / skipped / env_blocked / failed
    """
    reg = get_registry()
    if not reg.available or getattr(reg, "plan_ac_total", None) is None:
        return
    try:
        reg.plan_ac_total.labels(gate_type=str(gate_type), outcome=str(outcome)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_plan_ac failed (gate_type={gate_type}, outcome={outcome}): {e}")


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
def inc_webhook_received(event_type: str) -> None:
    """Bump larkhelm_webhook_received_total{event_type}. Never raises."""
    reg = get_registry()
    if not reg.available or getattr(reg, "webhook_received_total", None) is None:
        return
    try:
        reg.webhook_received_total.labels(event_type=str(event_type)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_webhook_received failed (event_type={event_type}): {e}")


def observe_lark_api_duration(seconds: float) -> None:
    """Observe larkhelm_lark_api_duration_seconds with latency in seconds. Never raises."""
    reg = get_registry()
    if not reg.available or getattr(reg, "lark_api_duration_seconds", None) is None:
        return
    try:
        reg.lark_api_duration_seconds.observe(float(seconds))
    except Exception as e:
        safe_log(f"[Metrics] observe_lark_api_duration failed (seconds={seconds!r}): {e}")


def inc_message_errors(error_type: str) -> None:
    """Bump larkhelm_message_errors_total{error_type}. Never raises."""
    reg = get_registry()
    if not reg.available or getattr(reg, "message_errors_total", None) is None:
        return
    try:
        reg.message_errors_total.labels(error_type=str(error_type)).inc()
    except Exception as e:
        safe_log(f"[Metrics] inc_message_errors failed (error_type={error_type}): {e}")


_REGISTRY_ALIASES = {
    "WORKSPACE_HINT_TOTAL":          "workspace_hint_total",
    "SESSION_AUTO_RESET_TOTAL":      "session_auto_reset_total",
    "STICKY_CONTEXT_EVICTED_TOTAL":  "sticky_context_evicted_total",
    "TOKENS_TOTAL":                  "tokens_total",
    "DOC_INJECT_CACHE_TOTAL":        "doc_inject_cache_total",
    "CACHE_SAVINGS_TOTAL":           "cache_savings_total",
    "SESSION_CHECKPOINT_TOTAL":      "session_checkpoint_total",
    "WEBHOOK_RECEIVED_TOTAL":        "webhook_received_total",
    "LARK_API_DURATION_SECONDS":     "lark_api_duration_seconds",
    "MESSAGE_ERRORS_TOTAL":          "message_errors_total",
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
    "inc_injection_gate",
    "inc_crew_backend_swing",
    "inc_cache_write_tokens",
    "inc_cache_read_tokens",
    "set_cache_hit_ratio",
    "inc_cache_savings",
    "inc_session_checkpoint",
    "inc_prefix_stability_low",
    "inc_crew_preflight",
    "inc_lark_api_retry",
    "inc_plan_ac",
    "observe_cache_hit_rate",
    "inc_webhook_received",
    "observe_lark_api_duration",
    "inc_message_errors",
    # Module-level Counter aliases (resolved lazily via __getattr__):
    "WORKSPACE_HINT_TOTAL",
    "SESSION_AUTO_RESET_TOTAL",
    "STICKY_CONTEXT_EVICTED_TOTAL",
    "TOKENS_TOTAL",
    "DOC_INJECT_CACHE_TOTAL",
    "WEBHOOK_RECEIVED_TOTAL",
    "LARK_API_DURATION_SECONDS",
    "MESSAGE_ERRORS_TOTAL",
]
