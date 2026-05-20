"""larkhelm · BackendSpec dataclass and BackendRegistry singleton"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import threading

from larkhelm.log import _debug_log


# Assumed per-call usage when converting per-1k-token rates into a USD-per-call
# estimate for `TaskProfile.cost_ceiling` filtering. Tunable here; not currently
# user-configurable (single global heuristic is sufficient until we have per-task
# usage telemetry).
COST_CEILING_ASSUMED_IN_TOKENS  = 1000
COST_CEILING_ASSUMED_OUT_TOKENS = 500


@dataclasses.dataclass
class BackendSpec:
    id: str
    provider: str           # "claude_cli" | "gemini_cli" | "kimi_cli"
                            # | "anthropic_api" | "google_api" | "openai_compat_api"
                            # | "deepseek_api"
    display_name: str
    role: str               # "orchestrator" | "worker" | "cheap"
    tags: list[str]         # e.g. ["vision", "tools", "cheap", "fast"]
    command: str = ""       # CLI backends: executable path
    model: str = ""         # model name passed to CLI (e.g. --model, -m)
    api_key: str = ""       # API backends: resolved key (no ${} placeholders)
    base_url: str = ""      # API backends: custom endpoint
    instructions: str = ""  # extra system instructions injected for this backend when it acts as orchestrator
    capabilities: str = ""  # human-readable description used in orchestrator routing decisions
    extra_args: list[str] = dataclasses.field(default_factory=list)  # extra CLI flags (e.g. ["--thinking"])
    healthy: bool = True
    enabled: bool = True
    free_tier_ok: bool = True        # if False, probe rejects free-tier model responses (e.g. gemini *-preview)
    last_error: str | None = None  # last health_check failure reason
    description: str = ""  # natural-language description for L2 intent classifier prompt
    trigger_phrases: list[str] = dataclasses.field(default_factory=list)  # keyword triggers for L1 heuristic routing
    intent_examples: list[str] = dataclasses.field(default_factory=list)  # few-shot anchor examples for L2 prompt
    capability_scores: dict[str, float] = dataclasses.field(default_factory=dict)
    cost_per_1k_input:  float = 0.0
    cost_per_1k_output: float = 0.0
    latency_tier:       str = "medium"   # "instant" | "fast" | "medium" | "slow"

    # ── Runtime health tracking (in-memory only, NOT persisted) ──────────────
    # Real-call traffic and periodic probes both feed these; the unified
    # health-tick loop in config._start_recover_thread reads them to decide
    # whether to schedule a fresh probe (recover unhealthy / re-probe stale).
    # Reset to defaults on bridge restart — startup probes will repopulate.
    #
    # Two clocks: wall-clock (`*_at`) for human display ("3 分钟前");
    # monotonic (`*_mono`) for staleness math, so NTP correction or
    # `date -s` jumps don't poison probe cadence. ``failure_window`` stores
    # monotonic timestamps for the same reason.
    last_used_at:     float = 0.0   # epoch of last record_call_success/failure (display)
    last_probed_at:   float = 0.0   # epoch of last set_probe_result (display)
    last_used_mono:   float = 0.0   # monotonic of last record_call_success/failure (decisions)
    last_probed_mono: float = 0.0   # monotonic of last set_probe_result (decisions)
    failure_window:   list[float] = dataclasses.field(default_factory=list)  # TRANSIENT failure monotonic timestamps


def _resolve_env_vars(raw: str) -> str:
    """Expand ${ENV_VAR} placeholders using os.environ."""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return re.sub(r'\$\{([^}]+)\}', _replace, raw)


def _normalize_str_list(value: object) -> list[str]:
    """Normalize a config value into list[str].

    Accepts:
      - None / missing: returns [].
      - list: keeps str elements only, strips each, drops empties.
      - str: splits by '\n', then by ',' per segment; strips; drops empties.
      - other types: returns [] (debug-logged, never raises).
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str):
        out: list[str] = []
        for line in value.split("\n"):
            for seg in line.split(","):
                token = seg.strip()
                if token:
                    out.append(token)
        return out
    _debug_log(f"[BackendRegistry] _normalize_str_list: invalid type {type(value).__name__}, falling back to []")
    return []


# Normalize PRD hyphen-style provider names to internal underscore names
_PROVIDER_ALIASES: dict[str, str] = {
    "claude-cli":   "claude_cli",
    "gemini-cli":   "gemini_cli",
    "kimi-cli":     "kimi_cli",
    "claude-api":   "anthropic_api",
    "gemini-api":   "google_api",
    "openai-api":   "openai_compat_api",
    "deepseek-api": "deepseek_api",
}


class BackendRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}
        self._lock = threading.RLock()

    def load(self, specs: list[dict]) -> None:
        with self._lock:
            self._specs.clear()
            for s in specs:
                tags = s.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                raw_provider = s.get("provider", "")
                provider = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
                # Also pull api_key/base_url from legacy "extra" dict if not top-level
                extra = s.get("extra", {})

                raw_api_key = s.get("api_key", "") or extra.get("api_key", "")
                resolved_api_key = _resolve_env_vars(raw_api_key)
                enabled = s.get("enabled", True)
                # Unresolved placeholder → disable backend without logging (avoids leaking var names)
                if "${" in resolved_api_key:
                    # Unresolved placeholder — backend never usable
                    resolved_api_key = ""
                    enabled = False
                elif raw_api_key and not resolved_api_key:
                    # Key was specified but env var resolved to empty string
                    resolved_api_key = ""
                    enabled = False

                # Auto-infer role when not explicitly set
                raw_role = s.get("role", "")
                if raw_role:
                    role = raw_role
                elif provider.endswith("_cli") and "tools" in list(tags):
                    role = "orchestrator"
                else:
                    role = "worker"

                extra_args = s.get("extra_args", [])
                if isinstance(extra_args, str):
                    extra_args = extra_args.split()

                raw_description = s.get("description", "")
                description = raw_description if isinstance(raw_description, str) else str(raw_description)
                trigger_phrases = _normalize_str_list(s.get("trigger_phrases"))
                intent_examples = _normalize_str_list(s.get("intent_examples"))

                raw_caps = s.get("capability_scores", {})
                capability_scores: dict[str, float] = {}
                if isinstance(raw_caps, dict):
                    for k, v in raw_caps.items():
                        try:
                            capability_scores[str(k)] = float(v)
                        except (TypeError, ValueError):
                            _debug_log(f"[BackendRegistry] {s.get('id', '?')}: invalid capability_score {k}={v!r}, skipping")

                def _to_float(name: str, default: float = 0.0) -> float:
                    val = s.get(name, default)
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        _debug_log(f"[BackendRegistry] {s.get('id', '?')}: invalid {name}={val!r}, falling back to {default}")
                        return default

                cost_per_1k_input  = _to_float("cost_per_1k_input")
                cost_per_1k_output = _to_float("cost_per_1k_output")
                raw_latency = s.get("latency_tier", "medium")
                latency_tier = raw_latency if raw_latency in ("instant", "fast", "medium", "slow") else "medium"

                spec = BackendSpec(
                    id=s["id"],
                    provider=provider,
                    display_name=s.get("display_name", s["id"]),
                    role=role,
                    tags=list(tags),
                    command=s.get("command", "") or extra.get("command", ""),
                    model=s.get("model", ""),
                    api_key=resolved_api_key,
                    base_url=s.get("base_url", "") or extra.get("base_url", ""),
                    instructions=s.get("instructions", ""),
                    capabilities=s.get("capabilities", ""),
                    extra_args=list(extra_args),
                    healthy=True,
                    enabled=enabled,
                    free_tier_ok=bool(s.get("free_tier_ok", True)),
                    description=description,
                    trigger_phrases=trigger_phrases,
                    intent_examples=intent_examples,
                    capability_scores=capability_scores,
                    cost_per_1k_input=cost_per_1k_input,
                    cost_per_1k_output=cost_per_1k_output,
                    latency_tier=latency_tier,
                )
                self._specs[spec.id] = spec

    def health_check(self) -> None:
        with self._lock:
            for spec in self._specs.values():
                if not spec.enabled:
                    continue
                spec.healthy = True   # reset before re-checking
                spec.last_error = None
                try:
                    if spec.provider.endswith("_cli"):
                        if not spec.command or not shutil.which(spec.command):
                            spec.healthy = False
                            spec.last_error = f"command not found: {spec.command!r}"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                    elif spec.provider == "anthropic_api":
                        try:
                            import anthropic  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            spec.last_error = "anthropic SDK not installed"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                            continue
                        if not spec.api_key or "${" in spec.api_key:
                            spec.healthy = False
                            spec.last_error = "api_key missing or unresolved"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                    elif spec.provider == "google_api":
                        try:
                            import google.genai  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            spec.last_error = "google-genai SDK not installed"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                            continue
                        if not spec.api_key or "${" in spec.api_key:
                            spec.healthy = False
                            spec.last_error = "api_key missing or unresolved"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                    elif spec.provider == "openai_compat_api":
                        try:
                            import openai  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            spec.last_error = "openai SDK not installed"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                            continue
                        if not spec.api_key or "${" in spec.api_key:
                            spec.healthy = False
                            spec.last_error = "api_key missing or unresolved"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                    elif spec.provider == "deepseek_api":
                        try:
                            import requests  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            spec.last_error = "requests package not installed"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                            continue
                        if not spec.api_key or "${" in spec.api_key:
                            spec.healthy = False
                            spec.last_error = "api_key missing or unresolved"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                except Exception as e:
                    spec.healthy = False
                    spec.last_error = str(e)
                    _debug_log(f"[BackendRegistry] {spec.id}: health_check error: {e}")

    def get(self, id: str) -> BackendSpec | None:
        with self._lock:
            return self._specs.get(id)

    def get_by_tag(self, tags: list[str], prefer_role: str = "") -> BackendSpec | None:
        """Return first BackendSpec matching ALL tags, healthy+enabled.

        If prefer_role is set, a matching spec with that role is returned first;
        falls back to any matching spec if no preferred-role match exists.
        """
        with self._lock:
            candidates = [
                s for s in self._specs.values()
                if s.enabled and s.healthy and all(t in s.tags for t in tags)
            ]
        if not candidates:
            return None
        if prefer_role:
            preferred = [s for s in candidates if s.role == prefer_role]
            if preferred:
                return preferred[0]
        return candidates[0]

    def get_orchestrator(self) -> BackendSpec | None:
        """Return first role='orchestrator' healthy+enabled BackendSpec."""
        with self._lock:
            for spec in self._specs.values():
                if spec.role == "orchestrator" and spec.healthy and spec.enabled:
                    return spec
        return None

    def get_orchestrator_chain(self) -> list[BackendSpec]:
        """Return ordered list of healthy+enabled backends for failover.

        Priority:
          1. role=orchestrator + 'tools' in tags + CLI provider
          2. role=orchestrator + 'tools' in tags + API provider
          3. role=orchestrator (any)
          4. any healthy+enabled (fallback)
        Deduplication via seen-set; all groups concatenated.
        """
        with self._lock:
            candidates = [s for s in self._specs.values() if s.healthy and s.enabled]

        seen: set[str] = set()
        result: list[BackendSpec] = []

        groups = [
            [s for s in candidates
             if s.role == "orchestrator" and "tools" in s.tags and s.provider.endswith("_cli")],
            [s for s in candidates
             if s.role == "orchestrator" and "tools" in s.tags and not s.provider.endswith("_cli")],
            [s for s in candidates if s.role == "orchestrator"],
            candidates,
        ]
        for group in groups:
            for spec in group:
                if spec.id not in seen:
                    seen.add(spec.id)
                    result.append(spec)
        return result

    def recover_check(self) -> None:
        """[DEPRECATED] Re-probe only healthy=False backends.

        Production health-tracking moved to the unified tick loop in
        ``config._start_recover_thread`` (which uses real probes via
        ``model_probe.probe_spec`` and tracks ``last_used_mono`` /
        ``last_probed_mono`` to schedule recovery + idle re-validation).
        This method is preserved only for legacy test fixtures
        (``tests/test_phase4.py``). May be removed in a future cleanup.

        Does NOT reset healthy=True backends (avoids unnecessary overhead).
        Thread-safe. Exceptions per-spec are caught and logged.
        """
        with self._lock:
            to_check = [s for s in self._specs.values() if not s.healthy and s.enabled]

        for spec in to_check:
            try:
                recovered = False
                if spec.provider.endswith("_cli"):
                    if spec.command and shutil.which(spec.command):
                        recovered = True
                elif spec.provider == "anthropic_api":
                    try:
                        import anthropic  # noqa: F401
                        if spec.api_key and "${" not in spec.api_key:
                            recovered = True
                    except ImportError:
                        pass
                elif spec.provider == "google_api":
                    try:
                        import google.genai  # noqa: F401
                        if spec.api_key and "${" not in spec.api_key:
                            recovered = True
                    except ImportError:
                        pass
                elif spec.provider == "openai_compat_api":
                    try:
                        import openai  # noqa: F401
                        if spec.api_key and "${" not in spec.api_key:
                            recovered = True
                    except ImportError:
                        pass
                elif spec.provider == "deepseek_api":
                    try:
                        import requests  # noqa: F401
                        if spec.api_key and "${" not in spec.api_key:
                            recovered = True
                    except ImportError:
                        pass

                if recovered:
                    with self._lock:
                        if not spec.healthy:  # re-check: could have failed again since snapshot
                            spec.healthy = True
                            spec.last_error = None
                    _debug_log(f"[BackendRegistry] recovered: {spec.id}")
                else:
                    _debug_log(f"[BackendRegistry] still unhealthy: {spec.id}")
            except Exception as e:
                _debug_log(f"[BackendRegistry] recover_check error for {spec.id}: {e}")

    def set_probe_result(self, spec_id: str, ok: bool | None, error: str = "") -> None:
        """Update healthy/last_error/last_probed_at for a spec after a probe completes.

        Called by both the startup probe (``model_probe.run_probes_async``) and
        the unified health-tick loop in ``config._start_recover_thread``.
        Thread-safe.

        ``ok`` semantics:
          * True — probe confirmed reachable: flip ``healthy=True``, clear error.
          * False — probe failed clearly: flip ``healthy=False``, record error.
          * None — probe result indeterminate (e.g. subprocess timeout). Do
            NOT mutate ``healthy``. Real-traffic ``record_call_failure`` is
            the authoritative health signal for these cases; the probe just
            updates ``last_probed_at`` so the staleness check knows the
            tick ran. Added to fix the Gemini/Claude probe false-positive
            where ``subprocess.TimeoutExpired`` returned ``True`` ("slow
            start = model exists") and silently masked real outages.

        Note: does NOT clear ``failure_window``. A successful probe only proves
        the backend's auth+connectivity work right now — it does not prove the
        real workload pattern works. If we cleared on success we'd mask a
        flapping backend (real call fails, probe succeeds, real call fails again
        within seconds). Time-based pruning in record_call_failure is enough.
        """
        import time as _time
        with self._lock:
            spec = self._specs.get(spec_id)
            if spec is None:
                return
            if ok is None:
                # Indeterminate — bookkeeping only, no healthy flip. Tag the
                # error string so /status shows the timeout reason next to
                # the last-probed timestamp.
                spec.last_probed_at = _time.time()
                spec.last_probed_mono = _time.monotonic()
                if error:
                    spec.last_error = f"probe indeterminate: {error}"
                return
            spec.healthy = ok
            spec.last_error = error if not ok else None
            spec.last_probed_at = _time.time()
            spec.last_probed_mono = _time.monotonic()

    def record_call_success(self, spec_id: str) -> None:
        """Real-traffic success: confirm healthy + bump last_used_at.

        Call this from ``backend_cli.run_*`` and ``backend_api.run_*`` after a
        normal return. Cheap (registry lock held briefly). Does NOT update
        ``last_probed_at`` — successful traffic counts as "recently validated"
        for the idle-stale check, but the probe-cadence accounting stays separate.

        Note: does NOT clear ``failure_window``. We keep the window so a backend
        flapping fail/success/fail/success within the window will still trip the
        threshold. Old timestamps prune themselves out via the window cutoff
        the next time record_call_failure runs.
        """
        import time as _time
        with self._lock:
            spec = self._specs.get(spec_id)
            if spec is None:
                return
            spec.healthy = True
            spec.last_error = None
            spec.last_used_at = _time.time()
            spec.last_used_mono = _time.monotonic()

    def record_call_failure(
        self,
        spec_id: str,
        err: str,
        category: str | None = None,
        transient_window_sec: float = 600.0,
        transient_threshold: int = 3,
    ) -> str:
        """Real-traffic failure: classify the error and update health accordingly.

        Returns the category string so callers can branch on it (e.g. for
        retry logic).

        * USER_CANCEL / TIMEOUT  → no change to health (still bumps last_used_at)
        * AUTH / QUOTA / MODEL_NOT_FOUND → flip healthy=False immediately
        * TRANSIENT → append to sliding window; flip only after ``transient_threshold``
          hits within ``transient_window_sec`` seconds

        ``category`` may be supplied by caller; otherwise classified from ``err``.
        ``transient_window_sec`` / ``transient_threshold`` are passed in by the
        caller (read from ``_cfg.BACKEND_TRANSIENT_*``) so the registry stays
        decoupled from the config module — avoids a circular import.
        """
        import time as _time
        from larkhelm.health_signals import classify_error, is_no_op, is_instant_unhealthy

        if category is None:
            category = classify_error(err)

        with self._lock:
            spec = self._specs.get(spec_id)
            if spec is None:
                return category
            now_wall = _time.time()
            now_mono = _time.monotonic()
            spec.last_used_at = now_wall
            spec.last_used_mono = now_mono

            if is_no_op(category):
                return category
            if is_instant_unhealthy(category):
                spec.healthy = False
                spec.last_error = f"{category.lower()}: {str(err)[:200]}"
                return category

            # TRANSIENT — sliding window. Use monotonic clock so NTP correction
            # or `date -s` jumps don't poison the cutoff math (a backwards
            # wall-clock jump would make every old timestamp look "in window"
            # and instantly trip the threshold).
            spec.failure_window.append(now_mono)
            cutoff = now_mono - max(transient_window_sec, 1.0)
            spec.failure_window[:] = [t for t in spec.failure_window if t >= cutoff]
            if len(spec.failure_window) >= max(transient_threshold, 1):
                spec.healthy = False
                spec.last_error = f"transient×{len(spec.failure_window)}: {str(err)[:200]}"
            return category

    def all_enabled(self) -> list[BackendSpec]:
        with self._lock:
            return [s for s in self._specs.values() if s.enabled]

    def snapshot(self, enabled_only: bool = False) -> list[BackendSpec]:
        """Return a one-shot copy of registered specs taken under the lock.

        Use this from external readers (e.g. ``/status``) instead of reaching
        into ``_lock`` / ``_specs`` directly. The returned list is yours to
        iterate freely; subsequent mutations to the registry do not affect it.
        Each :class:`BackendSpec` itself is shared (same instance), so reading
        per-spec mutable fields (``healthy``, ``last_error``, ``failure_window``)
        without re-acquiring the lock is technically racy but acceptable for
        UI purposes — values may be one tick stale.
        """
        with self._lock:
            specs = list(self._specs.values())
        if enabled_only:
            specs = [s for s in specs if s.enabled]
        return specs

    def rank_for_task(self, profile) -> list[BackendSpec]:
        """Rank healthy+enabled BackendSpecs for the given TaskProfile.

        Score = Σ profile.required_capabilities[k] * spec.capability_scores.get(k, 0.0).
        Falls back to len(set(profile.required_capabilities) & set(spec.tags))
        when no spec exposes capability_scores (preserves phase4 semantics
        for legacy configs — see NFR-COMPAT-02).
        Secondary sort: latency_tier preference, then cost_per_1k_output asc.

        ``profile`` is duck-typed via ``getattr`` so we don't need to import
        ``TaskProfile`` (which would create a circular dependency).
        """
        with self._lock:
            candidates = [s for s in self._specs.values() if s.enabled and s.healthy]

        required = dict(getattr(profile, "required_capabilities", {}) or {})
        latency_pref = getattr(profile, "latency_pref", "medium")
        require_tools = bool(getattr(profile, "require_tools", False))
        require_vision = bool(getattr(profile, "require_vision", False))
        cost_ceiling = getattr(profile, "cost_ceiling", None)

        if require_tools:
            candidates = [s for s in candidates if "tools" in s.tags]
        if require_vision:
            candidates = [s for s in candidates if "vision" in s.tags]
        if cost_ceiling is not None:
            # cost_ceiling is USD-per-call (see TaskProfile.cost_ceiling).
            # Estimate per-call cost using the assumed usage in COST_CEILING_ASSUMED_*.
            # Backends with zero per-1k rates (e.g., local/free) compute to 0 USD
            # and pass naturally — no special bypass needed.
            def _per_call_cost(spec: BackendSpec) -> float:
                return (
                    spec.cost_per_1k_input  * (COST_CEILING_ASSUMED_IN_TOKENS  / 1000.0)
                    + spec.cost_per_1k_output * (COST_CEILING_ASSUMED_OUT_TOKENS / 1000.0)
                )
            candidates = [s for s in candidates if _per_call_cost(s) <= cost_ceiling]

        any_caps = any(s.capability_scores for s in candidates)

        def _score(spec: BackendSpec) -> float:
            if any_caps:
                return sum(
                    float(weight) * spec.capability_scores.get(name, 0.0)
                    for name, weight in required.items()
                )
            return float(len(set(required.keys()) & set(spec.tags)))

        _LATENCY_RANK = {"instant": 0, "fast": 1, "medium": 2, "slow": 3}

        def _latency_distance(spec: BackendSpec) -> int:
            return abs(
                _LATENCY_RANK.get(spec.latency_tier, 2)
                - _LATENCY_RANK.get(latency_pref, 2)
            )

        def _sort_key(spec: BackendSpec):
            return (
                -_score(spec),
                _latency_distance(spec),
                spec.cost_per_1k_output,
                spec.cost_per_1k_input,
                spec.id,
            )

        return sorted(candidates, key=_sort_key)


# Singleton — populated by config._init_runtime()
BACKEND_REGISTRY: BackendRegistry = BackendRegistry()
