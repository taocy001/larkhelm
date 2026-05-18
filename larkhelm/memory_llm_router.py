"""larkhelm · LLM-driven memory router (Phase D / Phase 3).

The third retriever tier on top of KeywordRetriever / HybridRetriever.
Only used for genuinely complex tasks (`/crew`, `/dev complex`) where the
extra LLM round-trip is worth the precision boost.

Pipeline
--------
1. Underlying retriever (keyword or hybrid) yields ``pool_size`` candidates
   (default ``top_k × multiplier``, capped at ``LLM_ROUTER_MAX_POOL = 30``).
2. We serialise a compact metadata view of each candidate:
   ``{id, layer, kind, title, importance, char_len, body_head, stale}``.
3. A cheap-tier LLM (``_run_one_shot(prefer_cheap=True)``) is asked to
   return ``{"selected_ids": [...], "reasoning": "..."}`` in JSON.
4. Successful response → re-rank the candidates so that the LLM-selected
   ids come first (in the order returned), padded with the underlying
   retriever's ordering for any non-selected ids. Truncated to ``top_k``.
5. Any failure (JSON parse, network, budget, malformed ids) → drop back to
   the underlying retriever's output. Audit record always written.

Budget protection
-----------------
* Per ``chat_id`` rate limit: at most
  ``memory_llm_router_max_per_chat_per_min`` calls per rolling 60-second
  window (default 3). Over → cache miss falls back to underlying retriever
  with ``skipped_reason="rate_limit"``.
* Whole-feature time-to-live cache: identical ``(query_hash, slice_set_hash)``
  inside ``memory_llm_router_cache_ttl_sec`` (default 300s) → reuse the
  previous LLM verdict, ``cache_hit=True``. Cache size capped at 256
  entries (LRU-ish via insertion-order dict).

Safety
------
* All slice metadata is truncated (titles ≤120 chars, body_head ≤120 chars)
  before being sent to the cheap LLM — the LLM never sees full slice
  bodies, which limits both prompt-injection blast radius and token cost.
* Module imports only stdlib + larkhelm submodules. The cheap LLM call
  is gated behind ``_resolve_cheap_caller`` so the module loads cleanly
  even when ``memory._run_one_shot`` isn't importable (eg. early test
  bootstrap).
* Never raises into the caller: all exceptions are caught and logged with
  the underlying-retriever result returned unchanged.

The router does NOT decide whether to invoke the LLM — that's
``resolve_actual_mode`` (Phase 3 addition: returns ``"llm_router"`` mode
only when ``memory_llm_router_enabled=true`` and the chat is in the
``memory_llm_router_traffic`` bucket).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Callable, Iterable

from larkhelm.log import _debug_log
from larkhelm.memory_slice import (
    InjectionPolicy,
    MemorySlice,
    MemoryRetriever,
    RetrievalRequest,
    ScoredSlice,
)


# ── Module-level constants ────────────────────────────────────────────────

LLM_ROUTER_MAX_POOL = 30
"""Hard cap on candidates handed to the cheap LLM. Prevents pathological
token cost when ``top_k × multiplier`` would otherwise balloon."""

_TITLE_TRUNC = 120
_BODY_HEAD_TRUNC = 120
_CACHE_MAX_ENTRIES = 256
_RATE_WINDOW_SEC = 60.0
_DEFAULT_RATE_LIMIT = 3
_DEFAULT_CACHE_TTL = 300

# JSON extraction regex — handles the common "LLM wraps result in ```json"
# pattern as well as bare object responses. Greedy match → grab the whole
# braced block. Anchored at first ``{`` to avoid false-positive matches in
# the reasoning text.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# ── In-process state (per bridge instance) ────────────────────────────────

# OrderedDict implements LRU semantics via ``move_to_end`` and bounded size
# via ``popitem(last=False)``. Keys: ``(query_hash, slice_set_hash)``.
# Values: ``(ts, selected_ids_tuple)``.
_cache_lock = threading.Lock()
_cache: "OrderedDict[tuple[str, str], tuple[float, tuple[str, ...]]]" = OrderedDict()

# Per-chat rate limit: chat_id → deque[float] of recent call timestamps.
# Trimmed each access; total memory bounded by chat count × rate_limit.
_rate_lock = threading.Lock()
_rate_window: dict[str, deque[float]] = {}


def _config_int(key: str, default: int) -> int:
    """Read a positive integer config value with a safe default."""
    try:
        import larkhelm.config as _cfg
        cfg = getattr(_cfg, "config", {}) or {}
        v = int(cfg.get(key, default) or default)
        return max(0, v)
    except Exception:
        return default


def _config_float(key: str, default: float) -> float:
    try:
        import larkhelm.config as _cfg
        cfg = getattr(_cfg, "config", {}) or {}
        v = float(cfg.get(key, default) or default)
        return max(0.0, v)
    except Exception:
        return default


# ── Cache helpers ─────────────────────────────────────────────────────────


def _cache_key(query: str, candidate_ids: Iterable[str]) -> tuple[str, str]:
    """Build a stable cache key from query + candidate set.

    Cache hits require BOTH the query AND the candidate set to match —
    different candidate sets imply different relevant data, so reusing the
    verdict would be wrong.

    Idempotent under duplicate ids: passing ``["a", "a", "b"]`` yields the
    same key as ``["a", "b"]`` (set-semantics — review SF-02). Without
    this normalisation, a caller that drifts from passing a set to passing
    a list would silently cache-miss every time.
    """
    q_hash = hashlib.md5((query or "").encode("utf-8")).hexdigest()[:16]
    # Dedup + sort to make the hash insensitive to the underlying
    # retriever's ordering AND to duplicate-id input — what matters is
    # the SET of candidates, not their order or multiplicity.
    set_hash = hashlib.md5(
        ",".join(sorted(set(candidate_ids))).encode("utf-8")
    ).hexdigest()[:16]
    return (q_hash, set_hash)


def _cache_get(key: tuple[str, str], ttl_sec: float) -> tuple[str, ...] | None:
    """Return cached selected_ids if present and within TTL; else None.

    Side-effect: refreshes LRU position on hit (move_to_end).
    """
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, ids = entry
        if (time.monotonic() - ts) > ttl_sec:
            # Expired — proactively evict to keep _cache small.
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return ids


def _cache_put(key: tuple[str, str], ids: tuple[str, ...]) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), ids)
        _cache.move_to_end(key)
        # LRU eviction on overflow.
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)


def _cache_clear_for_tests() -> None:
    """Test-only: reset both cache and rate-limit state.

    Called by the unit-test fixtures so tests run hermetically. Never
    called in production.
    """
    with _cache_lock:
        _cache.clear()
    with _rate_lock:
        _rate_window.clear()


# ── Rate limiter ──────────────────────────────────────────────────────────


def _rate_check_and_record(chat_id: str, limit_per_min: int) -> bool:
    """Return True iff this chat can make a NEW LLM-router call right now.

    A True result also records the timestamp; callers should not "rollback"
    on subsequent failure paths — the rate limit is conservative on
    purpose to keep cheap-backend cost bounded under bug-driven hot loops.

    ``limit_per_min <= 0`` disables the limit entirely (always True), used
    by tests that explicitly want to skip rate gating.
    """
    if limit_per_min <= 0:
        return True
    now = time.monotonic()
    with _rate_lock:
        window = _rate_window.setdefault(chat_id, deque())
        # Trim entries older than the rolling window.
        cutoff = now - _RATE_WINDOW_SEC
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit_per_min:
            return False
        window.append(now)
        return True


def _rate_remaining(chat_id: str, limit_per_min: int) -> int:
    """How many more calls this chat can make right now (for diagnostics)."""
    if limit_per_min <= 0:
        return -1
    now = time.monotonic()
    with _rate_lock:
        window = _rate_window.get(chat_id)
        if not window:
            return limit_per_min
        cutoff = now - _RATE_WINDOW_SEC
        active = sum(1 for ts in window if ts >= cutoff)
        return max(0, limit_per_min - active)


# ── Prompt construction ───────────────────────────────────────────────────


def _serialise_slice_meta(s: MemorySlice) -> dict[str, Any]:
    """Compact metadata view of a slice; what the LLM sees."""
    body_head = (s.body or "").replace("\n", " ").strip()[:_BODY_HEAD_TRUNC]
    return {
        "id": s.id,
        "layer": s.layer,
        "kind": s.kind,
        "title": (s.title or "")[:_TITLE_TRUNC],
        "importance": round(float(s.importance or 0.0), 2),
        "char_len": int(s.char_len or len(s.body or "")),
        "body_head": body_head,
        "stale": bool(getattr(s, "stale", False)),
    }


def _build_router_prompt(
    request: RetrievalRequest,
    policy: InjectionPolicy,
    candidates: list[MemorySlice],
    *,
    top_k: int,
) -> str:
    """Build the JSON-mode prompt for the cheap router LLM."""
    metas = [_serialise_slice_meta(s) for s in candidates]
    head = (request.query or "")[:240]
    return (
        "You are a memory router. Pick the most relevant memory slices "
        f"for the user query so a downstream {policy.agent_type!s} agent "
        "can answer effectively.\n\n"
        f"User query (truncated 240 chars): {head!r}\n"
        f"Agent type: {policy.agent_type}\n"
        f"Sub-intent: {request.sub_intent or '(none)'}\n"
        f"Complexity: {request.complexity}\n\n"
        f"Candidates ({len(metas)} total — pick up to {top_k}, fewer is fine):\n"
        f"{json.dumps(metas, ensure_ascii=False, indent=2)}\n\n"
        "Return ONLY a JSON object on a single line:\n"
        '{"selected_ids": ["id1", "id2", ...], "reasoning": "one short '
        'sentence explaining the cut"}\n\n'
        "Constraints:\n"
        f"- selected_ids must be a non-empty subset of the candidate ids, in priority order\n"
        f"- selected_ids length <= {top_k}\n"
        "- Prefer recent + high-importance + non-stale slices when relevance ties\n"
        "- Output ONLY the JSON object. No prose, no markdown fences."
    )


# ── LLM response parser ───────────────────────────────────────────────────


def _parse_llm_response(text: str, candidate_ids: set[str]) -> tuple[str, ...] | None:
    """Extract ``selected_ids`` from an LLM response.

    Returns ``None`` on any parse error. Validates that returned ids are a
    subset of ``candidate_ids`` — silently drops anything else.
    Empty subset is treated as a parse failure (the LLM was supposed to
    return at least one id; an empty list is more likely a misunderstood
    instruction than a deliberate "no memory needed" verdict).
    """
    if not text:
        return None
    try:
        # Strip code fences if the LLM wrapped the JSON.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Drop the first line (``` or ```json) and any trailing fence.
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned[: -3]
        # Anchored regex — find the first complete brace block.
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            return None
        data = json.loads(m.group(0))
        ids_field = data.get("selected_ids")
        if not isinstance(ids_field, list):
            return None
        result = tuple(
            sid for sid in (str(x) for x in ids_field) if sid in candidate_ids
        )
        return result if result else None
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        _debug_log(f"[LLMRouter] parse failed: {e}")
        return None


# ── Circuit breaker (P3 REQ-04) ────────────────────────────────────────────
#
# A single module-level CircuitBreaker guards every cheap-backend call out of
# this router. Five consecutive failures (default; see
# ``llm_router_circuit_failures``) opens the circuit and short-circuits
# subsequent calls until ``llm_router_circuit_cooldown_sec`` elapses. Tests
# and operators can rebuild the breaker by calling :func:`_rebuild_circuit`.


_circuit_lock = threading.Lock()
_circuit: "Any | None" = None


def _build_circuit() -> "Any":
    """Construct a CircuitBreaker honouring the current config values."""
    from larkhelm.memory_circuit import CircuitBreaker, CircuitConfig
    try:
        import larkhelm.config as _cfg
        failures = int(getattr(_cfg, "LLM_ROUTER_CIRCUIT_FAILURES", 5) or 5)
        cooldown = float(getattr(_cfg, "LLM_ROUTER_CIRCUIT_COOLDOWN_SEC", 30.0) or 30.0)
    except Exception:
        failures, cooldown = 5, 30.0
    return CircuitBreaker(CircuitConfig(
        failure_threshold=max(1, failures),
        cool_down_sec=max(1.0, cooldown),
    ))


def _get_circuit() -> "Any":
    global _circuit
    if _circuit is None:
        with _circuit_lock:
            if _circuit is None:
                _circuit = _build_circuit()
    return _circuit


def _rebuild_circuit() -> None:
    """Test-only: drop the singleton so the next call re-reads config."""
    global _circuit
    with _circuit_lock:
        _circuit = None


def circuit_state() -> str:
    """Expose the current breaker state to the metrics layer (REQ-04)."""
    try:
        state = _get_circuit().current_state()
    except Exception:
        return "closed"
    try:
        # Sync the gauge whenever an observer asks (cheap, no-op when
        # prometheus-client is missing).
        from larkhelm.metrics import set_llm_router_circuit_state
        set_llm_router_circuit_state("cheap", state)
    except Exception:
        pass
    return state


def _call_with_circuit(caller: Callable[[str], str], prompt: str) -> "str | None":
    """Invoke ``caller(prompt)`` through the breaker.

    Returns ``None`` if the breaker is open (caller falls back to the
    underlying retriever). Re-raises any exception from ``caller`` after
    recording it as a failure — the existing try/except in ``retrieve``
    then maps it to ``skipped_reason="caller_exception"``.
    """
    cb = _get_circuit()
    if not cb.allow():
        try:
            from larkhelm.metrics import set_llm_router_circuit_state
            set_llm_router_circuit_state("cheap", cb.current_state())
        except Exception:
            pass
        return None
    try:
        result = caller(prompt)
    except Exception:
        cb.record_failure()
        try:
            from larkhelm.metrics import set_llm_router_circuit_state
            set_llm_router_circuit_state("cheap", cb.current_state())
        except Exception:
            pass
        raise
    cb.record_success()
    try:
        from larkhelm.metrics import set_llm_router_circuit_state
        set_llm_router_circuit_state("cheap", cb.current_state())
    except Exception:
        pass
    return result


# ── Cheap LLM caller (lazy lookup for test override) ──────────────────────


def _resolve_cheap_caller() -> Callable[[str], str] | None:
    """Look up ``memory._run_one_shot`` lazily.

    Returning ``None`` here is the equivalent of "the LLM tier is
    unavailable" — the retriever then falls back to its underlying pool.
    Imports are kept lazy because ``larkhelm.memory`` pulls in heavy
    dependencies and we want this module to be safely importable even in
    truncated bootstrap contexts.
    """
    try:
        from larkhelm.memory import _run_one_shot
        return lambda prompt: _run_one_shot(
            prompt, ns="memory_llm_router", prefer_cheap=True
        )
    except Exception as e:
        _debug_log(f"[LLMRouter] cheap caller import failed: {e}")
        return None


# ── Reorder helper ────────────────────────────────────────────────────────


def _reorder_scored_by_ids(
    underlying: list[ScoredSlice],
    selected_ids: tuple[str, ...],
    top_k: int,
) -> list[ScoredSlice]:
    """Put ``selected_ids`` first (in LLM order), then any leftovers; trim
    to ``top_k``."""
    by_id = {item.slice.id: item for item in underlying}
    out: list[ScoredSlice] = []
    seen: set[str] = set()
    for sid in selected_ids:
        item = by_id.get(sid)
        if item is None or sid in seen:
            continue
        # Tag the reason so audit / debug logs show the routing trail.
        out.append(dataclasses.replace(
            item,
            reason=(item.reason + ",llm_router" if item.reason else "llm_router"),
        ))
        seen.add(sid)
    # Backfill with underlying ordering for ids the LLM didn't select.
    for item in underlying:
        if item.slice.id in seen:
            continue
        out.append(item)
        seen.add(item.slice.id)
        if len(out) >= top_k:
            break
    return out[:top_k]


# ── Public router class ───────────────────────────────────────────────────


@dataclasses.dataclass
class RouterDiagnostics:
    """Per-call diagnostic counters for audit-v3 records.

    Reset every call. Populated by ``LLMRouterRetriever.retrieve`` and
    consumed by ``build_audit_record_v3`` (added to
    ``memory_retriever`` in this phase).
    """
    invoked: bool = False
    cache_hit: bool = False
    skipped_reason: str = ""  # "" | "rate_limit" | "no_cheap_caller" | "parse_failed" | ...
    elapsed_ms: int = 0
    selected_by_llm: int = 0


class LLMRouterRetriever:
    """Wraps an underlying retriever (keyword or hybrid) with an LLM
    re-ranking pass.

    This is NOT a ``MemoryRetriever`` Protocol implementor in the strict
    sense — its ``retrieve`` signature returns the same shape but the
    method also stashes a ``diagnostics`` attribute that the audit layer
    reads. Callers using the bare Protocol path get correct behaviour
    transparently.
    """

    def __init__(
        self,
        underlying: MemoryRetriever,
        *,
        cheap_caller: Callable[[str], str] | None = None,
    ) -> None:
        self.underlying = underlying
        # Tests inject ``cheap_caller`` directly; production resolves lazily
        # on each call so a transient import failure can recover.
        self._injected_caller = cheap_caller
        # Per-instance diagnostics; mutated by ``retrieve``.
        self.diagnostics = RouterDiagnostics()

    # ── retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        request: RetrievalRequest,
        policy: InjectionPolicy,
        candidate_slices: list[MemorySlice],
    ) -> list[ScoredSlice]:
        diag = RouterDiagnostics()
        self.diagnostics = diag
        t0 = time.monotonic()

        # Always run the underlying retriever first — both as the
        # candidate source for the LLM AND as the fallback if anything
        # goes wrong downstream.
        try:
            underlying_scored = self.underlying.retrieve(
                request, policy, candidate_slices,
            )
        except Exception as e:
            # Underlying blew up; we can't proceed. Re-raise so the
            # outer fail-open chain in ``memory_context._build_with_retriever``
            # falls back to the legacy v2 builder. The router itself
            # does NOT swallow the underlying failure.
            _debug_log(f"[LLMRouter] underlying.retrieve failed: {e}")
            raise

        if not underlying_scored:
            diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return underlying_scored

        top_k = max(1, int(policy.top_k or 6))

        # Build the candidate pool the LLM will see — cap at LLM_ROUTER_MAX_POOL
        # to keep prompt cost bounded.
        pool = [s.slice for s in underlying_scored[: LLM_ROUTER_MAX_POOL]]
        candidate_ids = {s.id for s in pool}

        # Cache lookup.
        ttl = _config_int("memory_llm_router_cache_ttl_sec", _DEFAULT_CACHE_TTL)
        key = _cache_key(request.query, candidate_ids)
        cached = _cache_get(key, float(ttl))
        if cached is not None:
            diag.invoked = False
            diag.cache_hit = True
            diag.selected_by_llm = len(cached)
            out = _reorder_scored_by_ids(underlying_scored, cached, top_k)
            diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        # Rate limit.
        limit = _config_int("memory_llm_router_max_per_chat_per_min", _DEFAULT_RATE_LIMIT)
        if not _rate_check_and_record(request.chat_id, limit):
            diag.skipped_reason = "rate_limit"
            diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return underlying_scored[:top_k]

        # Resolve the cheap caller (test inject overrides production lookup).
        caller = self._injected_caller or _resolve_cheap_caller()
        if caller is None:
            diag.skipped_reason = "no_cheap_caller"
            diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return underlying_scored[:top_k]

        # Build prompt + invoke LLM. Any exception → fallback.
        prompt = _build_router_prompt(request, policy, pool, top_k=top_k)
        diag.invoked = True
        try:
            raw = _call_with_circuit(caller, prompt)
        except Exception as e:
            _debug_log(f"[LLMRouter] cheap caller raised: {e}")
            diag.skipped_reason = "caller_exception"
            diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return underlying_scored[:top_k]
        if raw is None:
            # Circuit open — fail-open to underlying retriever.
            diag.skipped_reason = "circuit_open"
            diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return underlying_scored[:top_k]

        selected = _parse_llm_response(raw or "", candidate_ids)
        if selected is None or not selected:
            diag.skipped_reason = "parse_failed" if raw else "empty_response"
            diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return underlying_scored[:top_k]

        _cache_put(key, selected)
        diag.selected_by_llm = len(selected)
        out = _reorder_scored_by_ids(underlying_scored, selected, top_k)
        diag.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return out


__all__ = [
    "LLM_ROUTER_MAX_POOL",
    "LLMRouterRetriever",
    "RouterDiagnostics",
    "_cache_clear_for_tests",  # exposed for unit-test fixtures
    "circuit_state",
    "_rebuild_circuit",
]
