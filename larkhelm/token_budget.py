"""larkhelm · Backend-aware token budget manager (Week-2 milestone).

Maps each backend / provider to its context-window size and computes:

1. **Memory injection budget** — how many characters of retrieved memory
   should be composed into the prompt, scaled to the backend's capacity.
2. **API max_tokens** — a safe output-token ceiling that leaves headroom
   for the estimated input size.

All defaults are conservative (slightly below advertised max) to account
for tokenisation overhead, system instructions, and tool definitions.
Operators can override any value via ``config.json``
(``context_window_<backend_id>``).

Log prefix: ``[TokenBudget]``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from larkhelm.backend_registry import BackendSpec


# ── 9 well-known backends / providers ──────────────────────────────────────
#
# Values are the *advertised* context-window size in tokens.  The public
# functions below apply a safety margin (default 0.85) before using the
# number for budget math.
#
DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    # CLI backends (auto-discovered)
    "claude":            200_000,
    "gemini":          1_048_576,
    "kimi":              256_000,
    "kimi-code":         256_000,
    "deepseek":           64_000,
    # API backends
    "anthropic_api":     200_000,
    "google_api":      1_048_576,
    "openai_compat_api": 128_000,
    "deepseek_api":       64_000,
}

# Minimum context window we will ever assume (safety floor).
_MIN_CONTEXT_WINDOW = 32_000

# Safety factor: only use 85 % of the advertised window so tokenisation
# overhead, system instructions, and tool definitions don't blow the limit.
_DEFAULT_SAFETY_FACTOR = 0.85

# Memory-budget scale factors by context-window tier.
# Tier boundaries are intuitive rather than precise; the goal is to avoid
# choking small-window models while letting large-window ones breathe.
# Ordered **descending** so the first match is the highest applicable tier.
_TIER_SCALES: list[tuple[int, float]] = [
    (256_000, 1.20),   # large  → +20 % over POLICY_TABLE default
    (64_000,  1.00),   # medium → keep default
]

# Floor for any memory budget (chars) — never drop below this.
_MEMORY_BUDGET_FLOOR = 400

# Default per-agent_type token_budget from POLICY_TABLE (chars).
# Kept here so token_budget stays importable without dragging
# memory_retriever (and its heavy import graph) at module level.
_DEFAULT_AGENT_BUDGETS: dict[str, int] = {
    "chat": 1200,
    "btw":   800,
    "dev":  3000,
    "crew": 2400,
    "plan": 2000,
    "doc":   800,
}

# API output caps per backend tier.
# These are *upper* bounds; ``compute_api_max_tokens`` may choose a lower
# value when the estimated input is large.
_MAX_OUTPUT_CAPS: dict[str, int] = {
    "claude":            8192,
    "gemini":            8192,
    "kimi":              8192,
    "kimi-code":         8192,
    "deepseek":          8192,
    "anthropic_api":     8192,
    "google_api":        8192,
    "openai_compat_api": 8192,
    "deepseek_api":      8192,
}


def _config() -> dict:
    """Lazy import of runtime config dict."""
    try:
        import larkhelm.config as _cfg
        return getattr(_cfg, "config", {}) or {}
    except Exception:
        return {}


def resolve_context_window(spec: "BackendSpec | None") -> int:
    """Return the context-window size (tokens) for *spec*.

    Resolution order:
      1. Operator override: ``config.json["context_window_<spec.id>"]``
      2. Operator override: ``config.json["context_window_<spec.provider>"]``
      3. Hard-coded default from :data:`DEFAULT_CONTEXT_WINDOWS`
      4. Fallback: ``_MIN_CONTEXT_WINDOW`` (32 000)

    ``spec`` may be ``None`` (e.g. when the backend hasn't been selected
    yet); in that case step 4 applies.
    """
    if spec is None:
        return _MIN_CONTEXT_WINDOW

    # 0. If the spec already carries a resolved context_window, trust it.
    _cw = getattr(spec, "context_window", 0)
    if isinstance(_cw, int) and _cw > 0:
        return _cw

    cfg = _config()

    # 1. Per-id override
    key_id = f"context_window_{spec.id}"
    if key_id in cfg:
        try:
            val = int(cfg[key_id])
            if val > 0:  # 0 means "use built-in default"
                return max(_MIN_CONTEXT_WINDOW, val)
        except (TypeError, ValueError):
            pass

    # 2. Per-provider override
    key_prov = f"context_window_{spec.provider}"
    if key_prov in cfg:
        try:
            val = int(cfg[key_prov])
            if val > 0:  # 0 means "use built-in default"
                return max(_MIN_CONTEXT_WINDOW, val)
        except (TypeError, ValueError):
            pass

    # 3. Built-in default
    default = DEFAULT_CONTEXT_WINDOWS.get(spec.id) or DEFAULT_CONTEXT_WINDOWS.get(spec.provider)
    if default:
        return max(_MIN_CONTEXT_WINDOW, default)

    # 4. Floor
    return _MIN_CONTEXT_WINDOW


def compute_memory_char_budget(
    spec: "BackendSpec | None",
    agent_type: str = "chat",
    *,
    base_budget: int | None = None,
) -> int:
    """Backend-aware memory-injection budget in **characters**.

    The returned value is meant to replace ``InjectionPolicy.token_budget``
    (which is expressed in characters, not tokens, for historical reasons).

    Algorithm:
      * Look up the backend's context window.
      * Pick a tier scale (large +20 %, medium ±0 %, small −30 %).
      * Multiply the base budget (from ``POLICY_TABLE`` or *base_budget*)
        by the scale.
      * Clamp to ``_MEMORY_BUDGET_FLOOR``.

    When the feature flag ``backend_aware_budget_enabled`` is ``False``
    (or when *spec* is ``None``), the base budget is returned unchanged
    so existing callers see no behaviour change.
    """
    cfg = _config()
    if not bool(cfg.get("backend_aware_budget_enabled", False)):
        return base_budget if base_budget is not None else _DEFAULT_AGENT_BUDGETS.get(agent_type, 1200)

    window = resolve_context_window(spec)
    base = base_budget if base_budget is not None else _DEFAULT_AGENT_BUDGETS.get(agent_type, 1200)

    # Find the first tier whose boundary is <= window.
    # Default 0.70 for anything below the smallest boundary.
    scale = 0.70
    for boundary, s in _TIER_SCALES:
        if window >= boundary:
            scale = s
            break

    budget = int(base * scale)
    return max(_MEMORY_BUDGET_FLOOR, budget)


def compute_api_max_tokens(
    spec: "BackendSpec | None",
    *,
    input_tokens_est: int = 8000,
    min_output: int = 256,
    max_output_cap: int | None = None,
) -> int:
    """Compute a safe ``max_tokens`` (or ``max_output_tokens``) for an API call.

    Formula::

        safe_budget = context_window * safety_factor
        available   = safe_budget - input_tokens_est
        cap         = max_output_cap or backend-specific cap
        result      = clamp(available, min_output, cap)

    Parameters
    ----------
    spec:
        The backend that will receive the request.
    input_tokens_est:
        Conservative estimate of the prompt size in tokens (system
        instructions + memory + history + user message).  The default
        ``8000`` covers the typical three-tier memory injection (~4 500
        chars ≈ 1 500 tokens) + a few turns of history.
    min_output:
        Absolute floor; never go below this.
    max_output_cap:
        Hard ceiling.  When ``None``, the per-backend cap from
        :data:`_MAX_OUTPUT_CAPS` is used.

    Returns
    -------
    int
        The recommended ``max_tokens`` value.
    """
    if spec is None:
        return max_output_cap if max_output_cap is not None else 8192

    window = resolve_context_window(spec)
    safe_budget = int(window * _DEFAULT_SAFETY_FACTOR)
    available = safe_budget - max(0, input_tokens_est)

    cap = max_output_cap
    if cap is None:
        cap = _MAX_OUTPUT_CAPS.get(spec.id) or _MAX_OUTPUT_CAPS.get(spec.provider) or 8192

    return max(min_output, min(cap, available))


def apply_backend_aware_budget(
    policy,  # InjectionPolicy
    spec: "BackendSpec | None",
) -> object:  # InjectionPolicy
    """Return a copy of *policy* with ``token_budget`` adjusted for *spec*.

    This is a thin convenience wrapper around
    :func:`compute_memory_char_budget`.  When the feature flag is off or
    *spec* is ``None``, the original policy is returned unchanged
    (no copy) to avoid unnecessary allocation.
    """
    if spec is None:
        return policy

    cfg = _config()
    if not bool(cfg.get("backend_aware_budget_enabled", False)):
        return policy

    new_budget = compute_memory_char_budget(spec, policy.agent_type, base_budget=policy.token_budget)
    if new_budget == policy.token_budget:
        return policy

    import dataclasses as _dc
    return _dc.replace(policy, token_budget=new_budget)
