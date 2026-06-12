"""larkhelm · Backend-aware token budget manager.

Maps each backend / provider to its context-window size and computes
**API max_tokens** — a safe output-token ceiling that leaves headroom
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

# All backends currently share the same output cap (8192 tokens).
# Stored as a constant; if a future backend needs a different cap,
# introduce a per-backend dict then.
_DEFAULT_MAX_OUTPUT_CAP = 8192


def _config() -> dict:
    """Lazy import of runtime config dict."""
    try:
        import larkhelm.config as _cfg
        return getattr(_cfg, "config", {}) or {}
    except (ImportError, AttributeError):
        return {}


def resolve_context_window(spec: "BackendSpec | None") -> int:
    """Return the context-window size (tokens) for *spec*.

    Resolution order:
      1. Operator override: ``config.json["context_window_<spec.id>"]``
      2. Operator override: ``config.json["context_window_<spec.provider>"]``
      3. Hard-coded default from :data:`DEFAULT_CONTEXT_WINDOWS`
      4. Fallback: ``_MIN_CONTEXT_WINDOW`` (32 000)

    Value ``0`` in any override slot means "use built-in default" and is
    treated as absent.  ``spec`` may be ``None`` (e.g. when the backend
    hasn't been selected yet); in that case step 4 applies.
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
        cap         = max_output_cap or _DEFAULT_MAX_OUTPUT_CAP (8192)
        result      = clamp(available, min_output, cap)

    Parameters
    ----------
    spec:
        The backend that will receive the request.
    input_tokens_est:
        Conservative estimate of the prompt size in tokens.  The default
        ``8000`` is a worst-case budget covering system prompt, tool
        definitions, three-tier memory injection (~1 500 tokens), and
        several turns of conversation history.  For most requests the
        actual input is smaller; the cap (``_DEFAULT_MAX_OUTPUT_CAP``)
        is the binding constraint for large-context backends.
    min_output:
        Absolute floor; never go below this.
    max_output_cap:
        Hard ceiling.  When ``None``, ``_DEFAULT_MAX_OUTPUT_CAP`` (8192)
        is used for all backends.

    Returns
    -------
    int
        The recommended ``max_tokens`` value.
    """
    if spec is None:
        return max_output_cap if max_output_cap is not None else _DEFAULT_MAX_OUTPUT_CAP

    window = resolve_context_window(spec)
    safe_budget = int(window * _DEFAULT_SAFETY_FACTOR)
    available = safe_budget - max(0, input_tokens_est)

    cap = max_output_cap if max_output_cap is not None else _DEFAULT_MAX_OUTPUT_CAP

    return max(min_output, min(cap, available))


