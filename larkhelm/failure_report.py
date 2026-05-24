"""larkhelm · failure_report — generic admin-card failure reporter (P1-1a).

Thin, stateless abstraction for pushing operational failures to the admin
chat as an orange Feishu card. Designed as the single forward-facing entry
point for future migrations away from scattered ad-hoc admin cards (e.g.
cascade backoff exhausted, circuit breaker tripped, long-running task OOM
termination). This module is the **abstraction skeleton only** — no
existing call sites are migrated here; P1-1b/c will wire callers up.

Public surface
--------------

* :func:`emit` — sole entry point, ``(category, summary, detail="") -> None``.

Behaviour contract
------------------

* ``failure_report_card_enabled = False`` (default) → O(1) early return; no
  imports, no IO.
* flag-on + ``admin_chat_id`` empty → ``safe_log`` once + return.
* flag-on + ``admin_chat_id`` non-empty → render orange card, fire one
  ``send_card`` call, return ``None``.
* All branches **never raise**. Any error inside (config read, redact,
  render, send_card, log) is swallowed via ``safe_log``.

Future extension (P1-1c placeholder)
------------------------------------

A ``larkhelm_failure_report_emit_total{category, outcome}`` Counter will
be incremented inside ``emit`` once P1-1c lands. Intentionally not wired
this iteration — keeps the diff to a single new module + flag.
"""
from __future__ import annotations

__all__ = ["emit"]

_CATEGORY_MAX: int = 32
_SUMMARY_MAX:  int = 200
_DETAIL_MAX:   int = 800


def _truncate(value: str, limit: int) -> str:
    try:
        s = str(value or "")
    except Exception:
        s = ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _render_body(summary_clean: str, detail_clean: str) -> str:
    lines: list[str] = [summary_clean]
    if detail_clean:
        lines.append("")
        lines.append("**Detail:**")
        lines.append(detail_clean)
    return "\n".join(lines)


def emit(category: str, summary: str, detail: str = "") -> None:
    """Push a generic operational-failure card to the admin chat.

    Parameters
    ----------
    category : str
        Short failure tag (≤ 32 chars, snake_case recommended, e.g.
        ``"backoff_exhausted"`` / ``"circuit_open"`` / ``"oom_killed"``).
    summary : str
        One-line human-readable summary (≤ 200 chars). Passed through
        ``redact_error`` before truncation.
    detail : str, default ""
        Optional multi-line detail block (≤ 800 chars). Also redacted.
        Empty string suppresses the detail paragraph entirely.

    Guarantees
    ----------
    Never raises. Safe to invoke from ``finally`` / ``except`` blocks.
    """
    # Step 1 — flag-off fast path (no imports, no IO).
    try:
        import larkhelm.config as _cfg
        enabled = bool(getattr(_cfg, "FAILURE_REPORT_CARD_ENABLED", False))
    except Exception:
        return
    if not enabled:
        return

    # Step 2 — late-bind log helpers (safe_log / redact_error live in
    # larkhelm.log; importing here means flag-off path doesn't touch log).
    try:
        from larkhelm.log import redact_error, safe_log
    except Exception:
        return

    # Step 3 — admin_chat_id required to send anything.
    try:
        admin_chat = str(getattr(_cfg, "ADMIN_CHAT_ID", "") or "")
    except Exception:
        admin_chat = ""
    if not admin_chat:
        safe_log("[FailureReport] admin card skipped: admin_chat_id empty")
        return

    # Step 4 — sanitize + truncate. Each step wrapped so a malformed input
    # cannot escape this function.
    try:
        category_clean = _truncate(category, _CATEGORY_MAX)
        summary_clean = _truncate(redact_error(summary), _SUMMARY_MAX)
        detail_clean = _truncate(redact_error(detail), _DETAIL_MAX) if detail else ""
        body = _render_body(summary_clean, detail_clean)
    except Exception as e:
        safe_log(f"[FailureReport] render failed: {e}")
        return

    # Step 5 — lazy import lark_client to avoid early-boot import cycles
    # and to keep flag-off path zero-overhead.
    try:
        from larkhelm.lark_client import send_card
    except Exception as e:
        safe_log(f"[FailureReport] lark_client import failed: {e}")
        return

    try:
        send_card(admin_chat, f"⚠️ {category_clean}", body, color="orange")
    except Exception as e:
        safe_log(f"[FailureReport] send_card failed: {e}")
