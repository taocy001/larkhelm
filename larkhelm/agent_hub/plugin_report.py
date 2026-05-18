"""larkhelm · agent_hub.plugin_report — structured plugin-load report + admin card emitter.

REQ-07: when boot-time plugin loading hits import/instantiate/register
failures, surface them as a single orange Feishu card to the admin chat
instead of forcing operators to grep ``DEBUG_LOG``.

Two responsibilities only:

* :class:`PluginLoadReport` dataclass shape (loaded names, failed entries).
* :func:`emit_admin_card` rendering helper that calls
  :func:`larkhelm.lark_client.send_card`. Silently no-op when
  ``admin_chat_id`` is empty, when ``lark_client`` cannot import (early
  boot / single-file test), or when the send itself raises — this is a
  diagnostics path and must never block startup.

Security: card body lists ``spec``, ``stage``, ``reason`` truncated to
80 chars per entry. No tracebacks, no paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from larkhelm.log import safe_log as _safe_log


@dataclass
class PluginFailure:
    spec: str            # raw config / entry-point string
    stage: str           # "import" | "instantiate" | "register"
    reason: str          # already-trimmed, ≤ 80 chars expected


@dataclass
class PluginLoadReport:
    loaded: list[str] = field(default_factory=list)
    failed: list[PluginFailure] = field(default_factory=list)
    duration_sec: float = 0.0

    def has_failures(self) -> bool:
        return bool(self.failed)


def _truncate(value: str, limit: int = 80) -> str:
    s = str(value or "")
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _render_body(report: PluginLoadReport) -> str:
    lines: list[str] = []
    lines.append(
        f"**Plugin load summary** — loaded {len(report.loaded)}, "
        f"failed {len(report.failed)} ({report.duration_sec:.2f}s)"
    )
    if not report.failed:
        return "\n".join(lines)
    lines.append("")
    lines.append("**Failures:**")
    for f in report.failed:
        lines.append(
            f"- `{_truncate(f.spec, 60)}` · {_truncate(f.stage, 20)} · "
            f"{_truncate(f.reason, 80)}"
        )
    return "\n".join(lines)


def emit_admin_card(
    report: PluginLoadReport,
    admin_chat_id: str,
) -> None:
    """Push an orange card listing failed plugins to ``admin_chat_id``.

    No-op when ``admin_chat_id`` is empty/None or when ``lark_client``
    cannot send (boot ordering, no Feishu credentials, transient error).
    """
    if not admin_chat_id:
        _safe_log("[PluginReport] admin card skipped: admin_chat_id empty")
        return
    if not report.has_failures():
        _safe_log("[PluginReport] admin card skipped: no failures")
        return
    try:
        from larkhelm.lark_client import send_card  # lazy import to avoid early-boot cycles
    except Exception as e:
        _safe_log(f"[PluginReport] lark_client import failed: {e}")
        return
    body = _render_body(report)
    try:
        send_card(
            admin_chat_id,
            "⚠️ Plugin load failures",
            body,
            color="orange",
        )
    except Exception as e:
        _safe_log(f"[PluginReport] send_card failed: {e}")


def aggregate_failures(failures: Iterable[PluginFailure]) -> list[PluginFailure]:
    """Helper: drop duplicate (spec, stage) pairs while preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[PluginFailure] = []
    for f in failures:
        key = (f.spec, f.stage)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


__all__ = [
    "PluginFailure",
    "PluginLoadReport",
    "aggregate_failures",
    "emit_admin_card",
]
