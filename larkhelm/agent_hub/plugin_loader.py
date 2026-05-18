"""larkhelm · agent_hub.plugin_loader — discover third-party AgentExecutors.

Plugins register themselves via the ``larkhelm.agents`` entry-point group
*and* via the ``agent_plugins`` config list. Failures are surfaced as
structured :class:`PluginLoadReport` rows so :func:`bridge.boot` can push
a single admin card (REQ-07) instead of operators having to grep
``DEBUG_LOG``.

Public surface
--------------
* :func:`load_plugins` — returns :class:`PluginLoadReport` (previously
  ``int``; ``len(report.loaded)`` reproduces the old count).
* :func:`_load_from_entry_points` / :func:`_load_from_config` — keep their
  ``int`` return for legacy callers; when ``report=...`` is passed the
  loader also records failures into it.
"""
from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Optional

from larkhelm.agent_hub.agent_base import AGENT_REGISTRY, AgentExecutor
from larkhelm.agent_hub.plugin_report import PluginFailure, PluginLoadReport
# Centralized helper; kept as ``_safe_log`` so existing tests
# (`patch.object(plugin_loader, "_safe_log")`) keep working.
from larkhelm.log import safe_log as _safe_log


def _trim_reason(value: object, limit: int = 80) -> str:
    """Trim a free-form exception or message to ≤ 80 chars (REQ-07 §4)."""
    s = str(value or "")
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _record_failure(
    report: Optional[PluginLoadReport],
    spec: str,
    stage: str,
    reason: object,
) -> None:
    if report is None:
        return
    report.failed.append(PluginFailure(spec=spec, stage=stage, reason=_trim_reason(reason)))


def _instantiate(
    target,
    *,
    spec: str = "",
    report: Optional[PluginLoadReport] = None,
) -> "AgentExecutor | None":
    if isinstance(target, AgentExecutor):
        return target
    if isinstance(target, type) and issubclass(target, AgentExecutor):
        try:
            return target()
        except Exception as e:
            _safe_log(f"[plugin_loader] instantiate {target!r} failed: {e}")
            _record_failure(report, spec, "instantiate", e)
            return None
    if callable(target):
        try:
            inst = target()
        except Exception as e:
            _safe_log(f"[plugin_loader] callable {target!r} failed: {e}")
            _record_failure(report, spec, "instantiate", e)
            return None
        if isinstance(inst, AgentExecutor):
            return inst
        _safe_log(f"[plugin_loader] callable {target!r} did not return AgentExecutor")
        _record_failure(report, spec, "instantiate", "returned non-AgentExecutor")
    return None


def _load_from_entry_points(
    *, _entry_points_fn: "Callable[..., Any] | None" = None,
    report: Optional[PluginLoadReport] = None,
) -> int:
    """Scan ``importlib.metadata.entry_points`` for ``larkhelm.agents`` plugins.

    ``_entry_points_fn`` is a test hook: production callers leave it ``None``.
    Returns the count of successfully-registered agents.
    """
    count = 0
    if _entry_points_fn is None:
        try:
            from importlib.metadata import entry_points as _live_entry_points
            _entry_points_fn = _live_entry_points
        except Exception:
            return 0
    try:
        eps: Any = _entry_points_fn()
        if hasattr(eps, "select"):
            agent_eps = eps.select(group="larkhelm.agents")
        else:
            agent_eps = eps.get("larkhelm.agents", [])
    except Exception as e:
        _safe_log(f"[plugin_loader] entry_points scan failed: {e}")
        _record_failure(report, "<entry_points>", "import", e)
        return 0

    for ep in agent_eps:
        spec = str(getattr(ep, "name", "")) or repr(ep)
        try:
            target = ep.load()
        except Exception as e:
            _safe_log(f"[plugin_loader] entry-point {ep!r} load failed: {e}")
            _record_failure(report, spec, "import", e)
            continue
        agent = _instantiate(target, spec=spec, report=report)
        if agent is None:
            continue
        try:
            AGENT_REGISTRY.register(agent)
            count += 1
            if report is not None:
                report.loaded.append(getattr(agent, "agent_type", repr(agent)))
        except Exception as e:
            _safe_log(f"[plugin_loader] register {agent!r} failed: {e}")
            _record_failure(report, spec, "register", e)
    return count


def _load_from_config(
    config: dict,
    *, _import_module_fn: "Callable[..., Any] | None" = None,
    report: Optional[PluginLoadReport] = None,
) -> int:
    """Load plugins listed under ``config['agent_plugins']``. Returns success count."""
    count = 0
    if _import_module_fn is None:
        _import_module_fn = importlib.import_module
    plugins = config.get("agent_plugins") or []
    if not isinstance(plugins, list):
        return 0
    for plugin in plugins:
        if not isinstance(plugin, str) or not plugin.strip():
            continue
        target_path = plugin.strip()
        if ":" in target_path:
            module_name, _, attr = target_path.partition(":")
        else:
            module_name, _, attr = target_path.rpartition(".")
        if not module_name or not attr:
            _safe_log(f"[plugin_loader] invalid plugin spec: {plugin!r}")
            _record_failure(report, plugin, "import", "invalid plugin spec")
            continue
        try:
            module = _import_module_fn(module_name)
            target = getattr(module, attr)
        except Exception as e:
            _safe_log(f"[plugin_loader] import {plugin!r} failed: {e}")
            _record_failure(report, plugin, "import", e)
            continue
        agent = _instantiate(target, spec=plugin, report=report)
        if agent is None:
            continue
        try:
            AGENT_REGISTRY.register(agent)
            count += 1
            if report is not None:
                report.loaded.append(getattr(agent, "agent_type", repr(agent)))
        except Exception as e:
            _safe_log(f"[plugin_loader] register {agent!r} failed: {e}")
            _record_failure(report, plugin, "register", e)
    return count


def load_plugins(config: dict | None = None) -> PluginLoadReport:
    """Load plugins from entry points + ``config['agent_plugins']``.

    Returns a structured report. ``len(report.loaded)`` reproduces the
    old ``int`` contract for any caller that hasn't migrated yet.
    """
    cfg = config or {}
    report = PluginLoadReport()
    t0 = time.monotonic()
    _load_from_entry_points(report=report)
    _load_from_config(cfg, report=report)
    report.duration_sec = round(time.monotonic() - t0, 4)
    return report


__all__ = ["load_plugins"]
