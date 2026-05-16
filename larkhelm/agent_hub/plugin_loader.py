"""larkhelm · agent_hub.plugin_loader — discover third-party AgentExecutors.

Plugins register themselves via the ``larkhelm.agents`` entry-point group.
The expected entry point is a callable returning an :class:`AgentExecutor`
instance, or an :class:`AgentExecutor` subclass that can be instantiated
with no arguments.

Failures only emit ``_debug_log`` lines — never raised — so a broken plugin
cannot prevent larkhelm from starting (NFR-SEC-02).
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

from larkhelm.agent_hub.agent_base import AGENT_REGISTRY, AgentExecutor
# Centralized helper; previously re-defined locally.
from larkhelm.log import safe_log as _safe_log


def _instantiate(target) -> AgentExecutor | None:
    if isinstance(target, AgentExecutor):
        return target
    if isinstance(target, type) and issubclass(target, AgentExecutor):
        try:
            return target()
        except Exception as e:
            _safe_log(f"[plugin_loader] instantiate {target!r} failed: {e}")
            return None
    if callable(target):
        try:
            inst = target()
        except Exception as e:
            _safe_log(f"[plugin_loader] callable {target!r} failed: {e}")
            return None
        if isinstance(inst, AgentExecutor):
            return inst
        _safe_log(f"[plugin_loader] callable {target!r} did not return AgentExecutor")
    return None


def _load_from_entry_points(
    *, _entry_points_fn: "Callable[..., Any] | None" = None,
) -> int:
    """Scan ``importlib.metadata.entry_points`` for ``larkhelm.agents`` plugins.

    ``_entry_points_fn`` is a test hook: production callers leave it ``None``
    so the live ``from importlib.metadata import entry_points`` runs; tests
    pass ``lambda: fake_eps`` to inject a synthetic entry-point set without
    touching ``sys.modules``.
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
        # Python 3.10+: select() is the new API, .get() for older.
        if hasattr(eps, "select"):
            agent_eps = eps.select(group="larkhelm.agents")
        else:
            agent_eps = eps.get("larkhelm.agents", [])
    except Exception as e:
        _safe_log(f"[plugin_loader] entry_points scan failed: {e}")
        return 0

    for ep in agent_eps:
        try:
            target = ep.load()
        except Exception as e:
            _safe_log(f"[plugin_loader] entry-point {ep!r} load failed: {e}")
            continue
        agent = _instantiate(target)
        if agent is None:
            continue
        try:
            AGENT_REGISTRY.register(agent)
            count += 1
        except Exception as e:
            _safe_log(f"[plugin_loader] register {agent!r} failed: {e}")
    return count


def _load_from_config(
    config: dict,
    *, _import_module_fn: "Callable[..., Any] | None" = None,
) -> int:
    """Load plugins listed under ``config['agent_plugins']``.

    ``_import_module_fn`` is a test hook: production callers leave it
    ``None`` so ``importlib.import_module`` runs; tests pass
    ``lambda name: fake_mod`` to inject a synthetic module without
    touching ``sys.modules``.
    """
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
            continue
        try:
            module = _import_module_fn(module_name)
            target = getattr(module, attr)
        except Exception as e:
            _safe_log(f"[plugin_loader] import {plugin!r} failed: {e}")
            continue
        agent = _instantiate(target)
        if agent is None:
            continue
        try:
            AGENT_REGISTRY.register(agent)
            count += 1
        except Exception as e:
            _safe_log(f"[plugin_loader] register {agent!r} failed: {e}")
    return count


def load_plugins(config: dict | None = None) -> int:
    """Load plugins from entry points + config['agent_plugins']. Returns count."""
    cfg = config or {}
    n = _load_from_entry_points()
    n += _load_from_config(cfg)
    return n


__all__ = ["load_plugins"]
