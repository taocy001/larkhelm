"""larkhelm · agent_hub · task-aware backend selector.

``resolve_backend_for_task`` is the new entrypoint added in phase 5.
The legacy :func:`larkhelm.router.resolve_backend` is preserved for
backwards compatibility (NFR-COMPAT-01).
"""
from __future__ import annotations

from larkhelm.agent_hub.intent_types import TaskProfile
from larkhelm.backend_registry import BackendSpec


def _registry():
    """Resolve the live registry singleton at call time so tests can swap it."""
    import larkhelm.backend_registry as _br
    return _br.BACKEND_REGISTRY


def resolve_backend_for_task(
    chat_id: str,
    profile: TaskProfile,
    force_backend_id: str | None = None,
) -> BackendSpec:
    """Pick the best healthy+enabled BackendSpec for *profile*.

    1. Honor ``force_backend_id`` if it resolves to a healthy spec.
    2. Otherwise call ``BACKEND_REGISTRY.rank_for_task(profile)`` and
       return the top candidate.
    3. If no candidate is produced, fall back to legacy ``resolve_backend``.
    """
    registry = _registry()

    if force_backend_id:
        spec = registry.get(force_backend_id)
        if spec and spec.enabled and spec.healthy:
            return spec

    ranked = registry.rank_for_task(profile)
    if ranked:
        return ranked[0]

    from larkhelm.router import resolve_backend
    return resolve_backend(chat_id, "")


__all__ = ["resolve_backend_for_task"]
