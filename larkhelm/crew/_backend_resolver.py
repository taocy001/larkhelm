"""
larkhelm · Crew backend resolver (simplified)

Always uses the configured active backend (role=orchestrator in registry).
Validates tool capability when the agent declares an output_file.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from larkhelm.crew_types import AgentSpec, NoBackendAvailableError
from larkhelm.log import _debug_log

if TYPE_CHECKING:
    from larkhelm.backend_registry import BackendSpec


def _backend_has_tools(spec: "BackendSpec") -> bool:
    """Return True iff spec advertises 'tools' in its tags."""
    try:
        return "tools" in (spec.tags or [])
    except Exception:
        return False


def resolve_backend(
    spec: AgentSpec,
    fallback_orchestrator: bool = True,
    exclude_backend_ids: "frozenset[str] | None" = None,
) -> "BackendSpec":
    """Return the active BackendSpec that should serve spec.

    Always picks the configured orchestrator backend.  When the agent
    has an output_file, backends without the 'tools' tag are skipped
    (they would emit protocol tokens as plain text and corrupt the file).

    exclude_backend_ids: per-call set of backend ids to skip in the
    retry loop when a backend just produced a sentinel-violating artifact.
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY
    excluded = frozenset(exclude_backend_ids or ())

    chain = BACKEND_REGISTRY.get_orchestrator_chain()
    for cand in chain:
        if cand.id in excluded:
            continue
        if spec.output_file and not _backend_has_tools(cand):
            _debug_log(
                f"[Crew] BackendResolver: skipping {cand.id!r} — lacks tool "
                f"capability for agent with output_file={spec.output_file!r}"
            )
            continue
        return cand

    raise NoBackendAvailableError(
        task_profile="",
        reason=(
            "no healthy+enabled backend available"
            + (" (all excluded in retry loop)" if excluded else "")
            + "; check /status"
        ),
    )


def resolve_backend_preview() -> dict[str, str]:
    """Return {'active': backend_id} for /status display. Never raises."""
    try:
        from larkhelm.backend_registry import BACKEND_REGISTRY
        orch = BACKEND_REGISTRY.get_orchestrator()
        return {"active": orch.id if orch else "<none>"}
    except Exception as e:
        _debug_log(f"[Crew] BackendResolver preview failed: {e}")
        return {"active": "<none>"}
