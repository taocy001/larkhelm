"""
larkhelm · Crew backend resolver

Maps an :class:`AgentSpec` to a :class:`BackendSpec` using the new
``task_profile`` hint, with graceful fallback to the legacy ``model``
string-dispatch behavior when ``task_profile`` is empty.

Design reference: ``.crew_workspace/design.md`` §1.2 G3 / §6.1.

The resolver itself is stateless and side-effect-free (apart from one
``_debug_log`` line on fall-through paths). It does NOT call any
backend dispatch function — its sole job is to pick *which* backend
should serve. The caller (``_run_agent``) decides how to actually
invoke it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from larkhelm.agent_hub.intent_types import TaskProfile
from larkhelm.crew_types import AgentSpec, NoBackendAvailableError
from larkhelm.log import _debug_log

if TYPE_CHECKING:
    from larkhelm.backend_registry import BackendSpec


# ── Profile catalog ──────────────────────────────────────────────────────
# Five baseline profiles that cover the /dev pipeline plus generic chat.
# Numbers are first-pass weights derived from the Phase C PRD §3 G3
# discussion; future tuning will move these to ``config.task_profiles``
# but the in-code defaults remain authoritative when the config key is
# absent (which is the only mode supported in this phase).
TASK_PROFILES: dict[str, TaskProfile] = {
    "planner":  TaskProfile(
        complexity="complex",
        required_capabilities={"reasoning": 1.0, "long_context": 0.6, "tools": 0.6},
        # ``require_tools`` added 2026-05-22 after the DSML-leak incident:
        # planner agents need Read (to inspect existing code / upstream
        # artefacts) and Write (to persist prd.md / prd_criteria.json /
        # design.md / tasks.json). API-only backends without a tool layer
        # (e.g. DeepSeek with tags=["cheap","fast"]) cannot honour these
        # calls and instead emit the protocol tokens as plain text — that
        # text then landed in output_file, silently corrupting downstream
        # stages until the validator (commit 539cc45) caught it. Hard-gate
        # the routing so the cascade never starts.
        require_tools=True,
        latency_pref="medium",
    ),
    "engineer": TaskProfile(
        complexity="complex",
        required_capabilities={"coding": 1.0, "tools": 0.8},
        require_tools=True,
        latency_pref="medium",
    ),
    "qa":       TaskProfile(
        complexity="medium",
        required_capabilities={"coding": 0.6, "reasoning": 0.8, "tools": 0.8},
        require_tools=True,
        latency_pref="medium",
    ),
    "reviewer": TaskProfile(
        complexity="medium",
        required_capabilities={"reasoning": 1.0, "long_context": 0.5, "tools": 0.6},
        # Same reasoning as ``planner`` — reviewer reads the implementation
        # diff + writes review.md, both of which fundamentally require tool
        # support. Without ``require_tools`` an API-only backend would also
        # leak tool-call tokens into review.md.
        require_tools=True,
        latency_pref="medium",
    ),
    "chat":     TaskProfile(
        complexity="simple",
        required_capabilities={"chat": 1.0},
        latency_pref="fast",
    ),
}


# ``model`` strings recognised by the legacy direct-dispatch branch in
# ``_run_agent``. When a spec carries one of these AND no ``task_profile``
# we surface a synthetic BackendSpec marked as legacy-dispatch so the
# caller knows to take its old code path. The runner reads ``provider``
# off the resolved spec to choose between the API / CLI / hermes branches.
_LEGACY_DIRECT_MODELS: frozenset[str] = frozenset({"gemini", "kimi", "deepseek"})


def _is_legacy_direct(model: str) -> bool:
    if not model:
        return False
    if model in _LEGACY_DIRECT_MODELS:
        return True
    return model.startswith("hermes_")


def resolve_backend(
    spec: AgentSpec,
    fallback_orchestrator: bool = True,
) -> "BackendSpec":
    """Return the BackendSpec that should serve ``spec``.

    Priority (see design.md §6.1):

      1. ``spec.task_profile`` non-empty → ``BACKEND_REGISTRY.rank_for_task``
         on the matching :class:`TaskProfile`; first result wins.
      2. ``spec.model`` matches the legacy direct-dispatch set
         (``gemini`` / ``kimi`` / ``deepseek`` / ``hermes_*``) → return
         the registered BackendSpec for that id (so dispatch can read
         ``enabled``); for hermes_* a synthetic BackendSpec is returned
         since hermes is an orchestrator macro, not a single backend.
      3. ``fallback_orchestrator`` True →
         ``BACKEND_REGISTRY.get_orchestrator()``.
      4. Nothing usable → :class:`NoBackendAvailableError`.

    Importing ``BACKEND_REGISTRY`` lazily inside the function keeps this
    module safe to import from ``crew_types`` consumers without hauling
    the registry's startup chain into early bootstrap.
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY

    # Path 1 — task_profile (preferred new path)
    profile_name = spec.task_profile or ""
    if profile_name:
        profile = TASK_PROFILES.get(profile_name)
        if profile is None:
            _debug_log(
                f"[Crew] BackendResolver: unknown task_profile={profile_name!r}, "
                "falling through to legacy/orchestrator path"
            )
        else:
            ranked = BACKEND_REGISTRY.rank_for_task(profile)
            if ranked:
                return ranked[0]
            _debug_log(
                f"[Crew] BackendResolver: rank_for_task({profile_name}) returned "
                "no candidates; falling through"
            )

    # Path 2 — legacy direct-dispatch model strings
    if _is_legacy_direct(spec.model):
        if spec.model.startswith("hermes_"):
            # Synthetic spec — hermes is an orchestrator macro, not a backend.
            from larkhelm.backend_registry import BackendSpec as _BS
            return _BS(
                id=spec.model,
                provider="hermes",
                display_name=spec.model,
                role="orchestrator",
                tags=[],
                healthy=True,
                enabled=True,
            )
        legacy = BACKEND_REGISTRY.get(spec.model)
        if legacy is not None and legacy.enabled and legacy.healthy:
            return legacy
        # Disabled or unhealthy — keep falling through to the orchestrator
        # rather than handing the runner a dead spec.
        _debug_log(
            f"[Crew] BackendResolver: legacy model={spec.model!r} unavailable "
            f"(spec={'present' if legacy else 'missing'}, "
            f"enabled={getattr(legacy, 'enabled', '?')}, "
            f"healthy={getattr(legacy, 'healthy', '?')}); falling through"
        )

    # Path 3 — orchestrator fallback
    if fallback_orchestrator:
        orch = BACKEND_REGISTRY.get_orchestrator()
        if orch is not None:
            return orch

    # Path 4 — nothing usable
    raise NoBackendAvailableError(
        task_profile=profile_name,
        reason=("no healthy+enabled backend matched task_profile + no orchestrator "
                "fallback available; check /status"),
    )


def resolve_backend_preview() -> dict[str, str]:
    """Return ``{task_profile_name: top_backend_id_or_'<none>'}`` for /status.

    Best-effort diagnostic display — never raises. A failed lookup for any
    individual profile records ``"<none>"`` for that entry rather than
    poisoning the entire mapping.
    """
    out: dict[str, str] = {}
    try:
        from larkhelm.backend_registry import BACKEND_REGISTRY
    except Exception:
        return {name: "<none>" for name in TASK_PROFILES}

    for name, profile in TASK_PROFILES.items():
        try:
            ranked = BACKEND_REGISTRY.rank_for_task(profile)
            out[name] = ranked[0].id if ranked else "<none>"
        except Exception as e:
            _debug_log(f"[Crew] BackendResolver preview {name}: {e}")
            out[name] = "<none>"
    return out
