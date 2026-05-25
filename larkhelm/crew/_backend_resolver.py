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
        # P1 (design.md §3.3): cap the per-call estimate at $0.10 so
        # ``rank_for_task`` filters out expensive backends (Claude Opus,
        # Kimi-thinking, etc.) for plain chat — DeepSeek / smaller models
        # become the natural top pick.
        cost_ceiling=0.10,
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


def _backend_has_tools(spec: "BackendSpec") -> bool:
    """Return True iff ``spec`` advertises ``"tools"`` in its tags.

    F3 (2026-05-25): mirrors the gate inside
    :meth:`BackendRegistry.rank_for_task` so Path 2 (legacy direct
    dispatch by ``spec.model`` string) honours the same tool-capability
    contract as Path 1 (task_profile-based ranking). Backends without
    a tool layer (API-only DeepSeek, etc.) cannot honour an agent's
    Read / Write contract — they emit the protocol tokens as plain text
    and silently corrupt ``output_file``.
    """
    try:
        return "tools" in (spec.tags or [])
    except Exception:
        return False


def resolve_backend(
    spec: AgentSpec,
    fallback_orchestrator: bool = True,
    exclude_backend_ids: "frozenset[str] | None" = None,
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

    ``exclude_backend_ids`` (F4 2026-05-25): a per-call set of backend
    ids to skip across Path 1 / 2 / 3. Lets the runner's retry loop
    re-route around a backend that just emitted a contract-violating
    artifact, without poisoning the registry's global health state for
    other concurrent agents.

    Importing ``BACKEND_REGISTRY`` lazily inside the function keeps this
    module safe to import from ``crew_types`` consumers without hauling
    the registry's startup chain into early bootstrap.
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY
    excluded = frozenset(exclude_backend_ids or ())

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
            if excluded:
                ranked = [s for s in ranked if s.id not in excluded]
            if ranked:
                return ranked[0]
            _debug_log(
                f"[Crew] BackendResolver: rank_for_task({profile_name}) returned "
                "no candidates; falling through"
            )

    # Path 2 — legacy direct-dispatch model strings
    if _is_legacy_direct(spec.model) and spec.model not in excluded:
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
            # F3 (2026-05-25): require_tools gate. Path 1 has had this
            # check since the 2026-05-22 DSML leak; Path 2 (Manager-LLM
            # emitting explicit ``model="deepseek"``) bypassed it. The
            # 2026-05-25 DocsExpert failure was exactly this: Manager
            # planned the docs-review agent with model="deepseek" +
            # output_file="review_docs.md"; DeepSeek can't tool_use, so
            # it dumped DSML tokens as plain text, safety-net captured
            # those as the file, validator quarantined. Refuse the
            # dispatch upfront when the agent has a declared output_file
            # but the chosen backend lacks tool capability — let the
            # caller decide to abort / fall through to orchestrator
            # rather than burning 5 min on a guaranteed corrupt run.
            if spec.output_file and not _backend_has_tools(legacy):
                _debug_log(
                    f"[Crew] BackendResolver: refusing legacy model={spec.model!r} "
                    f"for agent with output_file={spec.output_file!r} — backend "
                    f"lacks tool capability (would leak protocol tokens). "
                    "Falling through to orchestrator."
                )
                # Intentional fall-through to Path 3.
            else:
                return legacy
        else:
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
        if orch is not None and orch.id not in excluded:
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
