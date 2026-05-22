"""Tests for ``larkhelm.crew._backend_resolver``."""
from __future__ import annotations

import pytest

from larkhelm.crew_types import AgentSpec, NoBackendAvailableError


# ── TASK_PROFILES catalog sanity ─────────────────────────────────────

def test_task_profiles_has_5_entries():
    from larkhelm.crew._backend_resolver import TASK_PROFILES
    assert set(TASK_PROFILES) == {"planner", "engineer", "qa", "reviewer", "chat"}


def test_engineer_profile_requires_tools():
    from larkhelm.crew._backend_resolver import TASK_PROFILES
    assert TASK_PROFILES["engineer"].require_tools is True


def test_qa_profile_requires_tools():
    from larkhelm.crew._backend_resolver import TASK_PROFILES
    assert TASK_PROFILES["qa"].require_tools is True


def test_planner_profile_requires_tools():
    """Pinned 2026-05-22 after the DSML-leak incident: planner agents need
    Read (existing code / upstream artefacts) AND Write (prd.md, tasks.json,
    design.md). Without ``require_tools=True`` the resolver may rank an
    API-only backend (e.g. DeepSeek tags=["cheap","fast"]) at the top,
    which then emits its tool-call protocol tokens as plain text into
    the output_file and corrupts the entire downstream pipeline.
    """
    from larkhelm.crew._backend_resolver import TASK_PROFILES
    assert TASK_PROFILES["planner"].require_tools is True


def test_reviewer_profile_requires_tools():
    """Same as planner — reviewer reads the diff and writes review.md, so
    tool support is mandatory. Without the gate the DSML-leak failure
    mode reproduces in the review stage."""
    from larkhelm.crew._backend_resolver import TASK_PROFILES
    assert TASK_PROFILES["reviewer"].require_tools is True


def test_planner_resolution_excludes_api_only_backend(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """End-to-end: with the standard fixture registry (claude + kimi have
    tools, deepseek does not), resolving a planner spec must return one of
    the tool-capable backends — never deepseek. Pins the routing fix that
    keeps API-only backends from being assigned file-I/O work.
    """
    from larkhelm.crew._backend_resolver import resolve_backend
    spec = fake_agent_spec(task_profile="planner")
    out = resolve_backend(spec)
    assert out.id in ("claude", "kimi"), (
        f"planner should route to a tool-capable backend, got {out.id!r}"
    )
    assert "tools" in out.tags


def test_reviewer_resolution_excludes_api_only_backend(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """Companion to planner: reviewer must also avoid api-only backends."""
    from larkhelm.crew._backend_resolver import resolve_backend
    spec = fake_agent_spec(task_profile="reviewer")
    out = resolve_backend(spec)
    assert out.id in ("claude", "kimi")
    assert "tools" in out.tags


def test_planner_falls_through_when_no_tool_backend_available(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """If every tool-capable backend is unhealthy, the planner profile
    yields no candidates and the resolver falls through to the orchestrator
    path. Pinning this prevents a future "silently re-allow deepseek when
    claude is down" regression — failing loudly is better than corrupt
    PRDs landing on disk.
    """
    from larkhelm.crew._backend_resolver import resolve_backend
    # Mark both tool-capable backends unhealthy
    fake_backend_registry._specs["claude"].healthy = False
    fake_backend_registry._specs["kimi"].healthy = False

    spec = fake_agent_spec(task_profile="planner")
    # Path 3 fallback: orchestrator lookup also finds no healthy orchestrator
    # (claude is the only one and it's marked unhealthy) → NoBackendAvailableError.
    with pytest.raises(NoBackendAvailableError):
        resolve_backend(spec)


def test_chat_profile_latency_fast():
    from larkhelm.crew._backend_resolver import TASK_PROFILES
    assert TASK_PROFILES["chat"].latency_pref == "fast"


# ── Path 1: task_profile resolution ──────────────────────────────────

def test_resolve_with_task_profile_picks_top_ranked(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """When task_profile is set, resolver uses BACKEND_REGISTRY.rank_for_task
    and returns the first candidate."""
    from larkhelm.crew._backend_resolver import resolve_backend
    spec = fake_agent_spec(task_profile="engineer")
    out = resolve_backend(spec)
    # claude has highest coding score (0.95); should win for engineer profile
    assert out.id == "claude"


def test_resolve_with_unknown_task_profile_falls_back(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """Unknown task_profile name → falls through to legacy/orchestrator path."""
    from larkhelm.crew._backend_resolver import resolve_backend
    spec = fake_agent_spec(task_profile="nonexistent_profile")
    out = resolve_backend(spec)
    # Falls back to orchestrator (claude is the only orchestrator role here)
    assert out.role == "orchestrator"
    assert out.id == "claude"


# ── Path 2: legacy model-string dispatch ─────────────────────────────

def test_resolve_with_model_kimi_returns_kimi(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend
    spec = fake_agent_spec(model="kimi", task_profile="")
    out = resolve_backend(spec)
    assert out.id == "kimi"


def test_resolve_with_disabled_legacy_falls_through(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """Disabled legacy backend should fall through to orchestrator, not return
    a dead spec."""
    from larkhelm.crew._backend_resolver import resolve_backend
    fake_backend_registry._specs["kimi"].enabled = False
    spec = fake_agent_spec(model="kimi", task_profile="")
    out = resolve_backend(spec)
    # Falls through → orchestrator
    assert out.id == "claude"


def test_resolve_hermes_synthesizes_spec(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend
    spec = fake_agent_spec(model="hermes_race", task_profile="")
    out = resolve_backend(spec)
    assert out.provider == "hermes"
    assert out.id == "hermes_race"


# ── Path 3 + 4: orchestrator fallback / no-backend error ─────────────

def test_resolve_falls_back_to_orchestrator(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend
    spec = fake_agent_spec(model="", task_profile="")
    out = resolve_backend(spec)
    assert out.id == "claude"  # the orchestrator


def test_resolve_no_backend_raises(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """Empty registry → NoBackendAvailableError carries task_profile + reason."""
    from larkhelm.crew._backend_resolver import resolve_backend
    fake_backend_registry._specs.clear()
    spec = fake_agent_spec(task_profile="engineer")
    with pytest.raises(NoBackendAvailableError) as ei:
        resolve_backend(spec)
    assert ei.value.task_profile == "engineer"
    assert "no" in ei.value.reason.lower() or "backend" in ei.value.reason.lower()


def test_resolve_no_backend_no_orchestrator_no_fallback(
    init_test_config, fake_agent_spec, fake_backend_registry,
):
    """fallback_orchestrator=False bypasses path 3; empty profile → error."""
    from larkhelm.crew._backend_resolver import resolve_backend
    fake_backend_registry._specs.clear()
    spec = fake_agent_spec(model="", task_profile="")
    with pytest.raises(NoBackendAvailableError):
        resolve_backend(spec, fallback_orchestrator=False)


# ── Preview ──────────────────────────────────────────────────────────

def test_preview_returns_dict_with_5_keys(
    init_test_config, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend_preview
    out = resolve_backend_preview()
    assert set(out) == {"planner", "engineer", "qa", "reviewer", "chat"}


def test_preview_returns_none_for_empty_registry(
    init_test_config, fake_backend_registry,
):
    from larkhelm.crew._backend_resolver import resolve_backend_preview
    fake_backend_registry._specs.clear()
    out = resolve_backend_preview()
    for v in out.values():
        assert v == "<none>"


def test_preview_never_raises(init_test_config, fake_backend_registry):
    """Even with weird registry state, preview returns dict, never raises."""
    from larkhelm.crew._backend_resolver import resolve_backend_preview

    class _Broken:
        def rank_for_task(self, profile):
            raise RuntimeError("boom")

    import larkhelm.backend_registry as _br_mod
    orig = _br_mod.BACKEND_REGISTRY
    try:
        _br_mod.BACKEND_REGISTRY = _Broken()
        out = resolve_backend_preview()
        assert all(v == "<none>" for v in out.values())
    finally:
        _br_mod.BACKEND_REGISTRY = orig
