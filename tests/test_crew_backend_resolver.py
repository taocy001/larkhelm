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
