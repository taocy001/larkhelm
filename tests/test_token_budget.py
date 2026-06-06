"""Tests for ``larkhelm.token_budget`` (Week-2 Backend-aware Context Budget)."""
from __future__ import annotations

import dataclasses

import pytest

import larkhelm.config as _cfg
@dataclasses.dataclass
class InjectionPolicy:
    agent_type: str
    token_budget: int
    layer_weights: dict = dataclasses.field(default_factory=dict)
    kind_priority: tuple = ()
from larkhelm.token_budget import (
    DEFAULT_CONTEXT_WINDOWS,
    _MIN_CONTEXT_WINDOW,
    compute_api_max_tokens,
    compute_memory_char_budget,
    resolve_context_window,
)

# Ensure a mutable config dict exists for monkeypatch tests.
if not hasattr(_cfg, "config"):
    _cfg.config = {}


# ── Minimal BackendSpec stub (avoids importing the full registry graph) ─────

@dataclasses.dataclass
class _StubSpec:
    id: str
    provider: str
    model: str = ""
    context_window: int = 0


# ── resolve_context_window ───────────────────────────────────────────────────

def test_resolve_default_by_id():
    spec = _StubSpec(id="claude", provider="claude_cli")
    assert resolve_context_window(spec) == DEFAULT_CONTEXT_WINDOWS["claude"]


def test_resolve_default_by_provider():
    # id unknown, provider known
    spec = _StubSpec(id="unknown", provider="anthropic_api")
    assert resolve_context_window(spec) == DEFAULT_CONTEXT_WINDOWS["anthropic_api"]


def test_resolve_fallback():
    spec = _StubSpec(id="weird", provider="weird_provider")
    assert resolve_context_window(spec) == _MIN_CONTEXT_WINDOW


def test_resolve_none():
    assert resolve_context_window(None) == _MIN_CONTEXT_WINDOW


def test_resolve_zero_config_falls_back_to_builtin(monkeypatch):
    """config[context_window_claude]=0 must fall through to DEFAULT_CONTEXT_WINDOWS, not 32K floor.

    This is the critical contract fixed in the Week-2 bug-fix commit:
    config.py setdefaults all context_window_* to 0 meaning "use built-in default".
    """
    monkeypatch.setitem(_cfg.config, "context_window_claude", 0)
    spec = _StubSpec(id="claude", provider="claude_cli", context_window=0)
    assert resolve_context_window(spec) == DEFAULT_CONTEXT_WINDOWS["claude"]


def test_resolve_positive_config_overrides_builtin(monkeypatch):
    """Operator-set positive value overrides the built-in default."""
    monkeypatch.setitem(_cfg.config, "context_window_claude", 300_000)
    spec = _StubSpec(id="claude", provider="claude_cli", context_window=0)
    assert resolve_context_window(spec) == 300_000


# ── compute_memory_char_budget ───────────────────────────────────────────────

def test_budget_disabled_flag_returns_base():
    """When backend_aware_budget_enabled is False, base budget is returned."""
    spec = _StubSpec(id="deepseek", provider="deepseek_api")
    # flag off → unchanged
    assert compute_memory_char_budget(spec, "chat", base_budget=1200) == 1200


def _set_budget_enabled(enabled: bool) -> None:
    """Helper: flip the runtime feature flag."""
    _cfg.config["backend_aware_budget_enabled"] = enabled


def test_budget_large_context_scales_up():
    """Gemini 1M window → +20 % over base."""
    _set_budget_enabled(True)
    try:
        spec = _StubSpec(id="gemini", provider="gemini_cli")
        budget = compute_memory_char_budget(spec, "chat", base_budget=1200)
        assert budget == int(1200 * 1.20)
    finally:
        _set_budget_enabled(False)


def test_budget_small_context_scales_down():
    """Tiny window (<64K) → -30 % under base."""
    _set_budget_enabled(True)
    try:
        spec = _StubSpec(id="tiny", provider="tiny")
        spec.context_window = 32_000
        budget = compute_memory_char_budget(spec, "chat", base_budget=1200)
        assert budget == int(1200 * 0.70)
    finally:
        _set_budget_enabled(False)


def test_budget_never_below_floor():
    """Even with tiny window and tiny base, floor is respected."""
    _set_budget_enabled(True)
    try:
        spec = _StubSpec(id="tiny", provider="tiny")
        budget = compute_memory_char_budget(spec, "chat", base_budget=100)
        assert budget >= 400
    finally:
        _set_budget_enabled(False)


def test_budget_none_spec():
    assert compute_memory_char_budget(None, "dev", base_budget=3000) == 3000


# ── compute_api_max_tokens ───────────────────────────────────────────────────

def test_api_max_tokens_large_window():
    """Claude 200K → cap 8192 is the limiting factor."""
    spec = _StubSpec(id="claude", provider="claude_cli")
    mt = compute_api_max_tokens(spec, input_tokens_est=8000)
    assert mt == 8192


def test_api_max_tokens_small_window_tight():
    """Tiny window → max_tokens shrinks to leave headroom."""
    spec = _StubSpec(id="tiny", provider="tiny")
    spec.context_window = 10_000
    mt = compute_api_max_tokens(spec, input_tokens_est=8000, min_output=256)
    # safe_budget = 8500, available = 500, clamped to min_output = 256
    assert mt == 500


def test_api_max_tokens_respects_min_output():
    spec = _StubSpec(id="tiny", provider="tiny")
    spec.context_window = 8_000
    mt = compute_api_max_tokens(spec, input_tokens_est=8000, min_output=256)
    assert mt == 256


def test_api_max_tokens_none_spec():
    assert compute_api_max_tokens(None) == 8192


# ── _fmt_token_block context window annotation (AC-07) ───────────────────────

def _make_stats_row(**kwargs):
    defaults = dict(input_tokens=100, output_tokens=50, cache_read=0, cache_create=0, calls=1, cost_usd=0.001)
    defaults.update(kwargs)
    return defaults


def test_fmt_token_block_shows_ctx_when_enabled(monkeypatch):
    """When backend_aware_budget_enabled=true, model label shows Xk ctx."""
    import larkhelm.config as _cfg
    from larkhelm.commands import _fmt_token_block
    from larkhelm.backend_registry import BACKEND_REGISTRY

    monkeypatch.setitem(_cfg.config, "backend_aware_budget_enabled", True)
    try:
        spec = BACKEND_REGISTRY.get("gemini")
        if spec is None:
            # Registry may not be initialised in test env; verify fail-open only.
            body = _fmt_token_block("label", {"gemini": _make_stats_row()})
            # Should render without crash; model name present even if no ctx annotation
            assert "gemini" in body
        else:
            body = _fmt_token_block("label", {"gemini": _make_stats_row()})
            assert "ctx）" in body or "K ctx" in body
    finally:
        monkeypatch.setitem(_cfg.config, "backend_aware_budget_enabled", False)


def test_fmt_token_block_no_ctx_when_disabled(monkeypatch):
    """When backend_aware_budget_enabled=false (default), no ctx annotation."""
    import larkhelm.config as _cfg
    from larkhelm.commands import _fmt_token_block

    monkeypatch.setitem(_cfg.config, "backend_aware_budget_enabled", False)
    body = _fmt_token_block("label", {"claude": _make_stats_row()})
    assert "ctx）" not in body


def test_fmt_token_block_failopen_on_registry_error(monkeypatch):
    """If BACKEND_REGISTRY.get() raises, label falls back to bare model name."""
    import larkhelm.config as _cfg
    import larkhelm.backend_registry as _breg
    from larkhelm.commands import _fmt_token_block

    monkeypatch.setitem(_cfg.config, "backend_aware_budget_enabled", True)

    class _BrokenRegistry:
        def get(self, _id):
            raise RuntimeError("registry broken")

    monkeypatch.setattr(_breg, "BACKEND_REGISTRY", _BrokenRegistry())
    try:
        body = _fmt_token_block("label", {"claude": _make_stats_row()})
        # fail-open: no crash, bare model name present
        assert "claude" in body
        assert "ctx）" not in body
    finally:
        monkeypatch.setitem(_cfg.config, "backend_aware_budget_enabled", False)


# ── BackendSpec.context_window auto-population (R2 coverage) ─────────────────

def test_backend_spec_context_window_auto_populated():
    """BackendRegistry.load() must call resolve_context_window and store the result."""
    from larkhelm.backend_registry import BackendRegistry
    reg = BackendRegistry()
    # Use "claude" so DEFAULT_CONTEXT_WINDOWS["claude"] is the expected fallback.
    reg.load([{
        "id": "claude",
        "provider": "claude_cli",
        "display_name": "Test Claude",
        "cmd": "claude",
        "model": "",
    }])
    spec = reg.get("claude")
    assert spec is not None
    assert spec.context_window > 0, "context_window must be resolved at load time"
    assert spec.context_window == DEFAULT_CONTEXT_WINDOWS["claude"]


# ── Adapter max_tokens wiring integration (R6 coverage) ──────────────────────

def test_anthropic_adapter_uses_compute_api_max_tokens():
    """AnthropicAdapter.prepare_request must set max_tokens from compute_api_max_tokens."""
    from larkhelm.backend_api_streaming import AnthropicAdapter
    from larkhelm.token_budget import compute_api_max_tokens

    spec = _StubSpec(id="claude", provider="anthropic_api", context_window=200_000)
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)

    kwargs = adapter.prepare_request(
        spec=spec,
        history=[],
        message="hello",
        extra_system="",
    )
    expected = compute_api_max_tokens(spec)
    assert kwargs["max_tokens"] == expected, (
        f"AnthropicAdapter max_tokens {kwargs['max_tokens']} != "
        f"compute_api_max_tokens {expected}"
    )


def test_openai_compat_adapter_uses_compute_api_max_tokens():
    """OpenAICompatAdapter.prepare_request must set max_tokens from compute_api_max_tokens."""
    from larkhelm.backend_api_streaming import OpenAICompatAdapter
    from larkhelm.token_budget import compute_api_max_tokens

    spec = _StubSpec(id="deepseek_api", provider="openai_compat_api", context_window=64_000)
    adapter = OpenAICompatAdapter.__new__(OpenAICompatAdapter)

    payload = adapter.prepare_request(
        spec=spec,
        history=[],
        message="hello",
        extra_system="",
    )
    expected = compute_api_max_tokens(spec)
    assert payload["max_tokens"] == expected, (
        f"OpenAICompatAdapter max_tokens {payload['max_tokens']} != "
        f"compute_api_max_tokens {expected}"
    )
