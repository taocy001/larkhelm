"""Test-mode bootstrap + shared fixtures.

The ``LARKHELM_TEST_MODE`` env var must be set BEFORE any larkhelm
module imports, so it lives at module scope. The fixtures defined
below are pure pytest helpers; they import larkhelm modules lazily so
they can be skipped on test files that don't use them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")


# ── Tiny config bootstrap ────────────────────────────────────────────
# Uses a fresh tmp_path for both DATA_DIR and the synthesized
# config.json. Marked as ``autouse=False`` so existing tests that
# import ``larkhelm.config`` directly aren't perturbed.

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a per-test data directory; sub-directories are created lazily."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def init_test_config(tmp_data_dir: Path) -> Path:
    """Initialize ``larkhelm.config`` with a temporary config.json + data dir.

    Re-runs ``_init_runtime`` per test so any per-test mutation doesn't
    bleed into the next test. Returns the config path so tests can also
    poke it.

    Bypasses live network probes via ``LARKHELM_TEST_MODE`` (set at
    module import). The synthesized config carries fake APP_ID /
    APP_SECRET so ``_init_runtime``'s required-field check passes.
    """
    import json
    cfg_path = tmp_data_dir / "config.json"
    cfg_path.write_text(json.dumps({
        "APP_ID":     "TEST_APP_ID",
        "APP_SECRET": "TEST_APP_SECRET",
        # Profile-aware tests want a fast turnaround; keep response timeout
        # small so the runner's wave loops don't accidentally wait minutes.
        "response_timeout": 30,
        "hard_timeout":     120,
    }))
    import larkhelm.config as _cfg
    _cfg._init_runtime(config_path=str(cfg_path), data_dir=str(tmp_data_dir))
    return cfg_path


@pytest.fixture
def fake_card_sender(monkeypatch):
    """Patch ``lark_client._patch_card_raw`` / ``send_card`` / ``_send_card_raw``
    to no-op recorders.

    Returns the recording list — each entry is a dict like
    ``{"kind": "patch", "mid": "...", "card": "..."}`` so tests can
    assert on the sequence of calls without hitting the live API.
    """
    recorded: list[dict] = []

    def _patch(mid, card):
        recorded.append({"kind": "patch", "mid": mid, "card": card})
        return True

    def _send_raw(chat_id, card, _fallback_text=None):
        synthetic_mid = f"fake_mid_{len(recorded)}"
        recorded.append({"kind": "send_raw", "chat_id": chat_id,
                         "card": card, "mid": synthetic_mid})
        return synthetic_mid

    def _send_card(chat_id, title, body, color="blue", note="",
                   buttons=None, normalize=True):
        synthetic_mid = f"fake_card_mid_{len(recorded)}"
        recorded.append({"kind": "send_card", "chat_id": chat_id,
                         "title": title, "body": body, "color": color,
                         "mid": synthetic_mid})
        return synthetic_mid

    def _send_card_reply(chat_id, msg_id, title, body, color="blue", note="",
                        buttons=None, normalize=True):
        synthetic_mid = f"fake_reply_mid_{len(recorded)}"
        recorded.append({"kind": "send_card_reply", "chat_id": chat_id,
                         "title": title, "body": body, "color": color,
                         "mid": synthetic_mid})
        return synthetic_mid

    def _reply_raw(message_id, card_json, in_thread=True):
        synthetic_mid = f"fake_reply_raw_mid_{len(recorded)}"
        recorded.append({"kind": "reply_raw", "message_id": message_id,
                         "card": card_json, "mid": synthetic_mid})
        return synthetic_mid

    import larkhelm.lark_client as _lc
    monkeypatch.setattr(_lc, "_patch_card_raw", _patch)
    monkeypatch.setattr(_lc, "_send_card_raw", _send_raw)
    monkeypatch.setattr(_lc, "send_card", _send_card)
    monkeypatch.setattr(_lc, "send_card_reply", _send_card_reply, raising=False)
    monkeypatch.setattr(_lc, "_reply_card_raw", _reply_raw, raising=False)

    return recorded


@pytest.fixture
def fake_backend_registry(monkeypatch):
    """Build a fresh :class:`BackendRegistry` populated with controllable specs
    and patch the module-level singleton(s) for the test scope.

    The fixture returns the registry instance so tests can mutate ``enabled``
    / ``healthy`` / ``capability_scores`` per spec to drive resolver paths.
    """
    from larkhelm.backend_registry import BackendRegistry, BackendSpec
    import larkhelm.backend_registry as _br_mod

    reg = BackendRegistry()
    # Default: 3 backends — orchestrator (claude + kimi), worker (deepseek)
    # Kimi is orchestrator because provider=kimi_cli + tags=[tools] (matches auto-inference rule).
    # Deepseek stays worker: no tools tag, API-only backend.
    reg._specs["claude"] = BackendSpec(
        id="claude", provider="claude_cli", display_name="Claude",
        role="orchestrator", tags=["tools", "vision"],
        capability_scores={"reasoning": 0.95, "coding": 0.95,
                           "long_context": 0.9, "tools": 0.95, "chat": 0.9},
        healthy=True, enabled=True, command="claude",
    )
    reg._specs["kimi"] = BackendSpec(
        id="kimi", provider="kimi_cli", display_name="Kimi",
        role="orchestrator", tags=["tools"],
        capability_scores={"reasoning": 0.8, "coding": 0.85,
                           "long_context": 0.85, "tools": 0.8, "chat": 0.85},
        healthy=True, enabled=True, command="kimi",
    )
    reg._specs["deepseek"] = BackendSpec(
        id="deepseek", provider="deepseek_api", display_name="DeepSeek",
        role="worker", tags=["cheap", "fast"],
        capability_scores={"reasoning": 0.7, "coding": 0.75,
                           "long_context": 0.65, "chat": 0.85},
        healthy=True, enabled=True, api_key="sk-test",
    )

    monkeypatch.setattr(_br_mod, "BACKEND_REGISTRY", reg)
    # config also caches its own reference
    try:
        import larkhelm.config as _cfg
        monkeypatch.setattr(_cfg, "BACKEND_REGISTRY", reg, raising=False)
    except Exception:
        pass
    return reg


# ── sys.modules injection fixtures (P0-3 REQ-09/10) ─────────────────
# Replace the legacy sys.modules-mocking test idiom with monkeypatch-backed
# fixtures. Auto-teardown is guaranteed by pytests monkeypatch lifecycle.

@pytest.fixture
def inject_module(monkeypatch):
    """Inject a fake module into ``sys.modules`` with automatic teardown.

    Usage::

        def test_x(inject_module):
            inject_module("anthropic", fake_mod)
            inject_module("anthropic.types", fake_types)
    """
    def _inject(name: str, module: object) -> None:
        monkeypatch.setitem(sys.modules, name, module)
    return _inject


@pytest.fixture
def unload_module(monkeypatch):
    """Make ``import <name>`` raise ImportError with automatic teardown.

    Setting ``sys.modules[name] = None`` is the documented way to make the
    import system refuse to (re)import a module — Python raises
    ``ModuleNotFoundError`` on the next ``import name`` statement. Reserved
    for tests that intentionally exercise the import-failure branch (e.g.
    ``onnxruntime`` missing, ``larkhelm.backend_api`` not available during
    agent_hub bootstrap).
    """
    def _unload(name: str) -> None:
        monkeypatch.setitem(sys.modules, name, None)
    return _unload
