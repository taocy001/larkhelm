"""P1 acceptance tests — ChatAgent cheap routing (design.md §7 AC-06..08).

Asserts:
  * AC-06: with the flag on AND a cheap candidate available,
    ``ChatAgent.execute`` passes the cheap backend id as ``force_backend_id``
    to ``_do_query``.
  * AC-07: when ``rank_for_task`` returns no candidates, the agent falls
    back to ``ctx.force_backend_id`` unchanged.
  * AC-08: with the flag off, the backend selector is never invoked
    (byte-compat with master).
"""
from __future__ import annotations

import os
import threading

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import pytest

import larkhelm.config as _cfg


@pytest.fixture(autouse=True)
def _stub_chat_model(monkeypatch):
    """``_get_chat_model`` reads ``_cfg.DEFAULT_MODEL`` which is not set in
    test mode unless ``init_test_config`` ran. Stub the lookup directly so
    each test can exercise ChatAgent.execute without bootstrapping config.
    """
    from larkhelm import chat_state as cs
    monkeypatch.setattr(cs, "_get_chat_model", lambda _c: "claude",
                        raising=False)
    yield


def _make_ctx(chat_id: str = "chat_p1", force_backend_id=None):
    from larkhelm.agent_hub.intent_types import AgentContext
    return AgentContext(
        chat_id=chat_id,
        user_msg_id="m1",
        text="hello",
        images=None,
        parent_id=None,
        cancel_ev=threading.Event(),
        cwd="/tmp",
        force_backend_id=force_backend_id,
    )


def _make_intent():
    from larkhelm.agent_hub.intent_types import IntentResult
    return IntentResult(agent_type="chat", confidence=0.9, layer="L1")


# ── AC-06: cheap backend wired into _do_query ─────────────────────────────

def test_chat_routed_to_cheap_backend(monkeypatch):
    monkeypatch.setattr(_cfg, "CHAT_AGENT_CHEAP_ROUTING_ENABLED", True,
                        raising=False)
    # Fake BackendSpec returned by rank_for_task.
    class _FakeSpec:
        id = "deepseek"
        enabled = True
        healthy = True

    from larkhelm.agent_hub import model_selector
    monkeypatch.setattr(
        model_selector, "resolve_backend_for_task",
        lambda chat_id, profile, force_backend_id=None: _FakeSpec(),
    )

    captured = {}

    def fake_do_query(**kwargs):
        captured.update(kwargs)

    import larkhelm.handlers._query as _query
    monkeypatch.setattr(_query, "_do_query", fake_do_query)

    from larkhelm.agent_hub.builtin.chat_agent import ChatAgent
    result = ChatAgent().execute(_make_intent(), _make_ctx())
    assert result.success is True
    assert captured["force_backend_id"] == "deepseek", captured


# ── AC-07: rank empty → fallback to ctx force_backend_id ──────────────────

def test_fallback_when_no_cheap_backend(monkeypatch, caplog):
    monkeypatch.setattr(_cfg, "CHAT_AGENT_CHEAP_ROUTING_ENABLED", True,
                        raising=False)
    from larkhelm.agent_hub import model_selector

    def _raise(chat_id, profile, force_backend_id=None):
        # Mirrors resolve_backend's "no candidate" behaviour: rank empty
        # falls through to the legacy resolver which can raise.
        raise RuntimeError("no_backend_available")

    monkeypatch.setattr(
        model_selector, "resolve_backend_for_task", _raise,
    )

    captured = {}

    def fake_do_query(**kwargs):
        captured.update(kwargs)

    import larkhelm.handlers._query as _query
    monkeypatch.setattr(_query, "_do_query", fake_do_query)

    from larkhelm.agent_hub.builtin.chat_agent import ChatAgent
    ctx = _make_ctx(force_backend_id="original")
    result = ChatAgent().execute(_make_intent(), ctx)
    assert result.success is True
    # Should preserve the caller's force_backend_id (not None, not "deepseek").
    assert captured["force_backend_id"] == "original"


# ── AC-08: routing disabled → selector untouched ──────────────────────────

def test_routing_disabled(monkeypatch):
    monkeypatch.setattr(_cfg, "CHAT_AGENT_CHEAP_ROUTING_ENABLED", False,
                        raising=False)
    calls = []

    def _spy(chat_id, profile, force_backend_id=None):
        calls.append((chat_id, profile, force_backend_id))
        raise AssertionError("must not be called when flag is off")

    from larkhelm.agent_hub import model_selector
    monkeypatch.setattr(model_selector, "resolve_backend_for_task", _spy)

    captured = {}

    def fake_do_query(**kwargs):
        captured.update(kwargs)

    import larkhelm.handlers._query as _query
    monkeypatch.setattr(_query, "_do_query", fake_do_query)

    from larkhelm.agent_hub.builtin.chat_agent import ChatAgent
    ctx = _make_ctx(force_backend_id="passthrough")
    result = ChatAgent().execute(_make_intent(), ctx)
    assert result.success is True
    assert calls == [], "resolve_backend_for_task should NOT be called"
    assert captured["force_backend_id"] == "passthrough"


# ── helper unit test — _resolve_cheap_backend_id ───────────────────────────

def test_resolve_cheap_backend_id_returns_none_when_disabled(monkeypatch):
    from larkhelm.agent_hub.builtin import chat_agent as ca
    monkeypatch.setattr(_cfg, "CHAT_AGENT_CHEAP_ROUTING_ENABLED", False,
                        raising=False)
    assert ca._resolve_cheap_backend_id("c") is None


def test_resolve_cheap_backend_id_returns_spec_id(monkeypatch):
    from larkhelm.agent_hub.builtin import chat_agent as ca
    monkeypatch.setattr(_cfg, "CHAT_AGENT_CHEAP_ROUTING_ENABLED", True,
                        raising=False)

    class _Spec:
        id = "kimi"
        enabled = True
        healthy = True

    from larkhelm.agent_hub import model_selector
    monkeypatch.setattr(
        model_selector, "resolve_backend_for_task",
        lambda c, p, force_backend_id=None: _Spec(),
    )
    assert ca._resolve_cheap_backend_id("c") == "kimi"
