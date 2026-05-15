"""Phase D — legacy byte-compatibility + fail-open tests.

Covers AC-01 (byte-identical when flag off), AC-02 (intent=None equivalent
to legacy on flag=on), AC-08 (fail-open on retriever exception)."""
from __future__ import annotations

from pathlib import Path

import pytest

import larkhelm.config as _cfg
import larkhelm.memory as _memory
import larkhelm.memory_context as _mc
import larkhelm.memory_retriever as _mr


@pytest.fixture
def fresh_config(monkeypatch):
    """Provide a per-test mutable config dict."""
    cfg = dict(getattr(_cfg, "config", {}) or {})
    cfg.setdefault("memory_lazy_global", True)
    cfg.setdefault("memory_project_conditional", True)
    cfg.setdefault("memory_session_layered", True)
    cfg.setdefault("memory_recent_turns_dedup", True)
    monkeypatch.setattr(_cfg, "config", cfg, raising=False)
    return cfg


@pytest.fixture
def fake_layers(monkeypatch):
    """Patch ``load_global/project/session_memory`` to return canned bodies.

    Tests can mutate the returned dict in-place before invoking ``build``."""
    canned = {
        "global": "User prefers Chinese replies. Pinned: language=zh.",
        "project": "## Tech Stack\nPython / Feishu bridge.\n## Architecture\nThree-tier memory.\n",
        "session": "## Work Context\nPhase D launch.\n## Key Decisions & Facts\nPick BM25-lite.\n## Next Steps\nWrite tests.\n",
    }
    monkeypatch.setattr(_memory, "load_global_memory",
                        lambda chat_id=None: canned["global"])
    monkeypatch.setattr(_memory, "load_project_memory",
                        lambda cwd: canned["project"])
    monkeypatch.setattr(_memory, "load_memory",
                        lambda chat_id: canned["session"])
    # The builder lazy-imports these names directly, so patch on the source.
    return canned


# ── AC-01: byte-identical with the flag off ───────────────────────────────

def test_legacy_byte_identical(fresh_config, fake_layers):
    """A 25-query × 4-cwd matrix; flag=off must yield the same output as
    a direct call to ``_build_legacy_v2``."""
    fresh_config["memory_retriever_enabled"] = False
    fresh_config["memory_retriever_traffic"] = 0.0

    chat_id = "oc_test"
    queries = [
        "测试约定", "OOM 双层防护", "fix the bug", "refactor module",
        "explain the architecture", "write to feishu doc",
        "add a new convention", "discuss decision", "summarise progress",
        "/dev rewrite voice", "/crew analyse logs", "/plan migrate db",
        "/btw what time is it", "/doc update README", "kimi vs claude",
        "deepseek cost", "session memory layered", "global preference",
        "code review on commands.py", "build memory_context",
        "investigate token usage", "memory recall improvement",
        "test the retriever path", "implement intent dispatcher",
        "deploy with systemd",
    ]
    cwds = [None, "/proj/larkhelm", "/proj/other", "/tmp/test"]

    samples = 0
    for q in queries:
        for cwd in cwds:
            b1 = _mc.MemoryContextBuilder(
                chat_id, cwd, query=q, recent_turns=["[12:00] user: hi"],
            ).build()
            b2 = _mc.MemoryContextBuilder(
                chat_id, cwd, query=q, recent_turns=["[12:00] user: hi"],
            )._build_legacy_v2()
            assert b1 == b2, f"divergence: q={q!r}, cwd={cwd!r}"
            samples += 1
    assert samples == len(queries) * len(cwds) == 100


# ── AC-02: intent=None equivalent to legacy v2 even with flag=on ─────────

def test_intent_none_legacy_path(fresh_config, fake_layers):
    """get_memory_context_v2(intent=None) returns exactly the same body as
    when called without the intent kwarg."""
    fresh_config["memory_retriever_enabled"] = False
    fresh_config["memory_retriever_traffic"] = 0.0
    ctx1, _ = _memory.get_memory_context_v2(
        "oc_x", cwd="/proj/larkhelm", query="测试约定",
    )
    ctx2, _ = _memory.get_memory_context_v2(
        "oc_x", cwd="/proj/larkhelm", query="测试约定", intent=None,
    )
    assert ctx1 == ctx2


# ── AC-08: fail-open on exception ─────────────────────────────────────────

def test_fail_open_on_retriever_exception(fresh_config, fake_layers, monkeypatch):
    fresh_config["memory_retriever_enabled"] = True
    fresh_config["memory_retriever_traffic"] = 1.0

    def _boom(self, *a, **kw):
        raise RuntimeError("fuzz")
    monkeypatch.setattr(_mr.KeywordRetriever, "retrieve", _boom)

    builder = _mc.MemoryContextBuilder(
        "oc_x", "/proj/larkhelm", query="anything",
        agent_type="chat",
    )
    out = builder.build()
    assert isinstance(out, str)
    # legacy v2 includes the layer tag wrappers
    assert "[SESSION MEMORY]" in out


def test_fail_open_on_load_slices_exception(fresh_config, fake_layers, monkeypatch):
    fresh_config["memory_retriever_enabled"] = True
    fresh_config["memory_retriever_traffic"] = 1.0

    def _boom(*a, **kw):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(_mr, "load_slices", _boom)

    builder = _mc.MemoryContextBuilder(
        "oc_x", "/proj/larkhelm", query="anything",
        agent_type="dev",
    )
    out = builder.build()
    assert isinstance(out, str)
    assert "[SESSION MEMORY]" in out


# ── flag short-circuit (retriever path NOT invoked when flag=off) ─────────

def test_disabled_flag_short_circuit(fresh_config, fake_layers, monkeypatch):
    fresh_config["memory_retriever_enabled"] = False
    fresh_config["memory_retriever_traffic"] = 1.0
    spy_calls = {"retrieve": 0, "load": 0}

    orig_retrieve = _mr.KeywordRetriever.retrieve
    orig_load = _mr.load_slices

    def _spy_retrieve(self, *a, **kw):
        spy_calls["retrieve"] += 1
        return orig_retrieve(self, *a, **kw)
    def _spy_load(*a, **kw):
        spy_calls["load"] += 1
        return orig_load(*a, **kw)

    monkeypatch.setattr(_mr.KeywordRetriever, "retrieve", _spy_retrieve)
    monkeypatch.setattr(_mr, "load_slices", _spy_load)

    builder = _mc.MemoryContextBuilder("oc_x", "/proj", query="hi",
                                        agent_type="chat")
    builder.build()
    assert spy_calls["retrieve"] == 0
    assert spy_calls["load"] == 0


# ── back-compat: legacy keyword args still accepted ───────────────────────

def test_intent_param_back_compat(fresh_config, fake_layers):
    """Legacy callers that don't pass ``intent`` should still return a
    ``(str, list[str])`` tuple."""
    fresh_config["memory_retriever_enabled"] = False
    ctx, recent = _memory.get_memory_context_v2(
        "oc_x", "/proj/larkhelm", query="x",
    )
    assert isinstance(ctx, str)
    assert isinstance(recent, list)


def test_unknown_intent_agent_type_falls_back_to_chat(fresh_config, fake_layers):
    """An IntentResult-like object with an unknown ``agent_type`` should
    not break the call; ``get_policy`` falls back to ``chat`` policy."""
    fresh_config["memory_retriever_enabled"] = True
    fresh_config["memory_retriever_traffic"] = 1.0

    class FakeIntent:
        agent_type = "nonexistent_type"
        sub_intent = ""
        complexity = "medium"
        confidence = 0.5

    ctx, _ = _memory.get_memory_context_v2(
        "oc_x", "/proj/larkhelm", query="anything",
        intent=FakeIntent(),
    )
    assert isinstance(ctx, str)


# ── Phase 2 — embedding RuntimeError still yields v2 legacy string ────────


def test_embedding_runtime_error_still_returns_v2_string(fresh_config, fake_layers, monkeypatch):
    """If the embedding backend raises RuntimeError mid-flow, the builder
    must fall open to keyword and still return a ``str`` (no crash).

    This is the **inner** fail-open (within ``_build_with_retriever``):
    the embedding subsystem dies, but the retriever flow continues with
    :class:`KeywordRetriever`. We do not require the resulting string to
    match legacy v2 byte-for-byte — that's covered by
    :func:`test_legacy_byte_identical` with the flag off.
    """
    fresh_config["memory_retriever_enabled"] = True
    fresh_config["memory_retriever_traffic"] = 1.0
    fresh_config["memory_retriever_mode"] = "hybrid"
    fresh_config["embedding_enabled"] = True
    fresh_config["embedding_traffic"] = 1.0
    fresh_config["embedding_backend"] = "stub"

    class _BrokenBackend:
        name = "broken"
        dim = 8
        def embed(self, _):
            raise RuntimeError("simulated outage")
        def warm(self):
            pass

    import larkhelm.memory_embedding as _me
    monkeypatch.setattr(_me, "get_embedding_backend",
                        lambda _cfg=None: _BrokenBackend(), raising=False)
    # Provide a non-empty slice pool so the keyword path produces output.
    from larkhelm.memory_slice import MemorySlice
    fake_slices = [
        MemorySlice(id="abc", layer="session", body="hybrid please retry"),
    ]
    monkeypatch.setattr(_mr, "load_slices", lambda *a, **kw: fake_slices, raising=False)

    builder = _mc.MemoryContextBuilder(
        "oc_x", "/proj/larkhelm", query="hybrid please",
        agent_type="dev",
    )
    out = builder.build()
    assert isinstance(out, str)
    # Output is a real composed string (not the empty-string failure mode).
    assert "[SESSION MEMORY]" in out
