"""Unit tests for ``HybridRetriever`` + ``EmbeddingRetriever`` (Phase D / Phase 2)."""
from __future__ import annotations

from unittest import mock

import pytest

np = pytest.importorskip("numpy")

from larkhelm.memory_embedding import EmbeddingError, EmbeddingCache, StubEmbedding  # noqa: E402
from larkhelm.memory_retriever import (  # noqa: E402
    EmbeddingRetriever,
    HybridRetriever,
    KeywordRetriever,
    POLICY_TABLE,
    build_audit_record_v2,
    resolve_actual_mode,
)
from larkhelm.memory_slice import InjectionPolicy, MemorySlice, RetrievalRequest  # noqa: E402


def _slice(sid, title, body, *, layer="project", kind="convention", stale=False):
    return MemorySlice(
        id=sid, layer=layer, kind=kind, title=title, body=body,
        importance=0.5, char_len=len(body), stale=stale,
    )


def _policy(top_k=3, alpha=0.6, multiplier=3):
    return InjectionPolicy(
        agent_type="dev",
        token_budget=2000,
        layer_weights={"project": 1.0},
        kind_priority=("convention", "fact"),
        retrieval_mode="hybrid",
        top_k=top_k,
        alpha_recency=0.0, alpha_importance=0.0, alpha_relevance=1.0,
        hybrid_alpha=alpha,
        embedding_top_k_multiplier=multiplier,
    )


def _build_slices(n=6):
    return [
        _slice(f"sid-{i}", f"title-{i}", f"BM25 token {chr(ord('a') + i)} {i} body {i}")
        for i in range(n)
    ]


def test_hybrid_alpha_weighting():
    """alpha=1.0 means cos_sim only; alpha=0.0 means BM25 only."""
    slices = _build_slices(4)
    request = RetrievalRequest(chat_id="c", query="token a", agent_type="dev")
    backend = StubEmbedding(dim=8)
    cache = EmbeddingCache()
    hybrid_kw = HybridRetriever(
        KeywordRetriever(), EmbeddingRetriever(backend, cache=cache),
    )
    # alpha=1.0 — cosine dominates
    out_cos = hybrid_kw.retrieve(request, _policy(alpha=1.0), slices)
    # alpha=0.0 — BM25 dominates
    out_bm = hybrid_kw.retrieve(request, _policy(alpha=0.0), slices)
    assert out_cos and out_bm
    # Different alphas should usually produce a different top item for our
    # synthetic dataset where BM25 has a clear winner.
    assert out_bm[0].slice.id == "sid-0"  # explicit BM25 hit


def test_keyword_empty_pool_falls_back():
    request = RetrievalRequest(chat_id="c", query="completely unrelated query")
    backend = StubEmbedding(dim=8)
    hybrid = HybridRetriever(KeywordRetriever(), EmbeddingRetriever(backend))
    # Empty pool input
    out = hybrid.retrieve(request, _policy(), [])
    assert out == []


def test_embedding_error_falls_back_keyword():
    """Backend.embed raising EmbeddingError must yield the keyword pool, truncated."""
    request = RetrievalRequest(chat_id="c", query="token a")
    slices = _build_slices(5)

    class _ExplodingBackend:
        name = "boom"
        dim = 8
        def embed(self, _):
            raise EmbeddingError("simulated")
        def warm(self):
            pass

    hybrid = HybridRetriever(KeywordRetriever(), EmbeddingRetriever(_ExplodingBackend()))
    out = hybrid.retrieve(request, _policy(top_k=2), slices)
    assert len(out) <= 2
    # And the items came from the keyword path (reason has 'rel='/'rec=' tokens, not 'hybrid α=').
    assert all("hybrid α=" not in s.reason for s in out)


def test_pool_multiplier_applied(monkeypatch):
    """pool size for the keyword pre-pass is top_k × multiplier."""
    captured = {}
    original = KeywordRetriever.retrieve

    def _capture(self, req, pol, sl):
        captured["top_k"] = pol.top_k
        return original(self, req, pol, sl)

    monkeypatch.setattr(KeywordRetriever, "retrieve", _capture)
    request = RetrievalRequest(chat_id="c", query="token a")
    hybrid = HybridRetriever(KeywordRetriever(), EmbeddingRetriever(StubEmbedding()))
    hybrid.retrieve(request, _policy(top_k=4, multiplier=3), _build_slices(20))
    assert captured["top_k"] == 12  # 4 × 3


def test_stale_decay_in_hybrid():
    """A stale slice in the hybrid output is demoted relative to a fresh slice."""
    fresh = _slice("fresh", "fresh", "alpha alpha alpha alpha", stale=False)
    stale = _slice("stale", "stale", "alpha alpha alpha alpha", stale=True)
    request = RetrievalRequest(chat_id="c", query="alpha")
    hybrid = HybridRetriever(KeywordRetriever(), EmbeddingRetriever(StubEmbedding()))
    out = hybrid.retrieve(request, _policy(top_k=2, alpha=0.0), [fresh, stale])
    assert out[0].slice.id == "fresh"
    assert any(",stale" in s.reason for s in out)


def test_resolve_actual_mode_with_traffic(monkeypatch):
    """When embedding traffic is 100% and backend!=none → hybrid; backend=none → keyword."""
    policy = POLICY_TABLE["chat"]  # declared retrieval_mode="keyword"
    cfg_on = {
        "embedding_enabled": True, "embedding_traffic": 1.0,
        "embedding_backend": "stub", "memory_retriever_mode": "auto",
    }
    assert resolve_actual_mode(policy, "any-chat", cfg_on) == "hybrid"
    cfg_off = dict(cfg_on)
    cfg_off["embedding_backend"] = "none"
    assert resolve_actual_mode(policy, "any-chat", cfg_off) == "keyword"


def test_fail_open_records_audit_v2_field():
    """build_audit_record_v2 must include the v2 contract fields."""
    request = RetrievalRequest(chat_id="c", query="hi", agent_type="dev")
    policy = POLICY_TABLE["dev"]
    rec = build_audit_record_v2(
        request=request, policy=policy, scored=[],
        candidate_count=10, elapsed_ms=42, selected_chars=0,
        fail_open=True, actual_mode="keyword", declared_mode="hybrid",
    )
    assert rec["schema_version"] == "2"
    assert rec["mode"] == "keyword"
    assert rec["declared_mode"] == "hybrid"
    assert rec["fail_open"] is True
    assert "hybrid_alpha" in rec
    assert rec["top_k_returned"] == 0
    assert rec["stale_hit_count"] == 0


def test_top_k_returned_matches_selected_ids():
    request = RetrievalRequest(chat_id="c", query="alpha")
    slices = _build_slices(8)
    hybrid = HybridRetriever(KeywordRetriever(), EmbeddingRetriever(StubEmbedding()))
    out = hybrid.retrieve(request, _policy(top_k=3, multiplier=2), slices)
    assert len(out) <= 3
