"""Phase D — tests for ``larkhelm.memory_retriever`` (Phase 1 recall core).

Covers REQ-02 / REQ-05–10 + AC-03 / AC-04 / AC-05 / AC-06:
policy table coverage, BM25-lite + entity boost, recency decay τ per layer,
require/exclude kinds, compose tag order, gating consistency with the
intent router, and the P99 performance budget."""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone

import pytest

import larkhelm.config as _cfg
from larkhelm._gating import hash_traffic_active
from larkhelm.memory_retriever import (
    POLICY_TABLE,
    KeywordRetriever,
    _bm25_lite_score,
    _compute_idf,
    _recency_score,
    _retriever_active,
    compose_slices_to_context,
    get_policy,
)
from larkhelm.memory_slice import (
    InjectionPolicy,
    MemorySlice,
    RetrievalRequest,
    ScoredSlice,
)


# ── 1. Policy table ────────────────────────────────────────────────────────

def test_policy_table_coverage():
    """AC-04: 6 agent_types and 3-factor weights sum to 1.0 ± 0.01."""
    expected_keys = {"chat", "btw", "dev", "crew", "plan", "doc"}
    assert set(POLICY_TABLE.keys()) == expected_keys
    for key, policy in POLICY_TABLE.items():
        total = policy.alpha_recency + policy.alpha_importance + policy.alpha_relevance
        assert abs(total - 1.0) <= 0.01, f"{key} sums to {total}"


def test_get_policy_unknown_fallback():
    p = get_policy("unknown_agent")
    assert p.agent_type == "chat"


# ── 2. BM25-lite scoring ───────────────────────────────────────────────────

def _slice_at(
    *,
    layer="project",
    title="",
    body="",
    kind="fact",
    importance=0.5,
    updated_at="",
    entities=(),
    keywords=(),
) -> MemorySlice:
    return MemorySlice(
        id=f"id_{abs(hash((layer, title, body))) % 10**8}",
        layer=layer,
        kind=kind,
        title=title,
        body=body,
        importance=importance,
        updated_at=updated_at,
        entities=entities,
        keywords=keywords,
        char_len=len(body),
    )


def test_bm25_relevance_query_term_match():
    """Slice mentioning OOM should outscore an unrelated slice for that query."""
    slices = [
        _slice_at(title="OOM 双层防护", body="cgroup memory.max=2.8G, V8 voice"),
        _slice_at(title="测试约定", body="集成测试 mock urllib"),
    ]
    req = RetrievalRequest(chat_id="c1", query="OOM voice protection")
    policy = get_policy("dev")
    scored = KeywordRetriever().retrieve(req, policy, slices)
    # OOM slice should win
    assert len(scored) == 2
    top = scored[0].slice
    assert "OOM" in top.title


# ── 3. Recency decay per layer ─────────────────────────────────────────────

def test_recency_decay_layer_tau():
    """Same Δdays but different layer τ → different recency scores."""
    from datetime import timedelta
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    thirty_days_ago = (now - timedelta(days=30)).isoformat()

    session_slice = _slice_at(layer="session", body="x", updated_at=thirty_days_ago)
    project_slice = _slice_at(layer="project", body="x", updated_at=thirty_days_ago)
    global_slice = _slice_at(layer="global", body="x", updated_at=thirty_days_ago)

    rs = _recency_score(session_slice, now)
    rp = _recency_score(project_slice, now)
    rg = _recency_score(global_slice, now)
    # Smaller τ (session=14) decays faster than project (30) or global (90)
    assert rs < rp < rg


def test_importance_default_neutral():
    """A slice with frontmatter-absent importance defaults to 0.5."""
    s = _slice_at(body="x")
    assert s.importance == 0.5


# ── 4. require / exclude kinds ─────────────────────────────────────────────

def test_require_kinds_force_include():
    pref = _slice_at(layer="global", body="boring", kind="preference",
                     importance=0.0)
    other = _slice_at(layer="project", body="not relevant", kind="fact",
                      importance=0.0)
    policy = InjectionPolicy(
        agent_type="chat",
        token_budget=1200,
        layer_weights={"session": 0.5, "global": 0.4, "project": 0.1},
        kind_priority=("preference", "fact"),
        require_kinds=("preference",),
        top_k=2,
        alpha_recency=0.3, alpha_importance=0.4, alpha_relevance=0.3,
    )
    req = RetrievalRequest(chat_id="c1", query="totally unrelated query")
    scored = KeywordRetriever().retrieve(req, policy, [pref, other])
    ids = {s.slice.kind for s in scored}
    assert "preference" in ids


def test_exclude_kinds_drop():
    inc = _slice_at(layer="project", body="incident-like", kind="incident")
    fact = _slice_at(layer="project", body="boring fact", kind="fact")
    policy = InjectionPolicy(
        agent_type="doc",
        token_budget=800,
        layer_weights={"project": 1.0},
        kind_priority=("fact",),
        exclude_kinds=("incident",),
        top_k=5,
        alpha_recency=0.2, alpha_importance=0.3, alpha_relevance=0.5,
    )
    req = RetrievalRequest(chat_id="c1", query="boring")
    scored = KeywordRetriever().retrieve(req, policy, [inc, fact])
    kinds = {s.slice.kind for s in scored}
    assert "incident" not in kinds


# ── 5. Entity boost ────────────────────────────────────────────────────────

def test_entity_boost_filepath():
    """AC-03 sub-case: slice with matching file-path entity scores high."""
    target = _slice_at(
        layer="project",
        title="OOM fix",
        body="cgroup tweaks for voice runner",
        entities=("larkhelm/voice/voice_runner.py",),
        kind="incident",
    )
    distractor = _slice_at(
        layer="project",
        title="unrelated",
        body="docs about something else",
        kind="fact",
    )
    req = RetrievalRequest(
        chat_id="c1",
        query="fix OOM in larkhelm/voice/voice_runner.py",
    )
    policy = get_policy("dev")
    scored = KeywordRetriever().retrieve(req, policy, [target, distractor])
    ids = [s.slice.id for s in scored[:3]]
    assert target.id in ids
    # Boosted slice should be the leader.
    assert scored[0].slice.id == target.id


def test_entity_boost_module_name():
    target = _slice_at(
        layer="project",
        body="changes to memory module",
        entities=("larkhelm.memory",),
    )
    req = RetrievalRequest(
        chat_id="c1",
        query="please look at larkhelm.memory loading",
    )
    policy = get_policy("dev")
    scored = KeywordRetriever().retrieve(req, policy, [target])
    assert scored
    # entity boost multiplies relevance to non-zero
    assert scored[0].relevance_score > 0


# ── 6. Compose ─────────────────────────────────────────────────────────────

def test_compose_tag_order_global_project_session():
    g = _slice_at(layer="global", title="G", body="g body")
    p = _slice_at(layer="project", title="P", body="p body")
    s = _slice_at(layer="session", title="S", body="s body")
    scored = [
        ScoredSlice(slice=g, score=0.9),
        ScoredSlice(slice=p, score=0.8),
        ScoredSlice(slice=s, score=0.7),
    ]
    policy = get_policy("dev")
    out = compose_slices_to_context(scored, policy, cwd="/proj")
    g_idx = out.index("[GLOBAL MEMORY]")
    p_idx = out.index("[PROJECT MEMORY — /proj]")
    s_idx = out.index("[SESSION MEMORY]")
    assert g_idx < p_idx < s_idx
    assert "[/GLOBAL MEMORY]" in out
    assert "[/PROJECT MEMORY]" in out
    assert "[/SESSION MEMORY]" in out


def test_compose_budget_trim_uses_smart_truncate():
    """When a layer's slice content exceeds its budget share, smart_truncate
    is applied (the output ends in an ellipsis marker)."""
    big = "x" * 5000
    s = _slice_at(layer="session", title="huge", body=big)
    scored = [ScoredSlice(slice=s, score=0.9)]
    # Very small budget → truncation must fire
    policy = InjectionPolicy(
        agent_type="chat",
        token_budget=200,
        layer_weights={"session": 1.0},
        kind_priority=("fact",),
        alpha_recency=0.3, alpha_importance=0.3, alpha_relevance=0.4,
    )
    out = compose_slices_to_context(scored, policy)
    # Ellipsis applied (either inline "…" or "\n…")
    assert "…" in out
    # And the body is shorter than the raw 5000 chars.
    assert len(out) < 5000


# ── 7. Gating consistency with intent_router (AC-05) ───────────────────────

def test_traffic_split_consistent_with_intent_router(monkeypatch):
    """`_retriever_active` and the intent_router gating must agree on the
    same chat_id at the same traffic % (NFR-DEPLOY-1)."""
    # Save and restore config to keep the test hermetic.
    orig = getattr(_cfg, "config", {})
    new_cfg = dict(orig)
    new_cfg.update({
        "intent_router_enabled":   True,
        "intent_router_traffic":   0.5,
        "memory_retriever_enabled": True,
        "memory_retriever_traffic": 0.5,
    })
    monkeypatch.setattr(_cfg, "config", new_cfg, raising=False)
    chats = [f"oc_{i:03d}" for i in range(100)]
    for c in chats:
        a = hash_traffic_active(c, "intent_router_enabled", "intent_router_traffic")
        b = _retriever_active(c)
        assert a == b, f"divergence at {c}: ir={a} mr={b}"


def test_traffic_split_extremes(monkeypatch):
    orig = getattr(_cfg, "config", {})
    cfg = dict(orig)
    cfg.update({"memory_retriever_enabled": True, "memory_retriever_traffic": 0.0})
    monkeypatch.setattr(_cfg, "config", cfg, raising=False)
    assert _retriever_active("c1") is False

    cfg2 = dict(orig)
    cfg2.update({"memory_retriever_enabled": True, "memory_retriever_traffic": 1.0})
    monkeypatch.setattr(_cfg, "config", cfg2, raising=False)
    assert _retriever_active("c1") is True

    cfg3 = dict(orig)
    cfg3.update({"memory_retriever_enabled": False, "memory_retriever_traffic": 1.0})
    monkeypatch.setattr(_cfg, "config", cfg3, raising=False)
    assert _retriever_active("c1") is False


# ── 8. P99 performance budget (AC-06) ──────────────────────────────────────

def test_perf_p99_under_20ms():
    """100 slice pool × 20 queries × 20 rounds; P99 retrieve time < 20 ms."""
    slices = []
    for i in range(100):
        slices.append(_slice_at(
            layer=("project" if i % 2 else "session"),
            title=f"Topic {i}",
            body=f"text body {i} oom voice retry token deepseek " * 4,
            kind=("convention" if i % 3 == 0 else "fact"),
        ))
    queries = [
        "oom voice retry", "deepseek backend", "memory schema",
        "session token", "fix the bug", "convention add",
        "voice runner OOM", "project memory layer", "chat policy",
        "intent router gating", "memory retriever budget",
        "load slices H2", "compose context", "policy table cover",
        "BM25 lite", "entity boost path", "Chinese tokenisation",
        "audit jsonl", "schema version", "kimi long context",
    ]
    policy = get_policy("dev")
    durations: list[float] = []
    retriever = KeywordRetriever()
    for _ in range(20):
        for q in queries:
            req = RetrievalRequest(chat_id="c1", query=q)
            t0 = time.perf_counter()
            retriever.retrieve(req, policy, slices)
            durations.append((time.perf_counter() - t0) * 1000.0)
    durations.sort()
    p99 = durations[int(0.99 * (len(durations) - 1))]
    assert p99 < 20.0, f"P99 was {p99:.2f}ms"


# ── 9. Helpers (idf / bm25_lite) ───────────────────────────────────────────

def test_compute_idf_empty():
    idf, avgdl = _compute_idf([])
    assert idf == {}
    assert avgdl == 0.0


def test_bm25_lite_no_query_terms():
    score = _bm25_lite_score([], ["a", "b"], {}, 2.0)
    assert score == 0.0


# ── 10. Audit (smoke) ──────────────────────────────────────────────────────

def test_audit_decision_smoke(tmp_path, monkeypatch):
    """_audit_decision should not raise even with an empty record."""
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(_cfg, "config", {"memory_retriever_audit_path": ""},
                        raising=False)
    from larkhelm.memory_retriever import _audit_decision
    _audit_decision({"ts": "now", "chat_id": "c1", "agent_type": "chat"})
    # Give the background writer a brief moment.
    time.sleep(0.1)


# ── 11. Phase 2 regression suite ───────────────────────────────────────────

def test_english_query_recall_unchanged():
    """Phase 2 must not regress English-only recall on the Phase 1 golden case."""
    slices = [
        _slice_at(title="OOM 双层防护", body="cgroup memory.max=2.8G, V8 voice"),
        _slice_at(title="测试约定", body="集成测试 mock urllib"),
    ]
    req = RetrievalRequest(chat_id="c1", query="OOM voice protection")
    policy = get_policy("dev")
    scored = KeywordRetriever().retrieve(req, policy, slices)
    assert scored[0].slice.title.startswith("OOM")


def test_cjk_bigram_lifts_recall():
    """A slice whose body only contains the CJK bigram still gets recalled."""
    target = _slice_at(layer="project", title="防护说明",
                       body="OOM 防护 是一种重要机制", kind="convention")
    distractor = _slice_at(layer="project", title="other",
                           body="完全无关 的 内容", kind="fact")
    req = RetrievalRequest(chat_id="c1", query="防护")
    policy = get_policy("dev")
    scored = KeywordRetriever().retrieve(req, policy, [target, distractor])
    assert scored[0].slice.id == target.id


def test_stale_decay_orders_correctly():
    """Stale slice with the same BM25 relevance ranks below the fresh sibling."""
    fresh = _slice_at(layer="project", title="A", body="alpha alpha alpha",
                      kind="convention")
    stale = MemorySlice(
        id="stale-x", layer="project", kind="convention",
        title="A-stale", body="alpha alpha alpha",
        importance=0.5, char_len=15, stale=True,
    )
    req = RetrievalRequest(chat_id="c1", query="alpha")
    policy = get_policy("dev")
    scored = KeywordRetriever().retrieve(req, policy, [fresh, stale])
    assert scored[0].slice.id == fresh.id
    stale_item = next(s for s in scored if s.slice.id == stale.id)
    assert "stale" in stale_item.reason


def test_resolve_actual_mode_decision_table(monkeypatch):
    """resolve_actual_mode handles the 5 main branches per design §1.3."""
    from larkhelm.memory_retriever import resolve_actual_mode
    chat_id = "any-chat"
    chat_policy = POLICY_TABLE["chat"]   # declared keyword
    dev_policy = POLICY_TABLE["dev"]    # declared hybrid

    # 1) auto + keyword policy + backend none → keyword
    cfg = {"memory_retriever_mode": "auto", "embedding_backend": "none",
           "embedding_enabled": False, "embedding_traffic": 0.0}
    assert resolve_actual_mode(chat_policy, chat_id, cfg) == "keyword"

    # 2) auto + hybrid policy + backend stub → hybrid
    cfg2 = {"memory_retriever_mode": "auto", "embedding_backend": "stub",
            "embedding_enabled": False, "embedding_traffic": 0.0}
    assert resolve_actual_mode(dev_policy, chat_id, cfg2) == "hybrid"

    # 3) traffic gate ON forces hybrid for chat policy
    cfg3 = {"memory_retriever_mode": "auto", "embedding_backend": "stub",
            "embedding_enabled": True, "embedding_traffic": 1.0}
    assert resolve_actual_mode(chat_policy, chat_id, cfg3) == "hybrid"

    # 4) embedding_backend=none collapses to keyword regardless of policy
    cfg4 = {"memory_retriever_mode": "auto", "embedding_backend": "none",
            "embedding_enabled": True, "embedding_traffic": 1.0}
    assert resolve_actual_mode(dev_policy, chat_id, cfg4) == "keyword"

    # 5) explicit override beats policy default
    cfg5 = {"memory_retriever_mode": "embedding", "embedding_backend": "stub"}
    assert resolve_actual_mode(chat_policy, chat_id, cfg5) == "embedding"


def test_audit_v2_record_fields():
    from larkhelm.memory_retriever import build_audit_record_v2
    req = RetrievalRequest(chat_id="c", query="hello world", agent_type="dev")
    rec = build_audit_record_v2(
        request=req, policy=POLICY_TABLE["dev"], scored=[],
        candidate_count=3, elapsed_ms=10, selected_chars=0,
        fail_open=False, actual_mode="hybrid", declared_mode="hybrid",
    )
    must_have = {"schema_version", "mode", "declared_mode", "hybrid_alpha",
                 "query_token_count", "top_k_returned", "stale_hit_count"}
    assert must_have.issubset(rec.keys())
