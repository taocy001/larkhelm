"""Unit tests for LLMRouterRetriever (Phase D · Phase 3).

The router decorates an underlying KeywordRetriever / HybridRetriever and
re-orders its output based on a cheap-LLM verdict. The full pipeline is
covered here without ever calling a real backend — ``cheap_caller`` is
always injected as a deterministic stub.

Test surface (organised top-down):

  1. Cache key + Cache TTL + LRU eviction (3 cases)
  2. Rate limiter window + 0-disables (3)
  3. Prompt construction + slice meta serialisation (2)
  4. LLM response parser (4 — JSON-fenced, bare, malformed, empty)
  5. Reorder helper preserves leftovers (1)
  6. End-to-end: happy path / cache hit / rate limit / no caller / LLM
     returns subset / LLM returns malformed JSON / LLM raises (7)
  7. _should_wrap_with_llm_router gating decision table (5)
  8. Audit record carries llm_router_* fields when diag provided (1)
"""
from __future__ import annotations

import time
import unittest
from typing import Any
from unittest.mock import patch

from larkhelm import memory_llm_router as r
from larkhelm.memory_llm_router import (
    LLM_ROUTER_MAX_POOL,
    LLMRouterRetriever,
    RouterDiagnostics,
    _build_router_prompt,
    _cache_clear_for_tests,
    _cache_get,
    _cache_key,
    _cache_put,
    _parse_llm_response,
    _rate_check_and_record,
    _rate_remaining,
    _reorder_scored_by_ids,
    _serialise_slice_meta,
)
from larkhelm.memory_retriever import (
    KeywordRetriever,
    _should_wrap_with_llm_router,
    build_audit_record_v2,
)
from larkhelm.memory_slice import (
    InjectionPolicy,
    MemorySlice,
    RetrievalRequest,
    ScoredSlice,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _slice(sid: str, *, body: str = "", title: str = "", layer: str = "project",
           kind: str = "convention", importance: float = 0.5,
           stale: bool = False) -> MemorySlice:
    return MemorySlice(
        id=sid, layer=layer, kind=kind, title=title or sid,
        body=body or f"body of {sid}",
        importance=importance, char_len=len(body or f"body of {sid}"),
        stale=stale,
    )


def _scored(slice_obj: MemorySlice, score: float = 0.5,
            reason: str = "") -> ScoredSlice:
    return ScoredSlice(slice=slice_obj, score=score, relevance_score=score,
                       reason=reason)


def _policy(*, top_k: int = 3, agent_type: str = "dev") -> InjectionPolicy:
    return InjectionPolicy(
        agent_type=agent_type,
        token_budget=2000,
        layer_weights={"project": 0.7, "session": 0.3},
        kind_priority=("convention", "fact"),
        retrieval_mode="hybrid",
        top_k=top_k,
        alpha_recency=0.3, alpha_importance=0.3, alpha_relevance=0.4,
    )


def _request(query: str = "alpha", *, chat_id: str = "c-test",
             complexity: str = "complex", agent_type: str = "dev") -> RetrievalRequest:
    # ``complexity="complex"`` matches the production Complexity literal
    # (agent_hub/intent_types.py:15). Initial v1 used "high" which is not
    # in the Complexity union — review MF-01 fixed.
    return RetrievalRequest(
        chat_id=chat_id, query=query, agent_type=agent_type,
        complexity=complexity,
    )


class _StubRetriever:
    """Underlying retriever stub returning a fixed ScoredSlice list."""
    def __init__(self, scored: list[ScoredSlice]) -> None:
        self._scored = list(scored)
        self.call_count = 0
    def retrieve(self, request: RetrievalRequest, policy: InjectionPolicy,
                 candidate_slices: list[MemorySlice]) -> list[ScoredSlice]:
        self.call_count += 1
        return list(self._scored)


# ─────────────────────────────────────────────────────────────────────────
#  1. Cache + LRU
# ─────────────────────────────────────────────────────────────────────────


class CacheTests(unittest.TestCase):
    def setUp(self):
        _cache_clear_for_tests()

    def test_cache_key_independent_of_order(self):
        a = _cache_key("hello", ["sid-1", "sid-2", "sid-3"])
        b = _cache_key("hello", ["sid-3", "sid-1", "sid-2"])
        self.assertEqual(a, b, "candidate-id order should not affect key")

    def test_cache_key_dedupes_duplicate_ids(self):
        """Regression SF-02: passing the same candidate id repeatedly
        must collapse to a single bucket. Without ``set(...)`` the
        key changed every time multiplicity drifted (e.g. caller
        accidentally passed a list instead of a set)."""
        a = _cache_key("q", ["sid-1", "sid-2"])
        b = _cache_key("q", ["sid-1", "sid-1", "sid-2", "sid-2"])
        c = _cache_key("q", ["sid-2", "sid-1", "sid-1", "sid-2"])
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_cache_hit_within_ttl(self):
        key = _cache_key("q", ["a", "b"])
        _cache_put(key, ("a",))
        # 60s window, hit immediately.
        self.assertEqual(_cache_get(key, 60.0), ("a",))

    def test_cache_expires_past_ttl(self):
        key = _cache_key("q", ["a", "b"])
        _cache_put(key, ("a",))
        # Forcibly back-date by patching the stored timestamp.
        with r._cache_lock:
            ts, ids = r._cache[key]
            r._cache[key] = (ts - 1000.0, ids)
        self.assertIsNone(_cache_get(key, 60.0),
                          "1000s-old entry should be evicted under 60s TTL")
        # Side-effect: expired entry is proactively removed.
        with r._cache_lock:
            self.assertNotIn(key, r._cache)


# ─────────────────────────────────────────────────────────────────────────
#  2. Rate limiter
# ─────────────────────────────────────────────────────────────────────────


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        _cache_clear_for_tests()

    def test_rate_limit_blocks_after_threshold(self):
        for _ in range(3):
            self.assertTrue(_rate_check_and_record("chat-rate", 3))
        # 4th call within the same window is rejected.
        self.assertFalse(_rate_check_and_record("chat-rate", 3))
        # And remaining is 0.
        self.assertEqual(_rate_remaining("chat-rate", 3), 0)

    def test_rate_limit_per_chat_isolation(self):
        for _ in range(3):
            _rate_check_and_record("chat-A", 3)
        # chat-B has its own bucket — unaffected.
        self.assertTrue(_rate_check_and_record("chat-B", 3))

    def test_rate_limit_zero_disables(self):
        # Special-case: 0 = unbounded (used by tests / debug).
        for _ in range(1000):
            self.assertTrue(_rate_check_and_record("c-unlim", 0))
        self.assertEqual(_rate_remaining("c-unlim", 0), -1)

    def test_rate_limit_concurrent_no_overshoot(self):
        """NH-03 regression: under 100-thread concurrent access against
        a 5/min limit, at most 5 calls succeed. Without the lock guard
        ``len(window) < limit`` would race against ``window.append``
        and let 6+ slots through."""
        import threading
        results: list[bool] = []
        results_lock = threading.Lock()

        def attempt():
            allowed = _rate_check_and_record("concurrent-chat", 5)
            with results_lock:
                results.append(allowed)

        threads = [threading.Thread(target=attempt) for _ in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()

        ok = sum(1 for r in results if r)
        self.assertEqual(ok, 5, f"expected exactly 5 successes, got {ok}")


# ─────────────────────────────────────────────────────────────────────────
#  3. Prompt + slice-meta serialisation
# ─────────────────────────────────────────────────────────────────────────


class PromptTests(unittest.TestCase):
    def test_slice_meta_truncates_long_fields(self):
        long_title = "T" * 500
        long_body = "B" * 2000
        s = _slice("sid", title=long_title, body=long_body, stale=True)
        meta = _serialise_slice_meta(s)
        # Titles capped at 120 chars; body_head at 120.
        self.assertLessEqual(len(meta["title"]), 120)
        self.assertLessEqual(len(meta["body_head"]), 120)
        self.assertTrue(meta["stale"])
        self.assertEqual(meta["layer"], "project")

    def test_prompt_contains_query_and_candidates(self):
        slices = [_slice(f"sid-{i}") for i in range(3)]
        prompt = _build_router_prompt(
            _request("how do I fix OOM"), _policy(top_k=2),
            slices, top_k=2,
        )
        self.assertIn("OOM", prompt)
        self.assertIn("dev", prompt)
        self.assertIn("sid-0", prompt)
        # Tells the LLM to return at most top_k.
        self.assertIn("<= 2", prompt)


# ─────────────────────────────────────────────────────────────────────────
#  4. Response parser
# ─────────────────────────────────────────────────────────────────────────


class ResponseParserTests(unittest.TestCase):
    def setUp(self):
        self.cand = {"sid-1", "sid-2", "sid-3"}

    def test_parses_bare_json(self):
        out = _parse_llm_response(
            '{"selected_ids": ["sid-2", "sid-1"], "reasoning": "x"}',
            self.cand,
        )
        self.assertEqual(out, ("sid-2", "sid-1"))

    def test_parses_code_fenced_json(self):
        out = _parse_llm_response(
            '```json\n{"selected_ids": ["sid-1"], "reasoning": "..."}\n```',
            self.cand,
        )
        self.assertEqual(out, ("sid-1",))

    def test_drops_unknown_ids(self):
        out = _parse_llm_response(
            '{"selected_ids": ["sid-1", "imposter-99", "sid-3"]}',
            self.cand,
        )
        self.assertEqual(out, ("sid-1", "sid-3"))

    def test_empty_or_malformed_returns_none(self):
        self.assertIsNone(_parse_llm_response("", self.cand))
        self.assertIsNone(_parse_llm_response("nothing useful here", self.cand))
        # Empty selection counts as parse failure (likely misunderstood prompt).
        self.assertIsNone(_parse_llm_response(
            '{"selected_ids": []}', self.cand,
        ))


# ─────────────────────────────────────────────────────────────────────────
#  5. Reorder helper
# ─────────────────────────────────────────────────────────────────────────


class ReorderTests(unittest.TestCase):
    def test_reorder_puts_selected_first_then_backfills(self):
        scored = [_scored(_slice(f"sid-{i}"), score=1.0 - i * 0.1)
                  for i in range(5)]
        # LLM picks 4 first, then 2.
        out = _reorder_scored_by_ids(scored, ("sid-4", "sid-2"), top_k=4)
        ids = [s.slice.id for s in out]
        self.assertEqual(ids[:2], ["sid-4", "sid-2"], "selected first")
        # Backfill keeps underlying order (excluding already-selected).
        self.assertEqual(ids[2:], ["sid-0", "sid-1"])

    def test_reorder_dedupes_repeated_selections(self):
        scored = [_scored(_slice(f"sid-{i}")) for i in range(3)]
        out = _reorder_scored_by_ids(scored, ("sid-1", "sid-1", "sid-2"), top_k=3)
        ids = [s.slice.id for s in out]
        self.assertEqual(ids, ["sid-1", "sid-2", "sid-0"])

    def test_reorder_tags_reason_with_llm_router(self):
        scored = [_scored(_slice("sid-1"), reason="bm25=0.5")]
        out = _reorder_scored_by_ids(scored, ("sid-1",), top_k=1)
        self.assertIn("llm_router", out[0].reason)
        self.assertIn("bm25", out[0].reason, "must preserve prior reason chain")


# ─────────────────────────────────────────────────────────────────────────
#  6. End-to-end retrieve()
# ─────────────────────────────────────────────────────────────────────────


class RetrieveTests(unittest.TestCase):
    def setUp(self):
        _cache_clear_for_tests()
        self.slices = [_slice(f"sid-{i}") for i in range(5)]
        self.scored = [_scored(self.slices[i], score=1.0 - i * 0.1)
                       for i in range(5)]
        self.policy = _policy(top_k=3)
        self.request = _request()

    def test_happy_path_llm_reorders(self):
        underlying = _StubRetriever(self.scored)
        def caller(prompt: str) -> str:
            return '{"selected_ids": ["sid-3", "sid-1"], "reasoning": "x"}'
        router = LLMRouterRetriever(underlying, cheap_caller=caller)
        out = router.retrieve(self.request, self.policy, self.slices)
        ids = [s.slice.id for s in out]
        # LLM-selected come first; backfill from underlying order.
        self.assertEqual(ids[0], "sid-3")
        self.assertEqual(ids[1], "sid-1")
        self.assertEqual(len(out), 3)
        # Diagnostics populated.
        self.assertTrue(router.diagnostics.invoked)
        self.assertFalse(router.diagnostics.cache_hit)
        self.assertEqual(router.diagnostics.selected_by_llm, 2)
        self.assertEqual(router.diagnostics.skipped_reason, "")

    def test_cache_hit_skips_llm(self):
        underlying = _StubRetriever(self.scored)
        calls = []
        def caller(prompt: str) -> str:
            calls.append(1)
            return '{"selected_ids": ["sid-2"], "reasoning": "y"}'
        router = LLMRouterRetriever(underlying, cheap_caller=caller)
        # First call populates the cache.
        router.retrieve(self.request, self.policy, self.slices)
        self.assertEqual(len(calls), 1)
        # Second call (same query + same candidate set) hits cache.
        out = router.retrieve(self.request, self.policy, self.slices)
        self.assertEqual(len(calls), 1, "second call must hit cache, not LLM")
        self.assertTrue(router.diagnostics.cache_hit)
        self.assertFalse(router.diagnostics.invoked)
        # And the reordering is still applied.
        self.assertEqual(out[0].slice.id, "sid-2")

    def test_rate_limit_skips_llm(self):
        with patch("larkhelm.memory_llm_router._config_int") as mock_int:
            # Rate limit = 1 per minute; cache TTL = 300s default.
            mock_int.side_effect = lambda k, d: 1 if "rate" in k or "max" in k else 300
            underlying = _StubRetriever(self.scored)
            calls = []
            def caller(prompt: str) -> str:
                calls.append(1)
                return '{"selected_ids": ["sid-2"]}'
            # Use different queries to avoid cache hit.
            req1 = _request("alpha unique")
            req2 = _request("beta unique")
            router = LLMRouterRetriever(underlying, cheap_caller=caller)
            router.retrieve(req1, self.policy, self.slices)
            router.retrieve(req2, self.policy, self.slices)
            # First fired the LLM, second hit the rate cap.
            self.assertEqual(len(calls), 1)
            self.assertEqual(router.diagnostics.skipped_reason, "rate_limit")

    def test_no_cheap_caller_falls_back(self):
        underlying = _StubRetriever(self.scored)
        # Cheap caller resolution returns None — feature unavailable.
        with patch("larkhelm.memory_llm_router._resolve_cheap_caller", return_value=None):
            router = LLMRouterRetriever(underlying)  # no inject
            out = router.retrieve(self.request, self.policy, self.slices)
        self.assertEqual(router.diagnostics.skipped_reason, "no_cheap_caller")
        # Underlying output preserved (first top_k).
        self.assertEqual([s.slice.id for s in out],
                         ["sid-0", "sid-1", "sid-2"])

    def test_llm_raises_falls_back_no_crash(self):
        underlying = _StubRetriever(self.scored)
        def caller(prompt: str) -> str:
            raise RuntimeError("simulated backend outage")
        router = LLMRouterRetriever(underlying, cheap_caller=caller)
        out = router.retrieve(self.request, self.policy, self.slices)
        self.assertEqual(router.diagnostics.skipped_reason, "caller_exception")
        # Underlying output preserved.
        self.assertEqual(len(out), 3)

    def test_llm_returns_garbage_falls_back(self):
        underlying = _StubRetriever(self.scored)
        def caller(prompt: str) -> str:
            return "totally not json"
        router = LLMRouterRetriever(underlying, cheap_caller=caller)
        out = router.retrieve(self.request, self.policy, self.slices)
        self.assertEqual(router.diagnostics.skipped_reason, "parse_failed")
        self.assertEqual(out, self.scored[:3])

    def test_pool_capped_at_max(self):
        # Build 50 candidates; router must only send LLM_ROUTER_MAX_POOL.
        many = [_slice(f"big-{i}") for i in range(50)]
        many_scored = [_scored(many[i], score=1.0 - i * 0.01) for i in range(50)]
        underlying = _StubRetriever(many_scored)
        captured_prompts: list[str] = []
        def caller(prompt: str) -> str:
            captured_prompts.append(prompt)
            return '{"selected_ids": ["big-0"]}'
        router = LLMRouterRetriever(underlying, cheap_caller=caller)
        router.retrieve(self.request, self.policy, many)
        # The LLM prompt must include ``Candidates (N total)`` with
        # N <= LLM_ROUTER_MAX_POOL.
        prompt = captured_prompts[0]
        self.assertIn(f"Candidates ({LLM_ROUTER_MAX_POOL} total", prompt)
        # And one of the cut-off ids should NOT appear.
        self.assertNotIn("big-49", prompt)

    def test_underlying_returns_empty_no_llm_call(self):
        underlying = _StubRetriever([])
        calls = []
        def caller(prompt: str) -> str:
            calls.append(1); return '{"selected_ids": []}'
        router = LLMRouterRetriever(underlying, cheap_caller=caller)
        out = router.retrieve(self.request, self.policy, self.slices)
        self.assertEqual(out, [])
        self.assertEqual(len(calls), 0, "no candidates → no LLM call")

    def test_underlying_raises_bubbles_up(self):
        class _Boom:
            def retrieve(self, *a, **k):
                raise RuntimeError("kw retriever broken")
        router = LLMRouterRetriever(_Boom(), cheap_caller=lambda p: "{}")
        with self.assertRaises(RuntimeError):
            router.retrieve(self.request, self.policy, self.slices)


# ─────────────────────────────────────────────────────────────────────────
#  7. _should_wrap_with_llm_router gate
# ─────────────────────────────────────────────────────────────────────────


class GateTests(unittest.TestCase):
    def setUp(self):
        self.policy = _policy()

    def test_off_by_default(self):
        self.assertFalse(_should_wrap_with_llm_router(
            _request(), self.policy, "chat-1", {"memory_llm_router_enabled": False},
        ))

    def test_enabled_but_traffic_zero(self):
        cfg = {"memory_llm_router_enabled": True, "memory_llm_router_traffic": 0.0}
        self.assertFalse(_should_wrap_with_llm_router(
            _request(), self.policy, "chat-1", cfg,
        ))

    def test_wrong_agent_type_filtered_out(self):
        # /chat at 100% traffic still must be filtered (only crew/dev allowed).
        cfg = {"memory_llm_router_enabled": True, "memory_llm_router_traffic": 1.0}
        self.assertFalse(_should_wrap_with_llm_router(
            _request(agent_type="chat"), self.policy, "chat-1", cfg,
        ))

    def test_complexity_not_complex_filtered_out(self):
        # The production Complexity literal is "simple"|"medium"|"complex"
        # (agent_hub/intent_types.py:15). The gate must reject every
        # value except "complex" (with "high" still tolerated as alias).
        cfg = {"memory_llm_router_enabled": True, "memory_llm_router_traffic": 1.0}
        for c in ("simple", "medium", "low", ""):
            self.assertFalse(_should_wrap_with_llm_router(
                _request(complexity=c), self.policy, "chat-1", cfg,
            ), f"complexity={c!r} must not gate in")

    def test_full_gate_open_complex(self):
        """Regression MF-01: 'complex' is the production literal —
        v1 of this gate checked 'high' which made the entire feature
        unreachable in production."""
        cfg = {"memory_llm_router_enabled": True, "memory_llm_router_traffic": 1.0}
        self.assertTrue(_should_wrap_with_llm_router(
            _request(complexity="complex", agent_type="crew"), self.policy, "chat-x", cfg,
        ), "production 'complex' must be accepted")

    def test_high_alias_still_accepted(self):
        """Keep the 'high' alias working so any third-party plugin
        that pre-dates Phase 3 doesn't silently lose routing."""
        cfg = {"memory_llm_router_enabled": True, "memory_llm_router_traffic": 1.0}
        self.assertTrue(_should_wrap_with_llm_router(
            _request(complexity="high", agent_type="dev"), self.policy, "chat-1", cfg,
        ))


# ─────────────────────────────────────────────────────────────────────────
#  8. Audit record additions
# ─────────────────────────────────────────────────────────────────────────


class AuditRecordTests(unittest.TestCase):
    def test_audit_includes_llm_router_fields_when_diag_given(self):
        diag = RouterDiagnostics(
            invoked=True, cache_hit=False, skipped_reason="",
            elapsed_ms=42, selected_by_llm=2,
        )
        rec = build_audit_record_v2(
            request=_request(),
            policy=_policy(),
            scored=[_scored(_slice("a"))],
            candidate_count=5,
            elapsed_ms=42,
            selected_chars=200,
            fail_open=False,
            actual_mode="hybrid",
            llm_router_diag=diag,
        )
        self.assertTrue(rec["llm_router_invoked"])
        self.assertFalse(rec["llm_router_cache_hit"])
        self.assertEqual(rec["llm_router_skipped"], "")
        self.assertEqual(rec["llm_router_selected_n"], 2)

    def test_audit_omits_llm_router_fields_when_diag_none(self):
        rec = build_audit_record_v2(
            request=_request(),
            policy=_policy(),
            scored=[_scored(_slice("a"))],
            candidate_count=5,
            elapsed_ms=42,
            selected_chars=200,
            fail_open=False,
            actual_mode="hybrid",
            llm_router_diag=None,
        )
        # Byte-compatibility with Phase 2: missing diag → no llm_router_* keys.
        self.assertNotIn("llm_router_invoked", rec)
        self.assertNotIn("llm_router_skipped", rec)


if __name__ == "__main__":
    unittest.main()
