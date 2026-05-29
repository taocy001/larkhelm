"""P2 REQ-07: tests for ``larkhelm.memory_session_compress``."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import memory_session_compress as _msc  # noqa: E402


# ── _score_sentence determinism ─────────────────────────────────────────


def test_score_deterministic_for_same_input():
    s = "user wants extract buffer"
    s1 = _msc._score_sentence(s, "user", 1700000000.0, {"buffer"}, 1700000010.0)
    s2 = _msc._score_sentence(s, "user", 1700000000.0, {"buffer"}, 1700000010.0)
    assert s1 == s2


def test_role_weight_affects_score():
    """milestone > user > assistant for the same sentence."""
    query_tokens: set[str] = set()
    sent = "a sentence"
    now = 1.0
    user_score = _msc._score_sentence(sent, "user", 0.5, query_tokens, now)
    assistant_score = _msc._score_sentence(sent, "assistant", 0.5, query_tokens, now)
    milestone_score = _msc._score_sentence(sent, "milestone", 0.5, query_tokens, now)
    assert milestone_score > user_score > assistant_score


def test_keyword_overlap_boosts_score():
    """A sentence matching all query tokens scores higher than one with none."""
    qtoks = {"buffer", "extract"}
    base = _msc._score_sentence("totally unrelated", "user", 0.0, qtoks, 1.0)
    boosted = _msc._score_sentence(
        "the extract buffer", "user", 0.0, qtoks, 1.0,
    )
    assert boosted > base


# ── _select_top_k ───────────────────────────────────────────────────────


def test_select_top_k_respects_budget():
    # Five sentences of 10 chars each + newlines; budget=25 should hold ≈2.
    scored = [
        _msc.CompressScore(sentence=f"sent {i}    ", role="user",
                           timestamp=0.0, score=float(10 - i), keep=False)
        for i in range(5)
    ]
    kept = _msc._select_top_k(scored, budget=25)
    # Top scorers picked; total length stays ≤ 25.
    assert sum(len(cs.sentence) + 1 for cs in kept) <= 25
    # Highest score selected.
    assert any(cs.sentence == "sent 0    " for cs in kept)


def test_select_top_k_reorders_to_original_doc_order():
    """After top-K, kept sentences appear in original document order."""
    scored = [
        _msc.CompressScore(sentence="first", role="user", timestamp=0.0,
                           score=1.0, keep=False),
        _msc.CompressScore(sentence="second", role="user", timestamp=0.0,
                           score=5.0, keep=False),
        _msc.CompressScore(sentence="third", role="user", timestamp=0.0,
                           score=3.0, keep=False),
    ]
    kept = _msc._select_top_k(scored, budget=20)
    # Even though "second" has the highest score, the output order should
    # match the input order.
    sentences = [cs.sentence for cs in kept]
    indices = [["first", "second", "third"].index(s) for s in sentences]
    assert indices == sorted(indices)


# ── smart_compress end-to-end ───────────────────────────────────────────


def test_smart_compress_short_input_returns_unchanged():
    text = "tiny text"
    assert _msc.smart_compress(text, budget=100) == text


def test_smart_compress_returns_under_budget():
    text = "句子一。句子二。句子三。句子四。句子五。句子六。" * 10
    out = _msc.smart_compress(text, budget=80, query="句子一")
    assert len(out) <= 80


def test_smart_compress_empty_query_still_works():
    """Query empty → score falls back to role + decay; output non-empty."""
    text = "[10:00:00] user: 第一句。\n[10:01:00] assistant: 第二句。\n[10:02:00] milestone: 第三句。"
    out = _msc.smart_compress(text, budget=100, query="")
    # Some sentences kept.
    assert out
    assert any(s in out for s in ("第一句", "第二句", "第三句"))


def test_smart_compress_zero_budget_returns_empty():
    assert _msc.smart_compress("anything", budget=0) == ""


# ── is_enabled() ────────────────────────────────────────────────────────


def test_is_enabled_default_true():
    assert _msc.is_enabled() is True


def test_is_enabled_honours_flag(monkeypatch):
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "MEMORY_SESSION_SMART_COMPRESS", True, raising=False)
    assert _msc.is_enabled() is True
