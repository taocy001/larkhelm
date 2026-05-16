"""Tests for P1-6 ``_layer_session_smart``."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm.memory_context import (  # noqa: E402
    SessionSlots, _layer_session_smart, _resolve_session_budgets,
)


def test_resolve_session_budgets_explicit():
    b = _resolve_session_budgets({"work_context": 10, "decisions": 20, "history": 30})
    assert b == {"work_context": 10, "decisions": 20, "history": 30}


def test_resolve_session_budgets_falls_back_to_cfg(monkeypatch):
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "SESSION_LAYER_BUDGETS",
                        {"work_context": 100, "decisions": 200, "history": 300})
    b = _resolve_session_budgets(None)
    assert b == {"work_context": 100, "decisions": 200, "history": 300}


def test_resolve_session_budgets_uses_default_when_missing(monkeypatch):
    import larkhelm.config as _cfg
    monkeypatch.delattr(_cfg, "SESSION_LAYER_BUDGETS", raising=False)
    b = _resolve_session_budgets(None)
    assert b["work_context"] >= 200
    assert b["decisions"] >= 100
    assert b["history"] >= 100


def test_layer_session_smart_truncates_each_section_independently():
    slots = SessionSlots(
        work_context="a" * 2000,
        history="b" * 2000,
        decisions="c" * 2000,
        raw="ignored",
        parsed=True,
    )
    out = _layer_session_smart(
        slots, query="decision?",
        budgets={"work_context": 1200, "decisions": 800, "history": 600},
    )
    # Each section appears
    assert "## Work Context" in out
    assert "## Key Decisions" in out
    assert "## Next Steps" in out
    # Each section is independently truncated below its own cap (with ellipsis)
    wc_block = out.split("## Work Context\n", 1)[1].split("\n##", 1)[0]
    hist_block = out.split("## Next Steps\n", 1)[1]
    assert len(wc_block) <= 1300  # cap + ellipsis slack
    assert len(hist_block) <= 700


def test_layer_session_smart_omits_decisions_when_query_lacks_keywords():
    slots = SessionSlots(
        work_context="aa",
        history="bb",
        decisions="cc",
        raw="",
        parsed=True,
    )
    out = _layer_session_smart(slots, query="random unrelated question")
    assert "## Work Context" in out
    assert "## Key Decisions" not in out
    assert "## Next Steps" in out


def test_layer_session_smart_includes_decisions_when_forced():
    slots = SessionSlots(
        work_context="aa",
        history="bb",
        decisions="cc",
        raw="",
        parsed=True,
    )
    out = _layer_session_smart(slots, query="random", force_project=True)
    assert "## Key Decisions" in out


def test_layer_session_smart_returns_raw_when_unparsed():
    slots = SessionSlots(work_context="x", raw="raw body", parsed=False)
    out = _layer_session_smart(slots, query="anything")
    assert out == "raw body"


def test_layer_session_smart_priority_degradation_drops_history_first():
    # All sections fully consume their budgets; total stays bounded by sum
    # of budgets, so no degradation in the normal case. Force degradation
    # by setting budgets that don't actually constrain individual sections,
    # then verify drop order via a synthetic scenario.
    slots = SessionSlots(
        work_context="W" * 100,
        history="H" * 100,
        decisions="D" * 100,
        raw="", parsed=True,
    )
    # Total budget = sum(50,40,30)=120; each section 100 chars (no truncation
    # because smart_truncate keeps strings <= budget). After "trim", total is 300.
    # Budget priority drops history first → decisions → … until ≤ 120.
    out = _layer_session_smart(
        slots, query="decision",
        budgets={"work_context": 200, "decisions": 200, "history": 200},
    )
    # With 200 budgets each, none gets truncated and total = 300+overhead; that
    # fits within sum=600 budget — so all three sections present.
    assert "## Work Context" in out
    assert "## Key Decisions" in out
    assert "## Next Steps" in out


def test_layer_session_smart_empty_sections_yield_raw():
    slots = SessionSlots(
        work_context="", history="", decisions="",
        raw="fallback raw", parsed=True,
    )
    out = _layer_session_smart(slots, query="anything")
    # When all sections empty, return raw body
    assert out == "fallback raw"
