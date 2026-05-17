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


def test_layer_session_smart_priority_degradation_drops_history_first(monkeypatch):
    """P1 review noted the priority-degradation branch was untested.

    Under normal inputs the branch is dead code: ``smart_truncate`` honours
    its budget (output ≤ budget + 1), and ``total_budget`` includes a +9
    ellipsis slack, so the ``_current_total > total_budget`` check never
    fires. To exercise the branch we monkey-patch ``smart_truncate`` to
    return strings that exceed their budget — simulating a future helper
    that doesn't clip — and assert sections are dropped in REVERSE priority
    (``history`` first, then ``decisions``, finally ``work_context``)
    until the total fits ``total_budget``.

    This pins both the drop ORDER and the existence of the branch so a
    future ``smart_truncate`` change that violates the budget contract
    can't silently make the safety net disappear.
    """
    import larkhelm.memory_context as mc

    # Each "truncated" section returns 80 chars. With budgets 50/40/30 →
    # ``total_budget`` = sum + 9 = 129. Initial combined = 240 (> 129).
    # Drop ``history`` → 160 (still > 129). Drop ``decisions`` → 80 (≤ 129)
    # → stop. ``work_context`` (highest priority) must survive.
    def _bloated_truncate(text, budget, *, slack_pct=0.15):
        return "X" * 80
    monkeypatch.setattr(mc, "smart_truncate", _bloated_truncate)

    slots = SessionSlots(
        work_context="W" * 100,
        history="H" * 100,
        decisions="D" * 100,
        raw="", parsed=True,
    )
    out = _layer_session_smart(
        slots, query="decision",   # decision-flavoured → include_decisions=True
        budgets={"work_context": 50, "decisions": 40, "history": 30},
    )

    assert "## Work Context" in out, (
        "highest-priority section must survive the degradation cascade"
    )
    assert "## Next Steps" not in out, (
        "history is lowest priority → dropped first"
    )
    assert "## Key Decisions" not in out, (
        "decisions is second-lowest → dropped second"
    )


def test_layer_session_smart_degradation_drops_history_only_when_decisions_fit(
    monkeypatch,
):
    """Drop-order regression: when dropping only ``history`` already brings
    the total below ``total_budget``, ``decisions`` must NOT be dropped."""
    import larkhelm.memory_context as mc

    # Make each "truncated" output exactly 60 chars. With budgets 50/40/30
    # → total_budget=129; combined = 60+60+60 = 180 (> 129). After dropping
    # ``history`` (60), combined = 120 (≤ 129) → stop. decisions stays.
    def _moderate_truncate(text, budget, *, slack_pct=0.15):
        return "Y" * 60
    monkeypatch.setattr(mc, "smart_truncate", _moderate_truncate)

    slots = SessionSlots(
        work_context="W" * 100, history="H" * 100, decisions="D" * 100,
        raw="", parsed=True,
    )
    out = _layer_session_smart(
        slots, query="decision",
        budgets={"work_context": 50, "decisions": 40, "history": 30},
    )
    assert "## Work Context" in out
    assert "## Key Decisions" in out, (
        "decisions section was dropped despite history alone freeing enough budget"
    )
    assert "## Next Steps" not in out, "history is the lowest priority and should drop"


def test_layer_session_smart_empty_sections_yield_raw():
    slots = SessionSlots(
        work_context="", history="", decisions="",
        raw="fallback raw", parsed=True,
    )
    out = _layer_session_smart(slots, query="anything")
    # When all sections empty, return raw body
    assert out == "fallback raw"
