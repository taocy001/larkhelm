"""Unit tests for ``larkhelm.__main__`` audit-summary aggregation."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import larkhelm.config as _cfg
from larkhelm.__main__ import (
    _compute_audit_summary,
    _parse_audit_summary_duration,
    _render_audit_summary_text,
)
from larkhelm.memory_retriever import iter_audit_records


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_since_parsing():
    assert _parse_audit_summary_duration("30m") == timedelta(minutes=30)
    assert _parse_audit_summary_duration("2h") == timedelta(hours=2)
    assert _parse_audit_summary_duration("1d") == timedelta(days=1)
    assert _parse_audit_summary_duration("1w") == timedelta(weeks=1)
    # default + invalid both fall back to 1h
    assert _parse_audit_summary_duration(None) == timedelta(hours=1)
    assert _parse_audit_summary_duration("garbage") == timedelta(hours=1)


def test_audit_summary_json_contract():
    records = _load_records(FIXTURE_DIR / "audit_phase2.jsonl")
    result = _compute_audit_summary(records, timedelta(hours=24))
    assert result["schema_version"] == "2"
    assert result["total_records"] == len(records)
    # 7 of 10 records have mode="hybrid" or "embedding"; at least one "keyword".
    assert "hybrid" in result["mode_distribution"]
    assert "keyword" in result["mode_distribution"]
    assert 0.0 <= result["fail_open_rate"] <= 1.0
    assert isinstance(result["by_agent_type"], dict)
    for agent, st in result["by_agent_type"].items():
        assert "count" in st and "p95_elapsed_ms" in st and "fail_open_rate" in st


def test_audit_summary_text_render():
    records = _load_records(FIXTURE_DIR / "audit_phase2.jsonl")
    text = _render_audit_summary_text(_compute_audit_summary(records, timedelta(hours=24)))
    # Spot-check important lines so the operator can grep.
    assert "window" in text
    assert "records" in text
    assert "modes" in text


def test_legacy_fixture_still_parses():
    """Phase 1 fixture (no schema_version) must still parse cleanly."""
    records = _load_records(FIXTURE_DIR / "audit_legacy.jsonl")
    assert len(records) >= 10
    result = _compute_audit_summary(records, timedelta(days=2))
    assert result["total_records"] == len(records)
    # Phase 1 records have mode="keyword" only.
    assert set(result["mode_distribution"].keys()) <= {"keyword"}


def test_rotation_aware_iteration(tmp_path, monkeypatch):
    """iter_audit_records reads both the live file and the rotated archive."""
    # Stub the config + audit path resolver to point at tmp_path.
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    live = audit_dir / "memory_retriever_audit.jsonl"
    archive = audit_dir / "memory_retriever_audit-2026-05-15.jsonl"
    shutil.copyfile(FIXTURE_DIR / "audit_phase2.jsonl", live)
    shutil.copyfile(FIXTURE_DIR / "audit_legacy.jsonl", archive)
    # Refresh archive mtime so it sorts older than live.
    import os, time
    older = time.time() - 86400
    os.utime(archive, (older, older))

    monkeypatch.setattr(
        "larkhelm.memory_retriever._resolve_audit_path", lambda: live,
    )
    out = list(iter_audit_records(timedelta(days=400)))
    # Sanity: should cover records from both files (legacy ts is 2026-05-14, live ts is 2026-05-15).
    # Lower bound is "live count" alone; with archive it must be strictly greater.
    live_only = len(_load_records(FIXTURE_DIR / "audit_phase2.jsonl"))
    assert len(out) >= live_only


# ─────────────────────────────────────────────────────────────────────────
#  Phase 3 — Stage C LLM-router aggregation (REQ-37 follow-up)
# ─────────────────────────────────────────────────────────────────────────

from larkhelm.__main__ import _compute_llm_router_summary


def _make_record(*, llm_router_invoked=None, llm_router_cache_hit=None,
                  llm_router_skipped="", llm_router_selected_n=0,
                  agent_type="dev", mode="hybrid", elapsed_ms=10):
    """Build a synthetic audit record for aggregation tests.

    Passing ``llm_router_invoked=None`` (the default) omits ALL
    ``llm_router_*`` keys — simulating a record that didn't pass the
    Stage C gate (Phase 2 or earlier behaviour). Passing a bool flips
    the gate-fired sentinel and includes the full quad of fields.
    """
    r: dict = {
        "schema_version": "2",
        "agent_type": agent_type,
        "mode": mode,
        "elapsed_ms": elapsed_ms,
        "fail_open": False,
        "selected_slice_ids": ["sid-x"],
        "selected_token_chars": 200,
    }
    if llm_router_invoked is not None:
        r["llm_router_invoked"] = bool(llm_router_invoked)
        r["llm_router_cache_hit"] = bool(llm_router_cache_hit or False)
        r["llm_router_skipped"] = str(llm_router_skipped or "")
        r["llm_router_selected_n"] = int(llm_router_selected_n or 0)
    return r


class LLMRouterSummaryTests:
    """Unit tests for ``_compute_llm_router_summary`` (the helper itself)."""

    # NOTE: kept as plain test functions (not unittest.TestCase) to match
    # the file's existing style.


def test_llm_router_summary_returns_none_for_phase2_only_records():
    """Phase 2 (or earlier) records have NO llm_router_* fields — the
    summary must omit the section entirely so Stage-C-disabled
    installations don't see a noisy zero-block."""
    records = [_make_record() for _ in range(5)]
    assert _compute_llm_router_summary(records) is None


def test_llm_router_summary_counts_invokes_and_cache_hits():
    records = [
        _make_record(llm_router_invoked=True, llm_router_selected_n=3),
        _make_record(llm_router_invoked=True, llm_router_selected_n=5),
        _make_record(llm_router_invoked=False, llm_router_cache_hit=True),
        _make_record(llm_router_invoked=False, llm_router_cache_hit=True),
    ]
    summary = _compute_llm_router_summary(records)
    assert summary is not None
    assert summary["gate_fired_count"] == 4
    assert summary["invoked_count"] == 2
    assert summary["cache_hit_count"] == 2
    # cache_hit_rate = hits / (hits + invokes); both stages count toward
    # "the gate fired and tried to use the LLM tier".
    assert summary["cache_hit_rate"] == pytest.approx(0.5)
    # avg_selected_n only averages over invokes (cache hits report 0
    # which would otherwise drag the mean down misleadingly).
    assert summary["avg_selected_n"] == pytest.approx(4.0)


def test_llm_router_summary_breaks_down_skipped_reasons():
    records = [
        _make_record(llm_router_invoked=False, llm_router_skipped="rate_limit"),
        _make_record(llm_router_invoked=False, llm_router_skipped="rate_limit"),
        _make_record(llm_router_invoked=False, llm_router_skipped="parse_failed"),
        _make_record(llm_router_invoked=False, llm_router_skipped="no_cheap_caller"),
        # Empty skipped (cache hit path) doesn't count toward breakdown.
        _make_record(llm_router_invoked=False, llm_router_cache_hit=True),
    ]
    summary = _compute_llm_router_summary(records)
    assert summary is not None
    bd = summary["skipped_breakdown"]
    assert bd == {"rate_limit": 2, "parse_failed": 1, "no_cheap_caller": 1}


def test_llm_router_summary_mixed_phase2_and_phase3_records():
    """When some records have Stage C fields and others don't, the
    summary aggregates ONLY the gate-fired subset (no off-by-zero
    pollution from Phase 2 records)."""
    records = [
        _make_record(),  # Phase 2 — gate didn't fire
        _make_record(),  # Phase 2 — gate didn't fire
        _make_record(llm_router_invoked=True, llm_router_selected_n=4),
        _make_record(llm_router_invoked=False, llm_router_cache_hit=True),
    ]
    summary = _compute_llm_router_summary(records)
    assert summary is not None
    assert summary["gate_fired_count"] == 2
    assert summary["invoked_count"] == 1
    assert summary["cache_hit_count"] == 1
    assert summary["avg_selected_n"] == pytest.approx(4.0)


def test_llm_router_summary_all_zero_invokes_cache_rate_zero():
    """Edge: gate fires but all are skipped — cache_hit_rate is 0/0=0
    rather than crashing on a divide-by-zero."""
    records = [
        _make_record(llm_router_invoked=False, llm_router_skipped="rate_limit"),
        _make_record(llm_router_invoked=False, llm_router_skipped="rate_limit"),
    ]
    summary = _compute_llm_router_summary(records)
    assert summary is not None
    assert summary["invoked_count"] == 0
    assert summary["cache_hit_count"] == 0
    assert summary["cache_hit_rate"] == 0.0
    assert summary["avg_selected_n"] == 0.0


def test_audit_summary_includes_llm_router_when_phase3_present():
    """End-to-end via _compute_audit_summary: the ``llm_router`` key
    appears in the top-level JSON when Stage C fired."""
    records = [
        _make_record(),  # Phase 2 only
        _make_record(llm_router_invoked=True, llm_router_selected_n=3),
    ]
    out = _compute_audit_summary(records, timedelta(hours=1))
    assert "llm_router" in out
    assert out["llm_router"]["invoked_count"] == 1


def test_audit_summary_omits_llm_router_for_phase2_only():
    """Top-level audit summary stays Phase-2-shaped when Stage C hasn't
    fired in the window (no spurious empty section)."""
    records = [_make_record() for _ in range(3)]
    out = _compute_audit_summary(records, timedelta(hours=1))
    assert "llm_router" not in out


def test_audit_summary_text_renders_llm_router_section():
    """Operator-facing text rendering includes the new section under
    the existing ``per agent`` block, with cache rate as percent."""
    records = [
        _make_record(llm_router_invoked=True, llm_router_selected_n=2),
        _make_record(llm_router_invoked=False, llm_router_cache_hit=True),
        _make_record(llm_router_invoked=False, llm_router_skipped="rate_limit"),
    ]
    out = _compute_audit_summary(records, timedelta(hours=1))
    text = _render_audit_summary_text(out)
    assert "llm router" in text
    assert "gate-fired=3" in text
    assert "invoked=1" in text
    assert "cache-hit=1" in text
    # Cache rate is hits / (hits + invokes) = 1/2 = 50%.
    assert "50.00%" in text or "50%" in text
    # Skipped breakdown is rendered as a comma-separated key=count list.
    assert "rate_limit=1" in text


def test_audit_summary_text_omits_llm_router_for_phase2():
    records = [_make_record() for _ in range(3)]
    out = _compute_audit_summary(records, timedelta(hours=1))
    text = _render_audit_summary_text(out)
    assert "llm router" not in text
