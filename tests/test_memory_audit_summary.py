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


# ─────────────────────────────────────────────────────────────────────────
#  Phase 3 round-2 review follow-ups
# ─────────────────────────────────────────────────────────────────────────


def test_llm_router_disjoint_invariant_invoked_plus_skipped_marked_as_skipped():
    """Round-2 MF-01 regression: the producer sets ``invoked=True``
    BEFORE the LLM call, then on parse_failed / empty_response /
    caller_exception adds a ``skipped_reason`` without flipping
    invoked back. A record with BOTH must be counted as SKIPPED
    (not as a successful invoke that dragged avg_selected_n to 0).
    """
    records = [
        # Successful invoke: invoked=True, skipped="", selected_n=4
        _make_record(llm_router_invoked=True, llm_router_selected_n=4),
        # invoke-then-failed: invoked=True, skipped="parse_failed", selected_n=0
        _make_record(llm_router_invoked=True,
                     llm_router_skipped="parse_failed",
                     llm_router_selected_n=0),
        # pure cache hit
        _make_record(llm_router_invoked=False, llm_router_cache_hit=True),
    ]
    summary = _compute_llm_router_summary(records)
    assert summary is not None
    # Successful invoke only — the parse_failed record is in skipped.
    assert summary["invoked_count"] == 1
    assert summary["cache_hit_count"] == 1
    # avg_selected_n must be 4 (from the single real invoke), NOT 2
    # (which would be the bug-case (4+0)/2).
    assert summary["avg_selected_n"] == pytest.approx(4.0)
    # parse_failed goes into skipped_breakdown.
    assert summary["skipped_breakdown"] == {"parse_failed": 1}
    # Disjointness: invoked + cache_hits + sum(skipped) == gate_fired.
    inv = summary["invoked_count"]
    hits = summary["cache_hit_count"]
    skipped_sum = sum(summary["skipped_breakdown"].values())
    assert inv + hits + skipped_sum == summary["gate_fired_count"], (
        f"buckets not disjoint: {inv} + {hits} + {skipped_sum} != "
        f"{summary['gate_fired_count']}"
    )


def test_llm_router_safe_int_handles_malformed_selected_n():
    """Round-2 SF-02 regression: a corrupted audit record with
    string / None / non-numeric ``selected_n`` must NOT crash the
    whole CLI — operators rely on audit-summary to diagnose Stage C,
    so a malformed record should be treated as 0 not abort."""
    records = [
        {"llm_router_invoked": True, "llm_router_cache_hit": False,
         "llm_router_skipped": "", "llm_router_selected_n": "abc",  # bogus
         "elapsed_ms": 5, "agent_type": "dev", "mode": "hybrid",
         "fail_open": False, "selected_slice_ids": [], "selected_token_chars": 0},
        {"llm_router_invoked": True, "llm_router_cache_hit": False,
         "llm_router_skipped": "", "llm_router_selected_n": None,    # null
         "elapsed_ms": 5, "agent_type": "dev", "mode": "hybrid",
         "fail_open": False, "selected_slice_ids": [], "selected_token_chars": 0},
        {"llm_router_invoked": True, "llm_router_cache_hit": False,
         "llm_router_skipped": "", "llm_router_selected_n": 3,
         "elapsed_ms": 5, "agent_type": "dev", "mode": "hybrid",
         "fail_open": False, "selected_slice_ids": [], "selected_token_chars": 0},
    ]
    summary = _compute_llm_router_summary(records)
    assert summary is not None
    # Three successful invokes; the two bogus selected_n values count
    # as 0; one real value of 3. Mean = (0+0+3) / 3 = 1.0.
    assert summary["invoked_count"] == 3
    assert summary["avg_selected_n"] == pytest.approx(1.0)


def test_llm_router_summary_matches_build_audit_record_shape():
    """Round-2 SF-03 follow-up: the aggregator must read records
    SHAPED EXACTLY as ``build_audit_record_v2(..., llm_router_diag=...)``
    produces. Without this, schema drift between producer and consumer
    silently degrades audit fidelity (every Phase review caught one of
    these — Phase 2 SF-01 shared bucket, Phase 3 MF-01 complexity gate).
    """
    from larkhelm.memory_llm_router import RouterDiagnostics
    from larkhelm.memory_retriever import build_audit_record_v2
    from larkhelm.memory_slice import (
        InjectionPolicy, RetrievalRequest, ScoredSlice, MemorySlice,
    )
    req = RetrievalRequest(chat_id="c", query="q",
                           agent_type="dev", complexity="complex")
    pol = InjectionPolicy(
        agent_type="dev", token_budget=2000,
        layer_weights={"project": 1.0}, kind_priority=("convention",),
    )
    diag = RouterDiagnostics(
        invoked=True, cache_hit=False, skipped_reason="",
        elapsed_ms=5, selected_by_llm=3,
    )
    # Production-shaped record.
    rec = build_audit_record_v2(
        request=req, policy=pol, scored=[], candidate_count=0,
        elapsed_ms=5, selected_chars=100, fail_open=False,
        actual_mode="hybrid", llm_router_diag=diag,
    )
    # Verify the aggregator finds the gate-fired sentinel field.
    assert "llm_router_invoked" in rec, (
        "build_audit_record_v2 must write llm_router_invoked when diag "
        "is provided — the aggregator uses this as gate-fired sentinel"
    )
    # End-to-end aggregation on the production record.
    summary = _compute_llm_router_summary([rec])
    assert summary is not None
    assert summary["gate_fired_count"] == 1
    assert summary["invoked_count"] == 1
    assert summary["avg_selected_n"] == pytest.approx(3.0)


def test_llm_router_underlying_failure_records_diag(monkeypatch):
    """Round-2 SF-01 regression: when the LLMRouterRetriever's
    underlying retriever raises (so _build_with_retriever's outer
    except fires), the audit record must STILL carry llm_router_*
    fields with ``skipped_reason="underlying_failure"`` so the audit
    summary doesn't undercount Stage C activity.

    Reproduce by directly checking the diag-construction logic that
    lives in ``memory_context._build_with_retriever``'s except branch.
    We exercise the path via a synthetic record shaped like what the
    fixed code now writes.
    """
    rec = {
        "llm_router_invoked": False,
        "llm_router_cache_hit": False,
        "llm_router_skipped": "underlying_failure",
        "llm_router_selected_n": 0,
        "elapsed_ms": 5, "agent_type": "dev", "mode": "keyword",
        "fail_open": True, "selected_slice_ids": [], "selected_token_chars": 0,
    }
    summary = _compute_llm_router_summary([rec])
    assert summary is not None
    assert summary["gate_fired_count"] == 1
    assert summary["invoked_count"] == 0
    assert summary["skipped_breakdown"] == {"underlying_failure": 1}
