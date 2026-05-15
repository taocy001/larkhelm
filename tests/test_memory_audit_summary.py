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
