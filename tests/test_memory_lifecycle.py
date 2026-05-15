"""Unit tests for ``larkhelm.memory_lifecycle`` (Phase D / Phase 2)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from larkhelm.memory_lifecycle import (
    SliceMeta,
    inject_stale_marks,
    load_slice_meta,
    mark_stale_slices,
    save_slice_meta,
    unstale_slice_id,
)
from larkhelm.memory_slice import MemorySlice


def test_load_slice_meta_missing_returns_empty(tmp_path):
    meta = load_slice_meta(tmp_path / "nope.meta.json")
    assert isinstance(meta, SliceMeta)
    assert meta.stale_slice_ids == ()


def test_save_slice_meta_roundtrip_and_perms(tmp_path):
    target = tmp_path / "global_x.meta.json"
    meta = SliceMeta(
        schema_version=1,
        updated_at="2026-05-15T10:00:00+08:00",
        stale_slice_ids=("aaa111", "bbb222"),
        last_gc_at="2026-05-15T03:00:00+08:00",
        gc_window_days=60,
    )
    save_slice_meta(target, meta)
    reloaded = load_slice_meta(target)
    assert reloaded.stale_slice_ids == ("aaa111", "bbb222")
    assert reloaded.gc_window_days == 60
    # File mode must be 0600 (sensitive metadata, REQ-44 + design.md §3.5).
    mode = target.stat().st_mode & 0o777
    assert mode == 0o600


def test_inject_stale_marks_decorates(tmp_path):
    md_path = tmp_path / "project_demo.md"
    md_path.write_text("# anything", encoding="utf-8")
    meta = SliceMeta(stale_slice_ids=("hit",))
    save_slice_meta(md_path.with_suffix(".meta.json"), meta)
    slices = [
        MemorySlice(id="hit", layer="project", body="A"),
        MemorySlice(id="miss", layer="project", body="B"),
    ]
    decorated = inject_stale_marks(slices, [("project", "demo", md_path)])
    by_id = {s.id: s for s in decorated}
    assert by_id["hit"].stale is True
    assert by_id["miss"].stale is False


def test_unstale_slice_id_removes(tmp_path, monkeypatch):
    """unstale_slice_id removes the id from every .meta.json under MEMORY_HOME_DIR."""
    monkeypatch.setattr("larkhelm.memory.MEMORY_HOME_DIR", tmp_path)
    a = tmp_path / "global_x.meta.json"
    b = tmp_path / "project_y.meta.json"
    save_slice_meta(a, SliceMeta(stale_slice_ids=("kill", "keep")))
    save_slice_meta(b, SliceMeta(stale_slice_ids=("kill",)))
    assert unstale_slice_id("kill") is True
    assert load_slice_meta(a).stale_slice_ids == ("keep",)
    assert load_slice_meta(b).stale_slice_ids == ()


def test_unstale_slice_id_returns_false_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("larkhelm.memory.MEMORY_HOME_DIR", tmp_path)
    save_slice_meta(tmp_path / "global_x.meta.json", SliceMeta(stale_slice_ids=("a",)))
    assert unstale_slice_id("not-there") is False


def test_mark_stale_dry_run_writes_nothing(tmp_path, monkeypatch):
    """dry_run=True must not create or touch the .meta.json sidecar."""
    monkeypatch.setattr("larkhelm.memory.MEMORY_HOME_DIR", tmp_path)
    # Stub the retriever's resolver + slice loader so the test is hermetic.
    md = tmp_path / "session_chat-A.md"
    md.write_text("body", encoding="utf-8")

    with mock.patch("larkhelm.memory_retriever._resolve_layer_files",
                    return_value=[("session", "chat-A", md)]), \
         mock.patch("larkhelm.memory_retriever.load_slices",
                    return_value=[MemorySlice(id="x", layer="session", body="b")]), \
         mock.patch("larkhelm.memory_retriever.iter_audit_records",
                    return_value=iter([])):
        n = mark_stale_slices("chat-A", None, dry_run=True, window_days=30)
    assert n >= 1
    # No file should have been written in dry-run mode.
    assert not (md.with_suffix(".meta.json")).exists()


def test_mark_stale_writes_meta_json(tmp_path, monkeypatch):
    monkeypatch.setattr("larkhelm.memory.MEMORY_HOME_DIR", tmp_path)
    md = tmp_path / "session_chat-A.md"
    md.write_text("body", encoding="utf-8")
    slice_x = MemorySlice(id="x", layer="session", body="b")
    slice_y = MemorySlice(id="y", layer="session", body="c")
    hit_record = {"ts": datetime.now(timezone.utc).isoformat(),
                  "chat_id": "chat-A", "selected_slice_ids": ["y"]}
    with mock.patch("larkhelm.memory_retriever._resolve_layer_files",
                    return_value=[("session", "chat-A", md)]), \
         mock.patch("larkhelm.memory_retriever.load_slices",
                    return_value=[slice_x, slice_y]), \
         mock.patch("larkhelm.memory_retriever.iter_audit_records",
                    return_value=iter([hit_record])):
        n = mark_stale_slices("chat-A", None, dry_run=False, window_days=30)
    assert n == 1  # only `x` is freshly stale (y was hit)
    meta = load_slice_meta(md.with_suffix(".meta.json"))
    assert meta.stale_slice_ids == ("x",)
    assert meta.gc_window_days == 30
