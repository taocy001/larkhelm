"""P2 REQ-05.1: tests for ``larkhelm.memory_global_slots``."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import memory_global_slots as _mgs  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_memory_home(monkeypatch, tmp_path: Path):
    import larkhelm.memory as _mem
    monkeypatch.setattr(_mem, "MEMORY_HOME_DIR", tmp_path / "memory", raising=False)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    # Wire a sender_open_id so _global_memory_file resolves to a real path.
    import larkhelm.chat_state as _cs
    with _cs._state_lock:
        _cs._chat_state_store.setdefault("oc_user", {})["sender_open_id"] = "ou_abc123"
    yield tmp_path / "memory"
    with _cs._state_lock:
        _cs._chat_state_store.pop("oc_user", None)


# ── parse_body ───────────────────────────────────────────────────────────


def test_parse_body_with_headings_returns_per_slot():
    body = (
        "## style\nshort answers, no fluff\n\n"
        "## format\nbullets > paragraphs\n\n"
        "## domain\nLinux + Python\n\n"
        "## expertise\nseasoned engineer"
    )
    slots = _mgs.parse_body(body)
    assert slots["style"].startswith("short answers")
    assert slots["format"].startswith("bullets")
    assert slots["domain"] == "Linux + Python"
    assert slots["expertise"] == "seasoned engineer"


def test_parse_body_legacy_no_headings_into_style():
    body = "freeform global memory line 1\nline 2"
    slots = _mgs.parse_body(body)
    # No ## headings → entire body lands in 'style' (legacy fallback).
    assert "freeform global memory" in slots["style"]
    assert slots["format"] == slots["domain"] == slots["expertise"] == ""


def test_parse_body_empty_returns_all_empty():
    slots = _mgs.parse_body("")
    assert all(v == "" for v in slots.values())
    assert set(slots) == set(_mgs.SLOT_NAMES)


# ── round-trip save/load ─────────────────────────────────────────────────


def test_save_load_roundtrip(isolated_memory_home):
    out = {
        "style": "terse",
        "format": "code",
        "domain": "memory layer",
        "expertise": "advanced",
    }
    _mgs.save_global_slots("oc_user", out)
    loaded = _mgs.load_global_slots("oc_user")
    assert loaded == out


def test_save_respects_slot_budget(isolated_memory_home):
    huge = "x" * (_mgs.SLOT_BUDGET + 200)
    _mgs.save_global_slots("oc_user", {"style": huge})
    loaded = _mgs.load_global_slots("oc_user")
    assert len(loaded["style"]) <= _mgs.SLOT_BUDGET


# ── merge_slot_update ────────────────────────────────────────────────────


def test_merge_slot_update_replaces_target_only():
    existing = {"style": "old", "format": "f1", "domain": "d1", "expertise": "e1"}
    new = _mgs.merge_slot_update(existing, "new style", "style")
    assert new["style"] == "new style"
    assert new["format"] == "f1"
    # Caller's dict is untouched.
    assert existing["style"] == "old"


def test_merge_slot_update_unknown_slot_dropped():
    existing = {s: "" for s in _mgs.SLOT_NAMES}
    new = _mgs.merge_slot_update(existing, "garbage", "made_up_slot")
    # Unknown slot → drop the update; existing dict unchanged.
    assert all(v == "" for v in new.values())


def test_merge_slot_update_typo_tolerant():
    existing = {s: "" for s in _mgs.SLOT_NAMES}
    new = _mgs.merge_slot_update(existing, "level", "expert")
    assert new["expertise"] == "level"


# ── render_for_context ───────────────────────────────────────────────────


def test_render_for_context_empty_returns_empty():
    assert _mgs.render_for_context({s: "" for s in _mgs.SLOT_NAMES}) == ""


def test_render_for_context_emits_only_nonempty_slots():
    out = _mgs.render_for_context({
        "style": "terse", "format": "", "domain": "Linux", "expertise": "",
    })
    assert "## style\nterse" in out
    assert "## domain\nLinux" in out
    assert "format" not in out
    assert "expertise" not in out


# ── is_enabled() honours flag ────────────────────────────────────────────


def test_is_enabled_default_false():
    assert _mgs.is_enabled() is False


def test_is_enabled_honours_flag(monkeypatch):
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "MEMORY_GLOBAL_PROFILE_SLOT_ENABLED", True, raising=False)
    assert _mgs.is_enabled() is True
