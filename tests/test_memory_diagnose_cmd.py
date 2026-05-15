"""Unit tests for ``_cmd_memory_diagnose`` (Phase D / Phase 2 REQ-38)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest import mock

import pytest


@pytest.fixture
def card_recorder(monkeypatch, fake_card_sender):
    """Make the ``fake_card_sender`` patches also reach the module-level binding
    in ``larkhelm.commands`` (where ``_cmd_memory_diagnose`` lives)."""
    import larkhelm.commands as _cmds
    import larkhelm.lark_client as _lc
    monkeypatch.setattr(_cmds, "send_card_reply", _lc.send_card_reply, raising=False)
    monkeypatch.setattr(_cmds, "send_card", _lc.send_card, raising=False)
    return fake_card_sender


def test_diagnose_renders_card(monkeypatch, card_recorder):
    """Records with selected slice ids are rendered as a single card body."""
    record = {
        "schema_version": "2",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_id": "chat-A", "agent_type": "dev",
        "mode": "hybrid", "declared_mode": "hybrid",
        "hybrid_alpha": 0.6, "query_head": "implement retriever",
        "query_token_count": 2, "candidate_slice_count": 10,
        "selected_slice_ids": ["abc123def456"],
        "selected_token_chars": 500, "top_k_returned": 1,
        "elapsed_ms": 99, "fail_open": False, "stale_hit_count": 0,
    }
    with mock.patch("larkhelm.memory_retriever.iter_audit_records",
                    return_value=iter([record])), \
         mock.patch("larkhelm.memory_retriever.load_slices", return_value=[]), \
         mock.patch("larkhelm.chat_state._get_cwd", return_value=None):
        from larkhelm.commands import _cmd_memory_diagnose
        _cmd_memory_diagnose("chat-A", "3", msg_id="msg-1")
    assert any(r.get("kind") == "send_card_reply" for r in card_recorder)
    body = next(r for r in card_recorder if r.get("kind") == "send_card_reply")["body"]
    assert "hybrid" in body
    assert "99ms" in body or "99 ms" in body or "elapsed" not in body  # don't pin format


def test_no_records_message(card_recorder):
    """Empty audit yields a friendly "no records" card."""
    with mock.patch("larkhelm.memory_retriever.iter_audit_records",
                    return_value=iter([])), \
         mock.patch("larkhelm.memory_retriever.load_slices", return_value=[]), \
         mock.patch("larkhelm.chat_state._get_cwd", return_value=None):
        from larkhelm.commands import _cmd_memory_diagnose
        _cmd_memory_diagnose("chat-A", "", msg_id=None)
    body = next(r for r in card_recorder if r.get("kind") == "send_card_reply")["body"]
    assert "24" in body and ("无召回" in body or "no records" in body.lower())


def test_card_omits_slice_body(card_recorder):
    """The diagnose card must NOT leak slice body — only titles, scores, modes."""
    record = {
        "schema_version": "2",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_id": "chat-A", "agent_type": "chat",
        "mode": "keyword", "declared_mode": "keyword",
        "hybrid_alpha": 0.6, "query_head": "secret query",
        "query_token_count": 2, "candidate_slice_count": 5,
        "selected_slice_ids": ["sensitive-id-12"],
        "selected_token_chars": 480, "top_k_returned": 1,
        "elapsed_ms": 25, "fail_open": False, "stale_hit_count": 0,
    }
    from larkhelm.memory_slice import MemorySlice
    fake_slice = MemorySlice(
        id="sensitive-id-12", layer="session",
        title="Sensitive Title",
        body="SECRET_PASSWORD=hunter2 (should not appear)",
    )
    with mock.patch("larkhelm.memory_retriever.iter_audit_records",
                    return_value=iter([record])), \
         mock.patch("larkhelm.memory_retriever.load_slices", return_value=[fake_slice]), \
         mock.patch("larkhelm.chat_state._get_cwd", return_value=None):
        from larkhelm.commands import _cmd_memory_diagnose
        _cmd_memory_diagnose("chat-A", "1", msg_id=None)
    body = next(r for r in card_recorder if r.get("kind") == "send_card_reply")["body"]
    assert "SECRET_PASSWORD" not in body
    assert "Sensitive Title" in body
