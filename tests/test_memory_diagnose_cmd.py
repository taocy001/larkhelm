"""Unit tests for ``_cmd_memory_diagnose`` (retriever removed)."""
from __future__ import annotations

import pytest


@pytest.fixture
def card_recorder(monkeypatch, fake_card_sender):
    import larkhelm.commands as _cmds
    import larkhelm.lark_client as _lc
    monkeypatch.setattr(_cmds, "send_card_reply", _lc.send_card_reply, raising=False)
    return fake_card_sender


def test_diagnose_returns_removed_stub(card_recorder):
    """Diagnose command returns a 'feature removed' stub card."""
    from larkhelm.commands import _cmd_memory_diagnose
    _cmd_memory_diagnose("chat-A", "3", msg_id="msg-1")
    assert any(r.get("kind") == "send_card_reply" for r in card_recorder)
    body = next(r for r in card_recorder if r.get("kind") == "send_card_reply")["body"]
    assert "移除" in body or "removed" in body.lower()
