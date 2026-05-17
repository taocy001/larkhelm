"""P2 AC-02: tests for ``larkhelm._message_pure`` five-function module.

Covers dedup hit/miss, ACL whitelist/reject, doc URL extraction (docx +
wiki + sheets), command vs. non-command routing. Indirectly contributes
to the AC-02 coverage requirement on ``handlers/_message.py`` because
the production routing path now calls
``_message_pure.extract_allowed_chat_decision`` for ACL.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm._message_pure import (  # noqa: E402
    AllowDecision,
    MessageKind,
    RouteDecision,
    classify_message_kind,
    extract_allowed_chat_decision,
    parse_doc_urls,
    route_to_command,
    should_skip_due_to_dedup,
)


# ── classify_message_kind ─────────────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ({"message_type": "text"},   MessageKind.TEXT),
    ({"message_type": "image"},  MessageKind.IMAGE),
    ({"message_type": "post"},   MessageKind.POST),
    ({"message_type": "audio"},  MessageKind.VOICE),
    ({"message_type": "file"},   MessageKind.FILE),
    ({"message_type": "sticker"}, MessageKind.UNKNOWN),
    ({},                          MessageKind.UNKNOWN),
    (None,                        MessageKind.UNKNOWN),
])
def test_classify_message_kind(raw, expected):
    assert classify_message_kind(raw) is expected


# ── extract_allowed_chat_decision ────────────────────────────────────────


def test_acl_empty_whitelist_allows_all():
    d = extract_allowed_chat_decision("oc_chat", allowed=set(), sender="user1")
    assert isinstance(d, AllowDecision)
    assert d.allowed and d.reason == "ok"


def test_acl_whitelist_hit_allows():
    d = extract_allowed_chat_decision("oc_chat", allowed={"oc_chat"}, sender="")
    assert d.allowed
    assert d.reason == "ok_no_sender"


def test_acl_whitelist_miss_rejects():
    d = extract_allowed_chat_decision("oc_bad", allowed={"oc_chat"})
    assert not d.allowed and d.reason == "acl_reject"


def test_acl_missing_chat_id_rejects():
    d = extract_allowed_chat_decision("", allowed={"oc_chat"})
    assert not d.allowed and d.reason == "missing_chat_id"


# ── should_skip_due_to_dedup ─────────────────────────────────────────────


def test_dedup_hit_returns_true():
    assert should_skip_due_to_dedup("evt_1", {"evt_1", "evt_2"}) is True


def test_dedup_miss_returns_false():
    assert should_skip_due_to_dedup("evt_3", {"evt_1", "evt_2"}) is False


def test_dedup_empty_event_id_falsey():
    assert should_skip_due_to_dedup("", {"evt_1"}) is False


# ── parse_doc_urls ───────────────────────────────────────────────────────


def test_parse_doc_urls_docx_wiki_sheets():
    text = (
        "看这个 https://xx.feishu.cn/docx/abc123 "
        "和 https://xx.feishu.cn/wiki/W_456 "
        "还有 https://xx.feishu.cn/sheets/S_789"
    )
    urls = parse_doc_urls(text)
    assert urls == [
        "https://xx.feishu.cn/docx/abc123",
        "https://xx.feishu.cn/wiki/W_456",
        "https://xx.feishu.cn/sheets/S_789",
    ]


def test_parse_doc_urls_dedupes_in_order():
    text = "a https://xx.feishu.cn/docx/aaa b https://xx.feishu.cn/docx/aaa"
    assert parse_doc_urls(text) == ["https://xx.feishu.cn/docx/aaa"]


def test_parse_doc_urls_returns_empty_when_no_match():
    assert parse_doc_urls("just some text") == []
    assert parse_doc_urls("") == []


# ── route_to_command ─────────────────────────────────────────────────────


def test_route_exact_command():
    d = route_to_command("/status", {"/help", "/status", "/reset"})
    assert isinstance(d, RouteDecision)
    assert d.is_command and d.handler_name == "/status" and d.args == ""


def test_route_prefix_command_extracts_args():
    d = route_to_command("/run echo hi", {"/run", "/cd"})
    assert d.is_command and d.handler_name == "/run"
    assert d.args == "echo hi"


def test_route_prefers_longer_match():
    """``/memory diagnose`` must win over ``/memory`` when both registered."""
    d = route_to_command("/memory diagnose 5", {"/memory", "/memory diagnose"})
    assert d.handler_name == "/memory diagnose"
    assert d.args == "5"


def test_route_non_command_returns_inactive():
    d = route_to_command("hello world", {"/help"})
    assert not d.is_command and d.handler_name == "" and d.args == ""


def test_route_empty_text():
    d = route_to_command("", {"/help"})
    assert not d.is_command
