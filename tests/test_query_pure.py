"""Tests for ``larkhelm.handlers._query_pure`` (P1-1 PR1)."""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

# Initialize a minimal config so _cfg.MAX_CARD_LEN etc. exist for these
# tests (which exercise card-shaping helpers that read config globals).
import larkhelm.config as _cfg  # noqa: E402

if not getattr(_cfg, "_runtime", None):
    import json as _json
    import pathlib as _pl
    _tmp = _pl.Path("/tmp") / "larkhelm-test-pure"
    _tmp.mkdir(parents=True, exist_ok=True)
    _cfg_path = _tmp / "config.json"
    _cfg_path.write_text(_json.dumps({
        "APP_ID": "X", "APP_SECRET": "Y",
        "response_timeout": 30, "hard_timeout": 120,
    }))
    _cfg._init_runtime(str(_cfg_path), str(_tmp))

from larkhelm.handlers import _query_pure  # noqa: E402


class _FakeSpec:
    def __init__(self, sid: str, healthy: bool = True):
        self.id = sid
        self.healthy = healthy
        self.display_name = sid


class _FakeRegistry:
    def __init__(self, chain_ids):
        self._chain = [_FakeSpec(s) for s in chain_ids]

    def get_orchestrator_chain(self):
        return list(self._chain)


# ── build_init_card ────────────────────────────────────────────────────


def test_build_init_card_basic():
    card = _query_pure.build_init_card("Claude", "/tmp/work", "chat-abc")
    assert isinstance(card, str)
    payload = json.loads(card)
    # Reasonable structural assertion: cancel button payload mentions chat id.
    assert "cancel:chat-abc" in card
    # JSON object form.
    assert isinstance(payload, dict)


def test_build_init_card_unicode_model_name():
    card = _query_pure.build_init_card("Kimi-中文", "/tmp/中文路径", "chat-1")
    assert "Kimi-中文" in card or "Kimi-\\u4e2d\\u6587" in card


# ── build_failover_chain ───────────────────────────────────────────────


def test_build_failover_chain_prepends_primary():
    reg = _FakeRegistry(["claude", "gemini"])
    primary = _FakeSpec("kimi")
    chain = _query_pure.build_failover_chain(primary, reg, force_direct=False)
    assert [s.id for s in chain] == ["kimi", "claude", "gemini"]


def test_build_failover_chain_force_direct_collapses():
    reg = _FakeRegistry(["claude", "gemini"])
    primary = _FakeSpec("kimi")
    chain = _query_pure.build_failover_chain(primary, reg, force_direct=True)
    assert [s.id for s in chain] == ["kimi"]


def test_build_failover_chain_reorders_existing_primary():
    reg = _FakeRegistry(["claude", "gemini"])
    primary = _FakeSpec("gemini")
    chain = _query_pure.build_failover_chain(primary, reg, force_direct=False)
    assert [s.id for s in chain] == ["gemini", "claude"]


def test_build_failover_chain_unhealthy_primary_not_prepended():
    reg = _FakeRegistry(["claude", "gemini"])
    primary = _FakeSpec("kimi", healthy=False)
    chain = _query_pure.build_failover_chain(primary, reg, force_direct=False)
    assert [s.id for s in chain] == ["claude", "gemini"]


# ── select_legacy_runner ───────────────────────────────────────────────


def test_select_legacy_runner_returns_correct_callable():
    from larkhelm.ai_runner import (
        query_claude, query_gemini, query_kimi, query_deepseek,
    )
    assert _query_pure.select_legacy_runner("claude") is query_claude
    assert _query_pure.select_legacy_runner("gemini") is query_gemini
    assert _query_pure.select_legacy_runner("kimi") is query_kimi
    assert _query_pure.select_legacy_runner("deepseek") is query_deepseek


def test_select_legacy_runner_unknown_falls_back_to_claude():
    from larkhelm.ai_runner import query_claude
    assert _query_pure.select_legacy_runner("unknown") is query_claude
    assert _query_pure.select_legacy_runner("") is query_claude


# ── format_completion_card ─────────────────────────────────────────────


def test_format_completion_card_basic_short_output():
    chunks, note, tools_payload = _query_pure.format_completion_card(
        m_name="Claude", output="hello", n_tools=2, elapsed="3s",
        final_tools=[{"name": "Read", "desc": "/tmp"}], max_card_len=3000,
    )
    assert chunks  # at least one chunk
    assert "使用了 2 次工具" in note
    assert "耗时 3s" in note
    assert tools_payload == [{"name": "Read", "desc": "/tmp"}]


def test_format_completion_card_empty_output():
    chunks, note, tools_payload = _query_pure.format_completion_card(
        m_name="Claude", output="", n_tools=0, elapsed="0s",
        final_tools=[], max_card_len=3000,
    )
    assert chunks == [""] or chunks == []
    assert "耗时" in note
    assert tools_payload is None


def test_format_completion_card_drops_huge_tools_payload():
    huge = [{"name": "X", "desc": "a" * 25_000}]
    _, _, payload = _query_pure.format_completion_card(
        m_name="Claude", output="ok", n_tools=1, elapsed="1s",
        final_tools=huge, max_card_len=3000,
    )
    assert payload is None


# ── cleanup_temp_images ────────────────────────────────────────────────


def test_cleanup_temp_images_unlinks_tmp_paths(tmp_path):
    img = tmp_path / "fake.png"
    img.write_text("hello")
    # Force the path under /tmp/ — symlink approach
    target = "/tmp/__larkhelm_test_q_pure_img.png"
    try:
        with open(target, "wb") as fh:
            fh.write(b"x")
        _query_pure.cleanup_temp_images([target])
        assert not os.path.exists(target)
    finally:
        try:
            os.unlink(target)
        except FileNotFoundError:
            pass


def test_cleanup_temp_images_ignores_none_and_non_tmp():
    # Use a path that's NOT under /tmp/ — only /tmp/ entries get unlinked.
    import pathlib as _pl
    keep_dir = _pl.Path.home() / ".larkhelm_test_pure_keep"
    keep_dir.mkdir(parents=True, exist_ok=True)
    outside = keep_dir / "keep.png"
    outside.write_text("keep me")
    try:
        _query_pure.cleanup_temp_images(None)
        _query_pure.cleanup_temp_images([str(outside)])
        assert outside.exists()
    finally:
        try:
            outside.unlink()
            keep_dir.rmdir()
        except Exception:
            pass


# ── inject_doc_and_memory ──────────────────────────────────────────────


def test_inject_doc_and_memory_no_doc_no_memory(monkeypatch):
    # Force all downstream paths to no-op.
    monkeypatch.setattr(
        "larkhelm.handlers._query._inject_doc_context",
        lambda msg, chat_id: msg, raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.log._get_recent_turns",
        lambda chat_id, dedup_prefix=None: "",
        raising=False,
    )
    monkeypatch.setattr("larkhelm.memory.load_memory", lambda chat_id: "")
    monkeypatch.setattr(
        "larkhelm.memory.get_memory_context_v2",
        lambda *a, **kw: ("", []),
    )

    msg, mem, deduped = _query_pure.inject_doc_and_memory(
        "hi", "chat-1", "/tmp", doc_auto_inject=True, has_doc_urls=False,
    )
    assert msg == "hi"
    assert mem == ""
    assert deduped == []


def test_inject_doc_and_memory_returns_enriched(monkeypatch):
    monkeypatch.setattr(
        "larkhelm.handlers._query._inject_doc_context",
        lambda msg, chat_id: f"[DOC]\n{msg}", raising=False,
    )
    monkeypatch.setattr(
        "larkhelm.log._get_recent_turns",
        lambda chat_id, dedup_prefix=None: "u: a\nb: c",
        raising=False,
    )
    monkeypatch.setattr("larkhelm.memory.load_memory", lambda chat_id: "")
    monkeypatch.setattr(
        "larkhelm.memory.get_memory_context_v2",
        lambda *a, **kw: ("[MEM]", ["u: a", "b: c"]),
    )
    msg, mem, deduped = _query_pure.inject_doc_and_memory(
        "hello", "chat-1", "/tmp", doc_auto_inject=True, has_doc_urls=False,
    )
    assert msg.startswith("[DOC]")
    assert mem == "[MEM]"
    assert deduped == ["u: a", "b: c"]


# ── extract_feishu_urls ────────────────────────────────────────────────


def test_extract_feishu_urls_finds_links():
    txt = "see https://abc.feishu.cn/docx/TOKEN and https://x.feishu.cn/wiki/W"
    urls = _query_pure.extract_feishu_urls(txt)
    assert len(urls) == 2


def test_extract_feishu_urls_empty():
    assert _query_pure.extract_feishu_urls("no urls here") == []
