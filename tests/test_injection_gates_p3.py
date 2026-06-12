"""Doc injection pins after relevance-gate removal.

The P2b/P3 doc relevance gate was deleted: ``_compute_doc_query_relevance``
was a stub that always returned 1.0, so the gate branch and its config flags
(``doc_inject_relevance_gate_enabled`` / ``doc_inject_relevance_threshold``)
could never trigger in production. These tests pin the post-removal contract.
"""
import pytest
from unittest.mock import patch, MagicMock

# ── 1. stub + gate fully removed ───────────────────────────────────────────

def test_relevance_stub_removed():
    """_compute_doc_query_relevance no longer exists in handlers._query."""
    from larkhelm.handlers import _query as qmod
    assert not hasattr(qmod, "_compute_doc_query_relevance")

def test_config_relevance_flags_removed():
    """DOC_INJECT_RELEVANCE_* module attributes are gone from config."""
    from larkhelm import config as cfg_mod
    assert not hasattr(cfg_mod, "DOC_INJECT_RELEVANCE_GATE_ENABLED")
    assert not hasattr(cfg_mod, "DOC_INJECT_RELEVANCE_THRESHOLD")

# ── 2. full content is always injected ─────────────────────────────────────

def test_full_content_injected():
    """The entire result.content is injected (no relevance gating)."""
    from larkhelm.handlers import _query as qmod
    mock_cfg = MagicMock()
    mock_cfg.DOC_INJECT_MAX_DOCS = 3
    mock_cfg.DOC_INJECT_MAX_CHARS = 20000
    mock_cfg.DOC_AUTO_INJECT = True
    mock_cfg.DOC_INJECT_CACHE_ENABLED = False

    doc_result = MagicMock()
    doc_result.title = "MyDoc"
    doc_result.content = "x" * 15000  # big content

    with patch.object(qmod, "_cfg", mock_cfg), \
         patch.object(qmod, "_extract_feishu_urls", return_value=["https://x.feishu.cn/docx/abc"]), \
         patch("larkhelm.lark_client.parse_doc_url", return_value=MagicMock()), \
         patch("larkhelm.lark_client.FeishuDocClient") as MockClient, \
         patch("larkhelm.lark_client.DocPermissionError", Exception), \
         patch("larkhelm.lark_client.DocError", Exception):
        MockClient.return_value.read.return_value = doc_result
        result = qmod._inject_doc_context("hello feishu link", "chat1")
    # Full content should be present
    assert "x" * 15000 in result
