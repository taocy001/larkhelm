"""Phase-3 injection gate tests: doc relevance scoring + truncation gate."""
import pytest
from unittest.mock import patch, MagicMock

# ── 1. _compute_doc_query_relevance ────────────────────────────────────────

def test_relevance_returns_1_when_embedding_disabled():
    """EMBEDDING_ENABLED=False → fail-open, returns 1.0."""
    from larkhelm.handlers._query import _compute_doc_query_relevance
    with patch("larkhelm.handlers._query._cfg") as mock_cfg:
        mock_cfg.EMBEDDING_ENABLED = False
        assert _compute_doc_query_relevance("query", "title", "snippet") == 1.0

def test_relevance_returns_1_on_import_error():
    """Import error of embedding backend → fail-open."""
    from larkhelm.handlers._query import _compute_doc_query_relevance
    with patch("larkhelm.handlers._query._cfg") as mock_cfg:
        mock_cfg.EMBEDDING_ENABLED = True
        with patch.dict("sys.modules", {"larkhelm.memory_embedding": None}):
            assert _compute_doc_query_relevance("q", "t", "s") == 1.0

def test_relevance_always_returns_1():
    """Embedding removed — function always returns 1.0 (fail-open)."""
    from larkhelm.handlers._query import _compute_doc_query_relevance
    assert _compute_doc_query_relevance("query", "title", "snippet") == 1.0

# ── 2. gate disabled → no change ──────────────────────────────────────────

def test_gate_off_full_content_injected():
    """DOC_INJECT_RELEVANCE_GATE_ENABLED=False → entire result.content injected."""
    # Build a mock _inject_doc_context call environment
    # (testing via direct invocation is easier than testing the full pipeline)
    # We verify that _injected_content == result.content when gate is off
    from larkhelm.handlers import _query as qmod
    mock_cfg = MagicMock()
    mock_cfg.DOC_INJECT_RELEVANCE_GATE_ENABLED = False
    mock_cfg.DOC_INJECT_RELEVANCE_THRESHOLD = 0.3
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
    assert "x" * 100 in result  # at least first 100 chars of content

# ── 3. gate enabled, low relevance → title-only ───────────────────────────

def test_gate_on_low_relevance_skips_fulltext():
    """sim < threshold → inject title hint only, skip body."""
    from larkhelm.handlers import _query as qmod
    mock_cfg = MagicMock()
    mock_cfg.DOC_INJECT_RELEVANCE_GATE_ENABLED = True
    mock_cfg.DOC_INJECT_RELEVANCE_THRESHOLD = 0.3
    mock_cfg.DOC_INJECT_MAX_DOCS = 3
    mock_cfg.DOC_INJECT_MAX_CHARS = 20000
    mock_cfg.DOC_AUTO_INJECT = True
    mock_cfg.DOC_INJECT_CACHE_ENABLED = False

    doc_result = MagicMock()
    doc_result.title = "IrrelevantDoc"
    doc_result.content = "irrelevant content " * 200

    with patch.object(qmod, "_cfg", mock_cfg), \
         patch.object(qmod, "_extract_feishu_urls", return_value=["https://x.feishu.cn/docx/abc"]), \
         patch.object(qmod, "_compute_doc_query_relevance", return_value=0.1), \
         patch("larkhelm.lark_client.parse_doc_url", return_value=MagicMock()), \
         patch("larkhelm.lark_client.FeishuDocClient") as MockClient, \
         patch("larkhelm.lark_client.DocPermissionError", Exception), \
         patch("larkhelm.lark_client.DocError", Exception):
        MockClient.return_value.read.return_value = doc_result
        result = qmod._inject_doc_context("hello world", "chat1")
    assert "相关度过低" in result
    assert "irrelevant content " * 5 not in result  # body not injected

# ── 4. gate enabled, mid relevance → truncated ────────────────────────────

def test_gate_on_mid_relevance_truncates():
    """0.3 ≤ sim < 0.6 → content truncated to DOC_INJECT_MAX_CHARS // 2."""
    from larkhelm.handlers import _query as qmod
    mock_cfg = MagicMock()
    mock_cfg.DOC_INJECT_RELEVANCE_GATE_ENABLED = True
    mock_cfg.DOC_INJECT_RELEVANCE_THRESHOLD = 0.3
    mock_cfg.DOC_INJECT_MAX_DOCS = 3
    mock_cfg.DOC_INJECT_MAX_CHARS = 1000  # half = 500
    mock_cfg.DOC_AUTO_INJECT = True
    mock_cfg.DOC_INJECT_CACHE_ENABLED = False

    doc_result = MagicMock()
    doc_result.title = "SomewhatRelated"
    doc_result.content = "A" * 800  # > 500

    with patch.object(qmod, "_cfg", mock_cfg), \
         patch.object(qmod, "_extract_feishu_urls", return_value=["https://x.feishu.cn/docx/abc"]), \
         patch.object(qmod, "_compute_doc_query_relevance", return_value=0.45), \
         patch("larkhelm.lark_client.parse_doc_url", return_value=MagicMock()), \
         patch("larkhelm.lark_client.FeishuDocClient") as MockClient, \
         patch("larkhelm.lark_client.DocPermissionError", Exception), \
         patch("larkhelm.lark_client.DocError", Exception):
        MockClient.return_value.read.return_value = doc_result
        result = qmod._inject_doc_context("hello", "chat1")
    # Content in result should be at most 500 chars (truncated)
    assert "A" * 500 in result
    assert "A" * 501 not in result

# ── 5. gate enabled, high relevance → full content ────────────────────────

def test_gate_on_high_relevance_full_content():
    """sim ≥ 0.6 → full content injected (no truncation)."""
    from larkhelm.handlers import _query as qmod
    mock_cfg = MagicMock()
    mock_cfg.DOC_INJECT_RELEVANCE_GATE_ENABLED = True
    mock_cfg.DOC_INJECT_RELEVANCE_THRESHOLD = 0.3
    mock_cfg.DOC_INJECT_MAX_DOCS = 3
    mock_cfg.DOC_INJECT_MAX_CHARS = 1000
    mock_cfg.DOC_AUTO_INJECT = True
    mock_cfg.DOC_INJECT_CACHE_ENABLED = False

    doc_result = MagicMock()
    doc_result.title = "HighlyRelevant"
    doc_result.content = "B" * 800

    with patch.object(qmod, "_cfg", mock_cfg), \
         patch.object(qmod, "_extract_feishu_urls", return_value=["https://x.feishu.cn/docx/abc"]), \
         patch.object(qmod, "_compute_doc_query_relevance", return_value=0.85), \
         patch("larkhelm.lark_client.parse_doc_url", return_value=MagicMock()), \
         patch("larkhelm.lark_client.FeishuDocClient") as MockClient, \
         patch("larkhelm.lark_client.DocPermissionError", Exception), \
         patch("larkhelm.lark_client.DocError", Exception):
        MockClient.return_value.read.return_value = doc_result
        result = qmod._inject_doc_context("hello", "chat1")
    assert "B" * 800 in result

# ── 6. config flag defaults ────────────────────────────────────────────────

def test_config_relevance_gate_default_false():
    """DOC_INJECT_RELEVANCE_GATE_ENABLED defaults to False."""
    from larkhelm import config as cfg_mod
    assert cfg_mod.DOC_INJECT_RELEVANCE_GATE_ENABLED is False

def test_config_relevance_threshold_default():
    """DOC_INJECT_RELEVANCE_THRESHOLD defaults to 0.3."""
    from larkhelm import config as cfg_mod
    assert cfg_mod.DOC_INJECT_RELEVANCE_THRESHOLD == pytest.approx(0.3)
