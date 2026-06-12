"""
P1 — handlers/_query.py unit tests

Coverage:
  - _extract_feishu_urls    Feishu document URL extraction (pure regex)
  - _inject_doc_context     automatic document context injection (mock FeishuDocClient)
"""
import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Initialize config ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_hqtest_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "doc_auto_inject": True,
    "doc_inject_max_chars": 500,
    "doc_inject_max_docs": 2,
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.handlers._query import _extract_feishu_urls, _inject_doc_context


# ═══════════════════════════════════════════════════════════════════
#  _extract_feishu_urls
# ═══════════════════════════════════════════════════════════════════

class TestExtractFeishuUrls(unittest.TestCase):
    def test_docx_url(self):
        text = "请看这个文档 https://mycompany.feishu.cn/docx/AbCdEfGhIjKl"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertIn("docx", urls[0])

    def test_wiki_url(self):
        text = "参考 https://myco.feishu.cn/wiki/SomeWikiToken 的内容"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertIn("wiki", urls[0])

    def test_sheets_url(self):
        text = "https://acme.feishu.cn/sheets/SheetToken123?sheet=abc"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)

    def test_multiple_urls(self):
        text = ("看 https://a.feishu.cn/docx/D1 和 "
                "https://b.feishu.cn/wiki/W2 这两个")
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 2)

    def test_no_feishu_url(self):
        text = "普通消息，没有文档链接，看 https://github.com/xxx"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 0)

    def test_empty_text(self):
        self.assertEqual(_extract_feishu_urls(""), [])

    def test_url_stops_at_whitespace(self):
        text = "https://x.feishu.cn/docx/ABC def"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertNotIn(" def", urls[0])

    def test_url_stops_at_chinese_bracket(self):
        text = "（https://x.feishu.cn/docx/ABC）"
        urls = _extract_feishu_urls(text)
        # URL should not include the closing bracket
        if urls:
            self.assertNotIn("）", urls[0])

    def test_docs_url(self):
        text = "https://company.feishu.cn/docs/DocToken"
        urls = _extract_feishu_urls(text)
        self.assertEqual(len(urls), 1)


# ═══════════════════════════════════════════════════════════════════
#  _inject_doc_context
# ═══════════════════════════════════════════════════════════════════

def _make_doc_result(title="测试文档", content="文档正文内容"):
    r = MagicMock()
    r.title = title
    r.content = content
    return r


class TestInjectDocContext(unittest.TestCase):
    def test_no_url_returns_original(self):
        text = "普通消息，没有飞书链接"
        result = _inject_doc_context(text, "chat1")
        self.assertEqual(result, text)

    # _inject_doc_context imports FeishuDocClient / parse_doc_url from larkhelm.lark_client internally,
    # so we must patch the lark_client module rather than the _query module
    _PATCH_CLIENT = "larkhelm.lark_client.FeishuDocClient"
    _PATCH_PARSE  = "larkhelm.lark_client.parse_doc_url"

    def test_url_triggers_injection(self):
        text = "请读 https://x.feishu.cn/docx/ABC"
        mock_client = MagicMock()
        mock_client.read.return_value = _make_doc_result("标题A", "内容A")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertIn("文档内容", result)
        self.assertIn("标题A", result)
        self.assertIn("内容A", result)
        self.assertIn(text, result)

    def test_permission_error_adds_placeholder(self):
        from larkhelm.lark_client import DocPermissionError
        text = "https://x.feishu.cn/docx/ABC"
        mock_client = MagicMock()
        mock_client.read.side_effect = DocPermissionError("no perm")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertIn("无读取权限", result)

    def test_doc_error_silently_skipped(self):
        from larkhelm.lark_client import DocError
        text = "https://x.feishu.cn/docx/ABC"
        mock_client = MagicMock()
        mock_client.read.side_effect = DocError("some error")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertEqual(result, text)

    def test_max_docs_limit(self):
        """Documents beyond doc_inject_max_docs=2 should not be injected."""
        text = ("https://x.feishu.cn/docx/A "
                "https://x.feishu.cn/docx/B "
                "https://x.feishu.cn/docx/C")
        call_count = []
        mock_client = MagicMock()

        def mock_read(ref, max_chars):
            call_count.append(1)
            return _make_doc_result(f"doc{len(call_count)}", f"content{len(call_count)}")

        mock_client.read.side_effect = mock_read

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            _inject_doc_context(text, "chat1")

        self.assertLessEqual(len(call_count), _cfg_module.DOC_INJECT_MAX_DOCS)

    def test_unrecognized_url_returns_original(self):
        """When parse_doc_url returns None the URL should be skipped."""
        text = "https://x.feishu.cn/some-unknown-path/token"
        mock_client = MagicMock()

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=None):
            result = _inject_doc_context(text, "chat1")

        self.assertEqual(result, text)
        mock_client.read.assert_not_called()

    def test_injected_content_prepended(self):
        """Injected content should appear before the original message."""
        text = "原始消息 https://x.feishu.cn/docx/T"
        mock_client = MagicMock()
        mock_client.read.return_value = _make_doc_result("标题", "内容")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertLess(result.index("文档内容"), result.index("原始消息"))

    def test_separator_between_injections_and_original(self):
        """There should be a --- separator between injected content and the original message."""
        text = "消息 https://x.feishu.cn/docx/T"
        mock_client = MagicMock()
        mock_client.read.return_value = _make_doc_result("T", "C")

        with patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=MagicMock()):
            result = _inject_doc_context(text, "chat1")

        self.assertIn("---", result)


class TestInjectDocContextSessionDedup(unittest.TestCase):
    """Session-level doc-injection dedup: the same (chat_id, backend,
    doc_token, content_hash) injected twice within one backend session
    downgrades the second injection to a one-line marker. Records are
    deferred — collected into ``pending_doc_records`` and only effective
    after the caller commits them via ``record_doc_injection`` (the query
    path does so after backend success). Clearing the sid
    (``chat_state._clear_sid(chat_id, model)``) drops only that backend's
    records.
    """

    _PATCH_CLIENT = "larkhelm.lark_client.FeishuDocClient"
    _PATCH_PARSE  = "larkhelm.lark_client.parse_doc_url"
    _CHAT = "chat_dedup"
    _TEXT = "请读 https://x.feishu.cn/docx/TOK1"

    def setUp(self):
        import larkhelm._context_cache as _cc
        _cc.reset_for_tests()

    def _inject_once(self, content: str, backend: str = "claude",
                     commit: bool = True, chat_id: str | None = None) -> str:
        """Run one injection; ``commit=True`` mimics the _do_query success
        path (records committed after the backend returned), ``commit=False``
        mimics /cancel or an all-backends-failed query."""
        chat_id = chat_id or self._CHAT
        mock_client = MagicMock()
        mock_client.read.return_value = _make_doc_result("标题A", content)
        ref = MagicMock()
        ref.token = "TOK1"
        pending: list[tuple[str, str]] = []
        # Disable the TTL doc cache so each call re-reads `content` —
        # the dedup record (not the cache) is the unit under test here.
        with patch.object(_cfg_module, "DOC_INJECT_CACHE_ENABLED", False), \
             patch(self._PATCH_CLIENT, return_value=mock_client), \
             patch(self._PATCH_PARSE, return_value=ref):
            result = _inject_doc_context(self._TEXT, chat_id, backend=backend,
                                         pending_doc_records=pending)
        if commit:
            from larkhelm._context_cache import record_doc_injection
            for tok, h in pending:
                record_doc_injection(chat_id, backend, tok, h)
        return result

    def test_second_injection_downgraded_to_marker(self):
        first = self._inject_once("正文内容V1")
        self.assertIn("正文内容V1", first)

        second = self._inject_once("正文内容V1")
        self.assertNotIn("正文内容V1", second)
        self.assertIn("本会话已注入且未变更", second)
        self.assertIn("标题A", second)
        # original message always survives
        self.assertIn(self._TEXT, second)

    def test_changed_content_reinjects_full_body(self):
        self._inject_once("正文内容V1")
        changed = self._inject_once("正文内容V2")
        self.assertIn("正文内容V2", changed)
        self.assertNotIn("本会话已注入且未变更", changed)

    def test_clear_sid_restores_full_injection(self):
        self._inject_once("正文内容V1")
        # /reset、session_guard、runner crash-recovery 都收敛到 _clear_sid
        from larkhelm.chat_state import _clear_sid
        _clear_sid(self._CHAT, "claude")
        again = self._inject_once("正文内容V1")
        self.assertIn("正文内容V1", again)
        self.assertNotIn("本会话已注入且未变更", again)

    def test_dedup_is_per_chat(self):
        self._inject_once("正文内容V1")
        other = self._inject_once("正文内容V1", chat_id="another_chat")
        self.assertIn("正文内容V1", other)

    # ── Cross-backend correctness (sessions are per-(chat, backend)) ──────

    def test_cross_backend_reinjects_full_body(self):
        """A body injected into the claude session must NOT be downgraded
        when the same URL is routed to another backend (/g, /model switch,
        failover): that backend's session never saw the body."""
        self._inject_once("正文内容V1", backend="claude")
        on_gemini = self._inject_once("正文内容V1", backend="gemini")
        self.assertIn("正文内容V1", on_gemini)
        self.assertNotIn("本会话已注入且未变更", on_gemini)
        # …and back on claude the record still holds.
        on_claude = self._inject_once("正文内容V1", backend="claude")
        self.assertIn("本会话已注入且未变更", on_claude)

    def test_clear_sid_only_drops_that_backends_records(self):
        self._inject_once("正文内容V1", backend="claude")
        self._inject_once("正文内容V1", backend="gemini")
        from larkhelm.chat_state import _clear_sid
        _clear_sid(self._CHAT, "gemini")
        # gemini session reset → full body again
        gem = self._inject_once("正文内容V1", backend="gemini")
        self.assertIn("正文内容V1", gem)
        # claude session untouched → still deduped
        cla = self._inject_once("正文内容V1", backend="claude")
        self.assertIn("本会话已注入且未变更", cla)

    def test_clear_doc_injections_without_backend_drops_all(self):
        self._inject_once("正文内容V1", backend="claude")
        self._inject_once("正文内容V1", backend="gemini")
        from larkhelm._context_cache import clear_doc_injections
        clear_doc_injections(self._CHAT)
        for b in ("claude", "gemini"):
            full = self._inject_once("正文内容V1", backend=b, commit=False)
            self.assertIn("正文内容V1", full)

    # ── Deferred commit (query failure / cancel leaves no record) ─────────

    def test_record_deferred_until_commit(self):
        from larkhelm._context_cache import doc_injection_seen
        import hashlib
        self._inject_once("正文内容V1", commit=False)
        h = hashlib.sha256("正文内容V1".encode("utf-8")).hexdigest()
        self.assertFalse(doc_injection_seen(self._CHAT, "claude", "TOK1", h))

    def test_failed_query_reinjects_full_body(self):
        """No commit (cancelled / all backends failed) → the next attempt
        must inject the full body again, not the marker."""
        self._inject_once("正文内容V1", commit=False)
        retry = self._inject_once("正文内容V1", commit=False)
        self.assertIn("正文内容V1", retry)
        self.assertNotIn("本会话已注入且未变更", retry)


class TestRunBackendSingleSidSkipsSystem(unittest.TestCase):
    """REQ-05 regression: claude_cli with non-empty sid must NOT re-inject
    system_prompt — the resumed session already carries that context, and
    re-passing it on every turn multiplies prompt size linearly.
    """

    def test_cli_resumed_skips_system_prompt(self):
        from larkhelm.handlers import _query as _q
        from larkhelm.backend_registry import BackendSpec

        spec = BackendSpec(
            id="claude", provider="claude_cli", display_name="Claude",
            role="orchestrator", tags=[], command="claude",
            healthy=True, enabled=True,
        )

        captured_kwargs = {}

        def _fake_run_claude(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return "stub-output"

        with patch.object(_q, "_load_sid", return_value="sess-123"), \
             patch("larkhelm.backend_cli.run_claude", side_effect=_fake_run_claude) as m:
            out = _q._run_backend_single(
                spec, "chat_x", "hello", cwd="/tmp",
                cancel_ev=None, on_text=None, on_tool=None,
                on_tool_result=None, on_soft_timeout=None,
                recent_turns="some recent turns text",
                extra_system="some system text",
            )

        self.assertEqual(out, "stub-output")
        # system_prompt must be None on resumed CLI sessions, regardless of
        # how juicy recent_turns / extra_system looked.
        self.assertIs(captured_kwargs.get("system_prompt"), None)
        self.assertEqual(m.call_count, 1)


if __name__ == "__main__":
    unittest.main()
