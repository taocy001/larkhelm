"""Coverage for L2 cheap-backend failure paths in ``intent_router``.

Specifically the branches inside ``_call_cheap_backend`` and ``_resolve_l2``:
- ``call_backend_oneshot`` import failure → ``_call_cheap_backend`` returns None
- ``call_backend_oneshot`` raise → returns None (and silently logs)
- cheap backend missing in registry → fallback chat
- malformed JSON / nested JSON / fenced JSON parsing
- invalid intent name → coerced to chat

review.md §6 backlog: "_call_cheap_backend import failure / call failure" and
"`_JSON_RE` 改 `JSONDecoder.raw_decode`" both fall into this file.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pytest

from larkhelm.agent_hub import intent_router
from larkhelm.agent_hub.intent_types import IntentResult


class TestExtractFirstJsonObject(unittest.TestCase):
    """The new raw_decode-based extractor must handle nested JSON correctly,
    where the previous non-greedy regex truncated at the first inner ``}``.
    """

    def test_simple_object(self):
        out = intent_router._extract_first_json_object('{"intent":"dev"}')
        self.assertEqual(out, {"intent": "dev"})

    def test_nested_object(self):
        text = 'prefix {"intent":"dev","meta":{"complexity":"complex"}} trailing'
        out = intent_router._extract_first_json_object(text)
        self.assertEqual(out, {"intent": "dev", "meta": {"complexity": "complex"}})

    def test_first_of_two_objects(self):
        text = 'noise {"intent":"crew"} more {"intent":"dev"}'
        out = intent_router._extract_first_json_object(text)
        self.assertEqual(out, {"intent": "crew"})

    def test_no_object_returns_none(self):
        self.assertIsNone(intent_router._extract_first_json_object("no json here"))
        self.assertIsNone(intent_router._extract_first_json_object(""))

    def test_unmatched_brace_returns_none(self):
        # raw_decode rejects malformed object, scan continues but finds nothing valid.
        self.assertIsNone(intent_router._extract_first_json_object("{not valid}"))

    def test_array_only_not_returned(self):
        # We require a dict; a top-level array doesn't qualify.
        self.assertIsNone(intent_router._extract_first_json_object("[1,2,3]"))


class TestParseL2Json(unittest.TestCase):

    def test_plain_json(self):
        self.assertEqual(intent_router._parse_l2_json('{"intent":"dev"}'), {"intent": "dev"})

    def test_json_in_fenced_block(self):
        self.assertEqual(
            intent_router._parse_l2_json('```json\n{"intent":"crew"}\n```'),
            {"intent": "crew"},
        )

    def test_fenced_without_json_label(self):
        self.assertEqual(
            intent_router._parse_l2_json('```\n{"intent":"plan"}\n```'),
            {"intent": "plan"},
        )

    def test_garbage_around_object(self):
        out = intent_router._parse_l2_json(
            'Here is your result:\n{"intent":"doc","reasoning":"…"} done.')
        self.assertEqual(out, {"intent": "doc", "reasoning": "…"})

    def test_nested_object_not_truncated(self):
        out = intent_router._parse_l2_json(
            'noise {"intent":"dev","meta":{"x":1}} tail')
        self.assertEqual(out, {"intent": "dev", "meta": {"x": 1}})

    def test_empty_returns_none(self):
        self.assertIsNone(intent_router._parse_l2_json(""))
        self.assertIsNone(intent_router._parse_l2_json("   "))

    def test_unparseable_returns_none(self):
        self.assertIsNone(intent_router._parse_l2_json("definitely not json"))


class TestCallCheapBackend:
    """Migrated off the legacy sys.modules-mocking idiom per REQ-09.

    The import-failure branch now uses the conftest ``unload_module`` fixture;
    the call-raises / call-succeeds branches inject a fake callable directly
    via the new ``_backend_call`` kwarg on ``_call_cheap_backend``.
    """

    def test_import_failure_returns_none(self, unload_module):
        """If ``larkhelm.backend_api`` cannot be imported we must return None
        without raising — required for NFR-SEC-02 graceful degradation.
        """
        unload_module("larkhelm.backend_api")
        out = intent_router._call_cheap_backend(MagicMock(), "sys", "text")
        assert out is None

    def test_call_raises_returns_none_and_logs(self):
        def _boom(*a, **kw):
            raise RuntimeError("network unreachable")

        with patch("larkhelm.log._debug_log") as dbg:
            out = intent_router._call_cheap_backend(
                MagicMock(), "sys", "text", _backend_call=_boom,
            )
        assert out is None
        # Logged with the [IntentRouter] prefix (PascalCase per CLAUDE.md
        # logging convention; previously [intent_router]).
        assert any("[IntentRouter]" in str(c.args[0]) for c in dbg.call_args_list), (
            f"expected [IntentRouter] log; got {dbg.call_args_list}"
        )

    def test_call_succeeds_returns_raw(self):
        out = intent_router._call_cheap_backend(
            MagicMock(), "sys", "text",
            _backend_call=lambda *a, **kw: '{"intent":"dev"}',
        )
        assert out == '{"intent":"dev"}'


class TestResolveL2Failures(unittest.TestCase):

    def test_no_cheap_backend_falls_back(self):
        """When no backend is tagged 'cheap', L2 collapses to fallback chat."""
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg:
            reg.get_by_tag.return_value = None
            out = intent_router._resolve_l2("自由文本")
        self.assertEqual(out.agent_type, "chat")
        self.assertEqual(out.layer, "fallback")

    def test_call_failure_falls_back(self):
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch.object(intent_router, "_call_cheap_backend", return_value=None):
            reg.get_by_tag.return_value = MagicMock()
            out = intent_router._resolve_l2("自由文本")
        self.assertEqual(out.agent_type, "chat")
        self.assertEqual(out.layer, "fallback")

    def test_unparseable_response_falls_back(self):
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch.object(intent_router, "_call_cheap_backend", return_value="not json"):
            reg.get_by_tag.return_value = MagicMock()
            out = intent_router._resolve_l2("自由文本")
        self.assertEqual(out.agent_type, "chat")
        self.assertEqual(out.layer, "fallback")

    def test_invalid_intent_coerced_to_chat(self):
        bad_json = '{"intent":"hack_root","complexity":"complex"}'
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch.object(intent_router, "_call_cheap_backend", return_value=bad_json):
            reg.get_by_tag.return_value = MagicMock()
            out = intent_router._resolve_l2("自由文本")
        self.assertEqual(out.agent_type, "chat")
        self.assertEqual(out.layer, "L2")
        self.assertEqual(out.complexity, "complex")

    def test_invalid_complexity_coerced_to_medium(self):
        bad_json = '{"intent":"dev","complexity":"galactic"}'
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch.object(intent_router, "_call_cheap_backend", return_value=bad_json):
            reg.get_by_tag.return_value = MagicMock()
            out = intent_router._resolve_l2("帮我实现一个 X")
        self.assertEqual(out.agent_type, "dev")
        self.assertEqual(out.complexity, "medium")

    def test_resolve_intent_l2_exception_falls_back(self):
        """Top-level resolve_intent wraps L2 in try/except for NFR-SEC-02."""
        with patch.object(intent_router, "_resolve_l2", side_effect=RuntimeError("boom")):
            out = intent_router.resolve_intent("一段不会触发 L1 的中性文本")
        self.assertIsInstance(out, IntentResult)
        self.assertEqual(out.agent_type, "chat")
        self.assertEqual(out.layer, "fallback")


if __name__ == "__main__":
    unittest.main()
