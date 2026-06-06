import unittest
import dataclasses
from unittest.mock import patch, MagicMock

from larkhelm.backend_registry import BackendSpec, BackendRegistry, _normalize_str_list
from larkhelm.log import _debug_log

class TestBackendRegistryMeta(unittest.TestCase):

    def setUp(self):
        # Reset the singleton instance for a clean slate before each test
        BackendRegistry._instance = None
        BackendRegistry._lock = None

    # AC-01 / AC-02 / AC-03 / AC-08
    def test_backend_spec_new_fields_and_defaults(self):
        spec = BackendSpec(
            id="test-id",
            provider="test-provider",
            display_name="Test Display Name",
            role="orchestrator",
            tags=[],
        )
        self.assertTrue(hasattr(spec, 'description'))
        self.assertTrue(hasattr(spec, 'trigger_phrases'))
        self.assertTrue(hasattr(spec, 'intent_examples'))

        self.assertEqual(spec.description, "")
        self.assertEqual(spec.trigger_phrases, [])
        self.assertEqual(spec.intent_examples, [])

        # AC-08: dataclasses.asdict includes new fields
        as_dict = dataclasses.asdict(spec)
        self.assertIn('description', as_dict)
        self.assertIn('trigger_phrases', as_dict)
        self.assertIn('intent_examples', as_dict)

        # AC-02: trigger_phrases and intent_examples are independent lists
        spec1 = BackendSpec(id="1", provider="p", display_name="d", role="r", tags=[])
        spec2 = BackendSpec(id="2", provider="p", display_name="d", role="r", tags=[])
        self.assertIsNot(spec1.trigger_phrases, spec2.trigger_phrases)
        self.assertIsNot(spec1.intent_examples, spec2.intent_examples)

    # _normalize_str_list tests
    def test_normalize_str_list_none_or_missing(self):
        self.assertEqual(_normalize_str_list(None), [])
        self.assertEqual(_normalize_str_list(""), []) # Empty string should also result in empty list after stripping and filtering
        self.assertEqual(_normalize_str_list("   "), [])

    def test_normalize_str_list_from_list(self):
        self.assertEqual(_normalize_str_list(["bug", " fix ", "", "feature"]), ["bug", "fix", "feature"])
        self.assertEqual(_normalize_str_list(["  "]), [])
        self.assertEqual(_normalize_str_list([]), [])

    # AC-06: trigger_phrases string normalization (split by comma, strip, drop empty)
    def test_normalize_str_list_from_string_comma(self):
        self.assertEqual(_normalize_str_list("bug, fix,, feature "), ["bug", "fix", "feature"])
        self.assertEqual(_normalize_str_list("  one,two , three  "), ["one", "two", "three"])
        self.assertEqual(_normalize_str_list("a"), ["a"])

    # PRD §1.1: multi-line string normalization (split by newline, then comma)
    def test_normalize_str_list_from_string_multi_line(self):
        self.assertEqual(_normalize_str_list("""a
b,c
  d , e """), ["a", "b", "c", "d", "e"])
        self.assertEqual(_normalize_str_list("""first
second, third

last"""), ["first", "second", "third", "last"])

    # PRD §3 P0: invalid type fallback to [] and not raise exception
    def test_normalize_str_list_invalid_type_fallback(self):
        with patch('larkhelm.backend_registry._debug_log') as mock_debug_log:
            self.assertEqual(_normalize_str_list({"key": "value"}), [])
            mock_debug_log.assert_called_once()
            mock_debug_log.reset_mock()
            self.assertEqual(_normalize_str_list(123), [])
            mock_debug_log.assert_called_once()

    # AC-04: BackendRegistry.load() with old config (no new fields)
    def test_load_old_config(self):
        old_specs = [
            {
                "id": "old-gemini",
                "provider": "gemini_cli",
                "display_name": "Old Gemini",
                "role": "orchestrator",
                "tags": ["tools"],
            }
        ]
        registry = BackendRegistry()
        registry.load(old_specs)

        spec = registry.get("old-gemini")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.description, "")
        self.assertEqual(spec.trigger_phrases, [])
        self.assertEqual(spec.intent_examples, [])

    # AC-05: BackendRegistry.load() with new config (explicitly declared fields)
    def test_load_new_config(self):
        new_specs = [
            {
                "id": "new-claude",
                "provider": "claude_cli",
                "display_name": "New Claude",
                "role": "worker",
                "tags": ["vision"],
                "description": "A new Claude model.",
                "trigger_phrases": ["imagine", "draw"],
                "intent_examples": ["draw a cat", "imagine a dog"],
            },
            {
                "id": "new-gemini",
                "provider": "gemini_cli",
                "display_name": "New Gemini",
                "role": "orchestrator",
                "tags": ["tools"],
                "description": "Another new Gemini model.",
                "trigger_phrases": "search,find,query",
                "intent_examples": "search for a cat, find a dog",
            },
            {
                "id": "malformed-spec",
                "provider": "test_cli",
                "display_name": "Malformed Spec",
                "role": "worker",
                "tags": [],
                "description": 123, # invalid type
                "trigger_phrases": {"invalid": "type"}, # invalid type
                "intent_examples": None # None type
            }
        ]
        registry = BackendRegistry()
        registry.load(new_specs)

        # Test new-claude
        spec_claude = registry.get("new-claude")
        self.assertIsNotNone(spec_claude)
        self.assertEqual(spec_claude.description, "A new Claude model.")
        self.assertEqual(spec_claude.trigger_phrases, ["imagine", "draw"])
        self.assertEqual(spec_claude.intent_examples, ["draw a cat", "imagine a dog"])

        # Test new-gemini
        spec_gemini = registry.get("new-gemini")
        self.assertIsNotNone(spec_gemini)
        self.assertEqual(spec_gemini.description, "Another new Gemini model.")
        self.assertEqual(spec_gemini.trigger_phrases, ["search", "find", "query"])
        self.assertEqual(spec_gemini.intent_examples, ["search for a cat", "find a dog"])

        # Test malformed-spec
        spec_malformed = registry.get("malformed-spec")
        self.assertIsNotNone(spec_malformed)
        self.assertEqual(spec_malformed.description, "123") # description is force-casted to str
        self.assertEqual(spec_malformed.trigger_phrases, []) # invalid type, normalized to empty list
        self.assertEqual(spec_malformed.intent_examples, []) # None type, normalized to empty list

    def test_load_description_not_stripped(self):
        new_specs = [
            {
                "id": "desc-test",
                "provider": "test",
                "display_name": "Desc Test",
                "role": "worker",
                "tags": [],
                "description": """  A description with leading and trailing spaces and
multiple lines.  """,
            }
        ]
        registry = BackendRegistry()
        registry.load(new_specs)
        spec = registry.get("desc-test")
        self.assertEqual(spec.description, """  A description with leading and trailing spaces and
multiple lines.  """)


class TestBackendSpecFields(unittest.TestCase):
    """AC-06 / AC-07: capability/cost/latency_tier defaults on BackendSpec."""

    def test_new_fields_defaults(self):
        spec = BackendSpec(id="t", provider="p", display_name="t", role="worker", tags=[])
        self.assertEqual(spec.capability_scores, {})
        self.assertEqual(spec.cost_per_1k_input, 0.0)
        self.assertEqual(spec.cost_per_1k_output, 0.0)
        self.assertEqual(spec.latency_tier, "medium")

    def test_load_capability_scores(self):
        registry = BackendRegistry()
        registry.load([
            {
                "id": "scored", "provider": "claude_cli", "display_name": "Scored",
                "role": "worker", "tags": ["tools"],
                "capability_scores": {"code": 0.9, "reasoning": 0.7},
                "cost_per_1k_input": 3.0, "cost_per_1k_output": 15.0,
                "latency_tier": "slow",
            },
        ])
        spec = registry.get("scored")
        self.assertEqual(spec.capability_scores, {"code": 0.9, "reasoning": 0.7})
        self.assertEqual(spec.cost_per_1k_input, 3.0)
        self.assertEqual(spec.cost_per_1k_output, 15.0)
        self.assertEqual(spec.latency_tier, "slow")

    def test_load_invalid_latency_falls_back(self):
        registry = BackendRegistry()
        registry.load([
            {"id": "b", "provider": "p", "display_name": "B", "role": "worker", "tags": [],
             "latency_tier": "warp-speed"},
        ])
        self.assertEqual(registry.get("b").latency_tier, "medium")

    def test_get_by_tag_unchanged_phase4(self):
        registry = BackendRegistry()
        registry.load([
            {"id": "cheap-id", "provider": "gemini_cli", "display_name": "Cheap",
             "role": "worker", "tags": ["cheap", "fast"]},
        ])
        spec = registry.get_by_tag(["cheap"])
        self.assertIsNotNone(spec)
        self.assertEqual(spec.id, "cheap-id")


if __name__ == '__main__':
    unittest.main()
