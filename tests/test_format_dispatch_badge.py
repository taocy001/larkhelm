"""P1-5b: pure-function unit tests for ``format_dispatch_badge``.

Table-driven coverage of the 5 layer branches the PRD pins (L1 /
explicit-as-L1 / L2 / fallback / microlearn) plus one unknown-future-value
case to lock the fail-open default. AC-01 also pins the public signature
via ``__annotations__`` reflection so a future drift in the type hints
(e.g. dropping ``IntentResult``) trips the test before it ships.
"""
import unittest

from larkhelm.agent_hub.intent_router import format_dispatch_badge
from larkhelm.agent_hub.intent_types import IntentResult


class TestFormatDispatchBadge(unittest.TestCase):

    def test_l1_returns_empty(self):
        intent = IntentResult(agent_type="dev", layer="L1", confidence=0.85)
        self.assertEqual(format_dispatch_badge(intent), "")

    def test_l1_explicit_command_returns_empty(self):
        # Explicit slash commands still set layer="L1"; the badge must
        # not leak through on this path either (PRD AC-02 D2).
        intent = IntentResult(
            agent_type="dev",
            layer="L1",
            confidence=1.0,
            is_explicit_command=True,
        )
        self.assertEqual(format_dispatch_badge(intent), "")

    def test_l2_returns_badge(self):
        intent = IntentResult(agent_type="dev", layer="L2", confidence=0.7)
        self.assertEqual(format_dispatch_badge(intent), "(L2)")

    def test_fallback_returns_l2_to_chat_arrow(self):
        intent = IntentResult(agent_type="chat", layer="fallback", confidence=0.0)
        self.assertEqual(format_dispatch_badge(intent), "(L2→chat)")

    def test_microlearn_returns_empty(self):
        intent = IntentResult(
            agent_type="dev", layer="microlearn", confidence=0.72,
        )
        self.assertEqual(format_dispatch_badge(intent), "")

    def test_unknown_layer_returns_empty(self):
        # Both today's "override" path and any future / typo'd value must
        # collapse to empty (fail-open per D3).
        self.assertEqual(
            format_dispatch_badge(IntentResult(agent_type="chat", layer="override")),
            "",
        )
        self.assertEqual(
            format_dispatch_badge(
                IntentResult(agent_type="chat", layer="unknown_future_value"),
            ),
            "",
        )

    def test_signature_annotations(self):
        # AC-01: the public function must keep its declared signature
        # (IntentResult in, str out). The module uses ``from __future__
        # import annotations`` so the raw ``__annotations__`` dict holds
        # forward-ref strings; ``typing.get_type_hints`` resolves them
        # against the module globals so we can identity-check the real
        # types rather than string-comparing names.
        import typing
        hints = typing.get_type_hints(format_dispatch_badge)
        self.assertIs(hints.get("intent"), IntentResult)
        self.assertIs(hints.get("return"), str)


if __name__ == "__main__":
    unittest.main()
