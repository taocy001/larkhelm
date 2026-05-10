"""CI gate for intent-router accuracy.

The Phase 5 PRD AC-13 mandates intent classification accuracy ≥ 0.85 on the
labeled fixture (``tests/fixtures/intent_labels.jsonl``). Before this file the
threshold was only enforced via ``scripts/eval_intent_accuracy.py`` which
nobody ran in CI; this test wires the same logic into ``unittest discover`` so
the bar can never silently regress.

Implementation: re-uses ``resolve_intent`` directly against the fixture and
asserts accuracy ≥ ``CI_THRESHOLD``. Marked ``CI_THRESHOLD`` slightly below
the production target (0.85) to absorb sampling noise from a 50-row fixture
while still catching meaningful regressions.
"""
from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from larkhelm.agent_hub.intent_router import resolve_intent


# Production target is 0.85 on a 1000-row dataset (see PRD AC-13).
# The CI fixture is a 50-row subset; use the same threshold so a regression
# below 0.85 here is definitely a real regression.
CI_THRESHOLD = 0.85
FIXTURE = Path(__file__).parent / "fixtures" / "intent_labels.jsonl"


def _load_fixture() -> list[dict]:
    samples: list[dict] = []
    with FIXTURE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


class TestIntentAccuracyCIGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.samples = _load_fixture()
        if not cls.samples:
            raise unittest.SkipTest("intent_labels.jsonl is empty")

    def test_accuracy_meets_threshold(self):
        correct = 0
        confusion: Counter[tuple[str, str]] = Counter()
        for s in self.samples:
            text = s["text"]
            expected = s["expected_agent_type"]
            predicted = resolve_intent(text).agent_type
            confusion[(expected, predicted)] += 1
            if predicted == expected:
                correct += 1
        total = len(self.samples)
        accuracy = correct / total

        self.assertGreaterEqual(
            accuracy, CI_THRESHOLD,
            msg=(
                f"Intent accuracy regressed: {accuracy:.4f} < {CI_THRESHOLD:.2f}"
                f" on {total} fixture rows.\n"
                f"Confusion (expected → predicted): {dict(confusion)}"
            ),
        )

    def test_fixture_covers_all_agent_types(self):
        """Sanity guard: the fixture must cover every builtin agent_type so
        regressions in any one branch register as accuracy drops."""
        seen = {s["expected_agent_type"] for s in self.samples}
        for required in ("chat", "dev", "crew", "plan", "doc"):
            self.assertIn(required, seen,
                          f"fixture missing rows for {required!r} (regression on this branch goes undetected)")


if __name__ == "__main__":
    unittest.main()
