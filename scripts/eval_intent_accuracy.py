#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

# Get the absolute path of the current file
current_file_abs_path = Path(__file__).resolve()

# Assuming the script is in <project_root>/scripts/
# The project root is one parent up from the script
project_root = current_file_abs_path.parent.parent

sys.path.insert(0, str(project_root))

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/fixtures/intent_labels.jsonl")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"fixture not found: {path}", file=sys.stderr)
        return 2

    from larkhelm.agent_hub.intent_router import resolve_intent

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if not samples:
        print("no samples", file=sys.stderr)
        return 2

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    correct = 0
    for s in samples:
        text = s["text"]
        expected = s["expected_agent_type"]
        intent = resolve_intent(text)
        predicted = intent.agent_type
        confusion[(expected, predicted)] += 1
        if predicted == expected:
            correct += 1

    total = len(samples)
    accuracy = correct / total

    print(f"Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Confusion matrix (rows=expected, cols=predicted):")
    agents = sorted({a for pair in confusion for a in pair})
    header = "         " + " ".join(f"{a:>8}" for a in agents)
    print(header)
    for exp in agents:
        row = [f"{confusion[(exp, pred)]:>8}" for pred in agents]
        print(f"{exp:>8} " + " ".join(row))

    if accuracy < args.threshold:
        print(f"\nFAIL: accuracy {accuracy:.4f} < threshold {args.threshold:.2f}")
        return 1
    print(f"\nPASS: accuracy {accuracy:.4f} ≥ threshold {args.threshold:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
