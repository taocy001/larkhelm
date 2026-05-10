#!/usr/bin/env python3
"""AC-14: benchmark resolve_intent latency.

Reports overall P50/P95/P99 plus a separate breakdown for entries that hit
the L1 fast path.

Usage::

    python3 scripts/bench_intent_latency.py [--n 500] [--path tests/fixtures/intent_labels.jsonl]
"""
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
import time
from pathlib import Path
from statistics import quantiles


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    samples = sorted(samples)
    k = max(0, min(len(samples) - 1, int(round(pct / 100.0 * (len(samples) - 1)))))
    return samples[k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--path", default="tests/fixtures/intent_labels.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"fixture not found: {path}", file=sys.stderr)
        return 2

    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            texts.append(json.loads(line)["text"])

    if not texts:
        print("no samples", file=sys.stderr)
        return 2

    from larkhelm.agent_hub.intent_router import resolve_intent

    all_lat: list[float] = []
    l1_lat: list[float] = []
    other_lat: list[float] = []
    for i in range(args.n):
        text = texts[i % len(texts)]
        t0 = time.monotonic()
        intent = resolve_intent(text)
        dt = (time.monotonic() - t0) * 1000.0
        all_lat.append(dt)
        if intent.layer == "L1":
            l1_lat.append(dt)
        else:
            other_lat.append(dt)

    def _report(label: str, samples: list[float]) -> None:
        if not samples:
            print(f"{label}: no samples")
            return
        print(
            f"{label}: n={len(samples)}, "
            f"P50={_percentile(samples, 50):.2f}ms, "
            f"P95={_percentile(samples, 95):.2f}ms, "
            f"P99={_percentile(samples, 99):.2f}ms, "
            f"avg={sum(samples)/len(samples):.2f}ms"
        )

    _report("all", all_lat)
    _report("L1 only", l1_lat)
    _report("non-L1", other_lat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
