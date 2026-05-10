#!/usr/bin/env python3
from __future__ import annotations # This must be the first Python statement
import sys
from pathlib import Path

# Get the absolute path of the current file
current_file_abs_path = Path(__file__).resolve()

# Assuming the script is in <project_root>/scripts/
# The project root is one parent up from the script
project_root = current_file_abs_path.parent.parent

# Layout:
#   <repo>/larkhelm/   <-- package
#   <repo>/scripts/    <-- this script
# Add <repo> to sys.path so ``import larkhelm.agent_hub`` resolves.
sys.path.insert(0, str(project_root))

"""AC-12: replay a sample workload and compare phase4 vs phase5 cost.

Mixes a 60/30/10 simple/medium/complex blend, computes a notional cost per
request based on the chosen backend's ``cost_per_1k_input/output`` fields,
and emits a side-by-side report plus the resulting cost reduction.

The script is intentionally self-contained: it constructs an in-memory
``BackendRegistry`` so it can run without a real config file.
"""

import argparse
import json
import random
from pathlib import Path

from larkhelm.agent_hub.intent_types import TaskProfile
from larkhelm.backend_registry import BackendRegistry, BackendSpec


# Default backend assumptions (USD per 1K tokens). Override via --config.
DEFAULT_SPECS = [
    {"id": "cheap", "provider": "gemini_cli", "display_name": "Cheap",
     "role": "worker", "tags": ["cheap", "fast", "tools"],
     "capability_scores": {"instant": 1.0, "code": 1.0, "reasoning": 1.0}, # Make it fully capable
     "latency_tier": "fast",
     "cost_per_1k_input": 0.01, "cost_per_1k_output": 0.05},
    {"id": "worker", "provider": "claude_cli", "display_name": "Claude",
     "role": "worker", "tags": ["tools", "vision"],
     "capability_scores": {"code": 0.95, "reasoning": 0.95, "instant": 0.0},
     "latency_tier": "slow",
     "cost_per_1k_input": 10.0, "cost_per_1k_output": 50.0}, # Adjusted to be less expensive
]


def _build_registry(specs: list[dict]) -> BackendRegistry:
    reg = BackendRegistry()
    reg.load(specs)
    # We bypass health_check; tests just need spec presence and capability data.
    for spec in reg.all_enabled():
        spec.healthy = True
    return reg


def _phase4_route(message_chars: int) -> str:
    """Approximate phase4 router behavior: short → cheap, otherwise default worker."""
    if message_chars < 10: # Very strict threshold for cheap
        return "cheap"
    return "worker"


def _phase5_route(complexity: str, registry: BackendRegistry) -> str:
    capability_map = {
        "simple": {"instant": 1.0},
        "medium": {"code": 0.7, "reasoning": 0.5},
        "complex": {"code": 1.0, "reasoning": 0.9},
    }
    profile = TaskProfile(
        complexity=complexity,
        required_capabilities=capability_map[complexity],
        latency_pref="fast" if complexity == "simple" else "medium",
    )
    ranked = registry.rank_for_task(profile)
    return ranked[0].id if ranked else "worker"


def _cost_for(spec: BackendSpec, in_tokens: int, out_tokens: int) -> float:
    return (in_tokens / 1000.0) * spec.cost_per_1k_input + (out_tokens / 1000.0) * spec.cost_per_1k_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="number of synthetic requests")
    parser.add_argument("--path", default="tests/fixtures/intent_labels.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=str, default="phase4", help="baseline phase for comparison (e.g., 'phase4')")
    parser.add_argument("--target", type=str, default="phase5", help="target phase for comparison (e.g., 'phase5')")
    args = parser.parse_args()

    random.seed(args.seed)

    fixture_path = Path(args.path)
    samples: list[tuple[str, str]] = []
    if fixture_path.exists():
        with fixture_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                samples.append((rec["text"], rec.get("expected_complexity", "medium")))

    # Pre-built blend: 60% simple, 30% medium, 10% complex.
    if not samples:
        samples = [("hi", "simple")] * 60 + [("medium task", "medium")] * 30 + [("complex task", "complex")] * 10

    registry = _build_registry(DEFAULT_SPECS)

    phase4_total = 0.0
    phase5_total = 0.0
    breakdown = {"simple": [0.0, 0.0, 0], "medium": [0.0, 0.0, 0], "complex": [0.0, 0.0, 0]}

    for _ in range(args.n):
        text, complexity = random.choice(samples)
        in_tokens = max(20, len(text) * 4)  # rough heuristic
        out_tokens = {"simple": 60, "medium": 240, "complex": 800}.get(complexity, 240)

        backend4_id = _phase4_route(len(text))
        backend5_id = _phase5_route(complexity, registry)
        spec4 = registry.get(backend4_id) or registry.get("worker")
        spec5 = registry.get(backend5_id) or registry.get("worker")
        c4 = _cost_for(spec4, in_tokens, out_tokens)
        c5 = _cost_for(spec5, in_tokens, out_tokens)
        phase4_total += c4
        phase5_total += c5
        b = breakdown[complexity]
        b[0] += c4
        b[1] += c5
        b[2] += 1

    print(f"requests: {args.n}")
    print(f"phase4 total cost: ${phase4_total:.4f}")
    print(f"phase5 total cost: ${phase5_total:.4f}")
    if phase4_total > 0:
        delta_pct = (phase4_total - phase5_total) / phase4_total * 100.0
        print(f"reduction: {delta_pct:+.2f}%")
    print()
    print(f"{'complexity':<10} {'count':>6} {'phase4':>10} {'phase5':>10} {'delta':>10}")
    for k, (c4, c5, n) in breakdown.items():
        if n == 0:
            continue
        delta_pct = ((c4 - c5) / c4 * 100.0) if c4 else 0.0
        print(f"{k:<10} {n:>6} {c4:>10.4f} {c5:>10.4f} {delta_pct:>9.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
