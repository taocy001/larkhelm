#!/usr/bin/env python3
"""Aggregate the memory subsystem's recent operational data into a single
markdown report. Designed to answer: "after the 4-week observation period
following the P0 + milestone + /memory gc cleanup, do we have evidence
that justifies the larger fact-based memory rewrite (P1-1)?"

Inputs (best-effort; missing files just print 0 / "n/a"):
  * ``DEBUG_LOG`` — `[Memory] ...` lines tell us how often the new gates
    fire (rejected summaries, generation timeouts, debounced milestones)
    and how often the auto-update path actually runs.
  * ``DATA_DIR/agent_audit.jsonl`` — Phase 5 dispatch events (intent_router
    has to be turned on for any data to land here; expected to be empty
    while灰度 stays at 0%).
  * ``DATA_DIR/intent_feedback.jsonl`` — explicit user "switch to plain
    chat" overrides; signals classification miss, NOT memory directly,
    but surfaces UX pain that often correlates.
  * ``LOG_DIR/all.jsonl`` — main turn-level event log; we use it to count
    ``role="milestone"`` entries and any ``/memory clear`` / ``/memory set``
    invocations (proxy for "user had to manually fix memory").
  * ``MEMORY_HOME_DIR`` (~/.larkhelm/memory) — file inventory.

Output: a markdown report on stdout. Pipe into ``larkhelm doc create
"Memory Observation Report 2026-06-06"`` if you want to attach it to the
team thread.

Decision rules baked into the report:
  * P0 gate firing >0% but <5% of generations → working as intended
  * P0 gate firing ≥5% → genuine LLM misbehavior, keep gate, no escalation
  * Manual `/memory clear session` rate ≥10% of session-update events →
    auto-memory frequently wrong; consider P1-1 (fact-based)
  * `/memory set` rate ≥5% of session-update events → users routinely
    override; same escalation signal
  * Milestone debounce skip rate ≥30% → /plan with many fast steps; cost
    bound is working but consider raising the debounce window
  * generate_memory timeout rate ≥3% → backend slow path; revisit timeout
    config; doesn't justify P1-1 by itself
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


# ── Pattern matchers (literal strings tied to memory.py / commands.py) ────

PAT_REJECTED  = re.compile(r"\[Memory\] rejected non-useful summary")
PAT_GEN_OK    = re.compile(r"\[Memory\] saved session_")
PAT_GEN_TIMEOUT = re.compile(r"\[Memory\] generate_memory timed out")
PAT_DEBOUNCE  = re.compile(r"\[Memory\] milestone .* debounced for")
PAT_MILESTONE_LOG = re.compile(r"\[Memory\] (?:project|global) layer auto-updated")
PAT_GC_RUN    = re.compile(r"\[Memory\] gc(?:\(apply\)|\(dry-run\))")


def _read_lines(path: Path, since_ts: float) -> list[str]:
    """Stream lines from path; tolerate rotated backup at <path>.1."""
    rows: list[str] = []
    for p in (path.with_name(path.name + ".1"), path):
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    rows.append(line)
        except OSError as e:
            print(f"# warning: read failed for {p}: {e}", file=sys.stderr)
    return rows


def _read_jsonl(path: Path, since_ts: float) -> list[dict]:
    rows: list[dict] = []
    for p in (path.with_name(path.name + ".1"), path):
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("ts", "")
                    if ts:
                        try:
                            t = datetime.fromisoformat(ts).timestamp()
                            if t < since_ts:
                                continue
                        except (TypeError, ValueError):
                            pass
                    rows.append(rec)
        except OSError as e:
            print(f"# warning: read failed for {p}: {e}", file=sys.stderr)
    return rows


def _scan_debug_log(rows: list[str]) -> dict:
    counters = Counter()
    for line in rows:
        if PAT_REJECTED.search(line):     counters["rejected"] += 1
        if PAT_GEN_OK.search(line):       counters["saved"] += 1
        if PAT_GEN_TIMEOUT.search(line):  counters["timeout"] += 1
        if PAT_DEBOUNCE.search(line):     counters["debounced"] += 1
        if PAT_MILESTONE_LOG.search(line):counters["cascade"] += 1
        if PAT_GC_RUN.search(line):       counters["gc_invocations"] += 1
    return dict(counters)


def _scan_jsonl_logs(rows: list[dict]) -> dict:
    out = {
        "total_turns": len(rows),
        "milestone_entries": 0,
        "memory_clear_session": 0,
        "memory_clear_other": 0,
        "memory_set_global": 0,
        "memory_set_project": 0,
        "memory_update_force": 0,
        "memory_gc_invocations": 0,
    }
    for r in rows:
        role = r.get("role", "")
        content = (r.get("content") or "").strip()
        if role == "milestone":
            out["milestone_entries"] += 1
            continue
        # Look for /memory subcommand invocations recorded as user turns.
        if role == "user" and content.startswith("/memory"):
            tail = content[len("/memory"):].strip().lower()
            if tail.startswith("clear session"):
                out["memory_clear_session"] += 1
            elif tail.startswith("clear"):
                out["memory_clear_other"] += 1
            elif tail.startswith("set global"):
                out["memory_set_global"] += 1
            elif tail.startswith("set project"):
                out["memory_set_project"] += 1
            elif tail == "update":
                out["memory_update_force"] += 1
            elif tail.startswith("gc"):
                out["memory_gc_invocations"] += 1
    return out


def _inventory_memory_home(home: Path) -> dict:
    out = {
        "exists": home.exists(),
        "session_count": 0, "project_count": 0, "global_count": 0,
        "session_bytes": 0, "project_bytes": 0, "global_bytes": 0,
        "oldest_project_age_days": None,
    }
    if not home.exists():
        return out
    now = datetime.now().timestamp()
    oldest = None
    for p in home.glob("*.md"):
        try:
            sz = p.stat().st_size
            mtime = p.stat().st_mtime
        except OSError:
            continue
        name = p.name
        if name.startswith("session_"):
            out["session_count"] += 1
            out["session_bytes"] += sz
        elif name.startswith("project_"):
            out["project_count"] += 1
            out["project_bytes"] += sz
            age = (now - mtime) / 86400
            if oldest is None or age > oldest:
                oldest = age
        elif name.startswith("global_"):
            out["global_count"] += 1
            out["global_bytes"] += sz
    out["oldest_project_age_days"] = int(oldest) if oldest is not None else None
    return out


def _percent(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num * 100 / denom:.1f}%"


def _decision(metric: str, value: float, threshold: float, op: str = ">=") -> str:
    """Format a decision tag for the report."""
    triggered = (value >= threshold) if op == ">=" else (value <= threshold)
    return ("🔴 ESCALATE" if triggered else "🟢 OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=28,
                        help="window size in days (default 28 = 4 weeks)")
    parser.add_argument("--debug-log", type=Path,
                        default=Path.home() / ".local/share/larkhelm/larkhelm.log")
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / ".local/share/larkhelm")
    parser.add_argument("--memory-home", type=Path,
                        default=Path.home() / ".larkhelm/memory")
    args = parser.parse_args()

    since_ts = (datetime.now() - timedelta(days=args.days)).timestamp()

    debug_lines = _read_lines(args.debug_log, since_ts)
    audit_rows = _read_jsonl(args.data_dir / "agent_audit.jsonl", since_ts)
    feedback_rows = _read_jsonl(args.data_dir / "intent_feedback.jsonl", since_ts)
    jsonl_rows = _read_jsonl(args.data_dir / "feishu_logs/all.jsonl", since_ts)

    dbg = _scan_debug_log(debug_lines)
    jsonl = _scan_jsonl_logs(jsonl_rows)
    inv = _inventory_memory_home(args.memory_home)

    saved = dbg.get("saved", 0)
    rejected = dbg.get("rejected", 0)
    timeout = dbg.get("timeout", 0)
    debounced = dbg.get("debounced", 0)
    cascade = dbg.get("cascade", 0)
    gen_attempts = saved + rejected + timeout

    rej_rate = (rejected / gen_attempts) if gen_attempts else 0.0
    timeout_rate = (timeout / gen_attempts) if gen_attempts else 0.0
    milestone_total = jsonl["milestone_entries"]
    debounce_rate = (debounced / (debounced + milestone_total)) if (debounced + milestone_total) else 0.0
    clear_rate = (jsonl["memory_clear_session"] / saved) if saved else 0.0
    set_rate = ((jsonl["memory_set_global"] + jsonl["memory_set_project"]) / saved) if saved else 0.0

    print(f"# Memory Observation Report ({args.days} days, generated {datetime.now().isoformat(timespec='seconds')})\n")
    print(f"Window: last **{args.days} days** of activity.\n")

    print("## Memory subsystem firing rates\n")
    print(f"| Metric | Count | Notes |")
    print(f"|---|---|---|")
    print(f"| Successful session memory saves | {saved} | from `[Memory] saved session_*` |")
    print(f"| `_is_useful_summary` rejections | {rejected} | LLM emitted refusal/empty |")
    print(f"| `generate_memory` timeouts | {timeout} | exceeded MEMORY_GENERATION_TIMEOUT |")
    print(f"| Cascade extracts (project+global updates) | {cascade} | |")
    print(f"| Milestone log entries (in JSONL) | {milestone_total} | from session_guard record_milestone |")
    print(f"| Milestone debounce skips | {debounced} | within 60s of previous |")
    print(f"| `/memory gc` invocations | {jsonl['memory_gc_invocations']} | user-explicit cleanup |\n")

    print("## Manual override signals\n")
    print(f"| User action | Count | Per-save % | Notes |")
    print(f"|---|---|---|---|")
    print(f"| `/memory clear session` | {jsonl['memory_clear_session']} | {_percent(jsonl['memory_clear_session'], saved)} | high → auto-memory often wrong |")
    print(f"| `/memory clear` (other layers) | {jsonl['memory_clear_other']} | — | |")
    print(f"| `/memory set global ...` | {jsonl['memory_set_global']} | {_percent(jsonl['memory_set_global'], saved)} | manual override |")
    print(f"| `/memory set project ...` | {jsonl['memory_set_project']} | {_percent(jsonl['memory_set_project'], saved)} | manual override |")
    print(f"| `/memory update` (force) | {jsonl['memory_update_force']} | — | user couldn't wait for auto |")
    print()

    print("## Memory home inventory\n")
    if not inv["exists"]:
        print(f"_Memory home directory `{args.memory_home}` does not exist._\n")
    else:
        print(f"| Layer | Files | Total size | Notes |")
        print(f"|---|---|---|---|")
        print(f"| session | {inv['session_count']} | {inv['session_bytes'] // 1024} KB | |")
        print(f"| project | {inv['project_count']} | {inv['project_bytes'] // 1024} KB | oldest: {inv['oldest_project_age_days']}d |")
        print(f"| global | {inv['global_count']} | {inv['global_bytes'] // 1024} KB | |")
        print()

    print("## Phase 5 router activity (intent_router only fires when 灰度 enabled)\n")
    print(f"- audit rows: **{len(audit_rows)}**")
    print(f"- feedback (force_chat) rows: **{len(feedback_rows)}**")
    print(f"- (Empty is expected as long as `intent_router_enabled: false` in config.json.)")
    print()

    print("## Decision rules (auto-evaluated)\n")
    print(f"| Signal | Value | Threshold | Verdict |")
    print(f"|---|---|---|---|")
    print(f"| P0 gate firing rate | {rej_rate*100:.1f}% | ≥5% | "
          f"{_decision('rej', rej_rate*100, 5)} |")
    print(f"| `generate_memory` timeout rate | {timeout_rate*100:.1f}% | ≥3% | "
          f"{_decision('to', timeout_rate*100, 3)} |")
    print(f"| `/memory clear session` per save | {clear_rate*100:.1f}% | ≥10% | "
          f"{_decision('cls', clear_rate*100, 10)} |")
    print(f"| `/memory set` overrides per save | {set_rate*100:.1f}% | ≥5% | "
          f"{_decision('set', set_rate*100, 5)} |")
    print(f"| Milestone debounce skip rate | {debounce_rate*100:.1f}% | ≥30% | "
          f"{_decision('deb', debounce_rate*100, 30)} |")
    print()

    print("## Interpretation cheat sheet\n")
    print("- All 🟢 → P0 cleanup did its job; **do not** start P1-1 fact-based rewrite.")
    print("- 🔴 on `/memory clear session` or `/memory set` → auto-memory frequently wrong;")
    print("  consider **global-only fact experiment** (≤500 lines, ~5 days) before any larger work.")
    print("- 🔴 on P0 gate firing rate ≥5% → gate is doing real work; keep but DO NOT escalate to P1-1")
    print("  (the gate solved exactly the problem P1-1 would have solved).")
    print("- 🔴 on debounce skip rate → consider raising `_MILESTONE_DEBOUNCE_SEC` to 120-180s.")
    print("- 🔴 on timeout rate → not a memory-architecture problem; investigate cheap-backend latency.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
