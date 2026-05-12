#!/usr/bin/env python3
"""Offline benchmark: measure ``_get_recent_turns`` byte savings on real logs.

Walks ``LOG_DIR/all.jsonl`` (and the rotated ``.jsonl.1`` backup if present),
groups records by ``chat_id``, replays each chat's last user/assistant turns
through ``log._prune_content`` and compares the before/after byte budget to
the orchestrator-context cap. Designed to validate PRD G2 (30–50% savings)
on production-shaped data without flipping a flag or rerunning the bridge.

Run before merging the pruning change:

    python3 scripts/measure_pruning_savings.py --data-dir ~/.local/share/larkhelm
    python3 scripts/measure_pruning_savings.py --chat-id oc_abcd1234

Does NOT mutate any file; not wired into CI.

Note: ``_byte_len`` / ``_stringify`` below are simplified copies of
``larkhelm.log`` helpers — they omit the bytes/bytearray branch present in the
runtime version. Production ``content`` is effectively always str / list /
dict, so the measurement deviation versus the runtime is ≤ 1%; the duplication
is intentional to keep this script free of bridge imports (avoids WebSocket
SDK side effects).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_data_dir(explicit: str | None) -> Path:
    """Mirror ``larkhelm.config`` resolution order without importing the
    full bridge (which would load the WebSocket SDK as a side-effect)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("LARKHELM_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    system = Path("/var/lib/larkhelm")
    if system.exists():
        return system
    return Path.home() / ".local" / "share" / "larkhelm"


def _stringify(value):
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _byte_len(value) -> int:
    if isinstance(value, str):
        try:
            return len(value.encode("utf-8"))
        except Exception:
            return 0
    try:
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _load_records(log_dir: Path, chat_id: str | None) -> dict[str, list[dict]]:
    """Return {chat_id: [records...]} from all.jsonl (+ .jsonl.1)."""
    by_chat: dict[str, list[dict]] = {}
    for fname in ("all.jsonl.1", "all.jsonl"):
        p = log_dir / fname
        if not p.exists():
            continue
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("role") not in ("user", "assistant"):
                        continue
                    cid = r.get("chat_id") or ""
                    if chat_id and cid != chat_id:
                        continue
                    by_chat.setdefault(cid, []).append(r)
        except Exception as e:
            print(f"# warn: failed to read {p}: {e}", file=sys.stderr)
    return by_chat


def _measure_chat(records: list[dict], prune_fn) -> tuple[int, int, int]:
    """Returns (before_bytes, after_bytes, blocks_pruned) for the last 12
    user/assistant turns (matching ``_get_recent_turns`` default of
    max_turns=6 ⇒ slice -12:)."""
    tail = records[-12:]
    before_sum = 0
    after_sum = 0
    blocks_pruned = 0
    for r in tail:
        raw = r.get("content", "")
        bytes_before = _byte_len(raw)
        pruned = prune_fn(raw)
        display = _stringify(pruned)
        bytes_after = len(display.encode("utf-8")) if display else 0
        before_sum += bytes_before
        after_sum += bytes_after
        if bytes_after < bytes_before:
            blocks_pruned += display.count("[tool_result truncated —")
    return before_sum, after_sum, blocks_pruned


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure pruning savings on real all.jsonl",
    )
    parser.add_argument("--data-dir", default=None,
                        help="larkhelm DATA_DIR (default: env LARKHELM_DATA_DIR "
                             "or ~/.local/share/larkhelm)")
    parser.add_argument("--chat-id", default=None,
                        help="Restrict to a single chat_id")
    parser.add_argument("--min-bytes", type=int, default=200,
                        help="Skip chats whose before-sum < this (default 200)")
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    log_dir = data_dir / ".feishu_logs"
    if not log_dir.exists():
        print(f"# error: log dir not found: {log_dir}", file=sys.stderr)
        return 2

    # Import lazily so a missing larkhelm install only fails after argparse.
    try:
        from larkhelm.log import _prune_content
    except Exception as e:
        print(f"# error: cannot import larkhelm.log: {e}", file=sys.stderr)
        return 2

    by_chat = _load_records(log_dir, args.chat_id)
    if not by_chat:
        print("# no matching records", file=sys.stderr)
        return 1

    rows: list[tuple[str, int, int, int]] = []
    grand_before = 0
    grand_after = 0
    grand_blocks = 0
    for cid, recs in by_chat.items():
        before, after, blocks = _measure_chat(recs, _prune_content)
        if before < args.min_bytes:
            continue
        rows.append((cid, before, after, blocks))
        grand_before += before
        grand_after += after
        grand_blocks += blocks

    rows.sort(key=lambda x: -(x[1] - x[2]))  # most-saved chats first

    print("| chat_id (head) | before | after | saved | saved% | blocks |")
    print("|---|---:|---:|---:|---:|---:|")
    for cid, before, after, blocks in rows:
        saved = before - after
        pct = (saved * 100 // before) if before > 0 else 0
        head = (cid or "<empty>")[:18]
        print(f"| `{head}` | {before} | {after} | {saved} | {pct}% | {blocks} |")

    saved = grand_before - grand_after
    pct = (saved * 100 // grand_before) if grand_before > 0 else 0
    print()
    print(f"**Total**: before={grand_before} bytes · after={grand_after} bytes "
          f"· saved={saved} bytes (**{pct}%**) · blocks={grand_blocks} "
          f"· chats={len(rows)}")

    target_lo, target_hi = 30, 50
    if pct < target_lo:
        print(f"\n⚠️  Saved% {pct}% < PRD G2 target {target_lo}–{target_hi}%."
              " Consider lowering _TOOL_RESULT_THRESHOLD to 300.")
    elif pct > target_hi:
        print(f"\nℹ️  Saved% {pct}% > PRD G2 target {target_lo}–{target_hi}%."
              " Workload is tool-heavy; threshold can stay at 500.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
