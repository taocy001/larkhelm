"""larkhelm · agent_hub · per-dispatch audit log.

Append-only JSONL at DATA_DIR/agent_audit.jsonl (0600). Used by
``/stats intent`` for daily aggregation (hit rate, latency, cost).
"""
from __future__ import annotations

import datetime
import json
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult
# Centralized helper; previously re-defined locally.
from larkhelm.log import safe_log as _safe_log


def _resolve_path() -> Path:
    """Resolve the JSONL path from config or DATA_DIR.

    Falls back to ``tempfile.gettempdir()`` when ``DATA_DIR`` is unset so the
    audit log never accidentally lands in the cwd.
    """
    import larkhelm.config as _cfg
    cfg = getattr(_cfg, "config", {}) or {}
    custom = cfg.get("intent_audit_path") or ""
    if custom:
        return Path(custom)
    data_dir = getattr(_cfg, "DATA_DIR", None)
    if data_dir is None:
        return Path(tempfile.gettempdir()) / "agent_audit.jsonl"
    return Path(data_dir) / "agent_audit.jsonl"


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        # Re-apply 0600 in case the file pre-existed with broader perms.
        # fchmod/fsync can fail on filesystems that don't support unix perms
        # or durable sync (e.g. CIFS); log and continue rather than abort.
        try:
            os.fchmod(fd, 0o600)
        except OSError as e:
            _safe_log(f"[agent_audit] fchmod 0600 failed for {path}: {e}")
        os.write(fd, (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        try:
            os.fsync(fd)
        except OSError as e:
            _safe_log(f"[agent_audit] fsync failed for {path}: {e}")
    finally:
        os.close(fd)


def write_audit(result: AgentResult, intent: IntentResult, ctx: AgentContext) -> None:
    record = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "chat_id": ctx.chat_id,
        "agent_type": intent.agent_type,
        "backend_id": result.backend_id,
        "duration_sec": result.duration_sec,
        "cost_usd": result.cost_usd,
        "success": result.success,
        "layer": intent.layer,
        "confidence": intent.confidence,
        "trace_id": uuid.uuid4().hex[:8],
    }
    try:
        _append_jsonl(_resolve_path(), record)
    except Exception as e:
        _safe_log(f"[agent_audit] write failed: {e}")


def aggregate_daily(
    date: str | None = None,
    path: Path | None = None,
    chat_id: str | None = None,
) -> dict:
    """Aggregate hit rate / latency / cost for a single date (default today).

    Returns a dict with ``date``, ``total``, ``per_agent``, ``avg_duration``,
    ``total_cost``, ``success_rate``. Returns zeroed stats if log missing.
    Corrupted lines (non-JSON) are skipped silently so a single bad line
    cannot break the whole report.

    Round-4 audit P1 (R4-1d): when ``chat_id`` is provided, restrict the
    aggregate to records whose ``chat_id`` field matches exactly. Without
    this, ``/stats intent`` invoked in chat A leaked aggregate volume /
    avg-duration for chats B, C, ... — a cross-chat side-channel that lets
    an outside observer infer other groups' activity. ``None`` keeps the
    global behaviour (used by CLI tooling / future admin endpoints).
    """
    p = path or _resolve_path()
    if date is None:
        date = datetime.datetime.now().astimezone().date().isoformat()

    per_agent_count: dict[str, int] = defaultdict(int)
    per_agent_success: dict[str, int] = defaultdict(int)
    per_agent_duration: dict[str, float] = defaultdict(float)
    total = 0
    total_success = 0
    total_duration = 0.0
    total_cost = 0.0

    if not p.exists():
        return {
            "date": date, "total": 0, "per_agent": {},
            "avg_duration": 0.0, "total_cost": 0.0, "success_rate": 0.0,
        }

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
                ts = str(rec.get("ts", ""))
                if not ts.startswith(date):
                    continue
                if chat_id is not None and rec.get("chat_id") != chat_id:
                    continue
                agent = str(rec.get("agent_type", "unknown"))
                total += 1
                per_agent_count[agent] += 1
                if rec.get("success"):
                    total_success += 1
                    per_agent_success[agent] += 1
                dur = float(rec.get("duration_sec", 0.0) or 0.0)
                total_duration += dur
                per_agent_duration[agent] += dur
                total_cost += float(rec.get("cost_usd", 0.0) or 0.0)
    except OSError as e:
        _safe_log(f"[agent_audit] aggregate_daily read failed for {p}: {e}")

    per_agent = {
        a: {
            "count": per_agent_count[a],
            "success": per_agent_success[a],
            "avg_duration": (per_agent_duration[a] / per_agent_count[a]) if per_agent_count[a] else 0.0,
        }
        for a in per_agent_count
    }
    return {
        "date": date,
        "total": total,
        "per_agent": per_agent,
        "avg_duration": (total_duration / total) if total else 0.0,
        "total_cost": total_cost,
        "success_rate": (total_success / total) if total else 0.0,
    }


__all__ = ["write_audit", "aggregate_daily"]
