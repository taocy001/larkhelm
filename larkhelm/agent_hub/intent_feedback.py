"""larkhelm · agent_hub · intent feedback log + pending registry.

Two responsibilities:

1. ``record_feedback()`` appends a misclassification record to
   ``intent_feedback.jsonl`` (mode 0600), used by ``/stats intent``.
2. ``register_pending()`` / ``resolve_pending()`` keep a small in-memory
   LRU mapping ``feedback_id → (IntentResult, AgentContext)`` so that
   "force_chat" card buttons can re-run with the original context.
"""
from __future__ import annotations

import datetime
import json
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

from larkhelm.agent_hub.intent_types import AgentContext, IntentResult
# Centralized helper; previously re-defined locally.
from larkhelm.log import safe_log as _safe_log


_LRU_CAP = 256


class _PendingEntry(NamedTuple):
    intent: IntentResult
    ctx: AgentContext
    text: str


_pending_lock = threading.Lock()
_pending: "OrderedDict[str, _PendingEntry]" = OrderedDict()


def _new_feedback_id() -> str:
    return f"fb_{uuid.uuid4().hex[:8]}"


def _resolve_path() -> Path:
    """Resolve the JSONL path from config or DATA_DIR.

    Production path: ``DATA_DIR / intent_feedback.jsonl`` (always set by
    ``_init_runtime``).  When ``DATA_DIR`` is unset (single-file invocation,
    early bootstrap, some test harnesses) we land in
    ``tempfile.gettempdir()`` instead of the cwd, so the audit file never
    leaks into the user's working directory by accident.
    """
    import larkhelm.config as _cfg
    cfg = getattr(_cfg, "config", {}) or {}
    custom = cfg.get("intent_feedback_path") or ""
    if custom:
        return Path(custom)
    data_dir = getattr(_cfg, "DATA_DIR", None)
    if data_dir is None:
        return Path(tempfile.gettempdir()) / "intent_feedback.jsonl"
    return Path(data_dir) / "intent_feedback.jsonl"


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_WRONLY | os.O_CREAT
    fd = os.open(str(path), flags, 0o600)
    try:
        # Re-apply 0600 in case the file already existed with broader perms.
        # fchmod can fail on filesystems that don't support unix perms (e.g.
        # CIFS mounts); we log and continue rather than abort the write so a
        # transient mount issue doesn't drop feedback events.
        try:
            os.fchmod(fd, 0o600)
        except OSError as e:
            _safe_log(f"[intent_feedback] fchmod 0600 failed for {path}: {e}")
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        os.write(fd, line)
        try:
            os.fsync(fd)
        except OSError as e:
            _safe_log(f"[intent_feedback] fsync failed for {path}: {e}")
    finally:
        os.close(fd)


def register_pending(feedback_id: str, intent: IntentResult, ctx: AgentContext, text: str = "") -> None:
    """Stash ``(intent, ctx)`` for later force_chat retrieval. LRU cap = 256."""
    with _pending_lock:
        if feedback_id in _pending:
            _pending.move_to_end(feedback_id)
        _pending[feedback_id] = _PendingEntry(intent=intent, ctx=ctx, text=text or intent.raw_text)
        while len(_pending) > _LRU_CAP:
            _pending.popitem(last=False)


def resolve_pending(feedback_id: str) -> _PendingEntry | None:
    with _pending_lock:
        entry = _pending.pop(feedback_id, None)
    return entry


def record_feedback(
    predicted: IntentResult,
    corrected: str,
    chat_id: str,
    feedback_id: str | None = None,
    text: str = "",
) -> str:
    """Append-only JSONL with the misclassification. Returns the feedback_id."""
    fid = feedback_id or _new_feedback_id()
    record = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "chat_id": chat_id,
        "text": text or predicted.raw_text,
        "predicted_intent": predicted.agent_type,
        "corrected_intent": corrected,
        "confidence": predicted.confidence,
        "layer": predicted.layer,
        "feedback_id": fid,
    }
    try:
        _append_jsonl(_resolve_path(), record)
    except Exception as e:
        _safe_log(f"[intent_feedback] write failed: {e}")
    return fid


def list_recent(n: int = 20, path: Path | None = None) -> list[dict]:
    p = path or _resolve_path()
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate partial / truncated lines so a corrupted tail
                # cannot break /stats intent reads.
                continue
    except OSError as e:
        _safe_log(f"[intent_feedback] list_recent read failed for {p}: {e}")
    return out


__all__ = [
    "record_feedback", "register_pending", "resolve_pending", "list_recent",
]
