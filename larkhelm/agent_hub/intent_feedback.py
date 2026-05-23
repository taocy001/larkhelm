"""larkhelm · agent_hub · intent feedback log + pending registry.

Three responsibilities:

1. ``record_feedback()`` appends a record to ``intent_feedback.jsonl``
   (mode 0600), used by ``/stats intent`` and the offline L1-keyword
   trainer (``scripts/train_intent_classifier.py``).
2. ``register_pending()`` / ``resolve_pending()`` keep a small in-memory
   LRU mapping ``feedback_id → (IntentResult, AgentContext)`` so that
   "force_chat" card buttons can re-run with the original context.
3. ``track_dispatch()`` / ``consume_dispatch()`` keep a per-chat
   recent-dispatch registry so the message router can attribute a
   subsequent ``/cancel`` or model-switch slash command back to the most
   recent dispatched intent. Powers the ``cancel_after_dispatch`` and
   ``agent_reswitch`` extended signals (Phase D follow-up, May 2026).

Schema extension (back-compat): every row gains an optional
``signal_type`` field. Pre-Phase-D rows lack it (treat them as
``force_chat``). Observational signals (``l1_gray_zone`` /
``l2_dispatched``) carry an empty ``corrected_intent`` — the trainer's
``_load_feedback`` already requires a non-empty label so they're skipped
there, which is the desired no-op.
"""
from __future__ import annotations

import datetime
import json
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

from larkhelm.agent_hub.intent_types import AgentContext, IntentResult
# Centralized helper; previously re-defined locally.
from larkhelm.log import safe_log as _safe_log


_LRU_CAP = 256

# Cap dispatch-history dict to keep memory bounded under a flood of
# unique chats. 512 keeps ~3-4× the typical chat-fanout headroom
# without ever crossing 100 KB of live state.
_DISPATCH_HISTORY_MAX = 512
# Hard ceiling for entry lifetime regardless of signal-window config —
# entries older than this are GC'd on the next track_dispatch() call so
# a chat that goes silent doesn't keep stale state forever.
_DISPATCH_HISTORY_HARD_TTL_SEC = 900.0  # 15 min


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


# ── Extended-signal config gate ────────────────────────────────────────


def _extended_signals_enabled() -> bool:
    """Master switch for the Phase-D follow-up signals.

    Defaults to True. Flip ``intent_feedback_extended_signals=false`` in
    config.json to roll back to the legacy force_chat-only behaviour
    without restarting the bridge.
    """
    try:
        import larkhelm.config as _cfg
        return bool(getattr(_cfg, "INTENT_FEEDBACK_EXTENDED_SIGNALS", True))
    except Exception:
        return True


def _signal_text_cap() -> int:
    """Max chars of ``text`` to persist on observational/inferred signal
    rows. Force_chat rows keep their full text for byte-compat with the
    pre-Phase-D records (the trainer already filters >2000 chars).
    """
    try:
        import larkhelm.config as _cfg
        cap = int(getattr(_cfg, "INTENT_FEEDBACK_SIGNAL_TEXT_MAX", 800) or 800)
    except Exception:
        cap = 800
    return max(0, cap)


def _cancel_window_sec() -> float:
    """Seconds within which a ``/cancel`` is attributed to the prior
    dispatch as a ``cancel_after_dispatch`` signal."""
    try:
        import larkhelm.config as _cfg
        return max(0.0, float(getattr(_cfg, "INTENT_FEEDBACK_CANCEL_WINDOW_SEC", 60.0) or 60.0))
    except Exception:
        return 60.0


def _truncate_text(text: str, cap: int | None = None) -> str:
    if not text:
        return ""
    limit = _signal_text_cap() if cap is None else cap
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "…"


def _bump_metric(signal_type: str) -> None:
    """Fire-and-forget metric increment. Never raises."""
    try:
        from larkhelm.metrics import inc_intent_feedback
        inc_intent_feedback(signal_type)
    except Exception as e:
        _safe_log(f"[IntentFeedback] metric inc failed (signal={signal_type}): {e}")


# ── Dispatch-history registry (for cancel / reswitch signal attribution) ──


class _DispatchEntry(NamedTuple):
    intent:   IntentResult
    chat_id:  str
    text:     str
    mono_ts:  float


_dispatch_history_lock = threading.Lock()
_dispatch_history: "OrderedDict[str, _DispatchEntry]" = OrderedDict()


def track_dispatch(chat_id: str, intent: IntentResult, text: str = "") -> None:
    """Stamp ``chat_id`` as having just dispatched ``intent``.

    Called from :class:`AgentDispatcher` after the ACL check passes and
    before the executor runs. Cheap (one lock acquisition + dict
    insertion); no I/O. Disabling extended signals turns this into a
    no-op so the legacy code path stays byte-identical.
    """
    if not _extended_signals_enabled():
        return
    now = time.monotonic()
    with _dispatch_history_lock:
        _dispatch_history[chat_id] = _DispatchEntry(
            intent=intent, chat_id=chat_id,
            text=text or intent.raw_text or "",
            mono_ts=now,
        )
        _dispatch_history.move_to_end(chat_id)
        # GC: drop stale entries first, then bound the dict size.
        if len(_dispatch_history) > _DISPATCH_HISTORY_MAX:
            for cid, ent in list(_dispatch_history.items()):
                if now - ent.mono_ts > _DISPATCH_HISTORY_HARD_TTL_SEC:
                    _dispatch_history.pop(cid, None)
            while len(_dispatch_history) > _DISPATCH_HISTORY_MAX:
                _dispatch_history.popitem(last=False)


def peek_dispatch(chat_id: str, max_age_sec: float = 0.0) -> "tuple[IntentResult, str, float] | None":
    """Return ``(intent, text, age_sec)`` if a dispatch was tracked within
    ``max_age_sec`` (0 = unlimited within the hard TTL). Does NOT pop.

    Returns None when extended signals are disabled, no entry exists, or
    the entry is older than ``max_age_sec``.
    """
    if not _extended_signals_enabled():
        return None
    with _dispatch_history_lock:
        ent = _dispatch_history.get(chat_id)
        if ent is None:
            return None
        age = time.monotonic() - ent.mono_ts
    if age > _DISPATCH_HISTORY_HARD_TTL_SEC:
        return None
    if max_age_sec > 0 and age > max_age_sec:
        return None
    return ent.intent, ent.text, age


def consume_dispatch(chat_id: str, max_age_sec: float = 0.0) -> "tuple[IntentResult, str, float] | None":
    """Same as :func:`peek_dispatch` but pops the entry on hit so a single
    user action (e.g. ``/cancel``) cannot trigger the same signal twice.
    """
    if not _extended_signals_enabled():
        return None
    with _dispatch_history_lock:
        ent = _dispatch_history.get(chat_id)
        if ent is None:
            return None
        age = time.monotonic() - ent.mono_ts
        if age > _DISPATCH_HISTORY_HARD_TTL_SEC:
            _dispatch_history.pop(chat_id, None)
            return None
        if max_age_sec > 0 and age > max_age_sec:
            return None
        _dispatch_history.pop(chat_id, None)
    return ent.intent, ent.text, age


def clear_dispatch_history_for_tests() -> None:
    """Test-only hook: wipe the per-chat dispatch registry."""
    with _dispatch_history_lock:
        _dispatch_history.clear()


# ── Persisted feedback writer ──────────────────────────────────────────


def record_feedback(
    predicted: IntentResult,
    corrected: str,
    chat_id: str,
    feedback_id: str | None = None,
    text: str = "",
    *,
    signal_type: str = "force_chat",
    metadata: dict | None = None,
) -> str:
    """Append-only JSONL with the misclassification. Returns the feedback_id.

    Pre-Phase-D callers (e.g. ``handlers/_card_action.force_chat``) leave
    ``signal_type`` defaulted so the only schema change visible on those
    records is the new ``signal_type="force_chat"`` field. Extended
    signals use :func:`record_signal` (which routes through here) and
    pass an empty ``corrected_intent`` when the signal is observational
    rather than corrective.

    ``metadata`` is whitelisted to JSON primitives (str/int/float/bool/
    None) so log analyzers and the trainer can rely on a flat shape.
    """
    fid = feedback_id or _new_feedback_id()
    full_text = text or predicted.raw_text or ""
    # force_chat: keep historical no-truncate behaviour so existing
    # trainers see byte-identical text for previously-recorded rows.
    # Every other signal_type goes through the configurable cap.
    persisted_text = full_text if signal_type == "force_chat" else _truncate_text(full_text)
    record = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "chat_id": chat_id,
        "text": persisted_text,
        "predicted_intent": predicted.agent_type,
        "corrected_intent": corrected or "",
        "confidence": predicted.confidence,
        "layer": predicted.layer,
        "feedback_id": fid,
        "signal_type": signal_type,
    }
    if metadata:
        clean: dict = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
        if clean:
            record["metadata"] = clean
    try:
        _append_jsonl(_resolve_path(), record)
    except Exception as e:
        _safe_log(f"[IntentFeedback] write failed (signal={signal_type}): {e}")
        return fid
    _bump_metric(signal_type)
    return fid


def record_signal(
    signal_type: str,
    intent: IntentResult,
    chat_id: str,
    *,
    corrected: str = "",
    text: str = "",
    metadata: dict | None = None,
) -> str | None:
    """Append an observational / inferred signal row.

    Returns the new ``feedback_id`` on write, ``None`` when extended
    signals are disabled. All failures are fire-and-forget — never
    blocks or raises into the main message path.
    """
    if not _extended_signals_enabled():
        return None
    if signal_type == "force_chat":
        # Force_chat has its own dedicated entry point with looser text
        # caps; refuse to alias it here so the contract stays clear.
        _safe_log("[IntentFeedback] record_signal refused signal_type='force_chat'; use record_feedback")
        return None
    return record_feedback(
        predicted=intent,
        corrected=corrected,
        chat_id=chat_id,
        text=text,
        signal_type=signal_type,
        metadata=metadata,
    )


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
    "record_feedback", "record_signal",
    "register_pending", "resolve_pending", "list_recent",
    "track_dispatch", "peek_dispatch", "consume_dispatch",
]
