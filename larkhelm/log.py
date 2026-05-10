"""larkhelm · structured log writing (Markdown + JSONL) and debug logging"""
from __future__ import annotations

import enum
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.concurrency import _jsonl_lock as _log_lock  # shared with token_stats.py

__all__ = [
    "_log_lock", "log_entry", "_read_logs", "_get_recent_turns",
    "_debug_log", "safe_log", "lazy_debug_log",
    "info", "warn", "error",
    "Level", "current_log_level",
    "rotate_jsonl_if_needed", "rotate_debug_log_if_needed",
]


# ── Level filter (LARKHELM_LOG_LEVEL env var) ─────────────────────────────
#
# Phase 4 of the logging unification (see CLAUDE.md "日志前缀规范"):
# every diagnostic write goes through ``_log_at(level, msg)``; ``_debug_log``
# is now equivalent to ``_log_at(Level.DEBUG, msg)`` so an operator setting
# ``LARKHELM_LOG_LEVEL=WARN`` can silence the ~250 existing DEBUG-level call
# sites in production without touching code. Default ``DEBUG`` preserves
# the full pre-Phase-4 verbosity for existing deployments.
#
# Why not just rip out _debug_log? 250+ call sites, each with its own
# ``[Module]`` prefix already encoding meaning; rewriting them as
# ``info/warn/error`` would be ~250 mechanical edits with high regression
# surface. Keeping ``_debug_log`` as the DEBUG entry-point + adding new
# typed helpers for new code is the minimum-disruption path.


class Level(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


_LEVEL_ORDER: dict[Level, int] = {
    Level.DEBUG: 0,
    Level.INFO: 1,
    Level.WARN: 2,
    Level.ERROR: 3,
}


def _resolve_level_from_env() -> Level:
    """Read ``LARKHELM_LOG_LEVEL`` once at import time. Unknown values fall
    back to DEBUG (with a stderr warning) so a typo doesn't accidentally
    silence the entire bridge. Read once on import — operators changing
    the env mid-process do NOT see live updates; that's intentional, the
    log filter should not flip mid-run."""
    raw = (os.environ.get("LARKHELM_LOG_LEVEL") or "").strip().upper()
    if not raw:
        return Level.DEBUG
    try:
        return Level(raw)
    except ValueError:
        # Don't go through _debug_log itself — it depends on this resolution.
        try:
            print(
                f"[Log] LARKHELM_LOG_LEVEL={raw!r} not in "
                "{DEBUG, INFO, WARN, ERROR}, falling back to DEBUG",
                file=sys.stderr,
            )
        except Exception:
            pass
        return Level.DEBUG


_min_level: Level = _resolve_level_from_env()


def current_log_level() -> Level:
    """Public read-only accessor for tests / /status reporting."""
    return _min_level


def _level_enabled(level: Level) -> bool:
    return _LEVEL_ORDER[level] >= _LEVEL_ORDER[_min_level]

_MAX_JSONL_BYTES = 100 * 1024 * 1024  # 100 MB
_jsonl_write_count = 0
_JSONL_ROTATION_CHECK_EVERY = 1000  # check rotation every N log entries
_rotation_lock = threading.Lock()   # separate lock so rotation never deadlocks with _log_lock

# DEBUG_LOG rotation. Threshold is stricter than all.jsonl because every
# _debug_log call also writes to stdout, so the file fills faster from
# verbose modules. We keep one backup (.log.1) and probe the size every
# _DEBUG_ROTATE_CHECK_EVERY writes to amortize the stat() cost.
_MAX_DEBUG_LOG_BYTES = 50 * 1024 * 1024  # 50 MB
_debug_write_count = 0
_DEBUG_ROTATE_CHECK_EVERY = 500


def rotate_jsonl_if_needed() -> None:
    """Rotate all.jsonl if it exceeds _MAX_JSONL_BYTES; keeps one backup (.jsonl.1)."""
    p = _cfg.LOG_DIR / "all.jsonl"
    with _rotation_lock:
        try:
            if p.exists() and p.stat().st_size > _MAX_JSONL_BYTES:
                backup = p.with_suffix(".jsonl.1")
                backup.unlink(missing_ok=True)
                p.rename(backup)
                _debug_log(f"[Log] all.jsonl rotated → all.jsonl.1 ({backup.stat().st_size // 1024 // 1024} MB)")
        except Exception as e:
            print(f"[Log] JSONL rotation failed: {e}", file=sys.stderr)


def rotate_debug_log_if_needed() -> None:
    """Rotate DEBUG_LOG if it exceeds _MAX_DEBUG_LOG_BYTES; keeps one backup (.log.1).

    Mirrors :func:`rotate_jsonl_if_needed`'s pattern (rename + drop older
    backup) so the file never grows unbounded in long-running deployments.
    Must NOT call ``_debug_log`` from inside ``_rotation_lock`` because
    ``_debug_log`` only takes ``_log_lock``; calling it after release is
    fine. Falls back to ``stderr`` print on any rotation failure to avoid
    recursing into the very file we're trying to rotate.
    """
    p = _cfg.DEBUG_LOG
    rotated_size_mb = None
    with _rotation_lock:
        try:
            if p.exists() and p.stat().st_size > _MAX_DEBUG_LOG_BYTES:
                # Build the backup name by appending ``.1`` rather than
                # ``with_suffix`` so we work for any DEBUG_LOG suffix
                # (default ``.log``; tests sometimes use ``.txt`` etc.).
                backup = p.with_name(p.name + ".1")
                backup.unlink(missing_ok=True)
                p.rename(backup)
                rotated_size_mb = backup.stat().st_size // 1024 // 1024
        except Exception as e:
            print(f"[Log] DEBUG_LOG rotation failed: {e}", file=sys.stderr)
    if rotated_size_mb is not None:
        # Best-effort post-rotation note; if it fails we don't care because
        # we just freed the disk space anyway.
        try:
            _debug_log(f"[Log] DEBUG_LOG rotated → {p.name}.1 ({rotated_size_mb} MB)")
        except Exception:
            pass


def _log_file(chat_id: str) -> Path:
    d = _cfg.LOG_DIR / chat_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{datetime.now().strftime('%Y-%m-%d')}.md"


def log_entry(
    chat_id: str, role: str, content: str,
    model: str = "claude", trace_id: str = None,
) -> None:
    global _jsonl_write_count
    now = datetime.now()
    model_tag = "Claude" if model == "claude" else "Gemini"
    role_label = {
        "user": "**用户**", "assistant": f"**{model_tag}**",
        "tool": "**工具**", "error": "**错误**", "shell": "**Shell**",
        "reset": "**♻️ 会话重置**",
    }.get(role, f"**{role}**")
    md_line = f"\n### {now.strftime('%H:%M:%S')} {role_label}\n\n{content}\n"
    record = {
        "ts":       now.isoformat(timespec="seconds"),
        "chat_id":  chat_id, "role": role, "content": content, "model": model,
        "is_error": role == "error",
    }
    if trace_id:
        record["trace_id"] = trace_id
    should_rotate = False
    with _log_lock:
        try:
            p = _log_file(chat_id)
            if not p.exists() or p.stat().st_size == 0:
                with p.open("a", encoding="utf-8") as f:
                    f.write(f"# {chat_id}  —  {now.strftime('%Y-%m-%d')}\n")
            with p.open("a", encoding="utf-8") as f:
                f.write(md_line)
        except OSError as e:
            print(f"[Log] MD 写入失败: {e}", file=sys.stderr)
        try:
            _cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (_cfg.LOG_DIR / "all.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[Log] JSONL 写入失败: {e}", file=sys.stderr)
        _jsonl_write_count += 1
        should_rotate = (_jsonl_write_count % _JSONL_ROTATION_CHECK_EVERY == 0)
    # Call outside the lock: rotate_jsonl_if_needed calls _debug_log which also needs _log_lock
    if should_rotate:
        rotate_jsonl_if_needed()


def _read_logs(chat_id: str) -> list[dict]:
    """Read records belonging to chat_id from all.jsonl (and .jsonl.1 backup if present).

    Reading without holding _log_lock is safe: the worst case is we miss the very last
    incomplete line (handled by the inner try/except).  Holding the lock for a full
    file scan would stall log writes for the entire duration.
    """
    log_dir = _cfg.LOG_DIR
    candidates = [log_dir / "all.jsonl.1", log_dir / "all.jsonl"]
    result = []
    for log_path in candidates:
        try:
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if r.get("chat_id") == chat_id:
                            result.append(r)
                    except Exception:
                        continue
        except Exception as e:
            _debug_log(f"[log] read_logs failed: {e}")
    return result


def _get_recent_turns(chat_id: str, max_turns: int = 6, max_chars: int = 2000) -> str:
    """Return last N user/assistant turns as compact context for orchestrator injection.

    Reads the tail of all.jsonl (100 KB) to avoid scanning the full file.
    Skips tool/error/shell entries — only user and assistant text.
    """
    TAIL_BYTES = 100 * 1024
    jsonl_path = _cfg.LOG_DIR / "all.jsonl"
    if not jsonl_path.exists():
        return ""
    try:
        with jsonl_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            raw = f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

    turns = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("chat_id") == chat_id and r.get("role") in ("user", "assistant"):
                turns.append(r)
        except Exception:
            continue

    turns = turns[-(max_turns * 2):]
    if not turns:
        return ""

    lines = ["[Recent conversation]"]
    for r in turns:
        role = "User" if r["role"] == "user" else "Assistant"
        content = r.get("content", "").strip()
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role}: {content}")

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[-max_chars:]
    return result


def _log_at(level: Level, msg: str) -> None:
    """Single write path used by ``_debug_log`` (DEBUG level) and the
    typed ``info`` / ``warn`` / ``error`` helpers.

    Format:
      * DEBUG: ``[HH:MM:SS] {msg}\\n`` — unchanged from pre-Phase-4 to
        preserve grep compatibility for the ~250 existing DEBUG callers.
      * INFO/WARN/ERROR: ``[HH:MM:SS] <LEVEL> {msg}\\n`` — explicit tag
        so operators (and ``scripts/memory_observation_report.py``) can
        filter by severity.

    Honors ``LARKHELM_LOG_LEVEL``: writes below ``_min_level`` are dropped
    silently.
    """
    if not _level_enabled(level):
        return
    global _debug_write_count
    ts = datetime.now().strftime('%H:%M:%S')
    if level is Level.DEBUG:
        line = f"[{ts}] {msg}\n"
    else:
        line = f"[{ts}] <{level.value}> {msg}\n"
    should_rotate = False
    with _log_lock:
        try:
            with _cfg.DEBUG_LOG.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
        _debug_write_count += 1
        # Probe size every N writes rather than every write to avoid stat()
        # on every diagnostic line. With N=500 and 50 MB threshold the
        # worst-case overshoot is bounded by the largest write batch.
        should_rotate = (_debug_write_count % _DEBUG_ROTATE_CHECK_EVERY == 0)
    # stdout 输出到文件（当 stdout 被重定向时）
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass
    # Call rotation OUTSIDE _log_lock: rotate_debug_log_if_needed acquires
    # _rotation_lock and (on success) re-enters _debug_log to record the
    # rotation event, which would deadlock if we still held _log_lock.
    if should_rotate:
        rotate_debug_log_if_needed()


def _debug_log(msg: str) -> None:
    """DEBUG-level diagnostic write. Equivalent to ``_log_at(Level.DEBUG, msg)``.

    Kept as a separate name (instead of inlining the call sites) because
    ~250 existing callers reference it; the gate behavior is identical
    when ``LARKHELM_LOG_LEVEL=DEBUG`` (the default).
    """
    _log_at(Level.DEBUG, msg)


def info(msg: str) -> None:
    """INFO-level diagnostic write. Format: ``[HH:MM:SS] <INFO> {msg}``."""
    _log_at(Level.INFO, msg)


def warn(msg: str) -> None:
    """WARN-level diagnostic write. Use for degraded-behavior notices that
    a operator-on-call would want to see (e.g. credential fetch failed,
    falling back to defaults). Replaces the legacy "print to stderr +
    _debug_log" double-write idiom in ``lark_client.py``.
    """
    _log_at(Level.WARN, msg)


def error(msg: str) -> None:
    """ERROR-level diagnostic write. Use for failures that interrupt
    a user-visible task (so the operator can correlate with a ticket)."""
    _log_at(Level.ERROR, msg)


def safe_log(msg: str) -> None:
    """``_debug_log`` 的"永不抛"版本，用于异常清理 / 日志降级路径。

    Replaces the 4 identical ``_safe_log`` copies that previously lived in
    ``agent_hub/{agent_dispatcher,agent_audit,intent_feedback,plugin_loader}.py``.
    The wrapping ``try/except Exception: pass`` guards against any regression
    that would make ``_debug_log`` itself raise (e.g. config not yet loaded
    in early bootstrap). All four call sites were behaviorally identical, so
    centralizing here removes ~28 lines of duplication and a future
    drift-risk.
    """
    try:
        _debug_log(msg)
    except Exception:
        pass


def lazy_debug_log(msg: str) -> None:
    """Diagnostic log helper safe for bootstrap / circular-import edges.

    Replaces the inline ``try: from larkhelm.log import _debug_log; _debug_log(msg)
    except Exception: pass`` pattern that was duplicated inside ``config.py``,
    ``agent_hub/agent_base.py.abort()`` and ``agent_hub/intent_router.py``.
    These call sites import ``larkhelm.log`` lazily (inside an exception
    handler) because they are reached from positions where the module may
    not yet be importable (e.g. a recovery thread fires before
    ``_init_runtime`` finishes, or a partial agent_hub import races with
    ``larkhelm.log`` import).

    Behaviorally identical to ``safe_log``; the separate name keeps the
    "this is a bootstrap-edge call site" intent visible at the call site
    and lets us evolve the two helpers independently if we ever need to.
    """
    try:
        _debug_log(msg)
    except Exception:
        pass
