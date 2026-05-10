"""larkhelm · structured log writing (Markdown + JSONL) and debug logging"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.concurrency import _jsonl_lock as _log_lock  # shared with token_stats.py

__all__ = [
    "_log_lock", "log_entry", "_read_logs", "_get_recent_turns",
    "_debug_log", "safe_log", "lazy_debug_log",
    "rotate_jsonl_if_needed", "rotate_debug_log_if_needed",
]

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


def _debug_log(msg: str) -> None:
    global _debug_write_count
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
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
