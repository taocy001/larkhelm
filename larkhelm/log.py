"""larkhelm · structured log writing (Markdown + JSONL) and debug logging"""
from __future__ import annotations

import enum
import json
import os
import re
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import larkhelm.config as _cfg
from larkhelm.concurrency import _jsonl_lock as _log_lock  # shared with token_stats.py
from larkhelm.secure_io import secure_open

__all__ = [
    "_log_lock", "log_entry", "_read_logs", "_read_logs_tail",
    "_get_recent_turns", "_get_recent_turns_uncached",
    "_get_conv_seqno",
    "_debug_log", "safe_log", "lazy_debug_log",
    "info", "warn", "error",
    "Level", "current_log_level",
    "rotate_jsonl_if_needed", "rotate_debug_log_if_needed",
    "redact_error",
]


# ── Secret redaction (used by crew failure cards before exception text is shown to users) ──
#
# Three shapes covered. The 20+ char floor on ``sk-…`` matches real Anthropic / OpenAI
# key shapes (typically 40+ chars) without false-positiving on stage names like
# ``sk-stage-001``. ``api_key=`` and ``Authorization:`` use word-boundary anchors so
# they only match the actual credential prefix, not a substring inside an unrelated
# identifier.
_SECRET_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;'\")]+"), r"\1***"),
    (re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)\S+"), r"\1***"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "sk-***"),
    # REQ-08: APP_SECRET/app_secret key-value pairs
    (re.compile(r"(?i)(APP_SECRET|app_secret)\s*[=\"']+\s*\S{8,}"), r"\1=***"),
    # REQ-08: DeepSeek ds- tokens
    (re.compile(r"\bds-[a-zA-Z0-9]{32,}\b"), "ds-***"),
    # REQ-08: high-entropy hex strings (40+ chars)
    (re.compile(r"\b[0-9a-f]{40,}\b"), "***hex***"),
    # Feishu APP_ID / APP_SECRET in JSON form: "APP_SECRET": "cli_xxx..."
    (re.compile(r'(?i)("APP_(?:ID|SECRET)"\s*:\s*")[^"]{8,}"'), r'\1***"'),
    # tenant_access_token / user_access_token values (token= or token:)
    (re.compile(r'(?i)(tokens?\s*[=:]\s*)[^\s,;\'"\)]{16,}'), r'\1***'),
)


def redact_error(text: str) -> str:
    """Redact common secret shapes from an error message.

    Patterns matched (case-insensitive where sensible):

      * ``api_key=xxx`` / ``api_key: xxx`` → ``api_key=***``
      * ``Authorization: Bearer xxx`` → ``Authorization: Bearer ***``
      * ``sk-[A-Za-z0-9_-]{20,}`` → ``sk-***`` (Anthropic / OpenAI style keys)

    Always returns a string; never raises. Idempotent. Safe to apply to
    any value — non-string inputs are coerced via ``str()``.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    out = text
    for pat, repl in _SECRET_PATTERNS:
        try:
            out = pat.sub(repl, out)
        except Exception:
            # Defensive: a pathological input shouldn't crash error handling.
            continue
    return out


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
_MAX_LOG_MD_BYTES = 10 * 1024 * 1024   # 10 MiB per Markdown shard
_TAIL_SCAN_DEFAULT_BYTES = 1 * 1024 * 1024  # 1 MiB tail window for _read_logs_tail
_jsonl_write_count = 0
_JSONL_ROTATION_CHECK_EVERY = 1000  # check rotation every N log entries
_rotation_lock = threading.Lock()   # separate lock so rotation never deadlocks with _log_lock

# Per-chat conversation sequence number.  Incremented ONLY when a ``user`` or
# ``assistant`` log entry is written (tool / shell / error / debug writes are
# excluded).  Used as the ``_context_cache.RecentTurnsKey.conv_seqno`` so the
# LRU cache for ``_get_recent_turns`` is invalidated by real turns only —
# fixing the Hit=0 problem caused by the former mtime_ns key that changed on
# every single ``log_entry`` write.  Access is guarded by ``_log_lock``.
_chat_conv_seqno: dict[str, int] = {}

# Pending Markdown shard-rotation events; appended by ``_log_file`` while it
# holds ``_log_lock`` and drained by ``log_entry`` *after* release so the
# ``info()`` call (which re-acquires ``_log_lock``) cannot deadlock.
_pending_md_rotation_msgs: list[str] = []

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
    """Active Markdown shard for chat_id today.

    First shard is ``{date}.md`` (legacy name preserved). Once it grows to
    ``_MAX_LOG_MD_BYTES`` the next ``log_entry`` writes into ``{date}-1.md``,
    then ``{date}-2.md``, etc. Returns the *currently active* shard so
    ``log_entry`` can append + add a header for newly created shards.
    """
    d = _cfg.LOG_DIR / chat_id
    d.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')

    max_n = 0
    prefix = f"{date_str}-"
    for p in d.glob(f"{date_str}-*.md"):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        try:
            n = int(stem[len(prefix):])
        except ValueError:
            continue
        if n > max_n:
            max_n = n

    if max_n == 0:
        current = d / f"{date_str}.md"
    else:
        current = d / f"{date_str}-{max_n}.md"

    try:
        if current.exists() and current.stat().st_size >= _MAX_LOG_MD_BYTES:
            new_n = max_n + 1
            # Defer the info() call: this function runs inside _log_lock and
            # info() re-acquires it (non-reentrant) — emitting here would
            # deadlock. log_entry drains the buffer after releasing the lock.
            _pending_md_rotation_msgs.append(
                f"[Log] logs/{chat_id}/{date_str}.md size > 10 MB, "
                f"rolling to {date_str}-{new_n}.md"
            )
            return d / f"{date_str}-{new_n}.md"
    except OSError:
        pass
    return current


def log_entry(
    chat_id: str, role: str, content: str,
    model: str = "claude", trace_id: str | None = None,
) -> None:
    global _jsonl_write_count
    now = datetime.now()
    model_tag = "Claude" if model == "claude" else "Gemini"
    role_label = {
        "user": "**用户**", "assistant": f"**{model_tag}**",
        "tool": "**工具**", "error": "**错误**", "shell": "**Shell**",
        "reset": "**♻️ 会话重置**",
    }.get(role, f"**{role}**")
    redacted = redact_error(content)
    md_line = f"\n### {now.strftime('%H:%M:%S')} {role_label}\n\n{redacted}\n"
    record = {
        "ts":       now.isoformat(timespec="seconds"),
        "chat_id":  chat_id, "role": role, "content": redacted, "model": model,
        "is_error": role == "error",
    }
    if trace_id:
        record["trace_id"] = trace_id
    should_rotate = False
    with _log_lock:
        try:
            p = _log_file(chat_id)
            if not p.exists() or p.stat().st_size == 0:
                date_str = now.strftime('%Y-%m-%d')
                stem = p.stem
                part_label = ""
                if stem != date_str and stem.startswith(date_str + "-"):
                    try:
                        n = int(stem[len(date_str) + 1:])
                        part_label = f" (part {n + 1})"
                    except ValueError:
                        part_label = ""
                with secure_open(p, "a", "utf-8") as f:
                    f.write(f"# {chat_id}  —  {date_str}{part_label}\n")
            with secure_open(p, "a", "utf-8") as f:
                f.write(md_line)
        except OSError as e:
            print(f"[Log] MD 写入失败: {e}", file=sys.stderr)
        try:
            _cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
            with secure_open(_cfg.LOG_DIR / "all.jsonl", "a", "utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[Log] JSONL 写入失败: {e}", file=sys.stderr)
        _jsonl_write_count += 1
        should_rotate = (_jsonl_write_count % _JSONL_ROTATION_CHECK_EVERY == 0)
        # Bump per-chat conversation seqno for user/assistant entries only.
        # Other roles (tool, shell, error, reset, …) do NOT invalidate the
        # ``_get_recent_turns`` LRU cache — only real conversation turns do.
        if role in ("user", "assistant"):
            _chat_conv_seqno[chat_id] = _chat_conv_seqno.get(chat_id, 0) + 1
        # Drain any deferred Markdown-shard rotation messages while we still
        # hold the lock — the drain itself doesn't take any other lock.
        rotation_msgs: list[str] = []
        if _pending_md_rotation_msgs:
            rotation_msgs = list(_pending_md_rotation_msgs)
            _pending_md_rotation_msgs.clear()
    # Call outside the lock: rotate_jsonl_if_needed calls _debug_log which also needs _log_lock
    if should_rotate:
        rotate_jsonl_if_needed()
    for msg in rotation_msgs:
        info(msg)


def _get_conv_seqno(chat_id: str) -> int:
    """Return the conversation sequence number for *chat_id*.

    The number is incremented each time a ``user`` or ``assistant`` entry
    is written to ``all.jsonl`` via :func:`log_entry`.  Tool / shell / error
    / debug entries do NOT bump the counter, so callers that read the same
    conversation state multiple times within one request (e.g. a retry
    path in ``_do_query``) will see a stable key and benefit from the
    ``_context_cache.cached_recent_turns`` LRU hit.

    Thread-safe: reads the value under ``_log_lock`` (same lock as
    ``log_entry``).  Returns 0 for an unknown chat_id (no turns logged yet).
    """
    with _log_lock:
        return _chat_conv_seqno.get(chat_id, 0)


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


def _read_logs_tail(chat_id: str, *, max_bytes: int = _TAIL_SCAN_DEFAULT_BYTES) -> list[dict]:
    """Read records belonging to chat_id from the *tail* of all.jsonl.

    Scans at most ``max_bytes`` from the end of ``all.jsonl``. Does NOT read
    ``all.jsonl.1`` — those records are by definition >100 MB old and not
    summary-relevant.

    When the scan window is truncated (file size > ``max_bytes``), emits a
    single ``_debug_log`` line so operators can correlate a missing record
    with the bound. The first decoded line may be partial (seek can land
    mid-record) — caught by the inner try/except, same pattern as
    ``_get_recent_turns``.
    """
    jsonl_path = _cfg.LOG_DIR / "all.jsonl"
    if not jsonl_path.exists():
        return []
    try:
        with jsonl_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size > max_bytes:
                f.seek(size - max_bytes)
                _debug_log(
                    f"[Memory] tail-scan truncated at {max_bytes} bytes for {chat_id[:8]}"
                )
            else:
                f.seek(0)
            raw = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        _debug_log(f"[Memory] _read_logs_tail failed: {e}")
        return []

    result: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("chat_id") == chat_id:
                result.append(r)
        except Exception:
            continue
    return result


# ── Recent-turn pruning (orchestrator context injection) ─────────────────
#
# Replaces ``tool_result`` block content > 500 bytes with a fixed placeholder
# when ``_get_recent_turns`` builds the ``[Recent conversation]`` block fed
# to the orchestrator. The JSONL on disk is unchanged — only the in-memory
# context string seen by the LLM. See ``.crew_workspace/design.md`` for
# the full rationale and acceptance criteria.

_TOOL_RESULT_THRESHOLD: int = 500
_PRUNE_MAX_DEPTH: int = 8
_TOOL_RESULT_PLACEHOLDER_FMT: str = "[tool_result truncated — {n} bytes]"


@dataclass
class PruningStats:
    """Process-local ring buffer for the last N ``_get_recent_turns`` calls.

    Writes guarded by ``threading.Lock``; reads also hold the lock to
    snapshot the deque safely. No persistence — process exit drops state.
    """
    capacity: int = 100
    calls: "deque[tuple[int, int]]" = field(
        default_factory=lambda: deque(maxlen=100)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, bytes_before: int, bytes_after: int) -> None:
        # 0-byte guard (design §7.4): avoid polluting saved_pct with no-op
        # samples (e.g. empty chat, JSON parse drop).
        if bytes_before <= 0:
            return
        with self.lock:
            self.calls.append((bytes_before, bytes_after))

    def summary(self) -> dict:
        with self.lock:
            samples = list(self.calls)
        window = len(samples)
        before_sum = sum(b for b, _ in samples)
        after_sum = sum(a for _, a in samples)
        if before_sum > 0:
            saved_pct = int((before_sum - after_sum) * 100 / before_sum)
        else:
            saved_pct = 0
        return {
            "window": window,
            "before_sum": before_sum,
            "after_sum": after_sum,
            "saved_pct": saved_pct,
        }


_pruning_stats: PruningStats = PruningStats()


def _maybe_rehydrate_json(s: str) -> Any:
    """Try ``json.loads(s)`` iff ``len(s) >= 50 and s[:1] in ('[', '{')``.

    50-char + first-char heuristic shortcuts the try/except cost for the
    common plain-dialog case. Returns parsed list/dict on success, else
    ``None``. Never raises.
    """
    if not isinstance(s, str):
        return None
    if len(s) < 50 or s[:1] not in ("[", "{"):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _content_byte_len(value: Any) -> int:
    """UTF-8 byte length of ``value``'s serialized form. Never raises."""
    if isinstance(value, str):
        try:
            return len(value.encode("utf-8"))
        except Exception:
            return 0
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except Exception:
        try:
            return len(repr(value).encode("utf-8"))
        except Exception:
            return 0


def _prune_content(content: Any, _depth: int = 0) -> Any:
    """Recursively prune ``tool_result`` blocks whose content > 500 bytes.

    Returns the *same reference* when no pruning occurred (G3 zero-regression
    guarantee). Safe for arbitrary input — never raises. Depth ≤ 8.
    """
    if _depth > _PRUNE_MAX_DEPTH:
        return content

    if isinstance(content, str):
        # JSON-string rehydration: the JSONL on-disk record stores content as
        # a str, but stream-json producers occasionally embed structured
        # content as a serialized list/dict. The 50-char + first-char
        # heuristic gates the try/except so plain dialog pays zero cost.
        if len(content) >= 50 and content[:1] in ("[", "{"):
            parsed = _maybe_rehydrate_json(content)
            if parsed is not None:
                # Rehydration restarts depth at 0 (design §1.2.3).
                pruned = _prune_content(parsed, _depth=0)
                if pruned is parsed:
                    return content  # no modification → preserve str identity
                try:
                    return json.dumps(pruned, ensure_ascii=False)
                except Exception:
                    return content
        return content

    if isinstance(content, list):
        new_items = None
        for i, item in enumerate(content):
            new_item = _prune_content(item, _depth=_depth + 1)
            if new_item is not item:
                if new_items is None:
                    new_items = list(content)
                new_items[i] = new_item
        return new_items if new_items is not None else content

    if isinstance(content, dict):
        # tool_result block: when its serialized content exceeds the
        # threshold, swap for a deterministic placeholder string.
        if content.get("type") == "tool_result":
            inner = content.get("content")
            if inner is not None:
                size = _content_byte_len(inner)
                if size > _TOOL_RESULT_THRESHOLD:
                    new_dict = dict(content)
                    new_dict["content"] = _TOOL_RESULT_PLACEHOLDER_FMT.format(n=size)
                    return new_dict
        # Recurse into non-truncated dicts (tool_use, text, etc.). Identity
        # preserved when no nested replacement happens.
        recurse_dict: dict | None = None
        for k, v in content.items():
            new_v = _prune_content(v, _depth=_depth + 1)
            if new_v is not v:
                if recurse_dict is None:
                    recurse_dict = dict(content)
                recurse_dict[k] = new_v
        return recurse_dict if recurse_dict is not None else content

    return content


def _stringify_for_display(value: Any) -> str:
    """Render pruned content as a string for the [Recent conversation] body.

    Strings pass through; lists/dicts are JSON-encoded; other types fall
    back to ``str()``. Never raises.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        try:
            return str(value)
        except Exception:
            return ""


_PLACEHOLDER_MARKER = "[tool_result truncated —"


def _get_recent_turns(
    chat_id: str, max_turns: int = 6, max_chars: int = 2000,
) -> str:
    """Cached wrapper — delegates to :func:`_get_recent_turns_uncached`.

    Wraps the original tail-read implementation with the
    ``_context_cache.cached_recent_turns`` LRU layer.  The cache key embeds
    ``(chat_id, max_turns, max_chars, conv_seqno)`` where ``conv_seqno`` is
    the per-chat integer returned by :func:`_get_conv_seqno` — incremented
    only on ``user`` / ``assistant`` writes.  Tool, shell, error, and debug
    entries do NOT change the key, so retries within the same request share
    the same cache slot (fixing the former Hit=0 caused by mtime_ns).

    When ``cfg.RECENT_TURNS_CACHE_ENABLED`` is False the body bypasses the
    cache and calls the uncached function directly (PR-prior byte-compat).
    """
    if not bool(getattr(_cfg, "RECENT_TURNS_CACHE_ENABLED", True)):
        return _get_recent_turns_uncached(chat_id, max_turns, max_chars)
    try:
        from larkhelm._context_cache import cached_recent_turns
    except Exception:
        # Cache module unavailable (early bootstrap / test mock) — fall
        # back to the uncached path so the call still works.
        return _get_recent_turns_uncached(chat_id, max_turns, max_chars)
    return cached_recent_turns(
        chat_id, max_turns, max_chars,
        conv_seqno=_get_conv_seqno(chat_id),
        loader=lambda: _get_recent_turns_uncached(chat_id, max_turns, max_chars),
    )


def _get_recent_turns_uncached(
    chat_id: str, max_turns: int = 6, max_chars: int = 2000,
) -> str:
    """Return last N user/assistant turns as compact context for orchestrator injection.

    Reads the tail of all.jsonl (100 KB) to avoid scanning the full file.
    Skips tool/error/shell entries — only user and assistant text. Per
    record, large ``tool_result`` block bodies are replaced with a
    placeholder before the 400-char dialog cap (design v1.0).
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

    total_before = 0
    total_after = 0

    lines = ["[Recent conversation]"]
    for r in turns:
        role = "User" if r["role"] == "user" else "Assistant"
        raw_content = r.get("content", "")

        bytes_before = _content_byte_len(raw_content)
        pruned = _prune_content(raw_content)
        display = _stringify_for_display(pruned)
        bytes_after = len(display.encode("utf-8")) if display else 0

        if bytes_after < bytes_before:
            blocks = display.count(_PLACEHOLDER_MARKER)
            try:
                _debug_log(
                    f"[Log] _get_recent_turns pruned chat={chat_id[:8]} "
                    f"blocks={blocks} saved={bytes_before - bytes_after}"
                )
            except Exception:
                pass

        total_before += bytes_before
        total_after += bytes_after

        content = display.strip()
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role}: {content}")

    if total_before > 0:
        try:
            _pruning_stats.record(total_before, total_after)
        except Exception:
            pass

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
            with secure_open(_cfg.DEBUG_LOG, "a", "utf-8") as f:
                f.write(line)
        except Exception:
            pass
        _debug_write_count += 1
        # Probe size every N writes rather than every write to avoid stat()
        # on every diagnostic line. With N=500 and 50 MB threshold the
        # worst-case overshoot is bounded by the largest write batch.
        should_rotate = (_debug_write_count % _DEBUG_ROTATE_CHECK_EVERY == 0)
    # Diagnostics go to stdout (preserved from pre-Phase-2 behaviour) so
    # ``test_log_level.py::TestLarkClientWarnConsolidation`` keeps its
    # "no stderr double-write" invariant. CLI subcommands that emit
    # machine-readable output (e.g. ``larkhelm memory audit-summary --json``)
    # are kept pipe-safe by ``__main__._cmd_memory`` not invoking
    # ``_init_runtime`` for the read-only branches that trigger Runner /
    # ModelProbe / MemoryGC boot logs.
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

    Replaces identical ``_safe_log`` copies that previously lived in agent_hub modules.
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
    ``agent_hub/agent_base.py.abort()``.
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
