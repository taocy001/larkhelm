"""larkhelm · structured log writing (Markdown + JSONL) and debug logging"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.concurrency import _jsonl_lock as _log_lock  # shared with token_stats.py

__all__ = ["_log_lock", "log_entry", "_read_logs", "_debug_log"]


def _log_file(chat_id: str) -> Path:
    d = _cfg.LOG_DIR / chat_id
    d.mkdir(exist_ok=True)
    return d / f"{datetime.now().strftime('%Y-%m-%d')}.md"


def log_entry(
    chat_id: str, role: str, content: str,
    model: str = "claude", trace_id: str = None,
) -> None:
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
    with _log_lock:
        try:
            p = _log_file(chat_id)
            if not p.exists() or p.stat().st_size == 0:
                with p.open("a", encoding="utf-8") as f:
                    f.write(f"# {chat_id}  —  {now.strftime('%Y-%m-%d')}\n")
            with p.open("a", encoding="utf-8") as f:
                f.write(md_line)
        except OSError as e:
            print(f"[log] MD 写入失败: {e}", file=sys.stderr)
        try:
            with (_cfg.LOG_DIR / "all.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[log] JSONL 写入失败: {e}", file=sys.stderr)


def _read_logs(chat_id: str) -> list[dict]:
    """Stream-read records belonging to chat_id from all.jsonl to avoid loading the entire file at once."""
    with _log_lock:
        log_path = _cfg.LOG_DIR / "all.jsonl"
    result = []
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
    except Exception:
        pass
    return result


def _debug_log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    with _log_lock:
        try:
            with _cfg.DEBUG_LOG.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
    # stdout 输出到文件（当 stdout 被重定向时）
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass
