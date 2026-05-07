"""larkhelm · semantic memory — per-chat persistent memory files

File format: DATA_DIR/memory/{chat_id}.md
  YAML frontmatter + Markdown body (≤ MEMORY_MAX_CHARS chars)

Usage flow:
  1. _do_query() calls inject_memory(chat_id, message) → prepended prompt
  2. After query, _do_query() calls maybe_auto_update(chat_id) in background thread
  3. /reset calls maybe_auto_update(chat_id, force=True) before clearing session
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.concurrency import _get_chat_lock
from larkhelm.chat_state import _get_turn_count
from larkhelm.log import _read_logs, _debug_log

MEMORY_MAX_CHARS = 2000
AUTO_UPDATE_EVERY = 20

_SUMMARIZE_PROMPT = """\
You are a memory assistant. Based on the conversation log below, write a concise \
persistent memory summary (max {max_chars} characters) in the same language as the \
conversation. Focus on: project context, user goals, key decisions, and important facts. \
Use Markdown headings. Output ONLY the summary — no preamble, no meta-commentary.

---CONVERSATION LOG---
{logs}
---END LOG---
"""


def _memory_file(chat_id: str) -> Path:
    d = _cfg.DATA_DIR / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{chat_id}.md"


def load_memory(chat_id: str) -> str | None:
    """Read memory file body (Markdown, no frontmatter). Returns None if absent."""
    try:
        f = _memory_file(chat_id)
        if not f.exists():
            return None
        text = f.read_text(encoding="utf-8")
        # Strip YAML frontmatter (--- ... ---)
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                return text[end + 4:].strip() or None
        return text.strip() or None
    except Exception as e:
        _debug_log(f"[memory] load_memory error {chat_id}: {e}")
        return None


def save_memory(chat_id: str, content: str) -> None:
    """Atomically write memory file (holds chat lock). content is Markdown body."""
    try:
        lock = _get_chat_lock(chat_id)
        with lock:
            now = datetime.now().isoformat(timespec="seconds")
            turns = _get_turn_count(chat_id)
            frontmatter = (
                f"---\nchat_id: {chat_id}\nupdated_at: \"{now}\"\n"
                f"turns: {turns}\nversion: 1\n---\n\n"
            )
            body = content[:MEMORY_MAX_CHARS]
            f = _memory_file(chat_id)
            tmp = f.with_suffix(".md.tmp")
            tmp.write_text(frontmatter + body, encoding="utf-8")
            tmp.replace(f)
            _debug_log(f"[memory] saved {chat_id} ({len(body)} chars, turns={turns})")
    except Exception as e:
        _debug_log(f"[memory] save_memory error {chat_id}: {e}")


def generate_memory(chat_id: str, recent_logs: str) -> str:
    """Call Claude CLI to generate a memory summary. Returns Markdown string (≤ MEMORY_MAX_CHARS)."""
    from larkhelm.backend_registry import BACKEND_REGISTRY
    from larkhelm.backend_cli import run_claude

    spec = BACKEND_REGISTRY.get_orchestrator()
    if spec is None:
        raise RuntimeError("No orchestrator backend available for memory generation")

    prompt = _SUMMARIZE_PROMPT.format(max_chars=MEMORY_MAX_CHARS, logs=recent_logs[:8000])

    collected: list[str] = []

    def _on_text(text: str, status: str = "typing") -> None:
        collected.clear()
        collected.append(text)

    try:
        output = run_claude(
            spec=spec,
            chat_id=chat_id,
            message=prompt,
            sid=None,
            cwd=str(_cfg.DATA_DIR),
            on_text=_on_text,
        )
        return (output or "".join(collected))[:MEMORY_MAX_CHARS]
    except Exception as e:
        _debug_log(f"[memory] generate_memory error {chat_id}: {e}")
        raise


def inject_memory(chat_id: str, message: str) -> str:
    """Prepend persistent memory block to message. Returns unchanged message if no memory."""
    content = load_memory(chat_id)
    if not content:
        return message
    return f"[PERSISTENT MEMORY]\n{content}\n[END MEMORY]\n\n{message}"


def maybe_auto_update(chat_id: str, force: bool = False) -> None:
    """Check if memory needs updating and run in a background thread if so.

    Triggers when turn_count % AUTO_UPDATE_EVERY == 0 (or force=True).
    Failures are logged but silently swallowed — old memory file is preserved.
    """
    turn_count = _get_turn_count(chat_id)
    if not force and (turn_count == 0 or turn_count % AUTO_UPDATE_EVERY != 0):
        return

    def _run():
        try:
            logs = _read_logs(chat_id)
            if not logs:
                return
            # Use last 40 records for context
            recent = logs[-40:]
            log_text = "\n".join(
                f"[{r['ts']}] {r['role']}: {r['content'][:500]}"
                for r in recent
                if r["role"] in ("user", "assistant")
            )
            if not log_text.strip():
                return
            content = generate_memory(chat_id, log_text)
            save_memory(chat_id, content)
        except Exception as e:
            _debug_log(f"[memory] maybe_auto_update error {chat_id}: {e}")

    threading.Thread(target=_run, daemon=True, name=f"memory-{chat_id[:8]}").start()
