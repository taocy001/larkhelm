"""larkhelm · three-tier persistent memory

Storage: ~/.larkhelm/memory/
  global_{open_id}.md    — user-level: preferences, style, cross-project habits (≤800 chars)
                           keyed by sender_open_id; returns None when open_id unknown (group safety)
  project_{hash16}.md    — project-level: tech stack, conventions, keyed by resolved cwd (≤1500 chars)
  session_{chat_id}.md   — session-level: current work, decisions, context (≤2000 chars)
                           auto-updated every 10 turns; cascades to project/global after each update

Injection order: global → project → session (passed via extra_system channel, new sessions only).

Auto-learning flow:
  1. Every 10 turns: session summary is regenerated from recent logs
  2. After each session update: background cascade tries to extract new facts into
     project memory (tech stack, conventions) and global memory (user preferences)
  3. Extraction uses "UNCHANGED" sentinel — only writes when genuinely new info is found
  4. /reset triggers a forced session snapshot + cascade before clearing the session

  inject_memory() is a legacy compatibility shim — do not call in new code.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import larkhelm.config as _cfg
from larkhelm.chat_state import _get_turn_count, _get_chat_state
from larkhelm.log import _read_logs, _debug_log

# ── Storage root ─────────────────────────────────────────────────────────────

MEMORY_HOME_DIR = Path.home() / ".larkhelm" / "memory"

# ── Per-layer size budgets ────────────────────────────────────────────────────

GLOBAL_MAX_CHARS  = 800
PROJECT_MAX_CHARS = 1500
SESSION_MAX_CHARS = 2000
TOTAL_MEMORY_BUDGET = 4500  # combined cap; tag overhead counted separately

_TAG_OVERHEAD_PER_LAYER = 50  # chars for open/close tag + surrounding newlines

# ── Auto-update tuning ────────────────────────────────────────────────────────

AUTO_UPDATE_EVERY         = 10   # session auto-update every N turns
MEMORY_GENERATION_TIMEOUT = 120  # seconds before abandoning a slow generate call
_EXTRACT_TIMEOUT          = 60   # shorter timeout for cascade extraction calls

# ── Lock pools ────────────────────────────────────────────────────────────────
# Both dicts are bounded to prevent unbounded growth in long-running processes.

_MAX_POOL_SIZE = 512

_update_locks: dict[str, threading.Lock] = {}
_update_locks_meta = threading.Lock()

_file_write_locks: dict[str, threading.Lock] = {}
_file_write_locks_meta = threading.Lock()


def _get_update_lock(chat_id: str) -> threading.Lock:
    with _update_locks_meta:
        if chat_id in _update_locks:
            return _update_locks[chat_id]
        if len(_update_locks) >= _MAX_POOL_SIZE:
            for old_key in list(_update_locks):
                if not _update_locks[old_key].locked():
                    del _update_locks[old_key]
                    break
        lock = threading.Lock()
        _update_locks[chat_id] = lock
        return lock


def _get_file_write_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _file_write_locks_meta:
        if key in _file_write_locks:
            return _file_write_locks[key]
        if len(_file_write_locks) >= _MAX_POOL_SIZE:
            for old_key in list(_file_write_locks):
                if not _file_write_locks[old_key].locked():
                    del _file_write_locks[old_key]
                    break
        lock = threading.Lock()
        _file_write_locks[key] = lock
        return lock


# ── Prompts ───────────────────────────────────────────────────────────────────

_SUMMARIZE_PROMPT = """\
You are a session memory assistant. Write a structured, concise memory summary \
(max {max_chars} chars) in the SAME LANGUAGE as the conversation.

Use these Markdown sections (omit any section that has nothing to say):

## Work Context
What project/repo, what task is in progress right now.

## Key Decisions & Facts
Important choices made, constraints discovered, technical conclusions.

## Next Steps
Pending items or open questions, if any.

Preserve important facts from existing memory; incorporate new information from the log.
Output ONLY the memory content — no preamble, no commentary.

{existing_memory_section}---RECENT CONVERSATION---
{logs}
---END LOG---
"""

# Extraction prompts: output "UNCHANGED" (exact, no punctuation) if nothing new to add.

_EXTRACT_PROJECT_PROMPT = """\
You are a project knowledge extractor. Review the session memory and decide whether it \
reveals NEW project-specific facts not already captured in existing project memory.

NEW facts to look for: tech stack choices, architecture decisions, file/folder conventions, \
testing approach, coding style rules, recurring patterns, project constraints.

If nothing new: output exactly the word UNCHANGED (nothing else).
Otherwise: output updated project memory (max {max_chars} chars, Markdown, same language) \
that merges existing facts with the new ones. Do not repeat information already in existing memory.

<existing_project_memory>
{existing}
</existing_project_memory>

<session_memory>
{session}
</session_memory>
"""

_EXTRACT_GLOBAL_PROMPT = """\
You are a personal preference extractor. Review the session memory and decide whether it \
reveals NEW cross-project user preferences not already captured in existing global memory.

NEW facts to look for: communication style, language preference, response format habits, \
working style, domain expertise, personal conventions that apply across ALL projects.

If nothing new: output exactly the word UNCHANGED (nothing else).
Otherwise: output updated global memory (max {max_chars} chars, Markdown, same language) \
that merges existing facts with the new ones. Do not repeat existing information.

<existing_global_memory>
{existing}
</existing_global_memory>

<session_memory>
{session}
</session_memory>
"""

# ── File helpers ──────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    MEMORY_HOME_DIR.mkdir(parents=True, exist_ok=True)


def _global_memory_file(chat_id: str | None = None) -> Path | None:
    """Return global memory path keyed by sender_open_id.

    Returns None when chat_id is absent or open_id cannot be determined,
    so the global layer is skipped rather than shared across all group members.
    global_default.md is intentionally NOT used as a fallback.
    """
    _ensure_dir()
    if not chat_id:
        return None
    try:
        state = _get_chat_state(chat_id)
        open_id = state.get("sender_open_id", "") or ""
        if not open_id:
            return None
        return MEMORY_HOME_DIR / f"global_{open_id}.md"
    except Exception:
        return None


def _project_memory_file(cwd: str) -> Path:
    _ensure_dir()
    canonical = str(Path(cwd).resolve())
    h = hashlib.md5(canonical.encode()).hexdigest()[:16]
    return MEMORY_HOME_DIR / f"project_{h}.md"


def _session_memory_file(chat_id: str) -> Path:
    _ensure_dir()
    return MEMORY_HOME_DIR / f"session_{chat_id}.md"


def _memory_file(chat_id: str) -> Path:
    """Backward-compat alias for _session_memory_file."""
    return _session_memory_file(chat_id)


# ── Low-level read/write ─────────────────────────────────────────────────────

def _load_md_body(path: Path | None) -> str | None:
    """Read a memory file body, stripping YAML frontmatter. Returns None if absent/empty."""
    if path is None:
        return None
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                return text[end + 4:].strip() or None
        return text.strip() or None
    except Exception as e:
        _debug_log(f"[Memory] load error {path.name}: {e}")
        return None


def _load_md_frontmatter(path: Path | None) -> dict[str, str]:
    """Parse YAML-like frontmatter key: value pairs from a memory file."""
    result: dict[str, str] = {}
    if path is None:
        return result
    try:
        if not path.exists():
            return result
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return result
        end = text.find("\n---", 3)
        if end == -1:
            return result
        for line in text[3:end].splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip().strip('"')
    except Exception as e:
        _debug_log(f"[Memory] frontmatter parse error: {e}")
    return result


def _save_md(path: Path | None, content: str, max_chars: int, extra_fm: str = "") -> None:
    """Atomically write a memory file with YAML frontmatter, protected by a per-file lock.

    Permissions: the tmp file is chmod'd 0600 BEFORE the atomic replace so the
    final file inherits user-only read/write regardless of process umask.
    Aligns with ``agent_audit._append_jsonl`` / ``intent_feedback._append_jsonl``
    which already enforce 0600 on JSONL audit files. Memory files contain
    distilled chat content (which may include user secrets they typed in),
    so 0644 default umask is unacceptable in shared environments.
    """
    if path is None:
        return
    lock = _get_file_write_lock(path)
    with lock:
        try:
            now = datetime.now().isoformat(timespec="seconds")
            fm = f'---\nupdated_at: "{now}"\n{extra_fm}---\n\n'
            body = content[:max_chars]
            tmp = path.with_suffix(".md.tmp")
            tmp.write_text(fm + body, encoding="utf-8")
            # Tighten perms BEFORE the rename so the final path is never
            # observable with broader bits; if chmod fails (filesystem
            # without unix perms, e.g. CIFS) we log and continue, not abort.
            try:
                os.chmod(tmp, 0o600)
            except OSError as _e:
                _debug_log(f"[Memory] chmod 0600 failed on {tmp.name}: {_e}")
            try:
                tmp.replace(path)
            except OSError:
                shutil.move(str(tmp), str(path))
            _debug_log(f"[Memory] saved {path.name} ({len(body)} chars)")
        except Exception as e:
            _debug_log(f"[Memory] save error {path.name}: {e}")


# ── Public load/save API ──────────────────────────────────────────────────────

def load_global_memory(chat_id: str | None = None) -> str | None:
    return _load_md_body(_global_memory_file(chat_id))


def save_global_memory(content: str, chat_id: str | None = None) -> None:
    _save_md(_global_memory_file(chat_id), content, GLOBAL_MAX_CHARS)


def load_project_memory(cwd: str) -> str | None:
    path = _project_memory_file(cwd)
    content = _load_md_body(path)
    if content is None:
        return None
    fm = _load_md_frontmatter(path)
    stored_cwd = fm.get("cwd", "")
    if stored_cwd:
        canonical = str(Path(cwd).resolve())
        stored_canonical = str(Path(stored_cwd).resolve())
        if stored_canonical != canonical:
            _debug_log(f"[Memory] cwd mismatch: stored={stored_cwd!r} vs requested={cwd!r}, skipping")
            return None
        if not Path(stored_canonical).exists():
            _debug_log(f"[Memory] project cwd gone, skipping: {stored_cwd}")
            return None
    return content


def save_project_memory(cwd: str, content: str) -> None:
    canonical = str(Path(cwd).resolve())
    _save_md(_project_memory_file(cwd), content, PROJECT_MAX_CHARS,
             f'cwd: "{canonical}"\n')


# ── Project-memory garbage collection (user-explicit /memory gc) ────────────

# Default age threshold for /memory gc: project files unmodified for 30+ days
# are treated as stale candidates. Picked conservatively because deleting
# memory is irreversible (no .trash/) — a one-month window comfortably covers
# returning to a project after a vacation but flags genuinely abandoned ones.
_GC_DEFAULT_DAYS = 30


def gc_project_memory(threshold_days: int = _GC_DEFAULT_DAYS,
                      apply: bool = False) -> dict:
    """Identify (and optionally delete) stale ``project_*.md`` files.

    A file is considered stale when ANY of:
      * it has not been modified in ``threshold_days`` days, OR
      * its frontmatter ``cwd`` no longer points to an existing directory
        (project was moved/deleted; memory file is now orphaned).

    The function NEVER touches ``session_*.md`` or ``global_*.md`` — those
    have different lifecycle semantics (session = active conversation,
    global = per-user singleton). Project memory is the only layer where
    files accumulate over time on a developer's machine.

    Parameters
    ----------
    threshold_days : int
        Files unmodified for this many days qualify on age. Must be ≥ 1
        (passing 0 would clear everything; we forbid that to avoid
        catastrophic mistakes from typo'd commands).
    apply : bool
        ``False`` (default) → dry-run, just report. ``True`` → actually
        unlink the files. Callers expose the dry-run path as the default
        UX so users can review before destruction.

    Returns
    -------
    dict with keys:
      * ``threshold_days``: echoed back
      * ``apply``: echoed back
      * ``scanned``: total project_*.md count
      * ``candidates``: list of dicts with
            {name, path, cwd, age_days (None if unknown), reason, deleted}
        where reason ∈ {"stale_age", "cwd_gone", "stale_age+cwd_gone"}
        and ``deleted`` is True iff ``apply`` was True AND the unlink
        actually succeeded.
      * ``errors``: list of {path, err} for unlink failures (apply mode)

    Failures during single-file scan are swallowed (logged) so one
    unreadable file doesn't break the whole report.
    """
    if threshold_days < 1:
        raise ValueError("threshold_days must be >= 1")

    _ensure_dir()
    now = datetime.now().timestamp()
    cutoff = now - threshold_days * 86400

    candidates: list[dict] = []
    errors: list[dict] = []
    scanned = 0
    for path in MEMORY_HOME_DIR.glob("project_*.md"):
        scanned += 1
        try:
            mtime = path.stat().st_mtime
            age_days = max(0, int((now - mtime) / 86400))
            stale_age = mtime < cutoff
            cwd_gone = False
            stored_cwd = ""
            try:
                fm = _load_md_frontmatter(path)
                stored_cwd = fm.get("cwd", "") or ""
                if stored_cwd:
                    cwd_gone = not Path(stored_cwd).expanduser().exists()
            except Exception as _e:
                _debug_log(f"[Memory] gc frontmatter read failed for {path.name}: {_e}")
            if not (stale_age or cwd_gone):
                continue
            reason_parts = []
            if stale_age:
                reason_parts.append("stale_age")
            if cwd_gone:
                reason_parts.append("cwd_gone")
            entry = {
                "name": path.name,
                "path": str(path),
                "cwd": stored_cwd,
                "age_days": age_days,
                "reason": "+".join(reason_parts),
                "deleted": False,
            }
            if apply:
                try:
                    # Acquire the same per-file write lock that ``_save_md``
                    # uses, so a concurrent cascade-extract write doesn't
                    # race the unlink. Non-blocking to avoid hanging gc on
                    # a stuck writer; we just skip and report.
                    write_lock = _get_file_write_lock(path)
                    if write_lock.acquire(blocking=False):
                        try:
                            path.unlink(missing_ok=True)
                            entry["deleted"] = True
                        finally:
                            write_lock.release()
                    else:
                        errors.append({"path": str(path),
                                       "err": "write lock busy, skipped"})
                except Exception as _e:
                    errors.append({"path": str(path), "err": str(_e)})
                    _debug_log(f"[Memory] gc unlink failed for {path.name}: {_e}")
            candidates.append(entry)
        except Exception as _e:
            errors.append({"path": str(path), "err": f"scan: {_e}"})
            _debug_log(f"[Memory] gc scan failed for {path.name}: {_e}")

    _debug_log(
        f"[Memory] gc{'(apply)' if apply else '(dry-run)'} "
        f"scanned={scanned} candidates={len(candidates)} "
        f"deleted={sum(1 for c in candidates if c['deleted'])} "
        f"errors={len(errors)} threshold_days={threshold_days}"
    )
    return {
        "threshold_days": threshold_days,
        "apply": apply,
        "scanned": scanned,
        "candidates": candidates,
        "errors": errors,
    }


def load_memory(chat_id: str) -> str | None:
    """Load session memory. Transparently migrates from old DATA_DIR location."""
    content = _load_md_body(_session_memory_file(chat_id))
    if content is None:
        old = _cfg.DATA_DIR / "memory" / f"{chat_id}.md"
        content = _load_md_body(old)
        if content is not None:
            save_memory(chat_id, content)
            try:
                old.unlink(missing_ok=True)
            except Exception as e:
                _debug_log(f"[Memory] session file cleanup failed: {e}")
    return content


def save_memory(chat_id: str, content: str) -> None:
    turns = _get_turn_count(chat_id)
    _save_md(_session_memory_file(chat_id), content, SESSION_MAX_CHARS,
             f"chat_id: {chat_id}\nturns: {turns}\nversion: 1\n")


# ── Context assembly ──────────────────────────────────────────────────────────

def get_memory_context(chat_id: str, cwd: str | None = None) -> str:
    """Build the combined memory context string for injection as extra_system.

    Enforces TOTAL_MEMORY_BUDGET (tag overhead counted): each layer is trimmed
    proportionally when the combined total exceeds the budget.
    Returns empty string when no memory is active.
    """
    parts: list[tuple[str, str, str]] = []

    g = load_global_memory(chat_id)
    if g:
        parts.append(("[GLOBAL MEMORY]", g, "[/GLOBAL MEMORY]"))

    if cwd:
        p = load_project_memory(cwd)
        if p:
            parts.append((f"[PROJECT MEMORY — {cwd}]", p, "[/PROJECT MEMORY]"))

    s = load_memory(chat_id)
    if s:
        parts.append(("[SESSION MEMORY]", s, "[/SESSION MEMORY]"))

    if not parts:
        return ""

    total = sum(len(c) + _TAG_OVERHEAD_PER_LAYER for _, c, _ in parts)
    if total > TOTAL_MEMORY_BUDGET:
        available = max(0, TOTAL_MEMORY_BUDGET - _TAG_OVERHEAD_PER_LAYER * len(parts))
        content_total = sum(len(c) for _, c, _ in parts)
        if content_total > 0:
            _debug_log(f"[Memory] budget trim: total={total} > {TOTAL_MEMORY_BUDGET}, available={available}")
            for i, (open_tag, content, close_tag) in enumerate(parts):
                budget_i = int(available * len(content) / content_total)
                if len(content) > budget_i:
                    parts[i] = (open_tag, content[:budget_i] + "…", close_tag)

    return "\n\n".join(f"{o}\n{c}\n{cl}" for o, c, cl in parts)


def get_project_memory_context(chat_id: str, cwd: str | None = None) -> str:
    """Build project + session memory context (no global layer).

    Used by crew agents for task-scoped context. Global layer is intentionally
    excluded to keep the function behaviour predictable.
    """
    parts: list[str] = []
    if cwd:
        p = load_project_memory(cwd)
        if p:
            parts.append(f"[PROJECT MEMORY — {cwd}]\n{p}\n[/PROJECT MEMORY]")
    s = load_memory(chat_id)
    if s:
        parts.append(f"[SESSION MEMORY]\n{s}\n[/SESSION MEMORY]")
    return "\n\n".join(parts) if parts else ""


def inject_memory(chat_id: str, message: str, cwd: str | None = None) -> str:
    """Prepend all active memory layers to the message.

    DEPRECATED — legacy compatibility shim. New code should call
    get_memory_context() and pass the result as extra_system.
    """
    ctx = get_memory_context(chat_id, cwd)
    if not ctx:
        return message
    return ctx + "\n\n" + message


# ── LLM one-shot helper ───────────────────────────────────────────────────────

_API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api")


def _run_one_shot(prompt: str, ns: str) -> str:
    """Run a single stateless LLM prompt and return the text output.

    Uses the orchestrator backend. ns is an isolated chat namespace so the call
    never touches any real chat's session state.
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY
    from larkhelm.perm import grant_yolo, revoke_yolo

    spec = BACKEND_REGISTRY.get_orchestrator()
    if spec is None:
        raise RuntimeError("No orchestrator backend available")

    collected: list[str] = []

    def _on_text(text: str, status: str = "typing") -> None:
        collected.clear()
        collected.append(text)

    try:
        if spec.provider in _API_PROVIDERS:
            import larkhelm.backend_api as _bapi
            fn = {
                "anthropic_api":    _bapi.run_anthropic,
                "google_api":       _bapi.run_google,
                "openai_compat_api": _bapi.run_openai_compat,
            }[spec.provider]
            output, _ = fn(spec=spec, chat_id=ns, message=prompt, history=[], on_text=_on_text)
        else:
            from larkhelm.backend_cli import run_claude, run_gemini, run_kimi
            grant_yolo(ns)
            try:
                if spec.provider == "gemini_cli":
                    output = run_gemini(spec=spec, chat_id=ns, message=prompt,
                                        sid=None, cwd=str(_cfg.DATA_DIR),
                                        on_text=_on_text, use_session=False)
                elif spec.provider == "kimi_cli":
                    output = run_kimi(spec=spec, chat_id=ns, message=prompt,
                                      sid=None, cwd=str(_cfg.DATA_DIR), on_text=_on_text)
                else:
                    output = run_claude(spec=spec, chat_id=ns, message=prompt,
                                        sid=None, cwd=str(_cfg.DATA_DIR), on_text=_on_text)
            finally:
                revoke_yolo(ns)
        return output or "".join(collected)
    finally:
        try:
            from larkhelm.chat_state import _clear_sid
            _clear_sid(ns, spec.id)
        except Exception as e:
            _debug_log(f"[Memory] clear_sid failed: {e}")


# ── Session memory generation ─────────────────────────────────────────────────

# Refusal/empty-output prefixes commonly emitted by LLMs when the input is
# unsuitable (rate-limited, content-policy, hallucinated apology). Comparing
# in lower-case + against the leading 80 chars catches variants like
# "I cannot fulfill this request" / "As an AI language model, I…" /
# "I'm sorry, but…". Anything matching is dropped — keeping the previous
# memory is strictly better than overwriting with garbage.
_USELESS_SUMMARY_PREFIXES: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "i'm afraid",
    "i'm unable",
    "i am unable",
    "unable to",
    "my apologies",
    "as an ai",
    "as a language model",
    "sorry, but",
    "抱歉",
    "很抱歉",
    "对不起",
    "我无法",
    "作为一个ai",
    "作为 ai",
)
_MIN_USEFUL_SUMMARY_CHARS = 50


def _is_useful_summary(text: str | None) -> bool:
    """Return True iff ``text`` looks like a real memory summary.

    Rejects: ``None``, empty/whitespace, too short (< 50 chars after strip),
    or output starting with a known refusal/apology prefix. Used by
    ``generate_memory`` and ``_try_extract_*`` so an LLM hiccup never
    overwrites a good memory file with garbage.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < _MIN_USEFUL_SUMMARY_CHARS:
        return False
    head_lower = stripped[:80].lower()
    for bad in _USELESS_SUMMARY_PREFIXES:
        if head_lower.startswith(bad):
            return False
    return True


def generate_memory(chat_id: str, recent_logs: str,
                    existing_memory: str | None = None) -> str:
    """Generate a session memory summary. Returns Markdown (≤SESSION_MAX_CHARS).

    Raises ``ValueError`` when the LLM output fails ``_is_useful_summary``
    (empty / refusal prefix / too short). The caller (``maybe_auto_update``)
    treats this as a soft failure and keeps the previous session memory
    unchanged, preventing pollution of subsequent ``existing_memory``
    inputs in a degenerative feedback loop.
    """
    if existing_memory:
        existing_memory_section = (
            f"---EXISTING MEMORY (preserve important facts and update)---\n"
            f"{existing_memory}\n"
            f"---END EXISTING MEMORY---\n\n"
        )
    else:
        existing_memory_section = ""

    prompt = _SUMMARIZE_PROMPT.format(
        max_chars=SESSION_MAX_CHARS,
        existing_memory_section=existing_memory_section,
        logs=recent_logs[:10000],
    )
    try:
        result = _run_one_shot(prompt, ns=f"_mem_{chat_id}")
    except Exception as e:
        _debug_log(f"[Memory] generate_memory error {chat_id}: {e}")
        raise
    if not _is_useful_summary(result):
        head = (result or "").strip()[:80].replace("\n", " ")
        _debug_log(
            f"[Memory] rejected non-useful summary for {chat_id[:8]} "
            f"(len={len(result or '')}, head={head!r}); keeping previous memory"
        )
        raise ValueError("non-useful summary output")
    return result[:SESSION_MAX_CHARS]


# ── Cascade extraction (project + global auto-learning) ──────────────────────

def _try_extract_project(session_content: str, cwd: str) -> None:
    """Extract project facts from a fresh session summary → update project layer if new info found.

    Runs in a background daemon thread. Writes only when the LLM finds genuinely new
    information (output != "UNCHANGED"). Safe to call concurrently; file write lock serialises.
    """
    try:
        existing = load_project_memory(cwd) or "(empty)"
        prompt = _EXTRACT_PROJECT_PROMPT.format(
            max_chars=PROJECT_MAX_CHARS,
            existing=existing,
            session=session_content,
        )
        ns = f"_proj_{hashlib.md5(cwd.encode()).hexdigest()[:8]}"
        result = _run_one_shot(prompt, ns=ns)
        result = (result or "").strip()
        if not result or result.upper() == "UNCHANGED":
            return
        if not _is_useful_summary(result):
            _debug_log(f"[Memory] project extract rejected non-useful output for {cwd!r}")
            return
        save_project_memory(cwd, result)
        _debug_log(f"[Memory] project layer auto-updated from session cascade ({len(result)} chars)")
    except Exception as e:
        _debug_log(f"[Memory] extract_project error for {cwd!r}: {e}")


def _try_extract_global(session_content: str, chat_id: str) -> None:
    """Extract user preferences from a fresh session summary → update global layer if new info found."""
    try:
        g_path = _global_memory_file(chat_id)
        if g_path is None:
            return  # no open_id (group chat) — skip global layer
        existing = _load_md_body(g_path) or "(empty)"
        prompt = _EXTRACT_GLOBAL_PROMPT.format(
            max_chars=GLOBAL_MAX_CHARS,
            existing=existing,
            session=session_content,
        )
        ns = f"_glob_{chat_id[:8]}"
        result = _run_one_shot(prompt, ns=ns)
        result = (result or "").strip()
        if not result or result.upper() == "UNCHANGED":
            return
        if not _is_useful_summary(result):
            _debug_log(f"[Memory] global extract rejected non-useful output for {chat_id[:8]}")
            return
        save_global_memory(result, chat_id=chat_id)
        _debug_log(f"[Memory] global layer auto-updated from session cascade ({len(result)} chars)")
    except Exception as e:
        _debug_log(f"[Memory] extract_global error for {chat_id[:8]}: {e}")


def _cascade_extract(session_content: str, chat_id: str) -> None:
    """Launch background threads to extract project and global facts from a fresh session summary.

    Each extraction thread is guarded by _EXTRACT_TIMEOUT so a slow LLM call cannot
    leak daemon threads indefinitely.
    """
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(chat_id)
    except Exception:
        cwd = None

    def _guarded(target, args, name):
        t = threading.Thread(target=target, args=args, daemon=True, name=name)
        t.start()
        t.join(timeout=_EXTRACT_TIMEOUT)
        if t.is_alive():
            _debug_log(f"[Memory] {name} extraction timed out ({_EXTRACT_TIMEOUT}s), daemon thread abandoned")

    def _run_cascade():
        threads = []
        if cwd:
            threads.append(threading.Thread(
                target=_guarded,
                args=(_try_extract_project, (session_content, cwd), f"memext-proj-{chat_id[:8]}"),
                daemon=True,
            ))
        threads.append(threading.Thread(
            target=_guarded,
            args=(_try_extract_global, (session_content, chat_id), f"memext-glob-{chat_id[:8]}"),
            daemon=True,
        ))
        for t in threads:
            t.start()

    threading.Thread(target=_run_cascade, daemon=True, name=f"memcascade-{chat_id[:8]}").start()


# ── Auto-update (session layer + cascade) ────────────────────────────────────

def maybe_auto_update(chat_id: str, force: bool = False,
                      on_done: Callable[[bool, str | None, str | None], None] | None = None,
                      ) -> None:
    """Check if session memory needs updating and run in a background thread if so.

    Triggers when turn_count % AUTO_UPDATE_EVERY == 0 (or force=True).
    After a successful session update, automatically cascades to extract new facts
    into project and global memory layers (background, non-blocking).

    on_done: optional callback(success, content, error_code)
    """
    turn_count = _get_turn_count(chat_id)
    if not force and (turn_count == 0 or turn_count % AUTO_UPDATE_EVERY != 0):
        return

    def _notify(success: bool, content: str | None, error: str | None) -> None:
        if on_done:
            try:
                on_done(success, content, error)
            except Exception as _cb_err:
                _debug_log(f"[Memory] on_done callback error: {_cb_err}")

    def _run():
        lock = _get_update_lock(chat_id)
        if not lock.acquire(blocking=False):
            _debug_log(f"[Memory] update already in progress for {chat_id[:8]}, skipping")
            _notify(False, None, "already_in_progress")
            return
        try:
            logs = _read_logs(chat_id)
            if not logs:
                _notify(False, None, "no_logs")
                return
            recent = logs[-50:]
            # Whitelist roles that semantically describe "what happened in
            # this session". The "milestone" role is added by
            # ``record_milestone`` and represents the completion of a
            # /dev /crew /plan task — without including it the LLM
            # summarizer never sees those events even when the user
            # invokes ``maybe_auto_update`` right after.
            #
            # ``model`` exclusion drops crew/dev/shell sub-task chatter
            # (which is voluminous and not summary-worthy); milestone
            # records use ``model="milestone"`` precisely so they pass
            # this filter.
            log_text = "\n".join(
                f"[{r['ts']}] {r['role']}: {r['content'][:600]}"
                for r in recent
                if r["role"] in ("user", "assistant", "milestone")
                and r.get("model") not in ("crew", "shell")
            )
            if not log_text.strip():
                _notify(False, None, "no_conversation_logs")
                return

            existing = load_memory(chat_id)

            result: list[str | None] = [None]
            err: list[Exception | None] = [None]

            def _gen():
                try:
                    result[0] = generate_memory(chat_id, log_text, existing_memory=existing)
                except Exception as e:
                    err[0] = e

            gen_t = threading.Thread(target=_gen, daemon=True, name=f"memgen-{chat_id[:8]}")
            gen_t.start()
            gen_t.join(timeout=MEMORY_GENERATION_TIMEOUT)
            if gen_t.is_alive():
                _debug_log(f"[Memory] generate_memory timed out ({MEMORY_GENERATION_TIMEOUT}s) for {chat_id[:8]}")
                _notify(False, None, f"timed_out_{MEMORY_GENERATION_TIMEOUT}s")
                return
            if err[0]:
                raise err[0]

            save_memory(chat_id, result[0])
            _notify(True, result[0], None)

            # Cascade: auto-extract project and global facts from the fresh session summary.
            # Runs in separate daemon threads — does not block the caller or the on_done callback.
            _cascade_extract(result[0], chat_id)

        except Exception as e:
            _debug_log(f"[Memory] maybe_auto_update error {chat_id}: {e}")
            _notify(False, None, str(e))
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True, name=f"memory-{chat_id[:8]}").start()


# ── Milestone hook (post-/dev /crew /plan completion) ────────────────────────

# Debounce: don't refresh memory more than once per chat per N seconds, even
# if multiple milestones fire close together. Cheap-LLM cost is bounded and
# the summarizer has nothing useful to say about deltas <60s apart anyway.
_MILESTONE_DEBOUNCE_SEC = 60
_last_milestone_ts: dict[str, float] = {}
_milestone_meta = threading.Lock()


def record_milestone(chat_id: str, kind: str, summary: str = "") -> None:
    """Record a task milestone and (debounced) trigger a session memory refresh.

    Called from the finally blocks of /dev (`_run_dev_crew_inner`),
    /crew (`_run_generic_crew_inner`) and /plan (`_run_plan`).

    Two responsibilities:

    1. Append a ``role="milestone"`` entry to the conversation log so the
       summarizer can include it the next time it runs (and so /history
       readers can see what big tasks happened).
    2. Force-trigger ``maybe_auto_update`` so the session memory captures
       the milestone immediately, instead of waiting up to 10 chat turns
       for the next ``_do_query`` to fire the regular auto-update.

    Failures swallowed: this is opportunistic — never break the milestone
    task itself.

    Debounce: if another milestone fired within ``_MILESTONE_DEBOUNCE_SEC``
    seconds, the memory regeneration is skipped (the log entry is still
    written so the next regenerate sees both events). Prevents pile-up
    when /plan runs many /dev steps back-to-back.
    """
    import time as _time
    msg = f"[Milestone] {kind}"
    if summary:
        msg += f": {summary[:200]}"

    # 1. Always log the milestone (cheap, helpful for /history and next regen).
    try:
        from larkhelm.log import log_entry
        log_entry(chat_id, "milestone", msg, model="milestone")
    except Exception as e:
        _debug_log(f"[Memory] milestone log_entry failed: {e}")

    # 2. Debounced force-refresh.
    now = _time.time()
    with _milestone_meta:
        last = _last_milestone_ts.get(chat_id, 0.0)
        if now - last < _MILESTONE_DEBOUNCE_SEC:
            _debug_log(
                f"[Memory] milestone {kind} debounced for {chat_id[:8]} "
                f"(last update {now - last:.1f}s ago)"
            )
            return
        _last_milestone_ts[chat_id] = now

    try:
        maybe_auto_update(chat_id, force=True)
    except Exception as e:
        _debug_log(f"[Memory] milestone trigger failed: {e}")
