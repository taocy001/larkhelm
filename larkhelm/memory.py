"""larkhelm · three-tier persistent memory

Storage: ~/.larkhelm/memory/
  global_{open_id}.md    — user-level: preferences, style, cross-project habits (≤800 chars)
                           keyed by sender_open_id; returns None when open_id unknown (group safety)
  project_{hash16}.md    — project-level: tech stack, conventions, keyed by resolved cwd (≤1500 chars)
  session_{chat_id}.md   — session-level: current work, decisions, context (≤2000 chars)
                           first auto-update at turn 3, then every 10 turns; cascades to project/global after each update

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
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import larkhelm.config as _cfg
from larkhelm.chat_state import _get_turn_count, _get_chat_state
from larkhelm.log import _read_logs, _read_logs_tail, _debug_log

# ── Storage root ─────────────────────────────────────────────────────────────

MEMORY_HOME_DIR = Path.home() / ".larkhelm" / "memory"

# ── Per-layer size budgets ────────────────────────────────────────────────────

GLOBAL_MAX_CHARS  = 800
PROJECT_MAX_CHARS = 1500
SESSION_MAX_CHARS = 2000
TOTAL_MEMORY_BUDGET = 4500  # combined cap; tag overhead counted separately

# ── Schema version (S47) ──────────────────────────────────────────────────────
#
# Bumped when the on-disk layout of a memory file changes in a way that
# requires the loader to migrate or refuse to read it. ``_save_md`` writes
# this as ``schema_version: "N"`` in the frontmatter; ``_load_md_frontmatter``
# returns it like any other field, and ``_check_schema_version`` warns when
# a file is newer than the binary supports.
#
# History:
#   1 — initial layout (frontmatter + body, used since pre-Phase B)
MEMORY_SCHEMA_VERSION = "1"

# Anchored at line start to avoid matching keys that merely *contain* the
# substring "schema_version" (e.g. last_schema_version_check, my_schema_version_note).
_SCHEMA_KEY_RE = re.compile(r'(?m)^schema_version\s*:')

# Bumped from 50 → 90 to reserve room for the per-layer meter line injected by
# ``get_memory_context()`` (e.g. ``[1850/2000 chars, 92%] ⚠️ near limit`` is
# ~36 chars; 90 leaves slack for unicode + newlines). TOTAL_MEMORY_BUDGET is
# unchanged so user content is preserved as before — only the bookkeeping
# allowance grows.
_TAG_OVERHEAD_PER_LAYER = 90

# ── /memory observe tuning ───────────────────────────────────────────────────

_OBSERVE_TAIL_BYTES        = 1 * 1024 * 1024   # all.jsonl tail window (1 MiB)
_OBSERVE_DEBUG_TAIL_BYTES  = 2 * 1024 * 1024   # DEBUG_LOG tail window (2 MiB)
_OBSERVE_WINDOW_DAYS       = 7
_NEAR_LIMIT_PCT            = 90

_CHEAP_FAIL_PAT = re.compile(r"\[Memory\] cheap backend .+? failed")
_UNCHANGED_PAT  = re.compile(
    r"\[Memory\] (?:project|global) extract rejected non-useful output"
    r"|\[Memory\] rejected non-useful summary"
)
_SAVE_OK_PAT    = re.compile(r"\[Memory\] saved session_[^\s]+\.md \(\d+ chars\)")
_TS_PAT         = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]")

# ── Auto-update tuning ────────────────────────────────────────────────────────

AUTO_UPDATE_EVERY         = 10   # session auto-update cadence after the first hit
AUTO_UPDATE_FIRST         = 3    # first-time threshold: trigger once at turn 3, then every AUTO_UPDATE_EVERY thereafter
MEMORY_GENERATION_TIMEOUT = 120  # seconds before abandoning a slow generate call
_EXTRACT_TIMEOUT          = 60   # shorter timeout for cascade extraction calls


def _should_auto_update(turn_count: int) -> bool:
    """True iff a non-forced auto-update should fire at this ``turn_count``.

    Fires at ``AUTO_UPDATE_FIRST`` (=3), then every ``AUTO_UPDATE_EVERY`` (=10)
    turns thereafter — i.e. turns 3, 13, 23, 33, …  Pure function; testable
    without touching state.
    """
    if turn_count == AUTO_UPDATE_FIRST:
        return True
    if turn_count > AUTO_UPDATE_FIRST and (turn_count - AUTO_UPDATE_FIRST) % AUTO_UPDATE_EVERY == 0:
        return True
    return False

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

# In-process body cache, invalidated on mtime change. Every ``_do_query`` reads
# all three layers via ``get_memory_context`` and the auto-update cascade reads
# them again inside ``_try_extract_*`` — that's 6+ disk opens per active turn.
# Caching cuts that to 3 opens for the first read and stat-only afterward
# (``path.stat`` is ~10× cheaper than open+read+strip-frontmatter).
#
# Invalidation contract:
#   * ``_save_md`` rewrites the file → mtime changes → next ``_load_md_body``
#     re-reads automatically.
#   * Other processes editing memory files manually (eg. ``larkhelm memory
#     import``) likewise touch mtime, so cross-process consistency is
#     preserved within mtime resolution (most filesystems: nanoseconds).
_mem_body_cache: dict[str, tuple[float, str | None]] = {}
_mem_body_cache_lock = threading.Lock()


def _load_md_body(path: Path | None) -> str | None:
    """Read a memory file body, stripping YAML frontmatter. Returns None if absent/empty.

    Uses an in-process mtime-keyed cache so repeated reads of the same file
    within a turn (or across turns when memory hasn't been rewritten) avoid
    re-opening the file. See module-level note above for the contract.
    """
    if path is None:
        return None
    try:
        if not path.exists():
            # Drop any stale entry for a now-deleted file so a future re-create
            # is observed on the next call.
            with _mem_body_cache_lock:
                _mem_body_cache.pop(str(path), None)
            return None
        st_mtime = path.stat().st_mtime
        key = str(path)
        with _mem_body_cache_lock:
            cached = _mem_body_cache.get(key)
            if cached is not None and cached[0] == st_mtime:
                return cached[1]
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                body = text[end + 4:].strip() or None
            else:
                body = text.strip() or None
        else:
            body = text.strip() or None
        with _mem_body_cache_lock:
            _mem_body_cache[key] = (st_mtime, body)
        return body
    except Exception as e:
        _debug_log(f"[Memory] load error {path.name}: {e}")
        return None


def _load_md_frontmatter(path: Path | None) -> dict[str, str]:
    """Parse YAML-like frontmatter key: value pairs from a memory file.

    Caller can read the ``schema_version`` field directly; ``_check_schema_version``
    is also available for a quick "is this file from a newer binary?" check.
    """
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


# Track which file paths we've already logged a schema warning for so the
# debug log doesn't fill with the same line every cascade tick. Lifetime
# matches the process; restart clears it (acceptable — operator gets a
# fresh warning after restart, which is signal not noise).
_SCHEMA_WARN_SEEN: set[str] = set()


def _check_schema_version(path: Path | None, fm: dict[str, str]) -> str:
    """Return the file's declared ``schema_version`` (default ``"1"``).

    Logs a one-shot warning when the file is *newer* than this binary
    understands. Files at or below the binary version are silently
    accepted — older files implicitly upgrade on next ``_save_md`` since
    we always re-stamp ``schema_version`` on write.
    """
    v = (fm.get("schema_version") or "1").strip()
    try:
        if path is not None and int(v) > int(MEMORY_SCHEMA_VERSION):
            key = str(path)
            if key not in _SCHEMA_WARN_SEEN:
                _SCHEMA_WARN_SEEN.add(key)
                _debug_log(
                    f"[Memory] schema_version={v} on {path.name} is newer than "
                    f"this binary supports ({MEMORY_SCHEMA_VERSION}); "
                    f"reading anyway, fields may be ignored"
                )
    except (TypeError, ValueError):
        # schema_version isn't numeric — treat as v1 silently. Future
        # versions can switch to semver if needed.
        pass
    return v


def _save_md(
    path: Path | None,
    content: str,
    max_chars: int,
    extra_fm: str = "",
    extra_fm_pairs: dict[str, str] | None = None,
) -> None:
    """Atomically write a memory file with YAML frontmatter, protected by a per-file lock.

    ``extra_fm_pairs`` is the structured form of ``extra_fm``: callers pass a
    ``{key: value}`` dict and the function renders each as a quoted YAML pair.
    The two arguments coexist (legacy callers still pass ``extra_fm`` strings).

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
            extra_pairs_text = ""
            if extra_fm_pairs:
                for k, v in extra_fm_pairs.items():
                    safe_v = str(v).replace('"', '\\"')
                    extra_pairs_text += f'{k}: "{safe_v}"\n'
            # S47: schema_version is written by every save so future loaders
            # can detect old / forward-incompatible files. ``extra_fm`` and
            # ``extra_fm_pairs`` retain precedence — a caller that explicitly
            # passes ``schema_version`` in either wins (used by migration
            # tools that re-stamp an older file at its declared version).
            #
            # The presence check must be ANCHORED at line start: a plain
            # substring search in ``extra_fm`` mis-fires on legitimate
            # keys that contain "schema_version" as a substring (e.g.
            # ``last_schema_version_check: ...``), suppressing the
            # auto-stamp and leaving the file with no version line.
            schema_line = ""
            has_in_pairs = "schema_version" in (extra_fm_pairs or {})
            has_in_extra = bool(_SCHEMA_KEY_RE.search(extra_fm))
            if not has_in_pairs and not has_in_extra:
                schema_line = f'schema_version: "{MEMORY_SCHEMA_VERSION}"\n'
            fm = f'---\nupdated_at: "{now}"\n{schema_line}{extra_fm}{extra_pairs_text}---\n\n'
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
    path = _global_memory_file(chat_id)
    content = _load_md_body(path)
    if content is not None:
        _check_schema_version(path, _load_md_frontmatter(path))
    return content


def save_global_memory(content: str, chat_id: str | None = None,
                       extra_fm_pairs: dict[str, str] | None = None) -> None:
    _save_md(_global_memory_file(chat_id), content, GLOBAL_MAX_CHARS,
             extra_fm_pairs=extra_fm_pairs)


def load_project_memory(cwd: str) -> str | None:
    path = _project_memory_file(cwd)
    content = _load_md_body(path)
    if content is None:
        return None
    fm = _load_md_frontmatter(path)
    _check_schema_version(path, fm)
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


def save_project_memory(cwd: str, content: str,
                        extra_fm_pairs: dict[str, str] | None = None) -> None:
    canonical = str(Path(cwd).resolve())
    _save_md(_project_memory_file(cwd), content, PROJECT_MAX_CHARS,
             f'cwd: "{canonical}"\n', extra_fm_pairs=extra_fm_pairs)


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
                # GC reads + may delete the file; we want to know about
                # forward-incompatible files BEFORE deciding to unlink one
                # the binary doesn't understand. The one-shot warning per
                # path keeps log noise bounded.
                _check_schema_version(path, fm)
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
    path = _session_memory_file(chat_id)
    content = _load_md_body(path)
    if content is not None:
        _check_schema_version(path, _load_md_frontmatter(path))
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

    Phase B (S49–S52) forwards to ``MemoryContextBuilder``. Default arguments
    (no ``query``, no ``recent_turns``) hit the fail-open paths in
    ``should_include_*`` so behaviour is byte-equivalent to the legacy
    implementation: every layer is included (subject to budget trim).
    """
    from larkhelm.memory_context import MemoryContextBuilder
    return MemoryContextBuilder(chat_id, cwd).build()


def get_memory_context_v2(
    chat_id: str,
    cwd: str | None = None,
    *,
    query: str = "",
    recent_turns: list[str] | None = None,
    has_doc_urls: bool = False,
    intent=None,
) -> tuple[str, list[str]]:
    """Build memory context AND return deduped recent turns in one pass.

    Returns ``(composed_memory, deduped_recent_turns)``. ``recent_turns`` is
    deduped against the *raw* session body (so dedup remains correct even
    when the session view is sliced down by ``memory_session_layered``).

    Phase D — ``intent``: when an :class:`IntentResult`-shaped object is
    supplied, its ``agent_type`` / ``sub_intent`` / ``complexity`` /
    ``confidence`` fields are forwarded to the builder so the retriever
    path (gated by ``memory_retriever_enabled``) can apply per-agent
    policy. When ``intent is None`` the call is byte-equivalent to the
    legacy v2 signature; this is duck-typed so third-party plugins can
    pass any namespace object exposing the four fields."""
    from larkhelm.memory_context import MemoryContextBuilder

    builder_kwargs: dict = dict(
        query=query, recent_turns=recent_turns,
        has_doc_urls=has_doc_urls,
    )
    if intent is not None:
        builder_kwargs["agent_type"] = getattr(intent, "agent_type", "chat") or "chat"
        builder_kwargs["sub_intent"] = getattr(intent, "sub_intent", "") or ""
        builder_kwargs["complexity"] = getattr(intent, "complexity", "medium") or "medium"
        try:
            builder_kwargs["confidence"] = float(getattr(intent, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            builder_kwargs["confidence"] = 0.0

    builder = MemoryContextBuilder(chat_id, cwd, **builder_kwargs)
    composed = builder.build()
    session_raw = load_memory(chat_id) or ""
    deduped = builder.deduped_recent_turns(session_raw)
    return composed, deduped


def get_project_memory_context(chat_id: str, cwd: str | None = None) -> str:
    """Build project + session memory context (no global layer).

    Used by crew agents for task-scoped context. Phase B forwards to
    ``MemoryContextBuilder.build_for_crew`` so per-layer changes (eg.
    project memory frontmatter) are applied uniformly.
    """
    from larkhelm.memory_context import MemoryContextBuilder
    return MemoryContextBuilder(chat_id, cwd, force_project=True).build_for_crew()


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

# Backends whose runner is reached via ``backend_api`` (SDK-based stream APIs).
# Note ``deepseek_api`` is NOT here — DeepSeek's runner historically lives in
# ``backend_cli`` (it's HTTP but predates the API/CLI split), so it gets its
# own branch in ``_dispatch_one_shot`` below.
_SDK_API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api")


def _dispatch_one_shot(spec, ns: str, prompt: str, on_text) -> str:
    """Invoke a one-shot LLM call against ``spec``; raises on backend error.

    Extracted from ``_run_one_shot`` so the retry-on-failure path (cheap →
    orchestrator) can call it twice without duplicating the dispatch tree.
    Cleans up the session sid in ``finally`` so a backend mid-call crash
    doesn't leak namespace state.
    """
    from larkhelm.perm import grant_yolo, revoke_yolo

    try:
        if spec.provider in _SDK_API_PROVIDERS:
            import larkhelm.backend_api as _bapi
            fn = {
                "anthropic_api":    _bapi.run_anthropic,
                "google_api":       _bapi.run_google,
                "openai_compat_api": _bapi.run_openai_compat,
            }[spec.provider]
            output, _ = fn(spec=spec, chat_id=ns, message=prompt,
                           history=[], on_text=on_text)
        elif spec.provider == "deepseek_api":
            # DeepSeek's runner lives in backend_cli (HTTP-via-requests, no SDK).
            # Routing it through the CLI ``else`` branch by accident used to
            # spawn ``run_claude`` with a DeepSeek spec — exactly the bug the
            # post-#3 audit caught.
            from larkhelm.backend_cli import run_deepseek
            output = run_deepseek(spec=spec, chat_id=ns, message=prompt,
                                  sid=None, cwd=str(_cfg.DATA_DIR),
                                  on_text=on_text)
        else:
            from larkhelm.backend_cli import run_claude, run_gemini, run_kimi
            grant_yolo(ns)
            try:
                if spec.provider == "gemini_cli":
                    output = run_gemini(spec=spec, chat_id=ns, message=prompt,
                                        sid=None, cwd=str(_cfg.DATA_DIR),
                                        on_text=on_text, use_session=False)
                elif spec.provider == "kimi_cli":
                    output = run_kimi(spec=spec, chat_id=ns, message=prompt,
                                      sid=None, cwd=str(_cfg.DATA_DIR), on_text=on_text)
                else:
                    output = run_claude(spec=spec, chat_id=ns, message=prompt,
                                        sid=None, cwd=str(_cfg.DATA_DIR), on_text=on_text)
            finally:
                revoke_yolo(ns)
        return output
    finally:
        try:
            from larkhelm.chat_state import _clear_sid
            _clear_sid(ns, spec.id)
        except Exception as e:
            _debug_log(f"[Memory] clear_sid failed: {e}")


def _run_one_shot(prompt: str, ns: str, prefer_cheap: bool = False,
                  cancel_ev: "threading.Event | None" = None) -> str:
    """Run a single stateless LLM prompt and return the text output.

    Backend selection:
      * ``prefer_cheap=True`` — try a ``tags=["cheap"]`` backend first
        (typically DeepSeek; see ``config._auto_discover_http``). On runtime
        failure, **falls back to the orchestrator with a single retry** so a
        DeepSeek quota / network hiccup doesn't break the cascade. Used by
        the memory cascade where the marginal quality cost of cheap is small
        but the price ratio is ~30× cheaper.
      * ``prefer_cheap=False`` (default) — orchestrator only; preserves
        legacy behaviour. No fallback — any failure propagates so the caller
        can mark the orchestrator unhealthy.

    ns is an isolated chat namespace so the call never touches any real
    chat's session state.
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY

    cheap_spec = None
    if prefer_cheap:
        cheap_spec = BACKEND_REGISTRY.get_by_tag(["cheap"])
        if cheap_spec is not None:
            _debug_log(f"[Memory] one_shot using cheap backend {cheap_spec.id} (ns={ns})")

    spec = cheap_spec or BACKEND_REGISTRY.get_orchestrator()
    if spec is None:
        raise RuntimeError("No backend available (neither cheap nor orchestrator)")

    collected: list[str] = []

    # P1-5: if a cancel_ev is supplied and midflight cancel is enabled,
    # wire the check into _on_text so the LLM stream aborts as soon as
    # the next chat-turn arrives, instead of waiting for the call to
    # complete.
    _midflight_check = _cascade_midflight_on_text(cancel_ev)

    def _on_text(text: str, status: str = "typing") -> None:
        if _midflight_check is not None:
            _midflight_check(text, status)
        collected.clear()
        collected.append(text)

    try:
        output = _dispatch_one_shot(spec, ns, prompt, _on_text)
        return output or "".join(collected)
    except Exception as cheap_err:
        # Runtime fallback: cheap path failed (network, quota, model error).
        # Try orchestrator once. Only kicks in when cheap was actually used —
        # avoids hiding orchestrator-direct failures from callers.
        if cheap_spec is None or spec.id != cheap_spec.id:
            raise
        orch_spec = BACKEND_REGISTRY.get_orchestrator()
        if orch_spec is None or orch_spec.id == cheap_spec.id:
            raise
        _debug_log(
            f"[Memory] cheap backend {cheap_spec.id} failed ({type(cheap_err).__name__}: "
            f"{str(cheap_err)[:120]}); retrying via orchestrator {orch_spec.id}"
        )
        collected.clear()
        output = _dispatch_one_shot(orch_spec, ns, prompt, _on_text)
        return output or "".join(collected)


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
        # ``prefer_cheap=True`` routes session summarization to DeepSeek (or
        # whichever backend is tagged ``cheap``) when healthy. Summary quality
        # for compressing 10K-char dialog → 2K-char paragraph is roughly
        # equivalent across modern LLMs but pricing differs ~30×.
        result = _run_one_shot(prompt, ns=f"_mem_{chat_id}", prefer_cheap=True)
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

# S53 hash short-circuit: a cascade with the same session_content as the
# previous successful extract can skip the LLM call entirely. We persist
# md5(session_content)[:16] + len into the project/global frontmatter and
# compare on the next call. The two-hash check is keyed by file (cwd-derived
# for project, chat_id-derived for global) so cross-chat contention is nil.

def _session_hash(session_content: str) -> str:
    return hashlib.md5((session_content or "").encode("utf-8")).hexdigest()[:16]


def _should_skip_extract_by_hash(prev_fm: dict, session_content: str,
                                 *, len_tolerance: int = 100) -> bool:
    """Return True iff the previous extract used the same session payload.

    Equivalence definition:
      * ``last_extracted_session_hash`` matches md5(session_content)[:16] AND
      * ``abs(prev_len - cur_len) < len_tolerance`` (defends against rare
        hash false-positive on near-empty inputs).

    Missing fields → False (no skip), so old memory files written before
    this feature continue to extract normally on next cascade.
    """
    if not prev_fm:
        return False
    if not _config_flag("memory_cascade_shortcircuit", True):
        return False
    prev_hash = (prev_fm.get("last_extracted_session_hash") or "").strip()
    if not prev_hash:
        return False
    cur_hash = _session_hash(session_content)
    if prev_hash != cur_hash:
        return False
    try:
        prev_len = int(prev_fm.get("last_extracted_session_len", "-1"))
    except (TypeError, ValueError):
        return False
    if prev_len < 0:
        return False
    return abs(prev_len - len(session_content)) < len_tolerance


def _config_flag(key: str, default: bool = True) -> bool:
    """Local copy of memory_context._config_flag — avoids cross-import cycle."""
    try:
        cfg = getattr(_cfg, "config", None) or {}
    except Exception:
        cfg = {}
    return bool(cfg.get(key, default))


# ── Cascade coordinator (S43) ────────────────────────────────────────────

# Global semaphore caps in-flight extract LLM calls so a busy chat can't
# spawn an unbounded number of daemon threads (token-blackhole risk). Default
# 4 mirrors typical orchestrator concurrency.


# ── Cascade observability counters (P1-5) ────────────────────────────
#
# Lightweight process-wide counters surfaced through
# :func:`get_cascade_stats` for the ``/metrics`` endpoint. These are
# advisory: they don't gate any behaviour and are intentionally not
# persisted across restarts.

from dataclasses import dataclass as _cs_dataclass


@_cs_dataclass
class CascadeStats:
    active: int = 0
    dropped_total: int = 0
    midflight_cancelled_total: int = 0


_cascade_stats = CascadeStats()
_cascade_stats_lock = threading.Lock()


def get_cascade_stats() -> dict[str, int]:
    """Return a thread-safe snapshot of cascade counters (P1-5)."""
    with _cascade_stats_lock:
        return {
            "active": _cascade_stats.active,
            "dropped_total": _cascade_stats.dropped_total,
            "midflight_cancelled_total": _cascade_stats.midflight_cancelled_total,
        }


def _cascade_inc_active() -> None:
    with _cascade_stats_lock:
        _cascade_stats.active += 1


def _cascade_dec_active() -> None:
    with _cascade_stats_lock:
        if _cascade_stats.active > 0:
            _cascade_stats.active -= 1


def _cascade_inc_dropped() -> None:
    with _cascade_stats_lock:
        _cascade_stats.dropped_total += 1


def _cascade_inc_midflight_cancelled() -> None:
    with _cascade_stats_lock:
        _cascade_stats.midflight_cancelled_total += 1


def _cascade_midflight_cancel_enabled() -> bool:
    """Whether ``on_text`` callbacks should poll ``cancel_ev`` (P1-5).

    Default ``True``; set ``memory_cascade_midflight_cancel=false`` to
    fall back to post-LLM cancel only (legacy behaviour).
    """
    try:
        cfg = getattr(_cfg, "config", None) or {}
    except Exception:
        cfg = {}
    return bool(cfg.get("memory_cascade_midflight_cancel", True))


def _cascade_midflight_on_text(cancel_ev: "threading.Event | None"):
    """Build the ``on_text`` callback shipped into ``_run_one_shot``.

    When ``cancel_ev`` is set during the LLM stream, raise
    :class:`QueryCancelledError` so the worker can abort before
    consuming more tokens / writing to disk.
    """
    if cancel_ev is None or not _cascade_midflight_cancel_enabled():
        return None

    from larkhelm.ai_runner import QueryCancelledError

    def _on_text(text: str, status: str = "typing") -> None:  # noqa: ARG001
        if cancel_ev.is_set():
            raise QueryCancelledError("cascade midflight cancelled")

    return _on_text


def _cascade_max_concurrent() -> int:
    try:
        cfg = getattr(_cfg, "config", None) or {}
        v = int(cfg.get("memory_cascade_max_concurrent", 4) or 4)
        return max(1, v)
    except Exception:
        return 4


_CASCADE_SEM: threading.BoundedSemaphore | None = None
_CASCADE_SEM_LOCK = threading.Lock()


def _get_cascade_sem() -> threading.BoundedSemaphore:
    global _CASCADE_SEM
    with _CASCADE_SEM_LOCK:
        if _CASCADE_SEM is None:
            _CASCADE_SEM = threading.BoundedSemaphore(_cascade_max_concurrent())
        return _CASCADE_SEM


# Per-chat cancel events: when a new cascade starts for chat_id X, the
# previous cascade's event is set so its workers can early-exit before
# making the (expensive) LLM call.
_active_cascade_cancels: dict[str, threading.Event] = {}
_active_cancels_lock = threading.Lock()


def _try_extract_project(session_content: str, cwd: str,
                         cancel_ev: threading.Event | None = None) -> None:
    """Extract project facts from a fresh session summary → update project layer if new info found.

    Runs in a background daemon thread. Writes only when the LLM finds genuinely new
    information (output != "UNCHANGED"). Safe to call concurrently; file write lock serialises.
    """
    try:
        if cancel_ev is not None and cancel_ev.is_set():
            _debug_log(f"[Memory] project extract cancelled before start for {cwd!r}")
            return
        proj_path = _project_memory_file(cwd)
        prev_fm = _load_md_frontmatter(proj_path)
        if _should_skip_extract_by_hash(prev_fm, session_content):
            _debug_log(
                f"[Memory] project cascade shortcircuit (hash match) for {cwd!r}"
            )
            return
        existing = load_project_memory(cwd) or "(empty)"
        prompt = _EXTRACT_PROJECT_PROMPT.format(
            max_chars=PROJECT_MAX_CHARS,
            existing=existing,
            session=session_content,
        )
        ns = f"_proj_{hashlib.md5(cwd.encode()).hexdigest()[:8]}"
        if cancel_ev is not None and cancel_ev.is_set():
            _debug_log(f"[Memory] project extract cancelled pre-LLM for {cwd!r}")
            return
        # See ``generate_memory`` — cheap-backend route applies here too.
        # 80%+ of these calls return ``UNCHANGED`` so we're predominantly
        # paying for input tokens we don't act on; using a cheap model
        # multiplies that "wasted input" by ~30× less per token.
        result = _run_one_shot(prompt, ns=ns, prefer_cheap=True, cancel_ev=cancel_ev)
        # Post-LLM cancel re-check: a newer cascade may have arrived during
        # the LLM call. Without this guard the old worker still writes its
        # (now-stale) session_hash into project frontmatter AFTER the new
        # worker started, briefly leaving the wrong hash on disk. The
        # atomic-rename in _save_md prevents corruption, but the stale hash
        # would suppress legitimate re-extracts until the next session turn.
        if cancel_ev is not None and cancel_ev.is_set():
            _debug_log(f"[Memory] project extract cancelled post-LLM for {cwd!r} (discarding result)")
            return
        result = (result or "").strip()
        if not result or result.upper() == "UNCHANGED":
            return
        if not _is_useful_summary(result):
            _debug_log(f"[Memory] project extract rejected non-useful output for {cwd!r}")
            return
        save_project_memory(cwd, result, extra_fm_pairs={
            "last_extracted_session_hash": _session_hash(session_content),
            "last_extracted_session_len":  str(len(session_content)),
        })
        _debug_log(f"[Memory] project layer auto-updated from session cascade ({len(result)} chars)")
    except Exception as e:
        # P1-5: bubble up midflight cancellation so _coordinated can
        # bump the dedicated counter. Other exceptions stay quarantined.
        from larkhelm.ai_runner import QueryCancelledError
        if isinstance(e, QueryCancelledError):
            raise
        _debug_log(f"[Memory] extract_project error for {cwd!r}: {e}")


def _try_extract_global(session_content: str, chat_id: str,
                        cancel_ev: threading.Event | None = None) -> None:
    """Extract user preferences from a fresh session summary → update global layer if new info found."""
    try:
        if cancel_ev is not None and cancel_ev.is_set():
            _debug_log(f"[Memory] global extract cancelled before start for {chat_id[:8]}")
            return
        g_path = _global_memory_file(chat_id)
        if g_path is None:
            return  # no open_id (group chat) — skip global layer
        prev_fm = _load_md_frontmatter(g_path)
        if _should_skip_extract_by_hash(prev_fm, session_content):
            _debug_log(
                f"[Memory] global cascade shortcircuit (hash match) for {chat_id[:8]}"
            )
            return
        existing = _load_md_body(g_path) or "(empty)"
        prompt = _EXTRACT_GLOBAL_PROMPT.format(
            max_chars=GLOBAL_MAX_CHARS,
            existing=existing,
            session=session_content,
        )
        ns = f"_glob_{chat_id[:8]}"
        if cancel_ev is not None and cancel_ev.is_set():
            _debug_log(f"[Memory] global extract cancelled pre-LLM for {chat_id[:8]}")
            return
        # Same reasoning as project extract (including post-LLM cancel re-check).
        result = _run_one_shot(prompt, ns=ns, prefer_cheap=True, cancel_ev=cancel_ev)
        if cancel_ev is not None and cancel_ev.is_set():
            _debug_log(f"[Memory] global extract cancelled post-LLM for {chat_id[:8]} (discarding result)")
            return
        result = (result or "").strip()
        if not result or result.upper() == "UNCHANGED":
            return
        if not _is_useful_summary(result):
            _debug_log(f"[Memory] global extract rejected non-useful output for {chat_id[:8]}")
            return
        save_global_memory(result, chat_id=chat_id, extra_fm_pairs={
            "last_extracted_session_hash": _session_hash(session_content),
            "last_extracted_session_len":  str(len(session_content)),
        })
        _debug_log(f"[Memory] global layer auto-updated from session cascade ({len(result)} chars)")
    except Exception as e:
        # P1-5: bubble up midflight cancellation so _coordinated can
        # bump the dedicated counter. Other exceptions stay quarantined.
        from larkhelm.ai_runner import QueryCancelledError
        if isinstance(e, QueryCancelledError):
            raise
        _debug_log(f"[Memory] extract_global error for {chat_id[:8]}: {e}")


def _cascade_extract(session_content: str, chat_id: str) -> None:
    """Launch background threads to extract project and global facts from a fresh session summary.

    Coordinator semantics (S43):

    1. A new cascade for ``chat_id`` cancels any previous in-flight cascade
       for the same chat (sets its cancel event so workers exit before the
       LLM call). Cancellation is best-effort; once the LLM call has begun
       it cannot be interrupted mid-flight.
    2. A global ``BoundedSemaphore`` caps in-flight extract calls to
       ``memory_cascade_max_concurrent`` (default 4). When the sem is full
       the new worker logs WARN and exits — better to drop one cascade than
       to let the daemon-thread count grow without bound.
    """
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(chat_id)
    except Exception:
        cwd = None

    # Swap the per-chat cancel event: signal old workers to bow out.
    new_ev = threading.Event()
    with _active_cancels_lock:
        old_ev = _active_cascade_cancels.get(chat_id)
        if old_ev is not None:
            old_ev.set()
        _active_cascade_cancels[chat_id] = new_ev

    def _coordinated(target, args, label: str):
        sem = _get_cascade_sem()
        if not sem.acquire(timeout=0.5):
            _debug_log(
                f"[Memory] cascade sem busy ({label}), abandoning extract for {chat_id[:8]}"
            )
            _cascade_inc_dropped()
            return
        _cascade_inc_active()
        try:
            try:
                target(*args, cancel_ev=new_ev)
            except Exception as _ce:
                # P1-5: midflight cancel raises QueryCancelledError out of
                # _run_one_shot's on_text. Map it to the dedicated counter;
                # other exceptions are already logged inside the extract
                # functions, but we still record them as drops.
                from larkhelm.ai_runner import QueryCancelledError
                if isinstance(_ce, QueryCancelledError):
                    _cascade_inc_midflight_cancelled()
                    _debug_log(
                        f"[Memory] cascade midflight cancelled ({label}) for {chat_id[:8]}"
                    )
                else:
                    _debug_log(f"[Memory] cascade {label} unexpected error: {_ce}")
        finally:
            _cascade_dec_active()
            try:
                sem.release()
            except ValueError:
                # BoundedSemaphore raises if released past initial value;
                # a bug elsewhere triggered an extra release. Log and move on.
                _debug_log(f"[Memory] cascade sem over-release on {label}")
            with _active_cancels_lock:
                # Only clear if we're still the active event (a newer cascade
                # may have already replaced us, in which case we leave it).
                if _active_cascade_cancels.get(chat_id) is new_ev:
                    _active_cascade_cancels.pop(chat_id, None)

    def _run_cascade():
        threads = []
        if cwd:
            threads.append(threading.Thread(
                target=_coordinated,
                args=(_try_extract_project, (session_content, cwd), "project"),
                daemon=True,
                name=f"memext-proj-{chat_id[:8]}",
            ))
        threads.append(threading.Thread(
            target=_coordinated,
            args=(_try_extract_global, (session_content, chat_id), "global"),
            daemon=True,
            name=f"memext-glob-{chat_id[:8]}",
        ))
        for t in threads:
            t.start()

    threading.Thread(target=_run_cascade, daemon=True, name=f"memcascade-{chat_id[:8]}").start()


# ── Auto-update (session layer + cascade) ────────────────────────────────────

def maybe_auto_update(chat_id: str, force: bool = False,
                      on_done: Callable[[bool, str | None, str | None], None] | None = None,
                      ) -> None:
    """Check if session memory needs updating and run in a background thread if so.

    Triggers at turn ``AUTO_UPDATE_FIRST`` (=3) and then every
    ``AUTO_UPDATE_EVERY`` (=10) turns thereafter — i.e. turns 3, 13, 23, … —
    or whenever ``force=True``. After a successful session update, automatically
    cascades to extract new facts into project and global memory layers
    (background, non-blocking).

    on_done: optional callback(success, content, error_code)
    """
    turn_count = _get_turn_count(chat_id)
    if not force and not _should_auto_update(turn_count):
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
            logs = _read_logs_tail(chat_id)
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


# ── /memory observe — capacity meter + summary health ─────────────────────────

def _layer_meter_line(chars: int, max_chars: int) -> str:
    """Format ``[N/M chars, X%]`` (+ ``⚠️ near limit`` when pct >= 90).

    Used both as the in-context inline meter (``get_memory_context``) and as
    the per-layer indicator in the ``/memory observe`` card. ``pct`` is integer
    division so callers can render it unambiguously; values >100% indicate the
    content was already at the budget cap before trim.
    """
    if max_chars <= 0:
        pct = 0
    else:
        pct = chars * 100 // max_chars
    base = f"[{chars}/{max_chars} chars, {pct}%]"
    if pct >= _NEAR_LIMIT_PCT:
        base += " ⚠️ near limit"
    return base


def _parse_debug_log_window(tail_bytes: int) -> dict:
    """Scan ``_cfg.DEBUG_LOG`` tail for cascade-health signals.

    Returns a dict with cheap_fail / unchanged / save_ok counters and the
    latest matched ``HH:MM:SS`` timestamps. On any I/O failure (file missing
    or unreadable) returns ``{"unavailable": True, "reason": ...}``; never
    raises. Time-window is approximated as "last ``tail_bytes`` of the file"
    — see PRD REQ-09: DEBUG_LOG carries only ``HH:MM:SS`` so a true 7-day
    filter is not feasible without log-format changes.
    """
    try:
        path = _cfg.DEBUG_LOG
    except AttributeError:
        return {"unavailable": True, "reason": "debug_log_unset"}

    try:
        if not path.exists():
            return {"unavailable": True, "reason": "debug_log_missing"}
    except Exception as e:
        _debug_log(f"[Memory] observe debug_log stat failed: {e}")
        return {"unavailable": True, "reason": "debug_log_stat_failed"}

    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size > tail_bytes:
                f.seek(size - tail_bytes)
            else:
                f.seek(0)
            raw = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        _debug_log(f"[Memory] observe debug_log read failed: {e}")
        return {"unavailable": True, "reason": "debug_log_read_failed"}

    cheap_fail = 0
    unchanged = 0
    save_ok = 0
    last_cheap_fail_ts: str | None = None
    last_save_ts: str | None = None
    last_unchanged_ts: str | None = None

    for line in raw.splitlines():
        ts_match = _TS_PAT.match(line)
        ts = ts_match.group(1) if ts_match else None
        if _CHEAP_FAIL_PAT.search(line):
            cheap_fail += 1
            if ts:
                last_cheap_fail_ts = ts
        if _UNCHANGED_PAT.search(line):
            unchanged += 1
            if ts:
                last_unchanged_ts = ts
        if _SAVE_OK_PAT.search(line):
            save_ok += 1
            if ts:
                last_save_ts = ts

    return {
        "unavailable": False,
        "cheap_fail_count": cheap_fail,
        "unchanged_count": unchanged,
        "save_ok_count": save_ok,
        "last_cheap_fail_ts": last_cheap_fail_ts,
        "last_save_ts": last_save_ts,
        "last_unchanged_ts": last_unchanged_ts,
    }


def _aggregate_memory_observation(
    chat_id: str,
    *,
    now: datetime | None = None,
) -> dict:
    """Aggregate three-layer memory health metrics for ``chat_id``.

    See ``.crew_workspace/design.md`` §3.1 for the full return schema. Reads
    three memory files (global/project/session) for size + updated_at, then
    scans the JSONL tail and DEBUG_LOG tail for cascade signals. Pure
    w.r.t. observable side effects: no writes, no LLM calls. Never raises
    on I/O errors — relevant fields are flagged ``unavailable=True``.

    ``now`` is accepted for tests asserting relative timestamps. Currently
    unused by the implementation; kept in the signature so future trends
    work can compute relative-day windows without an API break.
    """
    if now is None:
        now = datetime.now()

    # ── 1. resolve cwd for the project layer ─────────────────────────────────
    cwd: str | None = None
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(chat_id)
    except Exception as e:
        _debug_log(f"[Memory] observe _get_cwd failed: {e}")

    # ── 2. per-layer chars/pct/updated_at ────────────────────────────────────
    def _layer_stat(content: str | None, max_chars: int, path: Path | None) -> dict:
        chars = len(content or "")
        if max_chars > 0:
            pct = chars * 100 // max_chars
        else:
            pct = 0
        updated_at: str | None = None
        try:
            if path is not None:
                fm = _load_md_frontmatter(path)
                updated_at = fm.get("updated_at") or None
        except Exception as e:
            _debug_log(f"[Memory] observe frontmatter read failed: {e}")
        return {
            "chars": chars,
            "max_chars": max_chars,
            "pct": pct,
            "near_limit": pct >= _NEAR_LIMIT_PCT,
            "updated_at": updated_at,
        }

    try:
        g_content = load_global_memory(chat_id)
    except Exception as e:
        _debug_log(f"[Memory] observe load_global failed: {e}")
        g_content = None
    try:
        p_content = load_project_memory(cwd) if cwd else None
    except Exception as e:
        _debug_log(f"[Memory] observe load_project failed: {e}")
        p_content = None
    try:
        s_content = load_memory(chat_id)
    except Exception as e:
        _debug_log(f"[Memory] observe load_session failed: {e}")
        s_content = None

    g_path = _global_memory_file(chat_id)
    p_path = _project_memory_file(cwd) if cwd else None
    s_path = _session_memory_file(chat_id)

    layers = {
        "global":  _layer_stat(g_content, GLOBAL_MAX_CHARS,  g_path),
        "project": _layer_stat(p_content, PROJECT_MAX_CHARS, p_path),
        "session": _layer_stat(s_content, SESSION_MAX_CHARS, s_path),
    }

    # ── 3. DEBUG_LOG tail scan (cheap-fail / UNCHANGED / save-ok) ────────────
    debug = _parse_debug_log_window(_OBSERVE_DEBUG_TAIL_BYTES)

    if debug.get("unavailable"):
        recent_window: dict = {"unavailable": True, "reason": debug.get("reason", "debug_log_missing")}
        fallback: dict = {"unavailable": True, "reason": debug.get("reason", "debug_log_missing")}
    else:
        unchanged_count = int(debug.get("unchanged_count", 0))
        save_ok_count = int(debug.get("save_ok_count", 0))
        total_count = unchanged_count + save_ok_count
        unchanged_ratio = (unchanged_count / total_count) if total_count > 0 else 0.0

        cheap_fail_count = int(debug.get("cheap_fail_count", 0))
        # cheap-fail count is an under-count of "one-shot attempts" because
        # success lines are not currently emitted by ``_run_one_shot``. We
        # approximate total_one_shot as (cheap_fail + save_ok) — the bulk of
        # one-shot calls succeed and emit a save line, so the ratio is a
        # reasonable visible health signal even if not perfectly precise.
        total_one_shot = cheap_fail_count + save_ok_count
        ratio = (cheap_fail_count / total_one_shot) if total_one_shot > 0 else 0.0

        recent_window = {
            "window_days": _OBSERVE_WINDOW_DAYS,
            "unchanged_count": unchanged_count,
            "total_count": total_count,
            "unchanged_ratio": round(unchanged_ratio, 4),
            "unavailable": False,
        }
        fallback = {
            "window_days": _OBSERVE_WINDOW_DAYS,
            "count": cheap_fail_count,
            "total_one_shot": total_one_shot,
            "ratio": round(ratio, 4),
            "last_ts": debug.get("last_cheap_fail_ts"),
            "unavailable": False,
        }

    # ── 4. last_successful_update — prefer session frontmatter, fall back to log ─
    last_successful_update: str | None = layers["session"].get("updated_at")
    if last_successful_update is None and not debug.get("unavailable"):
        # Fall back to the last [Memory] saved session_*.md timestamp from
        # the debug log tail. Only HH:MM:SS — but better than nothing.
        last_successful_update = debug.get("last_save_ts")

    # ── 5. trends — REQ-13 reserved slot, populated from jsonl tail ─────────
    trends = _compute_session_trends(chat_id)

    # ── 6. recent-turns pruning summary (read-only snapshot) ────────────────
    # Read the ring-buffer summary from ``log._pruning_stats``. Failure to
    # import (very early bootstrap) or any unexpected error degrades to a
    # neutral "unavailable" struct so the observe card can still render.
    try:
        from larkhelm.log import _pruning_stats as _log_pruning_stats
        pruning = dict(_log_pruning_stats.summary())
        pruning["unavailable"] = False
    except Exception as e:
        _debug_log(f"[Memory] observe pruning summary unavailable: {e}")
        pruning = {
            "window": 0,
            "before_sum": 0,
            "after_sum": 0,
            "saved_pct": 0,
            "unavailable": True,
        }

    return {
        "chat_id": chat_id,
        "cwd": cwd,
        "layers": layers,
        "recent_window": recent_window,
        "fallback": fallback,
        "last_successful_update": last_successful_update,
        "trends": trends,
        "pruning": pruning,
    }


def _compute_session_trends(chat_id: str) -> list[int]:
    """Reserved for REQ-13 — returns [] until log_entry records session sizes."""
    return []
