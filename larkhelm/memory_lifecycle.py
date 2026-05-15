"""larkhelm · memory slice lifecycle — Phase D / Phase 2.

Tracks which slices have gone "stale" (no audit hits for ``window_days``)
and decorates loaded slices with ``stale=True`` so retrievers can apply
the configured relevance multiplier (``memory_stale_decay``, default 0.5).

This module is intentionally separate from :mod:`larkhelm.memory_retriever`
so that the retriever's keyword path stays import-cheap (no JSONL parsing
for every call) and so that the boot-time GC daemon can run independently
of any live retriever instance. The sidecar file format is documented in
``.crew_workspace/design.md`` §3.5.

Files written (all 0600):

    ~/.larkhelm/memory/{global,project,session}_<scope>.meta.json
        Sidecar to the ``.md`` slice file with the same stem. Contains
        ``stale_slice_ids`` and book-keeping (``last_gc_at``,
        ``gc_window_days``).

This module must NOT import :mod:`larkhelm.memory_retriever` (it is a
direct collaborator and would otherwise form a cycle on ``load_slices``).
Audit JSONL parsing is done with stdlib :mod:`json` only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Iterable

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.memory_slice import MemorySlice


SLICE_META_SCHEMA_VERSION = 1


@dataclasses.dataclass
class SliceMeta:
    """Sidecar metadata for one layer file (``.md`` ↔ ``.meta.json``)."""

    schema_version: int = SLICE_META_SCHEMA_VERSION
    updated_at: str = ""
    stale_slice_ids: tuple[str, ...] = ()
    last_gc_at: str = ""
    gc_window_days: int = 90


# ── load / save ───────────────────────────────────────────────────────────


def _meta_path_for_layer(md_path: Path) -> Path:
    """Return ``<stem>.meta.json`` next to the given ``.md`` file."""
    return md_path.with_suffix(".meta.json")


def load_slice_meta(meta_path: Path) -> SliceMeta:
    """Read and validate one ``.meta.json``.

    Missing file or any parse failure → fresh :class:`SliceMeta` (empty
    stale list). Callers should never have to handle exceptions; this is
    the same fail-open contract used by ``_resolve_audit_path`` etc.
    """
    try:
        if not meta_path.exists():
            return SliceMeta()
        raw = meta_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return SliceMeta()
        stale_ids = data.get("stale_slice_ids", []) or []
        if not isinstance(stale_ids, list):
            stale_ids = []
        return SliceMeta(
            schema_version=int(data.get("schema_version", SLICE_META_SCHEMA_VERSION)),
            updated_at=str(data.get("updated_at", "") or ""),
            stale_slice_ids=tuple(str(x) for x in stale_ids if isinstance(x, str)),
            last_gc_at=str(data.get("last_gc_at", "") or ""),
            gc_window_days=int(data.get("gc_window_days", 90) or 90),
        )
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] load_slice_meta failed for {meta_path.name}: {e}")
        return SliceMeta()


def save_slice_meta(meta_path: Path, meta: SliceMeta) -> None:
    """Atomically write ``meta`` to ``meta_path`` with mode 0600.

    Uses ``tempfile.NamedTemporaryFile`` + ``os.replace`` so concurrent
    readers never see a half-written file. Any failure is logged and
    swallowed — the caller treats meta as best-effort.
    """
    payload = {
        "schema_version": int(meta.schema_version or SLICE_META_SCHEMA_VERSION),
        "updated_at": meta.updated_at
            or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "stale_slice_ids": list(meta.stale_slice_ids or ()),
        "last_gc_at": meta.last_gc_at or "",
        "gc_window_days": int(meta.gc_window_days or 90),
    }
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile + os.replace gives atomic rename. We can't use
        # Path.write_text because we need to set mode 0600 BEFORE the rename.
        fd, tmp_name = tempfile.mkstemp(
            prefix=meta_path.name + ".", suffix=".tmp",
            dir=str(meta_path.parent),
        )
        try:
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
            os.write(fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(tmp_name, meta_path)
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] save_slice_meta failed for {meta_path.name}: {e}")


# ── inject_stale_marks ────────────────────────────────────────────────────


def inject_stale_marks(
    slices: list[MemorySlice],
    layer_paths: list[tuple[str, str, Path]] | Iterable[tuple[str, str, Path]],
) -> list[MemorySlice]:
    """Return a new list where slices in any ``.meta.json``'s
    ``stale_slice_ids`` get ``stale=True``.

    ``layer_paths`` is the triple list returned by
    :func:`larkhelm.memory_retriever._resolve_layer_files`. Missing meta
    files are silently treated as "no stale ids" (Phase 1 behaviour).
    """
    if not slices:
        return list(slices)
    stale_ids: set[str] = set()
    try:
        for entry in layer_paths or ():
            try:
                _layer, _scope, md_path = entry
            except Exception:
                continue
            meta = load_slice_meta(_meta_path_for_layer(md_path))
            stale_ids.update(meta.stale_slice_ids or ())
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] inject_stale_marks scan failed: {e}")
        return list(slices)
    if not stale_ids:
        return list(slices)

    out: list[MemorySlice] = []
    for s in slices:
        if s.id in stale_ids and not s.stale:
            out.append(dataclasses.replace(s, stale=True))
        else:
            out.append(s)
    return out


# ── mark_stale_slices (GC entry point) ────────────────────────────────────


def _iter_audit_records_for_window(window_days: int) -> Iterator[dict[str, Any]]:
    """Yield audit records (dict) emitted within the last ``window_days``.

    Uses :func:`larkhelm.memory_retriever.iter_audit_records` when
    available (rotation-aware). Falls back to a direct read of the audit
    JSONL path if the helper is missing (eg. when this module is imported
    before retriever's module is fully initialised — rare, but cheap to
    guard).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(window_days)))
    try:
        from larkhelm.memory_retriever import iter_audit_records
        for record in iter_audit_records(timedelta(days=window_days)):
            yield record
        return
    except Exception:
        pass
    # Fallback path: read the single canonical JSONL.
    try:
        from larkhelm.memory_retriever import _resolve_audit_path
        path = _resolve_audit_path()
    except Exception:
        return
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                ts = record.get("ts", "")
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except Exception:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    yield record
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] audit fallback read failed: {e}")


def mark_stale_slices(
    chat_id: str,
    cwd: str | None,
    *,
    dry_run: bool = False,
    window_days: int = 90,
) -> int:
    """Compute the never-hit set for this (chat_id, cwd) pair and write it
    to each layer's ``.meta.json``.

    Returns the count of slices newly flagged stale across all three layers.
    When ``dry_run`` is True, no files are written — useful for boot-time
    audits and tests.

    Algorithm (REQ-44):

      1. Resolve the three ``(layer, scope, .md)`` paths via the retriever.
      2. ``load_slices`` for current slice ids.
      3. Walk audit JSONL records in the trailing ``window_days``; collect
         ``selected_slice_ids``.
      4. Stale set = (current ids) − (hit ids).
      5. Persist to each layer's ``.meta.json`` (one file per layer).
    """
    try:
        from larkhelm.memory_retriever import _resolve_layer_files, load_slices
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] mark_stale_slices import failed: {e}")
        return 0

    try:
        triples = _resolve_layer_files(chat_id, cwd)
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] _resolve_layer_files failed: {e}")
        return 0

    try:
        slices = load_slices(chat_id, cwd)
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] load_slices failed: {e}")
        return 0

    if not slices:
        return 0

    by_layer: dict[str, set[str]] = {}
    for s in slices:
        by_layer.setdefault(s.layer, set()).add(s.id)

    hit_ids: set[str] = set()
    try:
        for record in _iter_audit_records_for_window(window_days):
            for sid in record.get("selected_slice_ids", []) or ():
                if isinstance(sid, str):
                    hit_ids.add(sid)
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] audit scan failed: {e}")

    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    newly_stale = 0
    for layer, _scope, md_path in triples:
        ids_for_layer = by_layer.get(layer, set())
        stale_ids = sorted(ids_for_layer - hit_ids)
        if dry_run:
            existing = load_slice_meta(_meta_path_for_layer(md_path))
            previously = set(existing.stale_slice_ids or ())
            newly_stale += sum(1 for sid in stale_ids if sid not in previously)
            continue
        meta_path = _meta_path_for_layer(md_path)
        existing = load_slice_meta(meta_path)
        previously = set(existing.stale_slice_ids or ())
        newly_stale += sum(1 for sid in stale_ids if sid not in previously)
        meta = SliceMeta(
            schema_version=SLICE_META_SCHEMA_VERSION,
            updated_at=now_iso,
            stale_slice_ids=tuple(stale_ids),
            last_gc_at=now_iso,
            gc_window_days=int(window_days),
        )
        save_slice_meta(meta_path, meta)
    return newly_stale


# ── unstale_slice_id ──────────────────────────────────────────────────────


def unstale_slice_id(slice_id: str) -> bool:
    """Remove ``slice_id`` from every ``.meta.json`` under
    ``~/.larkhelm/memory``. Returns True iff at least one file had the id.
    """
    if not slice_id:
        return False
    try:
        from larkhelm.memory import MEMORY_HOME_DIR
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] unstale_slice_id home import failed: {e}")
        return False
    found = False
    try:
        for meta_path in Path(MEMORY_HOME_DIR).glob("*.meta.json"):
            meta = load_slice_meta(meta_path)
            if slice_id not in (meta.stale_slice_ids or ()):
                continue
            new_ids = tuple(s for s in meta.stale_slice_ids if s != slice_id)
            new_meta = dataclasses.replace(
                meta,
                stale_slice_ids=new_ids,
                updated_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            )
            save_slice_meta(meta_path, new_meta)
            found = True
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] unstale_slice_id failed: {e}")
        return found
    return found


# ── iter_known_chat_cwd_pairs ─────────────────────────────────────────────


def iter_known_chat_cwd_pairs() -> Iterator[tuple[str, str | None]]:
    """Yield ``(chat_id, cwd)`` pairs known to the bridge.

    Two sources:

      1. ``larkhelm.chat_state._chat_state_store`` — every chat the bridge
         has seen since last restart (cwd is per-chat).
      2. ``MEMORY_HOME_DIR/session_*.md`` — chat ids whose session file
         exists but isn't loaded into chat_state (e.g. after a fresh
         bridge restart, before any traffic arrives).
    """
    seen: set[str] = set()
    try:
        from larkhelm.chat_state import _chat_state_store
        for chat_id, state in list(_chat_state_store.items()):
            try:
                cwd = str((state or {}).get("cwd") or "") or None
            except Exception:
                cwd = None
            if chat_id and chat_id not in seen:
                seen.add(chat_id)
                yield chat_id, cwd
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] iter chat_state failed: {e}")

    try:
        from larkhelm.memory import MEMORY_HOME_DIR
        for md_path in Path(MEMORY_HOME_DIR).glob("session_*.md"):
            chat_id = md_path.stem[len("session_"):]
            if chat_id and chat_id not in seen:
                seen.add(chat_id)
                yield chat_id, None
    except Exception as e:
        _debug_log(f"[MemoryLifecycle] iter session files failed: {e}")


__all__ = [
    "SliceMeta",
    "load_slice_meta",
    "save_slice_meta",
    "inject_stale_marks",
    "mark_stale_slices",
    "unstale_slice_id",
    "iter_known_chat_cwd_pairs",
]
