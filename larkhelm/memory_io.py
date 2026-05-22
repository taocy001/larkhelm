"""
larkhelm.memory_io — Export and import persistent state for backup and migration.

Covers:
  - DATA_DIR/state.json              (per-chat state)
  - DATA_DIR/sessions/*.sid          (CLI session IDs)
  - DATA_DIR/logs/all.jsonl[.1]      (conversation + token records)
  - DATA_DIR/logs/{chat_id}/*.md     (per-chat Markdown logs)
  - ~/.larkhelm/memory/*.md          (three-tier memory files)
  - config.json                      (sanitised — secrets redacted)

Standard library only: zipfile, json, hashlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_MANIFEST_VERSION = "1"
_MEMORY_HOME = Path.home() / ".larkhelm" / "memory"
# app_id identifies the Feishu bot application — treat as sensitive
_SENSITIVE_KEYS = frozenset({"secret", "api_key", "password", "token", "credential", "app_id"})

# API backend conversation-history filenames live in DATA_DIR/api_sessions/
# as ``{provider}_{chat_id}.json``. The export uses these prefixes to split
# the stem when applying ``--chat-ids`` filtering. Order is irrelevant —
# the helper iterates by descending length so longer prefixes win.
_API_SESSION_PREFIXES = (
    "anthropic_",
    "deepseek_",
    "openai_",
    "gemini_http_",
)


def _human_size(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def _resolve_data_dir(data_dir: Optional[str | Path]) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    try:
        import larkhelm.config as _cfg
        return _cfg.DATA_DIR
    except AttributeError:
        pass
    if "LARKHELM_DATA_DIR" in os.environ:
        return Path(os.environ["LARKHELM_DATA_DIR"])
    _sys = Path("/var/lib/larkhelm")
    if _sys.exists():
        return _sys
    _xdg = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return _xdg / "larkhelm"


def _sid_chat_id(stem: str) -> str:
    for pfx in ("gemini_", "kimi-code_", "kimi_", "deepseek_"):
        if stem.startswith(pfx):
            return stem[len(pfx):]
    return stem


def _api_session_chat_id(stem: str) -> Optional[str]:
    """Strip the longest matching provider prefix from an api_sessions file
    stem (without ``.json``). Returns the chat_id, or ``None`` when no known
    provider prefix matches (caller treats that as "not filterable by
    chat_id" and skips the file when a filter is active).
    """
    for pfx in sorted(_API_SESSION_PREFIXES, key=len, reverse=True):
        if stem.startswith(pfx):
            return stem[len(pfx):]
    return None


def _line_key(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write(dest: Path, data: bytes) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    try:
        os.chmod(dest, 0o600)
    except Exception:
        pass


def _redact_config(raw: dict) -> dict:
    """Recursively redact sensitive keys from a config dict."""
    out: dict = {}
    for k, v in raw.items():
        if any(s in k.lower() for s in _SENSITIVE_KEYS):
            out[k] = "***"
        elif isinstance(v, dict):
            out[k] = _redact_config(v)
        elif isinstance(v, list):
            _lst = []
            for _item in v:
                if isinstance(_item, dict):
                    _lst.append(_redact_config(_item))
                elif isinstance(_item, str) and any(s in _item.lower() for s in _SENSITIVE_KEYS):
                    _lst.append("***")
                else:
                    _lst.append(_item)
            out[k] = _lst
        else:
            out[k] = v
    return out


def export_memory(
    output_path: Optional[str | Path] = None,
    *,
    chat_ids: Optional[list[str]] = None,
    data_dir: Optional[str | Path] = None,
    include_debug_log: bool = False,
) -> Path:
    """Pack persistent state into a .zip archive.

    Args:
        output_path: destination file (.zip appended if missing).
                     Defaults to DATA_DIR/sessions/memory_export_{ts}.zip.
        chat_ids: if set, export only these chat IDs
        data_dir: override DATA_DIR; otherwise auto-detected
        include_debug_log: include larkhelm.log (can be large)

    Returns the resolved output path.
    """
    data = _resolve_data_dir(data_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path is None:
        out = data / "sessions" / f"memory_export_{ts}.zip"
    else:
        out = Path(output_path)
        if out.suffix != ".zip":
            out = Path(str(out) + ".zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    cid_set: Optional[set[str]] = set(chat_ids) if chat_ids else None
    entries: list[dict] = []

    try:
        from larkhelm import __version__ as _ver
    except Exception:
        _ver = "unknown"

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        def _add_file(src: Path, zpath: str, role: str) -> None:
            zf.write(src, zpath)
            entries.append({"zip_path": zpath, "orig_path": str(src), "role": role,
                            "size": src.stat().st_size})

        def _add_bytes(content: bytes, zpath: str, orig: Path, role: str) -> None:
            zf.writestr(zpath, content)
            entries.append({"zip_path": zpath, "orig_path": str(orig), "role": role,
                            "size": len(content)})

        # state.json
        state_f = data / "state.json"
        if state_f.exists():
            if cid_set:
                try:
                    raw = json.loads(state_f.read_text(encoding="utf-8"))
                    filtered = {k: v for k, v in raw.items() if k in cid_set}
                    _add_bytes(
                        json.dumps(filtered, ensure_ascii=False, indent=2).encode(),
                        "data/state.json", state_f, "state",
                    )
                except Exception:
                    _add_file(state_f, "data/state.json", "state")
            else:
                _add_file(state_f, "data/state.json", "state")

        # sessions/*.sid
        ses_dir = data / "sessions"
        if ses_dir.exists():
            for sid in sorted(ses_dir.glob("*.sid")):
                if cid_set and _sid_chat_id(sid.stem) not in cid_set:
                    continue
                _add_file(sid, f"data/sessions/{sid.name}", "session")

        # api_sessions/{provider}_{chat_id}.json — per-backend stream history
        api_sessions_dir = data / "api_sessions"
        api_session_count = 0
        if api_sessions_dir.exists():
            for sid in sorted(api_sessions_dir.glob("*.json")):
                cid = _api_session_chat_id(sid.stem)
                if cid_set and (cid is None or cid not in cid_set):
                    continue
                _add_file(sid, f"data/api_sessions/{sid.name}", "api_session")
                api_session_count += 1

        # logs/all.jsonl and all.jsonl.1
        log_dir = data / "logs"
        for jname in ("all.jsonl", "all.jsonl.1"):
            jfile = log_dir / jname
            if not jfile.exists():
                continue
            if cid_set:
                try:
                    kept = []
                    for ln in jfile.read_bytes().splitlines():
                        if not ln.strip():
                            continue
                        try:
                            rec = json.loads(ln)
                        except Exception:
                            continue
                        if rec.get("chat_id") in cid_set:
                            kept.append(ln)
                    _add_bytes(
                        b"\n".join(kept) + (b"\n" if kept else b""),
                        f"data/logs/{jname}", jfile, "jsonl",
                    )
                    continue
                except Exception:
                    pass
            _add_file(jfile, f"data/logs/{jname}", "jsonl")

        # logs/{chat_id}/*.md
        if log_dir.exists():
            for cdir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
                if cid_set and cdir.name not in cid_set:
                    continue
                for md in sorted(cdir.glob("*.md")):
                    _add_file(md, f"data/logs/{cdir.name}/{md.name}", "log_md")

        # ~/.larkhelm/memory/*.md
        if _MEMORY_HOME.exists():
            for mf in sorted(_MEMORY_HOME.glob("*.md")):
                if cid_set and mf.stem.startswith("session_"):
                    if mf.stem[len("session_"):] not in cid_set:
                        continue
                _add_file(mf, f"memory/{mf.name}", "memory")

        # config.json (redacted)
        try:
            import larkhelm.config as _cfg
            cfg_path = _cfg.CONFIG_PATH
        except AttributeError:
            cfg_path = None
        if cfg_path and cfg_path.exists():
            try:
                raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                safe_cfg = _redact_config(raw_cfg)
                _add_bytes(
                    json.dumps(safe_cfg, ensure_ascii=False, indent=2).encode(),
                    "config.json", cfg_path, "config_redacted",
                )
            except Exception:
                pass

        # debug log (opt-in)
        if include_debug_log:
            try:
                import larkhelm.config as _cfg
                dbg = _cfg.DEBUG_LOG
            except AttributeError:
                dbg = data / "larkhelm.log"
            if dbg.exists():
                _add_file(dbg, "data/larkhelm.log", "debug_log")

        zf.writestr("manifest.json", json.dumps({
            "format_version": _MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "app_version": _ver,
            "data_dir": str(data),
            "memory_home_dir": str(_MEMORY_HOME),
            "chat_ids_filter": sorted(cid_set) if cid_set else None,
            "files": entries,
        }, ensure_ascii=False, indent=2))

    try:
        from larkhelm.log import _debug_log
        _debug_log(
            f"[MemoryIO] exported {len(entries)} files "
            f"({api_session_count} api_sessions) → {out} "
            f"({_human_size(out.stat().st_size)})"
        )
    except Exception:
        pass

    return out


def import_memory(
    zip_path: str | Path,
    *,
    merge: bool = True,
    dry_run: bool = False,
    data_dir: Optional[str | Path] = None,
) -> dict:
    """Restore state from a zip archive created by export_memory().

    Args:
        zip_path: path to the .zip archive
        merge: True (default) = per-chat merge for state.json, deduplicate JSONL;
               False = overwrite all files unconditionally
        dry_run: report what would be written without touching disk
        data_dir: override DATA_DIR; otherwise auto-detected

    Returns:
        dict with keys:
          "written"  — list[str] of destination paths
          "skipped"  — list[tuple[str, str]] of (identifier, reason)
          "warnings" — list[str] of non-fatal messages
    """
    archive = Path(zip_path)
    data = _resolve_data_dir(data_dir)
    result: dict = {"written": [], "skipped": [], "warnings": []}

    lock = data / "larkhelm.lock"
    if lock.exists():
        result["warnings"].append(
            f"Bridge lock file found at {lock}. "
            "Stop the bridge before importing to prevent data corruption."
        )

    with zipfile.ZipFile(archive, "r") as zf:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except Exception as e:
            raise ValueError(f"Cannot read manifest.json from archive: {e}") from e

        fv = manifest.get("format_version")
        if fv != _MANIFEST_VERSION:
            # Warn and continue (AC-05) rather than hard-fail — allows cross-version restores
            result["warnings"].append(
                f"Archive format_version {fv!r} differs from expected {_MANIFEST_VERSION!r}; "
                "proceeding, but some fields may be ignored."
            )

        for entry in manifest.get("files", []):
            zpath: str = entry["zip_path"]
            role: str = entry.get("role", "")

            if role == "config_redacted":
                result["skipped"].append((zpath, "config not imported (secrets redacted; merge manually)"))
                continue

            dest = _dest_for(zpath, data)
            if dest is None:
                # Distinguish zip-slip rejection (known prefix but path escapes
                # its base) from a genuinely unknown prefix, so ops can tell
                # whether an entry was malicious or merely from an older/newer
                # archive layout.
                if zpath.startswith(("data/api_sessions/", "data/", "memory/")):
                    reason = "zip-slip rejected"
                else:
                    reason = "unrecognised path pattern"
                result["skipped"].append((zpath, reason))
                continue

            if dry_run:
                result["written"].append(str(dest))
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw = zf.read(zpath)
            except KeyError:
                result["skipped"].append((zpath, "file listed in manifest but missing from archive"))
                continue

            if role == "state" and merge and dest.exists():
                _merge_state(raw, dest, result)
            elif role == "jsonl" and merge and dest.exists():
                _merge_jsonl(raw, dest, result)
            else:
                _atomic_write(dest, raw)
                result["written"].append(str(dest))

    try:
        from larkhelm.log import _debug_log
        _debug_log(f"[MemoryIO] import done: {len(result['written'])} written, "
                   f"{len(result['skipped'])} skipped, {len(result['warnings'])} warnings")
    except Exception:
        pass

    return result


def _dest_for(zpath: str, data: Path) -> Optional[Path]:
    if zpath == "manifest.json":
        return None
    # Explicit api_sessions branch comes BEFORE the generic data/ branch so
    # zip-slip protection uses ``DATA_DIR/api_sessions/`` as the base (tighter
    # than ``DATA_DIR/``). Order matters — moving this below the data/ check
    # would let an api_sessions entry sneak into DATA_DIR root via "../".
    if zpath.startswith("data/api_sessions/"):
        base = (data / "api_sessions").resolve()
        rest = zpath[len("data/api_sessions/"):]
        dest = (data / "api_sessions" / rest).resolve()
        if not str(dest).startswith(str(base) + os.sep):
            return None  # zip slip protection
        return dest
    if zpath.startswith("data/"):
        dest = (data / zpath[5:]).resolve()
        if not str(dest).startswith(str(data.resolve()) + os.sep):
            return None  # zip slip protection
        return dest
    if zpath.startswith("memory/"):
        dest = (_MEMORY_HOME / zpath[7:]).resolve()
        if not str(dest).startswith(str(_MEMORY_HOME.resolve()) + os.sep):
            return None  # zip slip protection
        return dest
    return None


def _merge_state(incoming_raw: bytes, dest: Path, result: dict) -> None:
    try:
        incoming = json.loads(incoming_raw)
    except Exception as e:
        result["warnings"].append(f"state.json in archive is not valid JSON: {e}; skipped")
        result["skipped"].append(("data/state.json", f"invalid JSON: {e}"))
        return
    try:
        existing = json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        existing = {}

    # Deep merge: per-chat-id level. For chat IDs that exist in both, preserve
    # live state fields not present in the archive (e.g. newer crons, turn_count).
    for chat_id, inc_chat in incoming.items():
        if chat_id not in existing:
            existing[chat_id] = inc_chat
        else:
            ex_chat = existing[chat_id]
            # Start from archive values, then overlay live-only fields
            merged: dict = dict(inc_chat)
            for field, val in ex_chat.items():
                if field not in inc_chat:
                    merged[field] = val
            # Merge crons list by ID: live crons win over archived crons
            inc_crons = {c["id"]: c for c in inc_chat.get("crons", []) if isinstance(c, dict) and "id" in c}
            ex_crons  = {c["id"]: c for c in ex_chat.get("crons", [])  if isinstance(c, dict) and "id" in c}
            inc_crons.update(ex_crons)  # live wins
            if inc_crons or "crons" in inc_chat or "crons" in ex_chat:
                merged["crons"] = list(inc_crons.values())
            existing[chat_id] = merged

    _atomic_write(dest, json.dumps(existing, ensure_ascii=False, indent=2).encode())
    result["written"].append(str(dest))


def _merge_jsonl(incoming_raw: bytes, dest: Path, result: dict) -> None:
    try:
        seen: set[str] = set()
        try:
            for ln in dest.read_bytes().splitlines():
                s = ln.strip()
                if s:
                    seen.add(_line_key(s))
        except Exception:
            pass

        new_lines: list[bytes] = []
        for ln in incoming_raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            k = _line_key(s)
            if k not in seen:
                new_lines.append(s)
                seen.add(k)

        total_in = sum(1 for ln in incoming_raw.splitlines() if ln.strip())
        if new_lines:
            with open(dest, "ab") as f:
                f.write(b"\n".join(new_lines) + b"\n")
            result["warnings"].append(
                f"JSONL merge {dest.name}: {len(new_lines)} new line(s) appended, "
                f"{total_in - len(new_lines)} duplicate(s) skipped"
            )
        result["written"].append(str(dest))
    except Exception as e:
        result["warnings"].append(f"JSONL merge error for {dest.name}: {e}")
        result["skipped"].append((str(dest), f"JSONL merge error: {e}"))


def get_memory_status(chat_id: Optional[str] = None) -> dict:
    """Return a summary dict of current persistent state sizes (requires initialized _cfg)."""
    import larkhelm.config as _cfg

    def _dir_size(p: Path) -> int:
        if not p.exists():
            return 0
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    chats: list[dict] = []
    if _cfg.STATE_FILE.exists():
        try:
            state = json.loads(_cfg.STATE_FILE.read_text(encoding="utf-8"))
            for cid, s in sorted(state.items()):
                chats.append({
                    "chat_id": cid[:16] + "…" if len(cid) > 16 else cid,
                    "model": s.get("model", "?"),
                    "cwd": s.get("cwd", "?"),
                    "turn_count": s.get("turn_count", 0),
                })
        except Exception:
            pass

    n_sessions = 0
    ses_dir = _cfg.DATA_DIR / "sessions"
    if ses_dir.exists():
        n_sessions = len(list(ses_dir.glob("*.sid")))

    n_api_sessions, api_session_size = 0, 0
    api_dir = _cfg.DATA_DIR / "api_sessions"
    if api_dir.exists():
        api_files = list(api_dir.glob("*.json"))
        n_api_sessions = len(api_files)
        api_session_size = sum(f.stat().st_size for f in api_files)

    n_memory, mem_size = 0, 0
    if _MEMORY_HOME.exists():
        mem_files = list(_MEMORY_HOME.glob("*.md"))
        n_memory = len(mem_files)
        mem_size = sum(f.stat().st_size for f in mem_files)

    return {
        "chats": chats,
        "n_chats": len(chats),
        "n_sessions": n_sessions,
        "n_api_sessions": n_api_sessions,
        "api_session_size": api_session_size,
        "log_size": _dir_size(_cfg.LOG_DIR),
        "data_size": _dir_size(_cfg.DATA_DIR),
        "memory_files": n_memory,
        "memory_size": mem_size,
    }
