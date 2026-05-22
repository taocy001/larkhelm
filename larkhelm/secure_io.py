"""larkhelm · secure file I/O with 0o600 creation mask

Provides two public functions:
  secure_open()         — for append-log writes (Markdown shards, all.jsonl, DEBUG_LOG)
  secure_atomic_write() — for atomic config replacement (config.json, state.json)

Both guarantee that newly created files get mode 0o600 (rw-------).
Existing files are never chmod'd (O_CREAT semantics only).

Degradation on platforms lacking os.open(mode=...) (e.g. Windows native CPython):
  1. os.open without mode + os.fchmod
  2. If fchmod also unavailable: WARN log + continue with default permissions
Neither path raises — the caller's write is never blocked by a permission failure.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Union

O_FLAGS_CREATE_APPEND: int = os.O_WRONLY | os.O_CREAT | os.O_APPEND
O_FLAGS_CREATE_TRUNC: int = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
SENSITIVE_FILE_MODE: int = 0o600


def _warn_fallback(msg: str) -> None:
    """Emit a WARN to stderr.

    Deliberately avoids larkhelm.log: this function may be called while
    _log_at holds _log_lock (secure_open is used inside _log_at), so routing
    through log.warn would deadlock on _log_lock.
    """
    try:
        import sys
        print(f"[SecureIO] WARN: {msg}", file=sys.stderr)
    except Exception:
        pass


def _open_fd(path: Path, flags: int) -> int:
    """Open *path* with *flags* and creation mask 0o600.

    Tries os.open(..., 0o600) first. Falls back to os.open + os.fchmod on
    platforms that don't accept the mode positional arg. Falls back further
    to plain os.open with a WARN if fchmod is also unavailable.
    """
    path_str = str(path)
    try:
        return os.open(path_str, flags, SENSITIVE_FILE_MODE)
    except TypeError:
        fd = os.open(path_str, flags)
        try:
            os.fchmod(fd, SENSITIVE_FILE_MODE)
        except (AttributeError, OSError) as e:
            _warn_fallback(
                f"[SecureIO] fchmod fallback failed for {path}: {e}; "
                "file created with default permissions"
            )
        return fd


def secure_open(
    path: Path,
    mode: str = "a",
    encoding: str = "utf-8",
) -> IO[str]:
    """Open *path* for writing with creation mask 0o600.

    - Newly created files are opened with 0o600; existing files are not chmod'd.
    - ``mode="a"`` (default) uses O_APPEND for atomic multi-process appends.
    - Returns a text-mode file object; the caller is responsible for closing it.
    - Leaks no fd: os.fdopen failure closes the fd before re-raising.
    """
    if "a" in mode:
        flags = O_FLAGS_CREATE_APPEND
        fdopen_mode = "a"
    else:
        flags = O_FLAGS_CREATE_TRUNC
        fdopen_mode = "w"

    fd = _open_fd(path, flags)
    try:
        return os.fdopen(fd, fdopen_mode, encoding=encoding, closefd=True)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise


def secure_atomic_write(
    path: Path,
    data: Union[str, bytes],
    encoding: str = "utf-8",
) -> None:
    """Atomically write *data* to *path* with 0o600 creation mask.

    1. Writes to ``<path>.tmp`` (same directory) with 0o600.
    2. Flushes and closes the temp file.
    3. Calls ``os.replace(tmp, path)`` for atomic rename.

    The caller's existing locks (config_write_lock / state_lock) serialize
    concurrent writers — this function does not add its own locking.
    """
    if isinstance(data, str):
        data_bytes = data.encode(encoding)
    else:
        data_bytes = data

    tmp = Path(str(path) + ".tmp")
    fd = _open_fd(tmp, O_FLAGS_CREATE_TRUNC)
    try:
        f = os.fdopen(fd, "wb", closefd=True)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise

    try:
        f.write(data_bytes)
    finally:
        f.close()

    os.replace(tmp, path)
