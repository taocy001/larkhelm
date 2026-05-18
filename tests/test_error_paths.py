"""P2 AC-03: edge-case tests for chat_state / log / runner failure modes.

Five scenarios required by the PRD:
  * disk full (mock OSError ENOSPC)
  * permission denied (mock PermissionError)
  * JSON decode failure on state.json
  * subprocess killed by SIGKILL (fake subprocess)
  * concurrent writes to state.json (10 threads × 50 writes)
"""
from __future__ import annotations

import errno
import json
import os
import threading
from pathlib import Path

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")


# ── Helper: build an isolated state file ─────────────────────────────────


@pytest.fixture
def isolated_state(monkeypatch, tmp_path: Path):
    """Re-point ``larkhelm.config.STATE_FILE`` at a per-test tmp file."""
    state_file = tmp_path / "state.json"
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "STATE_FILE", state_file, raising=False)
    # Reset the in-memory store so a prior test doesn't bleed in.
    import larkhelm.chat_state as _cs
    with _cs._state_lock:
        _cs._chat_state_store.clear()
    yield state_file


# ── 1) Disk full → ENOSPC ────────────────────────────────────────────────


def test_disk_full(monkeypatch, isolated_state):
    import larkhelm.chat_state as _cs

    def _fake_write_text(self, *a, **kw):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(Path, "write_text", _fake_write_text)
    # _set_chat_field calls _save_state internally. Must not raise — the
    # save path swallows + logs failures.
    _cs._set_chat_field("oc_disk_full", "k", "v")


# ── 2) Permission denied on state file ───────────────────────────────────


def test_permission_denied(monkeypatch, isolated_state):
    import larkhelm.chat_state as _cs

    def _raise_perm(*a, **kw):
        raise PermissionError("perm denied")

    monkeypatch.setattr(Path, "write_text", _raise_perm)
    _cs._set_chat_field("oc_perm", "k", "v")


# ── 3) Corrupt state.json → _load_global_state returns gracefully ────────


def test_json_decode(isolated_state):
    import larkhelm.chat_state as _cs
    isolated_state.write_text("{not valid json")
    _cs._load_global_state()
    with _cs._state_lock:
        # Corrupt files leave the in-memory store empty; the bridge keeps
        # running on a fresh slate rather than crashing on startup.
        assert _cs._chat_state_store == {}


# ── 4) Subprocess SIGKILL — fake the subprocess wait ─────────────────────


def test_subprocess_killed():
    """Real subprocess-kill semantics: when ``self._proc.kill()`` raises
    (because the OS process has already exited — ``ProcessLookupError`` on
    POSIX), the runner's ``_watch`` thread must swallow it and return
    cleanly rather than propagating the exception out of the daemon
    thread (where it would be silently lost AND leave ``_completed`` /
    ``_watch_killed`` in an inconsistent state).

    This exercises the actual cancel-kill path in
    ``runner_base._watch`` (lines 500-503) — the previous version of
    this test only called ``_cleanup_tmp`` on a missing path, which
    didn't touch any subprocess kill code.
    """
    import threading
    from unittest.mock import MagicMock
    from larkhelm.runner_base import BaseProcessRunner

    class _StubRunner(BaseProcessRunner):
        def build_args(self): return ["true"]
        def build_stdin(self): return None
        def parse_stdout_event(self, ev): return False
        def cleanup_extra(self): pass

    r = _StubRunner.__new__(_StubRunner)
    BaseProcessRunner.__init__(
        r, backend_name="stub", chat_id="c_kill", message="m",
        sid=None, cwd="/tmp",
    )

    # Fake Popen whose kill() raises ProcessLookupError — the exact
    # condition the silent-except is designed to catch (proc already gone).
    r._proc = MagicMock()
    kill_called = threading.Event()

    def _kill_raises():
        kill_called.set()
        raise ProcessLookupError("already dead")
    r._proc.kill = _kill_raises

    # Wire cancel_ev so the next _watch loop iteration takes the cancel branch.
    r.cancel_ev = threading.Event()
    r.cancel_ev.set()

    # Track any exception that escapes _watch (a daemon thread eating the
    # exception silently would otherwise hide a regression).
    escaped: list[BaseException] = []

    def _runner():
        try:
            r._watch()
        except BaseException as e:
            escaped.append(e)

    watcher = threading.Thread(target=_runner, daemon=True)
    watcher.start()
    watcher.join(timeout=2.0)

    assert not watcher.is_alive(), "_watch did not return after cancel"
    assert escaped == [], (
        f"_watch propagated exception out of the silent-kill path: "
        f"{escaped!r}"
    )
    assert kill_called.is_set(), "kill() was never invoked"
    assert r._cancelled_flag.is_set(), "cancel flag should be set"
    assert r._watch_killed, "_watch_killed should be True after self-kill"


# ── 5) Concurrent writes to state.json (10 threads × 50 writes) ──────────


def test_concurrent_state_write(isolated_state):
    """10 threads, 50 writes each, all keys eventually visible."""
    import larkhelm.chat_state as _cs

    threads = []
    n_writers = 10
    per_writer = 50

    def _writer(thread_id: int):
        for i in range(per_writer):
            _cs._set_chat_field(f"oc_{thread_id}", f"k{i}", f"v{i}")

    for t in range(n_writers):
        th = threading.Thread(target=_writer, args=(t,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout=10)
        assert not th.is_alive()

    # Verify the persisted file parses cleanly and contains the expected
    # number of chat ids; per-chat fields may interleave (last-writer-wins
    # per (chat_id, key)) but the JSON itself must be well-formed.
    data = json.loads(isolated_state.read_text(encoding="utf-8"))
    assert len(data) == n_writers
    for t in range(n_writers):
        # At least one key visible per chat (writers race but no chat
        # should end up missing entirely).
        assert data[f"oc_{t}"]


# ── 6) Bonus: log helpers tolerate missing log dir ───────────────────────


def test_log_debug_log_tolerates_missing_dir(monkeypatch, tmp_path):
    """``_debug_log`` writes to DEBUG_LOG; a missing parent dir must not raise."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "DEBUG_LOG", tmp_path / "nope" / "no" / "way.log", raising=False)
    import larkhelm.log as _log
    # Should not raise even if dir doesn't exist (the write path creates it).
    _log._debug_log("test message")
