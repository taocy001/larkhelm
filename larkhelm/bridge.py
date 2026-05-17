#!/usr/bin/env python3
"""larkhelm · main entry point.

P2 REQ-02 / AC-02 — ``main()`` is composed from six small helpers, each
testable in isolation against fake ``lark.Client`` / ``ws.Client`` stubs:

* :func:`_install_pid_lock`    — exclusive flock at ``DATA_DIR/larkhelm.lock``
* :func:`_install_signal_handlers` — SIGTERM → graceful shutdown
* :func:`_initialise_clients`  — build lark.Client + fetch BOT_OPEN_ID
* :func:`_register_handlers`   — wire EventDispatcherHandler routes
* :func:`_start_background_threads` — cron / GC / health / watchdog / resume
* :func:`_post_init_notify`    — restart notification + startup banner

Side-effect-free pieces (parsing, dataclasses) live in
``larkhelm._message_pure``; this module only contains the orchestration
that the bridge process actually runs.
"""
from __future__ import annotations

import fcntl
import gc
import json as _json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import lark_oapi as lark
import lark_oapi.ws as ws

import larkhelm.lark_client as _lc
from larkhelm.config import _init_runtime
from larkhelm.log import _debug_log, rotate_jsonl_if_needed
from larkhelm.chat_state import _load_global_state, _state_lock, _chat_state_store
from larkhelm.concurrency import _cron_lock, set_shutting_down, wait_for_idle
from larkhelm.perm import _start_perm_server
from larkhelm.handlers import handle_message, handle_card_action, handle_reaction_created


# ── Module-level state ─────────────────────────────────────────────────────

# PID lock file — holds an exclusive flock for the lifetime of the process.
# Prevents multiple daemon instances from running simultaneously (a
# previously confirmed root cause of cumulative memory exhaustion).
_pid_lock_fd: Optional[int] = None

# Hooks that ``main()`` already installed — kept module-level so the test
# suite can introspect (e.g. assert that the signal handler is registered).
_signal_handlers_installed: bool = False


# ── PID lock ──────────────────────────────────────────────────────────────


def _install_pid_lock(data_dir: Path) -> bool:
    """Try to acquire an exclusive file lock at ``DATA_DIR/larkhelm.lock``.

    Returns True on success; False when another bridge instance already
    holds the lock. The fd is intentionally NOT closed — the lock is
    released only when the process exits.
    """
    global _pid_lock_fd
    lock_path = Path(data_dir) / "larkhelm.lock"
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.ftruncate(fd, len(f"{os.getpid()}\n"))
        _pid_lock_fd = fd
        return True
    except (OSError, IOError):
        return False


# Back-compat alias for any test / external code that imported the old
# private name. New code should use :func:`_install_pid_lock`.
def _acquire_pid_lock(data_dir) -> bool:
    return _install_pid_lock(Path(data_dir))


# ── Signal handlers ───────────────────────────────────────────────────────


def _handle_sigterm(signum, frame):
    """Graceful shutdown: cancel crews, drain queries, exit cleanly."""
    _debug_log("[Shutdown] 收到 SIGTERM，通知所有 crew 取消...")
    set_shutting_down()

    # P1-3: stop health server early so the orchestrator stops routing
    # traffic before we wait on in-flight queries.
    try:
        from larkhelm.health_server import stop_health_server
        stop_health_server()
    except Exception as _hs_err:
        _debug_log(f"[HealthServer] stop failed (continuing): {_hs_err}")

    # P2 REQ-06: flush any buffered extract updates so the SIGTERM doesn't
    # drop a pending session-summary cascade. The 10s cap prevents a stuck
    # backend from blocking shutdown indefinitely.
    try:
        from larkhelm import memory_extract_buffer as _meb
        _meb.flush_all_for_shutdown(timeout_sec=10.0)
    except Exception as _eb_err:
        _debug_log(f"[ExtractBuffer] shutdown flush failed (continuing): {_eb_err}")

    # First cancel all in-progress crews and update their cards
    from larkhelm.crew import cancel_all_crews, wait_crews_done
    cancel_all_crews(reason="服务即将重启")

    # Wait for crew threads to exit (up to 30s)
    crew_done = wait_crews_done(timeout=30.0)
    _debug_log(f"[Shutdown] Crew 线程{'已全部退出' if crew_done else '等待超时，强制继续'}")

    # Then wait for normal queries to finish (up to 60s)
    idle = wait_for_idle(timeout=60.0)
    if idle:
        _debug_log("[Shutdown] 所有任务已完成，正常退出。")
    else:
        _debug_log("[Shutdown] 等待超时，强制退出。")
    sys.exit(0)


def _install_signal_handlers() -> None:
    """Register SIGTERM → graceful shutdown.

    Idempotent: a duplicate call is a no-op. Safe to call from tests
    that drive the bridge through ``main()`` once and want to re-enter.
    """
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return
    signal.signal(signal.SIGTERM, _handle_sigterm)
    _signal_handlers_installed = True


# ── Client / handler wiring ───────────────────────────────────────────────


def _initialise_clients(cfg) -> Any:
    """Construct the singleton ``lark.Client``, fetch the bot's open_id,
    and return the client object so the caller can re-use it for the
    websocket builder.
    """
    _lc.client = (
        lark.Client.builder()
        .app_id(cfg.APP_ID)
        .app_secret(cfg.APP_SECRET)
        .build()
    )
    _lc._fetch_bot_open_id()
    return _lc.client


def _register_handlers(client) -> Any:
    """Build the EventDispatcherHandler — routes Feishu webhook events to
    ``handlers/*``. Returns the built handler so ``main()`` can pass it
    to the websocket client constructor.
    """
    return (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(handle_message)
        .register_p2_card_action_trigger(handle_card_action)
        .register_p2_im_message_reaction_created_v1(handle_reaction_created)
        .build()
    )


# ── Background threads ────────────────────────────────────────────────────

_GC_INTERVAL = 600  # full gc.collect every 10 minutes


def _start_gc_thread() -> None:
    """Periodic full GC so RSS gets handed back to the OS instead of
    accumulating across long-lived workers."""
    def _loop():
        while True:
            time.sleep(_GC_INTERVAL)
            try:
                collected = gc.collect(2)
                _debug_log(f"[GC] 周期性 gc.collect 完成，回收 {collected} 个对象")
            except Exception as e:
                _debug_log(f"[GC] gc.collect 异常: {e}")
    threading.Thread(target=_loop, daemon=True, name="gc-collector").start()


def _start_cron_scheduler() -> None:
    """Start the cron scheduler daemon thread; checks for due tasks once per minute."""
    from croniter import croniter
    from zoneinfo import ZoneInfo
    import larkhelm.config as _cfg
    from larkhelm.handlers import _do_query
    from larkhelm.chat_state import _get_chat_state as _gcs

    # Track last-fired ts per cron id to prevent double-firing within the
    # same minute due to scheduler sleep drift.
    _last_fired: dict[str, float] = {}

    def _loop():
        while True:
            try:
                now_aware = datetime.now(ZoneInfo(_cfg.CRON_TIMEZONE))
                now_ts = now_aware.timestamp()
                with _state_lock:
                    all_chats = list(_chat_state_store.keys())
                for chat_id in all_chats:
                    crons = _gcs(chat_id).get("crons", [])
                    for c in crons:
                        try:
                            cr = croniter(c["expr"], now_aware)
                            prev = cr.get_prev(datetime)
                            diff = (now_aware - prev).total_seconds()
                            cron_id = c["id"]
                            if diff < 65 and (now_ts - _last_fired.get(cron_id, 0)) > 50:
                                _last_fired[cron_id] = now_ts
                                _debug_log(f"[Cron] 触发 id={cron_id} chat={chat_id[:12]}")
                                threading.Thread(
                                    target=_do_query,
                                    args=(chat_id, c["query"], c["model"], None),
                                    daemon=True, name=f"cron-{cron_id}",
                                ).start()
                        except Exception as e:
                            _debug_log(f"[Cron] 任务检查异常 id={c.get('id')}: {e}")
            except Exception as e:
                _debug_log(f"[Cron] 调度器异常: {e}")
            time.sleep(60)

    threading.Thread(target=_loop, daemon=True, name="cron-scheduler").start()


def _start_memory_boot_warmup() -> None:
    """Phase D / Phase 2 — boot-time stale GC + embedding warmup.

    Daemon thread so traffic serving is never blocked. Failures inside
    the loop are caught and logged.
    """
    def _loop():
        try:
            import larkhelm.config as _cfg
            cfg = getattr(_cfg, "config", None) or {}
            from larkhelm.memory_lifecycle import (
                iter_known_chat_cwd_pairs, mark_stale_slices,
            )
            window_days = int(cfg.get("memory_stale_window_days", 90) or 90)
            for chat_id, cwd in iter_known_chat_cwd_pairs():
                try:
                    mark_stale_slices(chat_id, cwd, dry_run=False, window_days=window_days)
                except Exception as inner:
                    _debug_log(f"[MemoryLifecycle] boot warmup mark_stale failed: {inner}")
        except Exception as e:
            _debug_log(f"[MemoryLifecycle] boot warmup phase 1 (stale) failed: {e}")

        try:
            from larkhelm.memory_embedding import get_embedding_backend
            backend = get_embedding_backend()
            if backend is not None:
                backend.warm()
                _debug_log(
                    f"[MemoryRetriever] embedding backend '{backend.name}' warmed"
                )
        except Exception as e:
            _debug_log(f"[MemoryRetriever] embedding warmup failed: {e}")

    threading.Thread(target=_loop, daemon=True, name="boot-warmup").start()


def _start_background_threads(cfg) -> None:
    """Start every background daemon the bridge depends on.

    Order: perm-server (so AI subprocess permission checks have a target)
    → health endpoint → cron / GC / memory warmup → watchdog → crew /
    plan resume. Each block is best-effort and never re-raises so a
    failure in one thread doesn't abort startup.
    """
    if not cfg.SKIP_PERMISSIONS:
        _start_perm_server()

    # P1-3: optional HTTP /health /ready /metrics endpoint.
    try:
        from larkhelm.health_server import start_health_server
        start_health_server(
            getattr(cfg, "HEALTH_ENDPOINT_PORT", 0) or 0,
            getattr(cfg, "HEALTH_BIND_ADDR", "127.0.0.1") or "127.0.0.1",
        )
    except Exception as _hs_err:
        _debug_log(f"[HealthServer] start failed (continuing): {_hs_err}")

    _start_cron_scheduler()
    _start_gc_thread()
    _start_memory_boot_warmup()

    from larkhelm.memory_watchdog import start_memory_watchdog
    start_memory_watchdog(cfg.MEMORY_LIMIT_MB)
    _debug_log(
        f"[MemWatchdog] 内存限制: {cfg.MEMORY_LIMIT_MB} MB "
        f"(soft={int(cfg.MEMORY_LIMIT_MB*0.8)} MB)"
    )

    # On startup, resume any crew tasks that were interrupted last time.
    from larkhelm.crew import resume_interrupted_crews
    resume_interrupted_crews()

    # U17: surface any /plan whose bridge died mid-flight.
    try:
        from larkhelm.plan_persistence import resume_interrupted_plans
        resume_interrupted_plans()
    except Exception as _e:
        _debug_log(f"[Startup] resume_interrupted_plans failed: {_e}")


# ── Post-init notification ────────────────────────────────────────────────


def _post_init_notify(cfg) -> None:
    """Send the "restart complete" card (if any) and emit the startup banner."""
    _notify_path = cfg.DATA_DIR / "_restart_notify.json"
    try:
        if _notify_path.exists():
            _nd = _json.loads(_notify_path.read_text(encoding="utf-8"))
            # Delete first to avoid re-send on crash-loop.
            _notify_path.unlink(missing_ok=True)
            _nc = _nd.get("chat_id", "")
            if _nc and (time.time() - float(_nd.get("ts", 0))) < 300:
                from larkhelm.lark_client import send_card as _send_card
                _send_card(_nc, "✅ 升级完成，已重新连接",
                           "服务已成功重启并重新连接到飞书。", color="green")
                _debug_log(f"[Startup] upgrade restart notification sent to {_nc[:12]}")
    except Exception as _e:
        _debug_log(f"[Startup] restart notify error: {_e}")

    _debug_log("🚀 飞书 × Claude & Gemini 桥接 v2.0 启动")
    _debug_log(f"   Claude:    {cfg.CLAUDE_CMD}")
    _debug_log(f"   Gemini:    {cfg.GEMINI_CMD}")
    _debug_log(f"   默认模型:  {cfg.DEFAULT_MODEL}")
    _debug_log(f"   默认目录:  {cfg.DEFAULT_CWD}")
    _debug_log(f"   超时:      {cfg.RESPONSE_TIMEOUT}s")
    if cfg.ALLOWED_CHATS:
        _debug_log(f"   白名单:    {cfg.ALLOWED_CHATS}")


# ── Top-level entry point ────────────────────────────────────────────────


def main(config_path: str = None, data_dir: str = None) -> None:
    """Bridge entry point. Composed from the six helpers above so each
    block can be unit-tested in isolation (see tests/test_bridge_main.py).
    """
    import larkhelm.config as _cfg

    _init_runtime(config_path, data_dir)

    if not _install_pid_lock(Path(_cfg.DATA_DIR)):
        print(
            f"[larkhelm] 另一个实例正在运行（{_cfg.DATA_DIR}/larkhelm.lock 被锁定）。"
            "请先停止已有进程再重启。",
            file=sys.stderr,
        )
        sys.exit(1)

    rotate_jsonl_if_needed()
    _load_global_state()

    _initialise_clients(_cfg)
    event_handler = _register_handlers(_lc.client)

    _install_signal_handlers()
    _start_background_threads(_cfg)
    _post_init_notify(_cfg)

    ws_client = ws.Client(
        _cfg.APP_ID, _cfg.APP_SECRET,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )
    ws_client.start()


if __name__ == "__main__":
    main()
