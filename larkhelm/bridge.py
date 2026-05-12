#!/usr/bin/env python3
"""
larkhelm · main entry point (refactored, slim version)

This file only retains main() and _start_cron_scheduler().
Business logic has been split into:
  config.py       — runtime configuration
  chat_state.py   — persistent state
  concurrency.py  — concurrency control
  card_builder.py — card building
  lark_client.py  — Feishu API wrapper
  perm.py         — permission approval
  ai_runner.py    — AI subprocess runner
  commands.py     — command implementations
  handlers.py     — Feishu event handlers
  crew.py         — multi-agent collaboration
"""
import fcntl
import gc
import os
import signal
import sys
import threading
import time
from datetime import datetime

import lark_oapi as lark
import lark_oapi.ws as ws

import larkhelm.lark_client as _lc
from larkhelm.config import _init_runtime
from larkhelm.log import _debug_log, rotate_jsonl_if_needed
from larkhelm.chat_state import _load_global_state, _get_chat_state, _state_lock, _chat_state_store
from larkhelm.concurrency import _cron_lock, set_shutting_down, wait_for_idle
from larkhelm.perm import _start_perm_server
from larkhelm.handlers import handle_message, handle_card_action, handle_reaction_created

# PID lock file — holds an exclusive flock for the lifetime of the process.
# Prevents multiple daemon instances from running simultaneously, which was a
# root cause of cumulative memory exhaustion (each instance loads 200+ modules
# and accumulates in-memory state independently).
_pid_lock_fd = None


def _acquire_pid_lock(data_dir) -> bool:
    """Try to acquire an exclusive file lock. Returns False if another instance already holds it."""
    global _pid_lock_fd
    lock_path = data_dir / "larkhelm.lock"
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.ftruncate(fd, len(f"{os.getpid()}\n"))
        _pid_lock_fd = fd  # keep fd open — lock released when fd is closed (process exit)
        return True
    except (OSError, IOError):
        return False


_GC_INTERVAL = 600  # run gc.collect() every 10 minutes


def _start_gc_thread():
    """Background thread that periodically runs full GC to release memory back to OS."""
    def _loop():
        while True:
            time.sleep(_GC_INTERVAL)
            try:
                collected = gc.collect(2)  # full collection (generations 0,1,2)
                _debug_log(f"[GC] 周期性 gc.collect 完成，回收 {collected} 个对象")
            except Exception as e:
                _debug_log(f"[GC] gc.collect 异常: {e}")
    threading.Thread(target=_loop, daemon=True, name="gc-collector").start()


def _start_cron_scheduler():
    """Start the cron scheduler daemon thread; checks for due tasks once per minute."""
    from croniter import croniter
    from zoneinfo import ZoneInfo
    import larkhelm.config as _cfg
    from larkhelm.handlers import _do_query
    from larkhelm.chat_state import _get_chat_state as _gcs

    # Track the last-fired timestamp per cron id to prevent double-firing within the same minute due to sleep drift
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
                            # Fire within a 65s window of the scheduled time (tolerates sleep drift),
                            # and at least 50s since the last firing to prevent duplicate execution in the same minute
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


def main(config_path: str = None, data_dir: str = None) -> None:
    import larkhelm.config as _cfg

    _init_runtime(config_path, data_dir)

    if not _acquire_pid_lock(_cfg.DATA_DIR):
        print(
            f"[larkhelm] 另一个实例正在运行（{_cfg.DATA_DIR}/larkhelm.lock 被锁定）。"
            "请先停止已有进程再重启。",
            file=sys.stderr,
        )
        sys.exit(1)

    rotate_jsonl_if_needed()
    _load_global_state()

    if not _cfg.SKIP_PERMISSIONS:
        _start_perm_server()

    # Initialize the global lark client (other modules access it via lark_client.client)
    _lc.client = lark.Client.builder().app_id(_cfg.APP_ID).app_secret(_cfg.APP_SECRET).build()
    _lc._fetch_bot_open_id()

    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(handle_message)
        .register_p2_card_action_trigger(handle_card_action)
        .register_p2_im_message_reaction_created_v1(handle_reaction_created)
        .build()
    )

    # Graceful shutdown: stop accepting new messages on SIGTERM and wait for in-flight tasks to finish
    def _handle_sigterm(signum, frame):
        _debug_log("[Shutdown] 收到 SIGTERM，通知所有 crew 取消...")
        set_shutting_down()

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

    signal.signal(signal.SIGTERM, _handle_sigterm)

    _start_cron_scheduler()
    _start_gc_thread()

    from larkhelm.memory_watchdog import start_memory_watchdog
    start_memory_watchdog(_cfg.MEMORY_LIMIT_MB)
    _debug_log(f"[MemWatchdog] 内存限制: {_cfg.MEMORY_LIMIT_MB} MB (soft={int(_cfg.MEMORY_LIMIT_MB*0.8)} MB)")

    # On startup, resume any crew tasks that were interrupted last time
    from larkhelm.crew import resume_interrupted_crews
    resume_interrupted_crews()

    # U17: surface any /plan whose bridge died mid-flight. Sends an
    # informational card per interrupted plan — does NOT auto-resume
    # execution (state.cancel_ev / step-level checkpointing wouldn't
    # survive restart anyway; user awareness is the high-value bit).
    try:
        from larkhelm.plan_persistence import resume_interrupted_plans
        resume_interrupted_plans()
    except Exception as _e:
        _debug_log(f"[Startup] resume_interrupted_plans failed: {_e}")

    # If restarted via /upgrade, notify the requester that the bot is back online
    import json as _json
    _notify_path = _cfg.DATA_DIR / "_restart_notify.json"
    try:
        if _notify_path.exists():
            _nd = _json.loads(_notify_path.read_text(encoding="utf-8"))
            _notify_path.unlink(missing_ok=True)   # delete first to avoid re-send on crash-loop
            _nc = _nd.get("chat_id", "")
            if _nc and (time.time() - float(_nd.get("ts", 0))) < 300:
                from larkhelm.lark_client import send_card as _send_card
                _send_card(_nc, "✅ 升级完成，已重新连接", "服务已成功重启并重新连接到飞书。", color="green")
                _debug_log(f"[Startup] upgrade restart notification sent to {_nc[:12]}")
    except Exception as _e:
        _debug_log(f"[Startup] restart notify error: {_e}")

    _debug_log("🚀 飞书 × Claude & Gemini 桥接 v2.0 启动")
    _debug_log(f"   Claude:    {_cfg.CLAUDE_CMD}")
    _debug_log(f"   Gemini:    {_cfg.GEMINI_CMD}")
    _debug_log(f"   默认模型:  {_cfg.DEFAULT_MODEL}")
    _debug_log(f"   默认目录:  {_cfg.DEFAULT_CWD}")
    _debug_log(f"   超时:      {_cfg.RESPONSE_TIMEOUT}s")
    if _cfg.ALLOWED_CHATS:
        _debug_log(f"   白名单:    {_cfg.ALLOWED_CHATS}")

    ws_client = ws.Client(_cfg.APP_ID, _cfg.APP_SECRET,
                          event_handler=event_handler,
                          log_level=lark.LogLevel.INFO)
    ws_client.start()


if __name__ == "__main__":
    main()
