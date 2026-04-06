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
import signal
import sys
import threading
import time
from datetime import datetime

import lark_oapi as lark
import lark_oapi.ws as ws

import larkhelm.lark_client as _lc
from larkhelm.config import _init_runtime
from larkhelm.log import _debug_log
from larkhelm.chat_state import _load_global_state, _get_chat_state, _state_lock, _chat_state_store
from larkhelm.concurrency import _cron_lock, set_shutting_down, wait_for_idle
from larkhelm.perm import _start_perm_server
from larkhelm.handlers import handle_message, handle_card_action, handle_reaction_created


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


def main(config_path: str | None = None, data_dir: str | None = None) -> None:
    import larkhelm.config as _cfg

    _init_runtime(config_path, data_dir)
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

    # On startup, resume any crew tasks that were interrupted last time
    from larkhelm.crew import resume_interrupted_crews
    resume_interrupted_crews()

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
