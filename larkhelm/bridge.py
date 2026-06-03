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
from larkhelm.log import _debug_log, redact_error, rotate_jsonl_if_needed
from larkhelm.chat_state import (
    _load_global_state, _state_lock, _chat_state_store,
    _get_chat_state, _set_chat_field,
)
from larkhelm.concurrency import _cron_lock, set_shutting_down, is_shutting_down, wait_for_idle
from larkhelm.dedup import _save_to_disk as _dedup_flush
from larkhelm.failure_report import emit as _emit_failure_report
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

# Guard that ensures _run_shutdown_sequence() executes at most once, even if
# SIGTERM is delivered multiple times before ws_client.start() returns.
_shutdown_sequence_executed: bool = False

# Alert daemon throttle state: metric_name → unix timestamp of last alert sent.
_alert_throttle: dict[str, float] = {}
_ALERT_THROTTLE_SEC: int = 300  # 5-minute dedup window per metric


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


def _handle_sigterm(signum: int, frame: Any) -> None:
    """Signal handler: set the shutting-down flag and return immediately.

    Safety contract: no I/O, no lock acquisition, no sleep, no sys.exit.
    Idempotent: repeated SIGTERM calls are safe (set_shutting_down is idempotent).
    The actual cleanup sequence runs in main() after ws_client.start() returns.
    """
    set_shutting_down()


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


# ── Interruptible select + shutdown sequence ──────────────────────────────


def _make_interruptible_select() -> Optional[Any]:
    """Monkey-patch lark_oapi.ws.client._select with a shutting-down-aware coroutine.

    Returns the patched coroutine on success, or None if the attribute is absent
    (SDK internal API changed), signalling main() to use the daemon-thread fallback.
    """
    import asyncio as _asyncio

    async def _interruptible_select() -> None:
        while not is_shutting_down():
            await _asyncio.sleep(1.0)

    try:
        import lark_oapi.ws.client as _ws_client_mod
        _ws_client_mod._select = _interruptible_select
        return _interruptible_select
    except AttributeError:
        return None


def _run_shutdown_sequence() -> dict:
    """Execute the full graceful shutdown sequence after ws_client.start() returns.

    Steps (each wrapped in try/except so a single failure doesn't abort the rest):
      1. stop_health_server()
      2. flush_all_for_shutdown(timeout_sec=10.0)
      3. cancel_all_crews(reason="服务即将重启")
      4. wait_crews_done(timeout=30.0)
      5. wait_for_idle(timeout=60.0)
      6. time.sleep(2)

    Returns a dict with per-step outcomes for structured logging and test assertions:
        {
            "health_server_stopped": bool,
            "buffer_flushed": bool,
            "crews_cancelled": bool,
            "crews_done": bool,
            "idle_reached": bool,
            "final_status": "clean" | "timeout",
        }
    """
    global _shutdown_sequence_executed
    if _shutdown_sequence_executed:
        return {}
    _shutdown_sequence_executed = True

    result: dict = {
        "health_server_stopped": False,
        "buffer_flushed": False,
        "crews_cancelled": False,
        "crews_done": False,
        "idle_reached": False,
        "final_status": "clean",
    }

    try:
        from larkhelm.health_server import stop_health_server
        stop_health_server()
        result["health_server_stopped"] = True
    except Exception as e:
        _debug_log(f"[HealthServer] stop failed (continuing): {e}")

    try:
        from larkhelm import memory_extract_buffer as _meb
        _meb.flush_all_for_shutdown(timeout_sec=10.0)
        result["buffer_flushed"] = True
    except Exception as e:
        _debug_log(f"[ExtractBuffer] shutdown flush failed (continuing): {e}")

    try:
        _dedup_flush()
    except Exception as e:
        _debug_log(f"[Dedup] shutdown flush failed (continuing): {e}")

    try:
        from larkhelm.crew import cancel_all_crews
        cancel_all_crews(reason="服务即将重启")
        result["crews_cancelled"] = True
    except Exception as e:
        _debug_log(f"[Shutdown] cancel_all_crews failed (continuing): {e}")

    try:
        from larkhelm.crew import wait_crews_done
        done = wait_crews_done(timeout=30.0)
        result["crews_done"] = done
        _debug_log(f"[Shutdown] Crew 线程{'已全部退出' if done else '等待超时，强制继续'}")
    except Exception as e:
        _debug_log(f"[Shutdown] wait_crews_done failed (continuing): {e}")

    try:
        idle = wait_for_idle(timeout=60.0)
        result["idle_reached"] = idle
    except Exception as e:
        _debug_log(f"[Shutdown] wait_for_idle failed (continuing): {e}")
        idle = False

    time.sleep(2)

    if result["idle_reached"]:
        _debug_log("[Shutdown] 清理完成，正常退出")
    else:
        result["final_status"] = "timeout"
        _debug_log("[Shutdown] 清理完成（等待超时），正常退出")

    return result


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


def _persist_cron_result(
    chat_id: str,
    cron_id: str,
    status: str,
    error_text: str,
) -> None:
    """Write ``last_run_at`` / ``last_run_status`` / ``last_error`` back to the
    target cron entry inside ``state.json``.

    Holds ``_cron_lock`` for the re-fetch + modify + ``_set_chat_field`` so
    concurrent ``/cron add|del`` writes cannot drop the new fields. Wrapped
    in a top-level try/except — per REQ-07 the scheduler must NEVER raise
    out of a tick; any persistence failure is logged via ``_debug_log``.
    """
    try:
        from zoneinfo import ZoneInfo
        import larkhelm.config as _cfg
        ts = datetime.now(ZoneInfo(_cfg.CRON_TIMEZONE)).isoformat(timespec="seconds")
        with _cron_lock:
            crons = list(_get_chat_state(chat_id).get("crons", []))
            updated = False
            for entry in crons:
                if entry.get("id") == cron_id:
                    entry["last_run_at"] = ts
                    entry["last_run_status"] = status
                    entry["last_error"] = error_text if status == "error" else ""
                    updated = True
                    break
            if updated:
                _set_chat_field(chat_id, "crons", crons)
            else:
                _debug_log(
                    f"[Cron] _persist_cron_result: cron_id={cron_id} not found "
                    f"in chat={chat_id[:12]}"
                )
    except Exception as e:
        _debug_log(f"[Cron] _persist_cron_result failed id={cron_id}: {e}")


def _bump_failure_and_maybe_emit(
    chat_id: str,
    c: dict,
    exc: BaseException,
    consecutive_failures: dict[str, int],
) -> None:
    """Increment the per-cron consecutive-failure counter; if it reaches 3,
    push an admin failure card via ``failure_report.emit`` and reset the
    counter back to 0 so the next 3-strike window can build up again.

    REQ-03: emit exactly once per 3-failure window. After emit the counter
    is cleared regardless of whether the card actually went out (flag-off
    or empty admin_chat_id still counts as "handled" — re-firing every
    minute would only spam the operator).
    """
    try:
        cron_id = c.get("id", "")
        consecutive_failures[cron_id] = consecutive_failures.get(cron_id, 0) + 1
        count = consecutive_failures[cron_id]
        if count < 3:
            return
        try:
            exc_type = type(exc).__name__
            expr = c.get("expr", "")
            summary = (
                f"cron_id={cron_id} expr={expr} 连续失败 {count} 次 · {exc_type}"
            )
            query_preview = str(c.get("query", ""))[:80]
            detail = (
                f"chat={chat_id[:12]}\n"
                f"query={query_preview}\n"
                f"error={redact_error(repr(exc))[:400]}"
            )
            _emit_failure_report("cron", summary, detail)
        finally:
            consecutive_failures[cron_id] = 0
    except Exception as e:
        _debug_log(f"[Cron] _bump_failure_and_maybe_emit failed: {e}")


def _process_cron_tick(
    chat_id: str,
    c: dict,
    now_aware: datetime,
    last_fired: dict[str, float],
    consecutive_failures: dict[str, int],
) -> None:
    """Process a single cron entry for one scheduler tick.

    Mirrors the original per-cron ``try`` block from ``_loop``: parse the
    expression, check whether it just fired (diff < 65s) and was not
    already triggered this minute (>50s since last_fired), and on a hit
    spawn the ``_do_query`` daemon thread. Now also writes the
    ``last_run_*`` observability fields and accumulates failures for
    the admin-card threshold.

    Never raises — every internal failure is funneled to ``_debug_log``
    plus the failure-counter path so the scheduler loop above can keep
    iterating other chats / cron entries undisturbed.
    """
    from croniter import croniter
    from larkhelm.handlers import _do_query

    cron_id = c.get("id", "")
    try:
        cr = croniter(c["expr"], now_aware)
        prev = cr.get_prev(datetime)
        diff = (now_aware - prev).total_seconds()
        now_ts = now_aware.timestamp()
        if diff < 65 and (now_ts - last_fired.get(cron_id, 0)) > 50:
            last_fired[cron_id] = now_ts
            _debug_log(f"[Cron] 触发 id={cron_id} chat={chat_id[:12]}")
            threading.Thread(
                target=_do_query,
                args=(chat_id, c["query"], c["model"], None),
                daemon=True, name=f"cron-{cron_id}",
            ).start()
            consecutive_failures[cron_id] = 0
            _persist_cron_result(chat_id, cron_id, "ok", "")
    except Exception as e:
        _debug_log(f"[Cron] 任务检查异常 id={cron_id}: {e}")
        _bump_failure_and_maybe_emit(chat_id, c, e, consecutive_failures)
        _persist_cron_result(
            chat_id, cron_id, "error", redact_error(repr(e))[:200],
        )


def _start_cron_scheduler() -> None:
    """Start the cron scheduler daemon thread; checks for due tasks once per minute."""
    from zoneinfo import ZoneInfo
    import larkhelm.config as _cfg
    from larkhelm.chat_state import _get_chat_state as _gcs

    # Track last-fired ts per cron id to prevent double-firing within the
    # same minute due to scheduler sleep drift.
    _last_fired: dict[str, float] = {}
    # Track per-cron consecutive failure count for the admin-card threshold.
    _consecutive_failures: dict[str, int] = {}

    def _loop():
        while True:
            try:
                now_aware = datetime.now(ZoneInfo(_cfg.CRON_TIMEZONE))
                with _state_lock:
                    all_chats = list(_chat_state_store.keys())
                for chat_id in all_chats:
                    crons = _gcs(chat_id).get("crons", [])
                    for c in crons:
                        _process_cron_tick(
                            chat_id, c, now_aware,
                            _last_fired, _consecutive_failures,
                        )
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

        # Phase 2 — pre-fill memory layer LRU with recently-modified .md files
        try:
            import time as _time
            from larkhelm.memory import MEMORY_HOME_DIR
            from larkhelm._context_cache import cached_memory_layer
            cutoff = _time.time() - 86400
            for path in MEMORY_HOME_DIR.glob("*.md"):
                try:
                    if path.stat().st_mtime < cutoff:
                        continue
                    name = path.name
                    if name.startswith("global_"):
                        layer = "global"
                    elif name.startswith("project_"):
                        layer = "project"
                    elif name.startswith("session_"):
                        layer = "session"
                    else:
                        continue
                    cached_memory_layer(layer, path, loader=lambda p=path: p.read_text("utf-8"))
                except Exception as inner:
                    _debug_log(f"[BootWarmup] Phase 2 LRU fill failed for {path.name}: {inner}")
        except Exception as e:
            _debug_log(f"[BootWarmup] Phase 2 LRU warmup failed: {e}")

    threading.Thread(target=_loop, daemon=True, name="boot-warmup").start()


def _maybe_send_alert(metric_name: str, message: str, cfg) -> None:
    """Send an orange alert card if not throttled within _ALERT_THROTTLE_SEC."""
    now = time.time()
    last = _alert_throttle.get(metric_name, 0.0)
    if now - last < _ALERT_THROTTLE_SEC:
        return
    _alert_throttle[metric_name] = now
    try:
        import larkhelm.config as _acfg
        admin_chat_id = getattr(_acfg, "ADMIN_CHAT_ID", "") or ""
        if not admin_chat_id:
            admin_chat_id = getattr(_acfg, "DEFAULT_OWNER_OPEN_ID", "") or ""
        if not admin_chat_id:
            return
        _lc.send_card(admin_chat_id, "🔔 指标告警", message, color="orange")
    except Exception as _e:
        _debug_log(f"[MetricsAlert] send_card failed: {_e}")


def _run_alert_loop(cfg) -> None:
    """Polling loop that checks metric thresholds and sends throttled alerts."""
    import larkhelm.config as _acfg
    while not is_shutting_down():
        interval = getattr(_acfg, "METRICS_ALERT_INTERVAL_SEC", 60)
        try:
            time.sleep(max(1, interval))
        except Exception:
            time.sleep(60)
        if is_shutting_down():
            break
        try:
            from larkhelm.metrics import get_registry
            reg = get_registry()
            if not reg.available:
                continue
            aq_threshold = getattr(_acfg, "METRICS_ALERT_ACTIVE_QUERIES_THRESHOLD", 20)
            rss_threshold = getattr(_acfg, "METRICS_ALERT_RSS_BYTES_THRESHOLD", 2 * 1024 * 1024 * 1024)
            if reg.active_queries is not None:
                try:
                    aq_val = reg.active_queries._value.get()
                    if aq_val >= aq_threshold:
                        _maybe_send_alert(
                            "active_queries",
                            f"⚠️ 活跃查询数过多：{int(aq_val)} >= {aq_threshold}",
                            _acfg,
                        )
                except Exception:
                    pass
            if reg.memory_rss_bytes is not None:
                try:
                    rss_val = reg.memory_rss_bytes._value.get()
                    if rss_val >= rss_threshold:
                        rss_mb = int(rss_val) // (1024 * 1024)
                        thr_mb = rss_threshold // (1024 * 1024)
                        _maybe_send_alert(
                            "memory_rss_bytes",
                            f"⚠️ 内存占用过高：{rss_mb} MB >= {thr_mb} MB",
                            _acfg,
                        )
                except Exception:
                    pass
        except Exception as _e:
            _debug_log(f"[MetricsAlert] threshold check failed: {_e}")


def _start_metrics_alert_daemon(cfg) -> None:
    """Start alert polling daemon thread. Best-effort, never re-raises."""
    try:
        import larkhelm.config as _acfg
        if not getattr(_acfg, "METRICS_ALERT_ENABLED", True):
            return
        t = threading.Thread(
            target=_run_alert_loop, args=(cfg,),
            daemon=True, name="metrics-alert-daemon",
        )
        t.start()
    except Exception as _e:
        _debug_log(f"[MetricsAlert] start failed: {_e}")


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

    _start_metrics_alert_daemon(cfg)


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
                _prev = _nd.get("prev_head", "")
                _new = _nd.get("new_head", "")
                _subj = _nd.get("commit_subject", "")
                _body_lines = ["服务已成功重启并重新连接到飞书。"]
                if _new:
                    if _prev and _prev != _new:
                        _body_lines.append(f"\n已升级 `{_prev}` → `{_new}`")
                    else:
                        _body_lines.append(f"\n当前版本：`{_new}`")
                if _subj:
                    # Truncate long subjects (commit messages can be 100+ chars
                    # when the headline includes batched-change summaries).
                    _short_subj = _subj if len(_subj) <= 120 else _subj[:117] + "…"
                    _body_lines.append(f"\n> {_short_subj}")
                _send_card(_nc, "✅ 升级完成，已重新连接",
                           "\n".join(_body_lines), color="green")
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

    _emit_plugin_load_report(cfg)


def _emit_plugin_load_report(cfg) -> None:
    """P3 REQ-07: re-run the plugin loader so the structured report exists,
    then push an admin card when ``plugin_report_card_enabled=true``.

    The earlier ``_init_plugins`` invocation discarded the report; we
    duplicate the call here so a non-trivial failure is visible without
    requiring a second boot. The registry registration on the second
    call is idempotent (registers under the same agent_type).
    """
    try:
        if not getattr(cfg, "PLUGIN_REPORT_CARD_ENABLED", False):
            return
        admin = getattr(cfg, "ADMIN_CHAT_ID", "") or ""
        if not admin:
            _debug_log("[PluginReport] enabled but admin_chat_id empty; skipping card")
            return
        from larkhelm.agent_hub.plugin_loader import load_plugins
        from larkhelm.agent_hub.plugin_report import emit_admin_card
        report = load_plugins(getattr(cfg, "config", {}) or {})
        if report.has_failures():
            emit_admin_card(report, admin)
        else:
            _debug_log(
                f"[PluginReport] loaded {len(report.loaded)} plugin(s) "
                f"in {report.duration_sec:.2f}s, no failures"
            )
    except Exception as e:
        _debug_log(f"[PluginReport] emit failed: {e}")


# ── Top-level entry point ────────────────────────────────────────────────


def main(config_path: str = None, data_dir: str = None) -> None:
    """Bridge entry point. Composed from the six helpers above so each
    block can be unit-tested in isolation (see tests/test_bridge_main.py).

    LOGIC-C3 changes:
    1. Monkey-patches lark_oapi.ws.client._select before ws_client.start() so
       the blocking loop exits promptly when SIGTERM sets shutting_down.
    2. Calls _run_shutdown_sequence() after start() returns.
    3. Returns normally; no sys.exit in the happy path.
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

    patched_select = _make_interruptible_select()
    if patched_select is not None:
        ws_client.start()
    else:
        # Fallback: SDK internal API changed; run WS client in a daemon thread
        # and poll the shutting-down flag on the main thread.
        t = threading.Thread(target=ws_client.start, daemon=True, name="ws-client")
        t.start()
        while not is_shutting_down():
            time.sleep(1.0)

    _run_shutdown_sequence()


if __name__ == "__main__":
    main()
