"""larkhelm · memory watchdog — monitors process RSS and exits gracefully on OOM.

Config key: ``memory_limit_mb`` (int, in config.json).
  If absent at startup, auto-detected via psutil and written back to config.json so
  the user can inspect and tune it.

Auto-detect formula: max(512, min(total_ram_mb // 8, 4096))
  4 GB →  512 MB  |  8 GB → 1024 MB  |  16 GB → 2048 MB
 24 GB → 3072 MB  | 32 GB → 4096 MB

Thresholds:
  Soft (SOFT_RATIO × limit_mb, default 80%): warn + gc.collect()
  Hard (100% of limit_mb): gc.collect(), wait HARD_COOLDOWN_SEC, re-check;
    if still over → SIGTERM self (bridge.py graceful-shutdown path).
    launchd KeepAlive=true auto-restarts the process.

Poll interval: CHECK_INTERVAL_SEC (default 30 s).
SIGTERM is debounced by _SIGTERM_DEBOUNCE_SEC (60 s) to prevent restart storms
in pathological cases where GC cannot reclaim memory fast enough.
"""
from __future__ import annotations

import gc
import os
import signal
import threading
import time

SOFT_RATIO = 0.80           # warn at 80% of limit
CHECK_INTERVAL_SEC = 30     # poll interval (seconds)
HARD_COOLDOWN_SEC = 5       # pause after GC before re-checking hard limit
_SIGTERM_DEBOUNCE_SEC = 60  # min seconds between successive SIGTERM calls


def detect_memory_limit_mb() -> int:
    """Return a sensible RSS cap based on system total RAM.

    Formula: max(512, min(total_ram_mb // 8, 4096))
    Falls back to 2048 MB if psutil is unavailable.
    """
    try:
        import psutil
        total_mb = psutil.virtual_memory().total // (1024 * 1024)
        return max(512, min(total_mb // 8, 4096))
    except Exception:
        return 2048


def _rss_mb() -> float:
    """Return current process RSS in MB, or 0.0 on error."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def start_memory_watchdog(limit_mb: int, interval: int = CHECK_INTERVAL_SEC) -> None:
    """Start a daemon thread that monitors RSS and triggers graceful restart on OOM.

    Args:
        limit_mb: Hard RSS limit in MB (from config or auto-detected).
        interval:  Poll interval in seconds (default: CHECK_INTERVAL_SEC = 30).
    """
    soft_mb = limit_mb * SOFT_RATIO
    _last_sigterm: list[float] = [0.0]  # mutable cell for closure

    def _loop() -> None:
        # Lazy import so the module can be imported before bridge fully starts.
        from larkhelm.log import _debug_log, safe_log

        while True:
            try:
                time.sleep(interval)
            except Exception:
                time.sleep(interval)

            try:
                rss = _rss_mb()
                if rss <= 0:
                    continue  # psutil unavailable; skip check

                if rss >= limit_mb:
                    # ── Hard limit hit ────────────────────────────────────────
                    from larkhelm.log import error
                    error(
                        f"[MemWatchdog] RSS {rss:.0f} MB ≥ hard limit {limit_mb} MB "
                        "— 运行 GC 后重新检测"
                    )
                    gc.collect(2)
                    time.sleep(HARD_COOLDOWN_SEC)
                    rss_after = _rss_mb()
                    if rss_after >= limit_mb:
                        now = time.monotonic()
                        if now - _last_sigterm[0] > _SIGTERM_DEBOUNCE_SEC:
                            _last_sigterm[0] = now
                            error(
                                f"[MemWatchdog] GC 后 RSS {rss_after:.0f} MB 仍 ≥ {limit_mb} MB "
                                "— 发送 SIGTERM 触发 launchd 自动重启"
                            )
                            os.kill(os.getpid(), signal.SIGTERM)
                    else:
                        _debug_log(
                            f"[MemWatchdog] GC 释放内存成功: {rss:.0f} → {rss_after:.0f} MB"
                        )
                elif rss >= soft_mb:
                    # ── Soft limit hit ────────────────────────────────────────
                    from larkhelm.log import warn
                    warn(
                        f"[MemWatchdog] RSS {rss:.0f} MB ≥ soft limit {soft_mb:.0f} MB "
                        f"({int(SOFT_RATIO*100)}% of {limit_mb} MB) — 触发预防性 GC"
                    )
                    gc.collect(2)

            except Exception as exc:
                # Watchdog must never die — swallow all errors
                try:
                    safe_log(f"[MemWatchdog] 监控循环异常: {exc}")
                except Exception:
                    pass

    threading.Thread(target=_loop, daemon=True, name="memory-watchdog").start()
