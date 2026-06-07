"""larkhelm · BaseProcessRunner — shared subprocess spawn/stream/cleanup logic."""
from __future__ import annotations

import abc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.chat_state import _save_sid, _clear_sid
from larkhelm.runner_types import (
    OnText, OnTool, OnToolResult, OnSoftTimeout, OnStart, CancelEvent,
)

_MAX_STDERR_LINES = 100


class QueryCancelledError(Exception):
    """Raised when a query or crew agent is cancelled by the user or cancel_ev."""


# ── AI subprocess concurrency cap ─────────────────────────────────────────────
#
# The cap is *dynamic*: ``_init_ai_sem`` reads ``_cfg.MAX_AI_PROCS_CONFIG`` and
# either honours an explicit positive integer or runs ``_compute_max_procs`` to
# derive a memory-budget-aware value (cgroup MemoryMax → physical RAM fallback).
#
# Anti-staleness contract: code MUST acquire the semaphore via ``get_ai_sem()``
# at the *point of use*, not import-time. Pre-fix bug — ``from runner_base
# import _ai_proc_sem`` copied a binding into the importing module that
# survived ``_init_ai_sem`` re-creating the underlying object, so a chat
# could end up holding a slot in a sem with capacity-3 even after the
# runtime cap had been re-tightened to 2 by config or auto-detect. The fix
# is enforced by replacing every ``from ... import _ai_proc_sem`` consumer
# with ``get_ai_sem()`` calls, see ``ai_runner.py`` / ``runner_deepseek.py``
# / ``crew/_commands.py``.

MAX_AI_PROCS_DEFAULT = 2          # fallback when probing fails
# HARD_CEILING justification: WORKER_RSS_MB_DEFAULT (800) × 8 ≈ 6.4 GB peak AI
# RSS — already past the comfortable budget on commodity 8-GB hosts even
# without the bridge baseline. Beyond 8, the bottleneck shifts from memory
# to the Feishu card-update rate (one card per chat per ~0.3 s) and the
# per-chat lock contention; more parallel CLIs would just spend their time
# stalled on card edits. Treat as a "no host should ever exceed this"
# safety rail rather than a tunable.
HARD_CEILING = 8
WORKER_RSS_MB_DEFAULT = 800       # typical claude/kimi RSS during tool-burst (conservative)
BRIDGE_BASELINE_MB = 400          # python + lark_oapi + watchdog steady-state RSS
SAFETY_MARGIN_MB = 400            # extra buffer for cron / cleanup / transient spikes

_MAX_AI_PROCS = MAX_AI_PROCS_DEFAULT
_ai_proc_sem = threading.Semaphore(_MAX_AI_PROCS)
# Guards the ``_ai_proc_sem`` swap inside ``_init_ai_sem``. In practice
# ``_init_ai_sem`` only runs once at boot (called from ``config._init_runtime``)
# so contention is impossible — this lock is paranoia against a future caller
# that decides to re-probe at runtime. The hot read path (``get_ai_sem``)
# does NOT take this lock and relies on Python's atomic attribute load.
_ai_sem_lock = threading.Lock()

# Back-compat alias — some legacy code reads ``MAX_AI_PROCS`` directly. Kept
# as a *mirror* of ``_MAX_AI_PROCS``; assigned by ``_init_ai_sem`` and then
# treated as read-only. New code MUST use ``get_max_ai_procs()`` instead.
MAX_AI_PROCS = _MAX_AI_PROCS

_active_proc_count = 0
_active_proc_count_lock = threading.Lock()


def _detect_cgroup_memory_max() -> int | None:
    """Detect cgroup v2 ``memory.max`` for the *current* process.

    Walks ``/proc/self/cgroup`` to find the process's actual cgroup path,
    then probes ``/sys/fs/cgroup<path>/memory.max`` from the leaf up to
    root — returning the first concrete byte limit. The sentinel ``"max"``
    (= no limit) at a given level causes the walk to continue upward.

    Pre-fix bug: this function hard-coded
    ``/sys/fs/cgroup/memory.max`` (the *root* cgroup). On systemd-managed
    hosts the root file does not exist; the bridge runs under
    ``/system.slice/larkhelm.service``, whose ``memory.max`` lives at
    ``/sys/fs/cgroup/system.slice/larkhelm.service/memory.max``. The
    pre-fix function therefore always returned ``None`` and the prod-host
    2.8 GB cgroup limit was invisible to ``_compute_max_procs``.

    Returns the byte limit, or ``None`` for cgroup v1 hosts, unreadable
    filesystem, or genuine no-limit chains.
    """
    try:
        # Step 1 — resolve our cgroup path from /proc/self/cgroup.
        # v2 unified format: '0::<path>\n' (single line, hier_id=0, empty subsys).
        # v1 format: '<hier_id>:<subsys>:<path>\n' (multi-line, hier_id != 0).
        cgroup_path: str | None = None
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                    cgroup_path = parts[2]
                    break
        if cgroup_path is None:
            return None  # cgroup v1 host, or unrecognised format

        # Step 2 — generate candidate ``memory.max`` paths from leaf → root.
        # cgroup_path = '/system.slice/larkhelm.service' →
        #   /sys/fs/cgroup/system.slice/larkhelm.service/memory.max
        #   /sys/fs/cgroup/system.slice/memory.max
        #   /sys/fs/cgroup/memory.max
        rel = cgroup_path.strip("/")
        segs = rel.split("/") if rel else []
        candidates: list[str] = []
        for i in range(len(segs), -1, -1):
            sub = "/".join(segs[:i])
            candidates.append(
                "/sys/fs/cgroup" + ("/" + sub if sub else "") + "/memory.max"
            )

        # Step 3 — return the first concrete numeric limit; skip ``"max"``
        # sentinels and unreadable files, falling through to the next level.
        for cand in candidates:
            try:
                with open(cand, "r") as f:
                    val = f.read().strip()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if not val or val == "max":
                continue  # no limit at this level → walk upward
            try:
                return int(val)
            except ValueError:
                continue

        return None  # entire chain is unlimited or unreadable
    except Exception:
        return None


def _detect_physical_ram_mb() -> int:
    """Return physical RAM in MB.

    Linux: read ``MemTotal`` from /proc/meminfo.
    macOS / BSD: fall back to ``sysctl hw.memsize`` (the bridge runs locally
    on dev Macs too, where /proc/meminfo doesn't exist — without this the
    probe silently returned 4096 and capped MAX_AI_PROCS to 4 on a 24 GB box,
    *and* every claude SIGKILL got mislabelled "cgroup OOM").

    Final fallback is the conservative 4 GB that matches typical 4-GB VPS
    where the original OOM incidents occurred — guessing high would silently
    re-introduce the OOM.
    """
    # Linux — /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            first = f.readline()
        parts = first.split()  # "MemTotal:       16321448 kB"
        if len(parts) >= 2:
            return int(parts[1]) // 1024
    except (OSError, ValueError):
        pass

    # macOS / BSD — sysctl hw.memsize (bytes)
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip()) // (1024 * 1024)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    return 4096


def _compute_max_procs(worker_rss_mb: int = WORKER_RSS_MB_DEFAULT) -> tuple[int, str]:
    """Derive a memory-safe ``MAX_AI_PROCS`` value from the host's memory budget.

    Decision tree:
      1. ``cgroup memory.max`` → use as budget if present and finite.
      2. Otherwise → physical RAM (``MemTotal`` from /proc/meminfo).
      3. ``budget_available = budget - BRIDGE_BASELINE_MB - SAFETY_MARGIN_MB``.
      4. ``n = floor(budget_available / worker_rss_mb)``.
      5. Clamp to ``[1, HARD_CEILING]``.

    Returns ``(n, reason)`` where ``reason`` is a single human-readable line
    suitable for logging at INFO level.
    """
    cgroup_bytes = _detect_cgroup_memory_max()
    if cgroup_bytes is not None:
        available_mb = cgroup_bytes // (1024 * 1024)
        source = f"cgroup_max={available_mb / 1024:.1f}G"
    else:
        available_mb = _detect_physical_ram_mb()
        source = f"physical_ram={available_mb / 1024:.1f}G"

    budget_mb = available_mb - BRIDGE_BASELINE_MB - SAFETY_MARGIN_MB
    if budget_mb <= 0:
        # Tiny host (≤ 800 MB free after baselines) — floor to 1 anyway, but
        # surface that in the reason so operators investigate.
        reason = (f"{source}, budget={budget_mb}M (≤0!), worker_rss={worker_rss_mb}M "
                  f"→ floored to 1")
        return 1, reason

    n = budget_mb // worker_rss_mb
    n = max(1, min(HARD_CEILING, n))
    reason = (f"{source}, budget={budget_mb / 1024:.1f}G, "
              f"worker_rss={worker_rss_mb / 1024:.1f}G → {n}")
    return int(n), reason


def _init_ai_sem() -> None:
    """Resolve the effective concurrency cap and rebuild ``_ai_proc_sem``.

    Reads ``_cfg.MAX_AI_PROCS_CONFIG`` (set by ``config._init_runtime``).
    Values:

      * positive int → honoured verbatim (operator override)
      * ``None`` / 0 / negative / non-int → auto-detect via ``_compute_max_procs``

    Idempotent: if the resolved value already matches ``_MAX_AI_PROCS`` no
    new semaphore is constructed (avoids orphaning in-flight ``acquire``
    callers — though the caller protocol via ``get_ai_sem()`` makes that
    safe regardless).
    """
    global _MAX_AI_PROCS, _ai_proc_sem, MAX_AI_PROCS

    raw = getattr(_cfg, "MAX_AI_PROCS_CONFIG", None)
    # NB: ``bool`` is a subclass of ``int`` in Python, so ``max_ai_procs: true``
    # in JSON would otherwise be silently coerced to 1. We reject bools so the
    # user sees the warning path in ``config._init_runtime`` instead of a
    # mysterious cap of 1.
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        n, reason = raw, "config override"
    else:
        n, reason = _compute_max_procs()

    with _ai_sem_lock:
        if n == _MAX_AI_PROCS:
            # Update mirror just in case it drifted (paranoia; shouldn't happen)
            MAX_AI_PROCS = _MAX_AI_PROCS
            try:
                from larkhelm.log import info as _info
                _info(f"[Runner] MAX_AI_PROCS={_MAX_AI_PROCS} ({reason}) — unchanged")
            except Exception:
                pass
            return
        _MAX_AI_PROCS = n
        _ai_proc_sem = threading.Semaphore(n)
        MAX_AI_PROCS = n   # mirror — read-only after this point

    try:
        from larkhelm.log import info as _info
        _info(f"[Runner] MAX_AI_PROCS={n} ({reason})")
    except Exception:
        # Bootstrap path — log module may not be wired up yet.
        try:
            print(f"[Runner] MAX_AI_PROCS={n} ({reason})", file=sys.stderr)
        except Exception:
            pass


def get_ai_sem() -> threading.Semaphore:
    """Return the *current* AI process semaphore.

    All call sites that need to acquire/release the AI semaphore MUST go
    through this getter rather than ``from runner_base import _ai_proc_sem``
    so that ``_init_ai_sem`` rebuilds are visible everywhere — see the
    P0 staleness note at the top of this module.
    """
    return _ai_proc_sem


def get_max_ai_procs() -> int:
    """Return the current concurrency cap (replaces direct ``MAX_AI_PROCS`` reads)."""
    return _MAX_AI_PROCS


def active_proc_count() -> int:
    """Return number of AI subprocesses currently holding the semaphore."""
    with _active_proc_count_lock:
        return _active_proc_count


def _acquire_ai_sem(cancel_ev: threading.Event = None) -> threading.Semaphore:
    """Acquire the current AI process semaphore.

    Returns the *exact* semaphore instance that was acquired, so callers can
    release the same object even if ``_init_ai_sem`` rebuilds the live sem
    concurrently (idempotent in practice — rebuilds only happen at boot —
    but explicit symmetry beats relying on that invariant).

    Raises ``QueryCancelledError`` if ``cancel_ev`` fires before a slot opens.
    """
    sem = get_ai_sem()
    while not sem.acquire(timeout=1.0):
        if cancel_ev and cancel_ev.is_set():
            raise QueryCancelledError("cancelled while waiting for AI process slot")
        # Re-read in case _init_ai_sem swapped the global mid-wait (boot race).
        # In steady state this is the same object as before.
        sem = get_ai_sem()
    return sem


def _inc_active() -> None:
    global _active_proc_count
    with _active_proc_count_lock:
        _active_proc_count += 1


def _dec_active() -> None:
    global _active_proc_count
    with _active_proc_count_lock:
        _active_proc_count -= 1


def _truncate_tool_result(content: str, is_error: bool) -> str:
    """Non-error: first 200 chars to last newline. Error: last 200 chars."""
    if is_error:
        truncated = content[-200:]
        return ("...(truncated)\n" + truncated) if len(content) > 200 else truncated
    snippet = content[:200]
    nl = snippet.rfind("\n")
    return snippet[: nl + 1] if nl != -1 else snippet


class BaseProcessRunner(abc.ABC):
    """Abstract base for Claude / Kimi / Gemini subprocess runners.

    Subclasses implement build_args / build_stdin / parse_stdout_event / cleanup_extra.
    run() is the template method: semaphore → Popen → stdin → drain/watch → stdout loop → retry.
    """

    def __init__(
        self,
        backend_name: str,
        chat_id: str,
        message: str,
        sid: str | None,
        cwd: str,
        *,
        cancel_ev: CancelEvent | None = None,
        on_text: OnText | None = None,
        on_tool: OnTool | None = None,
        on_tool_result: OnToolResult | None = None,
        on_soft_timeout: OnSoftTimeout | None = None,
        on_start: OnStart | None = None,
        allow_retry: bool = False,
        images: list | None = None,
        session_namespace: str | None = None,
        command: str | None = None,
        use_session: bool = True,
        record_under: str | None = None,
        suppress_token_recording: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.chat_id = chat_id
        self.message = message
        self.sid = sid
        self.cwd = cwd
        self.cancel_ev = cancel_ev
        self.on_text = on_text
        self.on_tool = on_tool
        self.on_tool_result = on_tool_result
        self.on_soft_timeout = on_soft_timeout
        self.on_start = on_start
        self.allow_retry = allow_retry
        self.images: list = images or []
        self.session_namespace = session_namespace
        self.command = command
        self.use_session = use_session
        self.record_under = record_under
        self.suppress_token_recording = suppress_token_recording

        self._tmp_files: list[str] = []
        self._result_text: str = ""
        self._new_sid: str | None = None
        self._tool_start_times: dict[str, float] = {}
        self._stderr_buf: list[str] = []
        self._proc: subprocess.Popen | None = None
        self._completed = threading.Event()
        self._cancelled_flag = threading.Event()
        self._soft_timeout_flag = threading.Event()
        self._ctor_kwargs: dict = {}  # populated by subclass for _clone()
        # Partial-token bookkeeping for cancel / timeout paths
        # ----------------------------------------------------
        # Stats audit round-2 flagged that ``_record_tokens`` only fires
        # when ``parse_stdout_event`` sees the terminal ``result`` /
        # ``usage`` envelope — which never arrives if the user hits
        # /cancel or the idle hard-timeout kicks in mid-stream. Result:
        # every interrupted query contributed zero to ``token_stats``,
        # hiding real cost (a 300k-token /dev that ran 30 min then got
        # /cancel'd just vanished from /stats).
        #
        # Fix: each runner stashes the last ``usage`` dict it observes
        # in ``self._last_usage_seen`` and sets ``_tokens_recorded=True``
        # when ``_record_tokens`` actually fires. The default
        # ``cleanup_extra`` (called in the ``run()`` finally regardless
        # of cancel/timeout/normal-exit) checks the flag and, if still
        # False, writes a best-effort partial record:
        #   • if ``_last_usage_seen`` is non-empty → use those counts
        #     and flag the record ``estimated=True`` (since the stream
        #     was truncated and the SDK may have buffered more)
        #   • else → char-count fallback (len(message)/4 input,
        #     len(_result_text)/4 output) like the Kimi runner already
        #     does for its zero-usage CLI
        # Subclasses can override ``cleanup_extra`` if they need a
        # backend-specific fallback (e.g. Kimi already does char-count).
        self._last_usage_seen: dict | None = None
        self._tokens_recorded: bool = False
        # Idle-timeout tracking: ``_watch`` compares ``time.time() -
        # _last_activity_ts`` against RESPONSE_TIMEOUT / HARD_TIMEOUT, so a
        # task that keeps producing output (genuine 6-hour /dev pipeline,
        # long video gen, etc.) never trips the hard timeout. Only a truly
        # wedged subprocess (no stdout / stderr for the full window) is
        # killed. Pre-fix bug: the deadline was wall-clock-absolute, so
        # every 6-hour run died regardless of liveness.
        self._last_activity_ts: float = time.time()
        self._activity_lock = threading.Lock()
        # ``_watch_killed`` is set by ``_watch`` immediately before it calls
        # ``self._proc.kill()`` on its own initiative (idle hard timeout or
        # cancel_ev). ``_on_kill_signal`` reads this flag to distinguish:
        #   * True  → we killed it → ``TimeoutError`` (idle exceeded)
        #   * False → external SIGKILL (almost always cgroup OOM-killer
        #             targeting the node-based claude CLI whose total_vm
        #             routinely exceeds 20+ GB even with NODE_OPTIONS) →
        #             ``RuntimeError`` with "killed by OS" wording so the
        #             user sees the real cause instead of a misleading
        #             "执行超过 360 分钟" card.
        self._watch_killed: bool = False

    @property
    def _ns(self) -> str:
        return self.session_namespace if self.session_namespace is not None else self.chat_id

    # ------------------------------------------------------------------ hooks

    def build_env(self) -> dict:
        return {**os.environ, "DBUS_SESSION_BUS_ADDRESS": "", "GCM_CREDENTIAL_STORAGE": "file"}

    def _on_stderr_line(self, line: str) -> None:
        pass

    def _on_kill_signal(self) -> None:
        # Reached when the OS reports SIGKILL (-9). Two completely different
        # causes funnel through this signal:
        #
        #   A. ``_watch`` killed the subprocess intentionally (idle timeout
        #      exceeded, or cancel_ev raised). ``self._watch_killed`` is
        #      set just before that ``self._proc.kill()`` call. The user
        #      should see a timeout-shaped error.
        #
        #   B. The OS killed it with SIGKILL. The likely cause differs by
        #      platform, so the message branches on sys.platform — the old
        #      unconditional "cgroup OOM / systemctl / dmesg / node CLI"
        #      wording was Linux/systemd-specific and badly misled macOS dev
        #      users (no cgroup exists, claude is a native arm64 binary not
        #      node, and the box had 24 GB free when the kill fired).
        #      Pre-fix bug: this path raised TimeoutError too, so the user
        #      saw "执行超过 360 分钟" cards for what was actually a kill.
        if self._watch_killed:
            raise TimeoutError(
                f"{self.backend_name} force-killed (no output for ≥{_cfg.HARD_TIMEOUT}s)"
            )
        if sys.platform == "linux":
            raise RuntimeError(
                f"{self.backend_name} killed by OS (rc=-9, likely cgroup OOM). "
                "Check `systemctl status larkhelm` and dmesg for "
                "'task=claude ... oom-kill'. Probable causes: CLI virtual "
                "memory exceeded cgroup MemoryMax (default 2.8G), large file/"
                "image attachments expanding tool buffers, or runaway tool output."
            )
        # macOS / other: no cgroup. Most likely the kernel's memory-pressure
        # killer (jetsam) or the CLI process crashing. Point the operator at
        # the right diagnostics instead of Linux-only tooling.
        raise RuntimeError(
            f"{self.backend_name} killed by OS (rc=-9 / SIGKILL). On macOS this is "
            "usually the memory-pressure killer (jetsam) or a CLI crash, not a "
            "config OOM limit. Check Console.app / `log show --predicate "
            "'eventMessage CONTAINS \"jetsam\"'` and free memory; large "
            "attachments or runaway tool output can also trigger it."
        )

    def _handle_non_json_stdout(self, line: str) -> None:
        pass

    # ------------------------------------------------------------------ internals

    def _cleanup_tmp(self) -> None:
        for path in self._tmp_files:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    def _clone(self, **overrides) -> "BaseProcessRunner":
        kw = dict(self._ctor_kwargs)
        kw.update(overrides)
        sid = kw.pop("sid", self.sid)
        return type(self)(self.chat_id, self.message, sid, self.cwd, **kw)

    def _touch_activity(self) -> None:
        """Record a liveness signal — called from stdout / stderr drain loops.

        ``_watch`` resets its idle clock from this timestamp so a chatty
        subprocess never approaches the hard timeout.
        """
        with self._activity_lock:
            self._last_activity_ts = time.time()

    def _drain_stderr(self) -> None:
        try:
            for line in self._proc.stderr:
                # Any stderr byte is a liveness signal — even a "retry" /
                # "waiting on rate limit" log line proves the subprocess
                # hasn't wedged. Touching here BEFORE we discard empty
                # lines avoids a wedge that emits only whitespace.
                self._touch_activity()
                stripped = line.rstrip()
                if stripped:
                    if len(self._stderr_buf) < _MAX_STDERR_LINES:
                        self._stderr_buf.append(stripped)
                    self._on_stderr_line(stripped)
        except Exception:
            pass

    def _watch(self) -> None:
        """Idle-clock watcher. Timeout fires only when there's been no
        stdout / stderr / cancel poll for the configured window.

        Why idle, not wall-clock: a genuine long-running task (crew
        pipeline, /dev across many edits, 1-hour video transcode) emits
        progress continuously; the wall-clock version killed it the
        moment the elapsed window hit HARD_TIMEOUT regardless of
        liveness. The idle version kills a *wedged* subprocess — one
        that has stopped producing output entirely.

        ``RESPONSE_TIMEOUT`` (soft) and ``HARD_TIMEOUT`` (hard) keep
        their original semantics: soft → release the chat lock so the
        user can continue talking, hard → kill the process. Only the
        time origin changes (from start-time to last-activity).
        """
        soft_fired = False
        while True:
            if self._completed.is_set():
                return
            if self.cancel_ev and self.cancel_ev.is_set():
                _debug_log(f"[{self.backend_name}] user cancelled")
                self._cancelled_flag.set()
                self._watch_killed = True  # see _on_kill_signal — flags this as self-kill
                try:
                    self._proc.kill()
                except Exception:
                    pass
                return
            with self._activity_lock:
                idle = time.time() - self._last_activity_ts
            if idle >= _cfg.HARD_TIMEOUT:
                _debug_log(
                    f"[{self.backend_name}] hard idle timeout ({idle:.0f}s "
                    f"≥ {_cfg.HARD_TIMEOUT}s without output), force killing process"
                )
                self._watch_killed = True  # see _on_kill_signal — flags this as self-kill
                try:
                    self._proc.kill()
                except Exception:
                    pass
                return
            if not soft_fired and idle >= _cfg.RESPONSE_TIMEOUT:
                soft_fired = True
                _debug_log(
                    f"[{self.backend_name}] soft idle timeout ({idle:.0f}s "
                    f"≥ {_cfg.RESPONSE_TIMEOUT}s), releasing lock but keeping process running"
                )
                self._soft_timeout_flag.set()
                if self.on_soft_timeout:
                    try:
                        self.on_soft_timeout()
                    except Exception as e:
                        _debug_log(f"[{self.backend_name}] on_soft_timeout callback failed: {e}")
            # If the subprocess starts producing again after a soft fire,
            # don't re-arm soft — releasing the lock twice would confuse
            # the outer ``_do_query`` flow. Hard still applies.
            time.sleep(0.3)

    def _record_tokens(self, model: str, usage: dict, cost: float) -> None:
        self._tokens_recorded = True
        full_usage = {**usage, "cost_usd": cost}
        # Always update so ai_runner can extract via usage_holder after run()
        self._last_usage_seen = full_usage
        if getattr(self, 'suppress_token_recording', False):
            return  # recording deferred to caller (_synthesize)
        try:
            from larkhelm.token_stats import record_token_usage, resolve_record_chat_id
            record_id = resolve_record_chat_id(self.chat_id, self.record_under)
            record_token_usage(record_id, model, full_usage)
        except Exception as e:
            _debug_log(f"[{self.backend_name}] token_stats update failed: {e}")
        if "__crew_" in self.chat_id:
            try:
                from larkhelm.token_stats import record_crew_agent_tokens
                record_crew_agent_tokens(self.chat_id, model, full_usage)
            except Exception as e:
                _debug_log(f"[{self.backend_name}] token_stats update failed: {e}")

    def record_partial_tokens_if_needed(self, model_label: str) -> None:
        """Cancel / timeout safety net: persist whatever token data we have
        if the terminal ``result`` envelope never arrived.

        Called from ``cleanup_extra`` by runners that don't have their
        own backend-specific fallback (Kimi's char-count path already
        does its own thing inside ``cleanup_extra``; this helper is for
        Claude / Gemini / DeepSeek). Three branches:

          1. ``_tokens_recorded`` already True → terminal envelope made
             it through, nothing to do.
          2. ``_last_usage_seen`` populated from an intermediate event
             → record it with ``estimated=True`` (SDK may have buffered
             more deltas we never saw).
          3. Nothing observed but message / result_text non-empty →
             char-count fallback, ``estimated=True``.

        Safe to call multiple times — the flag short-circuits re-entries.
        """
        if getattr(self, 'suppress_token_recording', False):
            return  # recording deferred to caller
        if self._tokens_recorded:
            return
        usage = self._last_usage_seen
        try:
            if usage:
                payload = dict(usage)
                payload["estimated"] = True
                self._record_tokens(model_label, payload, 0.0)
                _debug_log(
                    f"[{self.backend_name}] partial tokens recorded from "
                    f"last_usage_seen (cancel/timeout): {payload}"
                )
                return
            msg = getattr(self, "message", "") or ""
            result_text = self._result_text or ""
            if not msg and not result_text:
                return
            self._record_tokens(model_label, {
                "input_tokens":  max(0, len(msg) // 4),
                "output_tokens": max(0, len(result_text) // 4),
                "cache_read":    0,
                "cache_create":  0,
                "estimated":     True,
            }, 0.0)
            _debug_log(
                f"[{self.backend_name}] partial tokens estimated from char "
                f"count (cancel/timeout)"
            )
        except Exception as e:
            _debug_log(f"[{self.backend_name}] partial-token record failed: {e}")

    @staticmethod
    def _summarize_tool_input(name: str, inp: dict) -> str:
        if isinstance(inp, dict):
            if name == "Bash" and "command" in inp:
                return inp["command"][:200]
            if name in ("Read", "Write", "Edit") and "file_path" in inp:
                return inp["file_path"]
            if name == "Glob" and "pattern" in inp:
                return inp["pattern"]
            if name == "Grep" and "pattern" in inp:
                return inp["pattern"]
            if name == "TodoWrite" and "todos" in inp:
                todos = inp["todos"]
                if isinstance(todos, list) and todos:
                    first = (
                        todos[0].get("content", "")[:40]
                        if isinstance(todos[0], dict)
                        else str(todos[0])[:40]
                    )
                    return f"{first}{'…' if len(todos) > 1 else ''} ({len(todos)} items)"
                return str(todos)[:80]
            if name == "Agent" and "prompt" in inp:
                return inp["prompt"][:100]
            return ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(inp.items())[:2])
        return str(inp)[:120]

    # ------------------------------------------------------------------ template method

    def run(self) -> str:
        args = self.build_args()
        env = self.build_env()
        stdin_content = self.build_stdin()

        popen_kw: dict = dict(
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=self.cwd, env=env,
        )
        if stdin_content is not None:
            popen_kw["stdin"] = subprocess.PIPE

        sem_acquired = _acquire_ai_sem(self.cancel_ev)
        sem_held = True
        active_inc = False
        try:
            try:
                self._proc = subprocess.Popen(args, **popen_kw)
            except FileNotFoundError:
                raise RuntimeError(f"{self.backend_name} CLI not found: {args[0]}")

            if self.on_start:
                try:
                    self.on_start()
                except Exception as e:
                    _debug_log(f"[{self.backend_name}] on_start callback failed: {e}")

            if stdin_content is not None:
                try:
                    self._proc.stdin.write(stdin_content + "\n")
                    self._proc.stdin.close()
                except OSError as e:
                    raise RuntimeError(f"{self.backend_name} stdin write failed: {e}")

            threading.Thread(target=self._drain_stderr, daemon=True).start()
            threading.Thread(target=self._watch, daemon=True).start()

            _inc_active()
            active_inc = True

            if self.on_text:
                try:
                    self.on_text("", status="init")
                except Exception as e:
                    _debug_log(f"[{self.backend_name}] on_text init callback failed: {e}")

            for line in self._proc.stdout:
                # Liveness signal BEFORE the strip-and-skip path so even
                # blank / non-JSON lines (e.g. status comments from some
                # CLI variants) count — the only thing the watcher cares
                # about is whether the pipe is still emitting.
                self._touch_activity()
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    self._handle_non_json_stdout(line)
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.parse_stdout_event(ev):
                    break
        finally:
            self._completed.set()
            if self._proc:
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # We're killing it ourselves (cleanup after the read loop
                    # ended), so flag it — otherwise the resulting rc=-9 would
                    # funnel into _on_kill_signal's external-SIGKILL branch and
                    # mislabel an already-completed query as an OOM kill.
                    self._watch_killed = True
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    except Exception:
                        pass
            if sem_held:
                # Release the *same* sem instance we acquired — not whatever
                # get_ai_sem() returns now. _init_ai_sem() can swap the live
                # sem (only at startup, but be defensive); releasing the new
                # one would inflate its capacity by 1 and leak the old slot.
                sem_acquired.release()
            if active_inc:
                _dec_active()
            self._cleanup_tmp()
            self.cleanup_extra()

        if self._cancelled_flag.is_set():
            raise QueryCancelledError("query cancelled")

        rc = self._proc.returncode
        # Only treat SIGKILL as fatal when we have no result. If the read loop
        # already parsed a complete result and broke out, a lingering process
        # killed during cleanup (finally block) is harmless — don't surface it
        # as an OOM/timeout error and discard a perfectly good answer.
        if rc == -9 and not self._result_text:
            self._on_kill_signal()

        if rc != 0 and not self._result_text:
            _debug_log(
                f"[{self.backend_name}] abnormal exit rc={rc}\n"
                + "\n".join(self._stderr_buf[-10:])
            )
            if self.allow_retry and self.sid:
                _debug_log(f"[{self.backend_name}] clearing session and retrying")
                _clear_sid(self._ns, self.backend_name)
                return self._clone(sid=None, allow_retry=False).run()
            raise RuntimeError(
                f"{self.backend_name} exited abnormally (rc={rc})\n"
                + ("\n".join(self._stderr_buf[-5:]) if self._stderr_buf else "no error output")
            )

        return self._result_text.strip()

    # ------------------------------------------------------------------ abstract

    @abc.abstractmethod
    def build_args(self) -> list[str]: ...

    @abc.abstractmethod
    def build_stdin(self) -> str | None: ...

    @abc.abstractmethod
    def parse_stdout_event(self, ev: dict) -> bool: ...

    @abc.abstractmethod
    def cleanup_extra(self) -> None: ...
