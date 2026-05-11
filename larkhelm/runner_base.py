"""larkhelm · BaseProcessRunner — shared subprocess spawn/stream/cleanup logic."""
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

_MAX_STDERR_LINES = 100


class QueryCancelledError(Exception):
    """Raised when a query or crew agent is cancelled by the user or cancel_ev."""


MAX_AI_PROCS = 3
_ai_proc_sem = threading.Semaphore(MAX_AI_PROCS)
_active_proc_count = 0
_active_proc_count_lock = threading.Lock()


def active_proc_count() -> int:
    """Return number of AI subprocesses currently holding the semaphore."""
    with _active_proc_count_lock:
        return _active_proc_count


def _acquire_ai_sem(cancel_ev: threading.Event = None) -> None:
    """Acquire AI process semaphore. Raises QueryCancelledError if cancel_ev fires first."""
    while not _ai_proc_sem.acquire(timeout=1.0):
        if cancel_ev and cancel_ev.is_set():
            raise QueryCancelledError("cancelled while waiting for AI process slot")


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
        cancel_ev: threading.Event | None = None,
        on_text=None,
        on_tool=None,
        on_tool_result=None,
        on_soft_timeout=None,
        on_start=None,
        allow_retry: bool = False,
        images: list | None = None,
        session_namespace: str | None = None,
        command: str | None = None,
        use_session: bool = True,
        record_under: str | None = None,
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
        #   B. The OS / cgroup OOM-killer killed it. cgroup MemoryMax on
        #      the larkhelm.service is typically 2.8G, and the node-based
        #      claude CLI's *virtual* memory routinely exceeds 20 GB even
        #      with NODE_OPTIONS=--max-old-space-size=384 (V8 only caps
        #      the old-gen heap, not buffer pools / mmap / JIT cache). The
        #      kernel selects the largest task in the cgroup — almost
        #      always the CLI subprocess, not the python bridge — as the
        #      OOM victim. Pre-fix bug: this path raised TimeoutError too,
        #      so the user saw "执行超过 360 分钟" cards for what was
        #      actually an OOM kill 5 minutes into the task.
        if self._watch_killed:
            raise TimeoutError(
                f"{self.backend_name} force-killed (no output for ≥{_cfg.HARD_TIMEOUT}s)"
            )
        raise RuntimeError(
            f"{self.backend_name} killed by OS (rc=-9, likely cgroup OOM). "
            "Check `systemctl status larkhelm` and dmesg for "
            "'task=claude ... oom-kill'. Probable causes: node CLI virtual "
            "memory exceeded cgroup MemoryMax (default 2.8G), large file/"
            "image attachments expanding tool buffers, or runaway tool output."
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
        if self.record_under:
            record_id = self.record_under
        elif "__crew_" in self.chat_id:
            record_id = self.chat_id.split("__crew_")[0]
        else:
            record_id = self.chat_id
        full_usage = {**usage, "cost_usd": cost}
        try:
            from larkhelm.token_stats import record_token_usage
            record_token_usage(record_id, model, full_usage)
        except Exception as e:
            _debug_log(f"[{self.backend_name}] token_stats update failed: {e}")
        if "__crew_" in self.chat_id:
            try:
                from larkhelm.token_stats import record_crew_agent_tokens
                record_crew_agent_tokens(self.chat_id, model, full_usage)
            except Exception as e:
                _debug_log(f"[{self.backend_name}] token_stats update failed: {e}")

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

        _acquire_ai_sem(self.cancel_ev)
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
                    try:
                        self._proc.kill()
                        self._proc.wait()
                    except Exception:
                        pass
            if sem_held:
                _ai_proc_sem.release()
            if active_inc:
                _dec_active()
            self._cleanup_tmp()
            self.cleanup_extra()

        if self._cancelled_flag.is_set():
            raise QueryCancelledError("query cancelled")

        rc = self._proc.returncode
        if rc == -9:
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
