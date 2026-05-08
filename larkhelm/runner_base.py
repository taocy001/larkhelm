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

    @property
    def _ns(self) -> str:
        return self.session_namespace if self.session_namespace is not None else self.chat_id

    # ------------------------------------------------------------------ hooks

    def build_env(self) -> dict:
        return {**os.environ, "DBUS_SESSION_BUS_ADDRESS": "", "GCM_CREDENTIAL_STORAGE": "file"}

    def _on_stderr_line(self, line: str) -> None:
        pass

    def _on_kill_signal(self) -> None:
        raise TimeoutError(f"{self.backend_name} force-killed (>{_cfg.HARD_TIMEOUT}s)")

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

    def _drain_stderr(self) -> None:
        try:
            for line in self._proc.stderr:
                stripped = line.rstrip()
                if stripped:
                    if len(self._stderr_buf) < _MAX_STDERR_LINES:
                        self._stderr_buf.append(stripped)
                    self._on_stderr_line(stripped)
        except Exception:
            pass

    def _watch(self) -> None:
        hard_deadline = time.time() + _cfg.HARD_TIMEOUT
        soft_deadline = time.time() + _cfg.RESPONSE_TIMEOUT
        soft_fired = False
        while time.time() < hard_deadline:
            if self._completed.is_set():
                return
            if self.cancel_ev and self.cancel_ev.is_set():
                _debug_log(f"[{self.backend_name}] user cancelled")
                self._cancelled_flag.set()
                try:
                    self._proc.kill()
                except Exception:
                    pass
                return
            if not soft_fired and time.time() >= soft_deadline:
                soft_fired = True
                _debug_log(
                    f"[{self.backend_name}] soft timeout ({_cfg.RESPONSE_TIMEOUT}s), "
                    "releasing lock but keeping process running"
                )
                self._soft_timeout_flag.set()
                if self.on_soft_timeout:
                    try:
                        self.on_soft_timeout()
                    except Exception as e:
                        _debug_log(f"[{self.backend_name}] on_soft_timeout callback failed: {e}")
            time.sleep(0.3)
        _debug_log(f"[{self.backend_name}] hard timeout ({_cfg.HARD_TIMEOUT}s), force killing process")
        try:
            self._proc.kill()
        except Exception:
            pass

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
        try:
            self._proc = subprocess.Popen(args, **popen_kw)
        except FileNotFoundError:
            _ai_proc_sem.release()
            self._cleanup_tmp()
            raise RuntimeError(f"{self.backend_name} CLI not found: {args[0]}")
        except Exception:
            _ai_proc_sem.release()
            self._cleanup_tmp()
            raise

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
                self._proc.kill()
                _ai_proc_sem.release()
                self._cleanup_tmp()
                raise RuntimeError(f"{self.backend_name} stdin write failed: {e}")

        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

        if self.on_text:
            try:
                self.on_text("", status="init")
            except Exception as e:
                _debug_log(f"[{self.backend_name}] on_text init callback failed: {e}")

        _inc_active()
        try:
            for line in self._proc.stdout:
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
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            _ai_proc_sem.release()
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
