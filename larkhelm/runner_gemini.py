"""larkhelm · GeminiRunner — Gemini CLI subprocess runner."""
import time

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.chat_state import _save_sid
from larkhelm.runner_base import BaseProcessRunner, _truncate_tool_result


class GeminiRunner(BaseProcessRunner):
    def __init__(
        self,
        chat_id: str,
        message: str,
        sid: str | None,
        cwd: str,
        *,
        cancel_ev=None,
        on_text=None,
        on_tool=None,
        on_tool_result=None,
        on_soft_timeout=None,
        allow_retry: bool = False,
        images: list | None = None,
        session_namespace: str | None = None,
        use_session: bool = True,
        record_under: str | None = None,
        command: str | None = None,
        model: str | None = None,
        extra_args: list | None = None,
        session_key: str | None = None,
    ) -> None:
        super().__init__(
            "gemini", chat_id, message, sid, cwd,
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            allow_retry=allow_retry, images=images, session_namespace=session_namespace,
            use_session=use_session, record_under=record_under, command=command,
        )
        self._model = model
        self._extra_args = list(extra_args) if extra_args else []
        self._session_key = session_key or "gemini"
        self._ctor_kwargs = dict(
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            allow_retry=allow_retry, images=images, session_namespace=session_namespace,
            use_session=use_session, record_under=record_under, command=command,
            model=model, extra_args=extra_args, session_key=session_key,
        )

    @property
    def _no_checkpointing(self) -> bool:
        return "--no-checkpointing" in self._extra_args

    def build_args(self) -> list[str]:
        cmd = self.command or _cfg.GEMINI_CMD
        args = [cmd, "-y", "--skip-trust", "--output-format", "stream-json"]
        if self._model:
            args += ["-m", self._model]
        args += ["-p", self.message]
        if self.sid and not self._no_checkpointing:
            args += ["--resume", self.sid]
        # Filter out larkhelm-internal flags that gemini CLI doesn't understand
        _INTERNAL_FLAGS = {"--no-checkpointing"}
        args += [a for a in self._extra_args if a not in _INTERNAL_FLAGS]
        _debug_log(
            f"[gemini] starting query cwd={self.cwd} resume={self.sid} "
            f"use_session={self.use_session} cmd={cmd} model={self._model or '(default)'}"
        )
        return args

    def build_stdin(self) -> None:
        return None

    def _on_kill_signal(self) -> None:
        if not self._soft_timeout_flag.is_set():
            raise TimeoutError(f"Gemini response timeout (>{_cfg.RESPONSE_TIMEOUT}s)")

    def _handle_non_json_stdout(self, line: str) -> None:
        if "Using FileKeychain fallback" not in line:
            _debug_log(f"[gemini] non-JSON STDOUT: {line[:200]}")

    def parse_stdout_event(self, ev: dict) -> bool:
        etype = ev.get("type", "")
        if etype in ("system", "init"):
            cand = ev.get("session_id")
            if cand:
                self._new_sid = cand
                if self.use_session and not self._no_checkpointing:
                    _save_sid(self.chat_id, cand, self._session_key)
        elif etype == "message" and ev.get("role") == "assistant":
            content = ev.get("content", "")
            if isinstance(content, list):
                text_chunk = "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and "text" in p
                )
            else:
                text_chunk = str(content)
            if not ev.get("delta", False) and self._result_text and text_chunk.startswith(self._result_text):
                self._result_text = text_chunk
            else:
                self._result_text += text_chunk
            if self.on_text:
                self.on_text(self._result_text, status="typing")
        elif etype == "tool_use":
            tid = ev.get("tool_id", "")
            name = ev.get("tool_name", "?")
            params = ev.get("parameters", {})
            self._tool_start_times[tid] = time.monotonic()
            if self.on_tool:
                self.on_tool(name, str(params)[:120], tool_id=tid)
        elif etype == "tool_result":
            tid = ev.get("tool_id", "")
            res = str(ev.get("output", ""))
            is_err = ev.get("status") != "success"
            start_t = self._tool_start_times.get(tid, time.monotonic())
            elapsed = time.monotonic() - start_t
            if self.on_tool_result:
                self.on_tool_result(tid, _truncate_tool_result(res, is_err), is_err, elapsed)
        elif etype == "result":
            cand = ev.get("session_id")
            if cand:
                self._new_sid = cand
                if self.use_session and not self._no_checkpointing:
                    _save_sid(self.chat_id, cand, self._session_key)
            usage = ev.get("usage", {})
            cost = ev.get("total_cost_usd", 0.0)
            if usage:
                self._record_tokens("gemini", {
                    "input_tokens":  usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_read":    0,
                    "cache_create":  0,
                }, cost)
            return True
        return False

    def cleanup_extra(self) -> None:
        pass
