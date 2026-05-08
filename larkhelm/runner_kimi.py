"""larkhelm · KimiRunner — Kimi CLI subprocess runner."""
import json
import time
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.chat_state import _save_sid
from larkhelm.runner_base import BaseProcessRunner, _truncate_tool_result

_KIMI_TOOL_MAP = {"Shell": "Bash", "FetchURL": "WebFetch", "SearchWeb": "WebSearch"}


def _build_kimi_stream_input(text: str, image_paths: list[str]) -> str:
    """Build multimodal stdin for Kimi --input-format stream-json."""
    import base64
    content: list[dict] = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = Path(path).suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
        except Exception as e:
            _debug_log(f"[Kimi Image] failed to read {path}: {e}")
    content.append({"type": "text", "text": text})
    return json.dumps({"role": "user", "content": content})


class KimiRunner(BaseProcessRunner):
    _KIMI_TOOL_MAP = _KIMI_TOOL_MAP

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
        on_start=None,
        allow_retry: bool = False,
        images: list | None = None,
        session_namespace: str | None = None,
        command: str | None = None,
        model: str | None = None,
        extra_args: list | None = None,
        session_key: str | None = None,
    ) -> None:
        super().__init__(
            "kimi", chat_id, message, sid, cwd,
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            on_start=on_start, allow_retry=allow_retry, images=images,
            session_namespace=session_namespace, command=command,
        )
        self._model = model
        self._extra_args = list(extra_args) if extra_args else []
        self._session_key = session_key or "kimi"
        self._ctor_kwargs = dict(
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            on_start=on_start, allow_retry=allow_retry, images=images,
            session_namespace=session_namespace, command=command,
            model=model, extra_args=extra_args, session_key=session_key,
        )

    def build_args(self) -> list[str]:
        cmd = self.command or _cfg.KIMI_CMD
        args = [
            cmd, "--print", "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            "--work-dir", self.cwd,
        ]
        if self._model:
            args += ["--model", self._model]
        if self.sid:
            args += ["--session", self.sid]
        if _cfg.SKIP_PERMISSIONS:
            args += ["--yolo"]
        args += self._extra_args
        _debug_log(
            f"[kimi] starting cwd={self.cwd} sid={self.sid} "
            f"skip_perm={_cfg.SKIP_PERMISSIONS} images={len(self.images)} "
            f"ns={self._ns} cmd={cmd} model={self._model or '(default)'}"
        )
        return args

    def build_stdin(self) -> str | None:
        if self.images:
            return _build_kimi_stream_input(self.message, self.images)
        return json.dumps({"role": "user", "content": self.message})

    def parse_stdout_event(self, ev: dict) -> bool:
        cand_sid = ev.get("session_id") or ev.get("session")
        if cand_sid and not self._new_sid:
            self._new_sid = cand_sid
            _save_sid(self._ns, cand_sid, self._session_key)

        role = ev.get("role", "")
        etype = ev.get("type", "")

        if role == "assistant":
            content = ev.get("content", "")
            tool_calls = ev.get("tool_calls") or []

            if content and isinstance(content, str):
                self._result_text += content
                if self.on_text:
                    self.on_text(self._result_text, status="typing")
            elif content and isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                chunk = "".join(text_parts)
                if chunk:
                    self._result_text += chunk
                    if self.on_text:
                        self.on_text(self._result_text, status="typing")

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                kimi_name = func.get("name", "?")
                name = self._KIMI_TOOL_MAP.get(kimi_name, kimi_name)
                try:
                    inp = json.loads(func.get("arguments", "{}"))
                except Exception:
                    inp = {}
                self._tool_start_times[tc_id] = time.monotonic()
                if self.on_tool:
                    self.on_tool(name, self._summarize_tool_input(name, inp), tool_id=tc_id)

        elif role == "tool":
            tc_id = ev.get("tool_call_id", "")
            content = ev.get("content", "")
            is_error = bool(ev.get("is_error", False))
            start_t = self._tool_start_times.get(tc_id, time.monotonic())
            elapsed = time.monotonic() - start_t
            if self.on_tool_result:
                self.on_tool_result(tc_id, _truncate_tool_result(str(content), is_error), is_error, elapsed)

        elif etype == "result" or role == "result":
            usage = ev.get("usage", {})
            cost = ev.get("total_cost_usd", 0.0)
            if usage:
                self._record_tokens("kimi", {
                    "input_tokens":  usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                    "cache_read":    0,
                    "cache_create":  0,
                }, cost)
            return True

        return False

    def cleanup_extra(self) -> None:
        pass
