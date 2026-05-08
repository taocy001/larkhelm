"""larkhelm · ClaudeRunner — Claude CLI subprocess runner."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.chat_state import _save_sid
from larkhelm.perm import is_yolo
from larkhelm.runner_base import BaseProcessRunner, _truncate_tool_result


def _build_stream_json_input(text: str, image_paths: list[str]) -> str:
    """Build stdin for Claude --input-format stream-json with base64-encoded images."""
    import base64
    content = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = Path(path).suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        except Exception as e:
            _debug_log(f"[Image] failed to read {path}: {e}")
    content.append({"type": "text", "text": text})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}})


class ClaudeRunner(BaseProcessRunner):
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
            "claude", chat_id, message, sid, cwd,
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            on_start=on_start, allow_retry=allow_retry, images=images,
            session_namespace=session_namespace, command=command,
        )
        self._model = model
        self._extra_args = list(extra_args) if extra_args else []
        self._session_key = session_key or "claude"
        self._ctor_kwargs = dict(
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            on_start=on_start, allow_retry=allow_retry, images=images,
            session_namespace=session_namespace, command=command,
            model=model, extra_args=extra_args, session_key=session_key,
        )

    def build_args(self) -> list[str]:
        cmd = self.command or _cfg.CLAUDE_CMD
        args = [cmd, "--print", "--output-format", "stream-json", "--verbose"]
        if self._model:
            args += ["--model", self._model]
        if self.images:
            args += ["--input-format", "stream-json"]

        if _cfg.SKIP_PERMISSIONS:
            args.append("--dangerously-skip-permissions")
        else:
            hook_settings = {
                "permissions": {
                    "allow": [
                        "Bash(*)", "Read(*)", "Write(*)", "Edit(*)",
                        "Glob(*)", "Grep(*)", "WebFetch(*)", "WebSearch(*)",
                        "TodoWrite(*)", "TodoRead(*)", "Agent(*)",
                    ]
                },
                "hooks": {
                    "PreToolUse": [{
                        "hooks": [{
                            "type": "command",
                            "command": f"python3 {_cfg.PERM_HOOK_SCRIPT}",
                            "timeout": _cfg.RESPONSE_TIMEOUT,
                        }]
                    }]
                },
            }
            fd, settings_file = tempfile.mkstemp(prefix="feishu_claude_", suffix=".json")
            os.chmod(settings_file, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(hook_settings, f)
            self._tmp_files.append(settings_file)
            args += ["--settings", settings_file]

        try:
            base: dict = {}
            user_mcp_cfg = _cfg.config.get("mcp_config_file", "")
            if user_mcp_cfg and Path(user_mcp_cfg).exists():
                try:
                    base = json.loads(Path(user_mcp_cfg).read_text())
                except Exception as e:
                    _debug_log(f"[ai_runner] mcp_config_file parse failed ({user_mcp_cfg}): {e}")
            larkhelm_bin = str(Path(sys.executable).parent / "larkhelm")
            servers = dict(base.get("mcpServers", {}))
            servers["larkhelm"] = {
                "command": larkhelm_bin,
                "args": [
                    "mcp-server",
                    "--config", str(_cfg.CONFIG_PATH),
                    "--data-dir", str(_cfg.DATA_DIR),
                ],
            }
            fd2, mcp_file = tempfile.mkstemp(prefix="larkhelm_mcp_", suffix=".json")
            os.chmod(mcp_file, 0o600)
            with os.fdopen(fd2, "w") as f:
                json.dump({**base, "mcpServers": servers}, f)
            self._tmp_files.append(mcp_file)
            args += ["--mcp-config", mcp_file]
        except Exception as e:
            _debug_log(f"[MCP] failed to build mcp config: {e}")

        if self.sid:
            args += ["--resume", self.sid]

        args += self._extra_args

        _debug_log(
            f"[claude] starting cwd={self.cwd} sid={self.sid} "
            f"skip_perm={_cfg.SKIP_PERMISSIONS} images={len(self.images)} "
            f"ns={self._ns} cmd={cmd} model={self._model or '(default)'}"
        )
        return args

    def build_env(self) -> dict:
        ns = self._ns
        return {
            **os.environ,
            "DBUS_SESSION_BUS_ADDRESS": "",
            "GCM_CREDENTIAL_STORAGE": "file",
            "FEISHU_CHAT_ID": ns,
            "FEISHU_PERM_SOCKET": _cfg.PERM_SOCKET_PATH,
            "FEISHU_PERM_YOLO": "1" if is_yolo(ns) else "0",
        }

    def build_stdin(self) -> str | None:
        if self.images:
            return _build_stream_json_input(self.message, self.images)
        return self.message

    def parse_stdout_event(self, ev: dict) -> bool:
        etype = ev.get("type", "")
        if etype in ("system", "init"):
            cand = ev.get("session_id")
            if cand:
                self._new_sid = cand
                _save_sid(self._ns, cand, self._session_key)
        elif etype == "assistant":
            for block in ev.get("message", {}).get("content", []) or []:
                btype = block.get("type", "")
                if btype == "text":
                    chunk = block.get("text", "")
                    if chunk:
                        self._result_text += chunk
                    if self.on_text:
                        self.on_text(self._result_text, status="typing")
                elif btype == "tool_use":
                    tool_id = block.get("id", "")
                    name = block.get("name", "?")
                    inp = block.get("input", {})
                    self._tool_start_times[tool_id] = time.monotonic()
                    if self.on_tool:
                        self.on_tool(name, self._summarize_tool_input(name, inp), tool_id=tool_id)
                elif btype == "thinking":
                    pass
        elif etype == "tool_result":
            tool_id = ev.get("tool_use_id", "")
            content_raw = ev.get("content", "")
            is_error = bool(ev.get("is_error", False))
            if isinstance(content_raw, list):
                content_str = "".join(
                    b.get("text", "") for b in content_raw
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                content_str = str(content_raw)
            start_t = self._tool_start_times.get(tool_id, time.monotonic())
            elapsed = time.monotonic() - start_t
            if self.on_tool_result:
                self.on_tool_result(tool_id, _truncate_tool_result(content_str, is_error), is_error, elapsed)
        elif etype == "error":
            err_obj = ev.get("error", {})
            err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
            print(f"[claude stream error] {err_msg}", file=sys.stderr)
            if self.on_text:
                self.on_text(f"stream error: {err_msg}", status="error")
        elif etype == "result":
            new_sid = ev.get("session_id") or self._new_sid
            if new_sid:
                self._new_sid = new_sid
                _save_sid(self._ns, new_sid, self._session_key)
            usage = ev.get("usage", {})
            cost = ev.get("total_cost_usd", 0.0)
            if usage:
                self._record_tokens("claude", {
                    "input_tokens":  usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_read":    usage.get("cache_read_input_tokens", 0),
                    "cache_create":  usage.get("cache_creation_input_tokens", 0),
                }, cost)
            return True
        return False

    def cleanup_extra(self) -> None:
        pass

    def _on_stderr_line(self, line: str) -> None:
        if "Warning: Python" in line and "lockfile expects" in line:
            print(line, file=sys.stderr)
