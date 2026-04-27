"""
larkhelm · AI subprocess runner

Contains:
  - _spawn_claude_proc()   Low-level Claude subprocess (formerly _run_claude, adds session_namespace param)
  - query_claude()         Public interface
  - query_gemini()         Gemini query
  - _build_stream_json_input()  Build multimodal stdin input
  - _truncate_tool_result()     Truncate tool results
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_MAX_STDERR_LINES = 100  # cap stderr drain buffers to prevent unbounded string accumulation

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.chat_state import _load_sid, _save_sid, _clear_sid
from larkhelm.perm import is_yolo

class QueryCancelledError(Exception):
    """Raised when a query or crew agent is cancelled by the user or cancel_ev."""


# Global concurrency limit: shared by all AI subprocesses (normal queries + crew agents + combined)
# On a 3.8 GB machine each Claude process uses ~300-350 MB; cap at 3 to avoid swap thrashing
MAX_AI_PROCS = 3
_ai_proc_sem = threading.Semaphore(MAX_AI_PROCS)

# Explicit counter to avoid relying on the private Semaphore._value attribute
_active_proc_count = 0
_active_proc_count_lock = threading.Lock()


def active_proc_count() -> int:
    """Return the number of AI subprocesses currently holding the semaphore (used by wait_for_idle)."""
    with _active_proc_count_lock:
        return _active_proc_count


def _acquire_ai_sem(cancel_ev: threading.Event = None) -> None:
    """Acquire the AI process semaphore. Raises QueryCancelledError if cancel_ev fires before a slot opens."""
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
    """Truncate tool result per spec: non-error keeps first 200 chars up to last newline; error keeps last 200 chars."""
    if is_error:
        truncated = content[-200:]
        return ("...(truncated)\n" + truncated) if len(content) > 200 else truncated
    else:
        snippet = content[:200]
        nl = snippet.rfind('\n')
        return snippet[:nl + 1] if nl != -1 else snippet


def _build_stream_json_input(text: str, image_paths: list[str]) -> str:
    """Build stdin input for Claude --input-format stream-json, supporting base64-encoded image content blocks."""
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


def _spawn_claude_proc(
    chat_id: str,
    message: str,
    sid: str,
    cwd: str,
    cancel_ev: threading.Event = None,
    on_text=None,
    on_tool=None,
    on_tool_result=None,
    allow_retry: bool = False,
    on_soft_timeout=None,
    on_start=None,
    images: list = None,
    session_namespace: str = None,
) -> str:
    """
    Spawn a Claude CLI subprocess and stream its output.

    session_namespace: If provided, use this namespace for session isolation (used by crew agents).
                       When None, chat_id is used as the namespace (normal queries).
    """
    # Session namespace determines where the sid is stored
    ns = session_namespace if session_namespace is not None else chat_id

    args = [_cfg.CLAUDE_CMD, "--print", "--output-format", "stream-json", "--verbose"]
    if images:
        args += ["--input-format", "stream-json"]
    settings_file: str = None
    if _cfg.SKIP_PERMISSIONS:
        args.append("--dangerously-skip-permissions")
    else:
        hook_settings_obj = {
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
            }
        }
        # Use a unique temp file per subprocess invocation to avoid races between concurrent queries
        fd, settings_file = tempfile.mkstemp(prefix="feishu_claude_", suffix=".json")
        os.chmod(settings_file, 0o600)
        with os.fdopen(fd, "w") as _f:
            json.dump(hook_settings_obj, _f)
        args += ["--settings", settings_file]
    if sid:
        args += ["--resume", sid]

    # MCP configuration
    _mcp_cfg = _cfg.config.get("mcp_config_file", "")
    if _mcp_cfg and Path(_mcp_cfg).exists():
        args += ["--mcp-config", _mcp_cfg]

    _debug_log(f"[claude] starting cwd={cwd} sid={sid} skip_perm={_cfg.SKIP_PERMISSIONS} "
               f"images={len(images) if images else 0} ns={ns}")
    env = {
        **os.environ,
        "DBUS_SESSION_BUS_ADDRESS": "",
        "GCM_CREDENTIAL_STORAGE": "file",
        "FEISHU_CHAT_ID": ns,
        "FEISHU_PERM_SOCKET": _cfg.PERM_SOCKET_PATH,
        "FEISHU_PERM_YOLO": "1" if is_yolo(ns) else "0",
    }

    _acquire_ai_sem(cancel_ev)
    try:
        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=cwd, env=env
        )
    except FileNotFoundError:
        _ai_proc_sem.release()
        if settings_file:
            Path(settings_file).unlink(missing_ok=True)
        raise RuntimeError(f"Claude CLI not found: {_cfg.CLAUDE_CMD}")
    except Exception:
        _ai_proc_sem.release()
        if settings_file:
            Path(settings_file).unlink(missing_ok=True)
        raise
    # Notify caller: process slot acquired, timeout countdown can begin
    if on_start:
        try:
            on_start()
        except Exception:
            pass

    try:
        if images:
            stdin_content = _build_stream_json_input(message, images)
        else:
            stdin_content = message
        proc.stdin.write(stdin_content + "\n")
        proc.stdin.close()
    except OSError as e:
        proc.kill()
        _ai_proc_sem.release()
        if settings_file:
            Path(settings_file).unlink(missing_ok=True)
        raise RuntimeError(f"Claude stdin write failed: {e}")

    stderr_buf: list[str] = []

    def _drain():
        try:
            for line in proc.stderr:
                stripped = line.rstrip()
                if stripped:
                    if len(stderr_buf) < _MAX_STDERR_LINES:
                        stderr_buf.append(stripped)
                    if "Warning: Python" in stripped and "lockfile expects" in stripped:
                        print(stripped, file=sys.stderr)
        except Exception:
            pass

    threading.Thread(target=_drain, daemon=True).start()

    completed = threading.Event()
    cancelled_flag = threading.Event()
    soft_timeout_flag = threading.Event()

    def _watch():
        hard_deadline = time.time() + _cfg.HARD_TIMEOUT
        soft_deadline = time.time() + _cfg.RESPONSE_TIMEOUT
        soft_fired = False
        while time.time() < hard_deadline:
            if completed.is_set():
                return
            if cancel_ev and cancel_ev.is_set():
                _debug_log("[claude] user cancelled")
                cancelled_flag.set()
                try:
                    proc.kill()
                except Exception:
                    pass
                return
            if not soft_fired and time.time() >= soft_deadline:
                soft_fired = True
                _debug_log(f"[claude] soft timeout ({_cfg.RESPONSE_TIMEOUT}s), releasing lock but keeping process running")
                soft_timeout_flag.set()
                if on_soft_timeout:
                    try:
                        on_soft_timeout()
                    except Exception:
                        pass
            time.sleep(0.3)
        _debug_log(f"[claude] hard timeout ({_cfg.HARD_TIMEOUT}s), force killing process")
        try:
            proc.kill()
        except Exception:
            pass

    threading.Thread(target=_watch, daemon=True).start()

    result_text, new_sid = "", None
    _tool_start_times: dict[str, float] = {}
    if on_text:
        on_text("", status="init")

    _inc_active()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type", "")
            if etype in ("system", "init"):
                cand = ev.get("session_id")
                if cand:
                    new_sid = cand
                    _save_sid(ns, new_sid, "claude")
            elif etype == "assistant":
                for block in ev.get("message", {}).get("content", []) or []:
                    btype = block.get("type", "")
                    if btype == "text":
                        chunk = block.get("text", "")
                        if chunk:
                            result_text += chunk
                        if on_text:
                            on_text(result_text, status="typing")
                    elif btype == "tool_use":
                        tool_id = block.get("id", "")
                        name = block.get("name", "?")
                        inp = block.get("input", {})
                        _tool_start_times[tool_id] = time.monotonic()
                        if on_tool:
                            if isinstance(inp, dict):
                                if name == "Bash" and "command" in inp:
                                    summary = inp["command"][:200]
                                elif name in ("Read", "Write", "Edit") and "file_path" in inp:
                                    summary = inp["file_path"]
                                elif name == "Glob" and "pattern" in inp:
                                    summary = inp["pattern"]
                                elif name == "Grep" and "pattern" in inp:
                                    summary = inp["pattern"]
                                elif name == "TodoWrite" and "todos" in inp:
                                    todos = inp["todos"]
                                    if isinstance(todos, list) and todos:
                                        first = todos[0].get("content", "")[:40] if isinstance(todos[0], dict) else str(todos[0])[:40]
                                        summary = f"{first}{'…' if len(todos) > 1 else ''} ({len(todos)} items)"
                                    else:
                                        summary = str(todos)[:80]
                                elif name == "Agent" and "prompt" in inp:
                                    summary = inp["prompt"][:100]
                                else:
                                    summary = ", ".join(
                                        f"{k}={repr(v)[:40]}"
                                        for k, v in list(inp.items())[:2]
                                    )
                            else:
                                summary = str(inp)[:120]
                            on_tool(name, summary, tool_id=tool_id)
                    elif btype == "thinking":
                        pass  # thinking blocks are not shown to the user
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
                start_t = _tool_start_times.get(tool_id, time.monotonic())
                elapsed = time.monotonic() - start_t
                truncated = _truncate_tool_result(content_str, is_error)
                if on_tool_result:
                    on_tool_result(tool_id, truncated, is_error, elapsed)
            elif etype == "error":
                err_obj = ev.get("error", {})
                err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                print(f"[claude stream error] {err_msg}", file=sys.stderr)
                if on_text:
                    on_text(f"stream error: {err_msg}", status="error")
            elif etype == "result":
                new_sid = ev.get("session_id") or new_sid
                if new_sid:
                    _save_sid(ns, new_sid, "claude")
                # Token usage accounting: map crew namespace back to the real chat_id (strip __crew_* suffix)
                record_id = chat_id.split("__crew_")[0] if "__crew_" in chat_id else chat_id
                usage = ev.get("usage", {})
                cost  = ev.get("total_cost_usd", 0.0)
                if usage:
                    try:
                        from larkhelm.token_stats import record_token_usage
                        record_token_usage(record_id, "claude", {
                            "input_tokens":  usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "cache_read":    usage.get("cache_read_input_tokens", 0),
                            "cache_create":  usage.get("cache_creation_input_tokens", 0),
                            "cost_usd":      cost,
                        })
                    except Exception:
                        pass
                    # Also track per crew-agent (for per-agent display in crew card)
                    if "__crew_" in chat_id:
                        try:
                            from larkhelm.token_stats import record_crew_agent_tokens
                            record_crew_agent_tokens(chat_id, "claude", {
                                "input_tokens":  usage.get("input_tokens", 0),
                                "output_tokens": usage.get("output_tokens", 0),
                                "cache_read":    usage.get("cache_read_input_tokens", 0),
                                "cache_create":  usage.get("cache_creation_input_tokens", 0),
                                "cost_usd":      cost,
                            })
                        except Exception:
                            pass
                break
    finally:
        completed.set()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _ai_proc_sem.release()
        _dec_active()
        if settings_file:
            Path(settings_file).unlink(missing_ok=True)

    if cancelled_flag.is_set():
        raise QueryCancelledError("query cancelled")

    rc = proc.returncode
    if rc == -9:
        raise TimeoutError(f"Claude force-killed (>{_cfg.HARD_TIMEOUT}s)")

    if rc != 0 and not result_text:
        _debug_log(f"[claude] abnormal exit rc={rc}\n" + "\n".join(stderr_buf[-10:]))
        if allow_retry and sid:
            _debug_log("[claude] clearing session and retrying")
            _clear_sid(ns, "claude")
            return _spawn_claude_proc(
                chat_id=chat_id, message=message, sid=None, cwd=cwd,
                cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
                on_tool_result=on_tool_result, allow_retry=False,
                images=images, session_namespace=session_namespace,
            )
        raise RuntimeError(
            f"Claude exited abnormally (rc={rc})\n"
            + ("\n".join(stderr_buf[-5:]) if stderr_buf else "no error output")
        )

    return result_text.strip()


def query_claude(chat_id: str, message: str, cwd: str,
                 cancel_ev: threading.Event = None,
                 on_tool=None, on_text=None, on_tool_result=None,
                 on_soft_timeout=None,
                 images: list = None) -> str:
    """Public interface: query Claude using chat_id as the session namespace."""
    sid = _load_sid(chat_id, "claude")
    return _spawn_claude_proc(
        chat_id=chat_id, message=message, sid=sid, cwd=cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=True,
        on_soft_timeout=on_soft_timeout, images=images,
        session_namespace=None,
    )


def _spawn_kimi_proc(
    chat_id: str,
    message: str,
    sid: str,
    cwd: str,
    cancel_ev: threading.Event = None,
    on_text=None,
    on_tool=None,
    on_tool_result=None,
    allow_retry: bool = False,
    on_soft_timeout=None,
    on_start=None,
    images: list = None,
    session_namespace: str = None,
) -> str:
    """Spawn a Kimi CLI subprocess and stream its output.

    Key differences from Claude:
    - Message delivery: stdin stream-json (role/content format)
    - Session resume: --session instead of --resume
    - Working directory: --work-dir flag (plus cwd= for safety)
    - Skip permissions: --yolo instead of --dangerously-skip-permissions
    - Output events: distinguished by role field; tool calls in tool_calls[] array
    - Tool name mapping: Shell→Bash, FetchURL→WebFetch, SearchWeb→WebSearch
    """
    ns = session_namespace if session_namespace is not None else chat_id

    args = [
        _cfg.KIMI_CMD, "--print", "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--verbose",
        "--work-dir", cwd,
    ]
    if sid:
        args += ["--session", sid]
    if _cfg.SKIP_PERMISSIONS:
        args += ["--yolo"]

    _debug_log(f"[kimi] starting cwd={cwd} sid={sid} skip_perm={_cfg.SKIP_PERMISSIONS} "
               f"images={len(images) if images else 0} ns={ns}")
    env = {**os.environ, "DBUS_SESSION_BUS_ADDRESS": "", "GCM_CREDENTIAL_STORAGE": "file"}

    _acquire_ai_sem(cancel_ev)
    try:
        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=cwd, env=env
        )
    except FileNotFoundError:
        _ai_proc_sem.release()
        raise RuntimeError(f"Kimi CLI not found: {_cfg.KIMI_CMD}")
    except Exception:
        _ai_proc_sem.release()
        raise

    if on_start:
        try:
            on_start()
        except Exception:
            pass

    try:
        if images:
            stdin_content = _build_kimi_stream_input(message, images)
        else:
            stdin_content = json.dumps({"role": "user", "content": message})
        proc.stdin.write(stdin_content + "\n")
        proc.stdin.close()
    except OSError as e:
        proc.kill()
        _ai_proc_sem.release()
        raise RuntimeError(f"Kimi stdin write failed: {e}")

    stderr_buf: list[str] = []

    def _drain():
        try:
            for line in proc.stderr:
                stripped = line.rstrip()
                if stripped and len(stderr_buf) < _MAX_STDERR_LINES:
                    stderr_buf.append(stripped)
        except Exception:
            pass

    threading.Thread(target=_drain, daemon=True).start()

    completed = threading.Event()
    cancelled_flag = threading.Event()
    soft_timeout_flag = threading.Event()

    def _watch():
        hard_deadline = time.time() + _cfg.HARD_TIMEOUT
        soft_deadline = time.time() + _cfg.RESPONSE_TIMEOUT
        soft_fired = False
        while time.time() < hard_deadline:
            if completed.is_set():
                return
            if cancel_ev and cancel_ev.is_set():
                _debug_log("[kimi] user cancelled")
                cancelled_flag.set()
                try:
                    proc.kill()
                except Exception:
                    pass
                return
            if not soft_fired and time.time() >= soft_deadline:
                soft_fired = True
                _debug_log(f"[kimi] soft timeout ({_cfg.RESPONSE_TIMEOUT}s), releasing lock but keeping process running")
                soft_timeout_flag.set()
                if on_soft_timeout:
                    try:
                        on_soft_timeout()
                    except Exception:
                        pass
            time.sleep(0.3)
        _debug_log(f"[kimi] hard timeout ({_cfg.HARD_TIMEOUT}s), force killing process")
        try:
            proc.kill()
        except Exception:
            pass

    threading.Thread(target=_watch, daemon=True).start()

    # Kimi tool name mapping to LarkHelm internal names (aligned with Claude)
    _KIMI_TOOL_MAP = {"Shell": "Bash", "FetchURL": "WebFetch", "SearchWeb": "WebSearch"}

    result_text, new_sid = "", None
    _tool_start_times: dict[str, float] = {}
    if on_text:
        on_text("", status="init")

    _inc_active()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Try to extract session_id from any event
            cand_sid = ev.get("session_id") or ev.get("session")
            if cand_sid and not new_sid:
                new_sid = cand_sid
                _save_sid(ns, new_sid, "kimi")

            role  = ev.get("role", "")
            etype = ev.get("type", "")

            if role == "assistant":
                content    = ev.get("content", "")
                tool_calls = ev.get("tool_calls") or []

                if content and isinstance(content, str):
                    result_text += content
                    if on_text:
                        on_text(result_text, status="typing")
                elif content and isinstance(content, list):
                    # kimi returns content as an array, e.g. [{"type":"think",...},{"type":"text","text":"..."}]
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    chunk = "".join(text_parts)
                    if chunk:
                        result_text += chunk
                        if on_text:
                            on_text(result_text, status="typing")

                for tc in tool_calls:
                    tc_id    = tc.get("id", "")
                    func     = tc.get("function", {})
                    kimi_name = func.get("name", "?")
                    name     = _KIMI_TOOL_MAP.get(kimi_name, kimi_name)
                    try:
                        inp = json.loads(func.get("arguments", "{}"))
                    except Exception:
                        inp = {}
                    _tool_start_times[tc_id] = time.monotonic()
                    if on_tool:
                        if name == "Bash" and "command" in inp:
                            summary = inp["command"][:200]
                        elif name in ("Read", "Write", "Edit") and "file_path" in inp:
                            summary = inp["file_path"]
                        elif name == "Glob" and "pattern" in inp:
                            summary = inp["pattern"]
                        elif name == "Grep" and "pattern" in inp:
                            summary = inp["pattern"]
                        elif name == "Agent" and "prompt" in inp:
                            summary = inp["prompt"][:100]
                        else:
                            summary = ", ".join(f"{k}={repr(v)[:40]}"
                                                for k, v in list(inp.items())[:2])
                        on_tool(name, summary, tool_id=tc_id)

            elif role == "tool":
                tc_id      = ev.get("tool_call_id", "")
                content    = ev.get("content", "")
                is_error   = bool(ev.get("is_error", False))
                content_str = str(content)
                start_t    = _tool_start_times.get(tc_id, time.monotonic())
                elapsed    = time.monotonic() - start_t
                truncated  = _truncate_tool_result(content_str, is_error)
                if on_tool_result:
                    on_tool_result(tc_id, truncated, is_error, elapsed)

            elif etype == "result" or role == "result":
                usage = ev.get("usage", {})
                cost  = ev.get("total_cost_usd", 0.0)
                if usage:
                    record_id = chat_id.split("__crew_")[0] if "__crew_" in chat_id else chat_id
                    try:
                        from larkhelm.token_stats import record_token_usage
                        record_token_usage(record_id, "kimi", {
                            "input_tokens":  usage.get("input_tokens",  usage.get("prompt_tokens", 0)),
                            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                            "cache_read":    0,
                            "cache_create":  0,
                            "cost_usd":      cost,
                        })
                    except Exception:
                        pass
                    if "__crew_" in chat_id:
                        try:
                            from larkhelm.token_stats import record_crew_agent_tokens
                            record_crew_agent_tokens(chat_id, "kimi", {
                                "input_tokens":  usage.get("input_tokens",  usage.get("prompt_tokens", 0)),
                                "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                                "cache_read":    0,
                                "cache_create":  0,
                                "cost_usd":      cost,
                            })
                        except Exception:
                            pass
                break
    finally:
        completed.set()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _ai_proc_sem.release()
        _dec_active()

    if cancelled_flag.is_set():
        raise QueryCancelledError("query cancelled")

    rc = proc.returncode
    if rc == -9:
        raise TimeoutError(f"Kimi force-killed (>{_cfg.HARD_TIMEOUT}s)")

    if rc != 0 and not result_text:
        _debug_log(f"[kimi] abnormal exit rc={rc}\n" + "\n".join(stderr_buf[-10:]))
        if allow_retry and sid:
            _debug_log("[kimi] clearing session and retrying")
            _clear_sid(ns, "kimi")
            return _spawn_kimi_proc(
                chat_id=chat_id, message=message, sid=None, cwd=cwd,
                cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
                on_tool_result=on_tool_result, allow_retry=False,
                session_namespace=session_namespace,
            )
        raise RuntimeError(
            f"Kimi exited abnormally (rc={rc})\n"
            + ("\n".join(stderr_buf[-5:]) if stderr_buf else "no error output")
        )

    return result_text.strip()


def _build_kimi_stream_input(text: str, image_paths: list[str]) -> str:
    """Build multimodal stdin input for Kimi --input-format stream-json."""
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


def query_kimi(chat_id: str, message: str, cwd: str,
               cancel_ev: threading.Event = None,
               on_tool=None, on_text=None, on_tool_result=None,
               on_soft_timeout=None,
               images: list = None,
               use_session: bool = True,
               record_under: str = None) -> str:
    """Public interface: query Kimi using chat_id as the session namespace.
    use_session=False disables session load/save (for crew agents).
    record_under: chat_id used for token usage recording (defaults to chat_id).
    """
    sid = _load_sid(chat_id, "kimi") if use_session else None
    return _spawn_kimi_proc(
        chat_id=chat_id, message=message, sid=sid, cwd=cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=True,
        on_soft_timeout=on_soft_timeout, images=images,
        session_namespace=None,
    )


def query_gemini(chat_id: str, message: str, cwd: str,
                 cancel_ev: threading.Event = None,
                 on_tool=None, on_text=None, on_tool_result=None,
                 on_soft_timeout=None,
                 use_session: bool = True,
                 record_under: str = None) -> str:
    """Query Gemini CLI (per-query process mode).
    use_session=False disables session load/save (for crew agents).
    record_under: chat_id used for token usage recording (defaults to chat_id).
    """
    sid = _load_sid(chat_id, "gemini") if use_session else None
    args = [_cfg.GEMINI_CMD, "-y", "--output-format", "stream-json", "-p", message]
    if sid:
        args += ["--resume", sid]

    env = {**os.environ, "DBUS_SESSION_BUS_ADDRESS": "", "GCM_CREDENTIAL_STORAGE": "file"}
    _debug_log(f"[gemini] starting query cwd={cwd} resume={sid} use_session={use_session}")

    _acquire_ai_sem(cancel_ev)
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=cwd, env=env
        )
    except FileNotFoundError:
        _ai_proc_sem.release()
        raise RuntimeError(f"Gemini CLI not found: {_cfg.GEMINI_CMD}")
    except Exception:
        _ai_proc_sem.release()
        raise

    stderr_buf: list[str] = []

    def _drain():
        try:
            for line in proc.stderr:
                stripped = line.rstrip()
                if stripped:
                    if len(stderr_buf) < _MAX_STDERR_LINES:
                        stderr_buf.append(stripped)
                    if "Keychain initialization encountered an error" in stripped:
                        _debug_log(f"[gemini] Stderr: {stripped}")
        except Exception:
            pass

    threading.Thread(target=_drain, daemon=True).start()

    completed = threading.Event()
    cancelled_flag = threading.Event()
    soft_timeout_flag = threading.Event()

    def _watch():
        hard_deadline = time.time() + _cfg.HARD_TIMEOUT
        soft_deadline = time.time() + _cfg.RESPONSE_TIMEOUT
        soft_fired = False
        while time.time() < hard_deadline:
            if completed.is_set():
                return
            if cancel_ev and cancel_ev.is_set():
                _debug_log("[gemini] user cancelled")
                cancelled_flag.set()
                try:
                    proc.kill()
                except Exception:
                    pass
                return
            if not soft_fired and time.time() >= soft_deadline:
                soft_fired = True
                _debug_log(f"[gemini] soft timeout ({_cfg.RESPONSE_TIMEOUT}s), releasing lock but keeping process running")
                soft_timeout_flag.set()
                if on_soft_timeout:
                    try:
                        on_soft_timeout()
                    except Exception:
                        pass
            time.sleep(0.3)
        _debug_log(f"[gemini] hard timeout ({_cfg.HARD_TIMEOUT}s), force killing process")
        try:
            proc.kill()
        except Exception:
            pass

    threading.Thread(target=_watch, daemon=True).start()

    result_text, new_sid = "", None
    _tool_start_times: dict[str, float] = {}
    if on_text:
        on_text("", status="init")

    _inc_active()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                if "Using FileKeychain fallback" not in line:
                    _debug_log(f"[gemini] non-JSON STDOUT: {line[:200]}")
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type", "")
            if etype in ("system", "init"):
                cand = ev.get("session_id")
                if cand:
                    new_sid = cand
                    if use_session:
                        _save_sid(chat_id, new_sid, "gemini")
            elif etype == "message" and ev.get("role") == "assistant":
                content = ev.get("content", "")
                if isinstance(content, list):
                    text_chunk = "".join(p.get("text", "") for p in content
                                         if isinstance(p, dict) and "text" in p)
                else:
                    text_chunk = str(content)
                if not ev.get("delta", False) and result_text and text_chunk.startswith(result_text):
                    # Non-delta full-text message that supersedes previous deltas: replace in full
                    result_text = text_chunk
                else:
                    result_text += text_chunk
                if on_text:
                    on_text(result_text, status="typing")
            elif etype == "tool_use":
                tid = ev.get("tool_id", "")
                name = ev.get("tool_name", "?")
                params = ev.get("parameters", {})
                _tool_start_times[tid] = time.monotonic()
                if on_tool:
                    summary = str(params)[:120]
                    on_tool(name, summary, tool_id=tid)
            elif etype == "tool_result":
                tid = ev.get("tool_id", "")
                res = str(ev.get("output", ""))
                is_err = ev.get("status") != "success"
                start_t = _tool_start_times.get(tid, time.monotonic())
                elapsed = time.monotonic() - start_t
                truncated = _truncate_tool_result(res, is_err)
                if on_tool_result:
                    on_tool_result(tid, truncated, is_err, elapsed)
            elif etype == "result":
                cand = ev.get("session_id")
                if cand:
                    new_sid = cand
                    if use_session:
                        _save_sid(chat_id, new_sid, "gemini")
                # Gemini token accounting (if usage field present)
                usage = ev.get("usage", {})
                if usage:
                    rec_id = record_under or chat_id
                    _gemini_cost = ev.get("total_cost_usd", 0.0)
                    try:
                        from larkhelm.token_stats import record_token_usage
                        record_token_usage(rec_id, "gemini", {
                            "input_tokens":  usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "cache_read":    0,
                            "cache_create":  0,
                            "cost_usd":      _gemini_cost,
                        })
                    except Exception:
                        pass
                    # Also track per crew-agent (for per-agent display in crew card)
                    if "__crew_" in chat_id:
                        try:
                            from larkhelm.token_stats import record_crew_agent_tokens
                            record_crew_agent_tokens(chat_id, "gemini", {
                                "input_tokens":  usage.get("input_tokens", 0),
                                "output_tokens": usage.get("output_tokens", 0),
                                "cache_read":    0,
                                "cache_create":  0,
                                "cost_usd":      _gemini_cost,
                            })
                        except Exception:
                            pass
                break
    finally:
        completed.set()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()  # always reap to prevent zombie; must not raise here
        _ai_proc_sem.release()
        _dec_active()

    if cancelled_flag.is_set():
        raise QueryCancelledError("query cancelled")
    rc = proc.returncode
    if rc == -9:
        if not soft_timeout_flag.is_set():
            raise TimeoutError(f"Gemini response timeout (>{_cfg.RESPONSE_TIMEOUT}s)")
    if not result_text and rc not in (0, None):
        raise RuntimeError(
            f"Gemini process exited abnormally rc={rc}\n"
            + ("\n".join(stderr_buf[-5:]) if stderr_buf else "no error output")
        )
    return result_text.strip()
