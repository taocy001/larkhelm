"""larkhelm · AI subprocess runner shim — delegates to runner_* modules."""
import threading

from larkhelm.runner_base import (
    QueryCancelledError,
    MAX_AI_PROCS,
    _ai_proc_sem,
    _active_proc_count,
    _MAX_STDERR_LINES,
    active_proc_count,
    _acquire_ai_sem,
    _inc_active,
    _dec_active,
    _truncate_tool_result,
)
from larkhelm.runner_claude import _build_stream_json_input
from larkhelm.runner_kimi import _build_kimi_stream_input

import larkhelm.config as _cfg
from larkhelm.chat_state import _load_sid


def _spawn_claude_proc(chat_id, message, sid, cwd, cancel_ev=None, on_text=None,
                       on_tool=None, on_tool_result=None, allow_retry=False,
                       on_soft_timeout=None, on_start=None, images=None,
                       session_namespace=None, command=None) -> str:
    from larkhelm.runner_claude import ClaudeRunner
    return ClaudeRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
    ).run()


def _spawn_kimi_proc(chat_id, message, sid, cwd, cancel_ev=None, on_text=None,
                     on_tool=None, on_tool_result=None, allow_retry=False,
                     on_soft_timeout=None, on_start=None, images=None,
                     session_namespace=None, command=None) -> str:
    from larkhelm.runner_kimi import KimiRunner
    return KimiRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
    ).run()


def _spawn_gemini_proc(chat_id, message, sid, cwd, cancel_ev=None, on_tool=None,
                       on_text=None, on_tool_result=None, on_soft_timeout=None,
                       use_session=True, record_under=None, command=None) -> str:
    from larkhelm.runner_gemini import GeminiRunner
    return GeminiRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        use_session=use_session, record_under=record_under, command=command,
    ).run()


def query_claude(chat_id, message, cwd, cancel_ev=None, on_tool=None, on_text=None,
                 on_tool_result=None, on_soft_timeout=None, images=None) -> str:
    from larkhelm.backend_cli import run_claude
    from larkhelm.backend_registry import BACKEND_REGISTRY, BackendSpec
    spec = BACKEND_REGISTRY.get("claude") or BackendSpec(
        id="claude", provider="claude_cli", display_name="Claude",
        role="orchestrator", tags=[], command=_cfg.CLAUDE_CMD,
    )
    return run_claude(
        spec=spec, chat_id=chat_id, message=message, sid=_load_sid(chat_id, "claude"),
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        images=images, allow_retry=True,
    )


def query_kimi(chat_id, message, cwd, cancel_ev=None, on_tool=None, on_text=None,
               on_tool_result=None, on_soft_timeout=None, images=None,
               use_session=True, record_under=None) -> str:
    from larkhelm.backend_cli import run_kimi
    from larkhelm.backend_registry import BACKEND_REGISTRY, BackendSpec
    spec = BACKEND_REGISTRY.get("kimi") or BackendSpec(
        id="kimi", provider="kimi_cli", display_name="Kimi",
        role="worker", tags=[], command=_cfg.KIMI_CMD,
    )
    return run_kimi(
        spec=spec, chat_id=chat_id, message=message,
        sid=_load_sid(chat_id, "kimi") if use_session else None,
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        images=images, allow_retry=True,
    )


def query_gemini(chat_id, message, cwd, cancel_ev=None, on_tool=None, on_text=None,
                 on_tool_result=None, on_soft_timeout=None,
                 use_session=True, record_under=None) -> str:
    from larkhelm.backend_cli import run_gemini
    from larkhelm.backend_registry import BACKEND_REGISTRY, BackendSpec
    spec = BACKEND_REGISTRY.get("gemini") or BackendSpec(
        id="gemini", provider="gemini_cli", display_name="Gemini",
        role="worker", tags=[], command=_cfg.GEMINI_CMD,
    )
    return run_gemini(
        spec=spec, chat_id=chat_id, message=message,
        sid=_load_sid(chat_id, "gemini") if use_session else None,
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        use_session=use_session,
    )
