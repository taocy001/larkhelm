"""larkhelm · CLI backend wrappers (Claude / Gemini / Kimi)

Wraps the internal ai_runner spawn functions, injecting spec.command
instead of reading from the global config. The public api_runner.query_*
functions become thin shims that delegate here.
"""
from __future__ import annotations

import threading

from larkhelm.backend_registry import BackendSpec

# Semaphore-related helpers imported at module level.
# ai_runner.py does NOT import backend_cli at module level, so no circular import.
from larkhelm.ai_runner import (
    _spawn_claude_proc,
    _spawn_kimi_proc,
    _spawn_gemini_proc,
)


def run_claude(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    sid: str,
    cwd: str,
    cancel_ev: threading.Event = None,
    on_text=None,
    on_tool=None,
    on_tool_result=None,
    on_soft_timeout=None,
    on_start=None,
    images: list = None,
    session_namespace: str = None,
    allow_retry: bool = False,
) -> str:
    return _spawn_claude_proc(
        chat_id=chat_id,
        message=message,
        sid=sid,
        cwd=cwd,
        cancel_ev=cancel_ev,
        on_text=on_text,
        on_tool=on_tool,
        on_tool_result=on_tool_result,
        allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout,
        on_start=on_start,
        images=images,
        session_namespace=session_namespace,
        command=spec.command or None,
    )


def run_gemini(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    sid: str,
    cwd: str,
    cancel_ev: threading.Event = None,
    on_text=None,
    on_tool=None,
    on_tool_result=None,
    on_soft_timeout=None,
    images: list = None,
    use_session: bool = True,
) -> str:
    return _spawn_gemini_proc(
        chat_id=chat_id,
        message=message,
        sid=sid,
        cwd=cwd,
        cancel_ev=cancel_ev,
        on_text=on_text,
        on_tool=on_tool,
        on_tool_result=on_tool_result,
        on_soft_timeout=on_soft_timeout,
        use_session=use_session,
        command=spec.command or None,
    )


def run_kimi(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    sid: str,
    cwd: str,
    cancel_ev: threading.Event = None,
    on_text=None,
    on_tool=None,
    on_tool_result=None,
    on_soft_timeout=None,
    on_start=None,
    images: list = None,
    session_namespace: str = None,
    allow_retry: bool = False,
) -> str:
    return _spawn_kimi_proc(
        chat_id=chat_id,
        message=message,
        sid=sid,
        cwd=cwd,
        cancel_ev=cancel_ev,
        on_text=on_text,
        on_tool=on_tool,
        on_tool_result=on_tool_result,
        allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout,
        on_start=on_start,
        images=images,
        session_namespace=session_namespace,
        command=spec.command or None,
    )
