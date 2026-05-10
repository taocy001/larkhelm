"""larkhelm · CLI backend wrappers (Claude / Gemini / Kimi / DeepSeek)

Wraps the internal ai_runner spawn functions, injecting spec.command
instead of reading from the global config. The public api_runner.query_*
functions become thin shims that delegate here.

Each ``run_*`` here also reports the call outcome to BackendRegistry so the
unified health-tick loop can short-circuit idle re-probes when a backend
is actively in use, and so transient/auth/quota failures observed from real
traffic flip ``healthy=False`` immediately (instead of waiting for the next
periodic probe).
"""
from __future__ import annotations

import threading

from larkhelm.backend_registry import BackendSpec, BACKEND_REGISTRY

# Semaphore-related helpers imported at module level.
# ai_runner.py does NOT import backend_cli at module level, so no circular import.
from larkhelm.ai_runner import (
    _spawn_claude_proc,
    _spawn_kimi_proc,
    _spawn_gemini_proc,
    _spawn_deepseek_proc,
    QueryCancelledError,
)


def _record_outcome(spec_id: str, exc: Exception | None) -> None:
    """Push call-outcome to BackendRegistry. Cancellation does NOT update health.

    Catches ``Exception`` (not ``BaseException``) so KeyboardInterrupt and
    SystemExit propagate without bookkeeping side-effects — those signal a
    process-level shutdown, not a backend fault.
    """
    try:
        if exc is None:
            BACKEND_REGISTRY.record_call_success(spec_id)
            return
        if isinstance(exc, QueryCancelledError):
            return  # user-initiated, not a backend fault
        # Pull thresholds from config so user can tune via config.json
        try:
            from larkhelm import config as _cfg
            window = float(getattr(_cfg, "BACKEND_TRANSIENT_WINDOW_SEC", 600.0))
            threshold = int(getattr(_cfg, "BACKEND_TRANSIENT_THRESHOLD", 3))
        except Exception:
            window, threshold = 600.0, 3
        BACKEND_REGISTRY.record_call_failure(
            spec_id, str(exc),
            transient_window_sec=window,
            transient_threshold=threshold,
        )
    except Exception:
        # Health tracking must NEVER mask the original error or break the call path.
        from larkhelm.log import safe_log
        safe_log(f"[BackendRegistry] _record_outcome failed for {spec_id}")


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
    system_prompt: str | None = None,
) -> str:
    try:
        out = _spawn_claude_proc(
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
            model=spec.model or None,
            extra_args=spec.extra_args or None,
            session_key=spec.id,
            system_prompt=system_prompt,
        )
    except Exception as e:
        _record_outcome(spec.id, e)
        raise
    _record_outcome(spec.id, None)
    return out


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
    try:
        out = _spawn_gemini_proc(
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
            model=spec.model or None,
            extra_args=spec.extra_args or None,
            session_key=spec.id,
        )
    except Exception as e:
        _record_outcome(spec.id, e)
        raise
    _record_outcome(spec.id, None)
    return out


def run_deepseek(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    sid: str,                     # accepted for parity; ignored (history file is canonical)
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
    use_session: bool = True,
    record_under: str = None,
    system_prompt: str | None = None,
) -> str:
    """Bridge BackendSpec → DeepSeekRunner. Despite the module name, this is
    HTTP, not CLI; placed here to mirror the run_claude/run_kimi/run_gemini
    contract and keep the dispatcher in ``_run_backend_single`` symmetric.
    """
    try:
        out = _spawn_deepseek_proc(
            chat_id=chat_id,
            message=message,
            sid=sid,
            cwd=cwd,
            cancel_ev=cancel_ev,
            on_text=on_text,
            on_tool=on_tool,
            on_tool_result=on_tool_result,
            on_soft_timeout=on_soft_timeout,
            on_start=on_start,
            images=images,
            session_namespace=session_namespace,
            use_session=use_session,
            record_under=record_under,
            model=spec.model or None,
            api_key=spec.api_key or None,
            base_url=spec.base_url or None,
            session_key=spec.id,
            system_prompt=system_prompt,
        )
    except Exception as e:
        _record_outcome(spec.id, e)
        raise
    _record_outcome(spec.id, None)
    return out


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
    try:
        out = _spawn_kimi_proc(
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
            model=spec.model or None,
            extra_args=spec.extra_args or None,
            session_key=spec.id,
        )
    except Exception as e:
        _record_outcome(spec.id, e)
        raise
    _record_outcome(spec.id, None)
    return out
