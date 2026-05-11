"""larkhelm · AI subprocess runner shim — delegates to runner_* modules."""
import threading

from larkhelm.runner_base import (
    QueryCancelledError,
    _active_proc_count,
    _MAX_STDERR_LINES,
    active_proc_count,
    _acquire_ai_sem,
    _inc_active,
    _dec_active,
    _truncate_tool_result,
    get_ai_sem,
    get_max_ai_procs,
)
from larkhelm.runner_claude import _build_stream_json_input
from larkhelm.runner_kimi import _build_kimi_stream_input
from larkhelm.runner_deepseek import DeepSeekRunner

import larkhelm.config as _cfg
from larkhelm.chat_state import _load_sid


# Back-compat shim. Older tests / external callers do
# ``from larkhelm.ai_runner import _ai_proc_sem`` or
# ``from larkhelm.ai_runner import MAX_AI_PROCS``. The straight ``from X import``
# pattern is exactly the P0 bug we're fixing (a stale binding survives
# ``_init_ai_sem`` rebuilding the live sem). To keep those imports working
# *and* always reflect the current sem / cap, expose them via a module-level
# ``__getattr__``. Internal larkhelm code MUST use ``get_ai_sem()`` /
# ``get_max_ai_procs()``; the shim is for tests that haven't been migrated
# and for any out-of-tree consumers.
def __getattr__(name):
    if name == "_ai_proc_sem":
        return get_ai_sem()
    if name == "MAX_AI_PROCS":
        return get_max_ai_procs()
    raise AttributeError(f"module 'larkhelm.ai_runner' has no attribute {name!r}")


def _spawn_claude_proc(chat_id, message, sid, cwd, cancel_ev=None, on_text=None,
                       on_tool=None, on_tool_result=None, allow_retry=False,
                       on_soft_timeout=None, on_start=None, images=None,
                       session_namespace=None, command=None,
                       model=None, extra_args=None, session_key=None,
                       system_prompt=None) -> str:
    from larkhelm.runner_claude import ClaudeRunner
    return ClaudeRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
        model=model, extra_args=extra_args, session_key=session_key,
        system_prompt=system_prompt,
    ).run()


def _spawn_kimi_proc(chat_id, message, sid, cwd, cancel_ev=None, on_text=None,
                     on_tool=None, on_tool_result=None, allow_retry=False,
                     on_soft_timeout=None, on_start=None, images=None,
                     session_namespace=None, command=None,
                     model=None, extra_args=None, session_key=None) -> str:
    from larkhelm.runner_kimi import KimiRunner
    return KimiRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
        model=model, extra_args=extra_args, session_key=session_key,
    ).run()


def _spawn_deepseek_proc(chat_id, message, sid, cwd, cancel_ev=None, on_text=None,
                         on_tool=None, on_tool_result=None, allow_retry=False,
                         on_soft_timeout=None, on_start=None, images=None,
                         session_namespace=None, command=None,
                         use_session=True, record_under=None,
                         model=None, extra_args=None, session_key=None,
                         api_key=None, base_url=None,
                         system_prompt=None) -> str:
    """Spawn a DeepSeek HTTP request via DeepSeekRunner.

    Despite the ``_proc`` suffix kept for naming symmetry with the subprocess
    runners, this is an HTTP call — there's no Popen.
    """
    return DeepSeekRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
        use_session=use_session, record_under=record_under,
        model=model, extra_args=extra_args, session_key=session_key,
        api_key=api_key, base_url=base_url, system_prompt=system_prompt,
    ).run()


def _spawn_gemini_proc(chat_id, message, sid, cwd, cancel_ev=None, on_tool=None,
                       on_text=None, on_tool_result=None, on_soft_timeout=None,
                       use_session=True, record_under=None, command=None,
                       model=None, extra_args=None, session_key=None) -> str:
    from larkhelm.runner_gemini import GeminiRunner
    return GeminiRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        use_session=use_session, record_under=record_under, command=command,
        model=model, extra_args=extra_args, session_key=session_key,
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


def query_deepseek(chat_id, message, cwd, cancel_ev=None, on_tool=None, on_text=None,
                   on_tool_result=None, on_soft_timeout=None, images=None,
                   use_session=True, record_under=None) -> str:
    """Legacy fallback path for /d / /deepseek when no BackendSpec is registered.

    The normal path goes through ``_run_backend_single`` (which uses the
    registry-resolved spec and ``backend_cli.run_deepseek``). This shim only
    fires when the registry chain is empty, mirroring ``query_kimi``.
    """
    from larkhelm.backend_cli import run_deepseek
    from larkhelm.backend_registry import BACKEND_REGISTRY, BackendSpec
    spec = BACKEND_REGISTRY.get("deepseek") or BackendSpec(
        id="deepseek", provider="deepseek_api", display_name="DeepSeek",
        role="worker", tags=["cheap", "fast"],
        api_key=getattr(_cfg, "DEEPSEEK_API_KEY", "") or "",
        base_url=getattr(_cfg, "DEEPSEEK_BASE_URL", "") or "",
        model=getattr(_cfg, "DEEPSEEK_MODEL", "") or "",
    )
    return run_deepseek(
        spec=spec, chat_id=chat_id, message=message,
        sid=_load_sid(chat_id, "deepseek") if use_session else None,
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        images=images, use_session=use_session, record_under=record_under,
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
