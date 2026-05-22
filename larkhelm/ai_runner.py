"""larkhelm · AI subprocess runner shim — delegates to runner_* modules.

This module exposes the historical ``_spawn_*_proc`` / ``query_*`` /
``MAX_AI_PROCS`` / ``_ai_proc_sem`` surface for back-compat. Every
function is now properly typed (S21+S26); the common spec-resolution
boilerplate in the ``query_*`` family is factored into
``_resolve_spec_or_default``.

The four ``_spawn_*_proc`` functions remain as thin per-runner shims
rather than a single ``**kwargs``-based dispatcher because each runner
class has a different positional/keyword parameter set; trying to
unify them through ``**kwargs`` would forfeit the very type checking
this commit is adding. The duplication is therefore intentional and
local — adding a new backend means adding one new wrapper here, not
modifying a fan-out dispatcher.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

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
from larkhelm.runner_types import (
    OnText, OnTool, OnToolResult, OnSoftTimeout, OnStart, CancelEvent,
)

import larkhelm.config as _cfg
from larkhelm.chat_state import _load_sid

if TYPE_CHECKING:
    # Imported only for type-checking to avoid a heavyweight import at
    # module load. BackendSpec is a frozen-ish dataclass with no
    # runtime side effects, so we could import it eagerly, but the
    # TYPE_CHECKING guard keeps the runtime import surface minimal
    # (mirrors what backend_cli.py does).
    from larkhelm.backend_registry import BackendSpec


# Back-compat shim. Older tests / external callers do
# ``from larkhelm.ai_runner import _ai_proc_sem`` or
# ``from larkhelm.ai_runner import MAX_AI_PROCS``. The straight ``from X import``
# pattern is exactly the P0 bug we're fixing (a stale binding survives
# ``_init_ai_sem`` rebuilding the live sem). To keep those imports working
# *and* always reflect the current sem / cap, expose them via a module-level
# ``__getattr__``. Internal larkhelm code MUST use ``get_ai_sem()`` /
# ``get_max_ai_procs()``; the shim is for tests that haven't been migrated
# and for any out-of-tree consumers.
def __getattr__(name: str):
    if name == "_ai_proc_sem":
        return get_ai_sem()
    if name == "MAX_AI_PROCS":
        return get_max_ai_procs()
    raise AttributeError(f"module 'larkhelm.ai_runner' has no attribute {name!r}")


# ─────────────────────────────────────────────────────────────────────
# Spawn wrappers — one per runner class.
# ─────────────────────────────────────────────────────────────────────
#
# These return ``str`` (the runner's final stdout-accumulated text).
# Keeping per-runner wrappers (rather than ``**kwargs`` dispatch) lets
# type checkers see each runner's actual signature and flag callers
# that pass the wrong subset of optional params.
# ─────────────────────────────────────────────────────────────────────

def _spawn_claude_proc(
    chat_id: str,
    message: str,
    sid: str | None,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_text: OnText | None = None,
    on_tool: OnTool | None = None,
    on_tool_result: OnToolResult | None = None,
    allow_retry: bool = False,
    on_soft_timeout: OnSoftTimeout | None = None,
    on_start: OnStart | None = None,
    images: list | None = None,
    session_namespace: str | None = None,
    command: str | None = None,
    model: str | None = None,
    extra_args: list[str] | None = None,
    session_key: str | None = None,
    system_prompt: str | None = None,
    suppress_token_recording: bool = False,
    usage_holder: dict | None = None,
) -> str:
    """Spawn Claude as a streamed subprocess and return its final text."""
    from larkhelm.runner_claude import ClaudeRunner
    runner = ClaudeRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
        model=model, extra_args=extra_args, session_key=session_key,
        system_prompt=system_prompt,
        suppress_token_recording=suppress_token_recording,
    )
    result = runner.run()
    if usage_holder is not None:
        usage_holder.update(runner._last_usage_seen or {})
    return result


def _spawn_kimi_proc(
    chat_id: str,
    message: str,
    sid: str | None,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_text: OnText | None = None,
    on_tool: OnTool | None = None,
    on_tool_result: OnToolResult | None = None,
    allow_retry: bool = False,
    on_soft_timeout: OnSoftTimeout | None = None,
    on_start: OnStart | None = None,
    images: list | None = None,
    session_namespace: str | None = None,
    command: str | None = None,
    model: str | None = None,
    extra_args: list[str] | None = None,
    session_key: str | None = None,
    suppress_token_recording: bool = False,
    usage_holder: dict | None = None,
) -> str:
    """Spawn Kimi as a streamed subprocess and return its final text."""
    from larkhelm.runner_kimi import KimiRunner
    runner = KimiRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
        model=model, extra_args=extra_args, session_key=session_key,
        suppress_token_recording=suppress_token_recording,
    )
    result = runner.run()
    if usage_holder is not None:
        usage_holder.update(runner._last_usage_seen or {})
    return result


def _spawn_deepseek_proc(
    chat_id: str,
    message: str,
    sid: str | None,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_text: OnText | None = None,
    on_tool: OnTool | None = None,
    on_tool_result: OnToolResult | None = None,
    allow_retry: bool = False,
    on_soft_timeout: OnSoftTimeout | None = None,
    on_start: OnStart | None = None,
    images: list | None = None,
    session_namespace: str | None = None,
    command: str | None = None,
    use_session: bool = True,
    record_under: str | None = None,
    model: str | None = None,
    extra_args: list[str] | None = None,
    session_key: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    suppress_token_recording: bool = False,
    usage_holder: dict | None = None,
) -> str:
    """Spawn a DeepSeek HTTP request via DeepSeekRunner.

    Despite the ``_proc`` suffix kept for naming symmetry with the subprocess
    runners, this is an HTTP call — there's no Popen.
    """
    runner = DeepSeekRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, allow_retry=allow_retry,
        on_soft_timeout=on_soft_timeout, on_start=on_start,
        images=images, session_namespace=session_namespace, command=command,
        use_session=use_session, record_under=record_under,
        model=model, extra_args=extra_args, session_key=session_key,
        api_key=api_key, base_url=base_url, system_prompt=system_prompt,
        suppress_token_recording=suppress_token_recording,
    )
    result = runner.run()
    if usage_holder is not None:
        usage_holder.update(runner._last_usage_seen or {})
    return result


def _spawn_gemini_proc(
    chat_id: str,
    message: str,
    sid: str | None,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_tool: OnTool | None = None,
    on_text: OnText | None = None,
    on_tool_result: OnToolResult | None = None,
    on_soft_timeout: OnSoftTimeout | None = None,
    use_session: bool = True,
    record_under: str | None = None,
    command: str | None = None,
    model: str | None = None,
    extra_args: list[str] | None = None,
    session_key: str | None = None,
    suppress_token_recording: bool = False,
    usage_holder: dict | None = None,
) -> str:
    """Spawn Gemini as a streamed subprocess and return its final text."""
    from larkhelm.runner_gemini import GeminiRunner
    runner = GeminiRunner(
        chat_id, message, sid, cwd,
        cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        use_session=use_session, record_under=record_under, command=command,
        model=model, extra_args=extra_args, session_key=session_key,
        suppress_token_recording=suppress_token_recording,
    )
    result = runner.run()
    if usage_holder is not None:
        usage_holder.update(runner._last_usage_seen or {})
    return result


# ─────────────────────────────────────────────────────────────────────
# Legacy fallback query_* path
# ─────────────────────────────────────────────────────────────────────
#
# These are the original direct-call entry points retained for callers
# that bypass the BackendRegistry (typically tests, crew fallback, and
# the legacy ``_do_query`` "last resort" branch). All four share a
# spec-resolution boilerplate that ``_resolve_spec_or_default`` factors
# out.
# ─────────────────────────────────────────────────────────────────────

def _resolve_spec_or_default(backend_id: str,
                             default_factory: Callable[[], "BackendSpec"]) -> "BackendSpec":
    """Return ``BACKEND_REGISTRY.get(backend_id)`` if present, else build a
    fresh default spec via ``default_factory``.

    Two paths can land here:

      * Registry hit — the normal case after bridge boot finishes config
        ingestion and registers every backend declared in
        ``config.json``.
      * Registry miss — usually a test that constructs an ``AiRunner``
        before ``BACKEND_REGISTRY`` is initialised, or a deployment where
        a backend was intentionally left out of config and the legacy
        path is the only way to invoke it.
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY
    spec = BACKEND_REGISTRY.get(backend_id)
    if spec is not None:
        return spec
    return default_factory()


def query_claude(
    chat_id: str,
    message: str,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_tool: OnTool | None = None,
    on_text: OnText | None = None,
    on_tool_result: OnToolResult | None = None,
    on_soft_timeout: OnSoftTimeout | None = None,
    images: list | None = None,
) -> str:
    """Legacy direct-invoke path for Claude (fallback when registry chain is empty)."""
    from larkhelm.backend_cli import run_claude
    from larkhelm.backend_registry import BackendSpec
    spec = _resolve_spec_or_default("claude", lambda: BackendSpec(
        id="claude", provider="claude_cli", display_name="Claude",
        role="orchestrator", tags=[], command=_cfg.CLAUDE_CMD,
    ))
    return run_claude(
        spec=spec, chat_id=chat_id, message=message, sid=_load_sid(chat_id, "claude"),
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        images=images, allow_retry=True,
    )


def query_kimi(
    chat_id: str,
    message: str,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_tool: OnTool | None = None,
    on_text: OnText | None = None,
    on_tool_result: OnToolResult | None = None,
    on_soft_timeout: OnSoftTimeout | None = None,
    images: list | None = None,
    use_session: bool = True,
    record_under: str | None = None,
) -> str:
    """Legacy direct-invoke path for Kimi (fallback when registry chain is empty)."""
    from larkhelm.backend_cli import run_kimi
    from larkhelm.backend_registry import BackendSpec
    spec = _resolve_spec_or_default("kimi", lambda: BackendSpec(
        id="kimi", provider="kimi_cli", display_name="Kimi",
        role="worker", tags=[], command=_cfg.KIMI_CMD,
    ))
    return run_kimi(
        spec=spec, chat_id=chat_id, message=message,
        sid=_load_sid(chat_id, "kimi") if use_session else None,
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        images=images, allow_retry=True,
    )


def query_deepseek(
    chat_id: str,
    message: str,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_tool: OnTool | None = None,
    on_text: OnText | None = None,
    on_tool_result: OnToolResult | None = None,
    on_soft_timeout: OnSoftTimeout | None = None,
    images: list | None = None,
    use_session: bool = True,
    record_under: str | None = None,
) -> str:
    """Legacy fallback path for /d / /deepseek when no BackendSpec is registered.

    The normal path goes through ``_run_backend_single`` (which uses the
    registry-resolved spec and ``backend_cli.run_deepseek``). This shim only
    fires when the registry chain is empty, mirroring ``query_kimi``.
    """
    from larkhelm.backend_cli import run_deepseek
    from larkhelm.backend_registry import BackendSpec
    spec = _resolve_spec_or_default("deepseek", lambda: BackendSpec(
        id="deepseek", provider="deepseek_api", display_name="DeepSeek",
        role="worker", tags=["cheap", "fast"],
        api_key=getattr(_cfg, "DEEPSEEK_API_KEY", "") or "",
        base_url=getattr(_cfg, "DEEPSEEK_BASE_URL", "") or "",
        model=getattr(_cfg, "DEEPSEEK_MODEL", "") or "",
    ))
    return run_deepseek(
        spec=spec, chat_id=chat_id, message=message,
        sid=_load_sid(chat_id, "deepseek") if use_session else None,
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        images=images, use_session=use_session, record_under=record_under,
    )


def query_gemini(
    chat_id: str,
    message: str,
    cwd: str,
    cancel_ev: CancelEvent | None = None,
    on_tool: OnTool | None = None,
    on_text: OnText | None = None,
    on_tool_result: OnToolResult | None = None,
    on_soft_timeout: OnSoftTimeout | None = None,
    use_session: bool = True,
    record_under: str | None = None,
) -> str:
    """Legacy direct-invoke path for Gemini (fallback when registry chain is empty)."""
    from larkhelm.backend_cli import run_gemini
    from larkhelm.backend_registry import BackendSpec
    spec = _resolve_spec_or_default("gemini", lambda: BackendSpec(
        id="gemini", provider="gemini_cli", display_name="Gemini",
        role="worker", tags=[], command=_cfg.GEMINI_CMD,
    ))
    return run_gemini(
        spec=spec, chat_id=chat_id, message=message,
        sid=_load_sid(chat_id, "gemini") if use_session else None,
        cwd=cwd, cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
        on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        use_session=use_session,
    )
