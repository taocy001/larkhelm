"""larkhelm · agent_hub.builtin.chat_agent — thin wrapper over ``_do_query``.

P1 (design.md §1.2 / §5.2): when
``config.CHAT_AGENT_CHEAP_ROUTING_ENABLED`` is True, resolve a cheap
backend via :func:`resolve_backend_for_task` using
``TASK_PROFILES['chat']`` (which carries ``cost_ceiling=0.10``) and pass
its id as ``force_backend_id`` to ``_do_query``. If no healthy candidate
remains (all unhealthy / all filtered by cost_ceiling), fall back to the
user's preferred ``_get_chat_model`` and the original ``ctx.force_backend_id``.
"""
from __future__ import annotations

import time
from typing import Optional

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


def _resolve_cheap_backend_id(chat_id: str,
                              force_backend_id: Optional[str] = None) -> Optional[str]:
    """Return the cheap-routing backend id, or None to fall back.

    Separate from ``execute`` so it stays unit-testable without standing up
    the full ``_do_query`` chain. Never raises — any failure (registry not
    yet built, profile import failed, etc.) returns None and the caller
    keeps its original behaviour.
    """
    try:
        import larkhelm.config as _cfg
        if not bool(getattr(_cfg, "CHAT_AGENT_CHEAP_ROUTING_ENABLED", True)):
            return None
        from larkhelm.agent_hub.model_selector import resolve_backend_for_task
        from larkhelm.crew._backend_resolver import TASK_PROFILES
        profile = TASK_PROFILES.get("chat")
        if profile is None:
            return None
        spec = resolve_backend_for_task(
            chat_id, profile, force_backend_id=force_backend_id,
        )
        if spec is None:
            return None
        spec_id = getattr(spec, "id", None)
        return str(spec_id) if spec_id else None
    except Exception as e:
        try:
            from larkhelm.log import _debug_log
            _debug_log(
                f"[ChatAgent] cheap routing fell back to chat model "
                f"(reason={e})"
            )
        except Exception:
            pass
        return None


class ChatAgent(AgentExecutor):
    agent_type = "chat"
    description = "普通对话 / 简单问答 / 短代码片段，复用 _do_query 路径"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            model = _get_chat_model(ctx.chat_id)
            cheap_id = _resolve_cheap_backend_id(
                ctx.chat_id, force_backend_id=ctx.force_backend_id,
            )
            if cheap_id:
                effective_force = cheap_id
                _debug_log(
                    f"[ChatAgent] cheap routing chat={ctx.chat_id[:12]} "
                    f"→ backend={cheap_id}"
                )
            else:
                effective_force = ctx.force_backend_id
            _do_query(
                chat_id=ctx.chat_id,
                message=ctx.text,
                model=model,
                user_msg_id=ctx.user_msg_id,
                images=ctx.images,
                parent_id=ctx.parent_id,
                force_backend_id=effective_force,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
