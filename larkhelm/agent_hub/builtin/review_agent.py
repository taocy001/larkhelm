"""larkhelm · agent_hub.builtin.review_agent — code review via reviewer backend.

Routes through the "reviewer" task profile (reasoning=1.0, long_context=0.5,
require_tools=True) so a capable backend is selected, then calls _do_query.
Distinct from DevAgent's /dev pipeline: this is a quick single-pass review,
not a full PM→implement→QA cycle.
"""
from __future__ import annotations

import time
from typing import Optional

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


def _resolve_reviewer_backend_id(chat_id: str,
                                  force_backend_id: Optional[str] = None) -> Optional[str]:
    """Return a backend id suited for code review, or None to fall back."""
    try:
        from larkhelm.agent_hub.model_selector import resolve_backend_for_task
        from larkhelm.crew._backend_resolver import TASK_PROFILES
        profile = TASK_PROFILES.get("reviewer")
        if profile is None:
            return None
        spec = resolve_backend_for_task(chat_id, profile, force_backend_id=force_backend_id)
        if spec is None:
            return None
        spec_id = getattr(spec, "id", None)
        return str(spec_id) if spec_id else None
    except Exception as e:
        try:
            from larkhelm.log import _debug_log
            _debug_log(f"[ReviewAgent] reviewer backend resolve failed: {e}")
        except Exception:
            pass
        return None


class ReviewAgent(AgentExecutor):
    agent_type = "reviewer"
    description = "代码审查 / checklist review / diff 分析，不跑完整 /dev 流水线"
    required_capabilities = ("reasoning",)

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            model = _get_chat_model(ctx.chat_id)
            reviewer_id = _resolve_reviewer_backend_id(
                ctx.chat_id, force_backend_id=ctx.force_backend_id,
            )
            if reviewer_id:
                effective_force = reviewer_id
                _debug_log(
                    f"[ReviewAgent] reviewer routing chat={ctx.chat_id[:12]} "
                    f"→ backend={reviewer_id}"
                )
            else:
                effective_force = ctx.force_backend_id

            _do_query(
                chat_id=ctx.chat_id,
                message=ctx.text,
                model=model,
                user_msg_id=ctx.user_msg_id,
                parent_id=ctx.parent_id,
                force_backend_id=effective_force,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
