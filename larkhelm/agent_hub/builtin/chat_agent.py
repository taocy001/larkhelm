"""larkhelm · agent_hub.builtin.chat_agent — thin wrapper over ``_do_query``."""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class ChatAgent(AgentExecutor):
    agent_type = "chat"
    description = "普通对话 / 简单问答 / 短代码片段，复用 _do_query 路径"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model

        start = time.monotonic()
        try:
            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=ctx.text,
                model=model,
                user_msg_id=ctx.user_msg_id,
                images=ctx.images,
                parent_id=ctx.parent_id,
                force_backend_id=ctx.force_backend_id,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
