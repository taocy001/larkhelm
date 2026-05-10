"""larkhelm · agent_hub.builtin.crew_agent — thin wrapper over ``cmd_crew``."""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class CrewAgent(AgentExecutor):
    agent_type = "crew"
    description = "多角色协作调研 / brainstorming / 长文档生成"
    required_capabilities = ("reasoning", "tools")

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.crew._commands import cmd_crew

        start = time.monotonic()
        try:
            cmd_crew(ctx.chat_id, ctx.text, ctx.user_msg_id)
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
