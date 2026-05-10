"""larkhelm · agent_hub.builtin.plan_agent — thin wrapper over ``cmd_plan``."""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class PlanAgent(AgentExecutor):
    agent_type = "plan"
    description = "多阶段开发计划：dev / review / fix / test 步骤化执行，含人工断点"
    required_capabilities = ("code", "tools", "reasoning")

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.cmd_plan import cmd_plan

        start = time.monotonic()
        try:
            cmd_plan(ctx.chat_id, ctx.text, ctx.user_msg_id)
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
