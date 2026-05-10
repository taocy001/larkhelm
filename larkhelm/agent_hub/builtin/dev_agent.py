"""larkhelm · agent_hub.builtin.dev_agent — thin wrapper over ``cmd_dev``."""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class DevAgent(AgentExecutor):
    agent_type = "dev"
    description = "完整软件开发流水线（PM→架构→实现→QA→Review），适合非平凡代码任务"
    required_capabilities = ("code", "tools")

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.crew._commands import cmd_dev

        start = time.monotonic()
        try:
            cmd_dev(ctx.chat_id, ctx.text, ctx.user_msg_id)
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
