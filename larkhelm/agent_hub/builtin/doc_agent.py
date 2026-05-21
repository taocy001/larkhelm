"""larkhelm · agent_hub.builtin.doc_agent — thin wrapper over ``_cmd_doc``.

After retiring the ``/doc`` user-facing slash command (方案B), the underlying
dispatcher moved from ``cmd_doc`` → ``doc_handlers``. DocAgent is the only
remaining caller of that natural-language entry point.
"""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class DocAgent(AgentExecutor):
    agent_type = "doc"
    description = "飞书文档/Wiki 读写：read/append/write/list 等子命令"
    required_capabilities = ("tools",)

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.doc_handlers import _cmd_doc

        start = time.monotonic()
        try:
            _cmd_doc(ctx.chat_id, ctx.text)
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
