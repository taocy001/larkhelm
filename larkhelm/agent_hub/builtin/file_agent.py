"""larkhelm · agent_hub.builtin.file_agent — file analysis via _do_query."""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class FileAgent(AgentExecutor):
    agent_type = "file"
    description = "文件内容分析：读取用户发送的文本/PDF/代码文件，注入 AI 上下文"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=ctx.text,
                model=model,
                user_msg_id=ctx.user_msg_id,
                files=ctx.files or [],
                parent_id=ctx.parent_id,
                force_backend_id=ctx.force_backend_id,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            _debug_log(f"[FileAgent] execute failed: {e}")
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
