"""larkhelm · agent_hub.builtin.translate_agent — translation via cheap AI backend.

Routes through the "chat" task profile (cheap, fast) since translation is a
simple transformation task. Injects a brief instruction prefix so the AI
responds with only the translated text, not an explanation.
"""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult

# Instruction prefix injected before the user text.  Kept minimal so the AI
# doesn't pad its reply with commentary.
_TRANSLATE_PREFIX = (
    "请将以下内容翻译成目标语言（中文↔英文自动互译；如明确指定语言则翻译到该语言）。"
    "只输出翻译结果，不要解释。\n\n"
)


class TranslateAgent(AgentExecutor):
    agent_type = "translate"
    description = "中英互译 / 多语言翻译，走 cheap 后端，仅输出译文"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            # Reuse the same cheap-routing logic ChatAgent uses.
            try:
                from larkhelm.agent_hub.builtin.chat_agent import _resolve_cheap_backend_id
                cheap_id = _resolve_cheap_backend_id(
                    ctx.chat_id, force_backend_id=ctx.force_backend_id,
                )
            except Exception:
                cheap_id = None

            effective_force = cheap_id or ctx.force_backend_id
            if cheap_id:
                _debug_log(
                    f"[TranslateAgent] cheap routing chat={ctx.chat_id[:12]} "
                    f"→ backend={cheap_id}"
                )

            # Strip any "翻译" / "translate" trigger words that the L1 router
            # matched on, so the AI gets clean source text.
            import re as _re
            text = ctx.text
            text = _re.sub(
                r"^\s*(翻译[一下成到]?|帮我翻译|请翻译|translate[:\s]?)",
                "", text, flags=_re.IGNORECASE,
            ).strip()
            message = _TRANSLATE_PREFIX + (text or ctx.text)

            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=message,
                model=model,
                user_msg_id=ctx.user_msg_id,
                parent_id=ctx.parent_id,
                force_backend_id=effective_force,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            _debug_log(f"[TranslateAgent] execute failed: {e}")
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
