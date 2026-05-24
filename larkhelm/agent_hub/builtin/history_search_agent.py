"""larkhelm · agent_hub.builtin.history_search_agent — search conversation history.

Reuses the existing BM25 / hybrid memory retriever (memory_retriever.py) to
find relevant past conversation slices, then injects them into _do_query so
the AI can answer questions about previous interactions.

Useful for:
  - "我们上次讨论的 X 是什么？"
  - "帮我找找之前关于 Y 的对话"
  - "之前你给我写的那个脚本在哪里？"
"""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult

_MAX_SLICES = 5
_MAX_CONTEXT_CHARS = 4000


class HistorySearchAgent(AgentExecutor):
    agent_type = "history_search"
    description = "搜索过往对话历史：找之前讨论的内容、脚本、结论等，使用 BM25 检索"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            retrieved = _retrieve_history(ctx.chat_id, ctx.text)
            _debug_log(f"[HistorySearchAgent] retrieved {len(retrieved)} slices for query={ctx.text[:60]!r}")

            if retrieved:
                parts = [
                    "[历史对话检索结果]\n",
                    "以下是与当前问题相关的历史对话片段：\n",
                ]
                for i, (score, body) in enumerate(retrieved, 1):
                    parts.append(f"--- 片段 {i} (相关度 {score:.2f}) ---\n{body[:800]}\n")
                parts.append(f"\n---\n**用户问题：** {ctx.text}\n")
                parts.append("请根据以上历史片段回答用户问题，如不相关则根据自身知识回答。")
            else:
                parts = [
                    f"[历史对话检索：未找到与「{ctx.text[:60]}」相关的历史片段]\n\n",
                    "请直接根据自身知识回答以下问题：\n",
                    ctx.text,
                ]

            augmented = "\n".join(parts)
            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=augmented,
                model=model,
                user_msg_id=ctx.user_msg_id,
                parent_id=ctx.parent_id,
                force_backend_id=ctx.force_backend_id,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            _debug_log(f"[HistorySearchAgent] execute failed: {e}")
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )


def _retrieve_history(chat_id: str, query: str) -> list[tuple[float, str]]:
    """Return top-k (score, body) slices from memory retriever."""
    try:
        from larkhelm.memory_retriever import (
            KeywordRetriever, RetrievalRequest, MemorySlice, SliceKind, SliceLayer,
        )
        import larkhelm.config as _cfg
        from larkhelm.log import _read_log_md_files

        # Collect slices from the chat's memory .md files.
        slices: list[MemorySlice] = []
        try:
            from larkhelm.memory import (
                _session_memory_file, _global_memory_file, _project_memory_file,
            )
            import re
            for path_fn, layer in [
                (_session_memory_file, SliceLayer.session),
                (_project_memory_file, SliceLayer.project),
                (_global_memory_file, SliceLayer.global_),
            ]:
                try:
                    path = path_fn(chat_id)
                    if path and path.exists():
                        text = path.read_text(encoding="utf-8", errors="replace")
                        # Split by markdown H2/H3 headings as slice boundaries.
                        chunks = re.split(r"^#{1,3}\s+", text, flags=re.MULTILINE)
                        for chunk in chunks:
                            chunk = chunk.strip()
                            if len(chunk) > 30:
                                slices.append(MemorySlice(
                                    slice_id="", layer=layer,
                                    kind=SliceKind.summary, body=chunk[:2000],
                                    source_path=str(path),
                                ))
                except Exception:
                    pass
        except Exception:
            pass

        if not slices:
            return []

        retriever = KeywordRetriever(slices)
        req = RetrievalRequest(
            query=query, top_k=_MAX_SLICES, chat_id=chat_id,
        )
        scored = retriever.retrieve(req)
        return [(s.score, s.slice.body) for s in scored if s.score > 0.01]
    except Exception:
        return []
