"""larkhelm · agent_hub.builtin.search_agent — web search + AI synthesis.

Default backend: DuckDuckGo Instant Answer API (no API key required, stdlib only).
Optional: set ``search_api_provider = "brave"`` + ``search_api_key`` in config.json
for richer search results via the Brave Search API.

Flow:
  1. Call search API → collect top snippets / abstracts.
  2. Build a context block: [搜索结果] + original question.
  3. Pass to _do_query so the AI can synthesise a grounded answer.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult

_DDG_API = "https://api.duckduckgo.com/"
_BRAVE_API = "https://api.search.brave.com/res/v1/web/search"
_FETCH_TIMEOUT = 10
_MAX_SNIPPETS = 5
_MAX_SNIPPET_CHARS = 400


def _search_ddg(query: str) -> list[dict[str, str]]:
    """DuckDuckGo Instant Answer API — no API key, stdlib only."""
    params = urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
    })
    url = f"{_DDG_API}?{params}"
    results: list[dict[str, str]] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "larkhelm/1.0"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode())
    except Exception:
        return results

    # Instant answer / abstract
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", ""),
            "snippet": data["AbstractText"][:_MAX_SNIPPET_CHARS],
            "url": data.get("AbstractURL", ""),
        })

    # Related topics (max _MAX_SNIPPETS)
    for item in data.get("RelatedTopics", []):
        if len(results) >= _MAX_SNIPPETS:
            break
        if not isinstance(item, dict) or "Text" not in item:
            continue
        results.append({
            "title": "",
            "snippet": item["Text"][:_MAX_SNIPPET_CHARS],
            "url": item.get("FirstURL", ""),
        })

    return results


def _search_brave(query: str, api_key: str) -> list[dict[str, str]]:
    """Brave Search API — requires ``search_api_key`` in config."""
    params = urllib.parse.urlencode({"q": query, "count": _MAX_SNIPPETS, "text_decorations": "0"})
    url = f"{_BRAVE_API}?{params}"
    results: list[dict[str, str]] = []
    try:
        req = urllib.request.Request(
            url, headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return results

    for r in data.get("web", {}).get("results", []):
        if len(results) >= _MAX_SNIPPETS:
            break
        results.append({
            "title": r.get("title", ""),
            "snippet": (r.get("description") or r.get("extra_snippets", [""])[0])[:_MAX_SNIPPET_CHARS],
            "url": r.get("url", ""),
        })
    return results


def _do_search(query: str) -> list[dict[str, str]]:
    """Dispatch to configured provider, fall back to DDG on any error."""
    try:
        import larkhelm.config as _cfg
        provider = getattr(_cfg, "SEARCH_API_PROVIDER", "ddg").lower()
        api_key = getattr(_cfg, "SEARCH_API_KEY", "")
        if provider == "brave" and api_key:
            results = _search_brave(query, api_key)
            if results:
                return results
    except Exception:
        pass
    return _search_ddg(query)


def _build_search_context(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"[搜索结果：未找到关于「{query}」的相关信息]\n\n请根据已有知识回答。"
    lines = [f"[搜索结果：关于「{query}」]\n"]
    for i, r in enumerate(results, 1):
        title = f"**{r['title']}** — " if r.get("title") else ""
        url = f"\n  来源: {r['url']}" if r.get("url") else ""
        lines.append(f"{i}. {title}{r['snippet']}{url}\n")
    lines.append("\n请综合以上搜索结果回答用户问题。")
    return "\n".join(lines)


class SearchAgent(AgentExecutor):
    agent_type = "search"
    description = "联网搜索（DuckDuckGo / Brave）并将结果注入 AI 上下文，适合「最新 X 是什么」类问题"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            # Extract the core search query from user message.
            query = ctx.text.strip()
            _debug_log(f"[SearchAgent] searching query={query[:80]!r}")

            results = _do_search(query)
            _debug_log(f"[SearchAgent] got {len(results)} results")

            search_context = _build_search_context(query, results)
            augmented = f"{search_context}\n\n---\n**用户问题：** {ctx.text}"

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
            _debug_log(f"[SearchAgent] execute failed: {e}")
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
