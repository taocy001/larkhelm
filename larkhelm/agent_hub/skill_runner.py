"""larkhelm · agent_hub · SkillRunner — generic executor for SkillDef instances.

Architecture
------------
``SkillExecutor`` is a concrete :class:`~larkhelm.agent_hub.agent_base.AgentExecutor`
that wraps a single :class:`~larkhelm.agent_hub.skill_types.SkillDef`.  The
:class:`~larkhelm.agent_hub.skill_registry.SkillRegistry` creates one
``SkillExecutor`` per skill and registers it in ``AGENT_REGISTRY`` so the
existing intent-dispatch pipeline routes to it transparently.

Context Injectors
-----------------
A **context injector** is a named Python callable registered via
:func:`register_injector`.  Skills reference injectors by name in their
``context_injectors`` list.  Before calling ``_do_query``, ``SkillExecutor``
runs each injector in order and concatenates the results into a context block
that is prepended to the AI call.

Built-in injectors
~~~~~~~~~~~~~~~~~~
``web_search``
    DuckDuckGo / Brave web search.  Extracted from the old ``SearchAgent``.

``shell_exec``
    Execute a shell command (with dangerous-command rejection).  Extracted from
    the old ``ShellAgent``.

``bm25_history``
    BM25 retrieval over the chat's memory files.  Extracted from the old
    ``HistorySearchAgent``.

Registering custom injectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    from larkhelm.agent_hub.skill_runner import register_injector

    def my_injector(text: str, ctx: AgentContext) -> str:
        data = fetch_something(text)
        return f"[MyData]\\n{data}\\n"

    register_injector("my_data", my_injector)

Then reference ``"my_data"`` in a SkillDef's ``context_injectors`` list.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult

# ── Context injector registry ─────────────────────────────────────────────

_INJECTORS: dict[str, Callable[[str, AgentContext], str]] = {}


def register_injector(name: str, fn: Callable[[str, AgentContext], str]) -> None:
    """Register a context injector callable under *name*."""
    _INJECTORS[name] = fn


def get_injector(name: str) -> Callable[[str, AgentContext], str] | None:
    return _INJECTORS.get(name)


# ── SkillExecutor ─────────────────────────────────────────────────────────


class SkillExecutor(AgentExecutor):
    """AgentExecutor wrapper for a SkillDef.

    Created and managed by :class:`~larkhelm.agent_hub.skill_registry.SkillRegistry`.
    Do not instantiate directly.
    """

    def __init__(self, skill: Any) -> None:           # skill: SkillDef (avoids circular)
        self._skill = skill
        self.agent_type: str = skill.id               # type: ignore[assignment]
        self.description: str = skill.description     # type: ignore[assignment]

    # AgentExecutor.can_handle defaults to exact agent_type match — correct.

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.log import _debug_log
        sk = self._skill
        start = time.monotonic()
        try:
            # 1. Apply strip pattern to clean trigger words from user text.
            text = _strip_trigger(ctx.text, sk.strip_trigger_pattern)

            # 2. Run context injectors in order; collect context blocks.
            context_blocks: list[str] = []
            for inj_name in sk.context_injectors:
                inj_fn = _INJECTORS.get(inj_name)
                if inj_fn is None:
                    _debug_log(f"[SkillRunner] unknown injector {inj_name!r} for skill {sk.id!r}")
                    continue
                try:
                    block = inj_fn(text, ctx)
                    if block:
                        context_blocks.append(block)
                except Exception as inj_exc:
                    _debug_log(f"[SkillRunner] injector {inj_name!r} failed: {inj_exc}")

            # 3. Build augmented message.
            parts: list[str] = []
            if sk.system_prompt:
                parts.append(sk.system_prompt)
            parts.extend(context_blocks)
            if ctx.text != text:
                parts.append(text)  # cleaned text (trigger words stripped)
            else:
                # No stripping happened: use the original text so the AI sees
                # the full request (context_blocks already carry the specifics).
                if not context_blocks:
                    parts.append(text)
                else:
                    parts.append(f"**用户请求：** {text}")

            augmented = "\n\n".join(p for p in parts if p)

            # 4. Resolve backend.
            backend_id = _resolve_skill_backend(ctx, sk.backend_profile)

            # 5. Call _do_query.
            from larkhelm.handlers._query import _do_query
            from larkhelm.chat_state import _get_chat_model
            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=augmented,
                model=model,
                user_msg_id=ctx.user_msg_id,
                parent_id=ctx.parent_id,
                force_backend_id=backend_id or ctx.force_backend_id,
                sender_open_id=ctx.extra.get("sender_open_id", ""),
            )
            return AgentResult(
                success=True,
                backend_id=backend_id or "",
                duration_sec=time.monotonic() - start,
            )
        except Exception as e:
            from larkhelm.log import _debug_log
            _debug_log(f"[SkillRunner] skill={sk.id!r} failed: {e}")
            return AgentResult(
                success=False,
                duration_sec=time.monotonic() - start,
                error=str(e),
            )


# ── Helper utilities ─────────────────────────────────────────────────────


def _strip_trigger(text: str, pattern: str) -> str:
    """Remove trigger words from text using *pattern* (regex or empty)."""
    if not pattern:
        return text
    try:
        import re
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip() or text
    except Exception:
        return text


def _resolve_skill_backend(ctx: AgentContext, profile: str) -> str | None:
    """Return a backend id suitable for *profile*, or None to fall back."""
    if ctx.force_backend_id:
        return ctx.force_backend_id
    if not profile or profile == "chat":
        # Try cheap routing the same way ChatAgent does.
        try:
            from larkhelm.agent_hub.builtin.chat_agent import _resolve_cheap_backend_id
            return _resolve_cheap_backend_id(ctx.chat_id)
        except Exception:
            return None
    try:
        from larkhelm.agent_hub.model_selector import resolve_backend_for_task
        from larkhelm.crew._backend_resolver import TASK_PROFILES
        tp = TASK_PROFILES.get(profile)
        if tp is None:
            return None
        spec = resolve_backend_for_task(ctx.chat_id, tp)
        if spec is None:
            return None
        return str(getattr(spec, "id", "") or "")
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[SkillRunner] backend resolve profile={profile!r}: {e}")
        return None


# ── Built-in context injectors ────────────────────────────────────────────
# These hold the implementation previously embedded in the Python agent classes.


def _injector_web_search(text: str, ctx: AgentContext) -> str:
    """DuckDuckGo / Brave web search → context block."""
    import json as _json
    import urllib.parse
    import urllib.request

    _DDG_API = "https://api.duckduckgo.com/"
    _BRAVE_API = "https://api.search.brave.com/res/v1/web/search"
    _TIMEOUT = 10
    _MAX_SNIPPETS = 5
    _SNIPPET_CHARS = 400

    def _ddg(q: str) -> list[dict]:
        params = urllib.parse.urlencode(
            {"q": q, "format": "json", "no_html": "1", "skip_disambig": "1"}
        )
        url = f"{_DDG_API}?{params}"
        out: list[dict] = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "larkhelm/1.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = _json.loads(resp.read().decode())
        except Exception:
            return out
        if data.get("AbstractText"):
            out.append({
                "title": data.get("Heading", ""),
                "snippet": data["AbstractText"][:_SNIPPET_CHARS],
                "url": data.get("AbstractURL", ""),
            })
        for item in data.get("RelatedTopics", []):
            if len(out) >= _MAX_SNIPPETS:
                break
            if isinstance(item, dict) and "Text" in item:
                out.append({
                    "title": "",
                    "snippet": item["Text"][:_SNIPPET_CHARS],
                    "url": item.get("FirstURL", ""),
                })
        return out

    def _brave(q: str, api_key: str) -> list[dict]:
        params = urllib.parse.urlencode(
            {"q": q, "count": _MAX_SNIPPETS, "text_decorations": "0"}
        )
        url = f"{_BRAVE_API}?{params}"
        out: list[dict] = []
        try:
            req = urllib.request.Request(
                url, headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                }
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = _json.loads(resp.read().decode())
        except Exception:
            return out
        for r in data.get("web", {}).get("results", []):
            if len(out) >= _MAX_SNIPPETS:
                break
            out.append({
                "title": r.get("title", ""),
                "snippet": (r.get("description") or r.get("extra_snippets", [""])[0])[:_SNIPPET_CHARS],
                "url": r.get("url", ""),
            })
        return out

    # Dispatch to configured provider.
    results: list[dict] = []
    try:
        import larkhelm.config as _cfg
        provider = getattr(_cfg, "SEARCH_API_PROVIDER", "ddg").lower()
        api_key = getattr(_cfg, "SEARCH_API_KEY", "")
        if provider == "brave" and api_key:
            results = _brave(text, api_key) or _ddg(text)
        else:
            results = _ddg(text)
    except Exception:
        results = _ddg(text)

    if not results:
        return f"[搜索结果：未找到关于「{text[:60]}」的相关信息]\n\n请根据已有知识回答。"
    lines = [f"[搜索结果：关于「{text[:60]}」]\n"]
    for i, r in enumerate(results, 1):
        title = f"**{r['title']}** — " if r.get("title") else ""
        url = f"\n  来源: {r['url']}" if r.get("url") else ""
        lines.append(f"{i}. {title}{r['snippet']}{url}\n")
    lines.append("\n请综合以上搜索结果回答用户问题。")
    return "\n".join(lines)


def _injector_shell_exec(text: str, ctx: AgentContext) -> str:
    """Execute a shell command and return stdout/stderr/rc as a context block."""
    import re
    import subprocess
    import shlex
    import os as _os

    _DANGEROUS_RE = re.compile(
        r"\b(rm\s+-[a-z]*r|mkfs|dd\s+if=|:(){ :|:&};:|chmod\s+777|chown|sudo|shutdown|reboot)\b",
        re.IGNORECASE,
    )
    _TIMEOUT = 30
    _MAX_OUT = 3000

    # Extract command from fenced code or raw text.
    fence = re.search(r"```(?:bash|sh|shell)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        cmd = fence.group(1).strip()
    elif re.match(r"^[\$>]?\s*([\w./~-])", text.strip()):
        cmd = re.sub(r"^[\$>\s]+", "", text.strip())
    else:
        cmd = text.strip()

    if _DANGEROUS_RE.search(cmd):
        try:
            from larkhelm.lark_client import send_card_reply
            send_card_reply(
                ctx.chat_id, ctx.user_msg_id,
                "⚠️ 命令被拒", f"检测到高危命令，已拒绝执行：\n```\n{cmd[:300]}\n```",
                color="red",
            )
        except Exception:
            pass
        return ""   # return empty → SkillExecutor will still call _do_query

    _SENSITIVE = ("SECRET", "TOKEN", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL")
    safe_env = {k: v for k, v in _os.environ.items()
                if not any(s in k.upper() for s in _SENSITIVE)}
    try:
        args = shlex.split(cmd)
        r = subprocess.run(
            args, shell=False, capture_output=True, text=True,
            timeout=_TIMEOUT, cwd=ctx.cwd, env=safe_env,
        )
        stdout, stderr, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", f"命令超时（>{_TIMEOUT}s）", -1
    except FileNotFoundError as e:
        stdout, stderr, rc = "", f"命令未找到: {e}", 127
    except Exception as e:
        stdout, stderr, rc = "", str(e), -1

    parts = [
        "以下是 shell 命令的执行结果，请根据用户原始问题给出解答。\n",
        f"**执行命令：** `{cmd}`\n**目录：** `{ctx.cwd}`\n**退出码：** `{rc}`",
    ]
    if stdout.strip():
        parts.append(f"\n**stdout：**\n```\n{stdout.strip()[:_MAX_OUT]}\n```")
    if stderr.strip():
        parts.append(f"\n**stderr：**\n```\n{stderr.strip()[:500]}\n```")
    if not stdout.strip() and not stderr.strip():
        parts.append("\n_（命令无输出）_")
    return "\n".join(parts)


def _injector_bm25_history(text: str, ctx: AgentContext) -> str:
    """BM25 memory retrieval → top-k slices as a context block."""
    _MAX_SLICES = 5

    try:
        from larkhelm.memory_retriever import (
            KeywordRetriever, RetrievalRequest, MemorySlice, SliceKind, SliceLayer,
        )
        from larkhelm.memory import (
            _session_memory_file, _global_memory_file, _project_memory_file,
        )
        import re

        slices: list[MemorySlice] = []
        for path_fn, layer in [
            (_session_memory_file, SliceLayer.session),
            (_project_memory_file, SliceLayer.project),
            (_global_memory_file,  SliceLayer.global_),
        ]:
            try:
                path = path_fn(ctx.chat_id)
                if path and path.exists():
                    txt = path.read_text(encoding="utf-8", errors="replace")
                    chunks = re.split(r"^#{1,3}\s+", txt, flags=re.MULTILINE)
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

        if not slices:
            return ""

        scored = KeywordRetriever(slices).retrieve(
            RetrievalRequest(query=text, top_k=_MAX_SLICES, chat_id=ctx.chat_id)
        )
        hits = [(s.score, s.slice.body) for s in scored if s.score > 0.01]
    except Exception:
        return ""

    if not hits:
        return f"[历史对话检索：未找到与「{text[:60]}」相关的历史片段]\n请直接根据自身知识回答。"
    parts = [
        "[历史对话检索结果]\n以下是与当前问题相关的历史对话片段：\n",
    ]
    for i, (score, body) in enumerate(hits, 1):
        parts.append(f"--- 片段 {i} (相关度 {score:.2f}) ---\n{body[:800]}\n")
    parts.append("请根据以上历史片段回答用户问题，如不相关则根据自身知识回答。")
    return "\n".join(parts)


# Register built-in injectors at module load.
register_injector("web_search",   _injector_web_search)
register_injector("shell_exec",   _injector_shell_exec)
register_injector("bm25_history", _injector_bm25_history)


__all__ = [
    "SkillExecutor",
    "register_injector",
    "get_injector",
]
