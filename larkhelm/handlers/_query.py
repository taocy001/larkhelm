"""
larkhelm · main query engine

Contains:
  - _extract_feishu_urls()   Extract Feishu document URLs from a message
  - _inject_doc_context()    Auto-inject Feishu document content into prompt
  - _do_query()              Execute an AI query with streaming card updates

The streaming-card state machine — render / push / tool tracking — lives
in ``_query_card_state.QueryCardState``. Splitting it out lets that
~280-line component get unit tested without spinning up the full chat
lock / backend resolution / Feishu API stack (P1 #8 / S2).
"""
import re
import threading
import time
import traceback
import uuid

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, log_entry
from larkhelm.concurrency import (
    _get_chat_lock, _get_cancel_event,
    _replace_cancel_event, _set_pending, _pop_pending, _update_pending_card_mid,
    _reset_cancel,
)
from larkhelm.lark_client import (
    send_card, send_card_reply, reply_card,
    _send_card_raw, _patch_card_raw, _reply_card_raw,
    _pin_task_card, react_to_message, delete_reaction,
    EMOJI_PROCESSING, EMOJI_DONE, EMOJI_ERROR,
    _index_reply,
)
from larkhelm.card_builder import _make_card, _split_md, _fmt_elapsed
from larkhelm.ai_runner import query_claude, query_gemini, query_kimi, query_deepseek, QueryCancelledError
from larkhelm.chat_state import _get_cwd, _load_sid, _get_turn_count, _increment_turn_count
from larkhelm.handlers._query_card_state import QueryCardState

# ── Card UX parameters ──────────────────────────────────────────────
# Constants live in _query_constants (dependency-free leaf module).
from larkhelm.handlers._query_constants import (
    CARD_PUSH_INTERVAL, CURSOR_INTERVAL,
    STALL_THRESHOLD, CURSOR_FRAMES, TOOL_HISTORY_CAP,
)


# ═══════════════════════════════════════════════════
#  Document URL auto-injection
# ═══════════════════════════════════════════════════

def _extract_feishu_urls(text: str) -> list:
    """Extract a list of Feishu document/Drive URLs from the given text."""
    return re.findall(r'https://[a-zA-Z0-9-]+\.feishu\.cn/[^\s\]>）]+', text)


def cleanup_temp_paths(paths: "list[str] | None") -> None:
    """Unlink temporary file paths produced during a query.

    Cleans paths under ``/tmp/`` (backward-compat with the inline temp-image
    cleanup) and paths under ``SESSION_DIR/{chat_id}/files/`` (file-download
    cache). Paths outside those two roots (e.g. ``SESSION_DIR/{chat_id}/imgs/``)
    are skipped safely so the image cache is never deleted between queries.
    """
    if not paths:
        return
    import os as _os
    # Resolve both roots so the prefix check is symlink-safe.
    # On macOS /tmp → /private/tmp; resolving the root means the comparison
    # works after realpath() expands the target path too.
    _tmp_root = _os.path.realpath("/tmp")
    try:
        session_dir_str = _os.path.realpath(str(_cfg.SESSION_DIR))
    except Exception:
        session_dir_str = ""
    for path in paths:
        try:
            if not path:
                continue
            p = _os.path.realpath(str(path))
            is_tmp = p.startswith(_tmp_root + "/") or p == _tmp_root
            is_session_files = bool(
                session_dir_str
                and p.startswith(session_dir_str)
                and "/files/" in p
            )
            if is_tmp or is_session_files:
                _os.unlink(p)
        except FileNotFoundError:
            pass
        except Exception as e:
            _debug_log(f"[Query] temp path cleanup failed: {e}")


def _format_age_hint(age_sec: int) -> str:
    """Render the parenthetical age hint for a cache-served doc payload.

    P5-OPT2: bucketed into 4 discrete bands so the injected user-message
    string stays byte-stable across small clock drifts. The Anthropic
    5-min ephemeral user-turn cache is keyed on exact byte content; the
    earlier per-minute rendering rotated the hint on every minute boundary
    and stopped repeated questions about the same doc from cache-hitting.

    Buckets (pinned by ``tests/test_context_cache.py``):

    - ``[0,   300)`` →  ``刚刚``           (< 5 min)
    - ``[300, 1800)`` → ``几分钟前``       (5..30 min)
    - ``[1800, 3600)`` → ``约半小时前``     (30..60 min)
    - ``>= 3600`` →    ``{age_sec // 3600} 小时前`` (floor hours)
    """
    if age_sec < 300:
        phrase = "刚刚"
    elif age_sec < 1800:
        phrase = "几分钟前"
    elif age_sec < 3600:
        phrase = "约半小时前"
    else:
        phrase = f"{age_sec // 3600} 小时前"
    return f"（缓存版本，{phrase}读取，如内容已变请提示刷新）"


def _inject_doc_context(text: str, chat_id: str, backend: str = "",
                        pending_doc_records: list[tuple[str, str]] | None = None,
                        ) -> str:
    """
    Detect Feishu document URLs in text, read their content, and prepend it to
    the prompt. At most DOC_INJECT_MAX_DOCS documents are injected, each capped
    at DOC_INJECT_MAX_CHARS characters. Failures are silently skipped so the
    normal conversation is unaffected.

    Successful reads are cached for ``DOC_INJECT_CACHE_TTL_SEC`` seconds per
    (chat_id, doc_token) — see ``_context_cache.cached_doc_read_with_meta``.
    On cache hit (P4 REQ-05) the injection text carries a parenthetical
    age hint so the model can warn the user the body may be stale.
    Permission / API errors propagate from the loader and are caught here
    (so the user still sees the "no permission" hint on each retry).

    Session-level dedup is keyed per (chat_id, ``backend``, doc_token) —
    sessions are per-(chat, backend), so a body injected into one backend's
    session must NOT suppress injection for another backend. ``backend``
    should be the routing target's backend id (same namespace as sid files).

    ``pending_doc_records`` (when given) collects ``(doc_token, content_hash)``
    tuples for every full body injected instead of recording them right away;
    the caller commits them via ``_context_cache.record_doc_injection`` only
    after the backend returns successfully, so a cancelled / failed query
    leaves no record. When ``None`` (direct callers / tests) no record is
    written — dedup then only ever downgrades based on previously committed
    records.
    """
    from larkhelm.lark_client import (
        FeishuDocClient, parse_doc_url,
        DocPermissionError, DocError,
    )
    urls = _extract_feishu_urls(text)
    if not urls:
        return text
    doc_client = FeishuDocClient()
    cache_enabled = bool(getattr(_cfg, "DOC_INJECT_CACHE_ENABLED", True))
    injections = []
    for url in urls:
        if len(injections) >= _cfg.DOC_INJECT_MAX_DOCS:
            break
        ref = parse_doc_url(url)
        if ref is None:
            continue
        try:
            age_hint = ""
            if cache_enabled:
                from larkhelm._context_cache import cached_doc_read_with_meta
                result_meta = cached_doc_read_with_meta(
                    chat_id, ref, _cfg.DOC_INJECT_MAX_CHARS,
                    loader=lambda r=ref: doc_client.read(
                        r, max_chars=_cfg.DOC_INJECT_MAX_CHARS
                    ),
                    backend=backend,
                )
                result = result_meta.payload
                if result_meta.from_cache and result_meta.age_sec is not None:
                    age_hint = _format_age_hint(result_meta.age_sec)
            else:
                result = doc_client.read(ref, max_chars=_cfg.DOC_INJECT_MAX_CHARS)
            label  = result.title or url
            header = f"[文档内容：《{label}》]"
            if age_hint:
                header = f"{header}\n{age_hint}"

            _injected_content = result.content

            # Session-level dedup: the same doc body was already injected
            # during the current (chat, backend) session (resumed CLI
            # sessions carry it from the first turn via --resume), so
            # re-injecting the full body is pure duplicate tokens. Downgrade
            # to a one-line marker. Records are committed by the caller after
            # backend success and cleared with the sid
            # (chat_state._clear_sid(chat_id, model)).
            # fail-open: any error here → full injection as before.
            _doc_token: str | None = None
            _content_hash = ""
            try:
                import hashlib
                from larkhelm._context_cache import doc_injection_seen
                _doc_token = str(getattr(ref, "token", "") or url)
                _content_hash = hashlib.sha256(
                    _injected_content.encode("utf-8", "replace")
                ).hexdigest()
                if doc_injection_seen(chat_id, backend, _doc_token, _content_hash):
                    injections.append(f"[文档《{label}》本会话已注入且未变更]")
                    try:
                        from larkhelm.metrics import inc_injection_gate as _inc_doc_ig3
                        _inc_doc_ig3("doc_inject", "skipped_session_dup")
                    except Exception:
                        pass
                    continue
            except Exception as _dd_err:
                _debug_log(f"[DoQuery] doc inject session dedup error (fail-open): {_dd_err}")
                _doc_token = None

            injections.append(
                f"{header}\n{_injected_content}\n[/文档内容]"
            )
            if _doc_token is not None and pending_doc_records is not None:
                pending_doc_records.append((_doc_token, _content_hash))
            try:
                from larkhelm.metrics import inc_injection_gate as _inc_doc_ig
                _inc_doc_ig("doc_inject", "injected")
                if len(_injected_content) > 10000:
                    _inc_doc_ig("doc_inject", "large_doc")
            except Exception:
                pass
        except DocPermissionError:
            injections.append(f"[文档 {url} 无读取权限，已跳过]")
        except DocError as _doc_e:
            _debug_log(f"[DoQuery] doc inject failed for {url}: {_doc_e}")
    if injections:
        return "\n\n".join(injections) + "\n\n---\n\n" + text
    return text


def _maybe_doc_usage_hint(text: str, chat_id: str, user_msg_id: str) -> bool:
    """Detect a bare unrecognised Feishu URL and reply with a usage card.

    Returns ``True`` (caller should short-circuit) only when ``text`` consists
    of *exactly one* Feishu URL that ``parse_doc_url`` cannot classify.
    All other shapes (no URL, multiple URLs, URL + extra text, URL parses
    successfully) return ``False`` so normal routing — including
    ``_inject_doc_context`` — continues unchanged.
    """
    urls = _extract_feishu_urls(text)
    if len(urls) != 1:
        return False
    # Allow only the bare-URL case to trigger the hint; if the user wrote
    # something around the URL we assume they meant to chat about it.
    if text.replace(urls[0], "").strip():
        return False
    from larkhelm.lark_client import parse_doc_url
    if parse_doc_url(urls[0]) is not None:
        return False
    from larkhelm.locale import _t
    from larkhelm.chat_state import _get_lang
    _lang = _get_lang(chat_id)
    body = _t(
        _lang,
        (
            "支持的飞书链接类型：\n"
            "- `docx` — 新版文档 `https://xxx.feishu.cn/docx/...`\n"
            "- `wiki` — Wiki 页面 `https://xxx.feishu.cn/wiki/...`\n"
            "- `sheets` — 电子表格 `https://xxx.feishu.cn/sheets/...`\n"
            "- `folder` / `drive` — 云盘文件夹 `https://xxx.feishu.cn/drive/folder/...`\n\n"
            "请检查链接类型是否正确，或在 URL 旁补一句你想做的事，"
            "机器人会按普通对话处理。"
        ),
        (
            "Supported Feishu URL types:\n"
            "- `docx` — Document `https://xxx.feishu.cn/docx/...`\n"
            "- `wiki` — Wiki page `https://xxx.feishu.cn/wiki/...`\n"
            "- `sheets` — Spreadsheet `https://xxx.feishu.cn/sheets/...`\n"
            "- `folder` / `drive` — Drive folder `https://xxx.feishu.cn/drive/folder/...`\n\n"
            "Please check the URL type, or add a sentence describing what you want to do "
            "and the bot will treat it as a normal message."
        ),
    )
    send_card_reply(
        chat_id, user_msg_id,
        _t(_lang, "⚠️ 飞书 URL 无法识别", "⚠️ Unrecognized Feishu URL"),
        body,
        color="orange",
    )
    return True


# ═══════════════════════════════════════════════════
#  AI query execution
# ═══════════════════════════════════════════════════

def _run_backend_single(spec, chat_id: str, message: str, cwd: str, cancel_ev,
                        on_text, on_tool, on_tool_result, on_soft_timeout,
                        images=None, extra_system: str = "",
                        recent_turns: str = "") -> str:
    """Run a single backend and return its output string.

    extra_system:
      - API backends: injected via the proper system channel (every call; stateless).
      - CLI backends: prepended as [System]...[User Query] ONLY when no existing session
        (sid=None). Resumed sessions already carry the context from the first turn, so
        re-injecting every message would multiply context size by N turns.

    recent_turns:
      Last ~6 turns serialized as plain text. Used to give CLI backends some
      conversational grounding on a fresh session (since after that they
      ``--resume`` and we never re-inject). **Dropped on API backends** because
      ``load_history()`` already passes the structured message list — including
      these same turns — and re-injecting them as a system blob is pure
      duplicate input tokens (~500 / call, ~50K / 100-turn session).
    """
    from larkhelm.backend_cli import run_claude, run_gemini, run_kimi, run_deepseek
    from larkhelm.backend_api import run_anthropic, run_google, run_openai_compat
    from larkhelm.api_session import load_history, save_history

    provider = spec.provider
    if provider == "anthropic_api":
        history = load_history(provider, chat_id)
        # NOTE: recent_turns intentionally omitted — history already carries it.
        api_extra = _maybe_strip_session_memory_for_api(
            extra_system, history, provider=provider, chat_id=chat_id)
        output, new_history = run_anthropic(spec, chat_id, message, history, cancel_ev, on_text,
                                            extra_system=api_extra)
        save_history(provider, chat_id, new_history)
    elif provider == "google_api":
        history = load_history(provider, chat_id)
        api_extra = _maybe_strip_session_memory_for_api(
            extra_system, history, provider=provider, chat_id=chat_id)
        output, new_history = run_google(spec, chat_id, message, history, cancel_ev, on_text,
                                         extra_system=api_extra)
        save_history(provider, chat_id, new_history)
    elif provider == "openai_compat_api":
        history = load_history(provider, chat_id)
        api_extra = _maybe_strip_session_memory_for_api(
            extra_system, history, provider=provider, chat_id=chat_id)
        output, new_history = run_openai_compat(spec, chat_id, message, history, cancel_ev, on_text,
                                                extra_system=api_extra)
        save_history(provider, chat_id, new_history)
    elif provider == "gemini_cli":
        sid = _load_sid(chat_id, spec.id)
        cli_recent = _maybe_drop_recent_turns_for_cli(
            chat_id, spec.id, sid, recent_turns, provider="gemini_cli",
        )
        cli_extra = "\n\n".join(filter(None, [extra_system, cli_recent]))
        cli_msg = f"[System]\n{cli_extra}\n\n[User Query]\n{message}" if (cli_extra and not sid) else message
        output = run_gemini(spec, chat_id, cli_msg, sid, cwd, cancel_ev,
                            on_text=on_text, on_tool=on_tool, on_tool_result=on_tool_result,
                            on_soft_timeout=on_soft_timeout)
    elif provider == "kimi_cli":
        sid = _load_sid(chat_id, spec.id)
        cli_recent = _maybe_drop_recent_turns_for_cli(
            chat_id, spec.id, sid, recent_turns, provider="kimi_cli",
        )
        cli_extra = "\n\n".join(filter(None, [extra_system, cli_recent]))
        cli_msg = f"[System]\n{cli_extra}\n\n[User Query]\n{message}" if (cli_extra and not sid) else message
        output = run_kimi(spec, chat_id, cli_msg, sid, cwd, cancel_ev,
                          on_text=on_text, on_tool=on_tool, on_tool_result=on_tool_result,
                          on_soft_timeout=on_soft_timeout, images=images)
    elif provider == "deepseek_api":
        # DeepSeek is HTTP + stateless. P1 REQ-04: when load_history()
        # returns a non-empty list, the runner is about to feed those
        # structured messages back to the backend, so re-injecting them
        # as recent_turns in the system channel is pure duplicate tokens.
        recent_for_ds = _maybe_drop_recent_turns_for_deepseek(
            chat_id, recent_turns, load_history_fn=load_history,
        )
        sys_combined = "\n\n".join(filter(None, [extra_system, recent_for_ds]))
        output = run_deepseek(spec, chat_id, message, sid=None, cwd=cwd,
                              cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
                              on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
                              system_prompt=sys_combined or None)
    else:  # claude_cli (default)
        sid = _load_sid(chat_id, spec.id)
        cli_recent = _maybe_drop_recent_turns_for_cli(
            chat_id, spec.id, sid, recent_turns, provider="claude_cli",
        )
        cli_extra = "\n\n".join(filter(None, [extra_system, cli_recent]))
        # CTX-C1: omit system_prompt on resumed sessions (sid != None) — the
        # session already contains the context from the first turn and Claude
        # CLI re-applies --system-prompt on every --resume call, causing
        # memory context to accumulate linearly (one extra copy per turn).
        # Gemini/Kimi already guard with `if (cli_extra and not sid)`.
        output = run_claude(spec, chat_id, message, sid, cwd, cancel_ev,
                            on_text=on_text, on_tool=on_tool, on_tool_result=on_tool_result,
                            on_soft_timeout=on_soft_timeout, images=images, allow_retry=True,
                            system_prompt=cli_extra if (cli_extra and not sid) else None)
    return output


def _maybe_strip_session_memory_for_api(
    extra_system: str, history: list, *, provider: str, chat_id: str,
) -> str:
    """Strip the ``[SESSION MEMORY]`` block from ``extra_system`` when the
    API session already carries structured history.

    The session-memory summary is distilled from the same recent chat-log
    turns that ``load_history()`` feeds back verbatim (up to 80K tokens —
    ``api_session._MAX_HISTORY_TOKENS``), so re-injecting the summary on
    every call is mostly duplicate input tokens AND rewrites the system
    block every ~10 turns, breaking the Anthropic prompt-cache prefix.
    Global / project layers stay — they are the stable prefix and have no
    verbatim counterpart in the history.

    A fresh API session (empty history) keeps the full injection: there
    the summary is the only continuity source (e.g. right after /reset,
    or for turns that happened on another backend).

    The boundary tags are owned by ``backend_api_streaming`` —
    ``_split_stable_volatile`` is the single parser so this gate and the
    layered cache_control split can never disagree on the block edges.
    fail-open: any error → return ``extra_system`` unchanged.

    Rollback switch: ``api_strip_session_memory_when_history=false``
    restores unconditional full injection (the summary is the only
    cross-backend continuity source — mixed-backend chats trade ~1K
    duplicate tokens/call for it).
    """
    if not extra_system:
        return extra_system
    if not bool(getattr(_cfg, "API_STRIP_SESSION_MEMORY_WHEN_HISTORY", True)):
        return extra_system
    try:
        from larkhelm.backend_api_streaming import _split_stable_volatile
        stable, volatile = _split_stable_volatile(extra_system)
    except Exception as e:
        _debug_log(f"[DoQuery] session memory split error (fail-open): {e}")
        return extra_system
    if not volatile:
        return extra_system
    if not history:
        try:
            from larkhelm.metrics import inc_injection_gate as _inc_sm
            _inc_sm("memory_session_api", "injected")
        except Exception:
            pass
        return extra_system
    try:
        from larkhelm.metrics import inc_injection_gate as _inc_sm2
        _inc_sm2("memory_session_api", "skipped_by_state")
    except Exception:
        pass
    _debug_log(
        f"[DoQuery] {provider} strip SESSION MEMORY chat={chat_id[:8]} "
        f"(history={len(history)} msgs)"
    )
    return stable


def _maybe_drop_recent_turns_for_cli(
    chat_id: str, spec_id: str, sid: str | None,
    recent_turns: str, *, provider: str,
) -> str:
    """REQ-04: return ``""`` when the CLI session is already resumable
    (``sid`` non-empty) AND ``CLI_SKIP_RECENT_TURNS_WHEN_SID`` is on, else
    return ``recent_turns`` unchanged.

    Resumed CLI sessions already carry the prior turns through
    ``--resume``/``--session``; re-injecting them in the system channel
    is pure duplicate input tokens (~500 tokens / call).
    """
    skip = bool(getattr(_cfg, "CLI_SKIP_RECENT_TURNS_WHEN_SID", True)) and bool(sid)
    if skip and recent_turns:
        _debug_log(
            f"[Cache] {provider} skip recent_turns chat={chat_id[:8]} sid present"
        )
        return ""
    return recent_turns


def _maybe_drop_recent_turns_for_deepseek(
    chat_id: str, recent_turns: str, *, load_history_fn,
) -> str:
    """REQ-04: drop ``recent_turns`` when ``load_history("deepseek_api", chat_id)``
    has any turns AND ``CLI_SKIP_RECENT_TURNS_WHEN_SID`` is on.

    The same flag governs both CLI sid and DeepSeek history because both
    signals encode the same fact ("the backend will see these turns
    structurally — no need to re-inject in the system channel").
    """
    if not bool(getattr(_cfg, "CLI_SKIP_RECENT_TURNS_WHEN_SID", True)):
        return recent_turns
    try:
        history = load_history_fn("deepseek_api", chat_id)
    except Exception as e:
        _debug_log(f"[Cache] deepseek_api load_history probe failed: {e}")
        return recent_turns
    if history:
        if recent_turns:
            _debug_log(
                f"[Cache] deepseek_api skip recent_turns chat={chat_id[:8]} "
                f"history={len(history)}"
            )
        return ""
    return recent_turns


def _do_query_with_delegation(
    chat_id: str,
    enriched_msg: str,
    orch_spec,
    worker_specs: dict,
    cwd: str,
    cancel_ev,
    on_text,
    on_tool,
    on_tool_result,
    on_soft_timeout,
    images=None,
    hop: int = 0,
    memory_ctx: str = "",
    recent_turns: str = "",
) -> str:
    """Execute query with delegation support (max 2 hops).

    Phase 1: Stream orchestrator response, buffer first 300 chars for DELEGATE detection.
    Phase 2: If DELEGATE found, run specialist; on_tool/on_tool_result show progress.
    Phase 3: Re-run orchestrator with specialist result for synthesis.
    Falls back to direct answer if specialist unavailable or delegation malformed.
    """
    from larkhelm.orchestration import build_orchestrator_system_prompt, _detect_delegation
    from larkhelm.backend_registry import BACKEND_REGISTRY

    # Build system prompt listing specialists, combined with memory context
    system_prompt = build_orchestrator_system_prompt(BACKEND_REGISTRY)
    orch_instructions = getattr(orch_spec, "instructions", "")
    combined_system = "\n\n".join(filter(None, [memory_ctx, system_prompt, orch_instructions]))

    if hop >= 2:
        # Retain orchestrator system prompt so the model knows its role and doesn't
        # re-emit DELEGATE: in the fallback path.
        return _run_backend_single(orch_spec, chat_id, enriched_msg, cwd, cancel_ev,
                                   on_text, on_tool, on_tool_result, on_soft_timeout, images,
                                   extra_system=combined_system,
                                   recent_turns=recent_turns)

    # Phase 1: run orchestrator, buffer ALL output until stream ends for DELEGATE detection.
    # We never flush to on_text during streaming — delegation can appear anywhere in the output.
    # Once complete, if DELEGATE found we clear the card; if not, we sync the full output.
    _buf_state = {"text": "", "delegation": None}

    def _buffered_on_text(text: str, status: str = "typing"):
        _buf_state["text"] = text
        # Optimisation: once we've seen END_DELEGATE we can stop checking
        if _buf_state["delegation"] is None and "END_DELEGATE" in text:
            result = _detect_delegation(text)
            if result:
                _buf_state["delegation"] = result

    orch_output = _run_backend_single(orch_spec, chat_id, enriched_msg, cwd, cancel_ev,
                                      _buffered_on_text, on_tool, on_tool_result, on_soft_timeout,
                                      images, extra_system=combined_system,
                                      recent_turns=recent_turns)

    # Final check on complete output
    delegation = _buf_state["delegation"] or _detect_delegation(orch_output)

    if not delegation:
        # Always sync final state; ensures card shows the complete output even
        # when flushed=True but the last incremental call predated stream end.
        on_text(orch_output)
        return orch_output

    # Phase 2: Specialist execution
    backend_id, sub_query = delegation
    specialist_spec = worker_specs.get(backend_id)

    if specialist_spec is None or not specialist_spec.healthy or not specialist_spec.enabled:
        _debug_log(f"[Delegation] specialist {backend_id} unavailable, falling back to direct orchestrator")
        on_text(f"> ⚠️ 专家 {backend_id} 不可用，正在直接回答...")
        # Pass fallback system prompt so orchestrator knows not to use DELEGATE format again
        from larkhelm.orchestration import _FALLBACK_SYSTEM
        fallback_system = "\n\n".join(filter(None, [memory_ctx, _FALLBACK_SYSTEM]))
        return _run_backend_single(orch_spec, chat_id, enriched_msg, cwd, cancel_ev,
                                   on_text, on_tool, on_tool_result, on_soft_timeout, images,
                                   extra_system=fallback_system,
                                   recent_turns=recent_turns)

    on_text(f"> 🔀 委托 {specialist_spec.display_name} 处理中...")
    on_tool(f"🔀 委托 {specialist_spec.display_name}", sub_query[:120], "delegation")

    _spec_start = time.monotonic()
    try:
        specialist_output = _run_backend_single(
            specialist_spec, chat_id, sub_query, cwd, cancel_ev,
            lambda t, **_: None,  # suppress specialist text from main card; accept status kwarg
            lambda *a, **_: None,
            lambda *a, **_: None,
            lambda: None,  # specialist must not trigger top-level soft-timeout / cancel_ev replacement
        )
    except Exception as _spec_err:
        _debug_log(f"[Delegation] specialist {backend_id} failed: {_spec_err}")
        specialist_output = f"[Specialist {backend_id} failed: {_spec_err}]"

    _spec_elapsed = time.monotonic() - _spec_start
    on_tool_result("delegation", specialist_output[:500], False, _spec_elapsed)

    # Return specialist output directly; no synthesis pass needed.
    # Orchestrator has full rolling history context and crafts self-contained sub_queries,
    # so the specialist's output is already the final answer.
    on_text(specialist_output)
    return specialist_output


def _post_query_memory_hook(chat_id: str, trace_id: str,
                            sender_open_id: str | None = None) -> None:
    """Run after a successful query: increment turn_count, dispatch to
    ``maybe_auto_update`` with the appropriate ``force`` flag for cold-start
    carry-over.

    Cold-start rule: when ``old_turn_count == 0`` (i.e. this is the first
    successful turn for this chat) the hook forces a summarization so the
    user feels memory continuity immediately rather than waiting for the
    regular 3-turn threshold. ``maybe_auto_update`` itself short-circuits
    when the chat has no readable history (``_read_logs_tail`` returns []),
    and ``_is_useful_summary`` rejects any thin output, so the force call is
    safe for brand-new chats too.

    ``sender_open_id`` is forwarded to ``maybe_auto_update`` so the global
    memory cascade writes to the TRIGGERING user's file (MEM-C1: ContextVar
    does not cross threads and the chat_state fallback is last-writer-wins
    in group chats).
    """
    try:
        from larkhelm.memory import maybe_auto_update
        old_count = _get_turn_count(chat_id)
        _increment_turn_count(chat_id)
        if old_count == 0:
            maybe_auto_update(chat_id, force=True,
                              sender_open_id=sender_open_id)
            return
        maybe_auto_update(chat_id, sender_open_id=sender_open_id)
    except Exception as _mc_err:
        _debug_log(f"[{trace_id}][DoQuery] post-query memory error: {_mc_err}")


def _apply_project_guide_gate(
    cwd: str, memory_ctx: str, is_cli_claude: bool = False
) -> "tuple[str, str]":
    """Apply the project-guide injection gate.

    Returns ``(updated_memory_ctx, metric_outcome)`` where outcome ∈
    {skipped_cli, injected, auto_discovered, not_found_auto, error, skipped}.
    Never raises.
    """
    if is_cli_claude:
        return memory_ctx, "skipped_cli"
    project_guide_path = _cfg.config.get("project_guide_path") or ""
    if project_guide_path:
        try:
            from pathlib import Path as _GPath
            _guide_path = _GPath(project_guide_path).expanduser()
            _guide_content = _guide_path.read_text(encoding="utf-8")
            if len(_guide_content) > 4000:
                _guide_content = _guide_content[:4000] + "…"
            memory_ctx = (
                f"[Project Guide]\n{_guide_content}\n[/Project Guide]\n\n"
                + memory_ctx
            )
            return memory_ctx, "injected"
        except Exception:
            return memory_ctx, "error"
    project_guide_auto_discover = bool(_cfg.config.get("project_guide_auto_discover"))
    if project_guide_auto_discover and cwd:
        from pathlib import Path as _GPath
        for _fname in ("CLAUDE.md", ".larkhelm_project.md"):
            _candidate = _GPath(cwd) / _fname
            try:
                if _candidate.exists():
                    _guide_content = _candidate.read_text(encoding="utf-8")
                    if len(_guide_content) > 4000:
                        _guide_content = _guide_content[:4000] + "…"
                    memory_ctx = (
                        f"[Project Guide]\n{_guide_content}\n[/Project Guide]\n\n"
                        + memory_ctx
                    )
                    return memory_ctx, "auto_discovered"
            except Exception:
                pass
        return memory_ctx, "not_found_auto"
    return memory_ctx, "skipped"


def _do_query(chat_id: str, message: str, model: str, user_msg_id: str = None,
              images: list = None, files: list = None,
              parent_id: str | None = None,
              force_backend_id: str | None = None,
              trace_id: str | None = None,
              sender_open_id: str | None = None,
              queue_card_mid: str | None = None):
    # Use caller-supplied trace_id (_message.py generates one BEFORE
    # logging the user entry so user/assistant pair share an id —
    # /stats duration pairing depends on it). Fall back to a fresh uuid
    # when called from a context that doesn't propagate trace_id.
    if not trace_id:
        trace_id = uuid.uuid4().hex[:12]

    chat_lock = _get_chat_lock(chat_id)
    cancel_ev = _get_cancel_event(chat_id)

    if not chat_lock.acquire(blocking=False):
        existing_mid = _set_pending(chat_id, message, model, user_msg_id)
        preview = message[:80].replace("\n", " ")
        queue_card = _make_card(
            "⏳ 排队中",
            f"将在当前任务完成后自动执行：\n\n> {preview}",
            color="orange",
            buttons=[("❌ 取消排队", f"cancel_queue:{chat_id}")]
        )
        # Pending-rollback pattern: if card emission fails, the user has
        # no visible queue indicator and no cancel button → roll back so
        # the next message can re-queue.
        try:
            if existing_mid:
                _patch_card_raw(existing_mid, queue_card)
            else:
                if user_msg_id:
                    mid = _reply_card_raw(user_msg_id, queue_card, in_thread=False)
                else:
                    mid = _send_card_raw(chat_id, queue_card)
                _update_pending_card_mid(chat_id, mid)
        except Exception as _card_err:
            _debug_log(
                f"[{trace_id}][DoQuery] queue card emission failed, rolling back "
                f"pending state: {_card_err}"
            )
            _pop_pending(chat_id)
        return

    _eyes_reaction_id: list[str | None] = [None]
    if user_msg_id:
        _eyes_reaction_id[0] = react_to_message(user_msg_id, EMOJI_PROCESSING)

    lock_released = False   # Set to True when the soft-timeout releases the lock early, preventing double-release in finally

    # ``start`` is captured BEFORE the outer try so the finally block's
    # ``time.time() - start`` is never UnboundLocal. Previously this line
    # lived inside the try (`_get_cwd` could raise), so an exception there
    # would crash the finally's elapsed-time math even though
    # ``record_query_start`` (which IS try/except-wrapped) had already run.
    start = time.time()

    # P1-3: bump the diagnostic active counter so /metrics surfaces it.
    try:
        from larkhelm.handlers._query_card_state import record_query_start
        record_query_start()
    except Exception:
        pass

    try:
        cwd = _get_cwd(chat_id)

        # Resolve backend early (pure registry lookup, no network) so the initial card
        # shows the correct model name instead of the legacy default_model value.
        m_name = {"claude": "Claude", "gemini": "Gemini", "kimi": "Kimi", "deepseek": "DeepSeek"}.get(model, model.capitalize())
        _early_spec = None  # pre-initialise; used by the recent_turns skip gate below
        try:
            from larkhelm.router import resolve_backend as _early_resolve
            from larkhelm.backend_registry import BACKEND_REGISTRY as _early_reg
            _early_has_docs = bool(_extract_feishu_urls(message))
            _early_spec = _early_resolve(chat_id, message, bool(images), _early_has_docs, force_backend_id)
            m_name = _early_spec.display_name
        except Exception:
            pass  # fall back to legacy m_name; full routing happens below

        init_card = _make_card(f"⏳ {m_name} 连接中",
                               f"> 正在启动...\n\n目录: `{cwd}`", color="grey",
                               buttons=[("🛑 取消", f"cancel:{chat_id}")])
        # When dispatched from the pending queue, reuse the existing
        # ``⏳ 排队中 / ❌ 取消排队`` card so the user sees one continuous
        # status thread (transition queued → running in place) rather than
        # an orphan "queued" card lingering next to a fresh progress card.
        # Falls back to send / reply if the patch fails (e.g. card mid
        # invalidated by Feishu retention sweep).
        mid: str | None = None
        if queue_card_mid:
            if _patch_card_raw(queue_card_mid, init_card):
                mid = queue_card_mid
        if not mid:
            if user_msg_id:
                mid = _reply_card_raw(user_msg_id, init_card, in_thread=False)
            else:
                mid = _send_card_raw(chat_id, init_card)
        if mid:
            _pin_task_card(chat_id, mid)

        # ── Card render / push / tool state machine ─────────────────────
        # All scalar + tool state plus the render and push logic lives in
        # QueryCardState (extracted per S2). The closures below are thin
        # adapters that bind the state's callbacks to this query's
        # ``log_entry`` invocation and supply the ``mid`` / cancel / stop
        # gates that the state class deliberately doesn't own.
        card_state = QueryCardState(chat_id=chat_id, model_name=m_name, start_time=start)

        def _push_if_needed(force: bool = False, include_cancel: bool = True):
            if cancel_ev.is_set() or _stop_hb.is_set():
                return
            rendered = card_state.render_body()
            need_push, combined = card_state.should_push(rendered, force=force)
            if need_push:
                btns = [("🛑 取消", f"cancel:{chat_id}")] if include_cancel else None
                card_json = _make_card(rendered.title, rendered.response_md, color="grey",
                                       tools_md=rendered.tools_md, tools_expanded=True,
                                       buttons=btns)
                with card_state.card_patch_lock:
                    # Re-check inside the lock: if the main thread already set _stop_hb
                    # and drained this lock, don't overwrite the final card.
                    if cancel_ev.is_set() or _stop_hb.is_set():
                        return
                    _patch_card_raw(mid, card_json)
                card_state.mark_pushed(combined)

        # ── Callbacks: thin shims that add per-query logging on top of
        #    the state's pure callbacks. ``on_tool`` is the only one that
        #    actually needs a wrapper (for log_entry); the others bind
        #    directly to ``card_state.on_*``.
        def on_tool(name: str, desc: str, tool_id: str = ""):
            card_state.on_tool(name, desc, tool_id)
            log_entry(chat_id, "tool", f"{name}: {desc}", model=model, trace_id=trace_id)

        on_tool_result = card_state.on_tool_result
        on_text = card_state.on_text

        # ── Soft-timeout callback ────────────────────────────────
        def _on_soft_timeout():
            nonlocal lock_released
            if lock_released:   # already fired once (e.g. during delegation phase 1 + phase 2)
                return
            elapsed_now = _fmt_elapsed(time.time() - start)
            _debug_log(f"[{trace_id}][DoQuery] soft timeout ({elapsed_now}), lock released, continuing in background")
            card_state.set_in_background(True)
            # Heartbeat keeps running in background (shows "后台·" prefix, hides cancel button).
            # It is stopped by the success/error finally block when the AI finishes.
            try:
                chat_lock.release()
            except RuntimeError:
                pass
            lock_released = True
            _replace_cancel_event(chat_id)
            pending = _pop_pending(chat_id)
            if pending:
                p_msg, p_model, p_user_msg_id, p_queue_mid = (
                    pending[0], pending[1], pending[2],
                    pending[3] if len(pending) >= 4 else None,
                )
                _debug_log(f"[Queue/SoftTimeout] processing queued message: {p_msg[:60]}")
                try:
                    threading.Thread(
                        target=_do_query,
                        args=(chat_id, p_msg, p_model, p_user_msg_id),
                        kwargs={"queue_card_mid": p_queue_mid},
                        daemon=True, name=f"query-{chat_id[:8]}",
                    ).start()
                except Exception as _te:
                    # LOGIC-C5: thread creation failed (e.g. OOM); re-push so the
                    # message isn't silently lost. Next incoming message will retry.
                    _debug_log(f"[Queue/SoftTimeout] thread start failed, re-queueing: {_te}")
                    _set_pending(chat_id, p_msg, p_model, p_user_msg_id)

        # ── Heartbeat thread ─────────────────────────────────────
        _stop_hb = threading.Event()

        def _heartbeat_loop():
            while not _stop_hb.is_set():
                try:
                    card_state.tick_cursor()
                    now = time.monotonic()
                    # Snapshot the three render-affecting flags atomically
                    # in ONE lock acquisition so the heartbeat decision can't
                    # observe a (in_bg=False, dirty=True) transient that
                    # occurs when ``_on_soft_timeout`` fires between two
                    # separate lock reads (see review of d4fbc7a, fix #1).
                    last_hb, in_bg, dirty_now = card_state.get_heartbeat_snapshot()
                    # After soft timeout the task runs in background; cancel button
                    # is no longer wired to the new cancel event, so hide it.
                    show_cancel = not in_bg
                    if now - last_hb >= CARD_PUSH_INTERVAL:
                        _push_if_needed(force=True, include_cancel=show_cancel)
                        card_state.update_heartbeat()
                    elif dirty_now:
                        _push_if_needed(force=False, include_cancel=show_cancel)
                except Exception as e:
                    _debug_log(f"[Heartbeat] exception: {e}")
                _stop_hb.wait(timeout=CURSOR_INTERVAL)

        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True,
                                     name=f"hb-{chat_id[:8]}")
        hb_thread.start()

        try:
            # ── Parent message injection (network call — here not in event thread) ──
            if parent_id:
                try:
                    # P2b: when the target is an API backend (which already carries
                    # conversation history structurally via load_history), the parent
                    # turn is already present in the structured history — re-injecting
                    # it as a text prefix is ~100–500 duplicate input tokens.
                    # fail-open: any exception in the gate → inject anyway.
                    _skip_parent = False
                    try:
                        _api_providers = {"anthropic_api", "google_api", "openai_compat_api"}
                        if (
                            bool(_cfg.config.get("parent_inject_skip_when_api_history", False))
                            and _early_spec is not None
                            and getattr(_early_spec, "provider", "") in _api_providers
                        ):
                            # Only skip when the API session already has history;
                            # on the very first turn history is empty and the
                            # parent message would not be present elsewhere.
                            from larkhelm.api_session import has_history as _hh
                            if _hh(_early_spec.provider, chat_id):
                                _skip_parent = True
                                from larkhelm.metrics import inc_injection_gate as _inc_pig
                                _inc_pig("parent_msg", "skipped_api")
                                _debug_log(
                                    f"[{trace_id}][DoQuery] parent inject skipped "
                                    f"(API backend with history: {_early_spec.provider})"
                                )
                    except Exception as _gate_err:
                        _debug_log(
                            f"[{trace_id}][DoQuery] parent inject gate error "
                            f"(fail-open): {_gate_err}"
                        )

                    if not _skip_parent:
                        from larkhelm.lark_client import _fetch_parent_message_text
                        parent_text = _fetch_parent_message_text(parent_id)
                        if parent_text:
                            try:
                                from larkhelm.metrics import inc_injection_gate as _inc_pig2
                                _inc_pig2("parent_msg", "injected")
                            except Exception:
                                pass
                            message = (
                                f"[用户回复了以下消息]\n\n{parent_text}\n\n---\n\n{message}"
                            )
                            _debug_log(f"[{trace_id}][DoQuery] injected parent context ({len(parent_text)} chars)")
                except Exception as _pe:
                    _debug_log(f"[{trace_id}][DoQuery] parent fetch error: {_pe}")

            # ── File content injection ───────────────────────────────────────
            # Prepend extracted file blocks before doc injection so file content
            # is in scope when Feishu doc URLs are resolved.
            if files:
                try:
                    from larkhelm.file_handler import files_to_prompt_fragment
                    _file_prefix = files_to_prompt_fragment(files)
                    if _file_prefix:
                        message = _file_prefix + message
                except Exception as _fe:
                    _debug_log(f"[{trace_id}][DoQuery] file inject error: {_fe}")

            # ── Doc injection + memory context ───────────────────────────────
            # Doc injection runs here (background thread, not SDK event callback) to avoid
            # blocking the event dispatch loop. Applied to the original message so memory
            # content cannot trigger redundant Feishu API reads.
            has_doc_urls = bool(_extract_feishu_urls(message))
            # Pending session-dedup records for full doc bodies injected this
            # query; committed only after the backend returns successfully
            # (under the backend that actually served the query) so /cancel,
            # backend failure or failover never leaves a record for a session
            # that never saw the body.
            _doc_pending: list[tuple[str, str]] = []
            if _cfg.DOC_AUTO_INJECT:
                try:
                    message = _inject_doc_context(
                        message, chat_id,
                        backend=(_early_spec.id if _early_spec is not None else model),
                        pending_doc_records=_doc_pending,
                    )
                except Exception as _doc_err:
                    _debug_log(f"[{trace_id}][DoQuery] doc inject error: {_doc_err}")

            # Memory is passed as extra_system (proper system channel) rather than prepended
            # to the user message, so the model receives clean user turn content.
            memory_ctx = ""
            recent_turns_list: list[str] = []
            try:
                from larkhelm.log import _get_recent_turns
                # P1 REQ-04 fast-path: skip the 100 KB tail-read entirely when the
                # target backend is a resumed CLI session.  ``_maybe_drop_recent_turns_for_cli``
                # would drop the result anyway (system_prompt is None for --resume),
                # so we save the disk I/O up-front.  _early_spec was resolved above
                # (None when routing failed → fall through to the normal read path).
                _skip_recent_turns = False
                if bool(getattr(_cfg, "CLI_SKIP_RECENT_TURNS_WHEN_SID", True)):
                    try:
                        _cli_providers = {"claude_cli", "gemini_cli", "kimi_cli"}
                        if (_early_spec is not None
                                and _early_spec.provider in _cli_providers
                                and _load_sid(chat_id, _early_spec.id)):
                            _skip_recent_turns = True
                            _debug_log(
                                f"[Cache] {_early_spec.provider} skip recent_turns "
                                f"pre-read chat={chat_id[:8]} sid present"
                            )
                    except Exception:
                        pass

                # API backends (anthropic_api / google_api / openai_compat_api)
                # carry conversation history via load_history() / save_history(),
                # so _run_backend_single unconditionally drops recent_turns for
                # them — the 100 KB disk read is always wasted.  Skip it here.
                # Flag named WHEN_HISTORY for config-file clarity; enforcement
                # is provider-level (not per-session history probe) because the
                # drop in _run_backend_single is also provider-level.
                if not _skip_recent_turns and bool(getattr(_cfg, "API_SKIP_RECENT_TURNS_WHEN_HISTORY", True)):
                    _api_providers = {"anthropic_api", "google_api", "openai_compat_api"}
                    if _early_spec is not None and _early_spec.provider in _api_providers:
                        _skip_recent_turns = True
                        _debug_log(
                            f"[Cache] {_early_spec.provider} skip recent_turns pre-read "
                            f"(API provider — history via load_history) chat={chat_id[:8]}"
                        )
                        try:
                            from larkhelm.metrics import inc_injection_gate
                            inc_injection_gate("recent_turns_api", "skipped_by_state")
                        except Exception:
                            pass

                if not _skip_recent_turns:
                    _raw_recent = _get_recent_turns(chat_id) or ""
                    if _raw_recent:
                        recent_turns_list = [
                            ln for ln in _raw_recent.splitlines() if ln.strip()
                        ]
            except Exception as _hist_err:
                _debug_log(f"[{trace_id}][DoQuery] rolling history error: {_hist_err}")

            try:
                from larkhelm.memory import get_memory_context_v2, maybe_auto_update
                memory_ctx, deduped_recent = get_memory_context_v2(
                    chat_id, cwd=cwd,
                    recent_turns=recent_turns_list,
                    sender_open_id=sender_open_id,
                )
            except Exception as _mem_err:
                _debug_log(f"[{trace_id}][DoQuery] memory context error: {_mem_err}")
                memory_ctx = ""
                deduped_recent = recent_turns_list

            try:
                from larkhelm.metrics import inc_injection_gate as _inc_ig2
                if bool(_cfg.config.get("project_guide_enabled")):
                    _is_cli_claude = getattr(_early_spec, "provider", "") == "claude_cli"
                    memory_ctx, _pg_outcome = _apply_project_guide_gate(
                        cwd, memory_ctx, _is_cli_claude
                    )
                    _inc_ig2("project_guide", _pg_outcome)
            except Exception as _pg_err:
                _debug_log(f"[{trace_id}][DoQuery] project_guide gate error: {_pg_err}")

            # ``recent_turns`` is kept separate from ``memory_ctx`` (rather than
            # concatenated as before) so the downstream ``_run_backend_single``
            # can drop it on API backends. API backends already receive the same
            # turns via ``load_history``; concatenating them into the system
            # channel was ~500 redundant input tokens per call (~50K / 100-turn
            # session). CLI + DeepSeek backends still need it because they don't
            # load a structured history.
            recent_turns = "\n".join(deduped_recent)

            from larkhelm.router import resolve_backend, LockedBackendUnavailableError
            from larkhelm.backend_registry import BACKEND_REGISTRY
            from larkhelm.health_signals import NO_HEALTH_UPDATE, NON_RETRIABLE
            from larkhelm.backend_cli import run_claude, run_gemini, run_kimi
            from larkhelm.backend_api import run_anthropic, run_google, run_openai_compat
            from larkhelm.api_session import load_history, save_history

            # Resolve primary backend (respects Rule 0 locked_backend, vision, doc routing)
            try:
                primary_spec = resolve_backend(chat_id, message, bool(images), has_doc_urls, force_backend_id)
                m_name = primary_spec.display_name
                card_state.update_model_name(m_name)
            except LockedBackendUnavailableError as _lbe:
                # User explicitly locked this backend; show error card, do not silently re-route
                _stop_hb.set()
                with card_state.card_patch_lock:   # drain any in-flight heartbeat patch before overwriting
                    pass
                hb_thread.join(timeout=0.5)
                if user_msg_id and _eyes_reaction_id[0]:
                    delete_reaction(user_msg_id, _eyes_reaction_id[0])
                    _eyes_reaction_id[0] = None
                from larkhelm.locale import _t as _lt
                from larkhelm.chat_state import _get_lang as _gl
                _lbe_lang = _gl(chat_id)
                send_card(chat_id,
                          _lt(_lbe_lang, "❌ 锁定后端不可用", "❌ Locked Backend Unavailable"),
                          f"{_lbe}\n\n" + _lt(_lbe_lang, "使用 **/lock off** 恢复自动路由。", "Use **/lock off** to restore auto-routing."),
                          color="red")
                return
            except Exception as _route_err:
                _debug_log(f"[{trace_id}][DoQuery] routing error: {_route_err}, using orchestrator chain")
                primary_spec = None

            # Build failover chain: primary spec first, then remaining orchestrators
            chain = BACKEND_REGISTRY.get_orchestrator_chain()
            if primary_spec is not None and primary_spec.healthy:
                chain_ids = [s.id for s in chain]
                if primary_spec.id not in chain_ids:
                    chain = [primary_spec] + chain
                elif chain and chain[0].id != primary_spec.id:
                    chain = [primary_spec] + [s for s in chain if s.id != primary_spec.id]

            # When user explicitly forces a backend (/c, /g, /k), skip delegation entirely:
            # run that single backend directly without orchestrator round-trip.
            _force_direct = bool(force_backend_id and primary_spec is not None and primary_spec.healthy)
            if _force_direct:
                chain = [primary_spec]

            successful_spec = None
            # Backend id the doc-injection dedup records get committed under
            # (overwritten below once the actually-used backend is known).
            _doc_commit_backend = model
            if not chain:
                # Last resort: fall back to legacy routing.
                # Same new-session-only rule: only prepend memory when no existing sid.
                _doc_commit_backend = (
                    "gemini" if model == "gemini" else
                    "kimi" if model == "kimi" else
                    "deepseek" if model == "deepseek" else "claude")
                _legacy_sid = _load_sid(chat_id, _doc_commit_backend)
                _legacy_msg = (f"[System]\n{memory_ctx}\n\n[User Query]\n{message}"
                               if memory_ctx and not _legacy_sid else message)
                if model == "gemini":
                    output = query_gemini(chat_id, _legacy_msg, cwd, cancel_ev,
                                          on_tool=on_tool, on_text=on_text,
                                          on_tool_result=on_tool_result,
                                          on_soft_timeout=_on_soft_timeout)
                elif model == "kimi":
                    output = query_kimi(chat_id, _legacy_msg, cwd, cancel_ev,
                                        on_tool=on_tool, on_text=on_text,
                                        on_tool_result=on_tool_result,
                                        on_soft_timeout=_on_soft_timeout,
                                        images=images)
                elif model == "deepseek":
                    output = query_deepseek(chat_id, _legacy_msg, cwd, cancel_ev,
                                            on_tool=on_tool, on_text=on_text,
                                            on_tool_result=on_tool_result,
                                            on_soft_timeout=_on_soft_timeout)
                else:
                    output = query_claude(chat_id, _legacy_msg, cwd, cancel_ev,
                                          on_tool=on_tool, on_text=on_text,
                                          on_tool_result=on_tool_result,
                                          on_soft_timeout=_on_soft_timeout,
                                          images=images)
            else:
                # Failover loop: iterate orchestrator chain, mark unhealthy on exception.
                # Delegation (worker_specs) is disabled: with a single orchestrator backend
                # the DELEGATE round-trip adds overhead with no benefit.  Worker-role
                # backends (e.g. DeepSeek) are still registered for other purposes but
                # should not receive user queries via the delegation path.
                worker_specs: dict = {}
                output = None
                last_err: Exception | None = None
                for attempt_spec in chain:
                    try:
                        m_name = attempt_spec.display_name
                        card_state.update_model_name(m_name)
                        if worker_specs:
                            output = _do_query_with_delegation(
                                chat_id, message, attempt_spec, worker_specs,
                                cwd, cancel_ev, on_text, on_tool, on_tool_result,
                                _on_soft_timeout, images=images, memory_ctx=memory_ctx,
                                recent_turns=recent_turns)
                        else:
                            output = _run_backend_single(
                                attempt_spec, chat_id, message, cwd, cancel_ev,
                                on_text, on_tool, on_tool_result, _on_soft_timeout, images,
                                extra_system=memory_ctx,
                                recent_turns=recent_turns)
                        successful_spec = attempt_spec
                        break
                    except QueryCancelledError:
                        raise
                    except TimeoutError:
                        raise
                    except Exception as _be:
                        _debug_log(f"[{trace_id}][DoQuery] backend {attempt_spec.id} failed: {_be}")
                        category = BACKEND_REGISTRY.record_call_failure(attempt_spec.id, str(_be))
                        last_err = _be
                        # OOM / timeout: host resource issue, not a retriable backend fault.
                        # Re-raise immediately so the outer handler shows an error card
                        # instead of silently falling back to a different backend.
                        if category in NON_RETRIABLE:
                            raise
                        # Show brief failover notice to user if more backends remain
                        remaining = [s for s in chain if s.healthy and s.id != attempt_spec.id]
                        if remaining and category not in NO_HEALTH_UPDATE:
                            card_state.set_current_text(
                                f"> ⚠️ {attempt_spec.display_name} 不可用，切换至 {remaining[0].display_name}..."
                            )

                if output is None:
                    raise RuntimeError(
                        f"所有 backend 均不可用。最近错误: {last_err}"
                    )

            elapsed = _fmt_elapsed(time.time() - start)
            if not output:
                output = "✅ 完成（无文本输出）"

            # Commit doc-injection dedup records: the backend returned
            # successfully, so its session now carries the injected bodies.
            # Committed under the backend that actually served the query
            # (failover may differ from the routing target used at injection
            # time) so /model switches and failover re-inject correctly.
            # Mid-risk aux op: failure only costs duplicate tokens later.
            if _doc_pending:
                try:
                    from larkhelm._context_cache import record_doc_injection
                    if successful_spec is not None:
                        _doc_commit_backend = successful_spec.id
                    for _doc_tok, _doc_hash in _doc_pending:
                        record_doc_injection(
                            chat_id, _doc_commit_backend, _doc_tok, _doc_hash)
                except Exception as _dc_err:
                    _debug_log(f"[{trace_id}][DoQuery] doc inject commit failed: {_dc_err}")

            # Strip <!--FILE:name-->...<!--/FILE--> markers before logging so
            # the log and card both show clean text. Files are sent after final card.
            _auto_files: list[tuple[str, str]] = []
            try:
                from larkhelm.file_handler import extract_file_markers
                output, _auto_files = extract_file_markers(output)
            except Exception as _mfe:
                _debug_log(f"[DoQuery] extract_file_markers failed: {_mfe}")

            log_model = successful_spec.id if successful_spec is not None else model
            log_entry(chat_id, "assistant", output, model=log_model, trace_id=trace_id)

            # Increment turn count and trigger memory auto-update in background
            # (with cold-start carry-over for first-turn chats whose all.jsonl
            # already has imported / pre-existing history).
            _post_query_memory_hook(chat_id, trace_id, sender_open_id)

            # Stop the heartbeat and wait for any in-flight patch to complete
            # before writing the final card.  join(timeout) alone is insufficient
            # because Feishu API calls can exceed 2 s; the lock drain is the real
            # guarantee — the join is just cleanup.
            _stop_hb.set()
            with card_state.card_patch_lock:   # blocks until any in-flight heartbeat patch finishes
                pass
            hb_thread.join(timeout=0.5)

            # Flush any still-in-flight tools into the completed list so the
            # final card / tools_list payload accurately reports them. The
            # serialised payload (sent to ``_make_card``) uses dict form for
            # back-compat with the card builder schema.
            card_state.snapshot_active_tools_as_completed()
            final_tool_records = card_state.snapshot_completed_tools()
            n_tools = len(final_tool_records)
            final_tools = [
                {
                    "name": t.name, "desc": t.desc,
                    "result": t.result,
                    "full_result": t.full_result,
                    "is_error": t.is_error, "elapsed": t.elapsed,
                }
                for t in final_tool_records
            ]

            note = (f"使用了 {n_tools} 次工具 · " if n_tools else "") + f"耗时 {elapsed}"

            chunks = _split_md(output.strip())
            n = len(chunks)

            # When the response spans multiple cards, keep the first card JSON small
            # (text only, no tools_list) to avoid exceeding Feishu's payload limit.
            # tools_list can be 100 KB+ with many tool calls; that causes both
            # _patch_card_raw and the fallback _send_card_raw to fail silently,
            # leaving the user with only the later numbered cards.
            first_title = f"🤖 {m_name}" + (f" (1/{n})" if n > 1 else "")
            _tools_payload = (final_tools if final_tools else None) if n == 1 else None
            if _tools_payload:
                import json as _json
                try:
                    if len(_json.dumps(_tools_payload, ensure_ascii=False)) > 20_000:
                        _tools_payload = None  # drop detailed results to stay within Feishu limit
                except Exception as e:
                    _debug_log(f"[query] tools payload serialize failed: {e}")
                    _tools_payload = None
            first_card = _make_card(first_title, chunks[0], color="blue",
                                    note=note if n == 1 else "",
                                    tools_list=_tools_payload,
                                    tools_expanded=False)
            if not _patch_card_raw(mid, first_card):
                _debug_log(f"[{trace_id}][DoQuery] final patch failed, falling back to send mid={mid}")
                fallback_text = f"[🤖 {m_name}]\n{chunks[0][:500]}"
                mid = _send_card_raw(chat_id, first_card, _fallback_text=fallback_text)

            for i, chunk in enumerate(chunks[1:], 2):
                is_last = (i == n)
                chunk_title = f"🤖 {m_name} ({i}/{n})"
                if is_last:
                    # Last card carries the note and tools summary
                    last_card = _make_card(chunk_title, chunk, color="blue",
                                           note=note,
                                           tools_list=final_tools if final_tools else None,
                                           tools_expanded=False)
                    fallback_text = f"[{chunk_title}]\n{chunk[:500]}"
                    _send_card_raw(chat_id, last_card, _fallback_text=fallback_text)
                else:
                    send_card(chat_id, chunk_title, chunk, color="blue")

            if user_msg_id and _eyes_reaction_id[0]:
                delete_reaction(user_msg_id, _eyes_reaction_id[0])
                _eyes_reaction_id[0] = None
                react_to_message(user_msg_id, EMOJI_DONE)

            # For long-running queries, reply to the user message with a completion notice to avoid spamming fast responses
            elapsed_sec = time.time() - start
            if user_msg_id and elapsed_sec >= 60:
                _reply_card_raw(
                    user_msg_id,
                    _make_card(f"✅ {m_name} 完成", f"耗时 {elapsed}", color="green"),
                    in_thread=False,
                )

            if mid:
                _index_reply(mid, chat_id, message, model)

            # Auto-send files extracted from AI output markers.
            if _auto_files:
                try:
                    from larkhelm.lark_client import send_text_as_file
                    for _af_name, _af_content in _auto_files:
                        send_text_as_file(chat_id, _af_content, _af_name,
                                          msg_id=user_msg_id)
                except Exception as _afe:
                    _debug_log(f"[DoQuery] auto file send failed: {_afe}")

        except QueryCancelledError:
            _stop_hb.set()
            with card_state.card_patch_lock: pass   # drain any in-flight heartbeat patch first
            elapsed = _fmt_elapsed(time.time() - start)
            # The card was synchronously updated to "cancelling" in the cancel callback response;
            # here we just patch with the final elapsed time; patch failure is safe (no stale cancel button)
            _patch_card_raw(mid, _make_card("🛑 已取消", f"查询已在 {elapsed} 后取消。", color="orange"))
            if user_msg_id and _eyes_reaction_id[0]:
                delete_reaction(user_msg_id, _eyes_reaction_id[0])
                _eyes_reaction_id[0] = None
        except TimeoutError as e:
            _stop_hb.set()
            with card_state.card_patch_lock: pass   # drain any in-flight heartbeat patch first
            elapsed = _fmt_elapsed(time.time() - start)
            log_entry(chat_id, "error", str(e), model=model, trace_id=trace_id)
            # Show the actual exception message rather than a hard-coded
            # ``HARD_TIMEOUT // 60`` blurb. Multiple paths funnel through
            # this except: BaseProcessRunner._on_kill_signal (idle window
            # exceeded), runner_gemini's response-timeout override, etc.
            # Hard-coding "360 分钟"
            # was the bug users hit when none of those reasons actually
            # applied — the card just always lied "360 分钟".
            err_msg = str(e).strip() or "未知超时"
            reply_card(chat_id, mid, f"⏰ 强制终止 ({elapsed})",
                       f"任务被超时机制终止：\n\n`{err_msg}`\n\n"
                       f"实际耗时：{elapsed}。可重新发送继续。\n"
                       "如确认任务正常运行被误杀，请增大 `hard_timeout` 配置 "
                       f"(当前: {_cfg.HARD_TIMEOUT // 60} 分钟，作用：子进程**无输出**超过该时长才会被判定卡死)。",
                       color="red")
            if user_msg_id and _eyes_reaction_id[0]:
                delete_reaction(user_msg_id, _eyes_reaction_id[0])
                react_to_message(user_msg_id, EMOJI_ERROR)
                _eyes_reaction_id[0] = None
        except Exception as e:
            _stop_hb.set()
            with card_state.card_patch_lock: pass   # drain any in-flight heartbeat patch first
            import sys
            print(traceback.format_exc(), file=sys.stderr)
            _debug_log(f"[{trace_id}][DoQuery] exception: {e}\n{traceback.format_exc()}")
            log_entry(chat_id, "error", str(e), model=model, trace_id=trace_id)
            lines = [l for l in str(e).splitlines() if l.strip()]
            last_line = lines[-1] if lines else str(e)
            reply_card(chat_id, mid, "❌ 错误", last_line, color="red",
                       note="执行 /status 查看详情")
            if user_msg_id and _eyes_reaction_id[0]:
                delete_reaction(user_msg_id, _eyes_reaction_id[0])
                react_to_message(user_msg_id, EMOJI_ERROR)
                _eyes_reaction_id[0] = None
        finally:
            _stop_hb.set()   # idempotent; ensures stop even if exception path skipped it
            hb_thread.join(timeout=1.0)

    finally:
        try:
            from larkhelm.handlers._query_card_state import record_query_end
            record_query_end(time.time() - start)
        except Exception:
            pass
        try:
            chat_lock.release()
        except RuntimeError:
            pass
        if not lock_released:
            pending = _pop_pending(chat_id)
            if pending:
                p_msg, p_model, p_user_msg_id, p_queue_mid = (
                    pending[0], pending[1], pending[2],
                    pending[3] if len(pending) >= 4 else None,
                )
                _debug_log(f"[Queue] processing queued message: {p_msg[:60]}")
                _reset_cancel(chat_id)
                threading.Thread(
                    target=_do_query,
                    args=(chat_id, p_msg, p_model, p_user_msg_id),
                    kwargs={"queue_card_mid": p_queue_mid},
                    daemon=True, name=f"query-{chat_id[:8]}"
                ).start()
        # Clean up image temp files created for this query.
        if images:
            import os as _os
            for _img in images:
                try:
                    if _img and str(_img).startswith("/tmp/"):
                        _os.unlink(_img)
                except Exception as e:
                    _debug_log(f"[query] temp image cleanup failed: {e}")
        # Clean up file downloads for this query.
        if files:
            try:
                cleanup_temp_paths([f.path for f in files])
            except Exception as _cte:
                _debug_log(f"[query] file temp cleanup failed: {_cte}")
