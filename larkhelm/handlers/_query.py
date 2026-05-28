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

# ── Card UX parameters (from config) ────────────────────────────────
# TOOL_HISTORY_CAP / STALL_THRESHOLD / CURSOR_FRAMES moved into
# ``_query_card_state.py`` along with the state machine that consumes
# them. The two intervals below stay here because ``_heartbeat_loop``
# (still in _do_query, not extracted) is the only consumer.
CARD_PUSH_INTERVAL = _cfg.CARD_PUSH_INTERVAL
CURSOR_INTERVAL    = _cfg.CURSOR_INTERVAL


def _should_use_query_session_v2(chat_id: str) -> bool:
    """P3 REQ-02: decide if this chat sees the v2 query path.

    Legacy boolean ``query_session_v2_enabled`` is preserved as a hard
    override: ``true`` forces v2 for every chat (traffic implicitly 1.0),
    ``false`` leaves the new ``query_session_v2_traffic`` knob in control.
    A traffic of 0.0 (default) keeps every chat on the legacy path → P2
    byte-compatibility.
    """
    if bool(_cfg.config.get("query_session_v2_enabled")):
        return True
    try:
        from larkhelm._gating import hash_bucket_allows
        traffic = float(getattr(_cfg, "QUERY_SESSION_V2_TRAFFIC", 0.0) or 0.0)
        return hash_bucket_allows(chat_id, traffic)
    except Exception as e:
        _debug_log(f"[DoQuery] v2 gating lookup failed, defaulting to legacy: {e}")
        return False


# ═══════════════════════════════════════════════════
#  Document URL auto-injection
# ═══════════════════════════════════════════════════

def _extract_feishu_urls(text: str) -> list:
    """Extract a list of Feishu document/Drive URLs from the given text."""
    return re.findall(r'https://[a-zA-Z0-9-]+\.feishu\.cn/[^\s\]>）]+', text)


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


def _inject_doc_context(text: str, chat_id: str) -> str:
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
            injections.append(
                f"{header}\n{result.content}\n[/文档内容]"
            )
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
    body = (
        "支持的飞书链接类型：\n"
        "- `docx` — 新版文档 `https://xxx.feishu.cn/docx/...`\n"
        "- `wiki` — Wiki 页面 `https://xxx.feishu.cn/wiki/...`\n"
        "- `sheets` — 电子表格 `https://xxx.feishu.cn/sheets/...`\n"
        "- `folder` / `drive` — 云盘文件夹 `https://xxx.feishu.cn/drive/folder/...`\n\n"
        "请检查链接类型是否正确，或在 URL 旁补一句你想做的事，"
        "机器人会按普通对话处理。"
    )
    send_card_reply(
        chat_id, user_msg_id,
        "⚠️ 飞书 URL 无法识别",
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
        output, new_history = run_anthropic(spec, chat_id, message, history, cancel_ev, on_text,
                                            extra_system=extra_system)
        save_history(provider, chat_id, new_history)
    elif provider == "google_api":
        history = load_history(provider, chat_id)
        output, new_history = run_google(spec, chat_id, message, history, cancel_ev, on_text,
                                         extra_system=extra_system)
        save_history(provider, chat_id, new_history)
    elif provider == "openai_compat_api":
        history = load_history(provider, chat_id)
        output, new_history = run_openai_compat(spec, chat_id, message, history, cancel_ev, on_text,
                                                extra_system=extra_system)
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


def _post_query_memory_hook(chat_id: str, trace_id: str) -> None:
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
    """
    try:
        from larkhelm.memory import maybe_auto_update
        old_count = _get_turn_count(chat_id)
        _increment_turn_count(chat_id)
        if old_count == 0:
            maybe_auto_update(chat_id, force=True)
            return
        maybe_auto_update(chat_id)
    except Exception as _mc_err:
        _debug_log(f"[{trace_id}][DoQuery] post-query memory error: {_mc_err}")


def _do_query(chat_id: str, message: str, model: str, user_msg_id: str = None,
              images: list = None, files: list = None,
              parent_id: str | None = None,
              force_backend_id: str | None = None,
              trace_id: str | None = None,
              sender_open_id: str | None = None,
              queue_card_mid: str | None = None):
    # P1-1 PR2: opt-in dispatch to the QuerySession rewrite. Default OFF so
    # existing behaviour is byte-identical until the flag flips.
    #
    # Fallback semantics (P1 round-2 review): the only SAFE fall-back to
    # legacy is when v2 hasn't committed any side effect yet — i.e. import
    # or construction failed. Once ``QuerySession.run()`` is entered, IT
    # owns the request: its internal try/except catches QueryCancelled /
    # Timeout / Exception and surfaces them via on_cancel / on_timeout /
    # on_error. Anything that *still* escapes is a v2 bug in unknown state
    # — falling back to legacy at that point would acquire the chat lock
    # a second time, emit a second init card, and run the LLM again. So we
    # log the escape and return: fail loud, never double-process.
    if _should_use_query_session_v2(chat_id):
        try:
            from larkhelm.handlers._query_session import QuerySession
            _session = QuerySession(
                chat_id=chat_id, message=message, model=model,
                user_msg_id=user_msg_id, images=images, files=files,
                parent_id=parent_id, force_backend_id=force_backend_id,
                sender_open_id=sender_open_id,
                queue_card_mid=queue_card_mid,
            )
        except Exception as _v2_setup_err:
            # Pre-side-effect failure (import / __init__) → legacy fallback is safe.
            _debug_log(
                f"[DoQuery] QuerySession setup failed, using legacy: {_v2_setup_err}"
            )
        else:
            try:
                _session.run()
            except Exception as _v2_run_err:
                _debug_log(
                    f"[DoQuery] QuerySession v2 raised post-setup; "
                    f"NOT falling back to avoid double-processing: {_v2_run_err}"
                )
            return   # v2 owned this request — success or failure.

    # Use caller-supplied trace_id (_message.py generates one BEFORE
    # logging the user entry so user/assistant pair share an id —
    # /stats duration pairing depends on it). Fall back to a fresh uuid
    # when called from a context that doesn't propagate trace_id (e.g.
    # crew sub-agent dispatch path).
    if not trace_id:
        trace_id = uuid.uuid4().hex[:12]

    chat_lock = _get_chat_lock(chat_id)
    cancel_ev = _get_cancel_event(chat_id)

    # Queue behind crew if one is running for this chat
    try:
        from larkhelm.crew._state import (
            current_owner, describe_active_owner, is_crew_running,
            owner_kind, subscribe_crew_done,
        )
        if is_crew_running(chat_id):
            existing_mid = _set_pending(chat_id, message, model, user_msg_id)
            preview = message[:80].replace("\n", " ")
            # C5 #14: ``is_crew_running`` is True for any ``_active_crew``
            # owner — including ``/plan``. Pre-C5 the card hardcoded
            # "⏳ Crew 运行中" + "当前 Crew 任务完成后自动执行" so when
            # the user was actually queued behind a plan they got the
            # wrong command name on every wait card (same family as
            # C4 #11 / #12). Decode the owner token and pick the title
            # and body to match. Falls back to the conservative "task"
            # wording for any unknown future owner class.
            owner_token = current_owner(chat_id)
            kind = owner_kind(owner_token)
            if kind == "plan":
                _q_title = "⏳ Plan 运行中"
                _q_body_lead = f"当前 {describe_active_owner(owner_token)} 完成后自动执行"
            elif kind == "crew":
                _q_title = "⏳ Crew 运行中"
                _q_body_lead = "当前 Crew 任务完成后自动执行"
            else:
                _q_title = "⏳ 任务运行中"
                _q_body_lead = "当前任务完成后自动执行"
            crew_queue_card = _make_card(
                _q_title,
                f"{_q_body_lead}：\n\n> {preview}",
                color="orange",
                buttons=[("❌ 取消排队", f"cancel_queue:{chat_id}")]
            )
            # Pending state is already written; if card emission fails
            # (Feishu 5xx / network), the user has no visible queue card
            # AND no way to cancel — roll back so the next message can
            # re-queue cleanly. Same fix lives in QuerySession (P1 round-3
            # follow-up); kept in sync here.
            try:
                if existing_mid:
                    _patch_card_raw(existing_mid, crew_queue_card)
                else:
                    if user_msg_id:
                        _mid = _reply_card_raw(user_msg_id, crew_queue_card, in_thread=False)
                    else:
                        _mid = _send_card_raw(chat_id, crew_queue_card)
                    _update_pending_card_mid(chat_id, _mid)
            except Exception as _card_err:
                _debug_log(
                    f"[DoQuery] queue card emission failed, rolling back "
                    f"pending state: {_card_err}"
                )
                _pop_pending(chat_id)
                return

            # subscribe_crew_done is race-safe: if crew ended between is_crew_running
            # and here, it returns a pre-set event so the watcher fires immediately.
            _done_ev = subscribe_crew_done(chat_id)

            def _after_crew(_ev=_done_ev, _cid=chat_id, _msg=message, _model=model, _uid=user_msg_id):
                _ev.wait(timeout=4 * 3600)
                pending = _pop_pending(_cid)
                if pending:
                    p_msg, p_model, p_user_msg_id, p_queue_mid = (
                        pending[0], pending[1], pending[2],
                        pending[3] if len(pending) >= 4 else None,
                    )
                    _reset_cancel(_cid)
                    threading.Thread(
                        target=_do_query,
                        args=(_cid, p_msg, p_model, p_user_msg_id),
                        kwargs={"queue_card_mid": p_queue_mid},
                        daemon=True, name=f"query-{_cid[:8]}",
                    ).start()

            threading.Thread(target=_after_crew, daemon=True,
                             name=f"crew-wait-{chat_id[:8]}").start()
            return
    except Exception as _crew_check_err:
        _debug_log(f"[{trace_id}][DoQuery] crew check error: {_crew_check_err}")

    if not chat_lock.acquire(blocking=False):
        existing_mid = _set_pending(chat_id, message, model, user_msg_id)
        preview = message[:80].replace("\n", " ")
        queue_card = _make_card(
            "⏳ 排队中",
            f"将在当前任务完成后自动执行：\n\n> {preview}",
            color="orange",
            buttons=[("❌ 取消排队", f"cancel_queue:{chat_id}")]
        )
        # Same pending-rollback pattern as the crew-queue branch above:
        # if card emission fails, the user has no visible queue indicator
        # and no cancel button → roll back so the next message can re-queue.
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
    # Hoisting the assignment also keeps semantics with the new
    # ``QuerySession.run`` parity fix (see _query_session.py:97).
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
                    from larkhelm.lark_client import _fetch_parent_message_text
                    parent_text = _fetch_parent_message_text(parent_id)
                    if parent_text:
                        message = (
                            f"[用户回复了以下消息]\n\n{parent_text}\n\n---\n\n{message}"
                        )
                        _debug_log(f"[{trace_id}][DoQuery] injected parent context ({len(parent_text)} chars)")
                    else:
                        from larkhelm.crew import get_recent_crew_context
                        crew_ctx = get_recent_crew_context(chat_id)
                        if crew_ctx:
                            message = (
                                f"[以下是刚完成的 Crew 任务「{crew_ctx['title']}」的交付结论，"
                                f"请结合它来回答我的问题]\n\n"
                                f"{crew_ctx['summary']}\n\n---\n\n{message}"
                            )
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
            if _cfg.DOC_AUTO_INJECT:
                try:
                    message = _inject_doc_context(message, chat_id)
                except Exception as _doc_err:
                    _debug_log(f"[{trace_id}][DoQuery] doc inject error: {_doc_err}")

            # Memory is passed as extra_system (proper system channel) rather than prepended
            # to the user message, so the model receives clean user turn content.
            #
            # Phase B: ``get_memory_context_v2`` lets the builder see the live
            # query + recent turns so it can apply lazy-global / project-conditional
            # / session-layered / recent-turns dedup. Default flags keep the
            # builder fail-open (everything injected) when in doubt.
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

                if not _skip_recent_turns and bool(getattr(_cfg, "API_SKIP_RECENT_TURNS_WHEN_HISTORY", True)):
                    _api_providers = {"anthropic_api", "google_api", "openai_compat_api"}
                    if _early_spec is not None and _early_spec.provider in _api_providers:
                        _skip_recent_turns = True
                        _debug_log(
                            f"[Cache] {_early_spec.provider} skip recent_turns "
                            f"pre-read (API backend) chat={chat_id[:8]}"
                        )
                        try:
                            from larkhelm.metrics import inc_injection_gate
                            inc_injection_gate("recent_turns_api", "skipped_by_state")
                        except Exception:
                            pass

                if not _skip_recent_turns:
                    # Compute a dedup_prefix from the session memory's Work
                    # Context slot so summarised content doesn't double-inject
                    # into the orchestrator. Any failure (memory not loaded,
                    # parse miss) cleanly degrades to ``dedup_prefix=None`` →
                    # byte-compatible with the PR-prior behaviour.
                    _dedup_prefix: str | None = None
                    try:
                        from larkhelm.memory import load_memory
                        from larkhelm.memory_context import extract_work_context
                        _session_raw = load_memory(chat_id)
                        _wc = extract_work_context(_session_raw)
                        _dedup_prefix = _wc or None
                    except Exception as _wc_err:
                        _debug_log(f"[{trace_id}][DoQuery] work_context extract error: {_wc_err}")
                        _dedup_prefix = None
                    try:
                        _raw_recent = _get_recent_turns(chat_id, dedup_prefix=_dedup_prefix) or ""
                    except Exception as _rt_err:
                        _debug_log(f"[{trace_id}][DoQuery] dedup recent_turns error: {_rt_err}, retrying without prefix")
                        _raw_recent = _get_recent_turns(chat_id) or ""
                    if _raw_recent:
                        recent_turns_list = [
                            ln for ln in _raw_recent.splitlines() if ln.strip()
                        ]
            except Exception as _hist_err:
                _debug_log(f"[{trace_id}][DoQuery] rolling history error: {_hist_err}")

            # Phase D: lift any IntentResult that _message.py staged before
            # falling through to _do_query so memory injection can use the
            # per-agent policy. Wrapped in try/except + None fallback so the
            # default v2 behaviour holds when intent_router_enabled is off
            # (or the chat_state import races during early bootstrap).
            _pending_intent = None
            try:
                from larkhelm.chat_state import _pop_pending_intent
                _pending_intent = _pop_pending_intent(chat_id)
            except Exception as _pi_err:
                _debug_log(f"[{trace_id}][DoQuery] _pop_pending_intent error: {_pi_err}")
                _pending_intent = None

            _MEMORY_SKIP_GLOBAL_INTENTS = {"dev", "crew", "shell"}

            try:
                from larkhelm.memory import get_memory_context_v2, maybe_auto_update
                memory_ctx, deduped_recent = get_memory_context_v2(
                    chat_id, cwd=cwd, query=message,
                    recent_turns=recent_turns_list,
                    has_doc_urls=has_doc_urls,
                    intent=_pending_intent,
                    sender_open_id=sender_open_id,
                )
            except Exception as _mem_err:
                _debug_log(f"[{trace_id}][DoQuery] memory context error: {_mem_err}")
                memory_ctx = ""
                deduped_recent = recent_turns_list

            try:
                from larkhelm.metrics import inc_injection_gate as _inc_ig
                _intent_type = getattr(_pending_intent, "agent_type", "") or ""
                if (bool(_cfg.config.get("memory_intent_policy_enabled"))
                        and _intent_type in _MEMORY_SKIP_GLOBAL_INTENTS
                        and "[GLOBAL MEMORY]" in memory_ctx):
                    import re as _re_ig
                    memory_ctx = _re_ig.sub(
                        r"\[GLOBAL MEMORY\].*?\[/GLOBAL MEMORY\]\n*",
                        "",
                        memory_ctx,
                        flags=_re_ig.DOTALL,
                    ).lstrip()
                    _inc_ig("memory_global", "skipped")
                else:
                    _inc_ig("memory_global", "injected")
            except Exception as _ig_err:
                _debug_log(f"[{trace_id}][DoQuery] memory_intent_policy gate error: {_ig_err}")

            try:
                from larkhelm.metrics import inc_injection_gate as _inc_ig2
                if bool(_cfg.config.get("project_guide_enabled")) and _cfg.config.get("project_guide_path"):
                    from pathlib import Path as _GPath
                    _is_cli_claude = getattr(_early_spec, "provider", "") == "claude_cli"
                    if _is_cli_claude:
                        _inc_ig2("project_guide", "skipped_cli")
                    else:
                        try:
                            _guide_path = _GPath(_cfg.config["project_guide_path"]).expanduser()
                            _guide_content = _guide_path.read_text(encoding="utf-8")
                            if len(_guide_content) > 4000:
                                _debug_log(
                                    f"[DoQuery] project_guide exceeds 4000 chars "
                                    f"({len(_guide_content)}), truncating"
                                )
                                _guide_content = _guide_content[:4000]
                            memory_ctx = (
                                f"[Project Guide]\n{_guide_content}\n[/Project Guide]\n\n"
                                + memory_ctx
                            )
                            _inc_ig2("project_guide", "injected")
                        except Exception as _pe:
                            _debug_log(f"[DoQuery] project_guide read failed: {_pe}")
                            _inc_ig2("project_guide", "error")
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
            from larkhelm.health_signals import NO_HEALTH_UPDATE
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
                send_card(chat_id, "❌ 锁定后端不可用",
                          f"{_lbe}\n\n使用 **/lock off** 恢复自动路由。", color="red")
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
            if not chain:
                # Last resort: fall back to legacy routing.
                # Same new-session-only rule: only prepend memory when no existing sid.
                _legacy_sid = _load_sid(chat_id,
                    "gemini" if model == "gemini" else
                    "kimi" if model == "kimi" else
                    "deepseek" if model == "deepseek" else "claude")
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
                # Failover loop: iterate orchestrator chain, mark unhealthy on exception
                # No delegation when the user explicitly forced a specific backend.
                worker_specs = {} if _force_direct else {
                    s.id: s for s in BACKEND_REGISTRY.all_enabled()
                    if s.healthy and s.role != "orchestrator"
                }
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
            _post_query_memory_hook(chat_id, trace_id)

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
            # exceeded), runner_gemini's response-timeout override, the
            # crew watcher's wall-clock kill, etc. Hard-coding "360 分钟"
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
                from larkhelm.handlers._query_pure import cleanup_temp_paths
                cleanup_temp_paths([f.path for f in files])
            except Exception as _cte:
                _debug_log(f"[query] file temp cleanup failed: {_cte}")
