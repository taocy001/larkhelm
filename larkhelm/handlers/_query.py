"""
larkhelm · main query engine

Contains:
  - _extract_feishu_urls()   Extract Feishu document URLs from a message
  - _inject_doc_context()    Auto-inject Feishu document content into prompt
  - _do_query()              Execute an AI query with streaming card updates
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
    send_card, reply_card, _send_card_raw, _patch_card_raw, _reply_card_raw,
    _pin_task_card, react_to_message, delete_reaction,
    EMOJI_PROCESSING, EMOJI_DONE, EMOJI_ERROR,
    _index_reply,
)
from larkhelm.card_builder import _make_card, _split_md, _fmt_elapsed
from larkhelm.ai_runner import query_claude, query_gemini, query_kimi, QueryCancelledError
from larkhelm.chat_state import _get_cwd, _load_sid, _increment_turn_count

# ── Card UX parameters (from config) ────────────────────────────────
TOOL_HISTORY_CAP   = _cfg.TOOL_HISTORY_CAP

CARD_PUSH_INTERVAL = _cfg.CARD_PUSH_INTERVAL
CURSOR_INTERVAL    = _cfg.CURSOR_INTERVAL
STALL_THRESHOLD    = _cfg.STALL_THRESHOLD
CURSOR_FRAMES      = _cfg.CURSOR_FRAMES


# ═══════════════════════════════════════════════════
#  Document URL auto-injection
# ═══════════════════════════════════════════════════

def _extract_feishu_urls(text: str) -> list:
    """Extract a list of Feishu document/Drive URLs from the given text."""
    return re.findall(r'https://[a-zA-Z0-9-]+\.feishu\.cn/[^\s\]>）]+', text)


def _inject_doc_context(text: str, chat_id: str) -> str:
    """
    Detect Feishu document URLs in text, read their content, and prepend it to
    the prompt. At most DOC_INJECT_MAX_DOCS documents are injected, each capped
    at DOC_INJECT_MAX_CHARS characters. Failures are silently skipped so the
    normal conversation is unaffected.
    """
    from larkhelm.lark_client import (
        FeishuDocClient, parse_doc_url,
        DocPermissionError, DocError,
    )
    urls = _extract_feishu_urls(text)
    if not urls:
        return text
    doc_client = FeishuDocClient()
    injections = []
    for url in urls:
        if len(injections) >= _cfg.DOC_INJECT_MAX_DOCS:
            break
        ref = parse_doc_url(url)
        if ref is None:
            continue
        try:
            result = doc_client.read(ref, max_chars=_cfg.DOC_INJECT_MAX_CHARS)
            label  = result.title or url
            injections.append(
                f"[文档内容：《{label}》]\n{result.content}\n[/文档内容]"
            )
        except DocPermissionError:
            injections.append(f"[文档 {url} 无读取权限，已跳过]")
        except DocError:
            pass  # Other errors: silently skip, do not disrupt AI query
    if injections:
        return "\n\n".join(injections) + "\n\n---\n\n" + text
    return text


# ═══════════════════════════════════════════════════
#  AI query execution
# ═══════════════════════════════════════════════════

def _run_backend_single(spec, chat_id: str, message: str, cwd: str, cancel_ev,
                        on_text, on_tool, on_tool_result, on_soft_timeout,
                        images=None, extra_system: str = "") -> str:
    """Run a single backend and return its output string.

    extra_system: for API backends, injected through the proper system channel;
                  for CLI backends, prepended as [System]...[User Query] text.
    """
    from larkhelm.backend_cli import run_claude, run_gemini, run_kimi
    from larkhelm.backend_api import run_anthropic, run_google, run_openai_compat
    from larkhelm.api_session import load_history, save_history
    from larkhelm.chat_state import _load_sid

    provider = spec.provider
    if provider == "anthropic_api":
        history = load_history(provider, chat_id)
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
        sid = _load_sid(chat_id, "gemini")
        cli_msg = f"[System]\n{extra_system}\n\n[User Query]\n{message}" if extra_system else message
        output = run_gemini(spec, chat_id, cli_msg, sid, cwd, cancel_ev,
                            on_text=on_text, on_tool=on_tool, on_tool_result=on_tool_result,
                            on_soft_timeout=on_soft_timeout)
    elif provider == "kimi_cli":
        sid = _load_sid(chat_id, "kimi")
        cli_msg = f"[System]\n{extra_system}\n\n[User Query]\n{message}" if extra_system else message
        output = run_kimi(spec, chat_id, cli_msg, sid, cwd, cancel_ev,
                          on_text=on_text, on_tool=on_tool, on_tool_result=on_tool_result,
                          on_soft_timeout=on_soft_timeout, images=images)
    else:  # claude_cli (default)
        sid = _load_sid(chat_id, "claude")
        cli_msg = f"[System]\n{extra_system}\n\n[User Query]\n{message}" if extra_system else message
        output = run_claude(spec, chat_id, cli_msg, sid, cwd, cancel_ev,
                            on_text=on_text, on_tool=on_tool, on_tool_result=on_tool_result,
                            on_soft_timeout=on_soft_timeout, images=images, allow_retry=True)
    return output


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
) -> str:
    """Execute query with delegation support (max 2 hops).

    Phase 1: Stream orchestrator response, buffer first 60 chars for DELEGATE detection.
    Phase 2: If DELEGATE found, run specialist; on_tool/on_tool_result show progress.
    Phase 3: Re-run orchestrator with specialist result for synthesis.
    Falls back to direct answer if specialist unavailable or delegation malformed.
    """
    from larkhelm.orchestration import build_orchestrator_system_prompt, _detect_delegation
    from larkhelm.backend_registry import BACKEND_REGISTRY

    if hop >= 2:
        return _run_backend_single(orch_spec, chat_id, enriched_msg, cwd, cancel_ev,
                                   on_text, on_tool, on_tool_result, on_soft_timeout, images)

    # Build system prompt listing specialists
    system_prompt = build_orchestrator_system_prompt(BACKEND_REGISTRY)

    # Phase 1: run orchestrator with buffered on_text to detect DELEGATE prefix
    BUFFER_THRESHOLD = 60
    _buf_state = {"text": "", "delegation": None, "flushed": False}

    def _buffered_on_text(text: str, status: str = "typing"):
        _buf_state["text"] = text
        if _buf_state["delegation"] is not None:
            return  # delegation already detected, hide all orchestrator text
        if len(text) >= BUFFER_THRESHOLD and not _buf_state["flushed"]:
            result = _detect_delegation(text)
            if result:
                _buf_state["delegation"] = result
                return  # hide from card
            _buf_state["flushed"] = True
            on_text(text, status)
        elif _buf_state["flushed"]:
            on_text(text, status)
        # else: still buffering (< BUFFER_THRESHOLD), show nothing yet

    orch_output = _run_backend_single(orch_spec, chat_id, enriched_msg, cwd, cancel_ev,
                                      _buffered_on_text, on_tool, on_tool_result, on_soft_timeout,
                                      images, extra_system=system_prompt)

    # Final check on complete output (catches END_DELEGATE arriving late)
    delegation = _buf_state["delegation"] or _detect_delegation(orch_output)

    if not delegation:
        # No delegation: flush accumulated text if heartbeat hasn't seen it yet
        if not _buf_state["flushed"]:
            on_text(orch_output)
        return orch_output

    # Phase 2: Specialist execution
    backend_id, sub_query = delegation
    specialist_spec = worker_specs.get(backend_id)

    if specialist_spec is None or not specialist_spec.healthy or not specialist_spec.enabled:
        _debug_log(f"[Delegation] specialist {backend_id} unavailable, falling back to direct orchestrator")
        on_text(f"> ⚠️ 专家 {backend_id} 不可用，正在直接回答...")
        # Call without extra_system so orchestrator won't receive DELEGATE instructions and answers directly
        return _run_backend_single(orch_spec, chat_id, enriched_msg, cwd, cancel_ev,
                                   on_text, on_tool, on_tool_result, on_soft_timeout, images)

    on_tool(f"🔀 委托 {specialist_spec.display_name}", sub_query[:120], "delegation")

    _spec_start = time.monotonic()
    try:
        specialist_output = _run_backend_single(
            specialist_spec, chat_id, sub_query, cwd, cancel_ev,
            lambda t, s="typing": None,  # suppress specialist text from main card
            lambda *a: None,
            lambda *a: None,
            on_soft_timeout,  # propagate so chat lock is released if specialist stalls
        )
    except Exception as _spec_err:
        _debug_log(f"[Delegation] specialist {backend_id} failed: {_spec_err}")
        specialist_output = f"[Specialist {backend_id} failed: {_spec_err}]"

    _spec_elapsed = time.monotonic() - _spec_start
    on_tool_result("delegation", specialist_output[:500], False, _spec_elapsed)

    # Phase 3: Orchestrator synthesis
    synthesis_msg = (
        f"{enriched_msg}\n\n"
        f"[{specialist_spec.display_name} specialist result:]\n{specialist_output}"
    )
    return _do_query_with_delegation(
        chat_id, synthesis_msg, orch_spec, worker_specs,
        cwd, cancel_ev, on_text, on_tool, on_tool_result, on_soft_timeout,
        images=images, hop=hop + 1,
    )


def _do_query(chat_id: str, message: str, model: str, user_msg_id: str = None,
              images: list = None):
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
        if existing_mid:
            _patch_card_raw(existing_mid, queue_card)
        else:
            if user_msg_id:
                mid = _reply_card_raw(user_msg_id, queue_card, in_thread=False)
            else:
                mid = _send_card_raw(chat_id, queue_card)
            _update_pending_card_mid(chat_id, mid)
        return

    _eyes_reaction_id: list[str | None] = [None]
    if user_msg_id:
        _eyes_reaction_id[0] = react_to_message(user_msg_id, EMOJI_PROCESSING)

    lock_released = False   # Set to True when the soft-timeout releases the lock early, preventing double-release in finally

    try:
        m_name = {"claude": "Claude", "gemini": "Gemini", "kimi": "Kimi"}.get(model, model.capitalize())
        cwd = _get_cwd(chat_id)
        spec = None  # resolved by resolve_backend inside the try block below
        start = time.time()

        init_card = _make_card(f"⏳ {m_name} 连接中",
                               f"> 正在启动...\n\n目录: `{cwd}`", color="grey",
                               buttons=[("🛑 取消", f"cancel:{chat_id}")])
        if user_msg_id:
            mid = _reply_card_raw(user_msg_id, init_card, in_thread=False)
        else:
            mid = _send_card_raw(chat_id, init_card)
        if mid:
            _pin_task_card(chat_id, mid)

        # ── Tool state ──────────────────────────────────────
        active_tools: dict[str, dict] = {}
        completed_tools: list[dict] = []
        _tools_lock = threading.Lock()

        # ── Card push state ──────────────────────────────────
        _dirty             = False
        _cursor_idx        = 0
        _current_text      = ""
        _last_pushed_body  = ""
        _last_heartbeat    = time.monotonic()
        _in_background     = False

        # Use nonlocal instead of single-element list
        _state_lock_local = threading.Lock()
        # Serialises _patch_card_raw calls between heartbeat and main thread.
        # The main thread acquires this (without holding it) after _stop_hb.set()
        # to wait for any in-flight heartbeat patch before writing the final card.
        _card_patch_lock  = threading.Lock()

        def _get_state():
            with _state_lock_local:
                return _dirty, _cursor_idx, _current_text, _last_pushed_body, _last_heartbeat, _in_background

        def _set_dirty(v: bool):
            nonlocal _dirty
            with _state_lock_local:
                _dirty = v

        def _set_current_text(v: str):
            nonlocal _current_text, _dirty
            with _state_lock_local:
                _current_text = v
                _dirty = True

        def _set_in_background(v: bool):
            nonlocal _in_background, _dirty
            with _state_lock_local:
                _in_background = v
                _dirty = True

        def _tick_cursor():
            nonlocal _cursor_idx
            with _state_lock_local:
                _cursor_idx = (_cursor_idx + 1) % len(CURSOR_FRAMES)

        def _update_heartbeat():
            nonlocal _last_heartbeat
            with _state_lock_local:
                _last_heartbeat = time.monotonic()

        # ── Render card content ──────────────────────────────────
        def _render_body() -> tuple[str, str | None, str]:
            elapsed = _fmt_elapsed(time.time() - start)

            with _tools_lock:
                act = dict(active_tools)
                comp = list(completed_tools)

            with _state_lock_local:
                cur_text = _current_text
                cursor_i = _cursor_idx
                in_bg    = _in_background

            tool_parts: list[str] = []
            n_hidden = max(0, len(comp) - TOOL_HISTORY_CAP)
            if n_hidden > 0:
                tool_parts.append(f"_+{n_hidden} 条更早记录已隐藏_")

            def _fmt_desc(d: str) -> str:
                if not d:
                    return ""
                if "\n" in d:
                    return f"\n```\n{d}\n```"
                return f"  \n`{d}`"

            for t in comp[-TOOL_HISTORY_CAP:]:
                icon = "✗" if t["is_error"] else "✓"
                desc_str = _fmt_desc(t.get("desc", ""))
                tool_parts.append(
                    f"{icon} **{t['name']}** ({_fmt_elapsed(t['elapsed'])}){desc_str}"
                )
            now_mono = time.monotonic()
            for t in act.values():
                tool_elapsed = now_mono - t["start"]
                desc_str = _fmt_desc(t.get("desc", ""))
                if tool_elapsed > STALL_THRESHOLD:
                    tool_parts.append(f"🔧 **{t['name']}** ⚠️ 响应停滞 ({_fmt_elapsed(tool_elapsed)}){desc_str}")
                else:
                    tool_parts.append(f"🔧 **{t['name']}** ({_fmt_elapsed(tool_elapsed)})…{desc_str}")
            tools_md = "\n\n".join(tool_parts) if tool_parts else None

            if cur_text.strip():
                cursor = CURSOR_FRAMES[cursor_i]
                response_md = cur_text.strip() + cursor
            elif not tool_parts:
                response_md = "> 正在思考..."
            else:
                response_md = ""

            bg_prefix = "后台·" if in_bg else ""
            if act:
                title = f"⚙️ {m_name} · {bg_prefix}工具调用中 ({elapsed})"
            elif cur_text.strip():
                title = f"✍️ {m_name} · {bg_prefix}回应中 ({elapsed})"
            else:
                title = f"⏳ {m_name} · {bg_prefix}思考中 ({elapsed})"

            return title, tools_md, response_md

        # ── Push card ─────────────────────────────────────
        def _push_if_needed(force: bool = False, include_cancel: bool = True):
            nonlocal _last_pushed_body, _dirty
            if cancel_ev.is_set() or _stop_hb.is_set():
                return
            title, tools_md, response_md = _render_body()
            combined = f"{title}||{tools_md}||{response_md}"
            with _state_lock_local:
                need_push = force or _dirty or combined != _last_pushed_body
            if need_push:
                btns = [("🛑 取消", f"cancel:{chat_id}")] if include_cancel else None
                card_json = _make_card(title, response_md, color="grey",
                                       tools_md=tools_md, tools_expanded=True,
                                       buttons=btns)
                with _card_patch_lock:
                    # Re-check inside the lock: if the main thread already set _stop_hb
                    # and drained this lock, don't overwrite the final card.
                    if cancel_ev.is_set() or _stop_hb.is_set():
                        return
                    _patch_card_raw(mid, card_json)
                with _state_lock_local:
                    _last_pushed_body = combined
                    _dirty = False

        # ── Callback: tool invocation ────────────────────────────────
        def on_tool(name: str, desc: str, tool_id: str = ""):
            with _tools_lock:
                now_mono = time.monotonic()
                for tid, t in list(active_tools.items()):
                    elapsed = now_mono - t["start"]
                    completed_tools.append({
                        "name": t["name"], "desc": t["desc"],
                        "result": "", "is_error": False, "elapsed": elapsed,
                    })
                active_tools.clear()
                active_tools[tool_id] = {"name": name, "desc": desc, "start": now_mono}
            _set_dirty(True)
            log_entry(chat_id, "tool", f"{name}: {desc}", model=model, trace_id=trace_id)

        # ── Callback: tool result ────────────────────────────────
        def on_tool_result(tool_id: str, result: str, is_error: bool, elapsed: float):
            with _tools_lock:
                info = active_tools.pop(tool_id, None)
                if info:
                    completed_tools.append({
                        "name":      info["name"],
                        "desc":      info["desc"],
                        "result":    result,
                        "full_result": result[:5000] if len(result) > 200 else "",
                        "is_error":  is_error,
                        "elapsed":   elapsed,
                    })
            _set_dirty(True)

        # ── Callback: streaming text ────────────────────────────────
        def on_text(text: str, status: str = "typing"):
            _set_current_text(text)

        # ── Soft-timeout callback ────────────────────────────────
        def _on_soft_timeout():
            nonlocal lock_released
            elapsed_now = _fmt_elapsed(time.time() - start)
            _debug_log(f"[{trace_id}][DoQuery] soft timeout ({elapsed_now}), lock released, continuing in background")
            _set_in_background(True)
            try:
                chat_lock.release()
            except RuntimeError:
                pass
            lock_released = True
            _replace_cancel_event(chat_id)
            pending = _pop_pending(chat_id)
            if pending:
                p_msg, p_model, p_user_msg_id, *_ = pending
                _debug_log(f"[Queue/SoftTimeout] processing queued message: {p_msg[:60]}")
                threading.Thread(
                    target=_do_query,
                    args=(chat_id, p_msg, p_model, p_user_msg_id),
                    daemon=True, name=f"query-{chat_id[:8]}",
                ).start()

        # ── Heartbeat thread ─────────────────────────────────────
        _stop_hb = threading.Event()

        def _heartbeat_loop():
            while not _stop_hb.is_set():
                try:
                    _tick_cursor()
                    now = time.monotonic()
                    with _state_lock_local:
                        last_hb = _last_heartbeat
                        dirty_now = _dirty
                        in_bg = _in_background
                    # After soft timeout the task runs in background; cancel button
                    # is no longer wired to the new cancel event, so hide it.
                    show_cancel = not in_bg
                    if now - last_hb >= CARD_PUSH_INTERVAL:
                        _push_if_needed(force=True, include_cancel=show_cancel)
                        _update_heartbeat()
                    elif dirty_now:
                        _push_if_needed(force=False, include_cancel=show_cancel)
                except Exception as e:
                    _debug_log(f"[Heartbeat] exception: {e}")
                _stop_hb.wait(timeout=CURSOR_INTERVAL)

        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True,
                                     name=f"hb-{chat_id[:8]}")
        hb_thread.start()

        try:
            # ── Memory injection + backend routing ──────────────────────────
            try:
                from larkhelm.memory import inject_memory, maybe_auto_update
                enriched_message = inject_memory(chat_id, message)
            except Exception as _mem_err:
                _debug_log(f"[{trace_id}][DoQuery] memory inject error: {_mem_err}")
                enriched_message = message

            has_doc_urls = bool(_extract_feishu_urls(message))

            from larkhelm.router import resolve_backend, LockedBackendUnavailableError
            from larkhelm.backend_registry import BACKEND_REGISTRY
            from larkhelm.backend_cli import run_claude, run_gemini, run_kimi
            from larkhelm.backend_api import run_anthropic, run_google, run_openai_compat
            from larkhelm.api_session import load_history, save_history

            # Resolve primary backend (respects Rule 0 locked_backend, vision, doc routing)
            try:
                primary_spec = resolve_backend(chat_id, enriched_message, bool(images), has_doc_urls)
                m_name = primary_spec.display_name
            except LockedBackendUnavailableError as _lbe:
                # User explicitly locked this backend; show error card, do not silently re-route
                _stop_hb.set()
                with _card_patch_lock:   # drain any in-flight heartbeat patch before overwriting
                    pass
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

            successful_spec = None
            if not chain:
                # Last resort: fall back to legacy routing
                if model == "gemini":
                    output = query_gemini(chat_id, enriched_message, cwd, cancel_ev,
                                          on_tool=on_tool, on_text=on_text,
                                          on_tool_result=on_tool_result,
                                          on_soft_timeout=_on_soft_timeout)
                elif model == "kimi":
                    output = query_kimi(chat_id, enriched_message, cwd, cancel_ev,
                                        on_tool=on_tool, on_text=on_text,
                                        on_tool_result=on_tool_result,
                                        on_soft_timeout=_on_soft_timeout,
                                        images=images)
                else:
                    output = query_claude(chat_id, enriched_message, cwd, cancel_ev,
                                          on_tool=on_tool, on_text=on_text,
                                          on_tool_result=on_tool_result,
                                          on_soft_timeout=_on_soft_timeout,
                                          images=images)
            else:
                # Failover loop: iterate orchestrator chain, mark unhealthy on exception
                worker_specs = {
                    s.id: s for s in BACKEND_REGISTRY.all_enabled()
                    if s.healthy and s.role != "orchestrator"
                }
                output = None
                successful_spec = None
                last_err: Exception | None = None
                for attempt_spec in chain:
                    try:
                        m_name = attempt_spec.display_name
                        if worker_specs:
                            output = _do_query_with_delegation(
                                chat_id, enriched_message, attempt_spec, worker_specs,
                                cwd, cancel_ev, on_text, on_tool, on_tool_result,
                                _on_soft_timeout, images=images)
                        else:
                            output = _run_backend_single(
                                attempt_spec, chat_id, enriched_message, cwd, cancel_ev,
                                on_text, on_tool, on_tool_result, _on_soft_timeout, images)
                        successful_spec = attempt_spec
                        break
                    except QueryCancelledError:
                        raise
                    except TimeoutError:
                        raise
                    except Exception as _be:
                        _debug_log(f"[{trace_id}][DoQuery] backend {attempt_spec.id} failed: {_be}, marking unhealthy")
                        attempt_spec.healthy = False
                        attempt_spec.last_error = str(_be)[:200]
                        last_err = _be
                        # Show brief failover notice to user if more backends remain
                        remaining = [s for s in chain if s.healthy and s.id != attempt_spec.id]
                        if remaining:
                            _set_current_text(
                                f"> ⚠️ {attempt_spec.display_name} 不可用，切换至 {remaining[0].display_name}..."
                            )

                if output is None:
                    raise RuntimeError(
                        f"所有 backend 均不可用。最近错误: {last_err}"
                    )

            elapsed = _fmt_elapsed(time.time() - start)
            if not output:
                output = "✅ 完成（无文本输出）"
            log_model = successful_spec.id if successful_spec is not None else model
            log_entry(chat_id, "assistant", output, model=log_model, trace_id=trace_id)

            # Increment turn count and trigger memory auto-update in background
            try:
                _increment_turn_count(chat_id)
                from larkhelm.memory import maybe_auto_update
                maybe_auto_update(chat_id)
            except Exception as _mc_err:
                _debug_log(f"[{trace_id}][DoQuery] post-query memory error: {_mc_err}")

            # Stop the heartbeat and wait for any in-flight patch to complete
            # before writing the final card.  join(timeout) alone is insufficient
            # because Feishu API calls can exceed 2 s; the lock drain is the real
            # guarantee — the join is just cleanup.
            _stop_hb.set()
            with _card_patch_lock:   # blocks until any in-flight heartbeat patch finishes
                pass
            hb_thread.join(timeout=0.5)

            with _tools_lock:
                now_mono = time.monotonic()
                for tid, t in list(active_tools.items()):
                    completed_tools.append({
                        "name": t["name"], "desc": t["desc"],
                        "result": "", "is_error": False,
                        "elapsed": now_mono - t["start"],
                    })
                active_tools.clear()
                n_tools = len(completed_tools)
                final_tools = list(completed_tools)

            note = (f"使用了 {n_tools} 次工具 · " if n_tools else "") + f"耗时 {elapsed}"

            chunks = _split_md(output.strip())
            n = len(chunks)

            # When the response spans multiple cards, keep the first card JSON small
            # (text only, no tools_list) to avoid exceeding Feishu's payload limit.
            # tools_list can be 100 KB+ with many tool calls; that causes both
            # _patch_card_raw and the fallback _send_card_raw to fail silently,
            # leaving the user with only the later numbered cards.
            first_title = f"🤖 {m_name}" + (f" (1/{n})" if n > 1 else "")
            first_card = _make_card(first_title, chunks[0], color="blue",
                                    note=note if n == 1 else "",
                                    tools_list=(final_tools if final_tools else None) if n == 1 else None,
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

        except QueryCancelledError:
            elapsed = _fmt_elapsed(time.time() - start)
            # The card was synchronously updated to "cancelling" in the cancel callback response;
            # here we just patch with the final elapsed time; patch failure is safe (no stale cancel button)
            _patch_card_raw(mid, _make_card("🛑 已取消", f"查询已在 {elapsed} 后取消。", color="orange"))
            if user_msg_id and _eyes_reaction_id[0]:
                delete_reaction(user_msg_id, _eyes_reaction_id[0])
                _eyes_reaction_id[0] = None
        except TimeoutError as e:
            elapsed = _fmt_elapsed(time.time() - start)
            log_entry(chat_id, "error", str(e), model=model, trace_id=trace_id)
            reply_card(chat_id, mid, f"⏰ 强制终止 ({elapsed})",
                       f"任务执行超过 {_cfg.HARD_TIMEOUT // 60} 分钟，进程已被强制终止。\n\n可重新发送继续。",
                       color="red")
            if user_msg_id and _eyes_reaction_id[0]:
                delete_reaction(user_msg_id, _eyes_reaction_id[0])
                react_to_message(user_msg_id, EMOJI_ERROR)
                _eyes_reaction_id[0] = None
        except Exception as e:
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
            _stop_hb.set()
            hb_thread.join(timeout=1.0)

    finally:
        try:
            chat_lock.release()
        except RuntimeError:
            pass
        if not lock_released:
            pending = _pop_pending(chat_id)
            if pending:
                p_msg, p_model, p_user_msg_id, *_ = pending
                _debug_log(f"[Queue] processing queued message: {p_msg[:60]}")
                _reset_cancel(chat_id)
                threading.Thread(
                    target=_do_query,
                    args=(chat_id, p_msg, p_model, p_user_msg_id),
                    daemon=True, name=f"query-{chat_id[:8]}"
                ).start()
