"""larkhelm · QuerySession (P1-1 PR2).

This module is the **opt-in** rewrite target for ``_do_query``. When
``config['query_session_v2_enabled']`` is True, the entry function
``_do_query`` delegates to :meth:`QuerySession.run`; otherwise the
938-line legacy path is unchanged.

The session is a dataclass carrying the per-query state that used to
live in 15+ closures (``mid``, ``start``, ``m_name``, ``cancel_ev``,
``card_state``, …). Behaviour matches the legacy code; the rewrite is
structural so the diff-able PR is small.

Default flag: ``false``. Tests pin the public contract via
``test_query_session.py``.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, log_entry
from larkhelm.card_builder import _make_card, _fmt_elapsed
from larkhelm.concurrency import (
    _get_chat_lock, _get_cancel_event,
    _replace_cancel_event, _set_pending, _pop_pending, _update_pending_card_mid,
    _reset_cancel,
)
from larkhelm.lark_client import (
    send_card, reply_card,
    _send_card_raw, _patch_card_raw, _reply_card_raw,
    _pin_task_card, react_to_message, delete_reaction,
    EMOJI_PROCESSING, EMOJI_DONE, EMOJI_ERROR,
    _index_reply,
)
from larkhelm.chat_state import _get_cwd
from larkhelm.ai_runner import QueryCancelledError
from larkhelm.handlers._query_card_state import (
    QueryCardState, record_query_start, record_query_end,
)
from larkhelm.handlers import _query_pure


@dataclass
class QuerySession:
    """Encapsulates one ``_do_query`` invocation (P1-1 PR2).

    Mutable fields are populated by ``run()`` as state evolves; tests
    construct a session and call individual helpers in isolation.
    """

    chat_id: str
    message: str
    model: str
    user_msg_id: "str | None" = None
    images: "list | None" = None
    parent_id: "str | None" = None
    force_backend_id: "str | None" = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Mutable state populated during run()
    mid: "str | None" = None
    start_time: float = 0.0
    m_name: str = ""
    cancel_ev: "threading.Event | None" = None
    chat_lock: "threading.Lock | None" = None
    card_state: "QueryCardState | None" = None
    hb_thread: "threading.Thread | None" = None
    stop_hb: threading.Event = field(default_factory=threading.Event)
    lock_released: bool = False
    eyes_reaction_id: "str | None" = None
    successful_spec: object = None
    output: "str | None" = None

    # ─────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.chat_lock = _get_chat_lock(self.chat_id)
        self.cancel_ev = _get_cancel_event(self.chat_id)

        # Pre-flight: queue behind crew if one is running
        if self._maybe_queue_behind_crew():
            return

        if not self.chat_lock.acquire(blocking=False):
            self._queue_behind_running_query()
            return

        if self.user_msg_id:
            self.eyes_reaction_id = react_to_message(self.user_msg_id, EMOJI_PROCESSING)

        record_query_start()
        try:
            cwd = _get_cwd(self.chat_id)
            self.start_time = time.time()
            self._resolve_initial_model()
            self.mid = self.emit_init_card(cwd)

            self.card_state = QueryCardState(
                chat_id=self.chat_id,
                model_name=self.m_name,
                start_time=self.start_time,
            )
            self.start_heartbeat()

            try:
                self._inject_parent_context()
                has_doc_urls = bool(_query_pure.extract_feishu_urls(self.message))
                enriched, memory_ctx, deduped = _query_pure.inject_doc_and_memory(
                    self.message, self.chat_id, cwd,
                    doc_auto_inject=_cfg.DOC_AUTO_INJECT,
                    has_doc_urls=has_doc_urls,
                )
                self.message = enriched
                recent_turns = "\n".join(deduped)

                self.output = self.run_failover_loop(
                    cwd, memory_ctx, recent_turns, has_doc_urls,
                )
                if self.output is None:
                    raise RuntimeError("所有 backend 均不可用。")
                self.finalise(self.output)
            except QueryCancelledError:
                self.on_cancel()
            except TimeoutError as e:
                self.on_timeout(e)
            except Exception as e:
                self.on_error(e)
            finally:
                self.stop_hb.set()
                if self.hb_thread is not None:
                    self.hb_thread.join(timeout=1.0)
        finally:
            elapsed_total = max(0.0, time.time() - (self.start_time or time.time()))
            record_query_end(elapsed_total)
            self._release_lock_safe()
            if not self.lock_released:
                self._drain_pending_queue()
            _query_pure.cleanup_temp_images(self.images)

    # ─────────────────────────────────────────────────────────────────
    # Stage helpers
    # ─────────────────────────────────────────────────────────────────

    def _resolve_initial_model(self) -> None:
        self.m_name = {
            "claude": "Claude", "gemini": "Gemini", "kimi": "Kimi", "deepseek": "DeepSeek",
        }.get(self.model, self.model.capitalize())
        try:
            from larkhelm.router import resolve_backend as _resolve
            from larkhelm.backend_registry import BACKEND_REGISTRY as _registry
            has_docs = bool(_query_pure.extract_feishu_urls(self.message))
            spec = _resolve(self.chat_id, self.message, bool(self.images), has_docs,
                            self.force_backend_id)
            self.m_name = spec.display_name
            del _registry  # silence linter; only used to mirror legacy import order
        except Exception:
            pass

    def emit_init_card(self, cwd: str) -> "str | None":
        card_json = _query_pure.build_init_card(self.m_name, cwd, self.chat_id)
        if self.user_msg_id:
            mid = _reply_card_raw(self.user_msg_id, card_json, in_thread=False)
        else:
            mid = _send_card_raw(self.chat_id, card_json)
        if mid:
            _pin_task_card(self.chat_id, mid)
        return mid

    def start_heartbeat(self) -> "threading.Thread | None":
        if self.card_state is None:
            return None

        from larkhelm.handlers._query import CARD_PUSH_INTERVAL, CURSOR_INTERVAL

        def _push_if_needed(force: bool, include_cancel: bool) -> None:
            if self.cancel_ev is None or self.cancel_ev.is_set() or self.stop_hb.is_set():
                return
            assert self.card_state is not None
            rendered = self.card_state.render_body()
            need_push, combined = self.card_state.should_push(rendered, force=force)
            if not need_push:
                return
            btns = [("🛑 取消", f"cancel:{self.chat_id}")] if include_cancel else None
            card_json = _make_card(
                rendered.title, rendered.response_md, color="grey",
                tools_md=rendered.tools_md, tools_expanded=True,
                buttons=btns,
            )
            with self.card_state.card_patch_lock:
                if self.cancel_ev.is_set() or self.stop_hb.is_set():
                    return
                _patch_card_raw(self.mid, card_json)
            self.card_state.mark_pushed(combined)

        def _loop():
            assert self.card_state is not None
            while not self.stop_hb.is_set():
                try:
                    self.card_state.tick_cursor()
                    now = time.monotonic()
                    last_hb, in_bg, dirty_now = self.card_state.get_heartbeat_snapshot()
                    show_cancel = not in_bg
                    if now - last_hb >= CARD_PUSH_INTERVAL:
                        _push_if_needed(True, show_cancel)
                        self.card_state.update_heartbeat()
                    elif dirty_now:
                        _push_if_needed(False, show_cancel)
                except Exception as e:
                    _debug_log(f"[QuerySession] heartbeat exception: {e}")
                self.stop_hb.wait(timeout=CURSOR_INTERVAL)

        t = threading.Thread(target=_loop, daemon=True,
                             name=f"hb-{self.chat_id[:8]}")
        t.start()
        self.hb_thread = t
        return t

    def run_failover_loop(
        self,
        cwd: str,
        memory_ctx: str,
        recent_turns: str,
        has_doc_urls: bool,
    ) -> "str | None":
        from larkhelm.router import resolve_backend, LockedBackendUnavailableError
        from larkhelm.backend_registry import BACKEND_REGISTRY
        from larkhelm.handlers._query import (
            _run_backend_single, _do_query_with_delegation,
        )

        assert self.card_state is not None

        # Resolve primary
        primary_spec = None
        try:
            primary_spec = resolve_backend(
                self.chat_id, self.message, bool(self.images),
                has_doc_urls, self.force_backend_id,
            )
            self.m_name = primary_spec.display_name
            self.card_state.update_model_name(self.m_name)
        except LockedBackendUnavailableError as e:
            self.stop_hb.set()
            with self.card_state.card_patch_lock:
                pass
            if self.hb_thread is not None:
                self.hb_thread.join(timeout=0.5)
            if self.user_msg_id and self.eyes_reaction_id:
                delete_reaction(self.user_msg_id, self.eyes_reaction_id)
                self.eyes_reaction_id = None
            send_card(self.chat_id, "❌ 锁定后端不可用",
                      f"{e}\n\n使用 **/lock off** 恢复自动路由。", color="red")
            return None
        except Exception as e:
            _debug_log(f"[{self.trace_id}][QuerySession] routing error: {e}")
            primary_spec = None

        force_direct = bool(
            self.force_backend_id and primary_spec is not None
            and getattr(primary_spec, "healthy", False)
        )
        chain = _query_pure.build_failover_chain(primary_spec, BACKEND_REGISTRY, force_direct)

        # Define callbacks for backend runners
        cs = self.card_state

        def on_tool(name: str, desc: str, tool_id: str = ""):
            cs.on_tool(name, desc, tool_id)
            log_entry(self.chat_id, "tool", f"{name}: {desc}",
                      model=self.model, trace_id=self.trace_id)

        on_tool_result = cs.on_tool_result
        on_text = cs.on_text

        def _on_soft_timeout() -> None:
            self.on_soft_timeout()

        if not chain:
            return self._run_legacy_fallback(
                cwd, memory_ctx, on_text, on_tool, on_tool_result, _on_soft_timeout,
            )

        worker_specs = {} if force_direct else {
            s.id: s for s in BACKEND_REGISTRY.all_enabled()
            if s.healthy and s.role != "orchestrator"
        }
        output: "str | None" = None
        last_err: "Exception | None" = None
        for attempt_spec in chain:
            try:
                self.m_name = attempt_spec.display_name
                cs.update_model_name(self.m_name)
                if worker_specs:
                    output = _do_query_with_delegation(
                        self.chat_id, self.message, attempt_spec, worker_specs,
                        cwd, self.cancel_ev, on_text, on_tool, on_tool_result,
                        _on_soft_timeout, images=self.images, memory_ctx=memory_ctx,
                        recent_turns=recent_turns,
                    )
                else:
                    output = _run_backend_single(
                        attempt_spec, self.chat_id, self.message, cwd, self.cancel_ev,
                        on_text, on_tool, on_tool_result, _on_soft_timeout, self.images,
                        extra_system=memory_ctx, recent_turns=recent_turns,
                    )
                self.successful_spec = attempt_spec
                break
            except QueryCancelledError:
                raise
            except TimeoutError:
                raise
            except Exception as e:
                _debug_log(
                    f"[{self.trace_id}][QuerySession] backend {attempt_spec.id} failed: {e}"
                )
                attempt_spec.healthy = False
                attempt_spec.last_error = str(e)[:200]
                last_err = e
                remaining = [s for s in chain if s.healthy and s.id != attempt_spec.id]
                if remaining:
                    cs.set_current_text(
                        f"> ⚠️ {attempt_spec.display_name} 不可用，"
                        f"切换至 {remaining[0].display_name}..."
                    )

        if output is None:
            raise RuntimeError(f"所有 backend 均不可用。最近错误: {last_err}")
        return output

    def _run_legacy_fallback(self, cwd, memory_ctx, on_text, on_tool,
                              on_tool_result, on_soft_timeout) -> str:
        from larkhelm.chat_state import _load_sid
        runner = _query_pure.select_legacy_runner(self.model)
        spec_id = (
            "gemini" if self.model == "gemini" else
            "kimi"   if self.model == "kimi"   else
            "deepseek" if self.model == "deepseek" else "claude"
        )
        sid = _load_sid(self.chat_id, spec_id)
        msg = (
            f"[System]\n{memory_ctx}\n\n[User Query]\n{self.message}"
            if memory_ctx and not sid else self.message
        )
        kwargs = dict(
            on_tool=on_tool, on_text=on_text,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
        )
        if self.model in ("kimi", "claude"):
            kwargs["images"] = self.images
        return runner(self.chat_id, msg, cwd, self.cancel_ev, **kwargs)

    # ─────────────────────────────────────────────────────────────────
    # Terminal handlers
    # ─────────────────────────────────────────────────────────────────

    def finalise(self, output: str) -> None:
        assert self.card_state is not None
        elapsed = _fmt_elapsed(time.time() - self.start_time)
        if not output:
            output = "✅ 完成（无文本输出）"
        log_model = (
            self.successful_spec.id
            if self.successful_spec is not None else self.model
        )
        log_entry(self.chat_id, "assistant", output, model=log_model,
                  trace_id=self.trace_id)

        try:
            from larkhelm.handlers._query import _post_query_memory_hook
            _post_query_memory_hook(self.chat_id, self.trace_id)
        except Exception as e:
            _debug_log(f"[QuerySession] post-query memory hook error: {e}")

        self.stop_hb.set()
        with self.card_state.card_patch_lock:
            pass
        if self.hb_thread is not None:
            self.hb_thread.join(timeout=0.5)

        self.card_state.snapshot_active_tools_as_completed()
        final_tool_records = self.card_state.snapshot_completed_tools()
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

        chunks, note, tools_payload = _query_pure.format_completion_card(
            self.m_name, output, n_tools, elapsed, final_tools,
            max_card_len=_cfg.MAX_CARD_LEN,
        )
        n = len(chunks)
        first_title = f"🤖 {self.m_name}" + (f" (1/{n})" if n > 1 else "")
        first_payload = tools_payload if n == 1 else None
        first_card = _make_card(
            first_title, chunks[0] if chunks else "", color="blue",
            note=note if n == 1 else "",
            tools_list=first_payload,
            tools_expanded=False,
        )
        if not _patch_card_raw(self.mid, first_card):
            _debug_log(
                f"[{self.trace_id}][QuerySession] final patch failed, falling back to send"
            )
            fallback_text = f"[🤖 {self.m_name}]\n{(chunks[0] if chunks else '')[:500]}"
            self.mid = _send_card_raw(self.chat_id, first_card, _fallback_text=fallback_text)

        for i, chunk in enumerate(chunks[1:], 2):
            is_last = (i == n)
            chunk_title = f"🤖 {self.m_name} ({i}/{n})"
            if is_last:
                last_card = _make_card(
                    chunk_title, chunk, color="blue",
                    note=note,
                    tools_list=final_tools if final_tools else None,
                    tools_expanded=False,
                )
                fallback_text = f"[{chunk_title}]\n{chunk[:500]}"
                _send_card_raw(self.chat_id, last_card, _fallback_text=fallback_text)
            else:
                send_card(self.chat_id, chunk_title, chunk, color="blue")

        if self.user_msg_id and self.eyes_reaction_id:
            delete_reaction(self.user_msg_id, self.eyes_reaction_id)
            self.eyes_reaction_id = None
            react_to_message(self.user_msg_id, EMOJI_DONE)

        if self.user_msg_id and (time.time() - self.start_time) >= 60:
            _reply_card_raw(
                self.user_msg_id,
                _make_card(f"✅ {self.m_name} 完成", f"耗时 {elapsed}", color="green"),
                in_thread=False,
            )

        if self.mid:
            _index_reply(self.mid, self.chat_id, self.message, self.model)

    def on_cancel(self) -> None:
        assert self.card_state is not None
        self.stop_hb.set()
        with self.card_state.card_patch_lock:
            pass
        elapsed = _fmt_elapsed(time.time() - self.start_time)
        _patch_card_raw(
            self.mid,
            _make_card("🛑 已取消", f"查询已在 {elapsed} 后取消。", color="orange"),
        )
        if self.user_msg_id and self.eyes_reaction_id:
            delete_reaction(self.user_msg_id, self.eyes_reaction_id)
            self.eyes_reaction_id = None

    def on_timeout(self, err: Exception) -> None:
        assert self.card_state is not None
        self.stop_hb.set()
        with self.card_state.card_patch_lock:
            pass
        elapsed = _fmt_elapsed(time.time() - self.start_time)
        log_entry(self.chat_id, "error", str(err), model=self.model,
                  trace_id=self.trace_id)
        err_msg = str(err).strip() or "未知超时"
        reply_card(
            self.chat_id, self.mid,
            f"⏰ 强制终止 ({elapsed})",
            f"任务被超时机制终止：\n\n`{err_msg}`\n\n实际耗时：{elapsed}。可重新发送继续。\n"
            f"如确认任务正常运行被误杀，请增大 `hard_timeout` 配置 "
            f"(当前: {_cfg.HARD_TIMEOUT // 60} 分钟，作用：子进程**无输出**超过该时长才会被判定卡死)。",
            color="red",
        )
        if self.user_msg_id and self.eyes_reaction_id:
            delete_reaction(self.user_msg_id, self.eyes_reaction_id)
            react_to_message(self.user_msg_id, EMOJI_ERROR)
            self.eyes_reaction_id = None

    def on_error(self, err: Exception) -> None:
        assert self.card_state is not None
        self.stop_hb.set()
        with self.card_state.card_patch_lock:
            pass
        import sys
        print(traceback.format_exc(), file=sys.stderr)
        _debug_log(
            f"[{self.trace_id}][QuerySession] exception: {err}\n{traceback.format_exc()}"
        )
        log_entry(self.chat_id, "error", str(err), model=self.model,
                  trace_id=self.trace_id)
        lines = [ln for ln in str(err).splitlines() if ln.strip()]
        last = lines[-1] if lines else str(err)
        reply_card(self.chat_id, self.mid, "❌ 错误", last, color="red",
                   note="执行 /status 查看详情")
        if self.user_msg_id and self.eyes_reaction_id:
            delete_reaction(self.user_msg_id, self.eyes_reaction_id)
            react_to_message(self.user_msg_id, EMOJI_ERROR)
            self.eyes_reaction_id = None

    def on_soft_timeout(self) -> None:
        if self.lock_released:
            return
        elapsed_now = _fmt_elapsed(time.time() - self.start_time)
        _debug_log(
            f"[{self.trace_id}][QuerySession] soft timeout ({elapsed_now}), lock released"
        )
        if self.card_state is not None:
            self.card_state.set_in_background(True)
        self._release_lock_safe()
        self.lock_released = True
        _replace_cancel_event(self.chat_id)
        pending = _pop_pending(self.chat_id)
        if pending:
            p_msg, p_model, p_user_msg_id, *_ = pending
            _debug_log(
                f"[Queue/QuerySession-SoftTimeout] processing queued message: {p_msg[:60]}"
            )
            threading.Thread(
                target=_re_dispatch_query,
                args=(self.chat_id, p_msg, p_model, p_user_msg_id),
                daemon=True, name=f"query-{self.chat_id[:8]}",
            ).start()

    # ─────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────

    def _maybe_queue_behind_crew(self) -> bool:
        try:
            from larkhelm.crew._state import is_crew_running, subscribe_crew_done
            if not is_crew_running(self.chat_id):
                return False
        except Exception as e:
            _debug_log(f"[QuerySession] crew check error: {e}")
            return False

        existing_mid = _set_pending(self.chat_id, self.message, self.model, self.user_msg_id)
        preview = self.message[:80].replace("\n", " ")
        card = _make_card(
            "⏳ Crew 运行中",
            f"当前 Crew 任务完成后自动执行：\n\n> {preview}",
            color="orange",
            buttons=[("❌ 取消排队", f"cancel_queue:{self.chat_id}")],
        )
        if existing_mid:
            _patch_card_raw(existing_mid, card)
        else:
            if self.user_msg_id:
                mid = _reply_card_raw(self.user_msg_id, card, in_thread=False)
            else:
                mid = _send_card_raw(self.chat_id, card)
            _update_pending_card_mid(self.chat_id, mid)

        try:
            from larkhelm.crew._state import subscribe_crew_done as _sub
            done_ev = _sub(self.chat_id)
        except Exception as e:
            _debug_log(f"[QuerySession] subscribe_crew_done error: {e}")
            return True

        def _after(_ev=done_ev, cid=self.chat_id):
            _ev.wait(timeout=4 * 3600)
            pending = _pop_pending(cid)
            if pending:
                p_msg, p_model, p_user_msg_id, *_ = pending
                _reset_cancel(cid)
                threading.Thread(
                    target=_re_dispatch_query,
                    args=(cid, p_msg, p_model, p_user_msg_id),
                    daemon=True, name=f"query-{cid[:8]}",
                ).start()

        threading.Thread(target=_after, daemon=True,
                         name=f"crew-wait-{self.chat_id[:8]}").start()
        return True

    def _queue_behind_running_query(self) -> None:
        existing_mid = _set_pending(self.chat_id, self.message, self.model, self.user_msg_id)
        preview = self.message[:80].replace("\n", " ")
        card = _make_card(
            "⏳ 排队中",
            f"将在当前任务完成后自动执行：\n\n> {preview}",
            color="orange",
            buttons=[("❌ 取消排队", f"cancel_queue:{self.chat_id}")],
        )
        if existing_mid:
            _patch_card_raw(existing_mid, card)
        else:
            if self.user_msg_id:
                mid = _reply_card_raw(self.user_msg_id, card, in_thread=False)
            else:
                mid = _send_card_raw(self.chat_id, card)
            _update_pending_card_mid(self.chat_id, mid)

    def _inject_parent_context(self) -> None:
        if not self.parent_id:
            return
        try:
            from larkhelm.lark_client import _fetch_parent_message_text
            parent_text = _fetch_parent_message_text(self.parent_id)
            if parent_text:
                self.message = (
                    f"[用户回复了以下消息]\n\n{parent_text}\n\n---\n\n{self.message}"
                )
                _debug_log(
                    f"[{self.trace_id}][QuerySession] injected parent context ({len(parent_text)} chars)"
                )
            else:
                from larkhelm.crew import get_recent_crew_context
                crew_ctx = get_recent_crew_context(self.chat_id)
                if crew_ctx:
                    self.message = (
                        f"[以下是刚完成的 Crew 任务「{crew_ctx['title']}」的交付结论，"
                        f"请结合它来回答我的问题]\n\n{crew_ctx['summary']}\n\n"
                        f"---\n\n{self.message}"
                    )
        except Exception as e:
            _debug_log(f"[{self.trace_id}][QuerySession] parent fetch error: {e}")

    def _release_lock_safe(self) -> None:
        if self.chat_lock is None:
            return
        try:
            self.chat_lock.release()
        except RuntimeError:
            pass

    def _drain_pending_queue(self) -> None:
        pending = _pop_pending(self.chat_id)
        if not pending:
            return
        p_msg, p_model, p_user_msg_id, *_ = pending
        _debug_log(f"[Queue/QuerySession] processing queued message: {p_msg[:60]}")
        _reset_cancel(self.chat_id)
        threading.Thread(
            target=_re_dispatch_query,
            args=(self.chat_id, p_msg, p_model, p_user_msg_id),
            daemon=True, name=f"query-{self.chat_id[:8]}",
        ).start()


def _re_dispatch_query(chat_id, message, model, user_msg_id):
    """Re-enter the main _do_query so the v2 flag is honoured again."""
    from larkhelm.handlers._query import _do_query
    _do_query(chat_id, message, model, user_msg_id)


__all__ = ["QuerySession"]
