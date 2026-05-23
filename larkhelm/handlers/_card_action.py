"""
larkhelm · card button callbacks

Contains:
  - handle_card_action()   Handle Feishu card button click events
"""
import threading

from lark_oapi.event.callback.model.p2_card_action_trigger import (
    P2CardActionTrigger, P2CardActionTriggerResponse, CallBackToast,
)

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.concurrency import _trigger_cancel, _replace_cancel_event, _pop_pending
from larkhelm.perm import (grant_yolo, _perm_lock, _perm_pending, _perm_decision,
                            _perm_tool_name, _perm_tool_input, _fmt_tool_body)
from larkhelm.card_builder import _make_card_dict


def handle_card_action(event: P2CardActionTrigger) -> P2CardActionTriggerResponse:
    resp = P2CardActionTriggerResponse()
    try:
        ctx    = event.event.context if event.event else None
        action = event.event.action  if event.event else None
        chat_id = ctx.open_chat_id if ctx else None
        cmd     = (action.value or {}).get("cmd", "").strip() if action else ""
        if not chat_id or not cmd:
            return resp
        if _cfg.ALLOWED_CHATS and chat_id not in _cfg.ALLOWED_CHATS:
            return resp
        _debug_log(f"[CardAction] {chat_id}: {cmd}")

        # Permission approval buttons
        if cmd.startswith("perm:"):
            parts = cmd.split(":", 2)
            if len(parts) == 3:
                action_name, tool_use_id = parts[1], parts[2]
                if action_name not in {"allow", "deny", "yolo"}:
                    return resp
                with _perm_lock:
                    tool_name  = _perm_tool_name.get(tool_use_id, "?")
                    tool_input = _perm_tool_input.get(tool_use_id, {})
                    if tool_use_id in _perm_pending:
                        if action_name == "yolo":
                            grant_yolo(chat_id)
                        _perm_decision[tool_use_id] = action_name
                        _perm_pending[tool_use_id].set()
                labels = {"allow": "✅ 已允许", "deny": "❌ 已拒绝", "yolo": "🚀 已允许全部"}
                result_title = labels.get(action_name, "已处理")
                result_color = "green" if action_name != "deny" else "red"
                body_sections = _fmt_tool_body(tool_name, tool_input, max_cmd=400)
                from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
                cb_card = CallBackCard()
                cb_card.type = "raw"
                cb_card.data = _make_card_dict(result_title, body_sections, color=result_color)
                resp.card = cb_card
                toast = CallBackToast()
                toast.type = "success"
                toast.content = result_title
                resp.toast = toast
            return resp

        # Crew human-confirmation breakpoint
        if cmd.startswith("crew_bp:"):
            parts = cmd.split(":", 2)
            if len(parts) == 3:
                bp_action, crew_id = parts[1], parts[2]
                from larkhelm.crew._state import _active_crew, _active_crew_lock
                with _active_crew_lock:
                    owned = _active_crew.get(chat_id) == crew_id
                if not owned:
                    _debug_log(f"[Security] crew_bp rejected: chat={chat_id} does not own crew={crew_id}")
                    return resp
                confirmed = (bp_action == "confirm")
                from larkhelm.crew import signal_breakpoint
                signal_breakpoint(crew_id, confirmed)
                from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
                cb_card = CallBackCard()
                cb_card.type = "raw"
                cb_card.data = _make_card_dict(
                    "✅ 继续执行" if confirmed else "🛑 已取消",
                    "决策已记录，继续执行后续阶段…" if confirmed else "Crew 任务已取消。",
                    color="green" if confirmed else "red",
                )
                resp.card = cb_card
                toast = CallBackToast()
                toast.type = "success"
                toast.content = "✅ 继续执行" if confirmed else "🛑 已取消"
                resp.toast = toast
            return resp

        # Streaming query cancel button
        if cmd.startswith("cancel:"):
            target_chat = cmd.split(":", 1)[1]
            if target_chat == chat_id:
                # First update the crew/dev card immediately (remove buttons), then send the cancel signal
                from larkhelm.crew import immediate_cancel_crew
                immediate_cancel_crew(chat_id)
                _trigger_cancel(chat_id)
                # Synchronously update the card via callback response to remove the cancel button
                # immediately so the user does not see it linger after clicking
                from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
                cb_card = CallBackCard()
                cb_card.type = "raw"
                cb_card.data = _make_card_dict("🛑 取消中", "取消信号已发送，请稍候…", color="orange")
                resp.card = cb_card
                toast = CallBackToast()
                toast.type = "success"
                toast.content = "🛑 取消信号已发送"
                resp.toast = toast
            return resp

        # Queue cancel button
        if cmd.startswith("cancel_queue:"):
            target_chat = cmd.split(":", 1)[1]
            if target_chat == chat_id:
                pending = _pop_pending(chat_id)
                from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
                cb_card = CallBackCard()
                cb_card.type = "raw"
                body_str = f"已取消排队：\n\n> {pending[0][:80]}" if pending else "排队任务已取消。"
                cb_card.data = _make_card_dict("✗ 排队已取消", body_str, color="grey")
                resp.card = cb_card
                toast = CallBackToast()
                toast.type = "success"
                toast.content = "❌ 排队已取消"
                resp.toast = toast
            return resp

        # Plan step confirmation buttons
        # U17: "🗑️ 清除提示" button on the bridge-restart interrupted-plan
        # notification card. Just deletes the persisted state file — no
        # execution restart involved (that's not what U17 promises; we
        # only notify, the user decides whether to re-run by hand).
        if cmd.startswith("plan_persist_clear:"):
            plan_id = cmd.split(":", 1)[1]
            try:
                from larkhelm.plan_persistence import clear_plan_state_button
                existed = clear_plan_state_button(plan_id)
            except Exception as e:
                _debug_log(f"[CardAction] plan_persist_clear failed: {e}")
                existed = False
            from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
            cb_card = CallBackCard()
            cb_card.type = "raw"
            label = "🗑️ 已清除中断提示" if existed else "🗑️ 提示已不存在"
            cb_card.data = _make_card_dict(label, "", color="grey")
            resp.card = cb_card
            toast = CallBackToast()
            toast.type = "success"
            toast.content = label
            resp.toast = toast
            return resp

        if cmd.startswith("plan_continue:") or cmd.startswith("plan_skip:") or cmd.startswith("plan_cancel:") or cmd.startswith("plan_retry:"):
            parts  = cmd.split(":", 1)
            action = parts[0].replace("plan_", "")   # "continue" | "skip" | "cancel" | "retry"
            plan_id = parts[1]
            from larkhelm.cmd_plan import signal_plan
            signal_plan(plan_id, action)
            labels = {"continue": "▶ 继续执行", "skip": "⏭ 已跳过", "cancel": "🛑 已取消", "retry": "🔄 重试中"}
            colors = {"continue": "green", "skip": "grey", "cancel": "red", "retry": "blue"}
            from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
            cb_card = CallBackCard()
            cb_card.type = "raw"
            cb_card.data = _make_card_dict(labels.get(action, action), "", color=colors.get(action, "grey"))
            resp.card = cb_card
            toast = CallBackToast()
            toast.type = "success"
            toast.content = labels.get(action, action)
            resp.toast = toast
            return resp

        # Phase 5: "switch to plain chat" button on the intent disclosure card
        if cmd.startswith("force_chat:"):
            feedback_id = cmd.split(":", 1)[1]
            try:
                from larkhelm.agent_hub import AgentDispatcher, IntentResult, record_feedback
                from larkhelm.agent_hub.intent_feedback import resolve_pending
            except Exception as e:
                _debug_log(f"[CardAction] force_chat import failed: {e}")
                return resp
            pending = resolve_pending(feedback_id)
            if pending is None:
                toast = CallBackToast()
                toast.type = "info"
                toast.content = "ℹ️ 该意图已过期，请重新发送消息"
                resp.toast = toast
                return resp
            try:
                record_feedback(
                    pending.intent, corrected="chat", chat_id=chat_id,
                    feedback_id=feedback_id, text=pending.text,
                )
            except Exception as e:
                _debug_log(f"[CardAction] record_feedback failed: {e}")
            _trigger_cancel(chat_id)
            # Cancel the in-flight intent dispatch (above), then SWAP the
            # per-chat cancel event for a fresh one so the new chat-agent
            # dispatch below doesn't observe the just-set flag and exit
            # immediately. The old task keeps its reference to the now-
            # orphaned set event and shuts down normally; the new task
            # picks up the fresh event via ``_get_cancel_event`` and runs
            # to completion. Without this swap the converted "normal task"
            # short-circuits within ~10s instead of executing.
            _replace_cancel_event(chat_id)
            override_intent = IntentResult(
                agent_type="chat", layer="override",
                is_explicit_command=True, raw_text=pending.text,
            )

            def _override_target():
                try:
                    AgentDispatcher().dispatch(override_intent, pending.ctx)
                except Exception as _ex:
                    _debug_log(f"[CardAction] force_chat dispatch failed: {_ex}")

            threading.Thread(target=_override_target, daemon=True,
                             name=f"force-chat-{chat_id[:8]}").start()
            from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
            cb_card = CallBackCard()
            cb_card.type = "raw"
            cb_card.data = _make_card_dict(
                "💬 已切换为普通对话",
                "已记录此次误判，下一次会更准。", color="grey",
            )
            resp.card = cb_card
            toast = CallBackToast()
            toast.type = "success"
            toast.content = "💬 已切换"
            resp.toast = toast
            return resp

        # Crew pause button
        if cmd.startswith("crew_pause:"):
            crew_id = cmd.split(":", 1)[1]
            from larkhelm.crew import pause_crew
            threading.Thread(target=pause_crew, args=(crew_id,), daemon=True,
                             name=f"pause-{crew_id[:8]}").start()
            toast = CallBackToast()
            toast.type = "success"
            toast.content = "⏸ 暂停信号已发送"
            resp.toast = toast
            return resp

        from larkhelm.commands import _dispatch_button_cmd
        threading.Thread(target=_dispatch_button_cmd, args=(chat_id, cmd), daemon=True).start()
        toast = CallBackToast()
        toast.type = "success"
        toast.content = "✅ 已发送"
        resp.toast = toast
    except Exception as e:
        _debug_log(f"[CardAction] exception: {e}")
    return resp
