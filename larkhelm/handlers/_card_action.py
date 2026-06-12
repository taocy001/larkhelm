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

        # Streaming query cancel button
        if cmd.startswith("cancel:"):
            target_chat = cmd.split(":", 1)[1]
            if target_chat == chat_id:
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

        from larkhelm.commands import _dispatch_button_cmd
        threading.Thread(target=_dispatch_button_cmd, args=(chat_id, cmd), daemon=True).start()
        toast = CallBackToast()
        toast.type = "success"
        toast.content = "✅ 已发送"
        resp.toast = toast
    except Exception as e:
        _debug_log(f"[CardAction] exception: {e}")
    return resp
