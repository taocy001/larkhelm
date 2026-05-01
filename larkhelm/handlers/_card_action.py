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
from larkhelm.concurrency import _trigger_cancel, _pop_pending
from larkhelm.perm import grant_yolo, _perm_lock, _perm_pending, _perm_decision, _perm_tool_name, _perm_tool_input


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
                confirm_elements: list = []
                if tool_name == "Bash":
                    cmd_text = tool_input.get("command", "").strip()
                    confirm_elements.append({"tag": "markdown",
                        "content": f"**命令：**\n```bash\n{cmd_text[:400] or '(空)'}\n```"})
                elif tool_name in ("Write", "Edit", "NotebookEdit"):
                    path = tool_input.get("file_path", tool_input.get("notebook_path", "?"))
                    old = tool_input.get("old_string", "")
                    new = tool_input.get("new_string", "")
                    detail = f"\n\n**修改：** {len(old.splitlines())} 行 → {len(new.splitlines())} 行" if old else ""
                    confirm_elements.append({"tag": "markdown",
                        "content": f"**工具：** `{tool_name}`\n\n**文件：** `{path}`{detail}"})
                else:
                    confirm_elements.append({"tag": "markdown",
                        "content": f"**工具：** `{tool_name}`"})
                from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
                cb_card = CallBackCard()
                cb_card.type = "raw"
                cb_card.data = {
                    "schema": "2.0",
                    "config": {"wide_screen_mode": True},
                    "header": {"template": result_color,
                               "title": {"tag": "plain_text", "content": result_title}},
                    "body": {"elements": confirm_elements},
                }
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
                confirmed = (bp_action == "confirm")
                from larkhelm.crew import signal_breakpoint
                signal_breakpoint(crew_id, confirmed)
                from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackCard
                cb_card = CallBackCard()
                cb_card.type = "raw"
                cb_card.data = {
                    "schema": "2.0",
                    "config": {"wide_screen_mode": True},
                    "header": {
                        "template": "green" if confirmed else "red",
                        "title": {"tag": "plain_text",
                                  "content": "✅ 继续执行" if confirmed else "🛑 已取消"},
                    },
                    "body": {"elements": [{"tag": "markdown",
                        "content": "决策已记录，继续执行后续阶段…" if confirmed else "Crew 任务已取消。",
                    }]},
                }
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
                cb_card.data = {
                    "schema": "2.0",
                    "config": {"wide_screen_mode": True},
                    "header": {"template": "orange",
                               "title": {"tag": "plain_text", "content": "🛑 取消中"}},
                    "body": {"elements": [{"tag": "markdown", "content": "取消信号已发送，请稍候…"}]},
                }
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
                if pending:
                    cb_card.data = {
                        "schema": "2.0",
                        "config": {"wide_screen_mode": True},
                        "header": {"template": "grey",
                                   "title": {"tag": "plain_text", "content": "✗ 排队已取消"}},
                        "body": {"elements": [{"tag": "markdown",
                                               "content": f"已取消排队：\n\n> {pending[0][:80]}"}]},
                    }
                else:
                    cb_card.data = {
                        "schema": "2.0",
                        "config": {"wide_screen_mode": True},
                        "header": {"template": "grey",
                                   "title": {"tag": "plain_text", "content": "✗ 排队已取消"}},
                        "body": {"elements": [{"tag": "markdown", "content": "排队任务已取消。"}]},
                    }
                resp.card = cb_card
                toast = CallBackToast()
                toast.type = "success"
                toast.content = "❌ 排队已取消"
                resp.toast = toast
            return resp

        # Plan step confirmation buttons
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
            cb_card.data = {
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"template": colors.get(action, "grey"),
                           "title": {"tag": "plain_text", "content": labels.get(action, action)}},
                "body": {"elements": []},
            }
            resp.card = cb_card
            toast = CallBackToast()
            toast.type = "success"
            toast.content = labels.get(action, action)
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
