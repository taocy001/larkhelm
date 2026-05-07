"""
larkhelm · message routing

Contains:
  - handle_message()           Handle Feishu text/image/rich-text message events
  - handle_reaction_created()  Handle emoji reaction events
"""
import json
import threading
import traceback

from lark_oapi.api.im.v1 import P2ImMessageReceiveV1, P2ImMessageReactionCreatedV1

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, log_entry
from larkhelm.dedup import _is_duplicate
from larkhelm.chat_state import _get_chat_model, _get_cwd, _is_btw_reply, _register_btw_msg, _set_chat_field
from larkhelm.concurrency import (
    _get_chat_lock, _trigger_cancel, _reset_cancel, _pop_pending,
)
from larkhelm.lark_client import (
    send_card, send_card_reply, _download_image,
    REACTION_ACTIONS, _reply_index, _reply_index_lock,
)
from larkhelm.concurrency import is_shutting_down
from larkhelm.handlers._query import _do_query


def handle_reaction_created(data: P2ImMessageReactionCreatedV1):
    try:
        ev = data.event
        if not ev:
            return
        msg_id     = ev.message_id
        emoji_type = ev.reaction_type.emoji_type if ev.reaction_type else ""
        operator   = ev.operator_type
        if operator == "bot" or not msg_id or not emoji_type:
            return

        _debug_log(f"[Reaction] msg={msg_id} emoji={emoji_type}")

        with _reply_index_lock:
            entry = _reply_index.get(msg_id)
        if not entry:
            return

        chat_id, prompt, model = entry
        if _cfg.ALLOWED_CHATS and chat_id not in _cfg.ALLOWED_CHATS:
            return

        action = REACTION_ACTIONS.get(emoji_type)
        if action == "positive":
            _debug_log(f"[Reaction] user liked chat={chat_id}")
        elif action == "retry":
            if is_shutting_down():
                return
            _debug_log(f"[Reaction] user requested retry chat={chat_id}")
            send_card(chat_id, "🔁 重试", "正在重新执行上一条查询...", color="grey")
            _reset_cancel(chat_id)
            threading.Thread(
                target=_do_query,
                args=(chat_id, prompt, model, None),
                daemon=True,
                name=f"retry-{chat_id[:8]}",
            ).start()
        else:
            _debug_log(f"[Reaction] unmapped emoji {emoji_type}, ignoring")
    except Exception as e:
        _debug_log(f"[Reaction] exception: {e}")


def handle_message(data: P2ImMessageReceiveV1):
    try:
        message = data.event.message
        if _is_duplicate(data.header.event_id, getattr(message, "message_id", "")):
            return
        chat_id = message.chat_id
        if not chat_id:
            return

        # Track the sender's open_id so doc creation can add them as collaborator
        try:
            sender_open_id = data.event.sender.sender_id.open_id if (
                data.event.sender and data.event.sender.sender_id
            ) else ""
            if sender_open_id:
                _set_chat_field(chat_id, "sender_open_id", sender_open_id)
        except Exception:
            pass

        if is_shutting_down():
            send_card(chat_id, '⏳ 服务升级中', '正在重启，请稍后重试。', color='orange')
            return

        if _cfg.ALLOWED_CHATS and chat_id not in _cfg.ALLOWED_CHATS:
            _debug_log(f"[ACL] rejected chat_id={chat_id}")
            return

        # Group chat @mention filter: only respond when the bot itself is mentioned.
        # If BOT_OPEN_ID is unknown (fetch failed at startup), fail-closed and ignore the message
        # rather than responding to all group messages indiscriminately.
        if message.chat_type != "p2p":  # covers "group" and any future non-DM chat types
            import larkhelm.lark_client as _lc_mod
            bot_id = _lc_mod.BOT_OPEN_ID
            if not bot_id:
                _debug_log("[Filter] BOT_OPEN_ID unknown, ignoring group message to avoid broadcast")
                return
            if not message.mentions:
                # No mentions at all in a group message — bot was not addressed
                return
            mentioned_ids = {
                m.id.open_id for m in message.mentions
                if m.id and m.id.open_id
            }
            if bot_id not in mentioned_ids:
                return

        _msg_images: list[str] = []

        if message.message_type == "text":
            try:
                raw = json.loads(message.content).get("text", "").strip()
            except (json.JSONDecodeError, AttributeError):
                return
            if not raw:
                return
            from larkhelm.commands import _strip_at_mention, _cmd_help
            text = _strip_at_mention(raw)
            if not text:
                _cmd_help(chat_id)
                return
        elif message.message_type == "image":
            try:
                img_content = json.loads(message.content)
                image_key = img_content.get("image_key", "")
            except (json.JSONDecodeError, AttributeError):
                image_key = ""
            if not image_key:
                send_card(chat_id, "🖼️ 图片消息", "无法获取图片信息。", color="orange")
                return
            local_path = _download_image(image_key, chat_id, message.message_id)
            if not local_path:
                send_card(chat_id, "🖼️ 图片下载失败", "无法下载图片，请稍后重试。", color="red")
                return
            _msg_images = [local_path]
            text = "请描述/分析这张图片"
        elif message.message_type == "post":
            try:
                raw_post = json.loads(message.content)
                post_content = raw_post.get("content", {})
                if isinstance(post_content, list):
                    title_text = raw_post.get("title", "")
                    paragraphs = post_content
                elif isinstance(post_content, dict):
                    lang_data = next(iter(post_content.values()), {})
                    title_text = lang_data.get("title", "")
                    paragraphs = lang_data.get("content", [])
                else:
                    return
                text_parts = []
                for para in paragraphs:
                    for elem in para:
                        tag = elem.get("tag", "")
                        if tag == "text":
                            t = elem.get("text", "").strip()
                            if t:
                                text_parts.append(t)
                        elif tag == "a":
                            t = elem.get("text", "").strip()
                            href = elem.get("href", "").strip()
                            if t and href:
                                text_parts.append(f"[{t}]({href})")
                            elif t:
                                text_parts.append(t)
                            elif href:
                                text_parts.append(href)
                        elif tag == "code_block":
                            t = elem.get("text", "").strip()
                            if t:
                                lang = elem.get("language", "").lower()
                                text_parts.append(f"```{lang}\n{t}\n```")
                        elif tag in ("img", "media"):
                            ik = elem.get("image_key", "")
                            if ik:
                                p = _download_image(ik, chat_id, message.message_id)
                                if p:
                                    _msg_images.append(p)
                if title_text:
                    text_parts.insert(0, title_text)
                combined = " ".join(text_parts).strip()
                if _msg_images and combined:
                    text = combined
                elif _msg_images:
                    text = "请描述/分析这些图片"
                elif combined:
                    text = combined
                else:
                    return
            except Exception as e:
                _debug_log(f"[Post] parse error: {e}")
                return
        else:
            return

        _debug_log(f"[MSG] {chat_id}: {text[:120]}")
        tl = text.lower().strip()

        # ── Synchronous control commands ──
        _mid = message.message_id

        from larkhelm.commands import (
            _cmd_reset, _cmd_status, _cmd_help, _cmd_pickup, _cmd_history,
            _cmd_stats, _cmd_cron, _cmd_cd, _cmd_pwd, _cmd_ls, _cmd_run,
            _cmd_model, _cmd_lock, _cmd_cli_native, _cmd_btw, _cmd_upgrade, _cmd_memory,
            _strip_at_mention,
        )
        from larkhelm.cmd_doc import _cmd_doc

        if tl == "/reset":
            _cmd_reset(chat_id, None, _mid); return
        if tl == "/reset claude":
            _cmd_reset(chat_id, "claude", _mid); return
        if tl == "/reset gemini":
            _cmd_reset(chat_id, "gemini", _mid); return
        if tl == "/reset kimi":
            _cmd_reset(chat_id, "kimi", _mid); return
        if tl in ("/reset permissions", "/reset perm"):
            _cmd_reset(chat_id, "perm", _mid); return
        if tl == "/reset memory":
            _cmd_reset(chat_id, "memory", _mid); return
        if tl == "/status":
            _cmd_status(chat_id, _mid); return
        if tl == "/help":
            _cmd_help(chat_id, _mid); return
        if tl == "/pickup":
            _cmd_pickup(chat_id, _mid); return
        if tl == "/history":
            _cmd_history(chat_id, msg_id=_mid); return
        if tl == "/history all":
            _cmd_history(chat_id, show_all=True, msg_id=_mid); return
        if tl == "/upgrade":
            _cmd_upgrade(chat_id, _mid); return
        if tl == "/stats":
            _cmd_stats(chat_id, _mid); return
        if tl == "/memory" or tl.startswith("/memory "):
            _cmd_memory(chat_id, text[7:].strip(), _mid); return
        if tl.startswith("/doc"):
            _cmd_doc(chat_id, text[4:].strip()); return
        if tl.startswith("/cron"):
            _cmd_cron(chat_id, text[5:].strip(), _mid); return
        if tl.startswith("/crew"):
            from larkhelm.crew import cmd_crew
            def _crew_target(*a):
                try:
                    cmd_crew(*a)
                except Exception as _e:
                    import traceback as _tb
                    _debug_log(f"[crew] unhandled exception: {_e}\n{_tb.format_exc()}")
            threading.Thread(target=_crew_target, args=(chat_id, text[5:].strip(), message.message_id),
                             daemon=True, name=f"crew-{chat_id[:8]}").start()
            return
        if tl == "/dev" or tl.startswith("/dev "):
            from larkhelm.crew import cmd_dev
            def _dev_target(*a):
                try:
                    cmd_dev(*a)
                except Exception as _e:
                    import traceback as _tb
                    _debug_log(f"[dev] unhandled exception: {_e}\n{_tb.format_exc()}")
            threading.Thread(target=_dev_target, args=(chat_id, text[5:].strip(), message.message_id),
                             daemon=True, name=f"dev-{chat_id[:8]}").start()
            return
        if tl.startswith("/plan"):
            from larkhelm.cmd_plan import cmd_plan
            def _plan_target(*a):
                try:
                    cmd_plan(*a)
                except Exception as _e:
                    import traceback as _tb
                    _debug_log(f"[plan] unhandled exception: {_e}\n{_tb.format_exc()}")
            threading.Thread(target=_plan_target, args=(chat_id, text[5:].strip(), message.message_id),
                             daemon=True, name=f"plan-{chat_id[:8]}").start()
            return
        if tl == "/cancel":
            chat_lock = _get_chat_lock(chat_id)
            is_running = not chat_lock.acquire(blocking=False)
            if not is_running:
                chat_lock.release()
            pending = _pop_pending(chat_id)
            _trigger_cancel(chat_id)
            if is_running:
                body = "已向当前任务发送中断信号。"
                if pending:
                    body += f"\n排队消息「{pending[0][:40]}」已取消。"
            else:
                body = "当前没有正在执行的任务。"
                if pending:
                    body += f"\n排队消息「{pending[0][:40]}」已取消。"
            send_card_reply(chat_id, _mid, "🛑 已取消", body, color="orange")
            return
        if tl == "/pwd":
            _cmd_pwd(chat_id, _mid); return
        if tl.startswith("/cd "):
            _cmd_cd(chat_id, text[4:].strip(), _mid); return
        if tl.startswith("/ls"):
            _cmd_ls(chat_id, text[3:].strip(), _mid); return
        if tl.startswith("/run "):
            threading.Thread(target=_cmd_run, args=(chat_id, text[5:].strip(), _mid),
                             daemon=True).start()
            return
        if tl == "/model" or tl.startswith("/model "):
            parts = text.split(None, 1)
            arg = parts[1].strip() if len(parts) == 2 else ""
            _cmd_lock(chat_id, arg, _mid)
            return
        if tl == "/lock" or tl.startswith("/lock "):
            parts = text.split(None, 1)
            arg = parts[1].strip() if len(parts) == 2 else ""
            _cmd_lock(chat_id, arg, _mid)
            return
        if tl.startswith("/rename "):
            import re as _re
            name = text[8:].strip()
            name = _re.sub(r'[`*_~]', '', name)[:30]
            if name:
                from larkhelm.chat_state import _set_chat_field as _scf
                _scf(chat_id, "name", name)
                send_card_reply(chat_id, _mid, "✅ 会话已命名", f"当前会话名称：**{name}**", color="green")
            else:
                from larkhelm.chat_state import _set_chat_field as _scf
                _scf(chat_id, "name", "")
                send_card_reply(chat_id, _mid, "✅ 会话名称已清除", "", color="green")
            return
        if tl.startswith("/btw ") or tl == "/btw":
            question = text[5:].strip() if tl.startswith("/btw ") else ""
            if not question:
                send_card_reply(chat_id, _mid, "⚠️ 用法", "`/btw <问题>` — 快速追问（不占用主锁）", color="orange")
                return
            _cmd_btw(chat_id, question, message.message_id)
            return
        if _is_btw_reply(chat_id, getattr(message, "parent_id", None)):
            _register_btw_msg(chat_id, message.message_id)
            _cmd_btw(chat_id, text, message.message_id)
            return

        # ── Model dispatch ──
        target_model = _get_chat_model(chat_id)
        prompt = text
        if text.startswith(("/g ", "/gemini ")):
            target_model = "gemini"
            prompt = text.split(" ", 1)[1].strip()
        elif text.startswith(("/c ", "/claude ")):
            target_model = "claude"
            prompt = text.split(" ", 1)[1].strip()
        elif text.startswith(("/k ", "/kimi ")):
            target_model = "kimi"
            prompt = text.split(" ", 1)[1].strip()

        # Vision routing: gemini CLI doesn't support image input; force to a vision-capable model
        if _msg_images and target_model not in ("claude", "kimi"):
            target_model = "claude"

        if not prompt:
            send_card_reply(chat_id, _mid, "⚠️ 空消息", "消息内容不能为空。", color="orange")
            return

        if prompt.startswith("/"):
            _cmd_cli_native(chat_id, target_model, prompt, _mid)
            return

        # Context injection: attempt in priority order
        parent_id = getattr(message, "parent_id", None)

        # Priority 1: user replied to a crew task card → inject crew summary
        crew_ctx = None
        if parent_id:
            from larkhelm.crew import get_crew_card_context
            crew_ctx = get_crew_card_context(parent_id)

        if crew_ctx:
            _debug_log(f"[MSG] injecting crew context '{crew_ctx['title'][:20]}' → {chat_id[:12]}")
            prompt = (
                f"[以下是刚完成的 Crew 任务「{crew_ctx['title']}」的交付结论，"
                f"请结合它来回答我的问题]\n\n"
                f"{crew_ctx['summary']}\n\n"
                f"---\n\n"
                f"{prompt}"
            )
        elif parent_id:
            # Priority 2: reply to any message (AI reply/user message/notification card) → fetch and inject original text
            from larkhelm.lark_client import _fetch_parent_message_text
            parent_text = _fetch_parent_message_text(parent_id)
            if parent_text:
                _debug_log(f"[MSG] injecting parent message context ({len(parent_text)} chars) → {chat_id[:12]}")
                prompt = (
                    f"[用户回复了以下消息]\n\n"
                    f"{parent_text}\n\n"
                    f"---\n\n"
                    f"{prompt}"
                )
            else:
                # Priority 3: parent fetch failed → fall back to sticky crew context
                from larkhelm.crew import get_recent_crew_context
                crew_ctx = get_recent_crew_context(chat_id)
                if crew_ctx:
                    _debug_log(f"[MSG] injecting sticky crew context '{crew_ctx['title'][:20]}' → {chat_id[:12]}")
                    prompt = (
                        f"[以下是刚完成的 Crew 任务「{crew_ctx['title']}」的交付结论，"
                        f"请结合它来回答我的问题]\n\n"
                        f"{crew_ctx['summary']}\n\n"
                        f"---\n\n"
                        f"{prompt}"
                    )
        else:
            # Priority 3 (no parent): fall back to sticky crew context
            from larkhelm.crew import get_recent_crew_context
            crew_ctx = get_recent_crew_context(chat_id)
            if crew_ctx:
                _debug_log(f"[MSG] injecting sticky crew context '{crew_ctx['title'][:20]}' → {chat_id[:12]}")
                prompt = (
                    f"[以下是刚完成的 Crew 任务「{crew_ctx['title']}」的交付结论，"
                    f"请结合它来回答我的问题]\n\n"
                    f"{crew_ctx['summary']}\n\n"
                    f"---\n\n"
                    f"{prompt}"
                )

        # Workspace context: if .crew_workspace/ has relevant files, tell the AI so it
        # can read them naturally (enables "fix failed /dev" or "revise /plan" via chat).
        try:
            from pathlib import Path as _Path
            _ws = _Path(_get_cwd(chat_id)) / ".crew_workspace"
            _ws_files = sorted(
                f.name for f in _ws.iterdir()
                if f.is_file() and f.suffix in (".md", ".json") and f.name != "crew_checkpoint.json"
            ) if _ws.is_dir() else []
            if _ws_files:
                prompt = (
                    f"[工作区] .crew_workspace/ 下有以下文件可供参考：{', '.join(_ws_files)}。"
                    f"如需了解当前任务背景，请读取这些文件。\n\n"
                    f"{prompt}"
                )
        except Exception:
            pass

        log_entry(chat_id, "user", prompt, model=target_model)
        _reset_cancel(chat_id)
        user_msg_id = message.message_id
        # Doc injection is done inside _do_query (background thread) to avoid blocking
        # the SDK event dispatch loop on slow Feishu API calls.
        threading.Thread(
            target=_do_query,
            kwargs={
                "chat_id": chat_id, "message": prompt, "model": target_model,
                "user_msg_id": user_msg_id,
                "images": _msg_images if _msg_images else None,
            },
            daemon=True,
            name=f"query-{chat_id[:8]}",
        ).start()

    except Exception as e:
        _debug_log(f"[HandleMsg] exception: {e}\n{traceback.format_exc()}")
