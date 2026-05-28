"""
larkhelm · message routing

Contains:
  - handle_message()           Handle Feishu text/image/rich-text message events
  - handle_reaction_created()  Handle emoji reaction events
"""
import json
import os
import re as _re
import threading
import time
import traceback
import uuid

from lark_oapi.api.im.v1 import P2ImMessageReceiveV1, P2ImMessageReactionCreatedV1

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, log_entry
from larkhelm.dedup import _is_duplicate
# P2 REQ-03: pure-fn helpers (no Feishu SDK dependency, unit-testable).
# Imported but only used at one branch below — handle_message is too
# entangled with SDK objects to convert wholesale, so the helpers are
# kept on the side and reused only where it doesn't change behaviour.
from larkhelm._message_pure import (
    extract_allowed_chat_decision as _pure_allow,
    parse_doc_urls as _pure_parse_doc_urls,
)
from larkhelm.chat_state import (
    _get_chat_model, _get_cwd, _is_btw_reply, _register_btw_msg, _set_chat_field,
    _get_voice_lang, _get_chat_state, _pop_chat_field,
)
from larkhelm.concurrency import (
    _get_chat_lock, _trigger_cancel, _reset_cancel, _pop_pending,
    _get_cancel_event,
)
from larkhelm.lark_client import (
    send_card, send_card_reply, _download_image, _download_message_file, update_card,
    REACTION_ACTIONS, _reply_index, _reply_index_lock,
    _make_card, _patch_card_raw, download_file_by_key,
)
from larkhelm.concurrency import is_shutting_down
from larkhelm.handlers._query import _do_query
from larkhelm.voice.transcribe import transcribe as transcribe_file
from larkhelm.voice.merge import add_voice


def _thread_error_card(chat_id: str, label: str, exc: Exception) -> None:
    """Log full traceback and notify the user with a red error card.

    Used by ``/plan`` / ``/crew`` / ``/dev`` daemon thread wrappers — those
    threads previously only wrote to ``_debug_log`` on uncaught exceptions, so
    the user saw the task silently disappear. The user-visible body truncates
    ``str(exc)`` to 200 chars; the full ``traceback.format_exc()`` still lands
    in the debug log under ``[<label>]`` for diagnosis.
    """
    try:
        _debug_log(
            f"[{label}] unhandled exception: {exc}\n{traceback.format_exc()}"
        )
    except Exception:
        pass
    try:
        send_card(
            chat_id, "❌ 任务失败",
            f"{label} 任务失败：{str(exc)[:200]}",
            color="red",
        )
    except Exception as _send_err:
        try:
            _debug_log(f"[{label}] error-card send failed: {_send_err}")
        except Exception:
            pass


def _intent_router_active(chat_id: str) -> bool:
    """Return True iff the phase-5 intent router should run for this chat.

    Delegates to the shared ``_gating.hash_traffic_active`` helper so that
    ``intent_router_traffic`` and ``memory_retriever_traffic`` bucket the
    same chat_id identically (AC-05 / NFR-DEPLOY-1). The helper short-
    circuits on flag=False without importing agent_hub so AC-10 still
    holds.
    """
    from larkhelm._gating import hash_traffic_active
    return hash_traffic_active(
        chat_id, "intent_router_enabled", "intent_router_traffic",
    )


# P3 REQ-02 / design.md §3.3: case-insensitive keyword set that gates the
# workspace-hint segment when WORKSPACE_HINT_KEYWORD_GATE=True. Compiled at
# import time (regex is fixed; satisfies NFR §4.1 "< 0.5 ms / message").
# P2a: expanded with code-task keywords so the gate also fires for code
# editing / fixing / refactoring prompts (avoids injecting workspace context
# into casual chat while still covering all actionable work requests).
_WORKSPACE_KEYWORD_RE = _re.compile(
    r"(workspace|计划|任务|设计|prd|design|tasks|review|qa|crew"
    r"|code|edit|implement|fix|refactor|debug|写代码|改代码|修复|重构)",
    _re.IGNORECASE,
)
_CREW_STICKY_KW_RE = _re.compile(
    r"(crew|/dev|/plan|agent|任务|流水线|pipeline|checkpoint)",
    _re.IGNORECASE,
)


def _apply_crew_sticky_context(chat_id: str, text: str, prompt: str) -> str:
    """Consume and optionally inject the most recent sticky crew summary.

    ``consume_recent_crew_context`` is **always** called so the injection
    counter advances and the TTL / max-injections eviction logic in
    ``crew/_state.py`` fires correctly — regardless of whether the keyword
    gate decides to skip the actual text injection.
    """
    gate_on = bool(_cfg.config.get("crew_sticky_keyword_gate_enabled"))
    kw_match = (not gate_on) or bool(_CREW_STICKY_KW_RE.search(text or ""))
    from larkhelm.crew import consume_recent_crew_context
    crew_ctx = consume_recent_crew_context(chat_id)
    if not crew_ctx:
        return prompt
    if gate_on and not kw_match:
        try:
            from larkhelm.metrics import inc_injection_gate as _inc
            _inc("crew_sticky", "skipped")
        except Exception:
            pass
        return prompt
    try:
        from larkhelm.metrics import inc_injection_gate as _inc
        _inc("crew_sticky", "injected")
    except Exception:
        pass
    from larkhelm.log import _debug_log as _dl
    _dl(f"[MSG] injecting sticky crew context '{crew_ctx['title'][:20]}' → {chat_id[:12]}")
    return (
        f"[以下是刚完成的 Crew 任务「{crew_ctx['title']}」的交付结论，"
        f"请结合它来回答我的问题]\n\n"
        f"{crew_ctx['summary']}\n\n"
        f"---\n\n"
        f"{prompt}"
    )


def _build_workspace_hint(chat_id: str, user_text: str) -> tuple[str, str]:
    """Return ``(injection_prefix, outcome)`` for the workspace-hint segment.

    The prefix is the literal text to prepend (incl. trailing ``\\n\\n``)
    or ``""`` when the caller must NOT touch the prompt. ``outcome`` is
    the metric label (always non-empty for telemetry parity).

    Outcomes (cf. design.md §3.4):
      * ``injected_passive``  — files exist; gate off OR keyword matched
      * ``skipped_by_gate``   — files exist; gate on AND keyword missed
      * ``skipped_empty``     — no ``.crew_workspace/`` dir or no files

    Wraps every filesystem touch in a broad ``except OSError`` so a
    permission flake on the ``.crew_workspace/`` directory degrades to
    "no hint" rather than tossing the user message back with a 500.
    """
    try:
        from pathlib import Path as _Path
        ws = _Path(_get_cwd(chat_id)) / ".crew_workspace"
        if not ws.is_dir():
            return "", "skipped_empty"
        files = sorted(
            f.name for f in ws.iterdir()
            if f.is_file() and f.suffix in (".md", ".json")
            and f.name != "crew_checkpoint.json"
        )
    except OSError:
        return "", "skipped_empty"

    if not files:
        return "", "skipped_empty"

    gate_on = bool(getattr(_cfg, "WORKSPACE_HINT_KEYWORD_GATE", False))
    if gate_on and _WORKSPACE_KEYWORD_RE.search(user_text or "") is None:
        return "", "skipped_by_gate"

    prefix = (
        f"[工作区] .crew_workspace/ 下存在以下文件：{', '.join(files)}。"
        f"如果与本次问题相关，再读取；否则忽略。\n\n"
    )
    return prefix, "injected_passive"


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
        sender_open_id = ""
        try:
            sender_open_id = data.event.sender.sender_id.open_id if (
                data.event.sender and data.event.sender.sender_id
            ) else ""
            if sender_open_id:
                _set_chat_field(chat_id, "sender_open_id", sender_open_id)
        except Exception as e:
            _debug_log(f"[message] sender_open_id save failed: {e}")

        if is_shutting_down():
            send_card(chat_id, '⏳ 服务升级中', '正在重启，请稍后重试。', color='orange')
            return

        # P2 REQ-03: ACL decision via the pure helper so the same logic is
        # unit-testable without spinning up a fake Feishu event object.
        _allow = _pure_allow(chat_id, _cfg.ALLOWED_CHATS, sender_open_id)
        if not _allow.allowed:
            _debug_log(f"[ACL] rejected chat_id={chat_id} reason={_allow.reason}")
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
                send_card(chat_id, "⚠️ 消息解析失败",
                          "富文本消息解析出错，请尝试以纯文本发送。", color="orange")
                return
        elif message.message_type == "file":
            # Parse file metadata common to all file branches.
            try:
                file_meta = json.loads(message.content)
                file_key = file_meta.get("file_key", "")
                file_name = file_meta.get("file_name", "")
            except (json.JSONDecodeError, AttributeError):
                return
            if not file_key:
                return

            # ── Branch 1: memory import (zip uploaded after /memory import) ──
            # _pop_chat_field atomically claims the pending slot; prevents two
            # concurrent file uploads from both entering the import path.
            _pending_ts = _pop_chat_field(chat_id, "pending_memory_import")
            _PENDING_TTL = 600
            if _pending_ts:
                # Accept both timestamp (new) and boolean True (legacy); expire after 10 minutes
                if isinstance(_pending_ts, float) and time.time() - _pending_ts > _PENDING_TTL:
                    send_card_reply(chat_id, message.message_id, "⏰ 等待超时",
                                    "导入等待已过期（10 分钟），请重新执行 `/memory import`。",
                                    color="orange")
                    return
                # Only accept .zip files; restore flag so the user can retry with a valid file.
                if not file_name.lower().endswith(".zip"):
                    _set_chat_field(chat_id, "pending_memory_import", _pending_ts)
                    send_card_reply(chat_id, message.message_id, "⚠️ 格式错误",
                                    "请发送 `.zip` 格式的导出文件。", color="orange")
                    return
                # Flag already cleared by _pop_chat_field; proceed with import.
                placeholder = send_card_reply(chat_id, message.message_id, "📥 导入中",
                                              "正在下载并导入记忆数据…", color="grey")
                try:
                    import tempfile
                    from pathlib import Path
                    from larkhelm.memory_io import import_memory
                    _fd, _tmp = tempfile.mkstemp(suffix=".zip", prefix=f"larkhelm_import_{chat_id[:8]}_")
                    os.close(_fd)
                    tmp_path = Path(_tmp)
                    ok = download_file_by_key(file_key, tmp_path)
                    if not ok:
                        if placeholder:
                            _patch_card_raw(placeholder, _make_card("❌ 下载失败",
                                                                     "无法下载文件，请确认权限。",
                                                                     color="red"))
                        return
                    report = import_memory(tmp_path)
                    n_written = len(report["written"])
                    n_skipped = len(report["skipped"])
                    lines = [f"**导入成功：** {n_written} 个文件"]
                    if n_skipped:
                        lines.append(f"**跳过：** {n_skipped} 个文件")
                    if report.get("warnings"):
                        lines.append(f"**警告：** {'；'.join(report['warnings'])}")
                    color = "green" if not n_skipped else "orange"
                    body = "\n\n".join(lines)
                    if placeholder:
                        _patch_card_raw(placeholder, _make_card("✅ 导入完成", body, color=color))
                    else:
                        send_card_reply(chat_id, message.message_id, "✅ 导入完成", body, color=color)
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                except Exception as e:
                    _debug_log(f"[memory import file] failed: {e}")
                    if placeholder:
                        _patch_card_raw(placeholder, _make_card("❌ 导入失败", str(e)[:300], color="red"))
                    else:
                        send_card_reply(chat_id, message.message_id, "❌ 导入失败", str(e)[:300], color="red")
                return

            # ── Branch 2: general file analysis (FILE_ENABLED gate) ──
            if not getattr(_cfg, "FILE_ENABLED", True):
                send_card(chat_id, "⚠️ 文件处理未启用",
                          "当前配置未启用文件处理功能，无法解析此文件。", color="orange")
                return

            from larkhelm.file_handler import process_file as _process_file
            _file_result = _process_file(file_key, file_name, chat_id, message.message_id)

            if _file_result.warnings and not _file_result.has_content:
                # Format rejected or all files failed — show warning and stop.
                warn_body = "\n\n".join(_file_result.warnings)
                send_card_reply(chat_id, message.message_id, "⚠️ 无法处理文件",
                                warn_body, color="orange")
                return

            if not _file_result.has_content:
                send_card(chat_id, "⚠️ 文件无法处理",
                          "文件内容为空或格式不受支持。", color="orange")
                return

            if _file_result.warnings:
                warn_body = "\n\n".join(_file_result.warnings)
                send_card_reply(chat_id, message.message_id, "⚠️ 部分文件处理失败",
                                warn_body, color="orange")

            _target_model = _get_chat_model(chat_id)
            _file_text = f"[文件: {file_name}]"
            _trace_id = uuid.uuid4().hex[:12]
            log_entry(chat_id, "user", _file_text, model=_target_model, trace_id=_trace_id)
            _reset_cancel(chat_id)
            try:
                from larkhelm.memory import _query_sender_open_id
                _query_sender_open_id.set(sender_open_id or "")
            except Exception:
                pass

            # Route through AgentDispatcher when intent router is active so
            # FileAgent benefits from ACL, audit, and reswitch hooks.
            _dispatched_via_agent = False
            if _intent_router_active(chat_id):
                try:
                    from larkhelm.agent_hub import AgentContext, AgentDispatcher
                    from larkhelm.agent_hub.intent_types import IntentResult
                    _file_intent = IntentResult(
                        agent_type="file",
                        sub_intent="file_analysis",
                        confidence=1.0,
                        is_explicit_command=False,
                        layer="override",
                    )
                    _file_ctx = AgentContext(
                        chat_id=chat_id,
                        user_msg_id=message.message_id,
                        text=_file_text,
                        images=None,
                        parent_id=getattr(message, "parent_id", None),
                        cancel_ev=_get_cancel_event(chat_id),
                        cwd=_get_cwd(chat_id),
                        files=_file_result.files,
                    )
                    threading.Thread(
                        target=AgentDispatcher().dispatch,
                        args=(_file_intent, _file_ctx),
                        daemon=True,
                        name=f"file-agent-{chat_id[:8]}",
                    ).start()
                    _dispatched_via_agent = True
                except Exception as _fae:
                    _debug_log(f"[FileAgent] AgentDispatcher route failed, falling back: {_fae}")

            if not _dispatched_via_agent:
                threading.Thread(
                    target=_do_query,
                    kwargs={
                        "chat_id": chat_id,
                        "message": _file_text,
                        "model": _target_model,
                        "user_msg_id": message.message_id,
                        "files": _file_result.files,
                        "trace_id": _trace_id,
                        "sender_open_id": sender_open_id,
                    },
                    daemon=True,
                    name=f"file-query-{chat_id[:8]}",
                ).start()
            return

        elif message.message_type == "audio":
            if not _cfg.VOICE_ENABLED:
                return
            try:
                audio_meta = json.loads(message.content)
                file_key = audio_meta.get("file_key", "")
                duration_ms = int(audio_meta.get("duration", 0) or 0)
            except Exception as e:
                _debug_log(f"[Voice] content parse failed: {e}")
                send_card_reply(chat_id, message.message_id, "⚠️ 音频解析失败",
                                "无法解析音频元数据。", color="orange")
                return
            if not file_key:
                _debug_log("[Voice] content missing file_key")
                send_card_reply(chat_id, message.message_id, "⚠️ 音频解析失败",
                                "音频消息缺少 file_key。", color="orange")
                return
            if duration_ms > _cfg.VOICE_MAX_DURATION_MS:
                _debug_log(
                    f"[Voice] rejected oversize duration={duration_ms}ms "
                    f"(limit={_cfg.VOICE_MAX_DURATION_MS}ms)"
                )
                send_card_reply(
                    chat_id, message.message_id, "⚠️ 音频过长",
                    f"时长 {duration_ms / 1000:.1f}s，超过上限 "
                    f"{_cfg.VOICE_MAX_DURATION_MS / 1000:.1f}s。",
                    color="orange",
                )
                return
            placeholder_mid = send_card_reply(
                chat_id, message.message_id, "🎤 转写中…",
                "正在识别语音…", color="grey",
            )
            audio_path = None
            try:
                audio_path = _download_message_file(
                    file_key, chat_id, message.message_id, kind="file",
                )
                if not audio_path:
                    _debug_log(f"[Voice] download failed file_key={file_key}")
                    update_card(placeholder_mid, "❌ 下载失败",
                                "无法下载音频文件，请稍后重试。", color="red")
                    return
                lang = _get_voice_lang(chat_id)
                result = transcribe_file(audio_path, lang=lang)
                text_out = (result.get("text") or "").strip()
                if not result.get("ok") or not text_out:
                    err_tag = result.get("error") or "空文本"
                    _debug_log(
                        f"[Voice] transcribe failed: ok={result.get('ok')} "
                        f"error={err_tag}"
                    )
                    # Map known engine-specific error tags to actionable hints
                    # so the user doesn't have to grep logs to recover.
                    hint_map = {
                        "dashscope_no_api_key":
                            "DashScope 未配置：在 systemd drop-in 注入 "
                            "`Environment=\"DASHSCOPE_API_KEY=sk-...\"` 后重启 bridge。",
                        "dashscope_empty_result":
                            "DashScope 返回空文本，可能是音频太短或口齿不清。",
                    }
                    user_hint = hint_map.get(err_tag, "")
                    if not user_hint:
                        if err_tag.startswith("dashscope_sdk_missing"):
                            user_hint = ("DashScope SDK 未安装，在 SSH 跑："
                                         "`pipx runpip larkhelm install dashscope`。")
                        elif err_tag.startswith("dashscope_status_4"):
                            user_hint = "DashScope 鉴权或参数错误，检查 API Key + 模型名。"
                        elif err_tag.startswith("dashscope_call_failed"):
                            user_hint = "DashScope 网络调用失败，检查网络或重试。"
                        elif err_tag in ("disabled", "load_failed"):
                            user_hint = ("本地模型加载失败，跑 `larkhelm voice probe` "
                                         "诊断；或切到 dashscope 引擎。")
                    body = f"未能从音频识别出文字（`{err_tag}`）。"
                    if user_hint:
                        body += f"\n\n{user_hint}"
                    update_card(placeholder_mid, "⚠️ 转写失败", body, color="orange")
                    return
                update_card(placeholder_mid, "✅ 转写完成", text_out, color="green")
                target_model = _get_chat_model(chat_id)
                parent_id = getattr(message, "parent_id", None)
                _debug_log(
                    f"[Voice] transcribed {len(text_out)}ch lang={lang} → "
                    f"dispatch model={target_model}"
                )
                # Mirror the text path's pre-dispatch hygiene (line 595-596):
                # log into the conversation .md so memory/auto-update sees the
                # voice-originated turn, and clear any stale /cancel event so
                # _do_query doesn't abort immediately on its first read.
                # Same trace_id wiring as the text path so duration
                # pairing in /stats works for voice-originated turns.
                voice_trace_id = uuid.uuid4().hex[:12]
                log_entry(chat_id, "user", text_out, model=target_model,
                          trace_id=voice_trace_id)
                _reset_cancel(chat_id)
                if _cfg.VOICE_MERGE_WINDOW_SEC > 0:
                    # Pass voice_trace_id so the eventual merged
                    # dispatch (only the HEAD fragment's trace_id
                    # survives the merge — see voice/merge.py
                    # ``_flush_locked``) lines up with the head's
                    # user log entry. Stats round-3 MUST-FIX.
                    add_voice(
                        chat_id, text_out, target_model,
                        user_msg_id=message.message_id, parent_id=parent_id,
                        trace_id=voice_trace_id,
                    )
                else:
                    # Off-thread the LLM call so the SDK event worker is not
                    # held for the full query duration (matches text path
                    # at line 600-611).
                    threading.Thread(
                        target=_do_query,
                        kwargs={
                            "chat_id": chat_id, "message": text_out,
                            "model": target_model,
                            "user_msg_id": message.message_id,
                            "parent_id": parent_id,
                            "trace_id": voice_trace_id,
                            "sender_open_id": sender_open_id,
                        },
                        daemon=True,
                        name=f"voice-query-{chat_id[:8]}",
                    ).start()
            finally:
                if (not _cfg.VOICE_KEEP_AUDIO) and audio_path:
                    try:
                        os.unlink(audio_path)
                    except Exception as e:
                        _debug_log(f"[Voice] cleanup unlink failed: {e}")
            return
        else:
            return

        _debug_log(f"[MSG] {chat_id}: {text[:120]}")
        tl = text.lower().strip()

        # ── Synchronous control commands ──
        _mid = message.message_id

        from larkhelm.commands import (
            _cmd_btw, _cmd_cli_native, _strip_at_mention,
        )

        # ── /cancel — touches per-chat lock + cancel-event; not registry-able ──
        if tl == "/cancel":
            chat_lock = _get_chat_lock(chat_id)
            is_running = not chat_lock.acquire(blocking=False)
            if not is_running:
                chat_lock.release()
            pending = _pop_pending(chat_id)
            _trigger_cancel(chat_id)
            # Phase D follow-up: a /cancel that lands inside the configured
            # window after an AgentDispatcher dispatch is strong evidence
            # the routing was wrong. We `consume_dispatch` (not `peek`) so
            # the same /cancel can't be billed against the same dispatch
            # twice. Skip when the prior intent was already ``chat`` —
            # chat cancellations are mostly "I changed my mind", not a
            # routing error.
            try:
                from larkhelm.agent_hub.intent_feedback import (
                    consume_dispatch as _consume_disp,
                    record_signal as _rec_signal,
                )
                from larkhelm.config import INTENT_FEEDBACK_CANCEL_WINDOW_SEC as _CANCEL_W
                _hit = _consume_disp(chat_id, max_age_sec=_CANCEL_W)
                if _hit is not None:
                    _prior_intent, _prior_text, _age = _hit
                    if _prior_intent.agent_type != "chat":
                        _rec_signal(
                            "cancel_after_dispatch", _prior_intent, chat_id,
                            corrected="chat", text=_prior_text,
                            metadata={"elapsed_sec": round(_age, 2),
                                      "was_running": bool(is_running)},
                        )
            except Exception as _ce:
                _debug_log(f"[IntentFeedback] cancel signal failed: {_ce}")
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

        # ── /memory diagnose [N] (Phase D / Phase 2 REQ-38) ──
        # Handled here (ahead of the registry) so the prefix-match on /memory
        # in command_registry doesn't double-dispatch. The body of the handler
        # lives in larkhelm.commands so test_memory_diagnose_cmd can import it
        # directly without spinning up the message router.
        if tl.startswith("/memory diagnose"):
            from larkhelm.commands import _cmd_memory_diagnose
            _cmd_memory_diagnose(
                chat_id, text[len("/memory diagnose"):].strip(), msg_id=_mid,
            )
            return

        # ── Registry-driven dispatch (S1+S7) ──
        # Covers /reset, /status, /help, /pickup, /upgrade, /history, /stats,
        # /memory, /cron, /crew, /dev, /plan, /pwd, /cd, /ls, /run,
        # /model (+ /lock alias), /voice. See command_registry._default_registrations.
        from larkhelm.command_registry import COMMAND_REGISTRY, DispatchContext
        _dctx = DispatchContext(chat_id=chat_id, msg_id=_mid, text=text, tl=tl,
                                sender_open_id=sender_open_id)
        if COMMAND_REGISTRY.dispatch(_dctx) == "handled":
            return

        # ── /rename — chat_state mutator; not registry-able ──
        if tl.startswith("/rename "):
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
            _cmd_btw(chat_id, question, message.message_id, sender_open_id=sender_open_id)
            return
        if _is_btw_reply(chat_id, getattr(message, "parent_id", None)):
            _register_btw_msg(chat_id, message.message_id)
            _cmd_btw(chat_id, text, message.message_id, sender_open_id=sender_open_id)
            return

        # ── Phase 5: intent router (gated by flag + traffic %) ──
        # Explicit slash commands above already returned early; only "free-form"
        # text reaches this point. Keep all imports lazy so flag=false (or unset)
        # means agent_hub is never imported (AC-10).
        if not text.startswith("/") and _intent_router_active(chat_id):
            try:
                from larkhelm.agent_hub import (
                    AgentContext, AgentDispatcher, resolve_intent,
                )
            except Exception as _ex:
                _debug_log(f"[IntentRouter] import failed: {_ex}")
            else:
                try:
                    has_doc_urls = ("feishu.cn/docx/" in text or "feishu.cn/wiki/" in text or "feishu.cn/sheets/" in text)
                    intent = resolve_intent(
                        text, _msg_images or None, has_doc_urls, chat_id,
                    )
                except Exception as _ex:
                    _debug_log(f"[IntentRouter] resolve_intent failed: {_ex}")
                    intent = None
                # Phase D: stash the resolved IntentResult on this chat so
                # the chat-agent fall-through (which exits this block without
                # calling AgentDispatcher) can pick it up inside _do_query and
                # forward to get_memory_context_v2(intent=...).
                # Restrict to agent_type == "chat" so dev/crew/plan/doc
                # intents (which go straight to AgentDispatcher and never
                # reach _do_query) don't leak into the *next* chat turn —
                # crew/dev paths synthesize their own IntentResult anyway.
                if intent is not None and intent.agent_type == "chat":
                    try:
                        from larkhelm.chat_state import _set_pending_intent
                        _set_pending_intent(chat_id, intent)
                    except Exception as _ex:
                        _debug_log(f"[IntentRouter] _set_pending_intent failed: {_ex}")
                if intent is not None and intent.agent_type != "chat":
                    # Mirror the legacy path's _reset_cancel: a stale chat-level
                    # cancel from the previous /cancel must be cleared, otherwise
                    # cmd_dev / cmd_crew / _do_query inside the dispatched Agent
                    # see is_set()=True and abort immediately. PlanAgent self-resets,
                    # the other three do not.
                    _reset_cancel(chat_id)
                    cancel_ev = _get_cancel_event(chat_id)
                    cwd       = _get_cwd(chat_id)
                    parent_id = getattr(message, "parent_id", None)
                    ctx = AgentContext(
                        chat_id=chat_id, user_msg_id=message.message_id,
                        text=text, images=_msg_images or None,
                        parent_id=parent_id, cancel_ev=cancel_ev, cwd=cwd,
                    )
                    threading.Thread(
                        target=AgentDispatcher().dispatch,
                        args=(intent, ctx), daemon=True,
                        name=f"agent-{intent.agent_type}-{chat_id[:8]}",
                    ).start()
                    return

        # ── Model dispatch ──
        target_model = _get_chat_model(chat_id)
        prompt = text
        force_backend_id: str | None = None
        if text.startswith(("/g ", "/gemini ")):
            target_model = "gemini"
            prompt = text.split(" ", 1)[1].strip()
            force_backend_id = "gemini"
        elif text.startswith(("/c ", "/claude ")):
            target_model = "claude"
            prompt = text.split(" ", 1)[1].strip()
            force_backend_id = "claude"
        elif text.startswith(("/k ", "/kimi ")):
            target_model = "kimi"
            prompt = text.split(" ", 1)[1].strip()
            force_backend_id = "kimi"
        elif text.startswith(("/d ", "/deepseek ")):
            target_model = "deepseek"
            prompt = text.split(" ", 1)[1].strip()
            force_backend_id = "deepseek"

        # Phase D follow-up: backend-override slash commands (/c /g /k /d)
        # bypass AgentDispatcher entirely, so the reswitch hook inside
        # the dispatcher can never see them. Detect here: if a prior
        # non-chat agent dispatch sits inside the reswitch window, the
        # user manually picking a backend means "should've been chat all
        # along". The plain backend prefix doesn't tell us a NEW agent
        # type, so we record corrected="chat" (the only legitimate
        # answer for backend-forcing slash commands).
        if force_backend_id is not None:
            try:
                from larkhelm.agent_hub.intent_feedback import (
                    consume_dispatch as _consume_disp,
                    record_signal as _rec_signal,
                )
                _hit = _consume_disp(chat_id, max_age_sec=120.0)
                if _hit is not None:
                    _prior, _prior_text, _age = _hit
                    if _prior.agent_type != "chat":
                        _rec_signal(
                            "agent_reswitch", _prior, chat_id,
                            corrected="chat", text=_prior_text,
                            metadata={
                                "elapsed_sec": round(_age, 2),
                                "new_agent": "chat",
                                "via": f"backend_override:{force_backend_id}",
                            },
                        )
            except Exception as _re:
                _debug_log(f"[IntentFeedback] reswitch (backend override) failed: {_re}")

        # Vision routing: gemini CLI and DeepSeek HTTP don't support image input;
        # force a vision-capable model.
        if _msg_images and target_model not in ("claude", "kimi"):
            target_model = "claude"

        if not prompt:
            send_card_reply(chat_id, _mid, "⚠️ 空消息", "消息内容不能为空。", color="orange")
            return

        if prompt.startswith("/"):
            _cmd_cli_native(chat_id, target_model, prompt, _mid)
            return

        # Bare-URL guard: a lone unrecognised Feishu URL gets an orange usage
        # card instead of being shipped to the AI (which would then waste a
        # turn explaining it can't read random URLs).
        from larkhelm.handlers._query import _maybe_doc_usage_hint
        if _maybe_doc_usage_hint(text, chat_id, _mid):
            return

        # Context injection: attempt in priority order.
        # _fetch_parent_message_text is a Feishu API call; it is NOT done here
        # (SDK event thread) to avoid blocking all event dispatch.  Instead,
        # parent_id is passed to _do_query which runs in a background thread.
        parent_id = getattr(message, "parent_id", None)

        # Priority 1: user replied to a crew task card → inject crew summary (local, no I/O)
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
            parent_id = None  # already handled; no need to fetch parent in _do_query
        elif not parent_id:
            # No parent at all → try sticky crew context (local, no I/O).
            # Delegated to _apply_crew_sticky_context which always calls
            # consume_recent_crew_context so the injection counter advances.
            prompt = _apply_crew_sticky_context(chat_id, text, prompt)
        # else: parent_id set and no crew card found → _do_query will fetch parent text in background

        # Workspace context: P3 REQ-01 passive phrasing + REQ-02 optional
        # keyword gate. _build_workspace_hint encodes the scanning + gate
        # logic so it stays unit-testable; one outcome metric per call
        # (cf. design.md §6.1 invariant).
        try:
            _ws_prefix, _ws_outcome = _build_workspace_hint(chat_id, text)
            if _ws_prefix:
                prompt = _ws_prefix + prompt
            try:
                from larkhelm.metrics import inc_workspace_hint as _inc_ws
                _inc_ws(_ws_outcome)
            except Exception as _metric_err:
                _debug_log(f"[Message] inc_workspace_hint failed: {_metric_err}")
        except Exception as e:
            _debug_log(f"[Message] workspace hint build failed: {e}")

        # Generate ``trace_id`` HERE — before the user log entry — so the
        # user side carries the same id ``_do_query`` will later use on
        # the assistant log entry. This is what powers ``_cmd_stats``
        # duration pairing under concurrent /btw or rapid resends; the
        # pre-fix FIFO pending_ts scrambled pairs when entries interleaved.
        # ``log_entry`` already accepts ``trace_id`` as a kwarg.
        trace_id = uuid.uuid4().hex[:12]
        log_entry(chat_id, "user", prompt, model=target_model, trace_id=trace_id)
        _reset_cancel(chat_id)
        user_msg_id = message.message_id
        # MEM-C1: propagate sender_open_id into the child thread via ContextVar so
        # group-chat queries never read a neighbour's global memory file.
        # Python threads inherit the parent's Context snapshot at start() time.
        try:
            from larkhelm.memory import _query_sender_open_id
            _query_sender_open_id.set(sender_open_id or "")
        except Exception:
            pass
        # Doc injection and parent message fetch are done inside _do_query (background
        # thread) to avoid blocking the SDK event dispatch loop on Feishu API calls.
        threading.Thread(
            target=_do_query,
            kwargs={
                "chat_id": chat_id, "message": prompt, "model": target_model,
                "user_msg_id": user_msg_id,
                "images": _msg_images if _msg_images else None,
                "parent_id": parent_id,
                "force_backend_id": force_backend_id,
                "trace_id": trace_id,
                "sender_open_id": sender_open_id,
            },
            daemon=True,
            name=f"query-{chat_id[:8]}",
        ).start()

    except Exception as e:
        _debug_log(f"[HandleMsg] exception: {e}\n{traceback.format_exc()}")
