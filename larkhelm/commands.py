"""
larkhelm · command implementations

Contains all _cmd_* functions, _dispatch_button_cmd(), and helper utilities.
"""
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, log_entry, _read_logs, warn
from larkhelm.chat_state import (
    _get_cwd, _set_chat_field, _get_chat_state, _get_chat_model,
    _load_sid, _clear_sid, _register_btw_msg,
    set_pending_doc_write, pop_pending_doc_write,
    _get_effort, _set_effort,
    _get_lang, _set_lang,
)
from larkhelm.locale import _t
from larkhelm.concurrency import (
    _get_chat_lock, _trigger_cancel, _pop_pending,
    _cron_lock, _get_btw_lock, _reset_cancel,
)
from larkhelm.token_stats import (
    get_token_stats, get_token_stats_persistent,
    summarize_crew_agent_tokens_for_chat,
    summarize_crew_agent_tokens_by_type,
    estimate_cache_savings,
    get_cache_savings_summary,
)
from larkhelm.claude_session_guard import (
    get_session_counters, clear_session_counters,
)
from larkhelm.perm import revoke_yolo, is_yolo
from larkhelm.doc_handlers import _cmd_doc_write_do  # backs the doc_write_confirm card button (DocAgent path)
from larkhelm.lark_client import (
    send_card, send_card_reply, reply_card, _send_card_raw, _patch_card_raw, _reply_card_raw,
    react_to_message, delete_reaction,
    EMOJI_PROCESSING, EMOJI_DONE, EMOJI_ERROR,
    send_permission_guide,
    upload_file_to_feishu, send_file_message, download_file_by_key,
)
from larkhelm.card_builder import _make_card, _fmt_elapsed


# ═══════════════════════════════════════════════════
#  /help renderer (P1-2b/P1-2c — data-driven)
# ═══════════════════════════════════════════════════

# Static narrative blocks. Indexed by key from ``_HELP_LAYOUT``.
# Splitting these out (rather than hardcoding inside _cmd_help) lets
# tests monkey-patch a single section, and lets future plugins inject
# / swap a section without touching the render function.
# Each value is a (zh, en) tuple.
_HELP_STATIC_SECTIONS: dict[str, tuple[str, str]] = {
    "intro": (
        "发消息直接提问，命令均以 `/` 开头",
        "Just send a message to ask anything. Commands start with `/`",
    ),
    "decision_tree": (
        (
            "**🧭 任务怎么选**\n"
            "💬 直接发消息 — 单轮问答 / 闲聊 / 代码片段解释\n"
            "🛠 **/dev** <需求> — 软件工程流水线（PM→架构→工程→QA→审查），产物通常一次 commit\n"
            "🤖 **/crew** <需求> — 动态规划，Manager 自动分解任务多 Agent 并行\n"
            "📋 **/plan** — 多阶段串行：`[dev]` `[review]` `[fix]` `[test]`，每步确认；支持飞书文档 URL\n"
            "💡 不确定时 → 直接发消息让 AI 判断"
        ),
        (
            "**🧭 Which command to use**\n"
            "💬 Just message — Q&A / chat / code snippets\n"
            "🛠 **/dev** <task> — software pipeline (PM→Arch→Eng→QA→Review), usually one commit\n"
            "🤖 **/crew** <task> — dynamic planning, Manager decomposes and runs agents in parallel\n"
            "📋 **/plan** — staged pipeline: `[dev]` `[review]` `[fix]` `[test]`, confirm each step; supports Feishu doc URLs\n"
            "💡 Unsure? → just message and let AI decide"
        ),
    ),
    "separator": ("---", "---"),
    "model_shortcuts": (
        (
            "**🎯 与指定模型对话**\n"
            "**/c <消息>** — Claude\n"
            "**/g <消息>** — Gemini\n"
            "**/k <消息>** — Kimi\n"
            "**/d <消息>** — DeepSeek"
        ),
        (
            "**🎯 Talk to a specific model**\n"
            "**/c <message>** — Claude\n"
            "**/g <message>** — Gemini\n"
            "**/k <message>** — Kimi\n"
            "**/d <message>** — DeepSeek"
        ),
    ),
    "reset_detail": (
        (
            "**♻️ 重置细分**\n"
            "**/reset** claude | gemini | kimi | deepseek — 单独重置某 backend 会话\n"
            "**/reset perm** — 权限审批\n"
            "**/reset memory** — 清除会话记忆（全局/项目保留）"
        ),
        (
            "**♻️ Reset options**\n"
            "**/reset** claude | gemini | kimi | deepseek — reset a specific backend session\n"
            "**/reset perm** — permission approvals\n"
            "**/reset memory** — clear session memory (global/project kept)"
        ),
    ),
    "memory_detail": (
        (
            "**🧠 记忆系统**（每 10 轮自动从对话中提取，无需手工维护）\n"
            "**/memory** — 查看三层（全局/项目/会话）\n"
            "**/memory observe** — 容量与健康度\n"
            "**/memory set** global | project <内容> — 手动覆盖偏好/项目记忆\n"
            "**/memory update** — 立即触发摘要 + 抽取\n"
            "**/memory clear** session | project | global | all — 清除指定记忆层\n"
            "**/memory list** — 列出记忆文件\n"
            "**/memory gc [天数] [apply]** — 清理过期记忆\n"
            "**/memory export** — 导出记忆 zip\n"
            "**/memory import [file_key]** — 导入记忆 zip"
        ),
        (
            "**🧠 Memory system** (auto-extracts from conversation every 10 turns)\n"
            "**/memory** — view three layers (global/project/session)\n"
            "**/memory observe** — capacity & health\n"
            "**/memory set** global | project <content> — manually set preference/project memory\n"
            "**/memory update** — trigger summary + extraction now\n"
            "**/memory clear** session | project | global | all — clear a memory layer\n"
            "**/memory list** — list memory files\n"
            "**/memory gc [days] [apply]** — clean up expired memory\n"
            "**/memory export** — export memory zip\n"
            "**/memory import [file_key]** — import memory zip"
        ),
    ),
    "doc_section": (
        (
            "**📄 飞书文档**\n"
            "**读** — 在消息里粘贴 `docx` / `wiki` / `sheets` URL，bridge 自动读取并注入上下文\n"
            "**写** — 终端跑 `larkhelm doc create|append|write`，或在飞书里用 `/run larkhelm doc create \"标题\"`"
        ),
        (
            "**📄 Feishu Docs**\n"
            "**Read** — paste a `docx` / `wiki` / `sheets` URL in a message; bridge auto-reads and injects context\n"
            "**Write** — run `larkhelm doc create|append|write` in terminal, or use `/run larkhelm doc create \"Title\"` in Feishu"
        ),
    ),
}

# Ordered layout descriptor. Each entry is either:
#   ("static", key)                              → emit _HELP_STATIC_SECTIONS[key]
#   ("group", title, rows)                       → render a titled group
# where each row is either a registry name (str) — looked up at render
# time so the renderer always reflects ``COMMAND_REGISTRY`` — or a
# (name, fallback_desc) tuple for commands handled in handlers/_message.py
# (per design.md D1 — those don't belong in the registry).
# Group titles and inline-row descriptions are (zh, en) tuples.
# Inline rows: str = registry lookup; (name, zh, en) = manual entry (not in registry).
_HELP_LAYOUT: tuple = (
    ("static", "intro"),
    ("static", "decision_tree"),
    ("static", "separator"),
    ("group", ("🚀 常用", "🚀 Common"), (
        "/reset",
        ("/cancel", "取消当前查询", "Cancel the current query"),
        "/cd", "/pwd", "/ls", "/run", "/pickup", "/status",
        ("/rename", "命名当前会话", "Name this session"),
        "/history", "/stats",
        "/help",
    )),
    ("static", "model_shortcuts"),
    ("group", ("⚙️ 模型与推理", "⚙️ Model & Reasoning"), ("/model", "/lock", "/effort")),
    ("static", "reset_detail"),
    ("static", "memory_detail"),
    ("static", "doc_section"),
    ("group", ("📦 其他命令", "📦 Other"), (
        "/voice", "/cron", "/lang",
        ("/btw", "快问，不占主锁", "Side question — doesn't hold the main lock"),
        "/upgrade",
    )),
)


def _render_help_body(lang: str = "zh") -> str:
    """Compose the /help card body from ``_HELP_LAYOUT`` + ``COMMAND_REGISTRY``.

    Pure function — no I/O, no chat_id. Hidden / unregistered rows are
    silently skipped; a group whose rows all vanish drops its title too.
    Truncates with a trailing hint if length reaches MAX_CARD_LEN.
    """
    from larkhelm.command_registry import COMMAND_REGISTRY
    from larkhelm.locale import _t

    parts: list[str] = []
    for sec in _HELP_LAYOUT:
        kind = sec[0]
        if kind == "static":
            zh_text, en_text = _HELP_STATIC_SECTIONS[sec[1]]
            parts.append(_t(lang, zh_text, en_text))
            continue
        # group — title is a (zh, en) tuple
        _, title_pair, rows = sec
        title = _t(lang, title_pair[0], title_pair[1]) if isinstance(title_pair, tuple) else title_pair
        rendered_rows: list[str] = []
        for row in rows:
            if isinstance(row, str):
                spec = COMMAND_REGISTRY.lookup(row)
                if spec is None or spec.hidden:
                    continue
                if lang == "en" and spec.description_en:
                    desc = spec.description_en
                else:
                    desc = spec.description or "（无描述）"
                rendered_rows.append(f"**{spec.name}** — {desc}")
            else:
                # (name, zh_desc, en_desc) or legacy (name, fallback_desc)
                if len(row) == 3:
                    name, zh_desc, en_desc = row
                    desc = _t(lang, zh_desc, en_desc)
                else:
                    name, desc = row
                rendered_rows.append(f"**{name}** — {desc}")
        if not rendered_rows:
            continue
        parts.append(f"**{title}**\n" + "\n".join(rendered_rows))

    body = "\n\n".join(parts)
    max_len = getattr(_cfg, "MAX_CARD_LEN", 3000)
    if len(body) >= max_len:
        hint = _t(lang, "...（命令清单已截断，请查看 README）",
                  "...（command list truncated, see README）")
        body = body[: max_len - 100] + "\n\n" + hint
    return body


# ═══════════════════════════════════════════════════
#  Shell command execution
# ═══════════════════════════════════════════════════

# Env-var key fragments that indicate credentials — filtered from subprocess environment
# to prevent accidental leakage via /run output or error messages.
_SENSITIVE_ENV_PREFIXES = frozenset({
    "API_KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL",
    "ACCESS_KEY", "PRIVATE_KEY", "AUTH_KEY",
    "GITHUB_PAT", "DSN", "WEBHOOK", "SMTP", "DATABASE_URL",
    "SENTRY", "SIGNING_KEY", "ENCRYPTION_KEY",
})


def _run_shell(chat_id: str, cmd: str) -> tuple[str, str, int]:
    import os as _os
    cwd = _get_cwd(chat_id)
    try:
        import shlex
        args = shlex.split(cmd)
    except ValueError as e:
        return "", f"命令格式错误: {e}", 1
    safe_env = {k: v for k, v in _os.environ.items()
                if not any(s in k.upper() for s in _SENSITIVE_ENV_PREFIXES)}
    # P3-a (W17): timeout is configurable via `shell_timeout_sec`; defaults to 30s.
    timeout_s = int(getattr(_cfg, "SHELL_TIMEOUT", 30) or 30)
    import signal as _signal
    try:
        proc = subprocess.Popen(
            args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=cwd, env=safe_env, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            return stdout, stderr, proc.returncode
        except subprocess.TimeoutExpired:
            try:
                _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait()
            except Exception:
                pass
            return "", f"命令超时（>{timeout_s}s）", -1
    except Exception as e:
        return "", str(e), -1


def _strip_at_mention(text: str) -> str:
    """Strip Feishu group-chat @mention prefix."""
    return re.sub(r'@\S+\s*', '', text).strip()


# ═══════════════════════════════════════════════════
#  Command implementations
# ═══════════════════════════════════════════════════

def _reset_backend(
    chat_id: str,
    msg_id: str | None,
    backend: str,
    *,
    clear_counters: bool = False,
    lang: str = "zh",
) -> None:
    """Clear session ID + optionally zero session counters for one backend.

    Sends a success card on completion. Handles API history clear for
    backends that support it (gemini, kimi). clear_counters=True zeroes
    claude_session_guard counters — used for claude and full-reset paths.
    """
    _clear_sid(chat_id, backend)
    if clear_counters:
        clear_session_counters(chat_id)
    label_map = {
        "claude": "Claude",
        "gemini": "Gemini",
        "kimi": "Kimi",
        "deepseek": "DeepSeek",
    }
    label = label_map.get(backend, backend)
    send_card_reply(chat_id, msg_id, _t(lang, "♻️ 已重置", "♻️ Reset"),
                    _t(lang,
                       f"{label} 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                       f"{label} session cleared (memory kept).\n\nTo also clear session memory: `/memory clear session`"),
                    color="green")


def _cmd_reset(chat_id: str, which: str = None, msg_id: str = None):
    """Unified reset logic. which=None resets everything; otherwise 'claude'/'gemini'/'perm'."""
    lang = _get_lang(chat_id)

    # Trigger memory snapshot before clearing session (async, non-blocking)
    if which in (None, "claude", "gemini", "kimi", "deepseek"):
        try:
            from larkhelm.memory import maybe_auto_update
            maybe_auto_update(chat_id, force=True)
        except Exception as e:
            _debug_log(f"[reset] maybe_auto_update failed: {e}")

    _api_clear_failed: bool = False

    if which is None:
        _clear_sid(chat_id, "claude")
        _clear_sid(chat_id, "gemini")
        _clear_sid(chat_id, "kimi")
        _clear_sid(chat_id, "deepseek")
        try:
            from larkhelm.api_session import clear_history as _clear_api_hist
            from larkhelm.backend_registry import BACKEND_REGISTRY as _REG
            for _spec in _REG.all_enabled():
                if _spec.provider in ("anthropic_api", "google_api", "openai_compat_api"):
                    _clear_api_hist(_spec.provider, chat_id)
        except Exception as e:
            _debug_log(f"[reset] clear_history failed: {e}")
            _api_clear_failed = True
        # P0 (design.md §6.5 AC-05): a full reset also zeroes the Claude
        # session counters so the very next record_token_usage starts fresh.
        clear_session_counters(chat_id)
        log_entry(chat_id, "reset", "reset:all", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, _t(lang, "⚠️ 部分重置", "⚠️ Partial Reset"),
                            _t(lang,
                               "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                               "Session ID cleared, but API history clear failed — the AI may still remember some context."),
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, _t(lang, "♻️ 已重置", "♻️ Reset"),
                            _t(lang,
                               "所有 AI 会话均已清空（三层记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                               "All AI sessions cleared (three-layer memory kept).\n\nTo also clear session memory: `/memory clear session`"),
                            color="green")
    elif which == "claude":
        _clear_sid(chat_id, "claude")
        try:
            from larkhelm.api_session import clear_history as _clear_api_hist
            _clear_api_hist("anthropic_api", chat_id)
        except Exception as e:
            _debug_log(f"[reset] clear_history failed: {e}")
            _api_clear_failed = True
        # P0 (design.md §6.5 AC-05): clear the session cache/turn counters
        # so the auto-reset threshold isn't already partially primed.
        clear_session_counters(chat_id)
        log_entry(chat_id, "reset", "reset:claude", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, _t(lang, "⚠️ 部分重置", "⚠️ Partial Reset"),
                            _t(lang,
                               "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                               "Session ID cleared, but API history clear failed — the AI may still remember some context."),
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, _t(lang, "♻️ 已重置", "♻️ Reset"),
                            _t(lang,
                               "Claude 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                               "Claude session cleared (memory kept).\n\nTo also clear session memory: `/memory clear session`"),
                            color="green")
    elif which == "gemini":
        _clear_sid(chat_id, "gemini")
        try:
            from larkhelm.api_session import clear_history as _clear_api_hist
            _clear_api_hist("google_api", chat_id)
        except Exception as e:
            _debug_log(f"[reset] clear_history failed: {e}")
            _api_clear_failed = True
        try:
            from larkhelm import session_guard as _sg
            _sg.clear_session_counters(chat_id, "gemini")
        except Exception as _e:
            _debug_log(f"[reset] clear gemini session counters failed: {_e}")
        log_entry(chat_id, "reset", "reset:gemini", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, _t(lang, "⚠️ 部分重置", "⚠️ Partial Reset"),
                            _t(lang,
                               "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                               "Session ID cleared, but API history clear failed — the AI may still remember some context."),
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, _t(lang, "♻️ 已重置", "♻️ Reset"),
                            _t(lang,
                               "Gemini 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                               "Gemini session cleared (memory kept).\n\nTo also clear session memory: `/memory clear session`"),
                            color="green")
    elif which == "kimi":
        _clear_sid(chat_id, "kimi")
        try:
            from larkhelm.api_session import clear_history as _clear_api_hist
            _clear_api_hist("openai_compat_api", chat_id)
        except Exception as e:
            _debug_log(f"[reset] clear_history failed: {e}")
            _api_clear_failed = True
        try:
            from larkhelm import session_guard as _sg
            _sg.clear_session_counters(chat_id, "kimi")
        except Exception as _e:
            _debug_log(f"[reset] clear kimi session counters failed: {_e}")
        log_entry(chat_id, "reset", "reset:kimi", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, _t(lang, "⚠️ 部分重置", "⚠️ Partial Reset"),
                            _t(lang,
                               "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                               "Session ID cleared, but API history clear failed — the AI may still remember some context."),
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, _t(lang, "♻️ 已重置", "♻️ Reset"),
                            _t(lang,
                               "Kimi 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                               "Kimi session cleared (memory kept).\n\nTo also clear session memory: `/memory clear session`"),
                            color="green")
    elif which == "deepseek":
        _clear_sid(chat_id, "deepseek")
        try:
            from larkhelm import session_guard as _sg
            _sg.clear_session_counters(chat_id, "deepseek")
        except Exception as _e:
            _debug_log(f"[reset] clear deepseek session counters failed: {_e}")
        log_entry(chat_id, "reset", "reset:deepseek", model="system")
        send_card_reply(chat_id, msg_id, _t(lang, "♻️ 已重置", "♻️ Reset"),
                        _t(lang,
                           "DeepSeek 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                           "DeepSeek session cleared (memory kept).\n\nTo also clear session memory: `/memory clear session`"),
                        color="green")
    elif which == "memory":
        try:
            from larkhelm.memory import _session_memory_file
            _session_memory_file(chat_id).unlink(missing_ok=True)
        except Exception as e:
            _debug_log(f"[reset] memory unlink failed: {e}")
        log_entry(chat_id, "reset", "reset:memory", model="system")
        send_card_reply(chat_id, msg_id, _t(lang, "♻️ 已重置", "♻️ Reset"),
                        _t(lang,
                           "会话记忆已清除（全局/项目记忆保留）。",
                           "Session memory cleared (global/project memory kept)."),
                        color="green")
    elif which in ("perm", "permissions"):
        revoke_yolo(chat_id)
        send_card_reply(chat_id, msg_id, _t(lang, "🔐 权限已重置", "🔐 Permissions Reset"),
                        _t(lang,
                           "「允许所有」已取消，后续工具调用将重新弹出审批。",
                           "YOLO mode cancelled — future tool calls will require approval again."),
                        color="green")


def _cmd_status(chat_id: str, msg_id: str = None):
    from larkhelm._version import __version__
    cwd = _get_cwd(chat_id)
    lang = _get_lang(chat_id)

    if _cfg.SKIP_PERMISSIONS:
        perm_status = _t(lang, "⏭️ 跳过（skip_permissions=true）", "⏭️ Skipped (skip_permissions=true)")
    elif is_yolo(chat_id):
        perm_status = _t(lang, "🚀 允许所有（发送 **/reset perm** 恢复审批）",
                              "🚀 All allowed (send **/reset perm** to restore)")
    else:
        perm_status = _t(lang, "🔐 正常审批", "🔐 Normal approval")

    # Active crew info
    crew_info = ""
    try:
        from larkhelm.crew import _active_crew, _active_crew_states, _active_crew_lock
        from larkhelm.crew._state import describe_active_owner
        with _active_crew_lock:
            crew_id    = _active_crew.get(chat_id)
            crew_state = _active_crew_states.get(chat_id)
        if crew_id and crew_state:
            _phase_label = {
                "planning": "规划中", "planned": "已规划", "running": "执行中",
                "synthesizing": "综合中", "breakpoint": "等待确认",
                "done": "已完成", "cancelled": "已取消",
            }.get(crew_state.phase, crew_state.phase)
            from larkhelm.crew_types import AgentStatus as _AgentStatus
            n_done  = sum(1 for a in crew_state.agents.values() if a.status == _AgentStatus.DONE)
            n_total = len(crew_state.agents)
            crew_info = f"**Crew 进行中** {crew_id[:8]}…　{_phase_label}　{n_done}/{n_total} 完成"
        elif crew_id:
            # C4 #12 (sister of C4 #11 / C3 #9): when ``/plan`` owns the
            # slot, ``_active_crew_states`` is empty (plan never writes
            # it) but ``_active_crew`` carries the ``plan:<id>`` token.
            # Pre-C4 this branch labelled the row "Crew 进行中 plan:abc…"
            # — both the wrong command type AND a truncated token that
            # bleeds the ``plan:`` prefix into the displayed hex. Route
            # through ``describe_active_owner`` so the user sees the
            # real owner (``/plan 任务 (id=...)`` vs ``/crew 或 /dev
            # 任务 (id=...)``), matching the /crew-status fix in C4 #11.
            crew_info = f"**任务进行中** {describe_active_owner(crew_id)}"
    except Exception as e:
        _debug_log(f"[status] crew info failed: {e}")

    # Token summary (current process lifetime, current chat)
    token_summary = ""
    stats = get_token_stats(chat_id)
    if stats:
        parts = []
        for mdl, m in stats.items():
            cost_str = f" ${m['cost_usd']:.3f}" if m["cost_usd"] else ""
            parts.append(f"{mdl} {m['input_tokens']+m['output_tokens']:,}tok{cost_str}")
        token_summary = "　　".join(parts)

    # Backend registry summary — includes disabled specs (with ⏸) so users
    # can confirm "gemini is enabled=false in config" rather than assuming the
    # backend was simply forgotten. Shows last-activity age (real call OR
    # probe, whichever is more recent) and transient-failure pressure so the
    # user can spot a flapping backend before it's flipped unhealthy.
    backend_summary = ""
    try:
        import time as _time
        from larkhelm.backend_registry import BACKEND_REGISTRY
        from larkhelm.api_session import load_history as _load_hist
        _API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api", "deepseek_api")
        _MAX_HIST = 40
        # Use the public snapshot() helper rather than reaching into ``_lock``
        # directly — keeps callers decoupled from registry internals so
        # future synchronization changes (e.g. moving to a fine-grained lock
        # per spec) won't break this code path.
        all_specs = BACKEND_REGISTRY.snapshot()

        def _fmt_ago(ts: float, _now: float = None) -> str:
            if not ts:
                return "—"
            now = _now if _now is not None else _time.time()
            delta = max(0, int(now - ts))
            if delta < 60:
                return f"{delta}s前"
            if delta < 3600:
                return f"{delta // 60}m前"
            if delta < 86400:
                return f"{delta // 3600}h前"
            return f"{delta // 86400}d前"

        if all_specs:
            now = _time.time()
            n_enabled = sum(1 for s in all_specs if s.enabled)
            n_healthy = sum(1 for s in all_specs if s.enabled and s.healthy)
            spec_lines = []
            # Enabled first (sorted by id), then disabled
            enabled = sorted([s for s in all_specs if s.enabled], key=lambda x: x.id)
            disabled = sorted([s for s in all_specs if not s.enabled], key=lambda x: x.id)
            # Aggregate last-probe time across all specs → shown once in header
            last_probed_all = max(
                (getattr(s, "last_probed_at", 0.0) or 0.0 for s in all_specs),
                default=0.0,
            )
            probe_note = (
                f" · {_t(lang, '上次检测', 'probed')} {_fmt_ago(last_probed_all, now)}"
                if last_probed_all else ""
            )
            for s in enabled + disabled:
                if not s.enabled:
                    icon = "⏸"
                elif s.healthy:
                    icon = "✅"
                else:
                    icon = "❌"
                # Session ID for CLI backends
                sid_str = ""
                if s.enabled and s.provider in ("claude_cli", "gemini_cli", "kimi_cli"):
                    _sid = _load_sid(chat_id, s.id)
                    if _sid:
                        sid_str = f" {_t(lang, '会话', 'session')} **{_sid[:12]}…**"
                # Failure pressure (sliding window for TRANSIENT)
                fw = getattr(s, "failure_window", []) or []
                fail_str = f" ⚠️{len(fw)}{_t(lang, '失败', 'fails')}" if fw else ""
                # Error detail or API history count
                err_str = ""
                if not s.healthy and s.enabled and s.last_error:
                    err_str = f" _{s.last_error[:150]}_"
                elif s.enabled and s.provider in _API_PROVIDERS:
                    hist_len = len(_load_hist(s.provider, chat_id))
                    if hist_len:
                        err_str = f" `{hist_len}/{_MAX_HIST} {_t(lang, '条历史', 'history')}`"
                line = f"  • {icon} **{s.id}**{sid_str}{fail_str}"
                if err_str:
                    line += f"\n    {err_str}"
                spec_lines.append(line)
            disabled_note = (
                f" · {len(disabled)} {_t(lang, '个已停用', 'disabled')}" if disabled else ""
            )
            backend_summary = (
                f"**AI Backends** {n_healthy}/{n_enabled} "
                f"{_t(lang, '可用', 'healthy')}{probe_note}{disabled_note}\n"
                + "\n".join(spec_lines)
            )
    except Exception as e:
        _debug_log(f"[status] backend summary failed: {e}")

    # Phase C: Crew Backend 调度预览 — show which backend each task_profile
    # currently resolves to so operators can see at a glance whether a
    # planner / engineer / qa task would route to the expected provider
    # (or to ``<none>`` if no backend matches).
    crew_backend_preview = ""
    try:
        from larkhelm.crew._backend_resolver import resolve_backend_preview
        preview = resolve_backend_preview()
        if preview:
            orch_id = preview.get("orchestrator", "<none>")
            crew_backend_preview = f"**{_t(lang, 'Crew/Dev 主调度', 'Crew/Dev Orchestrator')}** {orch_id}"
    except Exception as e:
        _debug_log(f"[status] crew backend preview failed: {e}")

    # Effort level
    effort_line = ""
    try:
        _effort = _get_effort(chat_id)
        if _effort:
            _effort_labels = {"low": "⚡ 快速", "medium": "⚖️ 均衡", "high": "🔍 深度", "xhigh": "🚀 极限"}
            _el = _effort_labels.get(_effort, _effort)
            _tk = _t(lang, " · 思维链已关闭", " · thinking off") if _effort == "low" else ""
            effort_line = f"**{_t(lang, '推理力度', 'Effort')}** {_el} ({_effort}){_tk}"
    except Exception:
        pass

    _name = _get_chat_state(chat_id).get('name', '').replace('**', '').replace('`', '')
    _name_label = _t(lang, '会话名', 'Name')
    lines = [
        f"**{_t(lang, '目录', 'Directory')}** {cwd}"
        + (f"　　**{_name_label}** {_name}" if _name else ""),
        f"**{_t(lang, '版本', 'Version')}** {__version__}",
        "",
        f"**{_t(lang, '权限模式', 'Permissions')}** {perm_status}",
        *([ effort_line ] if effort_line else []),
        *([ crew_info ] if crew_info else []),
        *([ f"**{_t(lang, 'Token（本次启动）', 'Tokens (session)')}** {token_summary}" ] if token_summary else []),
        *([ backend_summary ] if backend_summary else []),
        *([ crew_backend_preview ] if crew_backend_preview else []),
        "",
    ]

    send_card_reply(chat_id, msg_id,
                   _t(lang, "📊 运行状态", "📊 Status"),
                   "\n".join(lines), color="turquoise", normalize=False)


def _cmd_help(chat_id: str, msg_id: str = None):
    lang = _get_lang(chat_id)
    body = _render_help_body(lang)
    title = "📖 Help" if lang == "en" else "📖 帮助"
    send_card_reply(chat_id, msg_id, title, body, color="blue", normalize=False)


def _cmd_pickup(chat_id: str, msg_id: str = None):
    lang = _get_lang(chat_id)
    s_c  = _load_sid(chat_id, "claude")
    s_g  = _load_sid(chat_id, "gemini")
    s_k  = _load_sid(chat_id, "kimi")
    s_d  = _load_sid(chat_id, "deepseek")
    cwd  = _get_cwd(chat_id)
    lines = [f"{_t(lang, '**工作目录:**', '**Working dir:**')} `{cwd}`\n"]
    if s_c:
        lines.append(f"{_t(lang, '**Claude 接力：**', '**Claude resume:**')}\n```bash\ncd {cwd}\nclaude --resume {s_c}\n```")
    else:
        lines.append(_t(lang, "**Claude:** 无活跃会话", "**Claude:** no active session"))
    if s_g:
        lines.append(f"\n{_t(lang, '**Gemini 接力：**', '**Gemini resume:**')}\n```bash\ncd {cwd}\ngemini --resume {s_g}\n```")
    else:
        lines.append("\n" + _t(lang, "**Gemini:** 无活跃会话", "**Gemini:** no active session"))
    if s_k:
        lines.append(f"\n{_t(lang, '**Kimi 接力：**', '**Kimi resume:**')}\n```bash\ncd {cwd}\nkimi --session {s_k}\n```")
    else:
        lines.append("\n" + _t(lang, "**Kimi:** 无活跃会话", "**Kimi:** no active session"))
    # DeepSeek has no terminal CLI
    if s_d:
        lines.append("\n" + _t(lang, "**DeepSeek:** 无官方 CLI，暂不支持终端接力。如需继续对话请在飞书直接发消息。", "**DeepSeek:** no official CLI — resume not supported. Continue chatting in Feishu."))
    else:
        lines.append("\n" + _t(lang, "**DeepSeek:** 无活跃会话", "**DeepSeek:** no active session"))
    lines.append("\n" + _t(lang, "> 在终端运行上面命令即可无缝接力", "> Run the command above in your terminal to resume seamlessly"))
    send_card_reply(chat_id, msg_id, _t(lang, "🔗 终端接力", "🔗 Terminal Handoff"), "\n".join(lines), color="purple")


def _cmd_history(chat_id: str, show_all: bool = False, msg_id: str = None):
    """Display conversation history.
    By default shows only the current session since the last reset;
    show_all=True shows all records with separator lines at reset points.
    """
    lang = _get_lang(chat_id)
    records = _read_logs(chat_id)

    def _build_pairs(recs: list[dict]) -> list[tuple[dict, dict]]:
        pairs: list[tuple[dict, dict]] = []
        pending_user: dict = None
        for r in recs:
            if r["role"] == "user":
                pending_user = r
            elif r["role"] in ("assistant", "error") and pending_user:
                pairs.append((pending_user, r))
                pending_user = None
        return pairs

    def _pair_line(u: dict, a: dict) -> str:
        date_str = u["ts"][:10]
        time_str = u["ts"][11:16]
        q = u["content"][:60].replace("\n", " ")
        suffix = "…" if len(u["content"]) > 60 else ""
        m = a.get("model") or u.get("model", "")
        if m == "crew":
            model_tag = "⚙️"
        elif m == "claude":
            model_tag = "🤖"
        else:
            model_tag = "✨"
        icon = "❌" if a["role"] == "error" else "✅"
        return f"{icon} {model_tag} **{date_str} {time_str}** — {q}{suffix}"

    if not show_all:
        # ── Default: show only the current session since the last reset ──
        last_reset_idx = None
        for i, r in enumerate(records):
            if r["role"] == "reset":
                last_reset_idx = i

        if last_reset_idx is None:
            # Never reset: fall back to old behavior, most recent 10 entries
            current = records
            pre_count = 0
        else:
            current = records[last_reset_idx + 1:]
            pre_count = len(_build_pairs(records[:last_reset_idx]))

        pairs = _build_pairs(current)
        if not pairs:
            body = _t(lang, "_当前会话暂无对话记录_", "_No conversation records in this session_")
            if pre_count:
                body += "\n\n" + _t(lang, f"此前会话还有 **{pre_count}** 条记录，发送 `/history all` 查看", f"Previous sessions have **{pre_count}** more records — send `/history all` to view")
            send_card_reply(chat_id, msg_id, _t(lang, "📜 当前会话", "📜 Current Session"), body, color="blue")
            return

        parts = [_pair_line(u, a) for u, a in pairs]
        footer = ""
        if pre_count:
            footer = "\n\n" + _t(lang, f"_此前会话还有 **{pre_count}** 条记录，发送 `/history all` 查看_", f"_Previous sessions: **{pre_count}** more records — send `/history all` to view_")
        title = _t(lang, f"📜 当前会话（{len(pairs)} 条）", f"📜 This Session ({len(pairs)} entries)")
        send_card_reply(chat_id, msg_id, title, "\n".join(parts) + footer, color="blue", normalize=False)

    else:
        # ── /history all: all records, separator lines at resets, max 20 entries ──
        parts: list[str] = []
        pending_user: dict = None
        pair_count = 0
        MAX_PAIRS = 20

        for r in records:
            if pair_count >= MAX_PAIRS:
                break
            if r["role"] == "reset":
                ts = r["ts"][5:16].replace("T", " ")
                which = r.get("content", "reset:all").replace("reset:", "")
                label = _t(lang,
                    {"all": "全部", "claude": "Claude", "gemini": "Gemini"}.get(which, which),
                    {"all": "all", "claude": "Claude", "gemini": "Gemini"}.get(which, which),
                )
                parts.append(f"— ♻️ {_t(lang, f'重置（{label}）', f'reset ({label})')} {ts} —")
                pending_user = None
            elif r["role"] == "user":
                pending_user = r
            elif r["role"] in ("assistant", "error") and pending_user:
                parts.append(_pair_line(pending_user, r))
                pending_user = None
                pair_count += 1

        if not parts:
            send_card_reply(chat_id, msg_id, _t(lang, "📜 对话历史", "📜 History"), _t(lang, "_暂无对话记录_", "_No conversation records_"), color="blue")
            return

        total_pairs = len(_build_pairs(records))
        if lang == "en":
            title = f"📜 History ({pair_count} entries"
            title += f" of {total_pairs})" if total_pairs > pair_count else ")"
        else:
            title = f"📜 全部历史（{pair_count} 条"
            title += f"，共 {total_pairs} 条）" if total_pairs > pair_count else "）"
        send_card_reply(chat_id, msg_id, title, "\n".join(parts), color="blue", normalize=False)


def _fmt_token_block(label: str, data: dict, lang: str = "zh") -> str:
    """Format a {model: {...}} statistics dict as a markdown block."""
    from larkhelm.locale import _t
    if not data:
        return f"**{label}** — {_t(lang, '暂无数据', 'No data')}"
    lines = [f"**{label}**"]
    for model, m in sorted(data.items()):
        model_label = model
        try:
            if _cfg.config.get("backend_aware_budget_enabled"):
                from larkhelm.backend_registry import BACKEND_REGISTRY
                from larkhelm.token_budget import resolve_context_window
                spec = BACKEND_REGISTRY.get(model)
                cw = resolve_context_window(spec)
                model_label = f"{model}（{cw // 1000}K ctx）"
        except Exception:
            pass
        inp   = m["input_tokens"]
        out   = m["output_tokens"]
        cr    = m["cache_read"]
        cc    = m["cache_create"]
        calls = m["calls"]
        cost  = m["cost_usd"]

        # Field semantics now uniform across all runners (post-stats-audit):
        #   input_tokens — non-cached prompt tokens (NEW input)
        #   cache_read   — input served from cache (priced at ~10%)
        #   cache_create — input written to cache (Claude only; priced ~25% extra)
        #   output_tokens — completion tokens
        # All four buckets are disjoint and additive. Total counts ALL four:
        # leaving cache_read out of "合计" hides ~80% of the real consumption
        # on cache-heavy chats and was the most user-visible /stats bug
        # caught by the independent audit (commands.py:621 → 34× under-count
        # in a 50k-cache hit + 1k new + 500 output scenario).
        total = inp + out + cr + cc

        # Cache hit-rate = cache_read / (cache_read + non-cached input).
        # Previously the denominator was ``inp`` alone, which mathematically
        # could exceed 100% on Claude where ``inp`` and ``cr`` are disjoint
        # (audit reproduced 900% on cr=9000 / inp=1000).
        prompt_in_total = cr + inp
        hit_pct    = int(cr / max(prompt_in_total, 1) * 100)
        create_pct = int(cc / max(prompt_in_total, 1) * 100)

        # ``cost`` is set explicitly by runner_*; only show "—" when the
        # source had no cost field at all (None). A real $0 (cache full
        # hit or free-tier) renders as "$0.0000" so the user can tell
        # apart "we don't know" from "we know it was free".
        cost_str = "—" if cost is None else f"${cost:.4f}"
        # Single block per model: drop the redundant ``› **{model}**``
        # prefix that previously sat on the hit-rate header (the prefix is
        # already on the next line). The parenthesised ``（X%）`` in the
        # detail row is intentionally retained — three audit regressions
        # (cache-arithmetic / overlap cases) pin that exact substring.
        lines.append(
            f"› **{model_label}**  {calls} {_t(lang, '次', 'calls')}  "
            f"{_t(lang, '合计', 'total')} **{total:,}** tokens  "
            f"{_t(lang, '费用', 'cost')} **{cost_str}**\n"
            f"  {_t(lang, '缓存命中率', 'hit rate')} **{hit_pct}%**  "
            f"{_t(lang, '写入比', 'write ratio')} **{create_pct}%**\n"
            f"  {_t(lang, '新输入', 'new input')} {inp:,}  {_t(lang, '输出', 'output')} {out:,}\n"
            f"  {_t(lang, '缓存命中', 'cache hits')} {cr:,}（{hit_pct}%）  "
            f"{_t(lang, '缓存写入', 'cache writes')} {cc:,}（{create_pct}%）"
        )
    return "\n".join(lines)


def _render_crew_agent_breakdown(chat_id: str, lang: str = "zh") -> list[str]:
    """Build the markdown lines for the /stats "Crew Agents" block.

    Returns ``[]`` when there are no crew-agent tokens (caller suppresses
    the section). Honours :pydata:`config.STATS_AGENT_TYPE_BREAKDOWN_ENABLED`:

      * ``True``  → header line + one line per bucket, sorted by total
        tokens descending (P5 REQ-07).
      * ``False`` → single-line P2-byte-compat fallback
        (REQ-09; for the rare card-overflow escape hatch).
    """
    from larkhelm.locale import _t
    breakdown_on = bool(
        getattr(_cfg, "STATS_AGENT_TYPE_BREAKDOWN_ENABLED", True)
    )

    if not breakdown_on:
        summary = summarize_crew_agent_tokens_for_chat(chat_id)
        if not summary:
            return []
        crew_total = (
            summary["input_tokens"] + summary["output_tokens"]
            + summary["cache_read"] + summary["cache_create"]
        )
        cost_val = summary.get("cost_usd", 0.0)
        cost_str = "—" if cost_val is None else f"${cost_val:.4f}"
        header = _t(lang, "**🤖 Crew Agents（本进程）**", "**🤖 Crew Agents (this process)**")
        return [
            "---",
            (
                f"{header}\n"
                f"› {summary['agents']} agents  "
                f"{_t(lang, '合计', 'total')} **{crew_total:,}** tokens  "
                f"{_t(lang, '费用', 'cost')} **{cost_str}**"
            ),
        ]

    buckets = summarize_crew_agent_tokens_by_type(chat_id)
    if not buckets:
        return []

    def _bucket_total(item: tuple[str, dict]) -> int:
        _name, agg = item
        return (
            agg.get("input_tokens", 0) + agg.get("output_tokens", 0)
            + agg.get("cache_read", 0) + agg.get("cache_create", 0)
        )

    ordered = sorted(buckets.items(), key=_bucket_total, reverse=True)
    header = _t(lang, "**🤖 Crew Agents（本进程·按类型）**",
                "**🤖 Crew Agents (this process · by type)**")
    lines = ["---", header]
    for type_name, agg in ordered:
        total = _bucket_total((type_name, agg))
        cost_val = agg.get("cost_usd", 0.0)
        cost_str = "—" if cost_val is None else f"${cost_val:.4f}"
        lines.append(
            f"› **{type_name}** {agg['agents']} agents  "
            f"{_t(lang, '合计', 'total')} **{total:,}** tokens  "
            f"{_t(lang, '费用', 'cost')} **{cost_str}**"
        )
    return lines


def _cmd_stats_intent(chat_id: str, msg_id: str = None, date: str | None = None,
                      lang: str = "zh"):
    """Render today's intent dispatcher aggregate (hit rate / latency / cost)."""
    from larkhelm.locale import _t
    try:
        from larkhelm.agent_hub.agent_audit import aggregate_daily
    except Exception as e:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "📊 Intent 统计", "📊 Intent Stats"),
                        _t(lang, f"agent_hub 未启用或导入失败：{e}",
                           f"agent_hub not enabled or import failed: {e}"),
                        color="orange")
        return
    # Round-4 audit P1 (R4-1d): pass the current chat_id so the aggregate
    # only reflects this chat's own intent dispatches. Pre-fix returned
    # the GLOBAL aggregate across all chats — observer in chat A could
    # infer activity volume / agent mix for chats B, C, ... by polling
    # /stats intent. Now the only entry-point that hits the global path
    # is the (future) admin CLI.
    agg = aggregate_daily(date, chat_id=chat_id)
    if agg["total"] == 0:
        send_card_reply(chat_id, msg_id,
                        f"{_t(lang, '📊 Intent 统计', '📊 Intent Stats')} · {agg['date']}",
                        _t(lang, "_当日没有 Agent 调度记录_",
                           "_No agent dispatches recorded today_"),
                        color="grey")
        return
    # ``agg['total_cost']`` was always rendered as "$0.0000" because none
    # of the 5 builtin agents (chat / dev / crew / plan / doc) populate
    # ``AgentResult.cost_usd`` — it just emits the default 0.0. Showing
    # a hardcoded "$0.0000" pretended the dispatcher was free; the
    # accurate per-chat cost lives in ``/stats`` (token block). Suppress
    # the cost line entirely until at least one agent emits a real value.
    # Independent stats audit caught this as a UI lie.
    has_cost = bool(agg.get("total_cost"))
    header = (
        f"**{_t(lang, '成功率', 'Success rate')}：** {agg['success_rate'] * 100:.1f}%　·　"
        f"{_t(lang, '平均耗时', 'Avg duration')}：{agg['avg_duration']:.2f}s"
    )
    if has_cost:
        header += f"　·　{_t(lang, '成本', 'cost')}：${agg['total_cost']:.4f}"
    lines = [
        f"**{_t(lang, '调度总数', 'Total dispatches')}：** {agg['total']}",
        header,
        "",
        f"**{_t(lang, '按 Agent', 'By agent')}：**",
    ]
    for atype, info in sorted(agg["per_agent"].items()):
        lines.append(
            f"- `{atype}`：{info['count']} {_t(lang, '次', 'calls')}"
            f"（{_t(lang, '成功', 'success')} {info['success']}，"
            f"avg {info['avg_duration']:.2f}s）"
        )
    send_card_reply(chat_id, msg_id,
                    f"{_t(lang, '📊 Intent 统计', '📊 Intent Stats')} · {agg['date']}",
                    "\n".join(lines), color="turquoise")


def _cmd_stats_cache(chat_id: str, msg_id: str = None, lang: str = "zh"):
    """Render prompt-cache savings summary and per-backend session counters."""
    from larkhelm.locale import _t
    lines: list[str] = []
    try:
        savings = get_cache_savings_summary()
        if savings:
            lines.append(f"**{_t(lang, '累计 Cache 节省（估算）', 'Total cache savings (estimated)')}：**")
            for model, usd in sorted(savings.items()):
                lines.append(f"- `{model}`：${usd:.4f}")
        else:
            lines.append(_t(lang,
                            "_暂无 cache 节省数据（本次进程尚无 cache_read 记录）_",
                            "_No cache savings yet (no cache_read records this process)_"))
    except Exception as e:
        _debug_log(f"[stats] cache savings failed: {e}")
        lines.append(_t(lang, f"_cache savings 读取失败：{e}_",
                        f"_Failed to read cache savings: {e}_"))

    lines.append("")
    lines.append(f"**{_t(lang, '各 Backend 会话计数器', 'Backend session counters')}：**")
    try:
        from larkhelm import session_guard as _sg
        for _backend in ("claude", "gemini", "kimi", "deepseek"):
            _bsc = _sg.get_session_counters(chat_id, _backend)
            _sc_cache = int(_bsc.get("cache_read", 0) or 0)
            _sc_turns = int(_bsc.get("turns", 0) or 0)
            _t_cache = int(_bsc.get("threshold_cache_read", 0) or 0)
            _t_turns = int(_bsc.get("threshold_turns", 0) or 0)
            if _t_cache == 0 and _t_turns == 0 and _sc_cache == 0 and _sc_turns == 0:
                continue
            _pct_c = min(100, int(_sc_cache * 100 / max(1, _t_cache))) if _t_cache else 0
            _pct_t = min(100, int(_sc_turns * 100 / max(1, _t_turns))) if _t_turns else 0
            _pct = max(_pct_c, _pct_t)
            _turns_label = _t(lang, "轮", "turns")
            _threshold_label = _t(lang, f"（距阈值 {_pct}%）", f" ({_pct}% to threshold)")
            lines.append(
                f"- `{_backend}`：{_sc_turns} {_turns_label} / {_sc_cache:,} tokens cache_read"
                + (_threshold_label if _pct else "")
            )
    except Exception as e:
        _debug_log(f"[stats] cache counters failed: {e}")
        lines.append(_t(lang, f"_会话计数器读取失败：{e}_",
                        f"_Failed to read session counters: {e}_"))

    # Cache hit rate block
    try:
        from larkhelm.token_stats import get_cache_hit_rate_summary
        _hit_summary = get_cache_hit_rate_summary()
        if _hit_summary:
            lines.append("")
            lines.append(f"**{_t(lang, '命中率（process-local）', 'Hit rate (process-local)')}：**")
            for _m, _d in sorted(_hit_summary.items()):
                _r = int(_d.get("read", 0) or 0)
                _i = int(_d.get("input", 0) or 0)
                if _r <= 0:
                    continue
                _hr = int(_r * 100 / (_r + _i)) if (_r + _i) > 0 else 0
                lines.append(
                    f"- `{_m}`：{_t(lang, f'命中率 {_hr}%（本进程累计 {_r:,} tokens cache_read）', f'hit rate {_hr}% ({_r:,} tokens cache_read this process)')}"
                )
    except Exception as _he:
        _debug_log(f"[stats] cache hit rate failed: {_he}")

    # Prefix stability block
    try:
        from larkhelm.metrics import get_registry as _get_reg
        _preg = _get_reg()
        if _preg.available and getattr(_preg, "prefix_stability_low_total", None) is not None:
            _families = _preg.prefix_stability_low_total.collect()
            _by_backend: dict[str, int] = {}
            for _family in _families:
                for _sample in _family.samples:
                    _b = (_sample.labels or {}).get("backend", "unknown")
                    _by_backend[_b] = _by_backend.get(_b, 0) + int(_sample.value or 0)
            if _by_backend:
                lines.append("")
                lines.append(f"**{_t(lang, 'Prefix 稳定性', 'Prefix stability')}：**")
                for _b, _n in sorted(_by_backend.items()):
                    lines.append(f"- `{_b}`：{_t(lang, f'变更 {_n} 次', f'{_n} changes')}")
    except Exception as _pse:
        _debug_log(f"[stats] prefix stability failed: {_pse}")

    send_card_reply(chat_id, msg_id,
                    _t(lang, "💾 Cache 统计", "💾 Cache Stats"),
                    "\n".join(lines), color="turquoise")


def _cmd_stats(chat_id: str, msg_id: str = None, args: str = ""):
    """Display today / this-month / all-time token stats plus conversation activity for the current chat."""
    # Round-3 review P0 (R3-2): route `/stats intent [YYYY-MM-DD]` so the
    # optional date suffix reaches `aggregate_daily(date=...)`. The prior
    # `args.strip().lower()` collapsed the entire suffix into `sub`, so any
    # date argument was silently dropped and users could never inspect
    # historical intent aggregates. Split on whitespace, take the first
    # token as the subcommand and forward the rest verbatim (no lower()
    # — ISO dates are case-irrelevant but keep the original casing for
    # any future subcommand args).
    from larkhelm.locale import _t as _lt
    lang = _get_lang(chat_id)
    parts = (args or "").split(None, 1)
    sub = parts[0].lower() if parts else ""
    sub_args = parts[1].strip() if len(parts) > 1 else ""
    if sub == "intent":
        # sub_args is the optional `YYYY-MM-DD` date; empty → today.
        date = sub_args or None
        _cmd_stats_intent(chat_id, msg_id, date=date, lang=lang)
        return
    if sub == "cache":
        _cmd_stats_cache(chat_id, msg_id, lang=lang)
        return
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")
    month = now.strftime("%Y-%m")

    records = _read_logs(chat_id)
    today_records = [r for r in records if r["ts"].startswith(today)]

    # Round-4 audit P1 (R4-1c): /crew / /plan write ``role="user", model in
    # {crew,plan}`` as a command marker, not a real conversation turn.
    # Counting them as "对话" inflated the dialogue count by every command
    # invocation. ``model=""`` (legacy records missing the field entirely)
    # still counts — only filter on KNOWN non-conversation markers so old
    # JSONL stays accurate. ``shell`` would never appear with role="user"
    # in practice (commands.py:975 emits ``role="shell"``) but pin it
    # here defensively.
    _NON_CONVERSATION_MODELS = {"crew", "plan", "shell"}
    user_count  = sum(
        1 for r in today_records
        if r["role"] == "user"
        and r.get("model", "") not in _NON_CONVERSATION_MODELS
    )
    error_count = sum(1 for r in today_records if r["role"] == "error")

    # Duration pairing — round-3 stats fixes:
    #
    # (Fix #2) Pair user/assistant entries by ``trace_id`` when both
    # sides carry one (covers concurrent /btw + main task where FIFO
    # pending_ts was scrambled by interleaved entries). Fall back to
    # FIFO position for entries without trace_id so old JSONL records
    # (pre-trace_id-propagation) still contribute.
    #
    # (Fix #3) The previous ``if 0 < secs < 3600`` cap silently dropped
    # every long-running /dev / /crew query — those are normal at >1h
    # under the default ``hard_timeout=21600``. Raised to
    # ``HARD_TIMEOUT * 1.1`` so the cap kicks in only for clearly-bad
    # records (e.g. a missing pair-end making the gap span days).
    durations: list[float] = []
    _pending_by_trace: dict[str, datetime] = {}
    _pending_fifo: datetime | None = None
    _hard_cap_secs = float(getattr(_cfg, "HARD_TIMEOUT", 21600) or 21600) * 1.1

    def _accept_duration(end_ts: datetime, start_ts: datetime) -> None:
        secs = (end_ts - start_ts).total_seconds()
        if 0 < secs < _hard_cap_secs:
            durations.append(secs)

    for r in today_records:
        role = r.get("role", "")
        if role not in ("user", "assistant", "error"):
            continue
        try:
            ts = datetime.fromisoformat(r["ts"])
        except (KeyError, ValueError):
            continue
        tid = r.get("trace_id")
        if role == "user":
            if tid:
                _pending_by_trace[tid] = ts
            else:
                _pending_fifo = ts
        else:
            # assistant / error — prefer trace-id pairing; fall back to FIFO.
            paired_start: datetime | None = None
            if tid and tid in _pending_by_trace:
                paired_start = _pending_by_trace.pop(tid)
            elif _pending_fifo is not None:
                paired_start = _pending_fifo
                _pending_fifo = None
            if paired_start is not None:
                _accept_duration(ts, paired_start)
    # Round-3 review NICE-TO-HAVE: orphan trace_ids (user logged with
    # an id but no matching assistant entry by end of the scan) and
    # leftover FIFO entries surface as silent under-counts in the
    # duration average. Emit a single debug-log line so operators
    # bisecting "/stats numbers look wrong" can see the gap without
    # diffing JSONL by hand. Cheap path; only logs when non-zero.
    _orphan_traced = len(_pending_by_trace)
    _orphan_fifo = 1 if _pending_fifo is not None else 0
    if _orphan_traced or _orphan_fifo:
        _debug_log(
            f"[stats] _cmd_stats({chat_id[:12]}) pairing leftovers: "
            f"{_orphan_traced} trace_id orphans, {_orphan_fifo} FIFO orphan "
            f"(user logged but no matching assistant/error before scan end)"
        )
    avg = sum(durations) / len(durations) if durations else 0

    # Persistent stats across three time windows
    stats_today = get_token_stats_persistent(chat_id, date_prefix=today)
    stats_month = get_token_stats_persistent(chat_id, date_prefix=month)
    stats_all   = get_token_stats_persistent(chat_id, date_prefix=None)

    def _savings_line(stats: dict) -> str | None:
        total_savings = sum(
            estimate_cache_savings(model, m) for model, m in stats.items()
        )
        total_cache_read = sum(
            int(m.get("cache_read", 0) or 0) for m in stats.values()
        )
        total_input = sum(
            int(m.get("input_tokens", 0) or 0) + int(m.get("cache_read", 0) or 0)
            for m in stats.values()
        )
        if total_cache_read == 0 or total_savings <= 0:
            return None
        hit_pct = int(total_cache_read * 100 / total_input) if total_input > 0 else 0
        return _lt(lang,
                   f"缓存节省 **${total_savings:.4f}**（命中 {hit_pct}%）",
                   f"Cache savings **${total_savings:.4f}** (hit rate {hit_pct}%)")

    # In-memory stats for this process run (fallback if all.jsonl is empty)
    stats_mem   = get_token_stats(chat_id)

    _today_label = _lt(lang, f"📅 今日（{today}）", f"📅 Today ({today})")
    _month_label = _lt(lang, f"🗓 本月（{month}）", f"🗓 This month ({month})")
    _all_label   = _lt(lang, "📦 累计（全部）", "📦 All-time")

    _today_block = _fmt_token_block(_today_label, stats_today, lang)
    _month_block = _fmt_token_block(_month_label, stats_month, lang)
    _all_block   = _fmt_token_block(_all_label,   stats_all,   lang)

    def _append_savings(block: str, stats: dict) -> str:
        sl = _savings_line(stats)
        return (block + "\n" + sl) if sl else block

    _turns_word = _lt(lang, "轮", "turns")
    parts = [
        f"**{_lt(lang, '统计日期', 'Stats date')}：** {today}",
        (
            f"{_lt(lang, '今日对话', 'Today')}：**{user_count}** {_lt(lang, '次', 'queries')}"
            f"　{_lt(lang, '错误', 'errors')}：**{error_count}**　"
            f"{_lt(lang, '平均耗时', 'avg duration')}：**{_fmt_elapsed(avg) if avg else '—'}**"
        ),
        "---",
        _append_savings(_today_block, stats_today),
        "---",
        _append_savings(_month_block, stats_month),
        "---",
        _append_savings(_all_block, stats_all),
    ]

    # P0 (design.md §6.5 AC-03): show Claude session counters and how far
    # they are from triggering an auto-reset, so operators can spot a
    # prefix-bloat session before the threshold actually fires.
    try:
        sc = get_session_counters(chat_id)
        if sc.get("enabled"):
            sc_cache  = int(sc.get("cache_read", 0) or 0)
            sc_turns  = int(sc.get("turns", 0) or 0)
            t_cache   = max(1, int(sc.get("threshold_cache_read", 1) or 1))
            t_turns   = max(1, int(sc.get("threshold_turns", 1) or 1))
            pct_cache = min(100, int(sc_cache * 100 / t_cache))
            pct_turns = min(100, int(sc_turns * 100 / t_turns))
            pct = max(pct_cache, pct_turns)
            parts.append("---")
            parts.append(
                f"**{_lt(lang, '当前 session', 'Current session')}："
                f"{sc_turns} {_turns_word} / {sc_cache:,} tokens cache_read"
                f"{_lt(lang, f'（距离阈值 {pct}%）', f' ({pct}% to threshold)')}**"
            )
        # REQ-02: show counters for other backends when thresholds are configured
        try:
            from larkhelm import session_guard as _sg
            for _backend in ("gemini", "deepseek", "kimi"):
                _bsc = _sg.get_session_counters(chat_id, _backend)
                _t_cache = int(_bsc.get("threshold_cache_read", 0) or 0)
                _t_turns = int(_bsc.get("threshold_turns", 0) or 0)
                if _t_cache == 0 and _t_turns == 0:
                    continue
                _sc_cache = int(_bsc.get("cache_read", 0) or 0)
                _sc_turns = int(_bsc.get("turns", 0) or 0)
                _pct_c = min(100, int(_sc_cache * 100 / max(1, _t_cache))) if _t_cache else 0
                _pct_t = min(100, int(_sc_turns * 100 / max(1, _t_turns))) if _t_turns else 0
                _pct = max(_pct_c, _pct_t)
                parts.append(
                    f"**{_backend} session：{_sc_turns} {_turns_word} / {_sc_cache:,} tokens "
                    f"cache_read{_lt(lang, f'（距离阈值 {_pct}%）', f' ({_pct}% to threshold)')}**"
                )
        except Exception as _be:
            _debug_log(f"[stats] backend session counters render failed: {_be}")
    except Exception as e:
        _debug_log(f"[stats] session counters render failed: {e}")

    # If persistent data is empty (fresh deploy or upgrade from old version), show in-memory stats as fallback
    if not stats_all and stats_mem:
        parts.append("---")
        parts.append(_fmt_token_block(
            _lt(lang, "⚡ 本次启动（内存）", "⚡ This session (memory)"),
            stats_mem, lang))

    # Round-4 audit P1 (R4-1e) + P5 REQ-07/09: expose the in-memory
    # per-crew-agent counter. Counters reset on bridge restart so the
    # block labels "本进程" — totals are also rolled up into the parent
    # chat's persistent stats above, so this is a per-agent split, not a
    # separate accounting source. STATS_AGENT_TYPE_BREAKDOWN_ENABLED=false
    # restores the P2 single-line summary.
    parts.extend(_render_crew_agent_breakdown(chat_id, lang))

    send_card_reply(chat_id, msg_id,
                    _lt(lang, "📊 Token 统计", "📊 Token Stats"),
                    "\n\n".join(parts), color="turquoise")


def _cmd_cron(chat_id: str, args: str, msg_id: str = None):
    """Handle /cron command: add / list / del."""
    from croniter import croniter, CroniterBadCronError
    import uuid as _uuid

    lang = _get_lang(chat_id)
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""

    if sub == "list":
        crons = _get_chat_state(chat_id).get("crons", [])
        if not crons:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⏰ 定时任务", "⏰ Scheduled Tasks"),
                            _t(lang, "_暂无定时任务_", "_No scheduled tasks_"),
                            color="blue")
            return
        lines = []
        for c in crons:
            block = f"**`{c['id']}`** `{c['expr']}` [{c['model']}]\n{c['query'][:60]}"
            last_at = c.get("last_run_at", "")
            last_status = c.get("last_run_status", "")
            if not last_at or not last_status:
                status_line = _t(lang, "上次：从未执行", "Last: never run")
            else:
                icon = "✅" if last_status == "ok" else "❌"
                status_line = _t(lang, f"上次：{icon} {last_at}", f"Last: {icon} {last_at}")
                if last_status == "error":
                    last_error = c.get("last_error", "")
                    if last_error:
                        status_line += f"（{last_error[:60]}）"
            block += f"\n{status_line}"
            lines.append(block)
        send_card_reply(chat_id, msg_id,
                        _t(lang, f"⏰ 定时任务（{len(crons)} 条）", f"⏰ Scheduled Tasks ({len(crons)})"),
                        "\n\n---\n\n".join(lines), color="blue")
        return

    if sub == "del":
        cron_id = parts[1].strip() if len(parts) > 1 else ""
        if not cron_id:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⚠️ 用法", "⚠️ Usage"),
                            "`/cron del <id>`", color="orange")
            return
        with _cron_lock:
            crons = _get_chat_state(chat_id).get("crons", [])
            new_crons = [c for c in crons if c["id"] != cron_id]
            _set_chat_field(chat_id, "crons", new_crons)
        if len(new_crons) < len(crons if crons else []):
            send_card_reply(chat_id, msg_id,
                            _t(lang, "✅ 已删除", "✅ Deleted"),
                            _t(lang, f"定时任务 `{cron_id}` 已删除。",
                               f"Scheduled task `{cron_id}` deleted."),
                            color="green")
        else:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❓ 未找到", "❓ Not Found"),
                            _t(lang, f"没有 ID 为 `{cron_id}` 的定时任务。",
                               f"No scheduled task with ID `{cron_id}`."),
                            color="orange")
        return

    if sub == "add":
        rest = parts[1].strip() if len(parts) > 1 else ""
        m = re.match(r'^["\'](.+?)["\'\s](.+)$', rest) or re.match(
            r'^((?:\S+\s+){4}\S+)\s+(.+)$', rest)
        if not m:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⚠️ 用法", "⚠️ Usage"),
                            _t(lang,
                               '`/cron add "0 9 * * *" 每日早报查询`\n\n'
                               "cron 表达式为标准 5 字段（分 时 日 月 周）",
                               '`/cron add "0 9 * * *" daily briefing`\n\n'
                               "cron expression: standard 5-field (min hour day month weekday)"),
                            color="orange")
            return
        expr, query = m.group(1).strip(), m.group(2).strip()
        try:
            croniter(expr)
        except (CroniterBadCronError, Exception):
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 表达式错误", "❌ Invalid Expression"),
                            _t(lang,
                               f"`{expr}` 不是有效的 cron 表达式。\n\n示例：`0 9 * * *`（每天 9:00）",
                               f"`{expr}` is not a valid cron expression.\n\nExample: `0 9 * * *` (daily at 9:00)"),
                            color="red")
            return
        model = _get_chat_model(chat_id)
        cron_id = _uuid.uuid4().hex[:8]
        entry = {"id": cron_id, "expr": expr, "query": query,
                 "model": model, "created_at": datetime.now().isoformat(timespec="seconds")}
        with _cron_lock:
            crons = list(_get_chat_state(chat_id).get("crons", []))
            crons.append(entry)
            _set_chat_field(chat_id, "crons", crons)

        from zoneinfo import ZoneInfo
        tz = ZoneInfo(_cfg.CRON_TIMEZONE)
        nxt = croniter(expr, datetime.now(tz)).get_next(datetime)
        send_card_reply(chat_id, msg_id,
                        _t(lang, "✅ 定时任务已添加", "✅ Task Scheduled"),
                        _t(lang,
                           f"**ID：** `{cron_id}`\n\n"
                           f"**表达式：** `{expr}`（时区：{_cfg.CRON_TIMEZONE}）\n\n"
                           f"**查询：** {query[:80]}\n\n"
                           f"**下次执行：** {nxt.strftime('%Y-%m-%d %H:%M')}\n\n"
                           f"查看：`/cron list`　删除：`/cron del {cron_id}`",
                           f"**ID:** `{cron_id}`\n\n"
                           f"**Expression:** `{expr}` (timezone: {_cfg.CRON_TIMEZONE})\n\n"
                           f"**Query:** {query[:80]}\n\n"
                           f"**Next run:** {nxt.strftime('%Y-%m-%d %H:%M')}\n\n"
                           f"List: `/cron list`　Delete: `/cron del {cron_id}`"),
                        color="green")
        return

    send_card_reply(chat_id, msg_id,
                    _t(lang, "⚠️ 用法", "⚠️ Usage"),
                    _t(lang,
                       '`/cron add "<expr>" <查询>` — 添加定时任务\n'
                       "`/cron list` — 查看所有任务\n"
                       "`/cron del <id>` — 删除任务\n\n"
                       '示例：`/cron add "0 9 * * 1-5" 总结今日 git log`',
                       '`/cron add "<expr>" <query>` — add a scheduled task\n'
                       "`/cron list` — list all tasks\n"
                       "`/cron del <id>` — delete a task\n\n"
                       '`/cron add "0 9 * * 1-5" summarise today\'s git log`'),
                    color="orange")


def _check_cwd_root(p: Path) -> bool:
    """Return False if cwd_root is configured and p escapes it."""
    root = _cfg.config.get("cwd_root")
    if not root:
        return True
    try:
        p.resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _cmd_cd(chat_id: str, path: str, msg_id: str = None):
    lang = _get_lang(chat_id)
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(_get_cwd(chat_id)) / p
        p = p.resolve()
        if not p.is_dir():
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 目录不存在", "❌ Directory Not Found"),
                            f"`{p}`", color="red")
            return
        if not _check_cwd_root(p):
            _root = _cfg.config.get("cwd_root", "")
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 超出允许范围", "❌ Outside Allowed Path"),
                            _t(lang,
                               f"配置的 `cwd_root` 限制了可访问路径：`{_root}`\n\n"
                               f"请切换到 `{_root}` 内的子目录，例如 `/cd {_root}`",
                               f"Configured `cwd_root` restricts accessible paths: `{_root}`\n\n"
                               f"Switch to a subdirectory under `{_root}`, e.g. `/cd {_root}`"),
                            color="red")
            return
        _set_chat_field(chat_id, "cwd", str(p))
        send_card_reply(chat_id, msg_id,
                        _t(lang, "📁 目录已切换", "📁 Directory Changed"),
                        f"`{p}`", color="green")
    except Exception as e:
        send_card_reply(chat_id, msg_id, _t(lang, "❌ 错误", "❌ Error"), str(e), color="red")


def _cmd_pwd(chat_id: str, msg_id: str = None):
    lang = _get_lang(chat_id)
    send_card_reply(chat_id, msg_id,
                    _t(lang, "📁 当前目录", "📁 Current Directory"),
                    f"`{_get_cwd(chat_id)}`", color="blue")


def _cmd_ls(chat_id: str, path: str = "", msg_id: str = None):
    lang = _get_lang(chat_id)
    cwd = _get_cwd(chat_id)
    target = (Path(cwd) / path if path else Path(cwd)).resolve()
    if not _check_cwd_root(target):
        _root = _cfg.config.get("cwd_root", "")
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ 超出允许范围", "❌ Outside Allowed Path"),
                        _t(lang,
                           f"配置的 `cwd_root` 限制了可访问路径：`{_root}`\n\n"
                           f"请在 `{_root}` 内操作",
                           f"Configured `cwd_root` restricts accessible paths: `{_root}`\n\n"
                           f"Please operate within `{_root}`"),
                        color="red")
        return
    try:
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = [f"`{target}/`\n"]
        for e in entries[:60]:
            icon = "📁" if e.is_dir() else "📄"
            size = ""
            if e.is_file():
                s = e.stat().st_size
                size = f"  _{s//1024}KB_" if s >= 1024 else f"  _{s}B_"
            lines.append(f"{icon} `{e.name}`{size}")
        if len(entries) > 60:
            lines.append(f"\n_... {_t(lang, f'共 {len(entries)} 项', f'{len(entries)} entries total')}_")
        if not entries:
            lines.append(_t(lang, "（空目录）", "(empty directory)"))
        send_card_reply(chat_id, msg_id,
                        _t(lang, "📂 文件列表", "📂 File List"),
                        "\n".join(lines), color="blue", normalize=False)
    except Exception as e:
        send_card_reply(chat_id, msg_id, _t(lang, "❌ 错误", "❌ Error"), str(e), color="red")


def _cmd_run(chat_id: str, cmd: str, msg_id: str = None):
    lang = _get_lang(chat_id)
    cwd = _get_cwd(chat_id)
    mid = send_card_reply(chat_id, msg_id,
                          _t(lang, "⏳ 执行中", "⏳ Running"),
                          f"```bash\n{cmd}\n```\n{_t(lang, '目录:', 'dir:')} `{cwd}`",
                          color="grey")
    stdout, stderr, rc = _run_shell(chat_id, cmd)
    color = "green" if rc == 0 else "red"
    icon = "✅" if rc == 0 else "❌"
    body = f"```bash\n{cmd}\n```\n{_t(lang, '目录:', 'dir:')} `{cwd}`\n"
    if stdout.strip():
        body += f"\n{_t(lang, '**输出：**', '**Output:**')}\n```\n{stdout.strip()[:2000]}\n```"
    if stderr.strip():
        body += f"\n{_t(lang, '**错误：**', '**Stderr:**')}\n```\n{stderr.strip()[:500]}\n```"
    if not stdout.strip() and not stderr.strip():
        body += f"\n{_t(lang, '_（无输出）_', '_(no output)_')}"
    body += f"\n\n{_t(lang, '退出码:', 'exit code:')} `{rc}`"
    # Log only the command itself, not its output (stdout/stderr may contain passwords, tokens, etc.)
    log_entry(chat_id, "shell", f"$ {cmd}", model="shell")
    title = f"{icon} {_t(lang, '完成', 'Done')}" if rc == 0 else f"{icon} {_t(lang, '执行失败', 'Failed')}"
    reply_card(chat_id, mid, title, body, color=color)


def _cmd_lock(chat_id: str, args: str = "", msg_id: str = None) -> None:
    """Handle /lock command.

    /lock <backend_id>  — lock to specific backend (validates exists + healthy)
    /lock off           — clear locked_backend, restore auto-routing
    /lock               — show current locked_backend status
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY
    lang = _get_lang(chat_id)
    args = args.strip()

    if not args:
        # Show current locked backend
        locked_id = _get_chat_state(chat_id).get("locked_backend")
        if locked_id:
            spec = BACKEND_REGISTRY.get(locked_id)
            name = spec.display_name if spec else locked_id
            health = "✅" if (spec and spec.healthy) else "❌"
            send_card_reply(chat_id, msg_id,
                            _t(lang, "🔒 已锁定 Backend", "🔒 Backend Locked"),
                            _t(lang,
                               f"{health} **{name}** (`{locked_id}`)\n\n解锁: `/lock off`",
                               f"{health} **{name}** (`{locked_id}`)\n\nUnlock: `/lock off`"),
                            color="blue")
        else:
            specs = BACKEND_REGISTRY.all_enabled()
            lines = [
                _t(lang, "_当前使用自动路由_\n", "_Currently using auto-routing_\n"),
                _t(lang, "可用 Backends:", "Available Backends:"),
            ]
            for s in specs:
                mark = "✅" if s.healthy else "❌"
                lines.append(f"  {mark} `{s.id}` — {s.display_name} (`{s.provider}`)")
            lines.append(_t(lang, "\n锁定: `/lock <id>`", "\nLock: `/lock <id>`"))
            send_card_reply(chat_id, msg_id,
                            _t(lang, "🔓 自动路由中", "🔓 Auto-routing"),
                            "\n".join(lines), color="blue", normalize=False)
        return

    if args.lower() == "off":
        _set_chat_field(chat_id, "locked_backend", None)
        send_card_reply(chat_id, msg_id,
                        _t(lang, "🔓 已解锁", "🔓 Unlocked"),
                        _t(lang,
                           "已恢复自动路由，不再锁定特定 backend。",
                           "Restored auto-routing — no backend locked."),
                        color="green")
        return

    # Lock to specific backend
    backend_id = args
    spec = BACKEND_REGISTRY.get(backend_id)
    if spec is None:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ Backend 不存在", "❌ Backend Not Found"),
                        _t(lang,
                           f"未找到 backend: `{backend_id}`\n发送 `/lock` 查看所有可用 backends。",
                           f"Backend not found: `{backend_id}`\nSend `/lock` to see all available backends."),
                        color="red")
        return

    if not spec.enabled:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ Backend 已禁用", "❌ Backend Disabled"),
                        _t(lang,
                           f"`{backend_id}` 已被禁用（enabled=false），无法锁定。\n\n"
                           f"发送 `/lock` 查看可用 backends。",
                           f"`{backend_id}` is disabled (enabled=false) and cannot be locked.\n\n"
                           f"Send `/lock` to see available backends."),
                        color="red")
        return

    if not spec.healthy:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ Backend 不可用", "❌ Backend Unavailable"),
                        _t(lang,
                           f"`{backend_id}` 当前不可用（health check 失败）。\n\n"
                           f"错误: {spec.last_error or '未知'}\n\n"
                           f"可用: 等待恢复后重试，或 `/lock` 选择其他 backend。",
                           f"`{backend_id}` is currently unavailable (health check failed).\n\n"
                           f"Error: {spec.last_error or 'unknown'}\n\n"
                           f"Try again after it recovers, or `/lock` to choose another backend."),
                        color="red")
        return

    _set_chat_field(chat_id, "locked_backend", spec.id)
    send_card_reply(chat_id, msg_id,
                    _t(lang, "🔒 已锁定 Backend", "🔒 Backend Locked"),
                    _t(lang,
                       f"本 chat 已锁定到 **{spec.display_name}** (`{spec.id}`)。\n\n解锁: `/lock off`",
                       f"This chat is locked to **{spec.display_name}** (`{spec.id}`).\n\nUnlock: `/lock off`"),
                    color="green")


def _cmd_model(chat_id: str, model_name: str = "", msg_id: str = None):
    """Alias for /lock: switches default backend for this chat."""
    _cmd_lock(chat_id, model_name, msg_id)


def _cmd_voice(chat_id: str, args: str = "", msg_id: str = None) -> None:
    """Handle /voice command.

    /voice              → equivalent to /voice status
    /voice status       → status card (voice_lang + global VOICE_* + is_model_loaded())
    /voice lang <x>     → set voice_lang for this chat (whitelist: zh / en / auto)
    其它                → usage card
    Never raises; failures surface as user-visible cards.
    """
    from larkhelm.chat_state import _get_voice_lang, _set_voice_lang
    ui_lang = _get_lang(chat_id)
    args = (args or "").strip()

    if args == "" or args == "status":
        try:
            from larkhelm.voice.transcribe import is_model_loaded
            loaded = is_model_loaded()
        except Exception as e:
            _debug_log(f"[Voice] is_model_loaded probe failed: {e}")
            loaded = False
        voice_lang = _get_voice_lang(chat_id)
        enabled = _t(ui_lang, "✅ 开启", "✅ Enabled") if _cfg.VOICE_ENABLED else _t(ui_lang, "⏸️ 关闭", "⏸️ Disabled")
        engine = (getattr(_cfg, "VOICE_ENGINE", "faster_whisper") or "faster_whisper")
        api_key_set = bool(getattr(_cfg, "VOICE_API_KEY", ""))

        # Engine-specific status hints — collapse model-loaded vs api-key
        # status into one line so the card stays compact.
        if engine == "dashscope":
            # Probe SDK availability without importing transcribe (which would
            # in turn import the dashscope adapter module).
            try:
                import importlib.util as _ilu
                sdk_present = _ilu.find_spec("dashscope") is not None
            except Exception:
                sdk_present = False
            sdk_str = _t(ui_lang,
                         "✅ SDK 已装" if sdk_present else "⏸️ SDK 未装（pipx runpip larkhelm install dashscope）",
                         "✅ SDK installed" if sdk_present else "⏸️ SDK missing (pipx runpip larkhelm install dashscope)")
            key_str = _t(ui_lang,
                         "✅ 配置" if api_key_set else "⏸️ 未配置（设 $DASHSCOPE_API_KEY）",
                         "✅ Configured" if api_key_set else "⏸️ Not set (set $DASHSCOPE_API_KEY)")
            engine_lines = _t(ui_lang,
                              f"**引擎：** `dashscope`（云 API）\n"
                              f"**API Key：** {key_str}\n"
                              f"**SDK 状态：** {sdk_str}\n",
                              f"**Engine:** `dashscope` (cloud API)\n"
                              f"**API Key:** {key_str}\n"
                              f"**SDK:** {sdk_str}\n")
        else:
            loaded_str = _t(ui_lang,
                            "✅ 已加载" if loaded else "⏸️ 未加载（首次语音消息懒加载）",
                            "✅ Loaded" if loaded else "⏸️ Not loaded (lazy-loaded on first voice message)")
            engine_lines = _t(ui_lang,
                              f"**引擎：** `faster_whisper`（本地）\n"
                              f"**模型：** `{_cfg.VOICE_MODEL_SIZE}` ({_cfg.VOICE_COMPUTE_TYPE})\n"
                              f"**模型状态：** {loaded_str}\n",
                              f"**Engine:** `faster_whisper` (local)\n"
                              f"**Model:** `{_cfg.VOICE_MODEL_SIZE}` ({_cfg.VOICE_COMPUTE_TYPE})\n"
                              f"**Model status:** {loaded_str}\n")

        body = _t(ui_lang,
                  f"**总开关：** {enabled}\n"
                  f"{engine_lines}"
                  f"**当前语种：** `{voice_lang}`（默认 `{_cfg.VOICE_DEFAULT_LANG}`）\n"
                  f"**音频上限：** {_cfg.VOICE_MAX_DURATION_MS // 1000}s\n"
                  f"**合并窗口：** {_cfg.VOICE_MERGE_WINDOW_SEC}s（cap {_cfg.VOICE_MAX_MERGE}）\n\n"
                  f"切换语种：`/voice lang zh|en|auto`",
                  f"**Master switch:** {enabled}\n"
                  f"{engine_lines}"
                  f"**Language:** `{voice_lang}` (default `{_cfg.VOICE_DEFAULT_LANG}`)\n"
                  f"**Audio limit:** {_cfg.VOICE_MAX_DURATION_MS // 1000}s\n"
                  f"**Merge window:** {_cfg.VOICE_MERGE_WINDOW_SEC}s (cap {_cfg.VOICE_MAX_MERGE})\n\n"
                  f"Switch language: `/voice lang zh|en|auto`")
        send_card_reply(chat_id, msg_id, _t(ui_lang, "🎤 Voice 状态", "🎤 Voice Status"), body, color="blue")
        return

    if args.startswith("lang "):
        lang_arg = args[5:].strip().lower()
        if lang_arg in _cfg._VOICE_LANG_WHITELIST:
            _set_voice_lang(chat_id, lang_arg)
            send_card_reply(chat_id, msg_id,
                            _t(ui_lang, "✅ 语种已切换", "✅ Language Switched"),
                            f"voice_lang = `{lang_arg}`", color="green")
        else:
            send_card_reply(chat_id, msg_id,
                            _t(ui_lang, "❌ 语种无效", "❌ Invalid Language"),
                            _t(ui_lang, "可选值：`zh` / `en` / `auto`", "Valid values: `zh` / `en` / `auto`"),
                            color="red")
        return

    send_card_reply(chat_id, msg_id,
                    _t(ui_lang, "⚠️ 用法", "⚠️ Usage"),
                    _t(ui_lang,
                       "`/voice status` — 查看当前配置\n"
                       "`/voice lang <zh|en|auto>` — 切换语种",
                       "`/voice status` — show current configuration\n"
                       "`/voice lang <zh|en|auto>` — switch language"),
                    color="orange")


# Native command sets for Claude CLI / Gemini CLI / Kimi CLI
_CLAUDE_SESSION_CMDS = {
    "/compact", "/clear", "/cost", "/memory", "/mcp",
    "/model", "/review", "/pr_comments", "/vim",
    "/terminal-setup", "/doctor", "/bug", "/exit", "/quit",
    "/init", "/login", "/logout", "/migrate-installer",
    "/release-notes", "/settings",
}
_GEMINI_SESSION_CMDS = {
    "/compress", "/rewind", "/chat", "/resume",
    "/about", "/stats", "/tools", "/theme",
    "/clear", "/exit", "/quit",
}
_KIMI_SESSION_CMDS = {
    "/clear", "/exit", "/quit", "/history", "/compact",
}
# DeepSeek has no CLI, but we recognize a small set so /d /clear gives a useful hint
# rather than the generic "unknown command" reply.
_DEEPSEEK_SESSION_CMDS = {
    "/clear", "/reset", "/history", "/exit", "/quit",
}


def _cmd_cli_native(chat_id: str, model: str, cmd: str, msg_id: str = None):
    """Handle /c /<cmd>, /g /<cmd>, or /k /<cmd>: passthrough channel for AI CLI native commands."""
    lang      = _get_lang(chat_id)
    cmd_lower = cmd.lower().strip()
    m_name    = {"claude": "Claude", "gemini": "Gemini", "kimi": "Kimi", "deepseek": "DeepSeek"}.get(model, model.capitalize())

    if cmd_lower in ("/help", "--help", "-h"):
        if model == "deepseek":
            body = _t(lang,
                "DeepSeek 走 HTTP API，没有官方 CLI 会话命令。\n\n"
                "**桥接内置：**\n"
                "`/reset deepseek` — 清空会话历史\n"
                "`/status` — 查看当前 model / session 长度\n"
                "\n"
                "如需切回其他模型：`/model claude` 或 `/model gemini` 等。",
                "DeepSeek uses HTTP API and has no official CLI session commands.\n\n"
                "**Bridge built-ins:**\n"
                "`/reset deepseek` — Clear session history\n"
                "`/status` — View current model / session length\n"
                "\n"
                "To switch back to another model: `/model claude` or `/model gemini` etc.",
            )
            send_card_reply(chat_id, msg_id, f"📖 {m_name}", body, color="blue")
            return
        if model == "kimi":
            body = _t(lang,
                "**会话管理**\n"
                "`/clear` — 清除对话历史，开始新会话\n"
                "`/history` — 查看历史会话列表\n"
                "`/compact` — 压缩上下文\n"
                "\n"
                "**其他**\n"
                "`/exit` / `/quit` — 退出 Kimi CLI\n"
                "\n"
                "_以上命令仅在终端交互模式下可用。_\n"
                "_发送 `/pickup` 获取终端接力命令。_",
                "**Session Management**\n"
                "`/clear` — Clear conversation history, start new session\n"
                "`/history` — View session history list\n"
                "`/compact` — Compact context\n"
                "\n"
                "**Other**\n"
                "`/exit` / `/quit` — Exit Kimi CLI\n"
                "\n"
                "_These commands are only available in terminal interactive mode._\n"
                "_Send `/pickup` to get terminal relay commands._",
            )
            send_card_reply(chat_id, msg_id,
                            _t(lang, f"📖 {m_name} CLI 交互命令", f"📖 {m_name} CLI Commands"),
                            body, color="blue")
            return
        if model == "claude":
            body = _t(lang,
                "**会话管理**\n"
                "`/compact [指令]` — 压缩对话历史，可选保留重点指令\n"
                "`/clear` — 清除对话历史，开始新会话\n"
                "`/cost` — 显示本次会话 token 用量和费用\n"
                "\n"
                "**记忆与配置**\n"
                "`/memory` — 查看和管理记忆文件（CLAUDE.md 等）\n"
                "`/model` — 切换当前使用的模型\n"
                "`/mcp` — 管理 MCP 服务器连接状态\n"
                "\n"
                "**工具**\n"
                "`/review` — 代码 / PR 审查\n"
                "`/pr_comments` — 查看当前 PR 的评论\n"
                "`/init` — 初始化项目（生成 CLAUDE.md）\n"
                "\n"
                "**终端**\n"
                "`/vim` — 切换 Vim 键位模式\n"
                "`/terminal-setup` — 配置终端集成\n"
                "\n"
                "**其他**\n"
                "`/doctor` — 检查环境和配置\n"
                "`/bug` — 报告问题\n"
                "`/exit` / `/quit` — 退出 Claude CLI\n"
                "\n"
                "_以上命令仅在终端交互模式下可用。_\n"
                "_发送 `/pickup` 获取终端接力命令。_",
                "**Session Management**\n"
                "`/compact [instructions]` — Compact conversation history, optionally keep key instructions\n"
                "`/clear` — Clear conversation history, start new session\n"
                "`/cost` — Show token usage and cost for this session\n"
                "\n"
                "**Memory & Config**\n"
                "`/memory` — View and manage memory files (CLAUDE.md etc.)\n"
                "`/model` — Switch current model\n"
                "`/mcp` — Manage MCP server connections\n"
                "\n"
                "**Tools**\n"
                "`/review` — Code / PR review\n"
                "`/pr_comments` — View comments on current PR\n"
                "`/init` — Initialize project (generate CLAUDE.md)\n"
                "\n"
                "**Terminal**\n"
                "`/vim` — Toggle Vim keybinding mode\n"
                "`/terminal-setup` — Configure terminal integration\n"
                "\n"
                "**Other**\n"
                "`/doctor` — Check environment and configuration\n"
                "`/bug` — Report an issue\n"
                "`/exit` / `/quit` — Exit Claude CLI\n"
                "\n"
                "_These commands are only available in terminal interactive mode._\n"
                "_Send `/pickup` to get terminal relay commands._",
            )
        else:
            body = _t(lang,
                "**会话管理**\n"
                "`/compress` — 将当前上下文压缩为摘要，节省 Token\n"
                "`/rewind [n]` — 撤销最近 n 条消息（默认 1 条）\n"
                "`/chat` / `/resume` — 浏览并恢复历史会话\n"
                "`/clear` — 清屏\n"
                "\n"
                "**信息**\n"
                "`/about` — 显示版本信息\n"
                "`/stats` — 显示会话统计（token 用量等）\n"
                "\n"
                "**工具与配置**\n"
                "`/tools` — 查看可用工具列表\n"
                "`/memory` — 管理记忆文件\n"
                "`/theme` — 切换终端主题\n"
                "\n"
                "**其他**\n"
                "`/help` / `/?` — 显示帮助\n"
                "`/exit` / `/quit` — 退出 Gemini CLI\n"
                "\n"
                "_以上命令仅在终端交互模式下可用。_\n"
                "_发送 `/pickup` 获取终端接力命令。_",
                "**Session Management**\n"
                "`/compress` — Compress current context into a summary to save tokens\n"
                "`/rewind [n]` — Undo last n messages (default 1)\n"
                "`/chat` / `/resume` — Browse and resume past sessions\n"
                "`/clear` — Clear screen\n"
                "\n"
                "**Info**\n"
                "`/about` — Show version info\n"
                "`/stats` — Show session statistics (token usage etc.)\n"
                "\n"
                "**Tools & Config**\n"
                "`/tools` — View available tools\n"
                "`/memory` — Manage memory files\n"
                "`/theme` — Switch terminal theme\n"
                "\n"
                "**Other**\n"
                "`/help` / `/?` — Show help\n"
                "`/exit` / `/quit` — Exit Gemini CLI\n"
                "\n"
                "_These commands are only available in terminal interactive mode._\n"
                "_Send `/pickup` to get terminal relay commands._",
            )
        send_card_reply(chat_id, msg_id,
                        _t(lang, f"📖 {m_name} CLI 交互命令", f"📖 {m_name} CLI Commands"),
                        body, color="blue")
        return

    _all_cmds = {"claude": _CLAUDE_SESSION_CMDS, "gemini": _GEMINI_SESSION_CMDS,
                 "kimi": _KIMI_SESSION_CMDS, "deepseek": _DEEPSEEK_SESSION_CMDS}
    _pfx_map  = {"claude": "/c", "gemini": "/g", "kimi": "/k", "deepseek": "/d"}
    my_cmds   = _all_cmds.get(model, set())
    my_pfx    = _pfx_map.get(model, "/c")
    # Collect all other models' commands for cross-model detection
    other_info: list[tuple[str, str, str]] = [
        (nm, pfx, cmds)
        for nm, (pfx, cmds) in {
            "Claude":   ("/c", _CLAUDE_SESSION_CMDS),
            "Gemini":   ("/g", _GEMINI_SESSION_CMDS),
            "Kimi":     ("/k", _KIMI_SESSION_CMDS),
            "DeepSeek": ("/d", _DEEPSEEK_SESSION_CMDS),
        }.items()
        if nm.lower() != model
    ]

    if cmd_lower in my_cmds:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "🖥️ 需要在终端执行", "🖥️ Requires Terminal"),
                        _t(lang,
                           f"`{cmd}` 是 {m_name} CLI 的会话命令。\n\n"
                           f"桥接使用 `--print` 模式运行，该模式不支持 CLI 会话命令。\n"
                           f"发送 `/pickup` 接入终端后可直接输入 `{cmd}`。",
                           f"`{cmd}` is a {m_name} CLI session command.\n\n"
                           f"The bridge runs in `--print` mode, which does not support CLI session commands.\n"
                           f"Send `/pickup` to connect to a terminal, then type `{cmd}` directly.",
                        ),
                        color="orange")
        return

    for _other_name, _other_pfx, _other_cmds in other_info:
        if cmd_lower in _other_cmds:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❓ 模型不匹配", "❓ Model Mismatch"),
                            _t(lang,
                               f"`{cmd}` 是 {_other_name} CLI 的命令，不是 {m_name} CLI 的命令。\n\n"
                               f"试试：`{_other_pfx} {cmd}`",
                               f"`{cmd}` is a {_other_name} CLI command, not a {m_name} CLI command.\n\n"
                               f"Try: `{_other_pfx} {cmd}`",
                            ),
                            color="orange")
            return

    send_card_reply(chat_id, msg_id,
                    _t(lang, "❓ 未知命令", "❓ Unknown Command"),
                    _t(lang,
                       f"`{cmd}` 不是已知的 {m_name} CLI 命令。\n\n"
                       f"- 查看 CLI 帮助：`{my_pfx} /help`\n"
                       f"- 执行 Shell 命令：`/run {cmd[1:]}`\n"
                       f"- 直接向 AI 提问：`{my_pfx} {cmd[1:]} 是什么`",
                       f"`{cmd}` is not a known {m_name} CLI command.\n\n"
                       f"- View CLI help: `{my_pfx} /help`\n"
                       f"- Run a shell command: `/run {cmd[1:]}`\n"
                       f"- Ask the AI directly: `{my_pfx} what is {cmd[1:]}`",
                    ),
                    color="orange")


def _cmd_btw(chat_id: str, question: str, user_msg_id: str, *, sender_open_id: str = ""):
    """Handle /btw side-question: uses main session if free, otherwise a dedicated btw session; replies in the original message thread."""
    lang      = _get_lang(chat_id)
    chat_lock = _get_chat_lock(chat_id)
    btw_lock  = _get_btw_lock(chat_id)

    main_free = chat_lock.acquire(blocking=False)
    if main_free:
        chat_lock.release()

    def _run():
        eyes_id = react_to_message(user_msg_id, EMOJI_PROCESSING)

        with btw_lock:
            cwd = _get_cwd(chat_id)

            from larkhelm.backend_registry import BACKEND_REGISTRY as _reg
            from larkhelm.backend_cli import (
                run_claude as _bc_run_claude,
                run_gemini as _bc_run_gemini,
                run_kimi as _bc_run_kimi,
            )
            import larkhelm.backend_api as _bapi

            # Resolve backend via registry (respects /model user preference)
            _state = _get_chat_state(chat_id)
            _pref = _state.get("backend_id") or _state.get("model") or ""
            _spec = (_reg.get(_pref) if _pref else None)
            if _spec is None or not _spec.healthy or not _spec.enabled:
                _spec = _reg.get_orchestrator()
            if _spec is None:
                raise RuntimeError("No backend available for /btw")

            # SID: reuse main session when free, else dedicated btw session
            sid = None
            if _spec.provider.endswith("_cli"):
                sid_key = _spec.id if main_free else f"btw_{_spec.id}"
                sid = _load_sid(chat_id, sid_key)

            _thinking_text = _t(lang, "> 正在思考...", "> Thinking...")
            mid = _reply_card_raw(
                user_msg_id,
                _make_card("💬", _thinking_text, color="grey"),
                in_thread=False,
            )

            try:
                cur_text = [""]

                def _on_text(text, status="typing"):
                    cur_text[0] = text
                    if mid:
                        _patch_card_raw(mid, _make_card("💬", text.strip() or _thinking_text, color="grey"))

                # Inject memory context (L2)
                # N-1 follow-up: migrate to v2 with ``query=question`` so
                # the S50 lazy global / S51 project conditional gating
                # actually fires on /btw side-questions (which are often
                # casual or domain-unrelated to the current project).
                # Closes the last v1 user-facing entry point — chat / dev /
                # crew / plan / btw now all share the same memory contract.
                _btw_mem_ctx = ""
                try:
                    from larkhelm.memory import get_memory_context_v2
                    # Phase D: synthesise a btw-flavoured IntentResult so the
                    # retriever (when enabled) picks the smaller btw policy
                    # (token_budget=800, prefer preference/context_summary)
                    # rather than the default chat policy.
                    _btw_mem_ctx, _ = get_memory_context_v2(
                        chat_id, cwd=cwd, query=question,
                        sender_open_id=sender_open_id,
                    )
                except Exception as e:
                    _debug_log(f"[btw] memory load failed: {e}")

                _API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api")
                if _spec.provider == "claude_cli":
                    _btw_msg = (f"[System]\n{_btw_mem_ctx}\n\n[User Query]\n{question}"
                                if _btw_mem_ctx else question)
                    output = _bc_run_claude(
                        spec=_spec, chat_id=chat_id, message=_btw_msg, sid=sid, cwd=cwd,
                        cancel_ev=None, on_text=_on_text, allow_retry=True,
                    )
                elif _spec.provider == "gemini_cli":
                    _btw_msg = (f"[System]\n{_btw_mem_ctx}\n\n[User Query]\n{question}"
                                if _btw_mem_ctx else question)
                    output = _bc_run_gemini(
                        spec=_spec, chat_id=chat_id, message=_btw_msg, sid=sid, cwd=cwd,
                        cancel_ev=None, on_text=_on_text,
                    )
                elif _spec.provider == "kimi_cli":
                    _btw_msg = (f"[System]\n{_btw_mem_ctx}\n\n[User Query]\n{question}"
                                if _btw_mem_ctx else question)
                    output = _bc_run_kimi(
                        spec=_spec, chat_id=chat_id, message=_btw_msg, sid=sid, cwd=cwd,
                        cancel_ev=None, on_text=_on_text, allow_retry=True,
                    )
                elif _spec.provider == "deepseek_api":
                    from larkhelm.backend_cli import run_deepseek as _bc_run_deepseek
                    output = _bc_run_deepseek(
                        spec=_spec, chat_id=chat_id, message=question, sid=None, cwd=cwd,
                        cancel_ev=None, on_text=_on_text,
                        system_prompt=_btw_mem_ctx or None,
                    )
                elif _spec.provider in _API_PROVIDERS:
                    _fn = {"anthropic_api": _bapi.run_anthropic,
                           "google_api": _bapi.run_google,
                           "openai_compat_api": _bapi.run_openai_compat}[_spec.provider]
                    output, _ = _fn(spec=_spec, chat_id=chat_id, message=question,
                                    history=[], on_text=_on_text,
                                    extra_system=_btw_mem_ctx if _btw_mem_ctx else "")
                else:
                    raise RuntimeError(f"Unknown provider for /btw: {_spec.provider}")

                final = (output or cur_text[0]).strip() or _t(lang, "（无输出）", "(no output)")
                final_card = _make_card("💬", final, color="blue")
                if mid:
                    _patch_card_raw(mid, final_card)
                    _register_btw_msg(chat_id, mid)
                else:
                    new_mid = _reply_card_raw(user_msg_id, final_card, in_thread=False)
                    _register_btw_msg(chat_id, new_mid)

                if eyes_id:
                    delete_reaction(user_msg_id, eyes_id)
                react_to_message(user_msg_id, EMOJI_DONE)

            except Exception as e:
                err_card = _make_card(_t(lang, "❌ 旁注失败", "❌ /btw Failed"), str(e)[:200], color="red")
                if mid:
                    _patch_card_raw(mid, err_card)
                else:
                    _reply_card_raw(user_msg_id, err_card, in_thread=False)
                if eyes_id:
                    delete_reaction(user_msg_id, eyes_id)
                react_to_message(user_msg_id, EMOJI_ERROR)

    threading.Thread(target=_run, daemon=True, name=f"btw-{chat_id[:8]}").start()


# ═══════════════════════════════════════════════════
#  /memory command
# ═══════════════════════════════════════════════════

def _render_observe_card(obs: dict) -> tuple[str, str, str]:
    """Build (title, body_markdown, color) for the ``/memory observe`` card.

    Color: ``orange`` if any layer.near_limit, else ``blue``. Sections (固定顺序):
      1. 三层容量条 (含百分比 + ⚠️)
      2. 最近成功摘要时间
      3. 近 7 天 UNCHANGED ``m/n (X%)``
      4. 近 7 天记忆摘要降级到主模型次数（低成本模型失败）
    """
    layers = obs.get("layers", {}) or {}
    any_near = any((layers.get(k) or {}).get("near_limit") for k in ("global", "project", "session"))
    color = "orange" if any_near else "blue"

    def _meter(layer: dict | None, label: str) -> str:
        if not layer:
            return f"- **{label}**: _(无)_"
        chars = layer.get("chars", 0)
        m = layer.get("max_chars", 0)
        pct = layer.get("pct", 0)
        suffix = " ⚠️ near limit" if layer.get("near_limit") else ""
        return f"- **{label}**: `{chars}/{m} chars · {pct}%`{suffix}"

    lines = ["**容量**"]
    lines.append(_meter(layers.get("global"),  "全局"))
    lines.append(_meter(layers.get("project"), "项目"))
    lines.append(_meter(layers.get("session"), "会话"))

    last_ok = obs.get("last_successful_update") or "_(未知)_"
    lines.append("")
    lines.append(f"**上次成功摘要**: `{last_ok}`")

    # Pruning summary (per design v1.0 §5.3). Sourced from log._pruning_stats
    # via memory._aggregate_memory_observation; rendered as a percentage +
    # window size, or `_(尚无样本)_` when the ring buffer is empty.
    # Note: _pruning_stats is a per-process ring buffer — after a bridge
    # restart the window stays at 0 until the next few queries refill it.
    pr = obs.get("pruning") or {}
    if pr.get("unavailable") or int(pr.get("window", 0)) == 0:
        lines.append("**Pruning**: _(尚无样本，重启后需累计若干轮)_")
    else:
        lines.append(
            f"**Pruning**: `saved={int(pr.get('saved_pct', 0))}% "
            f"(last {int(pr['window'])} calls)`"
        )

    rw = obs.get("recent_window", {}) or {}
    if rw.get("unavailable"):
        lines.append(f"**近 {rw.get('window_days', 7)} 天 UNCHANGED**: _(不可用)_")
    else:
        m = rw.get("unchanged_count", 0)
        n = rw.get("total_count", 0)
        pct = int(round((rw.get("unchanged_ratio", 0.0) * 100)))
        lines.append(f"**近 {rw.get('window_days', 7)} 天 UNCHANGED**: `{m}/{n}` ({pct}%)")

    fb = obs.get("fallback", {}) or {}
    if fb.get("unavailable"):
        lines.append(f"**近 {fb.get('window_days', 7)} 天 记忆摘要降级到主模型**: _(不可用)_")
    else:
        k = fb.get("count", 0)
        pct = int(round((fb.get("ratio", 0.0) * 100)))
        ts = fb.get("last_ts")
        ts_suffix = f"（最近 `{ts}`）" if ts else ""
        lines.append(f"**近 {fb.get('window_days', 7)} 天 记忆摘要降级到主模型**: `{k} 次` ({pct}%){ts_suffix}")

    return "🔍 记忆观测", "\n".join(lines), color


def _cmd_memory_diagnose(chat_id: str, args: str = "", msg_id: str = None) -> None:
    """/memory diagnose — retriever audit log (feature removed)."""
    send_card_reply(chat_id, msg_id, "🔍 记忆诊断",
                    "召回审计功能已移除（retriever 基础设施已精简）。", color="grey")


def _cmd_memory_set_project_guide(chat_id: str, args: str, msg_id=None) -> None:
    """/memory set project_guide <auto|off|path <path>> — hot-update project guide config."""
    sub = (args or "").strip()
    if sub == "auto":
        _cfg.config["project_guide_enabled"] = True
        _cfg.config["project_guide_auto_discover"] = True
        _cfg.config["project_guide_path"] = ""
        _cfg.PROJECT_GUIDE_ENABLED = True
        _cfg.PROJECT_GUIDE_AUTO_DISCOVER = True
        _cfg.PROJECT_GUIDE_PATH = ""
        send_card_reply(
            chat_id, msg_id, "✅ Project Guide",
            "已开启自动发现模式（从 cwd 查找 CLAUDE.md / .larkhelm_project.md）",
            color="green",
        )
        return
    if sub == "off":
        _cfg.config["project_guide_enabled"] = False
        _cfg.config["project_guide_auto_discover"] = False
        _cfg.config["project_guide_path"] = ""
        _cfg.PROJECT_GUIDE_ENABLED = False
        _cfg.PROJECT_GUIDE_AUTO_DISCOVER = False
        _cfg.PROJECT_GUIDE_PATH = ""
        send_card_reply(
            chat_id, msg_id, "✅ Project Guide",
            "已关闭 Project Guide 注入",
            color="green",
        )
        return
    if sub.startswith("path "):
        path_str = sub[5:].strip()
        try:
            p = Path(path_str).expanduser().resolve()
        except Exception as _pe:
            send_card_reply(chat_id, msg_id, "⚠️ 路径错误", f"无法解析路径：{path_str}", color="orange")
            return
        if not p.exists():
            send_card_reply(chat_id, msg_id, "⚠️ 路径不存在", f"`{p}` 不存在", color="orange")
            return
        try:
            data_dir = _cfg.DATA_DIR
            if str(p).startswith(str(data_dir)):
                send_card_reply(
                    chat_id, msg_id, "⚠️ 路径限制",
                    "Project Guide 路径不能位于 DATA_DIR 内", color="orange",
                )
                return
        except Exception:
            pass
        _cfg.config["project_guide_enabled"] = True
        _cfg.config["project_guide_auto_discover"] = False
        _cfg.config["project_guide_path"] = str(p)
        _cfg.PROJECT_GUIDE_ENABLED = True
        _cfg.PROJECT_GUIDE_AUTO_DISCOVER = False
        _cfg.PROJECT_GUIDE_PATH = str(p)
        send_card_reply(
            chat_id, msg_id, "✅ Project Guide",
            f"已设置路径：`{p}`", color="green",
        )
        return
    send_card_reply(
        chat_id, msg_id, "⚠️ 用法",
        "`/memory set project_guide auto` — 自动发现模式\n"
        "`/memory set project_guide off` — 关闭\n"
        "`/memory set project_guide path <路径>` — 指定文件",
        color="orange",
    )


def _cmd_memory(chat_id: str, args: str = "", msg_id: str = None, *, sender_open_id: str = ""):
    """/memory — show/set/clear/update/gc/export/import/status the three-tier memory system.

    /memory                        show all active layers
    /memory set global <text>      overwrite global layer (≤500 chars)
    /memory set project <text>     overwrite project layer for current cwd (≤1000 chars)
    /memory clear [global|project|session]   clear one or all layers
    /memory update                 force-regenerate session layer from logs
    /memory list                   list every project_*.md file
    /memory gc [days] [apply]      clean up stale project memory (dry-run by
                                   default; ``days`` defaults to 30, must ≥ 1)
    /memory export                 export current chat persistent data as a zip file
    /memory import [file_key]      import from a zip file (reply with file or pass file_key)
    /memory status                 show summary of persistent state sizes
    """
    from larkhelm.memory import (
        load_global_memory, save_global_memory,
        load_project_memory, save_project_memory,
        load_memory, save_memory, maybe_auto_update,
        _global_memory_file, _project_memory_file, _session_memory_file,
        _load_md_frontmatter, _load_md_body, _ensure_dir, MEMORY_HOME_DIR,
        GLOBAL_MAX_CHARS, PROJECT_MAX_CHARS, AUTO_UPDATE_EVERY,
    )
    cwd = _get_cwd(chat_id)
    args = args.strip()
    sub = args.lower()
    lang = _get_lang(chat_id)

    # ── /memory export ───────────────────────────────────────────────────────
    if sub == "export":
        mid = send_card_reply(chat_id, msg_id,
                              _t(lang, "📦 导出中", "📦 Exporting"),
                              _t(lang, "正在打包记忆数据…", "Packaging memory data…"),
                              color="grey")
        try:
            from larkhelm.memory_io import export_memory
            zip_path = export_memory(chat_ids=[chat_id])
            file_key = upload_file_to_feishu(zip_path, file_type="stream")
            if file_key:
                send_file_message(chat_id, file_key, msg_id=msg_id)
                # Clean up the temporary export zip after successful upload
                try:
                    zip_path.unlink(missing_ok=True)
                except Exception:
                    pass
                if mid:
                    _patch_card_raw(mid, _make_card(
                        _t(lang, "✅ 导出完成", "✅ Export Complete"),
                        _t(lang,
                           f"文件已发送（{zip_path.name}）。",
                           f"File sent ({zip_path.name})."),
                        color="green"))
            else:
                if mid:
                    _patch_card_raw(mid, _make_card(
                        _t(lang, "❌ 上传失败", "❌ Upload Failed"),
                        _t(lang,
                           "文件生成成功但上传到飞书失败，请查看日志。",
                           "File generated but upload to Feishu failed — check logs."),
                        color="red"))
        except Exception as e:
            _debug_log(f"[memory export] failed: {e}")
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 导出失败", "❌ Export Failed"),
                            str(e)[:300], color="red")
        return

    # ── /memory import ───────────────────────────────────────────────────────
    if sub.startswith("import"):
        rest = args[6:].strip()  # len("import") == 6
        if rest:
            # Direct import by file_key
            file_key = rest
            mid = send_card_reply(chat_id, msg_id,
                                  _t(lang, "📥 导入中", "📥 Importing"),
                                  _t(lang, "正在下载并导入记忆数据…", "Downloading and importing memory data…"),
                                  color="grey")
            try:
                import tempfile
                from larkhelm.memory_io import import_memory
                _fd, _tmp = tempfile.mkstemp(suffix=".zip", prefix=f"larkhelm_import_{chat_id[:8]}_")
                os.close(_fd)
                tmp_path = Path(_tmp)
                ok = download_file_by_key(file_key, tmp_path)
                if not ok:
                    if mid:
                        _patch_card_raw(mid, _make_card(
                            _t(lang, "❌ 下载失败", "❌ Download Failed"),
                            _t(lang,
                               "无法通过 file_key 下载文件，请确认 key 正确。",
                               "Cannot download file via file_key — check the key."),
                            color="red"))
                    return
                report = import_memory(tmp_path)
                n_written = len(report["written"])
                n_skipped = len(report["skipped"])
                lines = [_t(lang,
                            f"**导入成功：** {n_written} 个文件",
                            f"**Imported:** {n_written} files")]
                if n_skipped:
                    lines.append(_t(lang,
                                    f"**跳过：** {n_skipped} 个文件",
                                    f"**Skipped:** {n_skipped} files"))
                if report.get("warnings"):
                    lines.append(_t(lang,
                                    f"**警告：** {'；'.join(report['warnings'])}",
                                    f"**Warnings:** {'; '.join(report['warnings'])}"))
                color = "green" if not n_skipped else "orange"
                body = "\n\n".join(lines)
                if mid:
                    _patch_card_raw(mid, _make_card(
                        _t(lang, "✅ 导入完成", "✅ Import Complete"), body, color=color))
                else:
                    send_card_reply(chat_id, msg_id,
                                    _t(lang, "✅ 导入完成", "✅ Import Complete"),
                                    body, color=color)
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as e:
                _debug_log(f"[memory import] failed: {e}")
                if mid:
                    _patch_card_raw(mid, _make_card(
                        _t(lang, "❌ 导入失败", "❌ Import Failed"),
                        str(e)[:300], color="red"))
                else:
                    send_card_reply(chat_id, msg_id,
                                    _t(lang, "❌ 导入失败", "❌ Import Failed"),
                                    str(e)[:300], color="red")
            return
        else:
            # Prompt user to reply with a file or pass file_key
            # Store timestamp so the pending flag auto-expires after 10 minutes
            import time as _time
            _set_chat_field(chat_id, "pending_memory_import", _time.time())
            send_card_reply(
                chat_id, msg_id,
                _t(lang, "📥 等待导入", "📥 Awaiting Import"),
                _t(lang,
                   "请直接回复发送 zip 文件，或发送：\n\n"
                   "`/memory import <file_key>`\n\n"
                   "_提示：file_key 可在文件消息的原始内容中获取。_",
                   "Please upload a memory zip file (generated by `/memory export`)\n\n"
                   "Or send: `/memory import <file_key>`"),
                color="blue")
            return

    # ── /memory status ───────────────────────────────────────────────────────
    if sub == "status":
        try:
            from larkhelm.memory_io import get_memory_status
            st = get_memory_status(chat_id)
            lines = [
                _t(lang, f"**Chat 数量：** {st['n_chats']}", f"**Chats:** {st['n_chats']}"),
                _t(lang, f"**Session ID 数：** {st['n_sessions']}", f"**Session IDs:** {st['n_sessions']}"),
                _t(lang,
                   f"**API 会话历史：** {st['n_api_sessions']} 个（{st['api_session_size'] // 1024} KB）",
                   f"**API session history:** {st['n_api_sessions']} ({st['api_session_size'] // 1024} KB)"),
                _t(lang,
                   f"**日志总大小：** {st['log_size'] // 1024} KB",
                   f"**Log size:** {st['log_size'] // 1024} KB"),
                _t(lang,
                   f"**数据总大小：** {st['data_size'] // 1024} KB",
                   f"**Data size:** {st['data_size'] // 1024} KB"),
                _t(lang,
                   f"**记忆文件数：** {st['memory_files']}（{st['memory_size'] // 1024} KB）",
                   f"**Memory files:** {st['memory_files']} ({st['memory_size'] // 1024} KB)"),
            ]
            # Cron health — independent of prometheus; reads chat_state.
            # Lock window only covers the shallow copy; aggregation runs
            # lock-free (PRD §4 non-functional).
            try:
                from larkhelm.chat_state import (
                    _state_lock as _obs_state_lock,
                    _chat_state_store as _obs_store,
                )
                with _obs_state_lock:
                    snapshot = [
                        dict(entry)
                        for st_dict in _obs_store.values()
                        for entry in st_dict.get("crons", [])
                    ]
                ok = err = never = 0
                for entry in snapshot:
                    status_ = entry.get("last_run_status", "")
                    if status_ == "ok":
                        ok += 1
                    elif status_ == "error":
                        err += 1
                    else:
                        never += 1
                total_cron = ok + err + never
                if total_cron == 0:
                    lines.append(_t(lang, "**Cron 健康度：** 无 cron 任务", "**Cron health:** no cron tasks"))
                else:
                    lines.append(_t(lang,
                                    f"**Cron 健康度：** ✅{ok} · ❌{err} · ❓{never}（共 {total_cron} 条）",
                                    f"**Cron health:** ✅{ok} · ❌{err} · ❓{never} ({total_cron} total)"))
            except Exception as inner:
                _debug_log(f"[memory status] cron health line failed: {inner}")
                lines.append(_t(lang, "**Cron 健康度：** n/a", "**Cron health:** n/a"))

            if st['chats']:
                lines.append(_t(lang, "\n**各 Chat 摘要：**", "\n**Chat summary:**"))
                for c in st['chats'][:10]:
                    lines.append(_t(lang,
                                    f"• `{c['chat_id']}` · {c['model']} · {c['turn_count']} 轮",
                                    f"• `{c['chat_id']}` · {c['model']} · {c['turn_count']} turns"))
                if len(st['chats']) > 10:
                    lines.append(_t(lang,
                                    f"_… 共 {len(st['chats'])} 个 chat_",
                                    f"_… {len(st['chats'])} chats total_"))
            send_card_reply(chat_id, msg_id,
                            _t(lang, "📊 记忆状态", "📊 Memory Status"),
                            "\n".join(lines), color="blue")
        except Exception as e:
            _debug_log(f"[memory status] failed: {e}")
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 查询失败", "❌ Query Failed"),
                            str(e)[:300], color="red")
        return

    # ── /memory diagnose [N] ─────────────────────────────────────────────────
    if sub == "diagnose" or sub.startswith("diagnose "):
        rest = args[len("diagnose"):].strip()
        _cmd_memory_diagnose(chat_id, rest, msg_id=msg_id)
        return

    # ── /memory observe ──────────────────────────────────────────────────────
    if sub == "observe":
        try:
            from larkhelm.memory import _aggregate_memory_observation
            obs = _aggregate_memory_observation(chat_id, sender_open_id=sender_open_id)
            title, body, color = _render_observe_card(obs)
            send_card_reply(chat_id, msg_id, title, body, color=color, normalize=False)
        except Exception as e:
            _debug_log(f"[Memory] observe failed: {e}")
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 观测失败", "❌ Observe Failed"),
                            str(e)[:300], color="red")
        return

    # ── /memory set global <text> ────────────────────────────────────────────
    if sub.startswith("set global ") or sub == "set global":
        text = args[11:].strip()  # len("set global ") == 11
        if not text:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⚠️ 用法", "⚠️ Usage"),
                            _t(lang,
                               "`/memory set global <内容>` — 设置全局记忆（最多 500 字符）",
                               "`/memory set global <text>` — set global memory (max 500 chars)"),
                            color="orange")
            return
        if _global_memory_file(chat_id, sender_open_id=sender_open_id) is None:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⚠️ 全局记忆不可用", "⚠️ Global Memory Unavailable"),
                            _t(lang,
                               "当前会话无法识别发送者身份，全局记忆已跳过（群聊安全保护）。",
                               "Sender identity cannot be determined in this chat — global memory skipped (group chat safety)."),
                            color="orange")
            return
        save_global_memory(text, chat_id=chat_id, sender_open_id=sender_open_id)
        send_card_reply(chat_id, msg_id,
                        _t(lang, "✅ 全局记忆已更新", "✅ Global Memory Updated"),
                        f"```\n{text[:200]}\n```", color="green")
        return

    # ── /memory set project_guide ... ───────────────────────────────────────
    if sub.startswith("set project_guide"):
        _pg_args = sub[len("set project_guide"):].strip()
        _cmd_memory_set_project_guide(chat_id, _pg_args, msg_id)
        return

    # ── /memory set project <text> ───────────────────────────────────────────
    if sub.startswith("set project ") or sub == "set project":
        text = args[12:].strip()  # len("set project ") == 12
        if not text:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⚠️ 用法", "⚠️ Usage"),
                            _t(lang,
                               f"`/memory set project <内容>` — 设置当前项目记忆（cwd: `{cwd}`，最多 1000 字符）",
                               f"`/memory set project <text>` — set project memory for current cwd: `{cwd}` (max 1000 chars)"),
                            color="orange")
            return
        save_project_memory(cwd, text)
        send_card_reply(chat_id, msg_id,
                        _t(lang, "✅ 项目记忆已更新", "✅ Project Memory Updated"),
                        _t(lang,
                           f"**目录**: `{cwd}`\n\n```\n{text[:200]}\n```",
                           f"**Directory**: `{cwd}`\n\n```\n{text[:200]}\n```"),
                        color="green")
        return

    # ── /memory clear <layer> ────────────────────────────────────────────────
    if sub.startswith("clear"):
        layer = args[5:].strip().lower()  # "all" | "global" | "project" | "session"
        if not layer:
            layer = "session"  # default: clear only session layer (safest)
        deleted = []
        try:
            if layer in ("all", "session"):
                _session_memory_file(chat_id).unlink(missing_ok=True)
                deleted.append(_t(lang, "会话", "session"))
            if layer in ("all", "global"):
                _gf = _global_memory_file(chat_id, sender_open_id=sender_open_id)
                if _gf:
                    _gf.unlink(missing_ok=True)
                deleted.append(_t(lang, "全局", "global"))
            if layer in ("all", "project"):
                _project_memory_file(cwd).unlink(missing_ok=True)
                deleted.append(_t(lang, "项目", "project"))
            if not deleted:
                send_card_reply(chat_id, msg_id,
                                _t(lang, "⚠️ 未知层级", "⚠️ Unknown Layer"),
                                _t(lang,
                                   "可选: `global` · `project` · `session` · `all`",
                                   "Options: `global` · `project` · `session` · `all`"),
                                color="orange")
                return
            if layer == "session":
                detail = _t(lang,
                            "已清除会话记忆，全局和项目记忆已保留。",
                            "Session memory cleared; global and project memory retained.")
            else:
                detail = _t(lang,
                            f"已删除：{'、'.join(deleted)}记忆。",
                            f"Deleted: {', '.join(deleted)} memory.")
            send_card_reply(chat_id, msg_id,
                            _t(lang, "🗑️ 已清除", "🗑️ Cleared"),
                            detail, color="green")
        except Exception as e:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 清除失败", "❌ Clear Failed"),
                            str(e)[:200], color="red")
        return

    # ── /memory update ───────────────────────────────────────────────────────
    if sub == "update":
        update_mid = send_card_reply(
            chat_id, msg_id,
            _t(lang, "🔄 生成记忆中", "🔄 Generating Memory"),
            _t(lang,
               "正在后台生成会话摘要，预计 10~30 秒，请稍候…",
               "Generating session summary in background, ~10–30 s — please wait…"),
            color="grey")

        _ERR_MSG_MAP_ZH = {
            "no_logs":              "当前会话暂无对话记录，无法生成摘要。",
            "no_conversation_logs": "当前会话暂无普通对话（只有系统/Shell 记录），无法生成摘要。",
        }
        _ERR_MSG_MAP_EN = {
            "no_logs":              "No conversation logs found — cannot generate summary.",
            "no_conversation_logs": "No regular conversation found (only system/shell records) — cannot generate summary.",
        }

        def _on_update_done(success: bool, content, error):
            if success and content:
                preview = content[:200]
                new_card = _make_card(
                    _t(lang, "✅ 会话记忆已更新", "✅ Session Memory Updated"),
                    f"```\n{preview}\n```", color="green")
            else:
                if error and error.startswith("timed_out_"):
                    secs = error.replace("timed_out_", "").replace("s", "")
                    err_msg = _t(lang,
                                 f"记忆生成超时（>{secs}s），请稍后重试。",
                                 f"Memory generation timed out (>{secs}s) — please retry.")
                else:
                    _err_map = _ERR_MSG_MAP_EN if lang == "en" else _ERR_MSG_MAP_ZH
                    _fallback = _t(lang, "记忆生成失败，请稍后重试。", "Memory generation failed — please retry.")
                    err_msg = _err_map.get(error or "", _fallback)
                new_card = _make_card(
                    _t(lang, "❌ 记忆生成失败", "❌ Memory Generation Failed"),
                    err_msg, color="red")
            if update_mid:
                _patch_card_raw(update_mid, new_card)
            else:
                send_card_reply(
                    chat_id, msg_id,
                    _t(lang, "✅ 会话记忆已更新", "✅ Session Memory Updated") if success
                    else _t(lang, "❌ 记忆生成失败", "❌ Memory Generation Failed"),
                    f"```\n{content[:200]}\n```" if success else err_msg,
                    color="green" if success else "red")

        maybe_auto_update(chat_id, force=True, on_done=_on_update_done)
        return

    # ── /memory gc [days] [apply] — clean up stale project memory ────────────
    # Examples:
    #   /memory gc            → dry-run with default 30-day threshold
    #   /memory gc 60         → dry-run with 60-day threshold
    #   /memory gc apply      → actually delete (default 30 days)
    #   /memory gc 60 apply   → actually delete with 60-day threshold
    if sub == "gc" or sub.startswith("gc ") or sub.startswith("gc\t"):
        from larkhelm.memory import gc_project_memory, _GC_DEFAULT_DAYS
        # Parse args: any int token = threshold_days; any "apply" token = apply mode.
        # Robust to either order so users don't have to remember which goes first.
        rest = args[2:].strip().split()
        threshold_days = _GC_DEFAULT_DAYS
        apply = False
        bad_tokens: list[str] = []
        for tok in rest:
            tl = tok.lower()
            if tl in ("apply", "--apply", "-y", "yes", "force"):
                apply = True
                continue
            try:
                threshold_days = int(tl)
            except ValueError:
                bad_tokens.append(tok)
        if bad_tokens:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⚠️ 用法", "⚠️ Usage"),
                            _t(lang,
                               f"无法识别参数：{' '.join(bad_tokens)}\n\n"
                               f"`/memory gc [天数] [apply]` — 清理 N 天未更新的项目记忆\n"
                               f"- `/memory gc` — 30 天 dry-run（默认，不删除）\n"
                               f"- `/memory gc 60` — 60 天 dry-run\n"
                               f"- `/memory gc apply` — 实际删除（30 天）\n"
                               f"- `/memory gc 60 apply` — 实际删除（60 天）",
                               f"Unrecognized argument(s): {' '.join(bad_tokens)}\n\n"
                               f"`/memory gc [days] [apply]` — clean up project memory not updated in N days\n"
                               f"- `/memory gc` — 30-day dry-run (default, no deletion)\n"
                               f"- `/memory gc 60` — 60-day dry-run\n"
                               f"- `/memory gc apply` — actually delete (30 days)\n"
                               f"- `/memory gc 60 apply` — actually delete (60 days)"),
                            color="orange")
            return
        if threshold_days < 1:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "⚠️ 阈值过低", "⚠️ Threshold Too Low"),
                            _t(lang,
                               "天数必须 ≥ 1（防止误清空所有项目记忆）。",
                               "Days must be ≥ 1 (prevents accidentally clearing all project memory)."),
                            color="orange")
            return

        try:
            report = gc_project_memory(threshold_days=threshold_days, apply=apply)
        except Exception as e:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ GC 失败", "❌ GC Failed"),
                            str(e)[:300], color="red")
            return

        scanned = report["scanned"]
        cands = report["candidates"]
        errs = report["errors"]
        if not cands:
            send_card_reply(
                chat_id, msg_id,
                _t(lang,
                   f"🧹 项目记忆 GC（>{threshold_days} 天）",
                   f"🧹 Project Memory GC (>{threshold_days} days)"),
                _t(lang,
                   f"扫描 {scanned} 个文件，**无可清理项**。",
                   f"Scanned {scanned} files — **nothing to clean up**."),
                color="green",
            )
            return

        # Build display: one row per candidate, capped to keep card readable.
        _MAX_ROWS = 30
        lines = []
        for c in cands[:_MAX_ROWS]:
            tag = "🗑️" if c["deleted"] else ("⚠️" if apply else "💤")
            cwd_disp = c["cwd"] or _t(lang, "_(无 cwd 元数据)_", "_(no cwd metadata)_")
            age_disp = f"{c['age_days']}d" if c["age_days"] is not None else "?"
            lines.append(
                f"{tag} `{c['name']}` · "
                + _t(lang, f"龄期 {age_disp}", f"age {age_disp}")
                + f" · `{c['reason']}`\n"
                f"   `{cwd_disp}`"
            )
        if len(cands) > _MAX_ROWS:
            lines.append(_t(lang,
                            f"_… 另有 {len(cands) - _MAX_ROWS} 个候选未列出_",
                            f"_… {len(cands) - _MAX_ROWS} more candidates not shown_"))
        if errs:
            lines.append(_t(lang, "\n**错误**：", "\n**Errors:**"))
            for e in errs[:5]:
                lines.append(f"- `{Path(e['path']).name}` — {e['err'][:120]}")
            if len(errs) > 5:
                lines.append(_t(lang,
                                f"_… 另有 {len(errs) - 5} 个错误_",
                                f"_… {len(errs) - 5} more errors_"))

        if apply:
            n_deleted = sum(1 for c in cands if c["deleted"])
            title = _t(lang,
                       f"🧹 项目记忆 GC 已执行（>{threshold_days} 天）",
                       f"🧹 Project Memory GC Done (>{threshold_days} days)")
            header = _t(lang,
                        f"扫描 {scanned}，已删除 **{n_deleted}**，错误 {len(errs)}\n\n",
                        f"Scanned {scanned}, deleted **{n_deleted}**, errors {len(errs)}\n\n")
            color = "green" if n_deleted and not errs else "orange"
        else:
            title = _t(lang,
                       f"🧹 项目记忆 GC · 预演（>{threshold_days} 天）",
                       f"🧹 Project Memory GC · Dry Run (>{threshold_days} days)")
            header = _t(lang,
                        f"扫描 {scanned}，发现 **{len(cands)}** 个候选。\n"
                        f"_这是预演，未删除任何文件。要实际清理：_\n"
                        f"`/memory gc {threshold_days} apply`\n\n",
                        f"Scanned {scanned}, found **{len(cands)}** candidates.\n"
                        f"_This is a dry run — no files deleted. To actually clean up:_\n"
                        f"`/memory gc {threshold_days} apply`\n\n")
            color = "blue"

        send_card_reply(chat_id, msg_id, title,
                        header + "\n\n".join(lines),
                        color=color, normalize=False)
        return

    # ── /memory list — show all project memory files ─────────────────────────
    if sub == "list":
        _ensure_dir()
        project_files = sorted(MEMORY_HOME_DIR.glob("project_*.md"))
        if not project_files:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "📂 项目记忆", "📂 Project Memory"),
                            _t(lang, "_暂无项目记忆文件_", "_No project memory files_"),
                            color="blue")
            return
        lines = []
        for pf in project_files:
            fm = _load_md_frontmatter(pf)
            stored_cwd = fm.get("cwd", "?")
            updated = (fm.get("updated_at", "?") or "?")[:10]
            body = _load_md_body(pf) or ""
            _stored_resolved = str(Path(stored_cwd).resolve()) if stored_cwd else ""
            _cwd_resolved = str(Path(cwd).resolve()) if cwd else ""
            mark = "📍" if _stored_resolved and _stored_resolved == _cwd_resolved else "📁"
            lines.append(
                f"{mark} **{pf.stem}**\n`{stored_cwd}`  "
                + _t(lang,
                     f"更新: {updated}  大小: {len(body)} 字符",
                     f"updated: {updated}  size: {len(body)} chars")
            )
        send_card_reply(chat_id, msg_id,
                        _t(lang,
                           f"📂 项目记忆（{len(project_files)} 个）",
                           f"📂 Project Memory ({len(project_files)} files)"),
                        "\n\n---\n\n".join(lines), color="blue", normalize=False)
        return

    # ── /memory — show all layers ────────────────────────────────────────────
    from larkhelm.chat_state import _get_turn_count as _gtc
    _turn = _gtc(chat_id)

    def _fm_meta(path) -> str:
        if path is None:
            return ""
        fm = _load_md_frontmatter(path)
        ts = (fm.get("updated_at") or "")[:16].replace("T", " ")
        return _t(lang, f" _(更新: {ts})_", f" _(updated: {ts})_") if ts else ""

    g = load_global_memory(chat_id, sender_open_id=sender_open_id)
    p = load_project_memory(cwd)
    s = load_memory(chat_id)

    _g_path = _global_memory_file(chat_id, sender_open_id=sender_open_id)
    _p_path = _project_memory_file(cwd)
    _s_path = _session_memory_file(chat_id)

    sections: list[str] = []
    if g:
        sections.append(
            _t(lang, f"### 🌐 全局记忆{_fm_meta(_g_path)}", f"### 🌐 Global Memory{_fm_meta(_g_path)}")
            + f"\n{g}")
    else:
        sections.append(_t(lang,
                           "### 🌐 全局记忆\n_（空）_ — `/memory set global <内容>` 设置",
                           "### 🌐 Global Memory\n_(empty)_ — set with `/memory set global <text>`"))

    if p:
        sections.append(
            _t(lang,
               f"### 📁 项目记忆 (`{cwd}`){_fm_meta(_p_path)}",
               f"### 📁 Project Memory (`{cwd}`){_fm_meta(_p_path)}")
            + f"\n{p}")
    else:
        sections.append(_t(lang,
                           f"### 📁 项目记忆 (`{cwd}`)\n_（空）_ — `/memory set project <内容>` 设置",
                           f"### 📁 Project Memory (`{cwd}`)\n_(empty)_ — set with `/memory set project <text>`"))

    if s:
        _next_turn = _turn + (AUTO_UPDATE_EVERY - (_turn % AUTO_UPDATE_EVERY) if _turn % AUTO_UPDATE_EVERY else AUTO_UPDATE_EVERY)
        sections.append(
            _t(lang,
               f"### 💬 会话记忆{_fm_meta(_s_path)} _(当前第 {_turn} 轮，第 {_next_turn} 轮时自动更新，可 `/memory update` 立即触发)_",
               f"### 💬 Session Memory{_fm_meta(_s_path)} _(turn {_turn}, auto-update at turn {_next_turn}, or `/memory update` now)_")
            + f"\n{s}")
    else:
        sections.append(_t(lang,
                           f"### 💬 会话记忆 _(当前第 {_turn} 轮)_\n_（空）_ — 每 {AUTO_UPDATE_EVERY} 轮自动生成，或 `/memory update` 立即生成",
                           f"### 💬 Session Memory _(turn {_turn})_\n_(empty)_ — auto-generated every {AUTO_UPDATE_EVERY} turns, or `/memory update` now"))

    body = "\n\n---\n\n".join(sections)
    body += _t(lang,
               "\n\n---\n`/memory set global|project <内容>` 写入 · "
               "`/memory clear session|project|global|all` 清除 · "
               "`/memory update` 更新会话 · `/memory list` 查看项目记忆文件",
               "\n\n---\n`/memory set global|project <text>` write · "
               "`/memory clear session|project|global|all` clear · "
               "`/memory update` update session · `/memory list` list project memory files")
    send_card_reply(chat_id, msg_id,
                    _t(lang, "🧠 三层记忆系统", "🧠 Memory System"),
                    body, color="blue", normalize=False)


# ═══════════════════════════════════════════════════
#  Button callback dispatch
# ═══════════════════════════════════════════════════

def _dispatch_button_cmd(chat_id: str, cmd: str):
    """Handle card button clicks (non-permission-approval buttons)."""
    tl = cmd.lower().strip()
    if tl in ("reset", "/reset"):
        _cmd_reset(chat_id, None)
    elif tl in ("reset claude", "/reset claude"):
        _cmd_reset(chat_id, "claude")
    elif tl in ("reset gemini", "/reset gemini"):
        _cmd_reset(chat_id, "gemini")
    elif tl in ("reset kimi", "/reset kimi"):
        _cmd_reset(chat_id, "kimi")
    elif tl in ("reset deepseek", "/reset deepseek"):
        _cmd_reset(chat_id, "deepseek")
    elif tl in ("reset permissions", "/reset permissions", "reset perm", "/reset perm"):
        _cmd_reset(chat_id, "perm")
    elif tl in ("status", "/status"):
        _cmd_status(chat_id)
    elif tl in ("help", "/help"):
        _cmd_help(chat_id)
    elif tl in ("/pickup", "pickup"):
        _cmd_pickup(chat_id)
    elif tl in ("/cancel", "cancel"):
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
        send_card(chat_id, "🛑 已取消", body, color="orange")
    elif tl.startswith("/model "):
        _cmd_model(chat_id, cmd[7:].strip())
    elif tl.startswith("effort "):
        level = tl[7:].strip()
        if level in ("low", "medium", "high", "xhigh"):
            _set_effort(chat_id, level)
            _cmd_effort(chat_id, level)
    elif tl.startswith("lang "):
        lang_val = tl[5:].strip()
        if lang_val in ("zh", "en"):
            _set_lang(chat_id, lang_val)
            _cmd_lang(chat_id, lang_val)
    elif tl == "doc_write_confirm":
        _cmd_doc_write_do(chat_id)
    elif tl == "doc_write_cancel":
        pop_pending_doc_write(chat_id)
        send_card(chat_id, "❎ 已取消", "文档写入操作已取消。", color="orange")



def _cmd_upgrade(chat_id: str, msg_id: str = None):
    """/upgrade: pull latest code and restart the service in-place via os.execv (PID unchanged)."""
    threading.Thread(target=_do_upgrade, args=(chat_id, msg_id), daemon=True,
                     name="upgrade").start()


def _do_upgrade(chat_id: str, msg_id: str = None):
    import os as _os
    import sys as _sys
    from larkhelm.concurrency import set_shutting_down, wait_for_idle
    from larkhelm.config import _resolve_source_dir
    from larkhelm import config as _cfg_mod

    lang = _get_lang(chat_id)

    # Re-resolve SOURCE_DIR / editable mode on every entry rather than reading
    # the boot-time frozen globals. pipx can flip install layout (wheel ↔
    # editable) while the bridge is running, which leaves ``_cfg.SOURCE_DIR``
    # pointing at a deleted ``<site-packages>/larkhelm`` directory and makes
    # /upgrade impossible without a full restart. Doing the resolve here means
    # the *next* /upgrade after a layout change just works.
    source_dir, editable = _resolve_source_dir(Path(_cfg_mod.__file__))

    # Two distinct failure modes — collapsing them under one "不是 git 仓库"
    # message misleads operators when the real problem is that the directory
    # was deleted by a reinstall.
    if not source_dir.exists():
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ 升级失败", "❌ Upgrade Failed"),
                        _t(lang,
                           f"`SOURCE_DIR` 已不存在：`{source_dir}`\n\n"
                           f"通常意味着 pipx / pip 重装时换了 install 模式，"
                           f"旧路径被删除。建议重启 bridge（让 `_init_runtime` "
                           f"重新解析）或 `pipx reinstall larkhelm` 后再 `/upgrade`。",
                           f"`SOURCE_DIR` no longer exists: `{source_dir}`\n\n"
                           f"This usually means pipx / pip reinstalled in a different mode and "
                           f"deleted the old path. Try restarting the bridge (so `_init_runtime` "
                           f"re-resolves) or run `pipx reinstall larkhelm` then `/upgrade` again."),
                        color="red")
        return
    if not (source_dir / ".git").exists():
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ 升级失败", "❌ Upgrade Failed"),
                        _t(lang,
                           f"`SOURCE_DIR` 不是 git 仓库：`{source_dir}`\n\n"
                           f"editable 安装：`pipx install -e <repo>` 或 "
                           f"`pip install -e <repo>`；\n"
                           f"非 editable 安装：保留原始源目录不要删 —— "
                           f"`/upgrade` 依赖 dist-info `direct_url.json` 定位",
                           f"`SOURCE_DIR` is not a git repository: `{source_dir}`\n\n"
                           f"Editable install: `pipx install -e <repo>` or `pip install -e <repo>`\n"
                           f"Non-editable install: keep the original source directory — "
                           f"`/upgrade` uses dist-info `direct_url.json` to locate it"),
                        color="red")
        return

    # Capture the pre-pull HEAD so the post-restart card can show
    # ``<old> → <new>``. Best-effort: if anything goes wrong we just omit
    # the prev_head field and the card falls back to "已升级到 <new>".
    prev_head = ""
    try:
        _pr = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if _pr.returncode == 0:
            prev_head = _pr.stdout.strip()
    except Exception as e:
        _debug_log(f"[Upgrade] prev_head probe failed: {e}")

    # Step 1: git pull
    send_card_reply(chat_id, msg_id,
                    _t(lang, "🔄 升级中", "🔄 Upgrading"),
                    _t(lang, "正在拉取最新代码…", "Pulling latest code…"),
                    color="grey")
    try:
        r = subprocess.run(
            ["git", "-C", str(source_dir), "pull", "--autostash"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ 升级失败", "❌ Upgrade Failed"),
                        _t(lang, f"git pull 异常：{e}", f"git pull error: {e}"),
                        color="red")
        return

    if r.returncode != 0:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ 升级失败", "❌ Upgrade Failed"),
                        f"```\n{(r.stderr or r.stdout)[:600]}\n```", color="red")
        return

    output = r.stdout.strip()
    if "Already up to date" in output:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "✅ 已是最新版本", "✅ Already Up to Date"),
                        output, color="green")
        return

    # Step 2: reinstall package into the running venv so execv picks up new
    # code. Editable installs reuse ``pip install -e`` (re-links the venv
    # to ``source_dir`` — cheap, idempotent). Non-editable installs use
    # ``--force-reinstall`` to re-stage the package files in site-packages
    # without converting the install to editable (operator chose non-
    # editable for a reason; don't silently flip the install mode).
    send_card_reply(chat_id, msg_id,
                    _t(lang, "🔄 升级中", "🔄 Upgrading"),
                    _t(lang,
                       f"**拉取完成：**\n```\n{output[:400]}\n```\n\n正在安装新版本…",
                       f"**Pull complete:**\n```\n{output[:400]}\n```\n\nInstalling new version…"),
                    color="blue")
    pip_install_cmd = [
        _sys.executable, "-m", "pip", "install", "--no-deps", "-q",
    ]
    # Use the locally-resolved `editable` flag rather than ``_cfg.EDITABLE_INSTALL``;
    # the resolver above already accounts for any post-boot install-mode flip.
    if editable:
        pip_install_cmd += ["-e", str(source_dir)]
    else:
        pip_install_cmd += ["--force-reinstall", str(source_dir)]
    try:
        ri = subprocess.run(
            pip_install_cmd,
            capture_output=True, text=True, timeout=120,
        )
        if ri.returncode != 0:
            send_card_reply(chat_id, msg_id,
                            _t(lang, "❌ 升级失败", "❌ Upgrade Failed"),
                            _t(lang,
                               f"pip install 失败：\n```\n{(ri.stderr or ri.stdout)[:400]}\n```",
                               f"pip install failed:\n```\n{(ri.stderr or ri.stdout)[:400]}\n```"),
                            color="red")
            return
    except Exception as e:
        send_card_reply(chat_id, msg_id,
                        _t(lang, "❌ 升级失败", "❌ Upgrade Failed"),
                        _t(lang, f"pip install 异常：{e}", f"pip install error: {e}"),
                        color="red")
        return

    # Step 3: wait for in-flight tasks to finish
    send_card_reply(chat_id, msg_id,
                    _t(lang, "🔄 升级中", "🔄 Upgrading"),
                    _t(lang,
                       "**安装完成。**\n\n正在等待进行中的任务完成后重启…",
                       "**Installation complete.**\n\nWaiting for in-flight tasks to finish before restarting…"),
                    color="blue")

    set_shutting_down()
    from larkhelm.crew import cancel_all_crews, wait_crews_done
    cancel_all_crews(reason=_t(lang,
                               "服务升级中，Crew 任务重启后将自动恢复",
                               "Service upgrading — Crew tasks will resume automatically after restart"))
    wait_crews_done(timeout=30.0)
    idle_ok = wait_for_idle(timeout=60.0)
    if not idle_ok:
        warn("[Upgrade] wait_for_idle timed out — proceeding with execv while queries still in flight")

    # Always notify any chat that still has an in-flight query — the execv
    # below severs the lark-oapi WebSocket and freezes their streaming cards
    # until reconnect, regardless of whether wait_for_idle returned cleanly
    # (a chat that became busy between idle-check and execv would otherwise
    # be silently stranded). Skip the upgrade originator: they already have
    # the "🔄 服务正在重启" card sent below.
    from larkhelm.concurrency import get_busy_chat_ids
    for busy_cid in get_busy_chat_ids():
        if busy_cid == chat_id:
            continue
        try:
            busy_lang = _get_lang(busy_cid)
            send_card(busy_cid,
                      _t(busy_lang, "⚠️ 查询已中断", "⚠️ Query Interrupted"),
                      _t(busy_lang,
                         "服务正在升级重启，当前查询被中断，请稍后重新发送。",
                         "Service is restarting for an upgrade. Your current query was interrupted — please resend it in a moment."),
                      color="orange")
        except Exception as e:
            _debug_log(f"[Upgrade] in-flight notify failed: {e}")

    # Capture the new HEAD + subject so _post_init_notify can render a
    # specific "升级 prev → new (<subject>)" confirmation. Best-effort:
    # any failure simply omits the corresponding field.
    new_head = ""
    commit_subject = ""
    try:
        _nr = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if _nr.returncode == 0:
            new_head = _nr.stdout.strip()
        _sr = subprocess.run(
            ["git", "-C", str(source_dir), "log", "-1", "--format=%s"],
            capture_output=True, text=True, timeout=10,
        )
        if _sr.returncode == 0:
            commit_subject = _sr.stdout.strip()
    except Exception as e:
        _debug_log(f"[Upgrade] new_head/subject probe failed: {e}")

    # Write a marker file so the new process can confirm back to the upgrade requester
    import json as _json
    _notify_path = _cfg.DATA_DIR / "_restart_notify.json"
    _notify_payload = {"chat_id": chat_id, "ts": time.time()}
    if prev_head:
        _notify_payload["prev_head"] = prev_head
    if new_head:
        _notify_payload["new_head"] = new_head
    if commit_subject:
        _notify_payload["commit_subject"] = commit_subject
    try:
        _notify_path.write_text(
            _json.dumps(_notify_payload),
            encoding="utf-8",
        )
    except Exception as e:
        _debug_log(f"[Upgrade] failed to write restart notify: {e}")

    # Step 3: replace process in-place with os.execv.
    # Must pass [executable, '-m', 'larkhelm'] + original args explicitly — passing sys.argv
    # directly is wrong because sys.argv[0] is the __main__.py path, not the program name,
    # so the new Python process would try to open sys.argv[1] ('start') as a script.
    _debug_log("[Upgrade] os.execv replacing process")
    send_card_reply(chat_id, msg_id,
                    _t(lang, "🔄 升级中", "🔄 Upgrading"),
                    _t(lang, "服务正在重启，连接将在数秒内恢复…", "Service restarting — connection will resume in a few seconds…"),
                    color="blue")
    time.sleep(1)   # Give send_card enough time to deliver
    _os.execv(_sys.executable, [_sys.executable, "-m", "larkhelm"] + _sys.argv[1:])


# ═══════════════════════════════════════════════════
#  /effort — Claude 推理力度控制
# ═══════════════════════════════════════════════════

_EFFORT_INFO = [
    # (level, icon_label, zh_desc, en_desc)
    ("low",    "⚡ 快速",  "简单问答、快速回复，关闭思维链",  "fast reply — thinking disabled"),
    ("medium", "⚖️ 均衡",  "默认模式，平衡速度与质量",        "default — balanced speed and quality"),
    ("high",   "🔍 深度",  "复杂分析、多步推理，提升准确性",  "deep reasoning for complex tasks"),
    ("xhigh",  "🚀 极限",  "Opus 专属，最强推理，耗时较长",   "Opus-only — max reasoning, takes longer"),
]
_EFFORT_LABELS_ZH = {
    "low":    "⚡ 快速",
    "medium": "⚖️ 均衡",
    "high":   "🔍 深度",
    "xhigh":  "🚀 极限",
}


def _cmd_effort(chat_id: str, args: str, msg_id: str = None):
    """/effort [low|medium|high|xhigh] — view or set per-chat Claude reasoning effort."""
    lang = _get_lang(chat_id)
    level = (args or "").strip().lower()
    valid = {"low", "medium", "high", "xhigh"}

    if level and level not in valid:
        send_card_reply(
            chat_id, msg_id,
            _t(lang, "⚠️ 无效力度", "⚠️ Invalid effort level"),
            _t(lang,
               "有效值：`low` / `medium` / `high` / `xhigh`\n\n直接发 `/effort` 查看选项卡。",
               "Valid values: `low` / `medium` / `high` / `xhigh`\n\nSend `/effort` to view options."),
            color="orange",
        )
        return

    if level:
        _set_effort(chat_id, level)

    current = _get_effort(chat_id) or ""
    current_label = _EFFORT_LABELS_ZH.get(current, _t(lang, "⚖️ 均衡（默认）", "⚖️ Medium (default)"))

    lines = [f"**{_t(lang, '当前', 'Current')}：{current_label}**\n"]
    for lvl, icon_label, desc_zh, desc_en in _EFFORT_INFO:
        desc = desc_en if lang == "en" else desc_zh
        mark = " ✓" if lvl == current else ""
        lines.append(f"- {icon_label} `{lvl}` — {desc}{mark}")
    lines.append(
        "\n💡 " + _t(lang,
                    "`low` 模式自动关闭扩展思维链，节省 token 开销",
                    "`low` automatically disables extended thinking to save tokens")
    )
    lines.append(_t(lang,
                    "下次对话即刻生效，`/reset claude` 不影响此设置",
                    "Takes effect immediately. `/reset claude` does not affect this setting."))

    buttons = [
        ("⚡ 快速", "effort low"),
        ("⚖️ 均衡", "effort medium"),
        ("🔍 深度", "effort high"),
        ("🚀 极限", "effort xhigh"),
    ]
    title = _t(lang, "🧠 推理力度", "🧠 Reasoning Effort")

    if msg_id:
        send_card_reply(chat_id, msg_id, title, "\n".join(lines), color="blue", buttons=buttons)
    else:
        send_card(chat_id, title, "\n".join(lines), color="blue", buttons=buttons)


# ═══════════════════════════════════════════════════
#  /lang — UI language switch (zh / en)
# ═══════════════════════════════════════════════════

def _cmd_lang(chat_id: str, args: str, msg_id: str = None):
    """/lang [zh|en] — view or switch bot UI language for this chat."""
    lang_arg = (args or "").strip().lower()

    if lang_arg and lang_arg not in ("zh", "en"):
        send_card_reply(
            chat_id, msg_id,
            "⚠️ 不支持的语言 / Unsupported language",
            "支持：`zh`（中文）· `en`（English）",
            color="orange",
        )
        return

    if lang_arg:
        _set_lang(chat_id, lang_arg)

    current = _get_lang(chat_id)
    flag = "🇨🇳" if current == "zh" else "🇺🇸"
    name_zh = "中文" if current == "zh" else "英文"
    name_en = "Chinese" if current == "zh" else "English"

    body = (
        f"**当前 / Current：{flag} {name_zh} / {name_en}** (`{current}`)\n\n"
        "切换 / Switch:"
    )
    buttons = [("🇨🇳 中文", "lang zh"), ("🇺🇸 English", "lang en")]

    if msg_id:
        send_card_reply(chat_id, msg_id, "🌐 界面语言 / Language", body, color="blue", buttons=buttons)
    else:
        send_card(chat_id, "🌐 界面语言 / Language", body, color="blue", buttons=buttons)

