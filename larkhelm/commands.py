"""
larkhelm · command implementations

Contains all _cmd_* functions, _dispatch_button_cmd(), and helper utilities.
"""
import json
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, log_entry, _read_logs
from larkhelm.chat_state import (
    _get_cwd, _set_chat_field, _get_chat_state, _get_chat_model,
    _load_sid, _clear_sid, _register_btw_msg,
    set_pending_doc_write, pop_pending_doc_write,
)
from larkhelm.concurrency import (
    _get_chat_lock, _trigger_cancel, _pop_pending,
    _cron_lock, _get_btw_lock, _reset_cancel,
)
from larkhelm.token_stats import get_token_stats, get_token_stats_persistent
from larkhelm.perm import revoke_yolo, is_yolo
from larkhelm.cmd_doc import _cmd_doc, _cmd_doc_write_do
from larkhelm.lark_client import (
    send_card, send_card_reply, reply_card, _send_card_raw, _patch_card_raw, _reply_card_raw,
    react_to_message, delete_reaction,
    EMOJI_PROCESSING, EMOJI_DONE, EMOJI_ERROR,
    send_permission_guide,
)
from larkhelm.card_builder import _make_card, _fmt_elapsed


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


# Model-rotation cycle for the "切换模型" button in /status and /help.
# Single source of truth — historically duplicated literally across both
# command bodies; kept module-level so future additions (e.g. a new
# provider) need exactly one edit.
_NEXT_MODEL_CYCLE = {
    "claude":   "gemini",
    "gemini":   "kimi",
    "kimi":     "deepseek",
    "deepseek": "claude",
}


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
    try:
        r = subprocess.run(
            args, shell=False, capture_output=True, text=True,
            timeout=30, cwd=cwd, env=safe_env
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "命令超时（>30s）", -1
    except Exception as e:
        return "", str(e), -1


def _strip_at_mention(text: str) -> str:
    """Strip Feishu group-chat @mention prefix."""
    return re.sub(r'@\S+\s*', '', text).strip()


# ═══════════════════════════════════════════════════
#  Command implementations
# ═══════════════════════════════════════════════════

def _cmd_reset(chat_id: str, which: str = None, msg_id: str = None):
    """Unified reset logic. which=None resets everything; otherwise 'claude'/'gemini'/'perm'."""
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
        log_entry(chat_id, "reset", "reset:all", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, "⚠️ 部分重置",
                            "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, "♻️ 已重置",
                            "所有 AI 会话均已清空（三层记忆已保留）。\n\n"
                            "如需同时清除会话记忆：`/memory clear session`", color="green")
    elif which == "claude":
        _clear_sid(chat_id, "claude")
        try:
            from larkhelm.api_session import clear_history as _clear_api_hist
            _clear_api_hist("anthropic_api", chat_id)
        except Exception as e:
            _debug_log(f"[reset] clear_history failed: {e}")
            _api_clear_failed = True
        log_entry(chat_id, "reset", "reset:claude", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, "⚠️ 部分重置",
                            "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, "♻️ 已重置",
                            "Claude 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                            color="green")
    elif which == "gemini":
        _clear_sid(chat_id, "gemini")
        try:
            from larkhelm.api_session import clear_history as _clear_api_hist
            _clear_api_hist("google_api", chat_id)
        except Exception as e:
            _debug_log(f"[reset] clear_history failed: {e}")
            _api_clear_failed = True
        log_entry(chat_id, "reset", "reset:gemini", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, "⚠️ 部分重置",
                            "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, "♻️ 已重置",
                            "Gemini 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                            color="green")
    elif which == "kimi":
        _clear_sid(chat_id, "kimi")
        try:
            from larkhelm.api_session import clear_history as _clear_api_hist
            _clear_api_hist("openai_compat_api", chat_id)
        except Exception as e:
            _debug_log(f"[reset] clear_history failed: {e}")
            _api_clear_failed = True
        log_entry(chat_id, "reset", "reset:kimi", model="system")
        if _api_clear_failed:
            send_card_reply(chat_id, msg_id, "⚠️ 部分重置",
                            "会话 ID 已清除，但 API 历史清除失败，AI 可能仍记得部分上下文。",
                            color="orange")
        else:
            send_card_reply(chat_id, msg_id, "♻️ 已重置",
                            "Kimi 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                            color="green")
    elif which == "deepseek":
        _clear_sid(chat_id, "deepseek")
        log_entry(chat_id, "reset", "reset:deepseek", model="system")
        send_card_reply(chat_id, msg_id, "♻️ 已重置",
                        "DeepSeek 会话已清空（记忆已保留）。\n\n如需同时清除会话记忆：`/memory clear session`",
                        color="green")
    elif which == "memory":
        try:
            from larkhelm.memory import _session_memory_file
            _session_memory_file(chat_id).unlink(missing_ok=True)
        except Exception as e:
            _debug_log(f"[reset] memory unlink failed: {e}")
        log_entry(chat_id, "reset", "reset:memory", model="system")
        send_card_reply(chat_id, msg_id, "♻️ 已重置", "会话记忆已清除（全局/项目记忆保留）。", color="green")
    elif which in ("perm", "permissions"):
        revoke_yolo(chat_id)
        send_card_reply(chat_id, msg_id, "🔐 权限已重置", "「允许所有」已取消，后续工具调用将重新弹出审批。", color="green")


def _cmd_status(chat_id: str, msg_id: str = None):
    s_c = _load_sid(chat_id, "claude")
    s_g = _load_sid(chat_id, "gemini")
    cwd = _get_cwd(chat_id)
    model = _get_chat_model(chat_id)

    def _ver(cmd):
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, timeout=15, text=True)
            return (r.stdout.strip() or r.stderr.strip()).split("\n")[0] if r.returncode == 0 else None
        except Exception as e:
            _debug_log(f"[status] version probe failed for {cmd!r}: {e}")
            return None

    s_k = _load_sid(chat_id, "kimi")
    s_d = _load_sid(chat_id, "deepseek")
    cv, gv, kv = _ver(_cfg.CLAUDE_CMD), _ver(_cfg.GEMINI_CMD), _ver(_cfg.KIMI_CMD)
    # DeepSeek is HTTP — "version" is just the configured model + base URL host
    if getattr(_cfg, "DEEPSEEK_API_KEY", ""):
        _ds_host = (_cfg.DEEPSEEK_BASE_URL or "").replace("https://", "").replace("http://", "").split("/", 1)[0]
        dv = f"{_cfg.DEEPSEEK_MODEL} @ {_ds_host}" if _ds_host else _cfg.DEEPSEEK_MODEL
    else:
        dv = None

    def _cli_status(ver, sid, name):
        if not ver:
            return f"❌ {name} 不可用"
        if sid:
            # DeepSeek's "sid" is JSON history; show length instead of opaque hash
            if name == "DeepSeek":
                try:
                    import json as _json
                    n_msgs = len(_json.loads(sid))
                    return f"✅ {name}  会话 **{n_msgs} 条历史**"
                except Exception:
                    pass
            return f"✅ {name}  会话 **{sid[:12]}…**"
        return f"✅ {name}  暂无会话"

    if _cfg.SKIP_PERMISSIONS:
        perm_status = "⏭️ 跳过（skip_permissions=true）"
    elif is_yolo(chat_id):
        perm_status = "🚀 允许所有（发送 **/reset perm** 恢复审批）"
    else:
        perm_status = "🔐 正常审批"

    # Active crew info
    crew_info = ""
    try:
        from larkhelm.crew import _active_crew, _active_crew_states, _active_crew_lock
        with _active_crew_lock:
            crew_id    = _active_crew.get(chat_id)
            crew_state = _active_crew_states.get(chat_id)
        if crew_id and crew_state:
            _phase_label = {
                "planning": "规划中", "planned": "已规划", "running": "执行中",
                "synthesizing": "综合中", "breakpoint": "等待确认",
                "done": "已完成", "cancelled": "已取消",
            }.get(crew_state.phase, crew_state.phase)
            n_done  = sum(1 for a in crew_state.agents.values() if a.status.value == "done")
            n_total = len(crew_state.agents)
            crew_info = f"**Crew 进行中** {crew_id[:8]}…　{_phase_label}　{n_done}/{n_total} 完成"
        elif crew_id:
            crew_info = f"**Crew 进行中** {crew_id[:8]}…"
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
            for s in enabled + disabled:
                if not s.enabled:
                    icon = "⏸"
                elif s.healthy:
                    icon = "✅"
                else:
                    icon = "❌"
                # Activity: prefer last_used_at (real traffic), fallback to last_probed_at
                last_used = getattr(s, "last_used_at", 0.0) or 0.0
                last_probed = getattr(s, "last_probed_at", 0.0) or 0.0
                if last_used >= last_probed:
                    activity = f"用 {_fmt_ago(last_used, now)}" if last_used else (f"探 {_fmt_ago(last_probed, now)}" if last_probed else "—")
                else:
                    activity = f"探 {_fmt_ago(last_probed, now)}"
                # Failure pressure (sliding window for TRANSIENT)
                fw = getattr(s, "failure_window", []) or []
                fail_str = f" ⚠️{len(fw)}失败" if fw else ""
                # Error detail
                err_str = ""
                if not s.healthy and s.enabled and s.last_error:
                    err_str = f" _{s.last_error[:80]}_"
                elif s.enabled and s.provider in _API_PROVIDERS:
                    hist_len = len(_load_hist(s.provider, chat_id))
                    if hist_len:
                        err_str = f" `{hist_len}/{_MAX_HIST}msgs`"
                spec_lines.append(f"  • {icon} **{s.id}** `{activity}`{fail_str}{err_str}")
            disabled_note = f" · {len(disabled)} disabled" if disabled else ""
            backend_summary = (
                f"**Backends** {n_healthy}/{n_enabled} healthy{disabled_note}\n"
                + "\n".join(spec_lines)
            )
    except Exception as e:
        _debug_log(f"[status] backend summary failed: {e}")

    lines = [
        f"**模型** {model}　　**目录** {cwd}"
        + (f"　　**会话名** {_get_chat_state(chat_id).get('name', '').replace('**','').replace('`','')}"
           if _get_chat_state(chat_id).get('name') else ""),
        "",
        _cli_status(cv, s_c, "Claude"),
        _cli_status(gv, s_g, "Gemini"),
        _cli_status(kv, s_k, "Kimi"),
        _cli_status(dv, s_d, "DeepSeek"),
        "",
        f"**权限模式** {perm_status}",
        *([ crew_info ] if crew_info else []),
        *([ f"**Token（本次启动）** {token_summary}" ] if token_summary else []),
        *([ backend_summary ] if backend_summary else []),
        "",
    ]

    tips = []
    if not s_c and not s_g and not s_k and not s_d:
        tips.append("💡 直接发消息开始第一次对话，会自动建立会话")
    else:
        tips.append("💡 **/pickup** — 获取在终端接力会话的命令")
        tips.append("💡 **/reset** — 清除会话，开始全新对话")
    # Use the same rotation constant as the button below so the tip text and
    # the action button always agree on what "next model" means. Previously
    # the tip ladder hardcoded its own (different) cycle: e.g. for kimi the
    # tip said "/model claude" while the button said "/model deepseek".
    _next_for_tip = _NEXT_MODEL_CYCLE.get(model, "claude")
    tips.append(f"💡 **/model {_next_for_tip}** — 切换默认模型为 {_next_for_tip.capitalize()}")

    lines += tips
    other_model = _NEXT_MODEL_CYCLE.get(model, "claude")
    buttons = [
        ("♻️ 重置会话", "/reset"),
        ("🔗 接入终端", "/pickup"),
        (f"切换 {other_model}", f"/model {other_model}"),
    ]
    send_card_reply(chat_id, msg_id, "📊 运行状态", "\n".join(lines), color="turquoise", buttons=buttons, normalize=False)


def _cmd_help(chat_id: str, msg_id: str = None):
    model = _get_chat_model(chat_id)
    other = _NEXT_MODEL_CYCLE.get(model, "claude")
    body = (
        f"**当前模型:** {model}　　发消息直接提问，命令均以 `/` 开头\n"
        "\n"
        "**🚀 常用操作**\n"
        f"**/reset** — 重置会话，开始新对话\n"
        f"**/pickup** — 获取在终端接力会话的命令\n"
        f"**/model {other}** — 切换到 {other}\n"
        "**/cd 路径** — 切换工作目录\n"
        "**/cancel** — 取消当前查询\n"
        "**/rename <名称>** — 给当前会话命名\n"
        "\n"
        "---\n"
        "\n"
        "**单条消息指定模型（本条生效，不改变默认）**\n"
        "**/c** 或 **/claude** 消息 — 本条用 Claude\n"
        "**/g** 或 **/gemini** 消息 — 本条用 Gemini\n"
        "**/k** 或 **/kimi** 消息 — 本条用 Kimi\n"
        "**/d** 或 **/deepseek** 消息 — 本条用 DeepSeek（HTTP API）\n"
        "\n"
        "**会话**\n"
        "**/reset** claude · gemini · kimi · deepseek — 单独重置会话\n"
        "**/reset perm** — 重置权限审批　　**/reset memory** — 清除会话记忆（全局/项目保留）\n"
        "**/lock** — 列出所有可用 backend 及健康状态\n"
        "**/lock <id>** — 持久锁定到指定 backend（后续所有消息生效）　**/lock off** — 解锁\n"
        "**/model** <id> — 同 /lock（两者完全等价，均接受任意 backend ID）\n"
        "\n"
        "**目录 & Shell**\n"
        "**/pwd**　**/ls** [路径]　**/run** 命令（30s 超时）\n"
        "\n"
        "**其他**\n"
        "**/status** — 查看运行状态　　**/help** — 此帮助\n"
        "**/history** [all] — 最近 10 条对话摘要　　**/stats** — 今日统计\n"
        "**/upgrade** — 更新 larkhelm 到最新版本\n"
        "**/cron add** \"<expr>\" <查询>　**/cron list**　**/cron del** <id> — 定时任务\n"
        "**/doc** read · append · write · create · setfolder — 飞书文档操作\n"
        "\n"
        "**🤖 多 Agent 协作**\n"
        "**/crew** <需求> — 动态规划：Manager 自动分解任务，多 Agent 并行执行\n"
        "**/dev** <需求> — 软件工程流水线：\n"
        "　　PM → **[确认]** → 架构师 → 工程师 → QA（失败重试 2×）→ 审查员（APPROVED · REJECTED 重试 1×）\n"
        "**/plan** — 多阶段编排，一条指令串行执行多个 dev/review/fix/test 步骤：\n"
        "　　**[dev]** 实现登录　**[review]** 安全审查　**[fix]** 修复问题　**[test]** 回归测试\n"
        "　　每步完成后等待确认，支持跳过或取消；也可 **/plan** <飞书文档URL> 从文档读取计划\n"
        "**/btw** <问题> — 快问（不占主任务锁，回复到消息线程）\n"
        "\n"
        "**🧠 记忆系统（自动学习，无需手工维护）**\n"
        "**/memory** — 查看三层记忆（全局/项目/会话）当前内容\n"
        "全局/项目记忆每 10 轮自动从对话中提取并更新；以下命令用于手工覆盖：\n"
        "**/memory set global <内容>** — 手动覆盖全局偏好\n"
        "**/memory set project <内容>** — 手动覆盖当前项目记忆\n"
        "**/memory update** — 立即触发会话摘要生成（同时触发全局/项目提取）\n"
        "**/memory clear session|project|global|all** — 清除指定层记忆\n"
        "**/memory list** — 查看所有项目记忆文件\n"
        "**/memory gc [天数] [apply]** — 清理 N 天未更新的项目记忆（默认预演 30 天）"
    )
    buttons = [
        ("♻️ 重置会话", "/reset"),
        ("🔗 接入终端", "/pickup"),
        ("📊 查看状态", "/status"),
        ("🛑 取消查询", "/cancel"),
        (f"切换 {other}", f"/model {other}"),
    ]
    send_card_reply(chat_id, msg_id, "📖 帮助", body, color="blue", buttons=buttons, normalize=False)


def _cmd_pickup(chat_id: str, msg_id: str = None):
    s_c  = _load_sid(chat_id, "claude")
    s_g  = _load_sid(chat_id, "gemini")
    s_k  = _load_sid(chat_id, "kimi")
    s_d  = _load_sid(chat_id, "deepseek")
    cwd  = _get_cwd(chat_id)
    lines = [f"**工作目录:** `{cwd}`\n"]
    if s_c:
        lines.append(f"**Claude 接力：**\n```bash\ncd {cwd}\nclaude --resume {s_c}\n```")
    else:
        lines.append("**Claude:** 无活跃会话")
    if s_g:
        lines.append(f"\n**Gemini 接力：**\n```bash\ncd {cwd}\ngemini --resume {s_g}\n```")
    else:
        lines.append("\n**Gemini:** 无活跃会话")
    if s_k:
        lines.append(f"\n**Kimi 接力：**\n```bash\ncd {cwd}\nkimi --session {s_k}\n```")
    else:
        lines.append("\n**Kimi:** 无活跃会话")
    # DeepSeek has no terminal CLI; show a one-liner curl scaffold using the persisted history file
    if s_d:
        from larkhelm.chat_state import _sid_file as _sf
        sid_path = _sf(chat_id, "deepseek")
        lines.append(
            f"\n**DeepSeek 接力（HTTP）：**\n"
            f"无官方 CLI；会话历史保存在 `{sid_path}`，可在脚本中 POST 到 "
            f"`{_cfg.DEEPSEEK_BASE_URL}/chat/completions` 复用。"
        )
    else:
        lines.append("\n**DeepSeek:** 无活跃会话")
    lines.append("\n> 在终端运行上面命令即可无缝接力")
    send_card_reply(chat_id, msg_id, "🔗 终端接力", "\n".join(lines), color="purple")


def _cmd_history(chat_id: str, show_all: bool = False, msg_id: str = None):
    """Display conversation history.
    By default shows only the current session since the last reset;
    show_all=True shows all records with separator lines at reset points.
    """
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
            body = "_当前会话暂无对话记录_"
            if pre_count:
                body += f"\n\n此前会话还有 **{pre_count}** 条记录，发送 `/history all` 查看"
            send_card_reply(chat_id, msg_id, "📜 当前会话", body, color="blue")
            return

        parts = [_pair_line(u, a) for u, a in pairs]
        footer = ""
        if pre_count:
            footer = f"\n\n_此前会话还有 **{pre_count}** 条记录，发送 `/history all` 查看_"
        title = f"📜 当前会话（{len(pairs)} 条）"
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
                label = {"all": "全部", "claude": "Claude", "gemini": "Gemini"}.get(which, which)
                parts.append(f"— ♻️ 重置（{label}）{ts} —")
                pending_user = None
            elif r["role"] == "user":
                pending_user = r
            elif r["role"] in ("assistant", "error") and pending_user:
                parts.append(_pair_line(pending_user, r))
                pending_user = None
                pair_count += 1

        if not parts:
            send_card_reply(chat_id, msg_id, "📜 对话历史", "_暂无对话记录_", color="blue")
            return

        total_pairs = len(_build_pairs(records))
        title = f"📜 全部历史（{pair_count} 条"
        title += f"，共 {total_pairs} 条）" if total_pairs > pair_count else "）"
        send_card_reply(chat_id, msg_id, title, "\n".join(parts), color="blue", normalize=False)


def _fmt_token_block(label: str, data: dict) -> str:
    """Format a {model: {...}} statistics dict as a markdown block."""
    if not data:
        return f"**{label}** — 暂无数据"
    lines = [f"**{label}**"]
    for model, m in sorted(data.items()):
        inp   = m["input_tokens"]
        out   = m["output_tokens"]
        cr    = m["cache_read"]
        cc    = m["cache_create"]
        total = inp + out
        calls = m["calls"]
        cost  = m["cost_usd"]

        # Cache hit rate: cache_read / (cache_read + non-cached input)
        # non-cached input = inp - cache_read (cache_read is already included in inp)
        hit_pct    = int(cr / max(inp, 1) * 100)
        create_pct = int(cc / max(inp, 1) * 100)

        cost_str = f"${cost:.4f}" if cost else "—"
        lines.append(
            f"› **{model}**  {calls} 次  合计 **{total:,}** tokens  费用 **{cost_str}**\n"
            f"  输入 {inp:,}  输出 {out:,}\n"
            f"  缓存命中 {cr:,}（{hit_pct}%）  缓存写入 {cc:,}（{create_pct}%）"
        )
    return "\n".join(lines)


def _cmd_stats_intent(chat_id: str, msg_id: str = None, date: str | None = None):
    """Render today's intent dispatcher aggregate (hit rate / latency / cost)."""
    try:
        from larkhelm.agent_hub.agent_audit import aggregate_daily
    except Exception as e:
        send_card_reply(chat_id, msg_id, "📊 Intent 统计",
                        f"agent_hub 未启用或导入失败：{e}", color="orange")
        return
    agg = aggregate_daily(date)
    if agg["total"] == 0:
        send_card_reply(chat_id, msg_id, f"📊 Intent 统计 · {agg['date']}",
                        "_当日没有 Agent 调度记录_", color="grey")
        return
    lines = [
        f"**调度总数：** {agg['total']}",
        f"**成功率：** {agg['success_rate'] * 100:.1f}%　·　"
        f"平均耗时：{agg['avg_duration']:.2f}s　·　"
        f"成本：${agg['total_cost']:.4f}",
        "",
        "**按 Agent：**",
    ]
    for atype, info in sorted(agg["per_agent"].items()):
        lines.append(
            f"- `{atype}`：{info['count']} 次（成功 {info['success']}，"
            f"avg {info['avg_duration']:.2f}s）"
        )
    send_card_reply(chat_id, msg_id, f"📊 Intent 统计 · {agg['date']}",
                    "\n".join(lines), color="turquoise")


def _cmd_stats(chat_id: str, msg_id: str = None, args: str = ""):
    """Display today / this-month / all-time token stats plus conversation activity for the current chat."""
    sub = (args or "").strip().lower()
    if sub == "intent":
        _cmd_stats_intent(chat_id, msg_id)
        return
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")
    month = now.strftime("%Y-%m")

    records = _read_logs(chat_id)
    today_records = [r for r in records if r["ts"].startswith(today)]

    user_count  = sum(1 for r in today_records if r["role"] == "user")
    error_count = sum(1 for r in today_records if r["role"] == "error")

    durations: list[float] = []
    pending_ts = None
    for r in today_records:
        if r["role"] == "user":
            pending_ts = datetime.fromisoformat(r["ts"])
        elif r["role"] in ("assistant", "error") and pending_ts:
            secs = (datetime.fromisoformat(r["ts"]) - pending_ts).total_seconds()
            if 0 < secs < 3600:
                durations.append(secs)
            pending_ts = None
    avg = sum(durations) / len(durations) if durations else 0

    # Persistent stats across three time windows
    stats_today = get_token_stats_persistent(chat_id, date_prefix=today)
    stats_month = get_token_stats_persistent(chat_id, date_prefix=month)
    stats_all   = get_token_stats_persistent(chat_id, date_prefix=None)

    # In-memory stats for this process run (fallback if all.jsonl is empty)
    stats_mem   = get_token_stats(chat_id)

    parts = [
        f"**统计日期：** {today}",
        f"今日对话：**{user_count}** 次　错误：**{error_count}** 次　"
        f"平均耗时：**{_fmt_elapsed(avg) if avg else '—'}**",
        "---",
        _fmt_token_block(f"📅 今日（{today}）", stats_today),
        "---",
        _fmt_token_block(f"🗓 本月（{month}）", stats_month),
        "---",
        _fmt_token_block("📦 累计（全部）", stats_all),
    ]

    # If persistent data is empty (fresh deploy or upgrade from old version), show in-memory stats as fallback
    if not stats_all and stats_mem:
        parts.append("---")
        parts.append(_fmt_token_block("⚡ 本次启动（内存）", stats_mem))

    send_card_reply(chat_id, msg_id, "📊 Token 统计", "\n\n".join(parts), color="turquoise")


def _cmd_cron(chat_id: str, args: str, msg_id: str = None):
    """Handle /cron command: add / list / del."""
    from croniter import croniter, CroniterBadCronError
    import uuid as _uuid

    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""

    if sub == "list":
        crons = _get_chat_state(chat_id).get("crons", [])
        if not crons:
            send_card_reply(chat_id, msg_id, "⏰ 定时任务", "_暂无定时任务_", color="blue")
            return
        lines = []
        for c in crons:
            lines.append(f"**`{c['id']}`** `{c['expr']}` [{c['model']}]\n{c['query'][:60]}")
        send_card_reply(chat_id, msg_id, f"⏰ 定时任务（{len(crons)} 条）",
                        "\n\n---\n\n".join(lines), color="blue")
        return

    if sub == "del":
        cron_id = parts[1].strip() if len(parts) > 1 else ""
        if not cron_id:
            send_card_reply(chat_id, msg_id, "⚠️ 用法", "`/cron del <id>`", color="orange")
            return
        with _cron_lock:
            crons = _get_chat_state(chat_id).get("crons", [])
            new_crons = [c for c in crons if c["id"] != cron_id]
            _set_chat_field(chat_id, "crons", new_crons)
        if len(new_crons) < len(crons if crons else []):
            send_card_reply(chat_id, msg_id, "✅ 已删除", f"定时任务 `{cron_id}` 已删除。", color="green")
        else:
            send_card_reply(chat_id, msg_id, "❓ 未找到", f"没有 ID 为 `{cron_id}` 的定时任务。", color="orange")
        return

    if sub == "add":
        rest = parts[1].strip() if len(parts) > 1 else ""
        m = re.match(r'^["\'](.+?)["\'\s](.+)$', rest) or re.match(
            r'^((?:\S+\s+){4}\S+)\s+(.+)$', rest)
        if not m:
            send_card_reply(chat_id, msg_id, "⚠️ 用法",
                            '`/cron add "0 9 * * *" 每日早报查询`\n\n'
                            "cron 表达式为标准 5 字段（分 时 日 月 周）",
                            color="orange")
            return
        expr, query = m.group(1).strip(), m.group(2).strip()
        try:
            croniter(expr)
        except (CroniterBadCronError, Exception):
            send_card_reply(chat_id, msg_id, "❌ 表达式错误",
                            f"`{expr}` 不是有效的 cron 表达式。\n\n示例：`0 9 * * *`（每天 9:00）",
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
        send_card_reply(chat_id, msg_id, "✅ 定时任务已添加",
                        f"**ID：** `{cron_id}`\n\n"
                        f"**表达式：** `{expr}`（时区：{_cfg.CRON_TIMEZONE}）\n\n"
                        f"**查询：** {query[:80]}\n\n"
                        f"**下次执行：** {nxt.strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"查看：`/cron list`　删除：`/cron del {cron_id}`",
                        color="green")
        return

    send_card_reply(chat_id, msg_id, "⚠️ 用法",
                    "`/cron add \"<expr>\" <查询>` — 添加定时任务\n"
                    "`/cron list` — 查看所有任务\n"
                    "`/cron del <id>` — 删除任务\n\n"
                    "示例：`/cron add \"0 9 * * 1-5\" 总结今日 git log`",
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
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(_get_cwd(chat_id)) / p
        p = p.resolve()
        if not p.is_dir():
            send_card_reply(chat_id, msg_id, "❌ 目录不存在", f"`{p}`", color="red")
            return
        if not _check_cwd_root(p):
            send_card_reply(chat_id, msg_id, "❌ 超出允许范围",
                            f"配置的 `cwd_root` 限制了可访问路径", color="red")
            return
        _set_chat_field(chat_id, "cwd", str(p))
        send_card_reply(chat_id, msg_id, "📁 目录已切换", f"`{p}`", color="green")
    except Exception as e:
        send_card_reply(chat_id, msg_id, "❌ 错误", str(e), color="red")


def _cmd_pwd(chat_id: str, msg_id: str = None):
    send_card_reply(chat_id, msg_id, "📁 当前目录", f"`{_get_cwd(chat_id)}`", color="blue")


def _cmd_ls(chat_id: str, path: str = "", msg_id: str = None):
    cwd = _get_cwd(chat_id)
    target = (Path(cwd) / path if path else Path(cwd)).resolve()
    if not _check_cwd_root(target):
        send_card_reply(chat_id, msg_id, "❌ 超出允许范围",
                        f"配置的 `cwd_root` 限制了可访问路径", color="red")
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
            lines.append(f"\n_... 共 {len(entries)} 项_")
        send_card_reply(chat_id, msg_id, "📂 文件列表", "\n".join(lines), color="blue", normalize=False)
    except Exception as e:
        send_card_reply(chat_id, msg_id, "❌ 错误", str(e), color="red")


def _cmd_run(chat_id: str, cmd: str, msg_id: str = None):
    cwd = _get_cwd(chat_id)
    mid = send_card_reply(chat_id, msg_id, "⏳ 执行中",
                          f"```bash\n{cmd}\n```\n目录: `{cwd}`", color="grey")
    stdout, stderr, rc = _run_shell(chat_id, cmd)
    color = "green" if rc == 0 else "red"
    icon = "✅" if rc == 0 else "❌"
    body = f"```bash\n{cmd}\n```\n目录: `{cwd}`\n"
    if stdout.strip():
        body += f"\n**输出：**\n```\n{stdout.strip()[:2000]}\n```"
    if stderr.strip():
        body += f"\n**错误：**\n```\n{stderr.strip()[:500]}\n```"
    if not stdout.strip() and not stderr.strip():
        body += "\n_（无输出）_"
    body += f"\n\n退出码: `{rc}`"
    # Log only the command itself, not its output (stdout/stderr may contain passwords, tokens, etc.)
    log_entry(chat_id, "shell", f"$ {cmd}", model="shell")
    reply_card(chat_id, mid, f"{icon} Shell", body, color=color)


def _cmd_lock(chat_id: str, args: str = "", msg_id: str = None) -> None:
    """Handle /lock command.

    /lock <backend_id>  — lock to specific backend (validates exists + healthy)
    /lock off           — clear locked_backend, restore auto-routing
    /lock               — show current locked_backend status
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY
    args = args.strip()

    if not args:
        # Show current locked backend
        locked_id = _get_chat_state(chat_id).get("locked_backend")
        if locked_id:
            spec = BACKEND_REGISTRY.get(locked_id)
            name = spec.display_name if spec else locked_id
            health = "✅" if (spec and spec.healthy) else "❌"
            send_card_reply(chat_id, msg_id, "🔒 已锁定 Backend",
                            f"{health} **{name}** (`{locked_id}`)\n\n解锁: `/lock off`",
                            color="blue")
        else:
            specs = BACKEND_REGISTRY.all_enabled()
            lines = ["_当前使用自动路由_\n", "可用 Backends:"]
            for s in specs:
                mark = "✅" if s.healthy else "❌"
                lines.append(f"  {mark} `{s.id}` — {s.display_name} (`{s.provider}`)")
            lines.append("\n锁定: `/lock <id>`")
            send_card_reply(chat_id, msg_id, "🔓 自动路由中",
                            "\n".join(lines), color="blue", normalize=False)
        return

    if args.lower() == "off":
        _set_chat_field(chat_id, "locked_backend", None)
        send_card_reply(chat_id, msg_id, "🔓 已解锁",
                        "已恢复自动路由，不再锁定特定 backend。", color="green")
        return

    # Lock to specific backend
    backend_id = args
    spec = BACKEND_REGISTRY.get(backend_id)
    if spec is None:
        send_card_reply(chat_id, msg_id, "❌ Backend 不存在",
                        f"未找到 backend: `{backend_id}`\n发送 `/lock` 查看所有可用 backends。",
                        color="red")
        return

    if not spec.enabled:
        send_card_reply(chat_id, msg_id, "❌ Backend 已禁用",
                        f"`{backend_id}` 已被禁用（enabled=false），无法锁定。\n\n"
                        f"发送 `/lock` 查看可用 backends。",
                        color="red")
        return

    if not spec.healthy:
        send_card_reply(chat_id, msg_id, "❌ Backend 不可用",
                        f"`{backend_id}` 当前不可用（health check 失败）。\n\n"
                        f"错误: {spec.last_error or '未知'}\n\n"
                        f"可用: 等待恢复后重试，或 `/lock` 选择其他 backend。",
                        color="red")
        return

    _set_chat_field(chat_id, "locked_backend", spec.id)
    send_card_reply(chat_id, msg_id, "🔒 已锁定 Backend",
                    f"本 chat 已锁定到 **{spec.display_name}** (`{spec.id}`)。\n\n解锁: `/lock off`",
                    color="green")


def _cmd_model(chat_id: str, model_name: str = "", msg_id: str = None):
    """Alias for /lock: switches default backend for this chat."""
    _cmd_lock(chat_id, model_name, msg_id)


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
    cmd_lower = cmd.lower().strip()
    m_name    = {"claude": "Claude", "gemini": "Gemini", "kimi": "Kimi", "deepseek": "DeepSeek"}.get(model, model.capitalize())

    if cmd_lower in ("/help", "--help", "-h"):
        if model == "deepseek":
            body = (
                "DeepSeek 走 HTTP API，没有官方 CLI 会话命令。\n\n"
                "**桥接内置：**\n"
                "`/reset deepseek` — 清空会话历史\n"
                "`/status` — 查看当前 model / session 长度\n"
                "\n"
                "如需切回其他模型：`/model claude` 或 `/model gemini` 等。"
            )
            send_card_reply(chat_id, msg_id, f"📖 {m_name}", body, color="blue")
            return
        if model == "kimi":
            body = (
                "**会话管理**\n"
                "`/clear` — 清除对话历史，开始新会话\n"
                "`/history` — 查看历史会话列表\n"
                "`/compact` — 压缩上下文\n"
                "\n"
                "**其他**\n"
                "`/exit` / `/quit` — 退出 Kimi CLI\n"
                "\n"
                "_以上命令仅在终端交互模式下可用。_\n"
                "_发送 `/pickup` 获取终端接力命令。_"
            )
            send_card_reply(chat_id, msg_id, f"📖 {m_name} CLI 交互命令", body, color="blue")
            return
        if model == "claude":
            body = (
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
                "_发送 `/pickup` 获取终端接力命令。_"
            )
        else:
            body = (
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
                "_发送 `/pickup` 获取终端接力命令。_"
            )
        send_card_reply(chat_id, msg_id, f"📖 {m_name} CLI 交互命令", body, color="blue")
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
        send_card_reply(chat_id, msg_id, "🖥️ 需要在终端执行",
                        f"`{cmd}` 是 {m_name} CLI 的会话命令。\n\n"
                        f"桥接使用 `--print` 模式运行，该模式不支持 CLI 会话命令。\n"
                        f"发送 `/pickup` 接入终端后可直接输入 `{cmd}`。",
                        color="orange")
        return

    for _other_name, _other_pfx, _other_cmds in other_info:
        if cmd_lower in _other_cmds:
            send_card_reply(chat_id, msg_id, "❓ 模型不匹配",
                            f"`{cmd}` 是 {_other_name} CLI 的命令，不是 {m_name} CLI 的命令。\n\n"
                            f"试试：`{_other_pfx} {cmd}`",
                            color="orange")
            return

    send_card_reply(chat_id, msg_id, "❓ 未知命令",
                    f"`{cmd}` 不是已知的 {m_name} CLI 命令。\n\n"
                    f"- 查看 CLI 帮助：`{my_pfx} /help`\n"
                    f"- 执行 Shell 命令：`/run {cmd[1:]}`\n"
                    f"- 直接向 AI 提问：`{my_pfx} {cmd[1:]} 是什么`",
                    color="orange")


def _cmd_btw(chat_id: str, question: str, user_msg_id: str):
    """Handle /btw side-question: uses main session if free, otherwise a dedicated btw session; replies in the original message thread."""
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

            mid = _reply_card_raw(
                user_msg_id,
                _make_card("💬", "> 正在思考...", color="grey"),
                in_thread=False,
            )

            try:
                cur_text = [""]

                def _on_text(text, status="typing"):
                    cur_text[0] = text
                    if mid:
                        _patch_card_raw(mid, _make_card("💬", text.strip() or "> 正在思考...", color="grey"))

                # Inject memory context (L2)
                _btw_mem_ctx = ""
                try:
                    from larkhelm.memory import get_memory_context as _get_mem_ctx
                    _btw_mem_ctx = _get_mem_ctx(chat_id, cwd=cwd)
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

                final = (output or cur_text[0]).strip() or "（无输出）"
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
                err_card = _make_card("❌ 旁注失败", str(e)[:200], color="red")
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

def _cmd_memory(chat_id: str, args: str = "", msg_id: str = None):
    """/memory — show/set/clear/update/gc the three-tier memory system.

    /memory                        show all active layers
    /memory set global <text>      overwrite global layer (≤500 chars)
    /memory set project <text>     overwrite project layer for current cwd (≤1000 chars)
    /memory clear [global|project|session]   clear one or all layers
    /memory update                 force-regenerate session layer from logs
    /memory list                   list every project_*.md file
    /memory gc [days] [apply]      clean up stale project memory (dry-run by
                                   default; ``days`` defaults to 30, must ≥ 1)
    """
    from larkhelm.memory import (
        load_global_memory, save_global_memory,
        load_project_memory, save_project_memory,
        load_memory, save_memory, maybe_auto_update,
        _global_memory_file, _project_memory_file, _session_memory_file,
        _load_md_frontmatter, _load_md_body, _ensure_dir, MEMORY_HOME_DIR,
        GLOBAL_MAX_CHARS, PROJECT_MAX_CHARS, AUTO_UPDATE_EVERY,
    )
    from larkhelm.chat_state import _get_cwd
    cwd = _get_cwd(chat_id)
    args = args.strip()
    sub = args.lower()

    # ── /memory set global <text> ────────────────────────────────────────────
    if sub.startswith("set global ") or sub == "set global":
        text = args[11:].strip()  # len("set global ") == 11
        if not text:
            send_card_reply(chat_id, msg_id, "⚠️ 用法",
                            "`/memory set global <内容>` — 设置全局记忆（最多 500 字符）",
                            color="orange")
            return
        if _global_memory_file(chat_id) is None:
            send_card_reply(chat_id, msg_id, "⚠️ 全局记忆不可用",
                            "当前会话无法识别发送者身份，全局记忆已跳过（群聊安全保护）。",
                            color="orange")
            return
        save_global_memory(text, chat_id=chat_id)
        send_card_reply(chat_id, msg_id, "✅ 全局记忆已更新",
                        f"```\n{text[:200]}\n```", color="green")
        return

    # ── /memory set project <text> ───────────────────────────────────────────
    if sub.startswith("set project ") or sub == "set project":
        text = args[12:].strip()  # len("set project ") == 12
        if not text:
            send_card_reply(chat_id, msg_id, "⚠️ 用法",
                            f"`/memory set project <内容>` — 设置当前项目记忆（cwd: `{cwd}`，最多 1000 字符）",
                            color="orange")
            return
        save_project_memory(cwd, text)
        send_card_reply(chat_id, msg_id, "✅ 项目记忆已更新",
                        f"**目录**: `{cwd}`\n\n```\n{text[:200]}\n```", color="green")
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
                deleted.append("会话")
            if layer in ("all", "global"):
                _gf = _global_memory_file(chat_id)
                if _gf:
                    _gf.unlink(missing_ok=True)
                deleted.append("全局")
            if layer in ("all", "project"):
                _project_memory_file(cwd).unlink(missing_ok=True)
                deleted.append("项目")
            if not deleted:
                send_card_reply(chat_id, msg_id, "⚠️ 未知层级",
                                "可选: `global` · `project` · `session` · `all`",
                                color="orange")
                return
            if layer == "session":
                detail = "已清除会话记忆，全局和项目记忆已保留。"
            else:
                detail = f"已删除：{'、'.join(deleted)}记忆。"
            send_card_reply(chat_id, msg_id, "🗑️ 已清除", detail, color="green")
        except Exception as e:
            send_card_reply(chat_id, msg_id, "❌ 清除失败", str(e)[:200], color="red")
        return

    # ── /memory update ───────────────────────────────────────────────────────
    if sub == "update":
        update_mid = send_card_reply(chat_id, msg_id, "🔄 生成记忆中",
                                     "正在后台生成会话摘要，预计 10~30 秒，请稍候…", color="grey")

        _ERR_MSG_MAP = {
            "no_logs":              "当前会话暂无对话记录，无法生成摘要。",
            "no_conversation_logs": "当前会话暂无普通对话（只有系统/Shell 记录），无法生成摘要。",
        }

        def _on_update_done(success: bool, content, error):
            if success and content:
                preview = content[:200]
                new_card = _make_card("✅ 会话记忆已更新",
                                     f"```\n{preview}\n```", color="green")
            else:
                if error and error.startswith("timed_out_"):
                    secs = error.replace("timed_out_", "").replace("s", "")
                    err_msg = f"记忆生成超时（>{secs}s），请稍后重试。"
                else:
                    err_msg = _ERR_MSG_MAP.get(error or "", "记忆生成失败，请稍后重试。")
                new_card = _make_card("❌ 记忆生成失败", err_msg, color="red")
            if update_mid:
                _patch_card_raw(update_mid, new_card)
            else:
                send_card_reply(chat_id, msg_id,
                                "✅ 会话记忆已更新" if success else "❌ 记忆生成失败",
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
            send_card_reply(chat_id, msg_id, "⚠️ 用法",
                            f"无法识别参数：{' '.join(bad_tokens)}\n\n"
                            f"`/memory gc [天数] [apply]` — 清理 N 天未更新的项目记忆\n"
                            f"- `/memory gc` — 30 天 dry-run（默认，不删除）\n"
                            f"- `/memory gc 60` — 60 天 dry-run\n"
                            f"- `/memory gc apply` — 实际删除（30 天）\n"
                            f"- `/memory gc 60 apply` — 实际删除（60 天）",
                            color="orange")
            return
        if threshold_days < 1:
            send_card_reply(chat_id, msg_id, "⚠️ 阈值过低",
                            "天数必须 ≥ 1（防止误清空所有项目记忆）。", color="orange")
            return

        try:
            report = gc_project_memory(threshold_days=threshold_days, apply=apply)
        except Exception as e:
            send_card_reply(chat_id, msg_id, "❌ GC 失败", str(e)[:300], color="red")
            return

        scanned = report["scanned"]
        cands = report["candidates"]
        errs = report["errors"]
        if not cands:
            send_card_reply(
                chat_id, msg_id,
                f"🧹 项目记忆 GC（>{threshold_days} 天）",
                f"扫描 {scanned} 个文件，**无可清理项**。",
                color="green",
            )
            return

        # Build display: one row per candidate, capped to keep card readable.
        _MAX_ROWS = 30
        lines = []
        for c in cands[:_MAX_ROWS]:
            tag = "🗑️" if c["deleted"] else ("⚠️" if apply else "💤")
            cwd_disp = c["cwd"] or "_(无 cwd 元数据)_"
            age_disp = f"{c['age_days']}d" if c["age_days"] is not None else "?"
            lines.append(
                f"{tag} `{c['name']}` · 龄期 {age_disp} · 原因 `{c['reason']}`\n"
                f"   `{cwd_disp}`"
            )
        if len(cands) > _MAX_ROWS:
            lines.append(f"_… 另有 {len(cands) - _MAX_ROWS} 个候选未列出_")
        if errs:
            lines.append("\n**错误**：")
            for e in errs[:5]:
                lines.append(f"- `{Path(e['path']).name}` — {e['err'][:120]}")
            if len(errs) > 5:
                lines.append(f"_… 另有 {len(errs) - 5} 个错误_")

        if apply:
            n_deleted = sum(1 for c in cands if c["deleted"])
            title = f"🧹 项目记忆 GC 已执行（>{threshold_days} 天）"
            header = f"扫描 {scanned}，已删除 **{n_deleted}**，错误 {len(errs)}\n\n"
            color = "green" if n_deleted and not errs else "orange"
        else:
            title = f"🧹 项目记忆 GC · 预演（>{threshold_days} 天）"
            header = (
                f"扫描 {scanned}，发现 **{len(cands)}** 个候选。\n"
                f"_这是预演，未删除任何文件。要实际清理：_\n"
                f"`/memory gc {threshold_days} apply`\n\n"
            )
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
            send_card_reply(chat_id, msg_id, "📂 项目记忆", "_暂无项目记忆文件_", color="blue")
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
            lines.append(f"{mark} **{pf.stem}**\n`{stored_cwd}`  更新: {updated}  大小: {len(body)} 字符")
        send_card_reply(chat_id, msg_id, f"📂 项目记忆（{len(project_files)} 个）",
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
        return f" _(更新: {ts})_" if ts else ""

    g = load_global_memory(chat_id)
    p = load_project_memory(cwd)
    s = load_memory(chat_id)

    _g_path = _global_memory_file(chat_id)
    _p_path = _project_memory_file(cwd)
    _s_path = _session_memory_file(chat_id)

    sections: list[str] = []
    if g:
        sections.append(f"### 🌐 全局记忆{_fm_meta(_g_path)}\n{g}")
    else:
        sections.append("### 🌐 全局记忆\n_（空）_ — `/memory set global <内容>` 设置")

    if p:
        sections.append(f"### 📁 项目记忆 (`{cwd}`){_fm_meta(_p_path)}\n{p}")
    else:
        sections.append(f"### 📁 项目记忆 (`{cwd}`)\n_（空）_ — `/memory set project <内容>` 设置")

    if s:
        _next_turn = _turn + (AUTO_UPDATE_EVERY - (_turn % AUTO_UPDATE_EVERY) if _turn % AUTO_UPDATE_EVERY else AUTO_UPDATE_EVERY)
        sections.append(f"### 💬 会话记忆{_fm_meta(_s_path)} _(当前第 {_turn} 轮，第 {_next_turn} 轮时自动更新，可 `/memory update` 立即触发)_\n{s}")
    else:
        sections.append(f"### 💬 会话记忆 _(当前第 {_turn} 轮)_\n_（空）_ — 每 {AUTO_UPDATE_EVERY} 轮自动生成，或 `/memory update` 立即生成")

    body = "\n\n---\n\n".join(sections)
    body += ("\n\n---\n`/memory set global|project <内容>` 写入 · "
             "`/memory clear session|project|global|all` 清除 · "
             "`/memory update` 更新会话 · `/memory list` 查看项目记忆文件")
    send_card_reply(chat_id, msg_id, "🧠 三层记忆系统", body, color="blue", normalize=False)


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

    # Step 1: git pull
    send_card_reply(chat_id, msg_id, "🔄 升级中", "正在拉取最新代码…", color="grey")
    try:
        r = subprocess.run(
            ["git", "-C", str(_cfg.SOURCE_DIR), "pull"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        send_card_reply(chat_id, msg_id, "❌ 升级失败", f"git pull 异常：{e}", color="red")
        return

    if r.returncode != 0:
        send_card_reply(chat_id, msg_id, "❌ 升级失败",
                        f"```\n{(r.stderr or r.stdout)[:600]}\n```", color="red")
        return

    output = r.stdout.strip()
    if "Already up to date" in output:
        send_card_reply(chat_id, msg_id, "✅ 已是最新版本", output, color="green")
        return

    # Step 2: reinstall package into the running venv so execv picks up new code
    send_card_reply(chat_id, msg_id, "🔄 升级中",
                    f"**拉取完成：**\n```\n{output[:400]}\n```\n\n"
                    "正在安装新版本…", color="blue")
    try:
        ri = subprocess.run(
            [_sys.executable, "-m", "pip", "install", "--no-deps", "-q",
             str(_cfg.SOURCE_DIR)],
            capture_output=True, text=True, timeout=120,
        )
        if ri.returncode != 0:
            send_card_reply(chat_id, msg_id, "❌ 升级失败",
                            f"pip install 失败：\n```\n{(ri.stderr or ri.stdout)[:400]}\n```",
                            color="red")
            return
    except Exception as e:
        send_card_reply(chat_id, msg_id, "❌ 升级失败", f"pip install 异常：{e}", color="red")
        return

    # Step 3: wait for in-flight tasks to finish
    send_card_reply(chat_id, msg_id, "🔄 升级中",
                    f"**安装完成。**\n\n正在等待进行中的任务完成后重启…", color="blue")

    set_shutting_down()
    from larkhelm.crew import cancel_all_crews, wait_crews_done
    cancel_all_crews(reason="服务升级中，Crew 任务重启后将自动恢复")
    wait_crews_done(timeout=30.0)
    idle = wait_for_idle(timeout=60.0)

    # If we timed out, notify affected chats so their streaming cards don't hang silently
    if not idle:
        from larkhelm.concurrency import get_busy_chat_ids
        for busy_cid in get_busy_chat_ids():
            try:
                send_card(busy_cid, "⚠️ 查询已中断",
                          "服务正在升级重启，当前查询被中断，请稍后重新发送。",
                          color="orange")
            except Exception as e:
                _debug_log(f"[upgrade] restart card failed: {e}")

    # Write a marker file so the new process can confirm back to the upgrade requester
    import json as _json
    _notify_path = _cfg.DATA_DIR / "_restart_notify.json"
    try:
        _notify_path.write_text(
            _json.dumps({"chat_id": chat_id, "ts": time.time()}),
            encoding="utf-8",
        )
    except Exception as e:
        _debug_log(f"[Upgrade] failed to write restart notify: {e}")

    # Step 3: replace process in-place with os.execv (PID unchanged, transparent to systemd)
    _debug_log("[Upgrade] os.execv replacing process")
    send_card_reply(chat_id, msg_id, "🔄 升级中", "服务正在重启，连接将在数秒内恢复…", color="blue")
    time.sleep(1)   # Give send_card enough time to deliver
    _os.execv(_sys.executable, _sys.argv)

