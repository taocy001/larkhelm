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
from larkhelm.perm import _perm_yolo, revoke_yolo
from larkhelm.cmd_doc import _cmd_doc, _cmd_doc_write_do
from larkhelm.lark_client import (
    send_card, send_card_reply, reply_card, _send_card_raw, _patch_card_raw, _reply_card_raw,
    react_to_message, delete_reaction,
    EMOJI_PROCESSING, EMOJI_DONE, EMOJI_ERROR,
    send_permission_guide,
)
from larkhelm.card_builder import _make_card, _fmt_elapsed
from larkhelm.ai_runner import query_gemini, _spawn_claude_proc


# ═══════════════════════════════════════════════════
#  Shell command execution
# ═══════════════════════════════════════════════════

def _run_shell(chat_id: str, cmd: str) -> tuple[str, str, int]:
    cwd = _get_cwd(chat_id)
    try:
        import shlex
        args = shlex.split(cmd)
    except ValueError as e:
        return "", f"命令格式错误: {e}", 1
    try:
        r = subprocess.run(
            args, shell=False, capture_output=True, text=True,
            timeout=30, cwd=cwd, env={**__import__("os").environ}
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

def _cmd_reset(chat_id: str, which: str | None = None, msg_id: str | None = None):
    """Unified reset logic. which=None resets everything; otherwise 'claude'/'gemini'/'perm'."""
    if which is None:
        _clear_sid(chat_id, "claude")
        _clear_sid(chat_id, "gemini")
        log_entry(chat_id, "reset", "reset:all", model="system")
        send_card_reply(chat_id, msg_id, "♻️ 已重置", "Claude 和 Gemini 会话均已清空。", color="green")
    elif which == "claude":
        _clear_sid(chat_id, "claude")
        log_entry(chat_id, "reset", "reset:claude", model="system")
        send_card_reply(chat_id, msg_id, "♻️ 已重置", "Claude 会话已清空。", color="green")
    elif which == "gemini":
        _clear_sid(chat_id, "gemini")
        log_entry(chat_id, "reset", "reset:gemini", model="system")
        send_card_reply(chat_id, msg_id, "♻️ 已重置", "Gemini 会话已清空。", color="green")
    elif which == "kimi":
        _clear_sid(chat_id, "kimi")
        log_entry(chat_id, "reset", "reset:kimi", model="system")
        send_card_reply(chat_id, msg_id, "♻️ 已重置", "Kimi 会话已清空。", color="green")
    elif which in ("perm", "permissions"):
        revoke_yolo(chat_id)
        send_card_reply(chat_id, msg_id, "🔐 权限已重置", "「允许所有」已取消，后续工具调用将重新弹出审批。", color="green")


def _cmd_status(chat_id: str, msg_id: str | None = None):
    s_c = _load_sid(chat_id, "claude")
    s_g = _load_sid(chat_id, "gemini")
    cwd = _get_cwd(chat_id)
    model = _get_chat_model(chat_id)

    def _ver(cmd):
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, timeout=15, text=True)
            return (r.stdout.strip() or r.stderr.strip()).split("\n")[0] if r.returncode == 0 else None
        except Exception:
            return None

    s_k = _load_sid(chat_id, "kimi")
    cv, gv, kv = _ver(_cfg.CLAUDE_CMD), _ver(_cfg.GEMINI_CMD), _ver(_cfg.KIMI_CMD)

    def _cli_status(ver, sid, name):
        if not ver:
            return f"❌ {name} 不可用"
        if sid:
            return f"✅ {name}  会话 `{sid[:12]}…`"
        return f"✅ {name}  暂无会话"

    if _cfg.SKIP_PERMISSIONS:
        perm_status = "⏭️ 跳过（skip_permissions=true）"
    elif chat_id in _perm_yolo:
        perm_status = "🚀 允许所有（发送 `/reset perm` 恢复审批）"
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
            crew_info = f"**Crew 进行中** `{crew_id[:8]}…`　{_phase_label}　{n_done}/{n_total} 完成"
        elif crew_id:
            crew_info = f"**Crew 进行中** `{crew_id[:8]}…`"
    except Exception:
        pass

    # Token summary (current process lifetime, current chat)
    token_summary = ""
    stats = get_token_stats(chat_id)
    if stats:
        parts = []
        for mdl, m in stats.items():
            cost_str = f" ${m['cost_usd']:.3f}" if m["cost_usd"] else ""
            parts.append(f"{mdl} {m['input_tokens']+m['output_tokens']:,}tok{cost_str}")
        token_summary = "　　".join(parts)

    lines = [
        f"**模型** {model}　　**目录** `{cwd}`"
        + (f"　　**会话名** {_get_chat_state(chat_id).get('name', '').replace('**','').replace('`','')}"
           if _get_chat_state(chat_id).get('name') else ""),
        "",
        _cli_status(cv, s_c, "Claude"),
        _cli_status(gv, s_g, "Gemini"),
        _cli_status(kv, s_k, "Kimi"),
        "",
        f"**权限模式** {perm_status}",
        *([ crew_info ] if crew_info else []),
        *([ f"**Token（本次启动）** {token_summary}" ] if token_summary else []),
        "",
    ]

    tips = []
    if not s_c and not s_g and not s_k:
        tips.append("💡 直接发消息开始第一次对话，会自动建立会话")
    else:
        tips.append("💡 `/pickup` — 获取在终端接力会话的命令")
        tips.append("💡 `/reset` — 清除会话，开始全新对话")
    if model == "claude":
        tips.append("💡 `/model gemini` — 切换默认模型为 Gemini")
    elif model == "gemini":
        tips.append("💡 `/model claude` — 切换默认模型为 Claude")
    else:
        tips.append("💡 `/model claude` — 切换默认模型为 Claude")

    lines += tips
    _next_models = {"claude": "gemini", "gemini": "kimi", "kimi": "claude"}
    other_model = _next_models.get(model, "claude")
    buttons = [
        ("♻️ 重置会话", "/reset"),
        ("🔗 接入终端", "/pickup"),
        (f"切换 {other_model}", f"/model {other_model}"),
    ]
    send_card_reply(chat_id, msg_id, "📊 运行状态", "\n".join(lines), color="turquoise", buttons=buttons)


def _cmd_help(chat_id: str, msg_id: str | None = None):
    model = _get_chat_model(chat_id)
    _next_models = {"claude": "gemini", "gemini": "kimi", "kimi": "claude"}
    other = _next_models.get(model, "claude")
    body = (
        f"**当前模型:** {model}　　发消息直接提问，命令均以 `/` 开头\n"
        "\n"
        "**🚀 常用操作**\n"
        f"`/reset` — 重置会话，开始新对话\n"
        f"`/pickup` — 获取在终端接力会话的命令\n"
        f"`/model {other}` — 切换到 {other}\n"
        f"`/cd 路径` — 切换工作目录\n"
        f"`/cancel` — 取消当前查询\n"
        f"`/rename <名称>` — 给当前会话命名\n"
        "\n"
        "---\n"
        "\n"
        "**查询**\n"
        "`/c 消息` `/claude 消息` — 强制用 Claude\n"
        "`/g 消息` `/gemini 消息` — 强制用 Gemini\n"
        "`/k 消息` `/kimi 消息` — 强制用 Kimi\n"
        "\n"
        "**会话**\n"
        "`/reset claude` / `/reset gemini` / `/reset kimi` — 单独重置\n"
        "`/reset perm` — 重置权限审批\n"
        "`/model claude` / `/model gemini` / `/model kimi` — 切换默认模型\n"
        "\n"
        "**目录 & Shell**\n"
        "`/pwd` `/ls [路径]` `/run 命令`（30s 超时）\n"
        "\n"
        "**其他**\n"
        "`/status` — 查看运行状态　　`/help` — 此帮助\n"
        "`/history` — 最近 10 条对话摘要　　`/stats` — 今日统计\n"
        "`/cron add \"<expr>\" <查询>` / `/cron list` / `/cron del <id>` — 定时任务\n"
        "`/doc read/append/write/create/setfolder` — 飞书文档操作\n"
        "\n"
        "**🤖 多 Agent 协作**\n"
        "`/crew <需求>` — 动态规划：Manager 自动分解任务，多 Agent 并行执行\n"
        "`/dev <需求>` — 软件工程流水线：\n"
        "　　PM → **[确认]** → 架构师 → 工程师 → QA（失败重试 2×）→ 审查员（APPROVED / REJECTED 重试 1×）\n"
        "`/btw <问题>` — 快问（不占主任务锁，回复到消息线程）"
    )
    buttons = [
        ("♻️ 重置会话", "/reset"),
        ("🔗 接入终端", "/pickup"),
        ("📊 查看状态", "/status"),
        ("🛑 取消查询", "/cancel"),
        (f"切换 {other}", f"/model {other}"),
    ]
    send_card_reply(chat_id, msg_id, "📖 帮助", body, color="blue", buttons=buttons)


def _cmd_pickup(chat_id: str, msg_id: str | None = None):
    s_c  = _load_sid(chat_id, "claude")
    s_g  = _load_sid(chat_id, "gemini")
    s_k  = _load_sid(chat_id, "kimi")
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
    lines.append("\n> 在终端运行上面命令即可无缝接力")
    send_card_reply(chat_id, msg_id, "🔗 终端接力", "\n".join(lines), color="purple")


def _cmd_history(chat_id: str, show_all: bool = False, msg_id: str | None = None):
    """Display conversation history.
    By default shows only the current session since the last reset;
    show_all=True shows all records with separator lines at reset points.
    """
    records = _read_logs(chat_id)

    def _build_pairs(recs: list[dict]) -> list[tuple[dict, dict]]:
        pairs: list[tuple[dict, dict]] = []
        pending_user: dict | None = None
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
        pending_user: dict | None = None
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


def _cmd_stats(chat_id: str, msg_id: str | None = None):
    """Display today / this-month / all-time token stats plus conversation activity for the current chat."""
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


def _cmd_cron(chat_id: str, args: str, msg_id: str | None = None):
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


def _cmd_cd(chat_id: str, path: str, msg_id: str | None = None):
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(_get_cwd(chat_id)) / p
        p = p.resolve()
        if not p.is_dir():
            send_card_reply(chat_id, msg_id, "❌ 目录不存在", f"`{p}`", color="red")
            return
        _set_chat_field(chat_id, "cwd", str(p))
        send_card_reply(chat_id, msg_id, "📁 目录已切换", f"`{p}`", color="green")
    except Exception as e:
        send_card_reply(chat_id, msg_id, "❌ 错误", str(e), color="red")


def _cmd_pwd(chat_id: str, msg_id: str | None = None):
    send_card_reply(chat_id, msg_id, "📁 当前目录", f"`{_get_cwd(chat_id)}`", color="blue")


def _cmd_ls(chat_id: str, path: str = "", msg_id: str | None = None):
    cwd = _get_cwd(chat_id)
    target = (Path(cwd) / path if path else Path(cwd)).resolve()
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


def _cmd_run(chat_id: str, cmd: str, msg_id: str | None = None):
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


def _cmd_model(chat_id: str, model_name: str, msg_id: str | None = None):
    model_name = model_name.lower().strip()
    if model_name not in ("claude", "gemini", "kimi"):
        send_card_reply(chat_id, msg_id, "⚠️ 无效模型",
                        f"可用: `claude` / `gemini` / `kimi`\n当前: `{_get_chat_model(chat_id)}`",
                        color="orange")
        return
    _set_chat_field(chat_id, "model", model_name)
    send_card_reply(chat_id, msg_id, "✅ 模型已切换",
                    f"本 chat 默认模型: **{model_name}**", color="green")


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


def _cmd_cli_native(chat_id: str, model: str, cmd: str, msg_id: str | None = None):
    """Handle /c /<cmd>, /g /<cmd>, or /k /<cmd>: passthrough channel for AI CLI native commands."""
    cmd_lower = cmd.lower().strip()
    m_name    = {"claude": "Claude", "gemini": "Gemini", "kimi": "Kimi"}.get(model, model.capitalize())

    if cmd_lower in ("/help", "--help", "-h"):
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

    _all_cmds = {"claude": _CLAUDE_SESSION_CMDS, "gemini": _GEMINI_SESSION_CMDS, "kimi": _KIMI_SESSION_CMDS}
    _pfx_map  = {"claude": "/c", "gemini": "/g", "kimi": "/k"}
    my_cmds   = _all_cmds.get(model, set())
    my_pfx    = _pfx_map.get(model, "/c")
    # Collect all other models' commands for cross-model detection
    other_info: list[tuple[str, str, str]] = [
        (nm, pfx, cmds)
        for nm, (pfx, cmds) in {
            "Claude": ("/c", _CLAUDE_SESSION_CMDS),
            "Gemini": ("/g", _GEMINI_SESSION_CMDS),
            "Kimi":   ("/k", _KIMI_SESSION_CMDS),
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
            cwd   = _get_cwd(chat_id)
            model = _get_chat_model(chat_id)
            if main_free:
                sid_key = "claude" if model == "claude" else "gemini"
            else:
                sid_key = f"btw_{model}"
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

                if model == "gemini":
                    output = query_gemini(chat_id, question, cwd, None, None, _on_text)
                else:
                    output = _spawn_claude_proc(
                        chat_id=chat_id, message=question, sid=sid, cwd=cwd,
                        cancel_ev=None, on_text=_on_text, allow_retry=True,
                    )

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



def _cmd_upgrade(chat_id: str, msg_id: str | None = None):
    """/upgrade: pull latest code and restart the service in-place via os.execv (PID unchanged)."""
    threading.Thread(target=_do_upgrade, args=(chat_id, msg_id), daemon=True,
                     name="upgrade").start()


def _do_upgrade(chat_id: str, msg_id: str | None = None):
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
            except Exception:
                pass

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

