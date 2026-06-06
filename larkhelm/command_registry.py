"""larkhelm · slash command registry (S1+S7 — Phase B)

Replaces the 600-line if/elif chain in ``handlers/_message.py`` with a
single dispatch table. Each command is one ``CommandSpec``; new commands
add a single ``register(...)`` call instead of editing message routing,
help text, and (for async tasks) a custom thread wrapper.

Conscious omissions: ``/cancel``, ``/rename``, ``/btw`` reply detection,
the ``/c`` / ``/g`` / ``/k`` / ``/d`` model shortcuts remain in ``_message.py`` because they touch the per-chat lock,
chat_state, parent_id detection, or cancel-event plumbing in ways that
don't generalise. Everything else lives here.

Imports are deliberately lazy (inside handlers / inside ``_default_registrations``)
so importing this module never drags in commands.py / doc_handlers.py / crew —
keeping ``handlers/_message.py`` import time minimal.

``CommandSpec.description`` / ``examples`` field semantics and the
recommended workflow for adding a new command are documented in
CLAUDE.md §"Adding a New Command".
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal


DispatchResult = Literal["handled", "unhandled"]


def _is_arg_sep(ch: str) -> bool:
    """True iff ``ch`` separates a slash-command token from its arguments.

    Historically the dispatcher required a single ASCII space — which silently
    broke ``/plan`` (and every other ``match_kind="prefix"`` command) whenever
    a Feishu mobile user pasted a newline, tab, or full-width space (U+3000)
    after the token. The user-facing symptom: ``/plan\\n<task>`` falls through
    to the model dispatch path and gets answered by Claude's built-in /plan
    instead of larkhelm's pipeline.

    We accept any Unicode whitespace character. ``str.isspace`` is True for
    ``' '``, ``'\\t'``, ``'\\n'``, ``'\\r'``, ``'\\v'``, ``'\\f'`` and
    ``'\\u3000'`` (full-width space) among others — exactly the set we want.
    """
    return bool(ch) and ch.isspace()


# ── Data ────────────────────────────────────────────────────────────────


@dataclass
class DispatchContext:
    """Per-message dispatch state.

    ``raw_args`` is set by ``CommandRegistry.dispatch`` AFTER a spec matches —
    it's the message text minus the matched command token (and a single
    leading space). Handlers should use ``raw_args``, not ``text``, for
    user input parsing.
    """
    chat_id: str
    msg_id: str
    text: str           # original message text (preserves case)
    tl: str             # text.lower().strip() (cached so each spec doesn't re-lower)
    raw_args: str = ""
    sender_open_id: str = ""  # open_id of the message sender (MEM-C1)


@dataclass(frozen=True)
class CommandSpec:
    """Metadata for a single slash command.

    ``match_kind``:
      * ``"exact"`` — ``tl`` must equal ``name`` (or one of ``aliases``)
      * ``"prefix"`` — ``tl`` must start with ``name`` followed by space, OR
        equal ``name`` exactly. ``raw_args`` is everything after the matched
        command token.

    ``sub_matches`` are checked first (exact match) — useful for variants
    like ``/reset claude`` / ``/reset gemini`` where the dispatcher wants
    to pick a single handler but the sub-command identifies a strategy.
    A handler can read the matched sub by inspecting ``ctx.tl``.

    ``usage_card`` is sent (orange) when ``match_kind == "prefix"`` AND
    ``raw_args`` ends up empty — saves N copies of the same orange-usage
    boilerplate.

    ``run_async=True`` wraps the handler in a daemon thread, surrounded by
    ``_thread_error_card(chat_id, thread_label, exc)``. Use for any
    handler that may block on I/O for more than ~1s.

    ``description`` is a ≤80-character one-line summary of the command
    (single source of truth for future help renderers — must read like
    the README "聊天命令" section). ``examples`` is 0–3 paste-ready
    usage samples (each ≤120 chars, no placeholders like ``<path>``,
    no embedded newlines / ``\\r``). Both fields are pure metadata —
    ``dispatch`` / ``matches`` / ``extract_args`` never read them. They
    exist for the upcoming ``/help`` auto-renderer and third-party
    plugins.
    """
    name: str
    handler: Callable[[DispatchContext], None]
    match_kind: Literal["exact", "prefix"] = "exact"
    aliases: tuple[str, ...] = ()
    sub_matches: tuple[str, ...] = ()
    usage_card: str | None = None
    run_async: bool = False
    thread_label: str = ""
    hidden: bool = False
    description: str = ""
    examples: tuple[str, ...] = ()

    # ── matching ───────────────────────────────────────────────────

    def _names(self) -> tuple[str, ...]:
        return (self.name,) + tuple(self.aliases)

    def matches(self, tl: str) -> bool:
        """Return True iff this spec wants to handle ``tl``."""
        if not tl:
            return False
        for sub in self.sub_matches:
            if tl == sub:
                return True
        for n in self._names():
            if tl == n:
                return True
            # Prefix match: ``tl`` must begin with the command token followed
            # by ANY Unicode whitespace (space, tab, newline, full-width
            # space). See ``_is_arg_sep`` for why we don't restrict to ASCII
            # space — Feishu mobile pastes routinely contain ``\n`` and
            # ``　`` between the slash command and its argument.
            if (self.match_kind == "prefix"
                    and len(tl) > len(n)
                    and tl.startswith(n)
                    and _is_arg_sep(tl[len(n)])):
                return True
        return False

    def extract_args(self, text: str) -> str:
        """Return the message text with the matched command token stripped.

        Used to populate ``DispatchContext.raw_args`` before calling the
        handler. We strip case-insensitively (the text can be mixed case)
        but preserve the original case of arguments — eg. ``/cd /Foo/Bar``
        yields ``"/Foo/Bar"`` not ``"/foo/bar"``.
        """
        tl = text.lower().lstrip()
        # Try sub_matches first (longest-first to avoid /reset eating /reset claude).
        # Separator check mirrors ``matches`` — any Unicode whitespace counts,
        # so ``/cmd\n<args>`` and ``/cmd　<args>`` parse the same as ``/cmd <args>``.
        for sub in sorted(self.sub_matches, key=len, reverse=True):
            if tl == sub:
                return ""
            if (len(tl) > len(sub)
                    and tl.startswith(sub)
                    and _is_arg_sep(tl[len(sub)])):
                return text.lstrip()[len(sub):].strip()
        for n in sorted(self._names(), key=len, reverse=True):
            if tl == n:
                return ""
            if (len(tl) > len(n)
                    and tl.startswith(n)
                    and _is_arg_sep(tl[len(n)])):
                return text.lstrip()[len(n):].strip()
        return ""


# ── Registry ────────────────────────────────────────────────────────────


class CommandRegistry:
    """Module-singleton registry. Order of insertion matters for ``_find_match``
    — register longer / more-specific names first if they share a prefix
    with another spec (we sort by length internally as a safety net).
    """

    def __init__(self):
        self._by_name: dict[str, CommandSpec] = {}
        self._ordered: list[CommandSpec] = []
        self._lock = threading.Lock()

    def register(self, spec: CommandSpec) -> None:
        """Add ``spec``. Raises ``ValueError`` on duplicate primary name."""
        with self._lock:
            if spec.name in self._by_name:
                raise ValueError(f"command already registered: {spec.name}")
            self._by_name[spec.name] = spec
            self._ordered.append(spec)

    def lookup(self, name: str) -> CommandSpec | None:
        return self._by_name.get(name)

    def iter_visible(self) -> Iterator[CommandSpec]:
        for spec in self._ordered:
            if not spec.hidden:
                yield spec

    def _find_match(self, tl: str) -> CommandSpec | None:
        """Resolve which spec (if any) wants to handle ``tl``.

        Iteration order favours longer command tokens — ``/reset claude``
        wins over ``/reset`` even if both registered. (We rely on each
        spec's own sub_matches list for real disambiguation; this is just
        an extra safety net.)
        """
        # Pass 1: exact-match wins immediately.
        for spec in self._ordered:
            if spec.matches(tl) and spec.match_kind == "exact":
                return spec
        # Pass 2: prefix matches, longest name first.
        prefix_matches = [s for s in self._ordered
                          if s.match_kind == "prefix" and s.matches(tl)]
        if not prefix_matches:
            return None
        prefix_matches.sort(key=lambda s: len(s.name), reverse=True)
        return prefix_matches[0]

    def dispatch(self, ctx: DispatchContext) -> DispatchResult:
        """Dispatch ``ctx`` to the matching spec; returns ``"handled"`` /
        ``"unhandled"``. Caller should treat ``"unhandled"`` as "fall through
        to the model dispatch path"."""
        spec = self._find_match(ctx.tl)
        if spec is None:
            return "unhandled"
        ctx.raw_args = spec.extract_args(ctx.text)
        # Empty-arg usage card path: skips the handler entirely.
        if (spec.match_kind == "prefix"
                and not ctx.raw_args
                and spec.usage_card
                and ctx.tl == spec.name.lower()):
            try:
                # Route via ``handlers._message`` so existing test mocks
                # patching ``_m.send_card_reply`` continue to catch the call.
                from larkhelm.handlers import _message as _m
                _m.send_card_reply(ctx.chat_id, ctx.msg_id, "⚠️ 用法",
                                   spec.usage_card, color="orange")
            except Exception as e:
                from larkhelm.log import _debug_log
                _debug_log(f"[CommandRegistry] usage_card send failed: {e}")
            return "handled"
        self._invoke(spec, ctx)
        return "handled"

    def _invoke(self, spec: CommandSpec, ctx: DispatchContext) -> None:
        if spec.run_async:
            label = spec.thread_label or spec.name.lstrip("/").capitalize()

            def _wrap():
                try:
                    spec.handler(ctx)
                except Exception as exc:
                    try:
                        from larkhelm.handlers._message import _thread_error_card
                        _thread_error_card(ctx.chat_id, label, exc)
                    except Exception as inner:
                        from larkhelm.log import _debug_log
                        _debug_log(f"[CommandRegistry] error-card recursion failed: {inner}")

            t = threading.Thread(target=_wrap, daemon=True,
                                 name=f"{label.lower()}-{ctx.chat_id[:8]}")
            t.start()
        else:
            spec.handler(ctx)


# ── Module-level singleton ─────────────────────────────────────────────

COMMAND_REGISTRY = CommandRegistry()


def register(spec: CommandSpec) -> None:
    COMMAND_REGISTRY.register(spec)


def register_simple(
    name: str,
    handler: Callable[[DispatchContext], None],
    *,
    aliases: tuple[str, ...] = (),
    match_kind: Literal["exact", "prefix"] = "exact",
    sub_matches: tuple[str, ...] = (),
    usage_card: str | None = None,
    run_async: bool = False,
    thread_label: str = "",
    hidden: bool = False,
    description: str = "",
    examples: tuple[str, ...] = (),
) -> None:
    COMMAND_REGISTRY.register(CommandSpec(
        name=name,
        handler=handler,
        aliases=aliases,
        match_kind=match_kind,
        sub_matches=sub_matches,
        usage_card=usage_card,
        run_async=run_async,
        thread_label=thread_label,
        hidden=hidden,
        description=description,
        examples=examples,
    ))


def dispatch(ctx: DispatchContext) -> DispatchResult:
    return COMMAND_REGISTRY.dispatch(ctx)


def iter_visible() -> Iterator[CommandSpec]:
    return COMMAND_REGISTRY.iter_visible()


def lookup(name: str) -> CommandSpec | None:
    return COMMAND_REGISTRY.lookup(name)


# ── Default registrations ──────────────────────────────────────────────


def _default_registrations() -> None:
    """Wire up the 30+ existing commands.

    Lazy imports inside each handler keep this module's import cost flat
    (handlers/_message.py imports it on the hot path).
    """

    # ── /reset (and family) ────────────────────────────────────────
    def _h_reset(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_reset
        which: str | None = None
        if ctx.tl == "/reset claude":
            which = "claude"
        elif ctx.tl == "/reset gemini":
            which = "gemini"
        elif ctx.tl == "/reset kimi":
            which = "kimi"
        elif ctx.tl == "/reset deepseek":
            which = "deepseek"
        elif ctx.tl in ("/reset permissions", "/reset perm"):
            which = "perm"
        elif ctx.tl == "/reset memory":
            which = "memory"
        _cmd_reset(ctx.chat_id, which, ctx.msg_id)

    register(CommandSpec(
        name="/reset",
        handler=_h_reset,
        match_kind="exact",
        sub_matches=(
            "/reset claude", "/reset gemini", "/reset kimi", "/reset deepseek",
            "/reset permissions", "/reset perm", "/reset memory",
        ),
        description="重置会话（按子命令清除指定 backend / 权限 / 记忆）",
        examples=("/reset", "/reset claude", "/reset memory"),
    ))

    # ── /status / /help / /pickup / /upgrade ───────────────────────
    def _h_status(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_status
        _cmd_status(ctx.chat_id, ctx.msg_id)
    register_simple("/status", _h_status,
                    description="查看服务运行状态：版本 / session ID / backend 健康")

    def _h_help(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_help
        _cmd_help(ctx.chat_id, ctx.msg_id)
    register_simple("/help", _h_help,
                    description="显示命令帮助卡片")

    def _h_pickup(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_pickup
        _cmd_pickup(ctx.chat_id, ctx.msg_id)
    register_simple("/pickup", _h_pickup,
                    description="获取在终端接力当前 Claude / Gemini / Kimi 会话的命令")

    def _h_upgrade(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_upgrade
        _cmd_upgrade(ctx.chat_id, ctx.msg_id)
    register_simple("/upgrade", _h_upgrade,
                    description="更新 larkhelm 到最新版本")

    # ── /history ───────────────────────────────────────────────────
    def _h_history(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_history
        if ctx.tl == "/history all":
            _cmd_history(ctx.chat_id, show_all=True, msg_id=ctx.msg_id)
        else:
            _cmd_history(ctx.chat_id, msg_id=ctx.msg_id)
    register(CommandSpec(
        name="/history",
        handler=_h_history,
        match_kind="exact",
        sub_matches=("/history all",),
        description="查看当前会话历史（默认最近 10 条，加 all 查看全部）",
        examples=("/history", "/history all"),
    ))

    # ── /stats / /memory / /cron ──────────────────────────────────
    def _h_stats(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_stats
        _cmd_stats(ctx.chat_id, ctx.msg_id, args=ctx.raw_args)
    register(CommandSpec(
        name="/stats",
        handler=_h_stats,
        match_kind="prefix",
        description="查看 Token 用量统计（加 intent 子命令查看意图路由分布）",
        examples=("/stats", "/stats intent"),
    ))

    def _h_memory(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_memory
        _cmd_memory(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(
        name="/memory",
        handler=_h_memory,
        match_kind="prefix",
        description="查看 / 管理三层记忆（全局 / 项目 / 会话）",
        examples=("/memory status", "/memory set global 偏好简短回答", "/memory clear session"),
    ))

    def _h_cron(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_cron
        _cmd_cron(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/cron",
        handler=_h_cron,
        match_kind="prefix",
        description="管理定时查询任务（add / list / del）",
        examples=('/cron add "0 9 * * *" 早会提醒', "/cron list", "/cron del abc123"),
    ))

    # ── /crew / /dev / /plan (async, error-carded) ─────────────────
    def _h_crew(ctx: DispatchContext) -> None:
        from larkhelm.crew import cmd_crew
        cmd_crew(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(
        name="/crew",
        handler=_h_crew,
        match_kind="prefix",
        run_async=True,
        thread_label="Crew",
        description="动态规划：Manager 自动分解任务，多 Agent 并行执行",
        examples=("/crew 调研竞品 X 与 Y 的差异",),
    ))

    def _h_dev(ctx: DispatchContext) -> None:
        from larkhelm.crew import cmd_dev
        cmd_dev(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(
        name="/dev",
        handler=_h_dev,
        match_kind="prefix",
        run_async=True,
        thread_label="Dev",
        description="软件工程流水线：PM → 架构 → 工程 → QA → 审查",
        examples=("/dev 给登录页加 SSO 支持", "/dev 修复登录超时 bug --no-confirm"),
    ))

    def _h_plan(ctx: DispatchContext) -> None:
        from larkhelm.cmd_plan import cmd_plan
        cmd_plan(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(
        name="/plan",
        handler=_h_plan,
        match_kind="prefix",
        run_async=True,
        thread_label="Plan",
        description="多阶段串行流水线：dev → review → fix → test，每步可确认",
        examples=("/plan 给设置页加深色模式",),
    ))

    # ── /cd / /pwd / /ls / /run ────────────────────────────────────
    def _h_pwd(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_pwd
        _cmd_pwd(ctx.chat_id, ctx.msg_id)
    register_simple("/pwd", _h_pwd,
                    description="显示当前工作目录")

    def _h_cd(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_cd
        _cmd_cd(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/cd",
        handler=_h_cd,
        match_kind="prefix",
        usage_card="`/cd <path>` — 切换工作目录",
        description="切换当前会话的工作目录",
        examples=("/cd /home/user/code/larkhelm",),
    ))

    def _h_ls(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_ls
        _cmd_ls(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/ls",
        handler=_h_ls,
        match_kind="prefix",
        description="列出目录文件（默认当前目录，最多 60 条）",
        examples=("/ls", "/ls /tmp"),
    ))

    def _h_run(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_run
        _cmd_run(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/run",
        handler=_h_run,
        match_kind="prefix",
        usage_card="`/run <command>` — 执行 shell 命令（30s 超时）",
        run_async=True,
        thread_label="Run",
        description="执行 Shell 命令（默认 30s 超时，可配 shell_timeout_sec）",
        examples=("/run uname -a", "/run df -h"),
    ))

    # ── /model + /lock (twin specs, shared handler) + /voice ───────
    # P1-2c: /lock used to be an alias of /model. Pulling it out as an
    # independent CommandSpec lets the help renderer (which iterates
    # COMMAND_REGISTRY by name) surface both entries — see design.md D3.
    def _h_model(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_lock
        _cmd_lock(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/model",
        handler=_h_model,
        match_kind="prefix",
        description="切换当前会话默认 backend（list / off / <id>）",
        examples=("/model claude", "/model kimi"),
    ))
    register(CommandSpec(
        name="/lock",
        handler=_h_model,
        match_kind="prefix",
        description="持久锁定 backend（list / off / <id>，/model 的同义入口）",
        examples=("/lock", "/lock kimi", "/lock off"),
    ))

    def _h_voice(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_voice
        _cmd_voice(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/voice",
        handler=_h_voice,
        match_kind="prefix",
        description="查看 / 切换语音转写设置（status / lang zh|en|auto）",
        examples=("/voice status", "/voice lang en"),
    ))


# Register defaults at import time. Python caches the module after the first
# import, so this runs exactly once per process under normal usage. Note:
# ``importlib.reload(larkhelm.command_registry)`` rebuilds COMMAND_REGISTRY
# from scratch (re-executing module body), so reload is effectively a clean
# slate — not a no-op. We don't rely on reload in production.
_default_registrations()
