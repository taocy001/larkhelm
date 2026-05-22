"""larkhelm · slash command registry (S1+S7 — Phase B)

Replaces the 600-line if/elif chain in ``handlers/_message.py`` with a
single dispatch table. Each command is one ``CommandSpec``; new commands
add a single ``register(...)`` call instead of editing message routing,
help text, and (for async tasks) a custom thread wrapper.

Conscious omissions: ``/cancel``, ``/rename``, ``/btw`` reply detection,
the ``/c`` / ``/g`` / ``/k`` / ``/d`` model shortcuts, and intent_router
flow remain in ``_message.py`` because they touch the per-chat lock,
chat_state, parent_id detection, or cancel-event plumbing in ways that
don't generalise. Everything else lives here.

Imports are deliberately lazy (inside handlers / inside ``_default_registrations``)
so importing this module never drags in commands.py / cmd_doc.py / crew —
keeping ``handlers/_message.py`` import time minimal.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal


DispatchResult = Literal["handled", "unhandled"]


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
            if self.match_kind == "prefix" and tl.startswith(n + " "):
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
        for sub in sorted(self.sub_matches, key=len, reverse=True):
            if tl == sub:
                return ""
            if tl.startswith(sub + " "):
                return text.lstrip()[len(sub):].strip()
        for n in sorted(self._names(), key=len, reverse=True):
            if tl == n:
                return ""
            if tl.startswith(n + " "):
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
    ))

    # ── /status / /help / /pickup / /upgrade ───────────────────────
    def _h_status(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_status
        _cmd_status(ctx.chat_id, ctx.msg_id)
    register_simple("/status", _h_status)

    def _h_help(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_help
        _cmd_help(ctx.chat_id, ctx.msg_id)
    register_simple("/help", _h_help)

    def _h_pickup(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_pickup
        _cmd_pickup(ctx.chat_id, ctx.msg_id)
    register_simple("/pickup", _h_pickup)

    def _h_upgrade(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_upgrade
        _cmd_upgrade(ctx.chat_id, ctx.msg_id)
    register_simple("/upgrade", _h_upgrade)

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
    ))

    # ── /stats / /memory / /doc / /cron ────────────────────────────
    def _h_stats(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_stats
        _cmd_stats(ctx.chat_id, ctx.msg_id, args=ctx.raw_args)
    register(CommandSpec(name="/stats", handler=_h_stats, match_kind="prefix"))

    def _h_memory(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_memory
        _cmd_memory(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(name="/memory", handler=_h_memory, match_kind="prefix"))

    def _h_doc(ctx: DispatchContext) -> None:
        from larkhelm.cmd_doc import _cmd_doc
        _cmd_doc(ctx.chat_id, ctx.raw_args)
    register(CommandSpec(name="/doc", handler=_h_doc, match_kind="prefix"))

    def _h_cron(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_cron
        _cmd_cron(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(name="/cron", handler=_h_cron, match_kind="prefix"))

    # ── /crew / /dev / /plan (async, error-carded) ─────────────────
    def _h_crew(ctx: DispatchContext) -> None:
        from larkhelm.crew import cmd_crew
        cmd_crew(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(name="/crew", handler=_h_crew, match_kind="prefix",
                         run_async=True, thread_label="Crew"))

    def _h_dev(ctx: DispatchContext) -> None:
        from larkhelm.crew import cmd_dev
        cmd_dev(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(name="/dev", handler=_h_dev, match_kind="prefix",
                         run_async=True, thread_label="Dev"))

    def _h_plan(ctx: DispatchContext) -> None:
        from larkhelm.cmd_plan import cmd_plan
        cmd_plan(ctx.chat_id, ctx.raw_args, ctx.msg_id, sender_open_id=ctx.sender_open_id)
    register(CommandSpec(name="/plan", handler=_h_plan, match_kind="prefix",
                         run_async=True, thread_label="Plan"))

    # ── /cd / /pwd / /ls / /run ────────────────────────────────────
    def _h_pwd(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_pwd
        _cmd_pwd(ctx.chat_id, ctx.msg_id)
    register_simple("/pwd", _h_pwd)

    def _h_cd(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_cd
        _cmd_cd(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/cd",
        handler=_h_cd,
        match_kind="prefix",
        usage_card="`/cd <path>` — 切换工作目录",
    ))

    def _h_ls(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_ls
        _cmd_ls(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(name="/ls", handler=_h_ls, match_kind="prefix"))

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
    ))

    # ── /model + /lock (alias) + /voice ────────────────────────────
    def _h_model(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_lock
        _cmd_lock(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(
        name="/model",
        handler=_h_model,
        match_kind="prefix",
        aliases=("/lock",),
    ))

    def _h_voice(ctx: DispatchContext) -> None:
        from larkhelm.commands import _cmd_voice
        _cmd_voice(ctx.chat_id, ctx.raw_args, ctx.msg_id)
    register(CommandSpec(name="/voice", handler=_h_voice, match_kind="prefix"))


# Register defaults at import time. Python caches the module after the first
# import, so this runs exactly once per process under normal usage. Note:
# ``importlib.reload(larkhelm.command_registry)`` rebuilds COMMAND_REGISTRY
# from scratch (re-executing module body), so reload is effectively a clean
# slate — not a no-op. We don't rely on reload in production.
_default_registrations()
