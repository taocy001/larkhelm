"""larkhelm · agent_hub · agent dispatcher.

Glue between :class:`IntentResult` and the registered :class:`AgentExecutor`.
Responsibilities:

  * ACL check (``intent.agent_type`` × ``chat_id`` glob)
  * Transparent intent card with a "switch to plain chat" button
  * Calls ``executor.execute(intent, ctx)``
  * Writes audit log; falls back to ChatAgent on errors (NFR-SEC-02)
"""
from __future__ import annotations

import fnmatch
import time

from larkhelm.agent_hub.agent_audit import write_audit
from larkhelm.agent_hub.agent_base import AGENT_REGISTRY, AgentExecutor, AgentRegistry
from larkhelm.agent_hub.intent_feedback import (
    _new_feedback_id, consume_dispatch, record_signal, register_pending,
    track_dispatch,
)
from larkhelm.agent_hub.intent_types import (
    AgentContext, AgentResult, IntentResult,
)


# Phase D follow-up: reswitch window in seconds. When the same chat
# dispatches a different agent_type within this window of a previous
# dispatch, the prior dispatch is logged as misrouted. 120s is wider
# than the cancel window (60s) because users sometimes wait through a
# brief "wrong" /dev disclosure card before manually switching.
_RESWITCH_WINDOW_SEC: float = 120.0
# Centralized helper; previously this module re-defined a local ``_safe_log``
# duplicating the same 7-line wrapper as 3 other agent_hub/ modules.
from larkhelm.log import safe_log as _safe_log


_AGENT_TITLE = {
    "dev": "🛠 Dev Agent",
    "crew": "👥 Crew Agent",
    "plan": "📋 Plan Agent",
    "doc": "📄 Doc Agent",
    "chat": "💬 Chat",
    "search": "🔎 Search",
}


class AgentDispatcher:
    def __init__(
        self,
        registry: AgentRegistry | None = None,
        acl: dict[str, list[str]] | None = None,
    ) -> None:
        self._registry = registry or AGENT_REGISTRY
        if acl is None:
            try:
                import larkhelm.config as _cfg
                acl = (getattr(_cfg, "config", {}) or {}).get("agent_acl") or {}
            except Exception:
                acl = {}
        # Snapshot the ACL once at construction time. Live config edits don't
        # reflow into already-instantiated dispatchers — admins should restart
        # the bridge after editing agent_acl, same convention as backend changes.
        self._acl = dict(acl) if acl else {}

    # ── ACL ──
    def _check_acl(self, agent_type: str, chat_id: str) -> bool:
        rules = self._acl.get(agent_type)
        if not rules:
            return True
        for pattern in rules:
            try:
                if fnmatch.fnmatch(chat_id, pattern):
                    return True
            except Exception:
                continue
        return False

    # ── Transparent intent card ──
    def _show_intent_card(self, intent: IntentResult, ctx: AgentContext) -> str:
        """Send the disclosure card with a "force chat" button. Returns ``feedback_id``."""
        feedback_id = _new_feedback_id()
        register_pending(feedback_id, intent, ctx, text=ctx.text)

        title = _AGENT_TITLE.get(intent.agent_type, f"🤖 {intent.agent_type}")
        body_lines = [
            f"已识别为 **{intent.agent_type}** 任务",
            f"复杂度：{intent.complexity}　·　置信度：{intent.confidence:.2f}　·　层级：{intent.layer}",
        ]
        if intent.reasoning:
            body_lines.append(f"\n> {intent.reasoning[:200]}")

        try:
            from larkhelm.lark_client import send_card
            mid = send_card(
                ctx.chat_id, title, "\n".join(body_lines), color="blue",
                buttons=[("💬 切换为普通对话", f"force_chat:{feedback_id}")],
            )
            return mid or feedback_id
        except Exception as e:
            _safe_log(f"[AgentDispatcher] intent card send failed: {e}")
            return feedback_id

    # ── Public dispatch ──
    def dispatch(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        start = time.monotonic()

        if not self._check_acl(intent.agent_type, ctx.chat_id):
            try:
                from larkhelm.lark_client import send_card
                send_card(
                    ctx.chat_id, "🚫 ACL 拒绝",
                    f"您当前不允许使用 **{intent.agent_type}** Agent。", color="red",
                )
            except Exception as e:
                _safe_log(f"[AgentDispatcher] ACL deny card failed: {e}")
            result = AgentResult(success=False, error="ACL denied")
            self._emit_audit(result, intent, ctx)
            return result

        executor = self._registry.match(intent) or self._registry.get("chat")

        # Phase D follow-up: if a prior dispatch is still inside the
        # reswitch window AND it picked a different agent_type, treat
        # the prior dispatch as misrouted. consume_ pops so the same
        # prior dispatch is only billed once. Only fires for non-
        # override dispatches (force_chat itself is already recorded
        # via record_feedback in handlers/_card_action.py — wiring it
        # here too would double-count). We pass ``corrected`` = the new
        # agent_type since the user just "switched to" that agent.
        if intent.layer != "override":
            try:
                prior = consume_dispatch(ctx.chat_id, max_age_sec=_RESWITCH_WINDOW_SEC)
                if (prior is not None
                        and prior[0].agent_type != intent.agent_type):
                    prior_intent, prior_text, age = prior
                    record_signal(
                        "agent_reswitch", prior_intent, ctx.chat_id,
                        corrected=intent.agent_type, text=prior_text,
                        metadata={
                            "elapsed_sec": round(age, 2),
                            "new_agent": intent.agent_type,
                            "new_layer": intent.layer,
                        },
                    )
            except Exception as e:
                _safe_log(f"[AgentDispatcher] reswitch detect failed: {e}")

        # Stamp THIS dispatch in the per-chat history registry BEFORE
        # running the executor so a fast /cancel from the user (or a
        # subsequent /c /chat reswitch) attributes correctly. Force-chat
        # overrides are not tracked — they are themselves the correction
        # signal, tracking them would create reswitch noise on the next
        # message.
        if intent.layer != "override":
            try:
                track_dispatch(ctx.chat_id, intent, text=ctx.text)
            except Exception as e:
                _safe_log(f"[AgentDispatcher] track_dispatch failed: {e}")

        # Intent disclosure card is suppressed in two cases:
        #   1. Explicit slash commands ("/dev …") — the user already chose
        #      the agent, redundant disclosure adds noise.
        #   2. layer == "override" — this dispatch came from the user clicking
        #      "force_chat" on a previous card; showing another card would
        #      loop the disclosure UX and risk re-prompting forever.
        if not intent.is_explicit_command and intent.layer != "override":
            self._show_intent_card(intent, ctx)

        if executor is None:
            _safe_log(f"[AgentDispatcher] no agent registered for {intent.agent_type!r}")
            result = AgentResult(success=False, error=f"agent {intent.agent_type!r} not registered")
            result.duration_sec = time.monotonic() - start
            self._emit_audit(result, intent, ctx)
            return result

        try:
            result = executor.execute(intent, ctx)
            if not isinstance(result, AgentResult):
                result = AgentResult(success=True, output=str(result))
        except Exception as e:
            _safe_log(f"[AgentDispatcher] {intent.agent_type} execute exception: {e}")
            result = self._fallback_to_chat(intent, ctx, error=str(e))

        if result.duration_sec == 0.0:
            result.duration_sec = time.monotonic() - start
        self._emit_audit(result, intent, ctx)
        return result

    def _fallback_to_chat(self, intent: IntentResult, ctx: AgentContext, error: str) -> AgentResult:
        chat_agent = self._registry.get("chat")
        # Phase D follow-up: a dispatched non-chat agent that fell back to
        # chat is by definition a misroute (the user's prompt actually
        # got answered by chat). Capture as ``dispatch_failed`` with the
        # exception class in metadata. ``ACL denied`` already returns
        # before this method, so dispatch_failed never duplicates ACL
        # rejects. Chat→chat fallbacks are no-ops (the predicted agent
        # IS chat), filtered to avoid self-correction noise.
        if intent.agent_type != "chat":
            try:
                record_signal(
                    "dispatch_failed", intent, ctx.chat_id,
                    corrected="chat", text=ctx.text,
                    metadata={"error": (error or "")[:200],
                              "fallback_agent": "chat"},
                )
            except Exception as _re:
                _safe_log(f"[AgentDispatcher] record_signal dispatch_failed failed: {_re}")
            # Pop the dispatch-history entry so a subsequent /cancel or
            # backend-override slash command can't ALSO bill this same
            # already-failed dispatch as cancel_after_dispatch / reswitch.
            try:
                consume_dispatch(ctx.chat_id, max_age_sec=0.0)
            except Exception as _ce:
                _safe_log(f"[AgentDispatcher] consume after fallback failed: {_ce}")
        if chat_agent is None:
            return AgentResult(success=False, error=error or "no chat agent")
        try:
            fallback_intent = IntentResult(
                agent_type="chat", layer="fallback", confidence=0.0,
                reasoning=f"fallback from {intent.agent_type}: {error}",
                raw_text=intent.raw_text,
            )
            return chat_agent.execute(fallback_intent, ctx)
        except Exception as e:
            return AgentResult(success=False, error=f"{error}; fallback failed: {e}")

    def _emit_audit(self, result: AgentResult, intent: IntentResult, ctx: AgentContext) -> None:
        try:
            write_audit(result, intent, ctx)
        except Exception as e:
            _safe_log(f"[AgentDispatcher] audit emit failed: {e}")


__all__ = ["AgentDispatcher"]
