"""
larkhelm · Crew failure-card emission

Three public functions all share the same contract:

  * **Never raise.** Network / lark API / state mutation problems are
    swallowed via ``_debug_log`` so a failure-reporting bug can never
    compound the underlying failure.
  * **Best-effort UI update.** Functions push the freshest possible
    snapshot to the user via ``_crew_update_card`` (in-place patch) or
    ``send_card`` (terminal banner) so the user sees the failure even
    if the heartbeat thread has already stopped.
  * **Redact secrets.** Every exception string passes through
    :func:`larkhelm.log.redact_error` before reaching the card body.

Design reference: ``.crew_workspace/design.md`` §6.2.
"""
from __future__ import annotations

import threading
import time

from larkhelm.crew_types import AgentStatus, CrewState
from larkhelm.crew_card import _crew_update_card
from larkhelm.lark_client import send_card
from larkhelm.log import _debug_log, redact_error


# F6 (2026-05-25): per-process throttle for the standalone red banner so
# that a stuck retry loop or duplicate ``emit_agent_failure`` call (the
# function is called from the wrapper but also potentially from other
# heartbeat paths) cannot spam a chat. Keyed by ``(crew_id, agent_id)``
# so each agent gets at most ONE red banner per crew run; in-memory dict
# is sufficient — crew_id is unique per ``/crew`` invocation and the set
# never grows past ``agents_per_crew * concurrent_crews``.
_banner_lock = threading.Lock()
_banner_seen: set[tuple[str, str]] = set()


def _banner_throttle_should_send(crew_id: str, agent_id: str) -> bool:
    """Return True iff we haven't already sent a red banner for this
    (crew_id, agent_id) pair in the current process lifetime.
    """
    key = (crew_id or "", agent_id or "")
    with _banner_lock:
        if key in _banner_seen:
            return False
        _banner_seen.add(key)
    return True


def _reset_banner_throttle_for_tests() -> None:
    """Test-only hook to clear the throttle set between unit tests."""
    with _banner_lock:
        _banner_seen.clear()


# ── OOM classification ──────────────────────────────────────────────
# The runner already maintains an OOM-marker tuple; we re-import lazily
# to keep this module's import surface minimal (and to avoid a circular
# import edge with ``_runner`` for tests that monkeypatch the resolver).
def _classify_oom(exc: Exception) -> bool:
    try:
        from larkhelm.crew._runner import _is_likely_oom_error
        return _is_likely_oom_error(exc)
    except Exception:
        return False


def _safe_error_repr(exc: Exception, max_chars: int = 300) -> str:
    """Render ``exc`` for inclusion in a user-facing card body.

    Goes through :func:`redact_error` so any leaked credentials in the
    exception's ``str()`` are stripped before display. Truncated to
    ``max_chars`` to keep the card body tight.
    """
    try:
        raw = str(exc)
    except Exception:
        raw = repr(exc)
    cleaned = redact_error(raw)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"
    return cleaned


def emit_agent_failure(
    state: CrewState,
    agent_id: str,
    stage: str,
    exc: Exception,
) -> None:
    """Update an agent's ``error`` field with a redacted ⚠️ line and force
    one heartbeat push.

    ``stage`` is a short tag — currently one of ``"run"`` /
    ``"backend_select"`` / ``"oom"`` / ``"timeout"`` — that gets
    embedded in the message so the user can tell *where* the failure
    happened.

    OOM-class failures get a friendlier "内存超限" prefix because the
    underlying exception text (``rc=-9``, ``killed by OS``) is rarely
    actionable for end users.
    """
    try:
        cleaned = _safe_error_repr(exc)
        is_oom = _classify_oom(exc)
        prefix = "⚠️ 内存超限：" if is_oom else f"⚠️ {stage} 失败："
        # The agent error field is what _build_card surfaces — write the
        # short prefixed version there, then force a heartbeat push.
        msg = f"{prefix}{cleaned}"
        try:
            with state.lock:
                ag = state.agents.get(agent_id)
                if ag is not None:
                    ag.error = msg
                    # Mark FAILED only when not already terminal — a
                    # caller may have already written DONE/CANCELLED/SKIPPED.
                    # SKIPPED added defensively: the
                    # ``TASK_ALREADY_COMPLETE`` short-circuit marks downstream
                    # agents SKIPPED, and a stray failure-card emit (e.g.
                    # from a heartbeat thread that already had the agent
                    # queued) must NOT flip SKIPPED back to FAILED.
                    if ag.status not in (AgentStatus.DONE, AgentStatus.CANCELLED,
                                         AgentStatus.SKIPPED):
                        ag.status = AgentStatus.FAILED
                        if ag.end_time is None:
                            ag.end_time = time.time()
        except Exception as e:
            _debug_log(f"[Crew] emit_agent_failure: state mutation failed: {e}")

        try:
            _crew_update_card(state)
        except Exception as e:
            _debug_log(f"[Crew] emit_agent_failure: card update failed: {e}")

        # F6 (2026-05-25): in-place card patch is invisible to a user
        # whose Feishu chat scrolled past the crew card or whose
        # heartbeat thread already stopped (2026-05-25 incident: agent_3
        # + agent_6 both validate-failed, but no banner ever surfaced).
        # Send a dedicated red banner for actionable stages so the user
        # sees the failure even in those cases. Throttled by
        # (crew_id, agent_id) so a retry storm can't spam the chat.
        if stage in ("validate", "backend_select", "oom", "timeout"):
            try:
                if _banner_throttle_should_send(state.crew_id, agent_id):
                    role = ""
                    try:
                        role = state.agents[agent_id].spec.role or ""
                    except Exception:
                        pass
                    suffix = f"（{role}）" if role else ""
                    title = f"⚠️ {agent_id} 失败 · {stage}{suffix}"
                    body_lines = [f"**原因**：{cleaned}"]
                    # When a quarantined .invalid file exists, point the
                    # user at it so they can recover content manually if
                    # the synth's sanitized excerpt isn't sufficient.
                    try:
                        spec = state.agents[agent_id].spec
                        if stage == "validate" and spec.output_file:
                            body_lines.append(
                                f"\n📄 原始输出已隔离为 "
                                f"`.crew_workspace/{spec.output_file}.invalid`，"
                                "如内容有效可手动 `mv` 去掉后缀恢复。"
                            )
                    except Exception:
                        pass
                    send_card(
                        state.chat_id, title, "\n".join(body_lines),
                        color="red",
                    )
            except Exception as e:
                _debug_log(f"[Crew] emit_agent_failure: red banner failed: {e}")

        _debug_log(f"[Crew] {agent_id}: {stage} failed: {cleaned}")
    except Exception as e:
        # Last-ditch swallow — emit_agent_failure must NEVER raise.
        try:
            _debug_log(f"[Crew] emit_agent_failure outer error: {e}")
        except Exception:
            pass


def emit_terminal_failure(
    chat_id: str,
    kind: str,
    reason: str,
    exc: "Exception | None" = None,
) -> None:
    """Send a standalone ⚠️ Feishu card after a crew/dev task crashes.

    Used by the outer ``_run_*_crew_inner`` ``except Exception`` blocks
    when an unexpected error escapes ``_run_crew``. Body always points
    the user at ``/status`` because the most common cause (a backend
    health flip) is diagnosable there.
    """
    try:
        label = "/dev" if kind == "dev" else "/crew"
        title = f"⚠️ {label} 失败 — {reason[:40]}"
        body_lines = [f"**任务终止**：{reason}"]
        if exc is not None:
            body_lines.append(f"\n**异常**：`{_safe_error_repr(exc)}`")
        body_lines.append(
            "\n💡 发送 `/status` 查看 Crew Backend 调度状态及各 backend 健康度。"
        )
        try:
            send_card(chat_id, title, "\n".join(body_lines), color="red")
        except Exception as e:
            _debug_log(f"[Crew] emit_terminal_failure: send_card failed: {e}")
        _debug_log(f"[Crew] terminal failure ({kind}/{chat_id[:8]}): {reason}")
    except Exception as e:
        try:
            _debug_log(f"[Crew] emit_terminal_failure outer error: {e}")
        except Exception:
            pass


def _fmt_wait_age(seconds: float) -> str:
    """Render wait duration as 「X 分钟」/「X 小时 Y 分钟」for age hints."""
    if seconds < 60:
        return "不到 1 分钟"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} 分钟"
    hours = minutes // 60
    rem_min = minutes % 60
    if rem_min == 0:
        return f"{hours} 小时"
    return f"{hours} 小时 {rem_min} 分钟"


def emit_breakpoint_timeout(state: CrewState) -> None:
    """Update the existing crew card to indicate a breakpoint auto-cancel.

    Called from ``_wait_for_breakpoint`` when the user did not click
    继续/取消 within ``CREW_BREAKPOINT_TIMEOUT_SEC``. The runner sets
    ``state.cancel_ev`` separately; this function is just the user-
    facing banner.

    P2-3a (W4/W6): writes ``phase="timeout"`` (new ``CrewPhase.TIMEOUT``
    value) instead of the historical ``"cancelled"`` so user-cancel vs
    auto-timeout stay distinguishable in checkpoints / metrics.
    P3-b (W18): the banner body now states **how long we actually
    waited** before timing out so users have a sense of the deadline.
    """
    try:
        # Resolve the configured deadline best-effort — fall back to the
        # documented default (1800s) so a stripped config can't crash here.
        try:
            import larkhelm.config as _cfg
            timeout_s = float(getattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 1800) or 1800)
        except Exception:
            timeout_s = 1800.0
        age_text = _fmt_wait_age(timeout_s)
        try:
            with state.lock:
                state.phase = "timeout"
                # Stamp the breakpoint agent's error so the agent details
                # pane shows why the task ended.
                bp_id = state.breakpoint_agent_id
                if bp_id and bp_id in state.agents:
                    ag = state.agents[bp_id]
                    if not ag.error:
                        ag.error = f"⏳ 等待 {age_text}人工确认无响应，已自动取消"
        except Exception as e:
            _debug_log(f"[Crew] emit_breakpoint_timeout: state mutation failed: {e}")
        try:
            _crew_update_card(state)
        except Exception as e:
            _debug_log(f"[Crew] emit_breakpoint_timeout: card update failed: {e}")
        # Also send a standalone notification so users with the chat
        # collapsed get a fresh card.
        try:
            send_card(
                state.chat_id,
                "⏳ 等待人工确认超时",
                f"**{state.plan.title}** 已自动取消（等待 {age_text}无响应）。\n\n"
                "需要继续时请重新发送 `/dev` 或 `/crew` 命令；"
                "已完成的阶段会从断点恢复。",
                color="orange",
            )
        except Exception as e:
            _debug_log(f"[Crew] emit_breakpoint_timeout: send_card failed: {e}")
    except Exception as e:
        try:
            _debug_log(f"[Crew] emit_breakpoint_timeout outer error: {e}")
        except Exception:
            pass
