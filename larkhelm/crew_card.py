"""
larkhelm · Crew Feishu card builder

Contains:
  - _build_card()         Build Crew/Dev summary card
  - _crew_update_card()   Push card update (patch)
  - _start_heartbeat()    Start heartbeat thread for periodic progress push
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from larkhelm.card_builder import _fmt_elapsed, _split_md, _make_card, _make_card_dict
from larkhelm.chat_state import _get_cwd
from larkhelm.crew_types import (
    AgentStatus, CrewState, CREW_RESULT_PREVIEW, CREW_CARD_INTERVAL,
)
from larkhelm.lark_client import _patch_card_raw, _send_card_raw
from larkhelm.log import _debug_log


def _backend_label(spec, agent_state) -> str:
    """Pick the most informative backend label for the card row.

    Phase-C migration left ``spec.model=""`` for all dev-pipeline specs
    (the resolver picks at runtime via ``spec.task_profile``), so the
    pre-existing ``[{spec.model}]`` rendered as ``[]``. Order of
    preference now:

      1. ``agent_state.actual_backend_id`` — the resolved backend id
         once the agent has started (most accurate).
      2. ``spec.model`` — legacy /crew specs that still hard-code a
         model string.
      3. ``spec.task_profile`` — pre-run hint when neither above is set.

    Returns ``""`` when nothing is known (very early PENDING with no
    profile); callers omit the bracket entirely in that case.
    """
    if agent_state and getattr(agent_state, "actual_backend_id", ""):
        return str(agent_state.actual_backend_id)
    if getattr(spec, "model", ""):
        return str(spec.model)
    if getattr(spec, "task_profile", ""):
        return str(spec.task_profile)
    return ""


_STATUS_ICON = {
    AgentStatus.PENDING:   "⏸",
    AgentStatus.RUNNING:   "🔄",
    AgentStatus.DONE:      "✅",
    AgentStatus.FAILED:    "❌",
    AgentStatus.CANCELLED: "🛑",
    AgentStatus.SKIPPED:   "⏭",
}


def _build_card(state: CrewState) -> str:
    plan    = state.plan
    agents  = state.agents
    phase   = state.phase
    elapsed = _fmt_elapsed(time.time() - state.start_time)

    n_total    = len(plan.agents)
    n_done     = sum(1 for a in agents.values() if a.status == AgentStatus.DONE)
    n_failed   = sum(1 for a in agents.values() if a.status == AgentStatus.FAILED)
    n_running  = sum(1 for a in agents.values() if a.status == AgentStatus.RUNNING)

    # Title & color
    _label = "Dev" if state.kind == "dev" else "Crew"
    if phase == "planning":
        title, color = f"🧠 {_label} · 规划中", "grey"
    elif phase == "planned":
        title, color = f"📋 {_label} · 即将执行  {n_total} 个 Agent", "blue"
    elif phase == "running":
        title = f"⚙️ {_label} · 执行中  {n_done}/{n_total} 完成"
        if n_running:
            title += f"  {n_running} 运行中"
        running_tok = sum(
            (a.tokens.get("input_tokens", 0) + a.tokens.get("output_tokens", 0))
            for a in agents.values() if a.tokens
        )
        running_tok_str = f"  {running_tok // 1000}k tok" if running_tok > 0 else ""
        title += f"  ({elapsed}){running_tok_str}"
        color = "blue"
    elif phase == "done":
        fail_note = f"  {n_failed} 个失败" if n_failed else ""
        total_tok = sum(
            (a.tokens.get("input_tokens", 0) + a.tokens.get("output_tokens", 0))
            for a in agents.values() if a.tokens
        )
        total_cost = sum(a.tokens.get("cost_usd", 0.0) for a in agents.values() if a.tokens)
        tok_str = f" · {total_tok // 1000}k tok" if total_tok > 0 else ""
        cost_str = f" ${total_cost:.2f}" if total_cost > 0.01 else ""
        title = f"✅ {_label} 完成  {n_total} Agent{fail_note} · {elapsed}{tok_str}{cost_str}"
        color = "green"
    elif phase == "synthesizing":
        title = f"🔗 {_label} · 综合中  {n_done}/{n_total} 完成  ({elapsed})"
        color = "blue"
    elif phase == "breakpoint":
        bp_ag = state.agents.get(state.breakpoint_agent_id)
        bp_role = bp_ag.spec.role if bp_ag else "Agent"
        title = f"⏸ 等待确认 · {bp_role}已完成  {n_done}/{n_total}  ({elapsed})"
        color = "yellow"
    elif phase == "cancelled":
        title, color = f"🛑 {_label} 已取消  ({elapsed})", "orange"
    elif phase == "timeout":
        # P2-3a (W4/W6): breakpoint auto-cancel — orange like ``cancelled``
        # so visual contract matches, but title is distinct so users see
        # the "auto" semantics rather than a self-initiated cancel.
        title, color = f"⏳ {_label} 等待超时  ({elapsed})", "orange"
    else:
        title, color = f"❌ {_label} 失败  ({elapsed})", "red"

    elements: list[dict] = []

    # Task title
    elements.append({
        "tag": "markdown",
        "content": f"**{plan.title}**",
    })

    # Agent status list (shown after planning completes)
    if phase != "planning":
        agent_lines = []
        for spec in plan.agents:
            a     = agents.get(spec.id)
            icon  = _STATUS_ICON.get(a.status if a else AgentStatus.PENDING, "?")
            dep   = f" ← {', '.join(spec.depends_on)}" if spec.depends_on else ""
            t_str = ""
            if a and a.start_time and a.end_time:
                t_str = f" ({_fmt_elapsed(a.end_time - a.start_time)})"
            elif a and a.start_time and a.status == AgentStatus.RUNNING:
                t_str = f" ({_fmt_elapsed(time.time() - a.start_time)}…)"
            _backend = _backend_label(spec, a)
            _backend_str = f" [{_backend}]" if _backend else ""
            agent_lines.append(
                f"{icon} **{spec.id}** {spec.role}{_backend_str}{dep}{t_str}"
            )

        elements.append({
            "tag": "collapsible_panel",
            "header": {"title": {"tag": "plain_text", "content": "**📋 任务计划**"}},
            "expanded": phase in ("planned", "running"),
            "elements": [{"tag": "markdown", "content": "\n".join(agent_lines)}],
        })

    # Architect's planned file list (shown after Architect produces output)
    if phase not in ("planning", "planned"):
        _fc_path = Path(_get_cwd(state.chat_id)) / ".crew_workspace" / "file_changes.json"
        if _fc_path.exists():
            try:
                _fc = json.loads(_fc_path.read_text(encoding="utf-8"))
                _action_icon = {"create": "➕", "modify": "✏️", "delete": "🗑️"}
                _file_lines = [
                    f"{_action_icon.get(f.get('action', ''), '•')} `{f.get('path', '?')}` — {f.get('desc', '')}"
                    for f in _fc.get("files", [])[:30]
                ]
                if _file_lines:
                    elements.append({
                        "tag": "collapsible_panel",
                        "header": {"title": {"tag": "plain_text", "content": "**📁 计划改动文件**"}},
                        "expanded": False,
                        "elements": [{"tag": "markdown", "content": "\n".join(_file_lines)}],
                    })
            except Exception as e:
                _debug_log(f"[crew_card] file list build failed: {e}")

    # Agent details (shown during running/synthesizing/breakpoint/done; flat markdown to avoid nested collapsibles)
    if phase in ("running", "synthesizing", "breakpoint", "done", "cancelled", "failed", "timeout"):
        agent_blocks: list[str] = []
        for spec in plan.agents:
            a = agents.get(spec.id)
            if not a or a.status == AgentStatus.PENDING:
                continue
            # Skip trigger_only placeholder DONE (result is empty)
            if spec.trigger_only and a.status == AgentStatus.DONE and not a.result:
                continue
            icon = _STATUS_ICON.get(a.status, "?")
            t_str = ""
            if a.start_time and a.end_time:
                t_str = f" · {_fmt_elapsed(a.end_time - a.start_time)}"
            elif a.start_time and a.status == AgentStatus.RUNNING:
                t_str = f" · {_fmt_elapsed(time.time() - a.start_time)}…"

            tok_info = ""
            if a.tokens:
                tok = a.tokens.get("input_tokens", 0) + a.tokens.get("output_tokens", 0)
                if tok > 0:
                    tok_info = f" · {tok // 1000}k tok"

            round_tag = f" · _{a.round_label}_" if a.round_label else ""
            # Build each agent's block with compact line joins (no intra-block blank lines)
            block: list[str] = [f"**{icon} {spec.role}**{t_str}{tok_info}{round_tag}"]

            if a.status == AgentStatus.FAILED:
                block.append(f"❌ 失败：{a.error[:200]}")
            elif a.status == AgentStatus.CANCELLED:
                block.append("🛑 已取消")
            elif a.status == AgentStatus.RUNNING:
                block.append((a.result[:200] if a.result else "运行中...") + " ▌")
            else:
                preview = a.result[:CREW_RESULT_PREVIEW]
                suffix  = "\n…（更多内容见结果文件）" if len(a.result) > CREW_RESULT_PREVIEW else ""
                block.append(preview + suffix)
                if a.feishu_doc_url:
                    block.append(f"📄 [飞书文档]({a.feishu_doc_url})")

            agent_blocks.append("\n".join(block))

        if agent_blocks:
            elements.append({
                "tag": "collapsible_panel",
                "header": {"title": {"tag": "plain_text", "content": "**🤖 Agent 详情**"}},
                "expanded": phase in ("running", "synthesizing", "breakpoint"),
                "elements": [{"tag": "markdown", "content": "\n\n---\n\n".join(agent_blocks)}],
            })

    # Final deliverable
    if phase == "done" and state.final_output:
        elements.append({"tag": "hr"})
        chunk = _split_md(state.final_output)[0]
        elements.append({
            "tag": "markdown",
            "content": chunk,
        })

    # ── Active phase: JSON 2.0 with cancel/pause buttons ─────────
    if phase in ("planning", "running", "planned", "synthesizing", "breakpoint"):
        if phase == "planning":
            body_md = "Manager 正在分析需求，生成任务计划…"
            return _make_card(title, body_md, color=color,
                              buttons=[("🛑 取消", f"cancel:{state.chat_id}")])

        body_parts: list[str] = [f"**{plan.title}**"]

        # Agent status list — one line per agent, no blank lines between them
        status_lines: list[str] = []
        for spec in plan.agents:
            a     = agents.get(spec.id)
            icon  = _STATUS_ICON.get(a.status if a else AgentStatus.PENDING, "?")
            dep   = f" ← {', '.join(spec.depends_on)}" if spec.depends_on else ""
            t_str = ""
            if a and a.start_time and a.end_time:
                t_str = f" ({_fmt_elapsed(a.end_time - a.start_time)})"
            elif a and a.start_time and a.status == AgentStatus.RUNNING:
                t_str = f" ({_fmt_elapsed(time.time() - a.start_time)}…)"
            _backend = _backend_label(spec, a)
            _backend_str = f" [{_backend}]" if _backend else ""
            status_lines.append(f"{icon} **{spec.id}** {spec.role}{_backend_str}{dep}{t_str}")
        if status_lines:
            body_parts.append("\n".join(status_lines))

        # Progress preview for running agents
        running_previews: list[str] = []
        for spec in plan.agents:
            a = agents.get(spec.id)
            if not a or a.status != AgentStatus.RUNNING:
                continue
            t_str = f" · {_fmt_elapsed(time.time() - a.start_time)}…" if a.start_time else ""
            preview = (a.result[:300] if a.result else "运行中...") + " ▌"
            running_previews.append(f"**🔄 {spec.role}**{t_str}\n\n{preview}")
        if running_previews:
            body_parts.append("---\n\n" + "\n\n---\n\n".join(running_previews))

        # Failed-agent details — surface ``a.error`` during the running
        # phase too. Without this, the user sees an ❌ icon on the agent
        # line but no reason; the rich "Agent 详情" block that DOES
        # include error text only fires for terminal phases (line 257+).
        # PRD §G2 promised "Agent 失败时收到明确的 ⚠️ 卡片（含失败
        # Agent ID、阶段、原因摘要）" — without this section, the
        # promise is only kept after the whole crew finishes, not at
        # the moment of failure. Caught by
        # ``test_phase_c_failure_card_roundtrip``.
        failure_blocks: list[str] = []
        for spec in plan.agents:
            a = agents.get(spec.id)
            if not a or a.status != AgentStatus.FAILED or not a.error:
                continue
            t_str = ""
            if a.start_time and a.end_time:
                t_str = f" · {_fmt_elapsed(a.end_time - a.start_time)}"
            failure_blocks.append(
                f"**❌ {spec.role} 失败**{t_str}\n\n{a.error[:400]}"
            )
        if failure_blocks:
            body_parts.append("---\n\n" + "\n\n---\n\n".join(failure_blocks))

        # Breakpoint: PRD preview + confirm/cancel buttons on the main card
        if phase == "breakpoint":
            from pathlib import Path as _Path
            prd_path = _Path(_get_cwd(state.chat_id)) / ".crew_workspace" / "prd.md"
            prd_preview = ""
            try:
                _text = prd_path.read_text(encoding="utf-8")
                prd_preview = _text[:600] + ("…" if len(_text) > 600 else "")
            except Exception:
                ag = state.agents.get(state.breakpoint_agent_id)
                if ag:
                    prd_preview = ag.result[:400]
            if prd_preview:
                body_parts.append(prd_preview)
            body_md = "\n\n".join(body_parts)
            return _make_card(title, body_md, color=color, normalize=False,
                              buttons=[("✅ 继续执行", f"crew_bp:confirm:{state.crew_id}"),
                                       ("❌ 取消",    f"crew_bp:cancel:{state.crew_id}")])

        body_md = "\n\n".join(body_parts)
        return _make_card(title, body_md, color=color, normalize=False,
                          buttons=[("🛑 取消", f"cancel:{state.chat_id}"),
                                   ("⏸ 暂停", f"crew_pause:{state.crew_id}")])

    # ── Terminal phase: rich text with pre-built element tree ────
    return json.dumps(_make_card_dict(title, raw_elements=elements, color=color),
                      ensure_ascii=False)


def _crew_update_card(state: CrewState):
    if not state.card_mid:
        return
    card = _build_card(state)
    ok = _patch_card_raw(state.card_mid, card)
    if not ok:
        # Schema mismatch (e.g. ErrCode 200830: schemaV2 cannot patch schemaV1).
        # Send a fresh card and redirect future patches to it.
        new_mid = _send_card_raw(state.chat_id, card)
        if new_mid:
            with state.lock:
                state.card_mid = new_mid
            _debug_log(f"[Crew] replaced stale card → {new_mid}")


def _start_heartbeat(state: CrewState, stop_ev: threading.Event) -> threading.Thread:
    """Push a progress card every CREW_CARD_INTERVAL seconds. Returns the thread."""
    def _loop():
        while not stop_ev.is_set():
            try:
                _crew_update_card(state)
            except Exception as e:
                _debug_log(f"[CrewHeartbeat] error: {e}")
            stop_ev.wait(timeout=CREW_CARD_INTERVAL)
    t = threading.Thread(target=_loop, daemon=True, name=f"crew-hb-{state.crew_id[:6]}")
    t.start()
    return t
