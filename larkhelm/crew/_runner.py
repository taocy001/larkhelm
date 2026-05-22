"""
larkhelm · Crew Agent executor
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from larkhelm.log import _debug_log, log_entry, warn
from larkhelm.ai_runner import QueryCancelledError
from larkhelm.crew_types import (
    HardFailError, AgentSpec, AgentState, AgentStatus, CrewState,
    NoBackendAvailableError,
    CREW_RESULT_PREVIEW,
)
from larkhelm.crew_card import _crew_update_card, _start_heartbeat
from larkhelm.crew._backend_resolver import resolve_backend
from larkhelm.crew._failure_card import (
    emit_agent_failure, emit_breakpoint_timeout,
)


def _workspace_dir(chat_id: str, crew_id: str) -> Path:
    import larkhelm.config as _cfg
    d = _cfg.SESSION_DIR / chat_id / f"crew_{crew_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _detect_fail_marker(spec: AgentSpec, result: str) -> bool:
    """Check whether the last non-empty line of the output contains the fail_marker."""
    if not spec.fail_marker or not result:
        return False
    last_line = next((l.strip() for l in reversed(result.split("\n")) if l.strip()), "")
    return spec.fail_marker in last_line


# Tool-call protocol tokens that some backends emit as plain text when the
# larkhelm runner does not translate them into ``tool_use`` events. Observed
# 2026-05-22 from a DeepSeek-flavoured planner backend writing
# ``<｜｜DSML｜｜tool_calls>...`` verbatim into ``prd.md`` / ``design.md``,
# which silently corrupted the entire downstream pipeline (architect, fixer,
# QA all burned full agents' worth of tokens on an unusable contract).
# Pinned narrow on purpose — broad regexes risk false-positives on docs
# that legitimately quote these tokens.
_OUTPUT_SENTINELS: tuple[str, ...] = (
    "<｜｜DSML｜｜tool_call",   # DeepSeek DSML (zero-width-ish full-width pipes)
    "<｜tool▁call",             # DeepSeek alt tokenizer form
    "<｜tool_call",             # DeepSeek alt tokenizer form
    "<|tool_call",              # generic OpenAI-style sentinel leak
    "<tool_call>",
    "<tool_calls>",
)


def _strip_fenced_code_blocks(text: str) -> str:
    """Remove ```...``` fenced blocks from ``text`` so downstream sentinel
    scans only see narrative prose, not quoted evidence.

    Motivation: documentation agents (QA, implementer-when-it-bails-out,
    reviewer) legitimately quote upstream corrupt output inside a code
    fence as evidence — see ``changes.md`` lines 22-35 of the 2026-05-22
    failure where the implementer embedded the full ``<｜｜DSML｜｜...>``
    block to explain why it couldn't proceed. Without this strip, those
    fenced quotes would trip the sentinel detector and the validator
    would mark legitimate bail-out reports as FAILED.

    Tradeoff: a malicious / sloppy backend could wrap its own leaked
    tool-call output in a pseudo-fence to bypass detection. Considered
    out of scope — the realistic adversary is "fenced quoted evidence",
    not "adversarially-crafted fence wrapper".
    """
    if "```" not in text:
        return text
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue   # drop the fence marker itself too
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def _validate_output_artifact(state: CrewState, agent_id: str, result: str) -> str:
    """Return a short problem description if the agent's output artifact is
    structurally corrupt, or "" if it looks OK.

    Two failure modes covered:
      1. Tool-call protocol tokens leaked into the artifact (see
         ``_OUTPUT_SENTINELS``). Downstream stages would parse these as
         markdown content and silently produce garbage. Sentinels inside
         fenced code blocks are ignored — see ``_strip_fenced_code_blocks``.
      2. ``.json`` ``output_file`` that is not parseable JSON. Downstream
         stages crash on ``json.load`` mid-wave. JSON validation does NOT
         strip fences (the whole file must parse).

    Scans the on-disk artifact when present (post Write tool) and falls back
    to the in-memory ``result`` otherwise. Only inspects the first 8 KiB —
    sentinels always appear at the head of the artifact, and capping the
    scan keeps the validator O(1) per agent regardless of artifact size.
    Never raises — defects in this helper must not crash the crew wrapper.
    """
    try:
        spec = state.agents[agent_id].spec
    except Exception:
        return ""

    candidates: list[tuple[str, str]] = []  # (label, content)
    if spec.output_file:
        try:
            from larkhelm.chat_state import _get_cwd
            cwd = _get_cwd(state.chat_id)
            out_path = Path(cwd) / ".crew_workspace" / spec.output_file
            if out_path.exists():
                content = out_path.read_text(encoding="utf-8", errors="replace")
                if content:
                    candidates.append((spec.output_file, content))
        except Exception as e:
            _debug_log(f"[Crew] {agent_id} validate: read {spec.output_file} failed: {e}")
    if not candidates and result:
        candidates.append(("result", result))

    for label, content in candidates:
        # Sentinel scan: strip fences first so legitimate quoted evidence
        # in docs (QA / implementer bail-out / review) does not trip.
        prose = _strip_fenced_code_blocks(content)
        head = prose[:8192]
        for s in _OUTPUT_SENTINELS:
            if s in head:
                return f"{label} contains tool-call sentinel {s!r}"
        # JSON validation runs on the full content — the file is supposed
        # to be parseable end-to-end, fences would already invalidate it.
        if spec.output_file and spec.output_file.endswith(".json"):
            try:
                json.loads(content)
            except Exception as e:
                return f"{label} is not valid JSON: {type(e).__name__}: {str(e)[:120]}"
    return ""


def _quarantine_invalid_output(state: CrewState, agent_id: str) -> None:
    """Rename a validation-failed ``output_file`` to ``<name>.invalid`` so
    the scheduler's partial-delivery rule (``_get_failed_dep``) does not
    interpret the corrupt file as "delivered" and let downstream agents run.

    Background: ``_get_failed_dep`` treats a FAILED upstream as non-blocking
    when its ``output_file`` exists and is non-empty (designed for the
    OOM-after-Write case where the artifact was atomically committed before
    the agent process died). That rule has no way to know that the file
    contents are structurally invalid — when our validator catches a
    tool-call leak, we have to invalidate the file ourselves, otherwise the
    crew cascades the same corruption through architect → implementer →
    QA → reviewer (observed 2026-05-22 dev failure).

    Atomic ``os.replace`` keeps the original bytes available for user
    inspection at the new ``.invalid`` path instead of deleting them.
    Failures here are logged and swallowed — quarantine is best-effort;
    the FAILED status itself still gets emitted by ``emit_agent_failure``.
    """
    try:
        spec = state.agents[agent_id].spec
    except Exception:
        return
    if not spec.output_file:
        return
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(state.chat_id)
        out_path = Path(cwd) / ".crew_workspace" / spec.output_file
        if not out_path.exists():
            return
        invalid_path = out_path.with_suffix(out_path.suffix + ".invalid")
        import os as _os
        _os.replace(out_path, invalid_path)
        warn(
            f"[Crew] {agent_id} quarantined invalid output: "
            f"{spec.output_file} → {invalid_path.name} "
            f"(inspect this file to see what the backend produced)"
        )
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} quarantine failed: {e}")


def _crew_owner_open_id(state: CrewState) -> str:
    """Return the open_id to transfer ownership to: chat sender > config default."""
    import larkhelm.config as _cfg
    from larkhelm.chat_state import _get_chat_state
    owner = _get_chat_state(state.chat_id).get("sender_open_id", "")
    return owner or _cfg.DEFAULT_OWNER_OPEN_ID


def _ensure_crew_folder(state: CrewState) -> None:
    """Create a per-project Feishu folder at crew start. Sets state.feishu_folder_token/url.
    Falls back silently if Feishu is unavailable or backend is 'local'.
    """
    import larkhelm.config as _cfg
    if _cfg.DOC_WRITE_BACKEND == "local":
        return

    project_name = (state.plan.title or "crew")[:40]
    owner_open_id = _crew_owner_open_id(state)

    from larkhelm.lark_client import FeishuDocClient, DocError
    doc_client = FeishuDocClient()

    try:
        if _cfg.DEFAULT_WIKI_SPACE_ID:
            ref = doc_client.create_wiki_node(
                _cfg.DEFAULT_WIKI_SPACE_ID, project_name, _cfg.DEFAULT_WIKI_PARENT_TOKEN,
                owner_open_id=owner_open_id,
            )
            node_token = ref.raw_url.split("/")[-1]
            state.feishu_folder_token = node_token
            state.feishu_folder_url   = f"https://feishu.cn/wiki/{node_token}"
        else:
            parent = _cfg.DEFAULT_DRIVE_FOLDER or ""
            folder_token = doc_client.create_folder(project_name, parent, owner_open_id=owner_open_id)
            state.feishu_folder_token = folder_token
            state.feishu_folder_url   = f"https://feishu.cn/drive/folder/{folder_token}"
        _debug_log(f"[Crew] 项目文件夹已创建: {state.feishu_folder_url}")
    except DocError as e:
        _debug_log(f"[Crew] 创建项目文件夹失败，将写入本地: {e}")


def _persist_result_to_output_file_if_missing(
    state: CrewState, agent_id: str, result: str,
) -> None:
    """Safety net: persist the in-memory ``result`` to the agent's declared
    ``output_file`` when the agent forgot (or failed) to call the Write tool.

    The agent prompt injection (in :func:`_run_agent`) tells the agent to
    Write its full output to ``.crew_workspace/{spec.output_file}``, but in
    practice ``state.agents[agent_id].result`` often holds only the LAST
    streamed text chunk (a short closing acknowledgement after the Write
    tool call). When the agent honours the contract, this is harmless —
    the on-disk file is the source of truth, and ``result`` is just a
    summary. When the agent skips the Write call (P3 PM did exactly this —
    emitted a ~39 K token PRD that never landed on disk), the architect
    downstream finds an empty / nonexistent file and falls back to
    reverse-engineering from source.

    Heuristic: write the fallback only when
      • ``spec.output_file`` is set
      • the file is missing OR materially smaller than ``result``
      • ``result`` is substantial (≥ 200 chars; below that it's almost
        certainly just a closing marker like "PRD written.")

    Failures are logged and swallowed — the safety net must never raise.
    """
    spec = state.agents[agent_id].spec
    if not spec.output_file:
        return
    if not result or len(result) < 200:
        # Result is just a closing marker; the agent presumably called Write
        # itself with the real payload. Don't risk overwriting a good file.
        return
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(state.chat_id)
        out_path = Path(cwd) / ".crew_workspace" / spec.output_file
        existing_size = 0
        if out_path.exists():
            try:
                existing_size = out_path.stat().st_size
            except OSError:
                existing_size = 0
        # If the on-disk file already covers ≥ 80% of result length, trust
        # the agent's own write — closing summary mismatch is fine.
        if existing_size >= int(len(result) * 0.8):
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: ``tmp`` + ``replace`` so a concurrent reader never
        # sees a torn file. Same pattern as ``memory_io.atomic_write``.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(result, encoding="utf-8")
        import os as _os
        _os.replace(tmp, out_path)
        # Surface at WARN so this never silently becomes "background noise":
        # every safety-net hit means the underlying agent failed to honour
        # the Write-tool contract, which is a real regression to chase
        # (Claude CLI perm denial / agent-prompt drift / token-limit cutoff
        # etc.). Independent review flagged that the original ``_debug_log``
        # alone gave operators zero signal.
        warn(
            f"[Crew] {agent_id} output_file safety-net wrote {len(result)} chars "
            f"to {spec.output_file} — agent skipped Write tool "
            f"(existing on disk: {existing_size}b). Investigate the agent's "
            f"tool-call trace to find why Write was bypassed."
        )
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} output_file safety-net failed: {e}")


def _sync_output_file(state: CrewState, agent_id: str) -> str:
    """Sync an agent's output_file to Feishu, inside the per-project folder.
    Returns the Feishu document URL on success, empty string on failure/local-only.
    Always falls back to local if Feishu is unavailable.
    """
    import larkhelm.config as _cfg
    if _cfg.DOC_WRITE_BACKEND == "local":
        return ""

    spec = state.agents[agent_id].spec
    if not spec.output_file:
        return ""

    from larkhelm.chat_state import _get_cwd
    cwd      = _get_cwd(state.chat_id)
    out_path = Path(cwd) / ".crew_workspace" / spec.output_file
    if not out_path.exists():
        return ""

    try:
        content = out_path.read_text(encoding="utf-8")
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} 读取 output_file 失败: {e}")
        return ""

    _STEM_NAMES = {
        "prd": "产品需求文档", "design": "技术设计方案",
        "changes": "代码变更记录", "qa_report": "测试报告", "review": "代码审查报告",
    }
    display_stem  = _STEM_NAMES.get(out_path.stem, out_path.stem)
    project_prefix = (state.plan.title or "crew")[:30].strip()
    title         = f"{project_prefix} · {display_stem}"
    folder_token  = state.feishu_folder_token
    owner_open_id = _crew_owner_open_id(state)

    from larkhelm.lark_client import FeishuDocClient, DocError, parse_doc_url
    doc_client = FeishuDocClient()

    # Reuse existing Feishu doc if this output_file was already synced (e.g. fixer reusing changes.md)
    existing_url = state.output_file_urls.get(spec.output_file, "")

    try:
        if existing_url:
            # Append to the already-created doc instead of creating a duplicate
            ref = parse_doc_url(existing_url)
            doc_client.append(ref, content)
            _debug_log(f"[Crew] {agent_id} 追加到已有飞书文档: {existing_url}")
            return existing_url
        elif _cfg.DEFAULT_WIKI_SPACE_ID:
            parent_token = folder_token or _cfg.DEFAULT_WIKI_PARENT_TOKEN
            ref        = doc_client.create_wiki_node(_cfg.DEFAULT_WIKI_SPACE_ID, title, parent_token,
                                                     owner_open_id=owner_open_id)
            node_token = ref.raw_url.split("/")[-1]
            doc_url    = f"https://feishu.cn/wiki/{node_token}"
            doc_client.append(ref, content)
        else:
            target = folder_token or _cfg.DEFAULT_DRIVE_FOLDER or ""
            ref     = doc_client.create_doc(title, target, owner_open_id=owner_open_id)
            doc_url = f"https://feishu.cn/docx/{ref.token}"
            doc_client.append(ref, content)

        with state.lock:
            state.output_file_urls[spec.output_file] = doc_url
        _debug_log(f"[Crew] {agent_id} 已同步到飞书: {doc_url}")
        return doc_url

    except DocError as e:
        _debug_log(f"[Crew] {agent_id} 飞书写入失败，保留本地文件: {e}")
        return ""


def _run_agent(state: CrewState, agent_id: str) -> str:
    """Execute a single agent and return the text result. Uses an isolated session namespace."""
    import larkhelm.config as _cfg
    from larkhelm.chat_state import _get_cwd
    from larkhelm.concurrency import _get_cancel_event
    from larkhelm.card_builder import _fmt_elapsed
    from larkhelm.perm import grant_yolo, revoke_yolo
    from larkhelm.ai_runner import query_gemini, query_kimi
    from larkhelm.crew._scheduler import _resolve_prompt
    from larkhelm.crew._state import _git_head
    from larkhelm.crew._hermes_orchestrator import _run_hermes_orchestrator

    spec      = state.agents[agent_id].spec
    cancel_ev = state.cancel_ev
    cwd       = _get_cwd(state.chat_id)
    workspace = _workspace_dir(state.chat_id, state.crew_id)

    # Session isolation: crew_ns is the namespace, isolated from the main session.
    # Crew agents do not use persistent sessions (sid=None):
    #   1. Carrying over the previous failure history on retry significantly increases token usage.
    #   2. Resumed runs inject context via the is_resuming prompt and do not rely on sessions.
    crew_ns = f"{state.chat_id}__crew_{state.crew_id}_{agent_id}"
    sid     = None  # No session reuse; each run starts a fresh conversation

    # Build the full prompt (system + placeholder resolution).
    # On retry, switch to the retry-specific role prompt (e.g. fixer mode).
    ag_state = state.agents[agent_id]
    active_system = spec.retry_system if (ag_state.retry_count > 0 and spec.retry_system) else spec.system
    active_prompt = spec.retry_prompt if (ag_state.retry_count > 0 and spec.retry_prompt) else spec.prompt
    
    # For Hermes orchestrator agents, the prompt is JSON params
    if spec.model.startswith("hermes_"):
        full_prompt = active_prompt
    else:
        resolved = _resolve_prompt(active_prompt, state)
        if active_system:
            full_prompt = f"{active_system}\n\n---\n\n{resolved}"
        else:
            full_prompt = resolved

    # Inject output_file requirement so the agent knows it must write the file
    if spec.output_file and not spec.model.startswith("hermes_"):
        full_prompt += (
            f"\n\n---\n\n**Output requirement**: When you have finished, write your complete "
            f"output to `.crew_workspace/{spec.output_file}` using the Write tool. "
            f"Do not skip this step."
        )

    # Inject Feishu doc URLs from completed upstream agents so this agent can reference them
    _upstream_doc_refs = []
    for _dep_id in spec.depends_on:
        _dep = state.agents.get(_dep_id)
        if _dep and _dep.feishu_doc_url:
            _upstream_doc_refs.append(f"- {_dep.spec.role}：{_dep.feishu_doc_url}")
    if _upstream_doc_refs and not spec.model.startswith("hermes_"):
        full_prompt += (
            "\n\n---\n\n**上游 Agent 飞书文档（可直接引用，与本地文件内容相同）：**\n"
            + "\n".join(_upstream_doc_refs)
        )

    # Inject downstream feedback (written by _execute on retry, truncated to 3000 chars to prevent token explosion)
    if ag_state.feedback:
        round_info = f" (retry {ag_state.retry_count})" if ag_state.retry_count else ""
        feedback_trimmed = ag_state.feedback[:3000]
        if len(ag_state.feedback) > 3000:
            feedback_trimmed += f"\n\n…（已截断，共 {len(ag_state.feedback)} 字符）"
        full_prompt = (
            f"⚠️ **Feedback from previous iteration{round_info} — please fix the following issues:**\n\n"
            f"{feedback_trimmed}\n\n"
            f"---\n\n"
            + full_prompt
        )

    # Resume context injection: inform the agent this task is being resumed from a checkpoint
    if state.is_resuming:
        _ws_path = Path(cwd) / ".crew_workspace"
        # Only list planning/document files (.md/.json); filter out intermediate artifacts like result.txt (noise)
        _plan_files = sorted(
            f.name for f in _ws_path.iterdir()
            if f.is_file() and f.suffix in (".md", ".json")
        ) if _ws_path.exists() else []
        _git_info = ""
        if state.git_head_before:
            _cur_head = _git_head(cwd)
            if _cur_head and _cur_head != state.git_head_before:
                _git_info = (f"\n- Code has been modified (start commit: {state.git_head_before},"
                             f" current: {_cur_head}). Run `git diff {state.git_head_before}`"
                             f" first to review completed changes and avoid duplicate work")
            elif _cur_head:
                _git_info = f"\n- No code changes detected (HEAD: {_cur_head})"
        _resume_prefix = (
            "⚠️ **Resuming task (previous execution was interrupted, continuing from checkpoint)**\n\n"
            "Resume notes:\n"
            "- `.crew_workspace/` contains the planning files from the last run; **use these as the baseline**, do not re-plan"
            + (_git_info or "")
            + ("\n- Existing planning files: " + ", ".join(_plan_files) if _plan_files else "")
            + "\n- Continue directly from the incomplete parts; skip already completed content\n\n---\n\n"
        )
        full_prompt = _resume_prefix + full_prompt

    # Resolve memory context for this agent (B1/B6 + Phase B S44 unification):
    # API backends receive it as extra_system; CLI backends get a [System] prefix.
    # hermes orchestrators have their own context and don't need memory injection.
    #
    # Phase B unifies crew agents with /chat and /plan: all three now call
    # ``get_memory_context`` (global + project + session). Previously crew
    # agents called ``get_project_memory_context`` which dropped the global
    # layer, hiding user-level preferences (style / language) from sub-agents
    # — a frequent source of "agent ignored my preferences" reports.
    # Per-layer S49–S52 budget logic in ``MemoryContextBuilder`` keeps the
    # token footprint bounded; passing ``full_prompt`` as the gating query
    # lets ``should_include_*`` make keyword-aware decisions.
    _crew_mem_ctx = ""
    if not spec.model.startswith("hermes_"):
        try:
            from larkhelm.memory import get_memory_context_v2
            # Phase D: map AgentSpec.task_profile → agent_type so the
            # retriever (when enabled) chooses a sensible per-agent policy.
            # Unknown profiles fall back to ``crew`` policy (large budget,
            # decision-heavy kind priority).
            _tp_to_agent_type = {
                "planner": "plan",
                "engineer": "dev",
                "qa": "dev",
                "reviewer": "dev",
                "chat": "chat",
            }
            _agent_type = _tp_to_agent_type.get(
                (spec.task_profile or ""), "crew",
            )
            _crew_intent = None
            try:
                from larkhelm.agent_hub.intent_types import IntentResult
                _crew_intent = IntentResult(
                    agent_type=_agent_type,
                    sub_intent=spec.id,
                    complexity="complex",
                    confidence=0.9,
                    layer="override",
                    raw_text=full_prompt,
                )
            except Exception as _ie:
                _debug_log(f"[Crew] IntentResult synth failed: {_ie}")
            _crew_mem_ctx, _ = get_memory_context_v2(
                state.chat_id, cwd=str(cwd), query=full_prompt,
                intent=_crew_intent,
                sender_open_id=state.sender_open_id,
            )
        except Exception as e:
            _debug_log(f"[Crew] memory load failed: {e}")

    # Timeout control: start countdown only after acquiring the process slot (semaphore),
    # so waiting time is not counted toward the timeout.
    agent_cancel  = threading.Event()
    _slot_ready   = threading.Event()   # fired when the semaphore slot is acquired

    def _timeout_watcher():
        """Forward crew-level cancellation; emit a soft-timeout log line.

        **No hard kill** — that responsibility belongs to
        ``BaseProcessRunner._watch`` which now measures **idle** time. The
        previous version of this watcher set ``hard_deadline = time.time()
        + max(spec.timeout * 2, HARD_TIMEOUT)`` and unconditionally fired
        ``agent_cancel`` at the wall-clock mark — meaning a long but
        actively-streaming agent (multi-hour /dev pipeline producing
        continuous output) got cancelled at the same instant a wedged
        agent did. The idle-clock fix in ``runner_base`` only takes
        effect if this outer wall-clock kill doesn't preempt it first.
        Forwarding cancel_ev + logging soft is the only remaining job.
        """
        # Phase 1: wait for process slot while monitoring crew-level cancellation.
        # Also honours ``agent_cancel`` — set by ``_run_agent``'s own
        # ``finally`` block at the bottom of this function (line ~472),
        # which fires when the agent body exits early (e.g.
        # ``NoBackendAvailableError`` raised before any subprocess starts).
        # Without this check the watcher would self-spin for the full
        # sem-wait window after each failed agent, accumulating one
        # daemon thread per failure until the crew completes. Cheap fix
        # per review OBS-01; regression test:
        # ``test_timeout_watcher_exits_when_agent_cancel_set``.
        while not _slot_ready.is_set():
            if cancel_ev.is_set() or agent_cancel.is_set():
                agent_cancel.set()
                return
            time.sleep(0.3)
        # Phase 2: process has started; begin counting spec.timeout from now
        soft_deadline = time.time() + spec.timeout
        soft_fired = False
        while True:
            if cancel_ev.is_set():
                agent_cancel.set()
                return
            # Stop polling once the inner runner has signalled completion —
            # otherwise this thread spins until process exit (harmless but
            # wasteful) and may delay agent cleanup.
            if agent_cancel.is_set():
                return
            if not soft_fired and time.time() >= soft_deadline:
                soft_fired = True
                _debug_log(
                    f"[Crew] {agent_id} soft timeout ({spec.timeout}s), "
                    "BaseProcessRunner idle-clock continues to gate hard kill"
                )
            time.sleep(0.3)

    threading.Thread(target=_timeout_watcher, daemon=True).start()

    # Progress card update: only cache the latest text; heartbeat (_start_heartbeat) pushes the card
    def _on_text(text: str, status: str = "typing"):
        with state.lock:
            state.agents[agent_id].result = text

    def _on_start():
        """Callback when the process slot is acquired: update status and start the timeout countdown."""
        _slot_ready.set()
        with state.lock:
            state.agents[agent_id].status     = AgentStatus.RUNNING
            state.agents[agent_id].start_time = time.time()
        _crew_update_card(state)

    # Crew agents automatically grant all permissions (user authorized via /crew command)
    grant_yolo(crew_ns)
    try:
        # Resolve which backend should serve this agent BEFORE entering the
        # dispatch branches. The resolver checks task_profile first
        # (Phase C path), then falls back to legacy model-string handling
        # for backward-compat. The resolved BackendSpec drives the branch
        # selection below via its ``provider`` field.
        try:
            resolved = resolve_backend(spec)
        except NoBackendAvailableError:
            # Re-raise so the wrapper can record the failure and surface
            # a "无可用 backend" card; not retried because this is a config
            # / health issue, not a transient subprocess error.
            raise

        # Record the actual backend id picked by the resolver so the
        # crew card can show "[claude] PM" / "[kimi] engineer" instead
        # of the now-empty ``[spec.model]`` placeholder. Phase-C left
        # ``spec.model=""`` and ``spec.task_profile="planner"/...`` so
        # the model is only known after this resolve call.
        with state.lock:
            state.agents[agent_id].actual_backend_id = resolved.id

        # Resolved hermes_* synthesizes a provider="hermes" BackendSpec —
        # rebuild the dispatch decision around ``resolved.provider`` so
        # existing branches stay intact.
        if resolved.provider == "hermes":
            _disp_kind = resolved.id        # e.g. "hermes_race"
        elif resolved.provider == "gemini_cli":
            _disp_kind = "gemini"
        elif resolved.provider == "kimi_cli":
            _disp_kind = "kimi"
        elif resolved.provider == "deepseek_api":
            _disp_kind = "deepseek"
        else:
            _disp_kind = "orchestrator"

        if _disp_kind == "gemini":
            _on_start()   # Gemini has no semaphore callback; trigger directly
            _gemini_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                           if _crew_mem_ctx else full_prompt)
            output = query_gemini(
                chat_id=crew_ns,
                message=_gemini_msg,
                cwd=cwd,
                cancel_ev=agent_cancel,
                on_text=_on_text,
                on_tool=None,
                on_tool_result=None,
                use_session=False,          # No session reuse; prevents carrying history into retries
                record_under=state.chat_id, # Token stats recorded under the real chat_id
            )
        elif _disp_kind == "kimi":
            _on_start()   # Kimi has no semaphore callback; trigger directly
            _kimi_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                         if _crew_mem_ctx else full_prompt)
            output = query_kimi(
                chat_id=crew_ns,
                message=_kimi_msg,
                cwd=cwd,
                cancel_ev=agent_cancel,
                on_text=_on_text,
                use_session=False,
                record_under=state.chat_id,
            )
        elif _disp_kind == "deepseek":
            from larkhelm.ai_runner import query_deepseek
            _on_start()   # DeepSeek has no semaphore callback; trigger directly
            _ds_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                       if _crew_mem_ctx else full_prompt)
            output = query_deepseek(
                chat_id=crew_ns,
                message=_ds_msg,
                cwd=cwd,
                cancel_ev=agent_cancel,
                on_text=_on_text,
                use_session=False,
                record_under=state.chat_id,
            )
        elif _disp_kind.startswith("hermes_"):
            # Hermes multi-agent orchestrator modes: race, split, review.
            # ``provider == "hermes"`` was previously checked here too,
            # but ``_disp_kind`` is set to ``resolved.id`` exactly when
            # ``resolved.provider == "hermes"`` (line 356), and every
            # hermes BackendSpec id begins with ``hermes_``, so the
            # second clause was unreachable. Review §4 cleanup.
            _on_start()
            output = _run_hermes_orchestrator(
                state=state,
                agent_id=agent_id,
                spec=spec,
                cancel_ev=agent_cancel,
                on_text=_on_text,
            )
        else:
            from larkhelm.backend_cli import run_claude as _bc_run_claude, run_gemini as _bc_run_gemini, run_kimi as _bc_run_kimi
            # ``resolved`` is whatever the backend resolver picked — could be
            # the orchestrator default OR a task_profile-ranked candidate.
            # Either way we hand off to one of the backend_cli / backend_api
            # branches below based on its provider, so claude-disabled hosts
            # naturally fall through to kimi/deepseek (PRD AC-06).
            _spec = resolved
            _API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api")
            _cli_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                        if _crew_mem_ctx else full_prompt)
            if _spec.provider in _API_PROVIDERS:
                import larkhelm.backend_api as _bapi
                _fn = {"anthropic_api": _bapi.run_anthropic,
                       "google_api": _bapi.run_google,
                       "openai_compat_api": _bapi.run_openai_compat}[_spec.provider]
                _on_start()
                output, _ = _fn(spec=_spec, chat_id=crew_ns, message=full_prompt,
                                history=[], cancel_ev=agent_cancel, on_text=_on_text,
                                extra_system=_crew_mem_ctx)
            elif _spec.provider == "gemini_cli":
                _on_start()
                output = _bc_run_gemini(spec=_spec, chat_id=crew_ns, message=_cli_msg,
                                        sid=None, cwd=cwd, cancel_ev=agent_cancel,
                                        on_text=_on_text, use_session=False)
            elif _spec.provider == "kimi_cli":
                _on_start()
                output = _bc_run_kimi(spec=_spec, chat_id=crew_ns, message=_cli_msg,
                                      sid=None, cwd=cwd, cancel_ev=agent_cancel,
                                      on_text=_on_text)
            else:
                output = _bc_run_claude(
                    spec=_spec, chat_id=crew_ns, message=_cli_msg,
                    sid=sid, cwd=cwd, cancel_ev=agent_cancel,
                    on_text=_on_text, on_tool=None, on_tool_result=None,
                    allow_retry=False, on_start=_on_start, session_namespace=crew_ns,
                )
    except QueryCancelledError:
        if cancel_ev.is_set():
            raise  # Crew-level cancellation → propagate → wrapper marks CANCELLED
        # Per-agent timeout only → FAILED (not CANCELLED)
        raise RuntimeError(f"Agent timed out (terminated after {_fmt_elapsed(spec.timeout)})")
    finally:
        agent_cancel.set()  # always stop the timeout watcher thread, even on abnormal exit
        revoke_yolo(crew_ns)

    result = output.strip() or "（无输出）"

    # Write result to file so downstream agents can read the full text via the Read tool
    result_file = workspace / f"{agent_id}_result.txt"
    try:
        result_file.write_text(result, encoding="utf-8")
    except Exception as e:
        _debug_log(f"[Crew] result file write failed: {e}")

    # For Hermes orchestrator agents, also write a formatted summary to a markdown file
    if spec.model.startswith("hermes_"):
        _write_hermes_summary(state, agent_id, result, workspace)

    return result


def _write_hermes_summary(state: CrewState, agent_id: str, result: str, workspace: Path) -> None:
    """Write a formatted markdown summary of Hermes orchestrator results for Feishu doc sync."""
    spec = state.agents[agent_id].spec
    mode = spec.model.replace("hermes_", "")
    
    summary_path = workspace / f"{agent_id}_summary.md"
    lines = [
        f"# {spec.role} 结果 ({mode} 模式)",
        "",
        f"**任务:** {state.plan.title}",
        f"**Agent:** {agent_id}",
        f"**模式:** {mode}",
        f"**完成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        result,
        "",
        "---",
        "",
        f"*由 LarkHelm Hermes Orchestrator 自动生成*",
    ]
    try:
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        _debug_log(f"[HermesOrchestrator] summary written to {summary_path}")
    except Exception as e:
        _debug_log(f"[HermesOrchestrator] failed to write summary: {e}")


# OOM detection signature set. The two main sources of OOM in this
# project's history:
#
#   * cgroup OOM-killer killing the node CLI (claude/kimi/gemini)
#     subprocess. ``runner_base._on_kill_signal`` wraps SIGKILL into
#     ``RuntimeError(..."killed by OS (rc=-9)..."``).
#   * V8 internal "FATAL ERROR: Reached heap limit" — the node CLI
#     prints this to stderr and exits non-zero with a different
#     message shape; the runner reports it as ``"abnormal exit
#     rc=N\n<stderr-tail>"``.
#
# Match conservatively against both shapes; over-matching here just
# means an unrelated transient error gets the 8s backoff instead of
# 1s (harmless), while under-matching means an OOM gets 1s and likely
# OOMs again (the bug we're fixing). The substrings are deliberately
# lowercased / case-insensitive checked.
_OOM_ERROR_MARKERS = (
    "killed by os",                   # runner_base._on_kill_signal
    "rc=-9",                          # explicit SIGKILL exit code
    "cgroup oom",                     # explicit kernel cgroup OOM
    "out of memory",                  # generic
    "memorymax",                      # systemd cgroup property name
    "reached heap limit",             # V8 native OOM
    "javascript heap out of memory",  # node default OOM phrase
    "fatal error: ineffective mark-compacts near heap limit",  # V8 specific
)


def _is_likely_oom_error(exc: Exception) -> bool:
    """Return True iff the exception message matches a known OOM
    signature. Used to bias the retry backoff toward giving the
    cgroup time to reclaim memory before re-spawning.

    Conservative + idempotent — never raises; bad exception object
    just falls through to False.
    """
    try:
        msg = str(exc).lower()
    except Exception:
        return False
    return any(marker in msg for marker in _OOM_ERROR_MARKERS)


def _log_oom_diagnostics(agent_id: str) -> None:
    """Snapshot the larkhelm cgroup's memory state into the debug log.

    Sole purpose: forensic. When OOM hits in production it's often
    investigated hours later when the kernel ring buffer / journalctl
    has already rotated. Persisting one line per OOM into
    ``_cfg.DEBUG_LOG`` keeps the signal close to the failure for
    later grep.

    Fail-soft: any disk / IO error is swallowed so this diagnostic
    helper can never compound an already-bad situation.
    """
    try:
        cgroup_root = Path("/sys/fs/cgroup/system.slice/larkhelm.service")
        if not cgroup_root.is_dir():
            return
        snapshot = {}
        for f in ("memory.current", "memory.high", "memory.max",
                  "memory.peak", "memory.swap.current"):
            try:
                snapshot[f] = (cgroup_root / f).read_text(encoding="utf-8").strip()
            except OSError:
                snapshot[f] = "?"
        _debug_log(
            f"[Crew] {agent_id} OOM diagnostic — cgroup state: " +
            " ".join(f"{k}={v}" for k, v in snapshot.items())
        )
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} OOM diagnostic snapshot failed: {e}")


def _run_agent_wrapper(state: CrewState, agent_id: str) -> None:
    """Agent execution shell: catches exceptions, updates state, detects exit markers.
    Process-level failures (subprocess crashes, etc.) are retried at most once.
    """
    from larkhelm.chat_state import _get_cwd
    from larkhelm.crew._state import _git_auto_commit

    last_exc: Exception = None
    backend_select_failure: bool = False  # set when NoBackendAvailableError surfaces
    validate_failure: bool = False        # set when _validate_output_artifact fires twice

    for proc_attempt in range(2):   # attempt 0 = first try, attempt 1 = retry
        try:
            result = _run_agent(state, agent_id)
            needs_retry = _detect_fail_marker(state.agents[agent_id].spec, result)
            if needs_retry:
                _debug_log(f"[Crew] {agent_id} fail marker detected, pending retry")
            # Output-file safety net: agents are instructed (via the prompt
            # injection in _run_agent) to call the Write tool to persist
            # their full output to ``.crew_workspace/{spec.output_file}``.
            # Reality: some agents emit a short closing marker (the result
            # field captures the last text chunk, often ≤100 chars) and
            # rely on Write — but if the Write call was skipped, truncated,
            # or wrote to the wrong path, the full PRD/design/etc. is gone
            # and downstream agents read an empty / missing file. This
            # observably bit P3 (PM emitted a 39196-token PRD that never
            # landed on disk; architect had to reverse-engineer it from
            # source + memory).
            #
            # Fallback: if the expected file is missing or too small AND
            # the in-memory result is substantially longer, persist
            # ``result`` to the expected path before the sync step runs.
            _persist_result_to_output_file_if_missing(state, agent_id, result)
            # Artifact contract check: if the on-disk file (or in-memory
            # result) contains a tool-call protocol leak or malformed JSON,
            # circuit-break before downstream agents read it. Retry once on
            # the first attempt (transient model glitch); on the second
            # failure fall through to ``emit_agent_failure(stage="validate")``
            # so the user gets a targeted card instead of watching the rest
            # of the pipeline burn tokens on an unusable contract.
            artifact_issue = _validate_output_artifact(state, agent_id, result)
            if artifact_issue:
                if proc_attempt == 0:
                    warn(
                        f"[Crew] {agent_id} output validation failed "
                        f"({artifact_issue}) — retrying with same backend"
                    )
                    with state.lock:
                        state.agents[agent_id].status     = AgentStatus.PENDING
                        state.agents[agent_id].start_time = None
                        state.agents[agent_id].result     = ""
                    time.sleep(1)
                    continue
                last_exc = RuntimeError(
                    f"output artifact contract violation: {artifact_issue}"
                )
                validate_failure = True
                # Quarantine the corrupt file BEFORE breaking out, so the
                # scheduler's partial-delivery rule does not see a
                # nominally-present output_file and let downstream agents
                # cascade-read the garbage (regression observed 2026-05-22).
                _quarantine_invalid_output(state, agent_id)
                break
            # Sync output_file before updating state to avoid holding the lock during IO
            feishu_url = _sync_output_file(state, agent_id)
            # Auto-commit after code-modification agents succeed (when dev_auto_commit=true)
            commit_hash = ""
            if not needs_retry and agent_id in ("implementer", "fixer"):
                cwd = _get_cwd(state.chat_id)
                commit_hash = _git_auto_commit(cwd, agent_id)
                if commit_hash:
                    with state.lock:
                        state.phase_commits[agent_id] = commit_hash
                    _debug_log(f"[Git] {agent_id} auto-committed: {commit_hash}")
            # Capture per-agent token stats
            crew_ns = f"{state.chat_id}__crew_{state.crew_id}_{agent_id}"
            try:
                from larkhelm.token_stats import get_crew_agent_tokens
                agent_tokens = get_crew_agent_tokens(crew_ns)
            except Exception:
                agent_tokens = {}
            with state.lock:
                state.agents[agent_id].status         = AgentStatus.DONE
                state.agents[agent_id].result         = result
                state.agents[agent_id].end_time       = time.time()
                state.agents[agent_id].needs_retry    = needs_retry
                state.agents[agent_id].feishu_doc_url = feishu_url
                state.agents[agent_id].tokens         = agent_tokens
            _debug_log(f"[Crew] {agent_id} done ({len(result)} chars)")
            _crew_update_card(state)
            return

        except QueryCancelledError:
            with state.lock:
                state.agents[agent_id].status   = AgentStatus.CANCELLED
                state.agents[agent_id].end_time = time.time()
            _crew_update_card(state)
            return

        except NoBackendAvailableError as e:
            # Config / health failure — retrying won't help, so skip the
            # second attempt and fall straight to the failure card branch
            # below. emit_agent_failure tags this as ``stage="backend_select"``
            # so the user sees the targeted "无可用 backend" hint.
            last_exc = e
            backend_select_failure = True
            break

        except Exception as e:
            last_exc = e
            if proc_attempt == 0:
                # OOM-aware retry backoff. A claude/node CLI killed by the
                # cgroup OOM-killer (rc=-9) leaves the cgroup near its
                # memory.max — page cache and V8 native allocations don't
                # release instantly. Retrying after 1s usually re-trips
                # the same OOM. 8s gives the kernel a chance to reclaim
                # + lets MAX_AI_PROCS=2 wave back below the high-water
                # mark before a second large process starts. Other errors
                # (HTTP transient, parse error, etc.) keep the original
                # 1s — no point waiting 8s on a quick recoverable failure.
                _oom = _is_likely_oom_error(e)
                backoff_sec = 8 if _oom else 1
                if _oom:
                    _debug_log(
                        f"[Crew] {agent_id} OOM-class failure on attempt 1/2 — "
                        f"backing off {backoff_sec}s before retry "
                        f"(let cgroup reclaim memory). Error: {str(e)[:200]}"
                    )
                    _log_oom_diagnostics(agent_id)
                else:
                    _debug_log(f"[Crew] {agent_id} process failed (attempt 1/2), retrying in {backoff_sec}s: {e}")
                time.sleep(backoff_sec)
                with state.lock:   # Reset to PENDING for retry
                    state.agents[agent_id].status     = AgentStatus.PENDING
                    state.agents[agent_id].start_time = None
                    state.agents[agent_id].result     = ""
                continue  # proceed to second attempt

    # Both attempts failed (or first attempt was a non-retryable backend-select
    # failure). Emit the user-facing ⚠️ card via _failure_card so the user
    # actually sees what went wrong; the helper handles state mutation +
    # heartbeat push internally and never raises.
    if validate_failure:
        stage = "validate"
    elif backend_select_failure:
        stage = "backend_select"
    else:
        stage = "run"
    if last_exc is None:
        last_exc = RuntimeError("unknown error")
    emit_agent_failure(state, agent_id, stage, last_exc)


# ═══════════════════════════════════════════════════════════════
#  Breakpoint confirmation
# ═══════════════════════════════════════════════════════════════

def _wait_for_breakpoint(state: CrewState, agent_id: str) -> bool:
    """Pause after PM completes, update main card to show 继续/取消 buttons and wait for the user to click. Returns True=continue, False=cancel."""
    import larkhelm.config as _cfg
    from larkhelm.crew._state import (
        _breakpoint_meta, _breakpoint_events, _breakpoint_results,
    )

    crew_id = state.crew_id

    # Register wait event.
    # Note: if signal_breakpoint is called before this function registers the event (theoretical race),
    # _breakpoint_results already has a value and should not be overwritten; fire bp_ev directly instead.
    bp_ev = threading.Event()
    with _breakpoint_meta:
        _breakpoint_events[crew_id] = bp_ev
        if crew_id in _breakpoint_results:
            # Signal arrived early; wake immediately without overwriting the existing result
            bp_ev.set()
        else:
            _breakpoint_results[crew_id] = False  # Default: cancel on timeout

    # Update main card to breakpoint phase — _build_card shows "继续/取消" buttons for this phase
    with state.lock:
        state.phase              = "breakpoint"
        state.breakpoint_agent_id = agent_id
    _crew_update_card(state)

    # Wait for user decision; poll to support /cancel interruption.
    # Phase C: the deadline is sourced from ``_cfg.CREW_BREAKPOINT_TIMEOUT_SEC``
    # (default 1800s, configurable via crew_breakpoint_timeout_sec) instead of
    # the previous hardcoded ``min(RESPONSE_TIMEOUT*2, 3600)``. On timeout we
    # both set ``cancel_ev`` (so the executor breaks out of its wave loop) and
    # emit a dedicated breakpoint-timeout card (so the user sees why the task
    # ended).
    # Trust whatever ``_cfg.CREW_BREAKPOINT_TIMEOUT_SEC`` resolves to —
    # ``_init_runtime`` already floors the configured value at 60s, and
    # tests need to monkeypatch lower for fast assertions.
    bp_timeout = int(getattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 1800))
    bp_deadline = time.time() + bp_timeout
    timed_out = False
    while time.time() < bp_deadline:
        if state.cancel_ev.is_set():
            _debug_log("[Crew] breakpoint wait interrupted by /cancel")
            break
        if bp_ev.wait(timeout=min(2.0, max(0.0, bp_deadline - time.time()))):  # clamp to avoid overshooting deadline
            break
    else:
        # Loop exited via condition (no break) — that's the timeout branch.
        timed_out = True

    with _breakpoint_meta:
        confirmed = _breakpoint_results.pop(crew_id, False)
        _breakpoint_events.pop(crew_id, None)

    if timed_out and not confirmed:
        _debug_log(f"[Crew] breakpoint timeout after {bp_timeout}s, auto-cancelling")
        state.cancel_ev.set()
        emit_breakpoint_timeout(state)
        return False

    return confirmed


# ═══════════════════════════════════════════════════════════════
#  Scheduler
# ═══════════════════════════════════════════════════════════════

def _execute(state: CrewState, total_timeout: int):
    """Execute all agents in topological waves, supporting retry fallback when QA/Reviewer fails."""
    import larkhelm.config as _cfg
    from larkhelm.chat_state import _get_cwd
    from larkhelm.crew._scheduler import (
        _topo_waves, _topo_waves_subset, _get_failed_dep,
    )
    from larkhelm.crew._checkpoint import _save_checkpoint

    cancel_ev    = state.cancel_ev
    deadline     = time.time() + total_timeout

    # Create per-project Feishu folder before running any agents
    _ensure_crew_folder(state)

    wave_queue   = deque(_topo_waves(state.plan.agents))
    retry_counts: dict[str, int] = {spec.id: 0 for spec in state.plan.agents}

    # Workspace path resolved once and passed into ``_get_failed_dep`` for
    # the partial-delivery rule (see scheduler docstring). When an upstream
    # FAILED agent has already written its declared ``output_file``, we
    # let downstream agents run instead of cascading the FAILED into N
    # more skipped agents. Real motivator: implementer's claude CLI gets
    # OOM-killed after the Write tool atomically committed changes.md,
    # but fixer/qa/reviewer were all marked "skipped because upstream
    # implementer failed". Now fixer can actually run.
    try:
        _crew_ws_path = Path(_get_cwd(state.chat_id)) / ".crew_workspace"
    except Exception:
        _crew_ws_path = None

    while wave_queue:
        if cancel_ev.is_set():
            raise QueryCancelledError("Crew cancelled")
        if time.time() > deadline:
            with state.lock:
                for spec in state.plan.agents:
                    if state.agents[spec.id].status == AgentStatus.PENDING:
                        state.agents[spec.id].status = AgentStatus.FAILED
                        state.agents[spec.id].error  = "Crew 总超时"
            break

        wave = wave_queue.popleft()

        # ── Phase 4: failure propagation — skip agents whose upstream is FAILED ──
        runnable: list[AgentSpec] = []
        with state.lock:
            for spec in wave:
                failed_dep = _get_failed_dep(state, spec, _crew_ws_path)
                if failed_dep:
                    _debug_log(f"[Crew] {spec.id} skipped because upstream {failed_dep} failed")
                    ag = state.agents[spec.id]
                    ag.status   = AgentStatus.FAILED
                    ag.error    = f"upstream {failed_dep} failed"
                    ag.end_time = time.time()
                elif spec.trigger_only and state.agents[spec.id].retry_count == 0:
                    # Skip on first wave: mark as DONE (trigger pending), do not actually execute
                    ag = state.agents[spec.id]
                    ag.status   = AgentStatus.DONE
                    ag.result   = ""   # empty result, sentinel
                    ag.end_time = time.time()
                    _debug_log(f"[Crew] {spec.id} trigger_only, skipping first wave")
                else:
                    runnable.append(spec)

        if not runnable:
            _crew_update_card(state)
            continue

        threads = [
            threading.Thread(
                target=_run_agent_wrapper,
                args=(state, spec.id),
                daemon=True,
                name=f"crew-{spec.id}-{state.crew_id[:6]}",
            )
            for spec in runnable
        ]
        for t in threads:
            t.start()

        # Wait for this wave to complete; check cancel every 0.5s
        done_ev = threading.Event()
        def _waiter(ts=threads):
            for t in ts:
                t.join()
            done_ev.set()
        threading.Thread(target=_waiter, daemon=True).start()
        while not done_ev.is_set():
            if cancel_ev.is_set():
                raise QueryCancelledError("Crew cancelled")
            done_ev.wait(timeout=0.5)

        # ── Check if any agent in this wave needs retry (read needs_retry under lock) ──
        retry_specs: list[AgentSpec] = []
        with state.lock:
            for spec in wave:
                if state.agents[spec.id].needs_retry:
                    retry_specs.append(spec)

        for spec in retry_specs:
            ag = state.agents[spec.id]
            if retry_counts[spec.id] >= spec.max_retries:
                _debug_log(f"[Crew] {spec.id} reached max retries ({spec.max_retries}), giving up")
                with state.lock:
                    ag.needs_retry = False
                if spec.is_gatekeeper:
                    # Gatekeeper final rejection: clear remaining queue and go straight to synthesis
                    _debug_log(f"[Crew] gatekeeper {spec.id} final rejection, skipping remaining tasks")
                    wave_queue.clear()
                if spec.hard_fail_on_exhaust:
                    raise HardFailError(f"{spec.role}（{spec.id}）最终失败，已重试 {spec.max_retries} 次")
                continue

            # Not enough remaining time for a retry round (estimate: retry rounds × RESPONSE_TIMEOUT)
            retry_num = retry_counts[spec.id] + 1
            est_needed = (len(spec.retry_target) + 1) * _cfg.RESPONSE_TIMEOUT
            if time.time() + est_needed > deadline:
                _debug_log(f"[Crew] {spec.id} insufficient time remaining (~{est_needed}s needed), skipping retry #{retry_num}")
                with state.lock:
                    ag.needs_retry = False
                if spec.hard_fail_on_exhaust:
                    raise HardFailError(f"{spec.role} ({spec.id}) retries abandoned due to timeout")
                continue

            retry_counts[spec.id] += 1
            _debug_log(f"[Crew] {spec.id} triggering retry #{retry_counts[spec.id]}, resetting: {spec.retry_target}")

            # Build feedback: if the triggering agent has an output_file, pass only the file path reference.
            # (The retry_target agent's system prompt already instructs it to read that file;
            #  passing content directly would double-transmit and waste ~1000-3000 tokens/retry.)
            # If no output_file (/crew dynamic scenario), pass the beginning of the output (errors are typically at the top).
            _cwd_fb = _get_cwd(state.chat_id)
            if ag.spec.output_file:
                _fb_path = Path(_cwd_fb) / ".crew_workspace" / ag.spec.output_file
                feedback = (
                    f"Failure report: {_fb_path}\n"
                    f"Use the Read tool to read this file for details, then fix each issue."
                )
            else:
                # Take the beginning (bug list is typically first) rather than the end (usually just TESTS_FAILED etc.)
                feedback = ag.result[:2000] if ag.result else ag.error[:1000]

            # Reset this agent + agents in retry_target
            targets = list(spec.retry_target) + [spec.id]
            with state.lock:
                for tid in targets:
                    if tid not in state.agents:
                        continue
                    ta = state.agents[tid]
                    ta.status     = AgentStatus.PENDING
                    ta.result     = ""
                    ta.error      = ""
                    ta.start_time = None
                    ta.end_time   = None
                    ta.needs_retry = False
                    ta.retry_count += 1
                    ta.round_label = f"Round {ta.retry_count + 1}"
                    # Only inject feedback into upstream retry_target agents (e.g. engineer), not into self
                    if tid != spec.id:
                        ta.feedback = feedback

            # Re-enqueue the retry agent subset
            retry_waves = _topo_waves_subset(state.plan.agents, set(targets))
            wave_queue.extendleft(reversed(retry_waves))
            _crew_update_card(state)
            break  # At most one retry trigger per wave

        # ── Save checkpoint: this wave is complete ────────────────────
        if not retry_specs:
            completed_ids = [spec.id for spec in state.plan.agents
                             if state.agents[spec.id].status in
                             (AgentStatus.DONE, AgentStatus.FAILED)]
            _save_checkpoint(state, completed_ids)

        # ── PM short-circuit: ``TASK_ALREADY_COMPLETE`` marker ─────────
        # When PM (or any planner-tier first-wave agent) detects that the
        # task's acceptance points are already met by the current codebase,
        # it emits this single-line marker. Drain the remaining waves —
        # downstream agents would just waste tokens and time re-discovering
        # "nothing to do". The synthesis phase still runs so the user gets
        # a nice "already complete" card with PM's reason.
        if not retry_specs and _check_task_already_complete(state, wave):
            with state.lock:
                for _spec in state.plan.agents:
                    _ag = state.agents.get(_spec.id)
                    if _ag and _ag.status == AgentStatus.PENDING:
                        _ag.status = AgentStatus.SKIPPED
                        _ag.error = "TASK_ALREADY_COMPLETE — PM 判定任务已在代码中完成"
            wave_queue.clear()
            break

        # ── Phase 3.1: breakpoint — pause after an agent with breakpoint=True completes ──
        if not retry_specs:  # Only check breakpoints when there are no retries (skip during retry)
            for spec in wave:   # Iterate original wave; state guard ensures only DONE agents trigger
                ag = state.agents.get(spec.id)
                if (spec.breakpoint and ag and ag.status == AgentStatus.DONE):
                    confirmed = _wait_for_breakpoint(state, spec.id)
                    if not confirmed:
                        raise QueryCancelledError("User cancelled")
                    # Restore main card to running state
                    with state.lock:
                        state.phase = "running"
                    _crew_update_card(state)
                    break


# ── PM TASK_ALREADY_COMPLETE marker ─────────────────────────────────────

_TASK_COMPLETE_MARKER = "TASK_ALREADY_COMPLETE"


def _extract_task_complete_marker(state: CrewState) -> "tuple[bool, str]":
    """Single source of truth for the marker detection. Returns
    ``(hit, reason)`` where:

      • ``hit`` is True iff PM is DONE, not a retry, AND the marker
        appears at the start of either PM's in-memory result OR the
        first non-blank line of ``.crew_workspace/prd.md``.
      • ``reason`` is the trailing free-text after ``TASK_ALREADY_COMPLETE:``
        (empty if no colon / nothing after).

    Round-2 review caught that ``_check_task_already_complete`` had the
    prd.md fallback but ``_synthesize`` only inspected ``pm_state.result``,
    so PM honouring the system-prompt contract by writing only the file
    (a very common shape — PM emits "PRD written." as result after the
    Write tool call) would correctly short-circuit ``_execute`` but
    silently trigger the LLM synthesis path. Sharing this helper
    guarantees the two phases never disagree.
    """
    pm_state = state.agents.get("pm")
    if not pm_state or pm_state.status != AgentStatus.DONE:
        return False, ""
    if pm_state.retry_count > 0:
        return False, ""

    # Primary source: in-memory result. Fallback: first non-blank line of
    # prd.md (covers the case where the agent honoured the system-prompt
    # contract by writing the marker to the file but the in-memory result
    # is a short closing summary).
    candidates: list[str] = []
    head = (pm_state.result or "").lstrip()
    if head:
        candidates.append(head)
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(state.chat_id)
        prd_path = Path(cwd) / ".crew_workspace" / "prd.md"
        if prd_path.exists():
            file_head = prd_path.read_text(encoding="utf-8").lstrip()
            if file_head:
                candidates.append(file_head)
    except Exception:
        pass

    for cand in candidates:
        if cand.startswith(_TASK_COMPLETE_MARKER):
            first_line = cand.splitlines()[0].strip()
            reason = first_line.removeprefix(_TASK_COMPLETE_MARKER).lstrip(": ").strip()
            return True, reason
    return False, ""


def _check_task_already_complete(state: CrewState, wave) -> bool:
    """Return True iff a just-completed first-wave PM agent emitted the
    ``TASK_ALREADY_COMPLETE: <reason>`` marker, signalling that the
    codebase already satisfies the user's acceptance points.

    Only fires when:
      • the wave contains an agent with id ``"pm"`` (dev pipeline shape;
        /crew dynamic plans are unaffected) — guard so this helper can't
        misfire on a non-dev plan whose first agent happens to be named
        differently.
      • the marker is found in either PM's in-memory result or in
        ``.crew_workspace/prd.md`` (see ``_extract_task_complete_marker``).

    The check is read-only — the caller is responsible for marking the
    remaining agents as SKIPPED.
    """
    if not any(s.id == "pm" for s in wave):
        return False
    hit, _reason = _extract_task_complete_marker(state)
    return hit


def _execute_from(state: CrewState, total_timeout: int, skip_ids: set):
    """Continue execution after a set of already-completed agents, skipping agents in skip_ids."""
    from larkhelm.chat_state import _get_cwd
    from larkhelm.crew._scheduler import _topo_waves, _get_failed_dep
    from larkhelm.crew._checkpoint import _save_checkpoint

    cancel_ev  = state.cancel_ev
    deadline   = time.time() + total_timeout
    all_waves  = _topo_waves(state.plan.agents)
    wave_queue = deque()

    # Same partial-delivery context as ``_execute``; see commentary there.
    try:
        _crew_ws_path = Path(_get_cwd(state.chat_id)) / ".crew_workspace"
    except Exception:
        _crew_ws_path = None

    for wave in all_waves:
        # If all agents in this wave are already completed, skip the wave
        if all(spec.id in skip_ids for spec in wave):
            continue
        wave_queue.append(wave)

    # Skipped agent states are already restored from checkpoint; keep them as-is.
    # Reset incomplete agents to PENDING (they may have been interrupted mid-run last time).
    with state.lock:
        for spec in state.plan.agents:
            if spec.id not in skip_ids:
                ag = state.agents[spec.id]
                if ag.status not in (AgentStatus.DONE, AgentStatus.FAILED):
                    ag.status     = AgentStatus.PENDING
                    ag.result     = ""
                    ag.error      = ""
                    ag.start_time = None
                    ag.end_time   = None

    state.phase = "running"
    _crew_update_card(state)

    while wave_queue:
        if cancel_ev.is_set():
            raise QueryCancelledError("Crew cancelled")
        if time.time() > deadline:
            break

        wave = wave_queue.popleft()
        runnable: list[AgentSpec] = []
        with state.lock:
            for spec in wave:
                if spec.id in skip_ids:
                    continue
                failed_dep = _get_failed_dep(state, spec, _crew_ws_path)
                if failed_dep:
                    ag = state.agents[spec.id]
                    ag.status = AgentStatus.FAILED
                    ag.error  = f"upstream {failed_dep} failed"
                    ag.end_time = time.time()
                elif spec.trigger_only and state.agents[spec.id].retry_count == 0:
                    ag = state.agents[spec.id]
                    ag.status  = AgentStatus.DONE
                    ag.result  = ""
                    ag.end_time = time.time()
                else:
                    runnable.append(spec)

        if not runnable:
            _crew_update_card(state)
            continue

        threads = [
            threading.Thread(target=_run_agent_wrapper, args=(state, spec.id),
                             daemon=True, name=f"crew-{spec.id}-{state.crew_id[:6]}")
            for spec in runnable
        ]
        for t in threads:
            t.start()
        done_ev = threading.Event()
        def _waiter(ts=threads):
            for t in ts: t.join()
            done_ev.set()
        threading.Thread(target=_waiter, daemon=True).start()
        while not done_ev.is_set():
            if cancel_ev.is_set():
                raise QueryCancelledError("Crew cancelled")
            done_ev.wait(timeout=0.5)

        # Save checkpoint: this wave is complete
        completed = [spec.id for spec in state.plan.agents
                     if state.agents[spec.id].status in
                     (AgentStatus.DONE, AgentStatus.FAILED)]
        _save_checkpoint(state, completed)


# ═══════════════════════════════════════════════════════════════
#  Synthesis (Manager summary)
# ═══════════════════════════════════════════════════════════════

def _synthesize(state: CrewState) -> str:
    """Manager synthesizes all agent results and produces the final deliverable."""
    from larkhelm.chat_state import _get_cwd
    from larkhelm.perm import grant_yolo, revoke_yolo
    from larkhelm.backend_cli import run_claude as _bc_run_claude
    from larkhelm.backend_registry import BACKEND_REGISTRY as _reg

    cancel_ev = state.cancel_ev
    cwd       = _get_cwd(state.chat_id)

    # ── TASK_ALREADY_COMPLETE short-circuit ─────────────────────────────
    # When PM short-circuited the pipeline (see ``_check_task_already_complete``)
    # the downstream agents are SKIPPED — there's nothing meaningful to
    # synthesise. Return a templated card body so the user sees a clean
    # "已完成" message instead of a synthesis-LLM hallucinating about empty
    # inputs.
    #
    # Round-2 review caught that this branch previously only inspected
    # ``pm_state.result``, but ``_check_task_already_complete`` also
    # falls back to prd.md. The asymmetry meant a PM that wrote the
    # marker only to the file (the more common shape: PM calls Write,
    # then emits "Done." as the in-memory result) correctly short-
    # circuited ``_execute`` but quietly hit this branch's "no" path
    # and ran a full synthesis LLM call. Fixed by sharing
    # ``_extract_task_complete_marker``.
    _marker_hit, _marker_reason = _extract_task_complete_marker(state)
    if _marker_hit:
        skipped_ids = [s.id for s in state.plan.agents
                       if state.agents[s.id].status == AgentStatus.SKIPPED]
        return (
            "**✅ 任务已经在代码中完成**\n\n"
            f"PM 判定：{_marker_reason or '（PM 未给出详细说明）'}\n\n"
            f"已跳过的阶段：{', '.join(skipped_ids) if skipped_ids else '（无）'}\n\n"
            "若你认为这个判断不对，请用 `/dev --no-confirm <更具体的需求>` 强制重跑。"
        )

    # No synthesis_prompt and only one agent → return its result directly
    plan = state.plan
    if not plan.synthesis_prompt and len(plan.agents) == 1:
        only = list(state.agents.values())[0]
        return only.result if only.status == AgentStatus.DONE else only.error

    # Build synthesis prompt.
    # Agents with output_file: pass only the path (Claude will use Read to retrieve it, avoiding double transmission).
    # Agents without output_file: pass a summary (the only way to get their result).
    parts = []
    _cwd_synth = _get_cwd(state.chat_id)
    for spec in plan.agents:
        a = state.agents[spec.id]
        if a.status == AgentStatus.DONE:
            if spec.output_file:
                out_path = Path(_cwd_synth) / ".crew_workspace" / spec.output_file
                feishu_note = f"\n飞书文档：{a.feishu_doc_url}" if a.feishu_doc_url else ""
                parts.append(f"## {spec.role} ({spec.id})\nOutput file: {out_path}{feishu_note}")
            else:
                workspace   = _workspace_dir(state.chat_id, state.crew_id)
                result_file = workspace / f"{spec.id}_result.txt"
                preview     = a.result[:CREW_RESULT_PREVIEW]
                suffix      = "…" if len(a.result) > CREW_RESULT_PREVIEW else ""
                parts.append(
                    f"## {spec.role} ({spec.id})\n{preview}{suffix}\n"
                    f"Full output: {result_file}"
                )
        elif a.status == AgentStatus.FAILED:
            if a.result:
                # Partial output from timeout or similar; include in synthesis, annotate truncation
                preview = a.result[:CREW_RESULT_PREVIEW]
                suffix  = "…(truncated due to timeout)" if len(a.result) > CREW_RESULT_PREVIEW else "(truncated due to timeout)"
                parts.append(f"## {spec.role} ({spec.id})\n⚠️ Incomplete execution, partial output:\n{preview}{suffix}")
            else:
                parts.append(f"## {spec.role} ({spec.id})\nExecution failed: {a.error}")

    if not parts:
        return "All agents produced no results."

    synthesis_prompt = (plan.synthesis_prompt or "Please synthesize the outputs of all agents above and produce the final delivery report.")
    full_prompt      = synthesis_prompt + "\n\n" + "\n\n".join(parts)

    synth_ns = f"{state.chat_id}__crew_{state.crew_id}_synth"
    synth_cancel = threading.Event()

    def _watch_cancel():
        while not synth_cancel.is_set():
            if cancel_ev.is_set():
                synth_cancel.set()
                return
            time.sleep(0.3)
    threading.Thread(target=_watch_cancel, daemon=True).start()

    grant_yolo(synth_ns)
    _synth_spec = _reg.get_orchestrator()
    if _synth_spec is None:
        raise RuntimeError("No orchestrator backend available for synthesis")
    _API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api")
    usage_holder: dict = {}
    model_label = _synth_spec.model or _synth_spec.id
    try:
        if _synth_spec.provider in _API_PROVIDERS:
            import larkhelm.backend_api as _bapi
            _fn = {"anthropic_api": _bapi.run_anthropic,
                   "google_api": _bapi.run_google,
                   "openai_compat_api": _bapi.run_openai_compat}[_synth_spec.provider]
            result, _ = _fn(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                            history=[], cancel_ev=synth_cancel, on_text=None,
                            suppress_token_recording=True, usage_holder=usage_holder)
        elif _synth_spec.provider == "gemini_cli":
            from larkhelm.backend_cli import run_gemini as _bc_run_gemini
            result = _bc_run_gemini(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                                    sid=None, cwd=cwd, cancel_ev=synth_cancel,
                                    on_text=None, use_session=False,
                                    suppress_token_recording=True, usage_holder=usage_holder)
        elif _synth_spec.provider == "kimi_cli":
            from larkhelm.backend_cli import run_kimi as _bc_run_kimi
            result = _bc_run_kimi(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                                  sid=None, cwd=cwd, cancel_ev=synth_cancel, on_text=None,
                                  suppress_token_recording=True, usage_holder=usage_holder)
        else:
            result = _bc_run_claude(
                spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                sid=None, cwd=cwd, cancel_ev=synth_cancel,
                on_text=None, allow_retry=False, session_namespace=synth_ns,
                suppress_token_recording=True, usage_holder=usage_holder,
            )
        if usage_holder:
            from larkhelm.token_stats import record_token_usage
            record_token_usage(state.chat_id, model_label, usage_holder)
    finally:
        synth_cancel.set()
        revoke_yolo(synth_ns)

    return result.strip() or "Synthesis phase produced no output."


# ═══════════════════════════════════════════════════════════════
#  Main flow
# ═══════════════════════════════════════════════════════════════

def _run_crew(state: CrewState, total_timeout: int):
    """Complete crew execution flow (runs in a dedicated thread)."""
    from larkhelm.chat_state import _get_cwd
    from larkhelm.card_builder import _fmt_elapsed, _split_md, _make_card
    from larkhelm.lark_client import _reply_card_raw, send_card
    from larkhelm.crew._state import _register_crew_card
    from larkhelm.crew._checkpoint import _clear_checkpoint

    hb_stop   = threading.Event()
    hb_thread = _start_heartbeat(state, hb_stop)

    try:
        # ── Execute agents ───────────────────────────────
        with state.lock:
            state.phase = "running"
        _crew_update_card(state)

        try:
            _execute(state, total_timeout)
        except QueryCancelledError:
            with state.lock:
                state.phase = "cancelled"
            _crew_update_card(state)
            # Clear checkpoint so the next bridge restart doesn't auto-resume
            # a crew the user already gave up on (e.g. breakpoint timeout,
            # /cancel button, /cancel text command). Exception: during a
            # SIGTERM shutdown, ``cancel_all_crews`` deliberately re-saved
            # the checkpoint with ``phase="running"`` *immediately before*
            # setting ``cancel_ev``, precisely so ``resume_interrupted_crews``
            # can pick up where we left off after the restart. Preserve that.
            from larkhelm.concurrency import is_shutting_down
            if not is_shutting_down():
                _clear_checkpoint(state.chat_id)
            return
        except HardFailError as e:
            with state.lock:
                state.phase = "failed"
                state.final_output = f"❌ Pipeline hard failure: {e}"
            _crew_update_card(state)
            log_entry(state.chat_id, "error", str(e), model="crew")
            # Hard-fail is terminal — never resume.
            _clear_checkpoint(state.chat_id)
            return

        if state.cancel_ev.is_set():
            with state.lock:
                state.phase = "cancelled"
            _crew_update_card(state)
            from larkhelm.concurrency import is_shutting_down
            if not is_shutting_down():
                _clear_checkpoint(state.chat_id)
            return

        # ── Synthesis ────────────────────────────────────
        with state.lock:
            state.phase = "synthesizing"
        _crew_update_card(state)

        try:
            final = _synthesize(state)
        except Exception as e:
            _debug_log(f"[Crew] synthesis failed: {e}")
            # Fall back to concatenating all completed results when synthesis fails
            final = "\n\n---\n\n".join(
                f"**{state.agents[spec.id].spec.role}**\n{state.agents[spec.id].result}"
                for spec in state.plan.agents
                if state.agents[spec.id].status == AgentStatus.DONE
            ) or "Synthesis failed, no results available."

        # ── Git change statistics (appended to delivery report) ──────
        try:
            _cwd = _get_cwd(state.chat_id)
            # Prefer uncommitted changes; if already committed, show the most recent commit's changes
            _diff = subprocess.run(
                ["git", "diff", "--stat", "HEAD"],
                cwd=_cwd, capture_output=True, text=True, timeout=10,
            )
            _diff_out = _diff.stdout.strip()
            if not _diff_out:
                # Engineer may have already committed; check the most recent commit
                _log = subprocess.run(
                    ["git", "log", "-1", "--stat", "--no-merges", "--format="],
                    cwd=_cwd, capture_output=True, text=True, timeout=10,
                )
                _diff_out = _log.stdout.strip()
            if _diff_out:
                final = final + "\n\n---\n\n**📊 Code change statistics**\n```\n" + _diff_out + "\n```"
        except Exception:
            pass  # Silently ignore: non-git repo, git not installed, timeout, etc.

        # Append Feishu folder link, per-agent doc links, and git commit info
        _cwd_val = _get_cwd(state.chat_id)
        from larkhelm.crew._state import _git_head
        _ws_note = "\n\n---\n\n"
        if state.feishu_folder_url:
            _ws_note += f"📁 [飞书项目文件夹]({state.feishu_folder_url})"
        else:
            _ws_note += f"📁 Workspace: `{_cwd_val}/.crew_workspace/`"
        # List per-agent Feishu doc links (deduplicated: same URL only once)
        _seen_urls: set[str] = set()
        _doc_links = []
        for spec in state.plan.agents:
            _ag = state.agents[spec.id]
            if _ag.feishu_doc_url and _ag.feishu_doc_url not in _seen_urls:
                _seen_urls.add(_ag.feishu_doc_url)
                _doc_links.append(f"[{spec.role}]({_ag.feishu_doc_url})")
        if _doc_links:
            _ws_note += "\n📄 " + "  ·  ".join(_doc_links)
        if state.phase_commits:
            _commits_str = "  ".join(
                f"`{aid}:{h}`" for aid, h in state.phase_commits.items()
            )
            _ws_note += f"\n🔖 Auto-committed: {_commits_str}"
        elif state.git_head_before:
            _cur = _git_head(_cwd_val)
            if _cur and _cur != state.git_head_before:
                _ws_note += f"\n📝 Code changed (start: `{state.git_head_before}` → current: `{_cur}`)"
        final = final + _ws_note

        # Stop the heartbeat before writing the terminal state.  Without this,
        # a heartbeat that read the old phase ("synthesizing") just before we set
        # "done" can complete its API call after _crew_update_card and overwrite
        # the final card.
        hb_stop.set()
        hb_thread.join(timeout=10.0)

        with state.lock:
            state.phase        = "done"
            state.final_output = final
        _crew_update_card(state)

        # Completion notification: reply to the original user message so the user receives an alert
        _label   = "Dev" if state.kind == "dev" else "Crew"
        elapsed  = _fmt_elapsed(time.time() - state.start_time)
        n_agents = len(state.plan.agents)
        notify_body = (
            f"**{state.plan.title}** completed\n\n"
            f"{n_agents} agents · elapsed {elapsed}"
        )
        notify_card = _make_card(f"✅ {_label} complete", notify_body, color="green")
        if state.trigger_msg_id:
            notify_mid = _reply_card_raw(state.trigger_msg_id, notify_card, in_thread=False)
        else:
            notify_mid = send_card(state.chat_id, f"✅ {_label} complete", notify_body, color="green")

        # Extra-long output: send additional cards (filter empty chunks to avoid blank cards)
        chunks = _split_md(final)
        if len(chunks) > 1:
            for chunk in chunks[1:]:
                if chunk.strip():
                    send_card(state.chat_id, "📄 Continued (Crew output)", chunk, color="green")

        log_entry(state.chat_id, "assistant", final, model="crew")
        _clear_checkpoint(state.chat_id)
        # Register both progress card and notification card in crew_card_index so replying to either hits context
        if state.card_mid:
            _register_crew_card(state.card_mid, state.chat_id, state.plan.title, final)
        if notify_mid:
            _register_crew_card(notify_mid, state.chat_id, state.plan.title, final)

    finally:
        hb_stop.set()
