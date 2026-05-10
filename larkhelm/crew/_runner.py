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

from larkhelm.log import _debug_log, log_entry
from larkhelm.ai_runner import QueryCancelledError
from larkhelm.crew_types import (
    HardFailError, AgentSpec, AgentState, AgentStatus, CrewState,
    CREW_RESULT_PREVIEW,
)
from larkhelm.crew_card import _crew_update_card, _start_heartbeat


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

    # Resolve memory context for this agent (B1/B6):
    # API backends receive it as extra_system; CLI backends get a [System] prefix.
    # hermes orchestrators have their own context and don't need memory injection.
    _crew_mem_ctx = ""
    if not spec.model.startswith("hermes_"):
        try:
            from larkhelm.memory import get_project_memory_context
            _crew_mem_ctx = get_project_memory_context(state.chat_id, cwd=str(cwd))
        except Exception as e:
            _debug_log(f"[Crew] memory load failed: {e}")

    # Timeout control: start countdown only after acquiring the process slot (semaphore),
    # so waiting time is not counted toward the timeout.
    agent_cancel  = threading.Event()
    _slot_ready   = threading.Event()   # fired when the semaphore slot is acquired

    def _timeout_watcher():
        # Phase 1: wait for process slot while monitoring crew-level cancellation
        while not _slot_ready.is_set():
            if cancel_ev.is_set():
                agent_cancel.set()
                return
            time.sleep(0.3)
        # Phase 2: process has started; begin counting spec.timeout from now
        # Use a longer grace period: soft timeout releases the semaphore but keeps process running
        soft_deadline = time.time() + spec.timeout
        hard_deadline = time.time() + max(spec.timeout * 2, _cfg.HARD_TIMEOUT)
        soft_fired = False
        while time.time() < hard_deadline:
            if cancel_ev.is_set():
                agent_cancel.set()
                return
            if not soft_fired and time.time() >= soft_deadline:
                soft_fired = True
                _debug_log(f"[Crew] {agent_id} soft timeout ({spec.timeout}s), releasing lock but keeping process running")
            time.sleep(0.3)
        agent_cancel.set()

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
        if spec.model == "gemini":
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
        elif spec.model == "kimi":
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
        elif spec.model.startswith("hermes_"):
            # Hermes multi-agent orchestrator modes: race, split, review
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
            from larkhelm.backend_registry import BACKEND_REGISTRY as _reg
            _spec = _reg.get_orchestrator()
            if _spec is None:
                raise RuntimeError("No orchestrator backend available for crew agent")
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


def _run_agent_wrapper(state: CrewState, agent_id: str) -> None:
    """Agent execution shell: catches exceptions, updates state, detects exit markers.
    Process-level failures (subprocess crashes, etc.) are retried at most once.
    """
    from larkhelm.chat_state import _get_cwd
    from larkhelm.crew._state import _git_auto_commit

    last_exc: Exception = None

    for proc_attempt in range(2):   # attempt 0 = first try, attempt 1 = retry
        try:
            result = _run_agent(state, agent_id)
            needs_retry = _detect_fail_marker(state.agents[agent_id].spec, result)
            if needs_retry:
                _debug_log(f"[Crew] {agent_id} fail marker detected, pending retry")
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

        except Exception as e:
            last_exc = e
            if proc_attempt == 0:
                _debug_log(f"[Crew] {agent_id} process failed (attempt 1/2), retrying in 1s: {e}")
                time.sleep(1)
                with state.lock:   # Reset to PENDING for retry
                    state.agents[agent_id].status     = AgentStatus.PENDING
                    state.agents[agent_id].start_time = None
                    state.agents[agent_id].result     = ""
                continue  # proceed to second attempt

    # Both attempts failed
    with state.lock:
        state.agents[agent_id].status   = AgentStatus.FAILED
        state.agents[agent_id].error    = str(last_exc)[:300] if last_exc else "unknown error"
        state.agents[agent_id].end_time = time.time()
    _debug_log(f"[Crew] {agent_id} permanently failed (both attempts failed): {last_exc}")
    _crew_update_card(state)


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

    # Wait for user decision; poll to support /cancel interruption; max 60 minutes
    bp_deadline = time.time() + min(_cfg.RESPONSE_TIMEOUT * 2, 3600)
    while time.time() < bp_deadline:
        if state.cancel_ev.is_set():
            _debug_log(f"[Crew] breakpoint wait interrupted by /cancel")
            break
        if bp_ev.wait(timeout=2.0):   # wait returns True when the user has clicked
            break

    with _breakpoint_meta:
        confirmed = _breakpoint_results.pop(crew_id, False)
        _breakpoint_events.pop(crew_id, None)

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
                failed_dep = _get_failed_dep(state, spec)
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


def _execute_from(state: CrewState, total_timeout: int, skip_ids: set):
    """Continue execution after a set of already-completed agents, skipping agents in skip_ids."""
    from larkhelm.crew._scheduler import _topo_waves, _get_failed_dep
    from larkhelm.crew._checkpoint import _save_checkpoint

    cancel_ev  = state.cancel_ev
    deadline   = time.time() + total_timeout
    all_waves  = _topo_waves(state.plan.agents)
    wave_queue = deque()

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
                failed_dep = _get_failed_dep(state, spec)
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
    try:
        if _synth_spec.provider in _API_PROVIDERS:
            import larkhelm.backend_api as _bapi
            _fn = {"anthropic_api": _bapi.run_anthropic,
                   "google_api": _bapi.run_google,
                   "openai_compat_api": _bapi.run_openai_compat}[_synth_spec.provider]
            result, _ = _fn(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                            history=[], cancel_ev=synth_cancel, on_text=None)
        elif _synth_spec.provider == "gemini_cli":
            from larkhelm.backend_cli import run_gemini as _bc_run_gemini
            result = _bc_run_gemini(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                                    sid=None, cwd=cwd, cancel_ev=synth_cancel,
                                    on_text=None, use_session=False)
        elif _synth_spec.provider == "kimi_cli":
            from larkhelm.backend_cli import run_kimi as _bc_run_kimi
            result = _bc_run_kimi(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                                  sid=None, cwd=cwd, cancel_ev=synth_cancel, on_text=None)
        else:
            result = _bc_run_claude(
                spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                sid=None, cwd=cwd, cancel_ev=synth_cancel,
                on_text=None, allow_retry=False, session_namespace=synth_ns,
            )
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
            return
        except HardFailError as e:
            with state.lock:
                state.phase = "failed"
                state.final_output = f"❌ Pipeline hard failure: {e}"
            _crew_update_card(state)
            log_entry(state.chat_id, "error", str(e), model="crew")
            return

        if state.cancel_ev.is_set():
            with state.lock:
                state.phase = "cancelled"
            _crew_update_card(state)
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
