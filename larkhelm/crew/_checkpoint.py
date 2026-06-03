"""
larkhelm · Crew checkpoint (resume from breakpoint)
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

from larkhelm.log import _debug_log
from larkhelm.ai_runner import QueryCancelledError
from larkhelm.crew_types import AgentSpec, AgentState, AgentStatus, CrewPlan, CrewState, HardFailError


_CHECKPOINT_FILE = ".crew_workspace/crew_checkpoint.json"


def _save_checkpoint(state: CrewState, completed_wave_ids: list[str], phase: str = ""):
    """Persist checkpoint after each wave completes, recording finished agent results and current progress."""
    from larkhelm.chat_state import _get_cwd
    cwd = _get_cwd(state.chat_id)
    path = Path(cwd) / _CHECKPOINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize AgentSpec (drop non-serializable dataclass fields)
    def _ser_spec(spec: AgentSpec) -> dict:
        return dataclasses.asdict(spec)

    # Serialize completed/failed/cancelled/skipped agent states. SKIPPED is
    # included so a resume-after-restart correctly picks up the
    # ``TASK_ALREADY_COMPLETE`` short-circuit — without it, a checkpoint
    # saved between PM's marker-emit and the synthesis would reload the
    # downstream agents as PENDING and run them anyway, undoing the early-
    # exit. (Defensive: today the short-circuit clears the checkpoint
    # before this race can fire, but future scheduler changes might
    # insert a save in between.)
    agents_snap: dict = {}
    phase_outputs: dict = {}
    _exit_status_map = {"completed": "PASS", "failed": "FAIL", "skipped": "SKIP", "cancelled": "SKIP"}
    with state.lock:
        for ag_id, ag in state.agents.items():
            if ag.status in (AgentStatus.DONE, AgentStatus.FAILED,
                             AgentStatus.CANCELLED, AgentStatus.SKIPPED):
                agents_snap[ag_id] = {
                    "status":      ag.status.value,
                    "result":      ag.result,
                    "error":       ag.error,
                    "retry_count": ag.retry_count,
                    "round_label": ag.round_label,
                }
                phase_outputs[ag_id] = {
                    "summary":     ag.result[:400],
                    "output_file": ag.spec.output_file,
                    "exit_status": _exit_status_map.get(ag.status.value, "UNKNOWN"),
                }

    checkpoint = {
        "schema_version": 2,
        "crew_id":    state.crew_id,
        "chat_id":    state.chat_id,
        "card_mid":   state.card_mid,
        "start_time": state.start_time,
        "phase":      phase if phase else state.phase,
        "kind":       state.kind,
        "git_head_before": state.git_head_before,
        "phase_commits":   state.phase_commits,
        "plan": {
            "title":              state.plan.title,
            "synthesis_prompt":   state.plan.synthesis_prompt,
            "max_qa_retry_rounds": state.plan.max_qa_retry_rounds,
            "agents":             [_ser_spec(s) for s in state.plan.agents],
        },
        "agents":             agents_snap,
        "completed_wave_ids": completed_wave_ids,
        "phase_outputs":      phase_outputs,
    }
    try:
        import os as _os
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        _os.replace(tmp, path)
        _debug_log(f"[Checkpoint] saved: {path}")
    except Exception as e:
        _debug_log(f"[Checkpoint] save failed: {e}")


def _clear_checkpoint(chat_id: str):
    """Clear the checkpoint after a task completes or is fully cancelled."""
    from larkhelm.chat_state import _get_cwd
    cwd  = _get_cwd(chat_id)
    path = Path(cwd) / _CHECKPOINT_FILE
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        _debug_log(f"[Checkpoint] delete failed: {e}")


def _migrate_v1_to_v2(data: dict) -> dict:
    """Upgrade checkpoint from schema_version 1 to 2 in-place.

    Mutations:
    - data["schema_version"] = 2
    - each agent spec in data["plan"]["agents"]: add fallback_agent_id="" if absent
    - each agent snapshot in data["agents"]: map status "done" to "completed"

    Returns the mutated dict. Never raises (fail-soft).
    """
    try:
        data.pop("version", None)
        data["schema_version"] = 2
        for spec in data.get("plan", {}).get("agents", []):
            spec.setdefault("fallback_agent_id", "")
        data.setdefault("phase_outputs", {})
        for snap in data.get("agents", {}).values():
            if snap.get("status") == "done":
                snap["status"] = "completed"
    except Exception as e:
        _debug_log(f"[Checkpoint] migrate v1→v2 failed: {e}")
    return data


def _load_checkpoint(chat_id: str) -> "dict":
    """Read the checkpoint file and return a dict, or None if unavailable."""
    from larkhelm.chat_state import _get_cwd
    cwd  = _get_cwd(chat_id)
    path = Path(cwd) / _CHECKPOINT_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        schema_v = data.get("schema_version")
        version_v = data.get("version")
        if schema_v == 2:
            return data
        if version_v == 1:
            return _migrate_v1_to_v2(data)
        # Unknown format
        return None
    except Exception as e:
        _debug_log(f"[Checkpoint] read failed: {e}")
        return None


def _rebuild_state_from_checkpoint(data: dict) -> "CrewState":
    """Rebuild CrewState from a checkpoint dict, restoring completed agent results."""
    try:
        plan_data = data["plan"]
        specs = []
        legacy_specs_seen = 0
        for s in plan_data["agents"]:
            # Phase C: pre-task_profile checkpoints lack the new field.
            # AgentSpec(**s) handles it via the dataclass default ("") so
            # the resolver auto-falls back to model-string dispatch — but
            # we log once so operators investigating slow recovery can
            # tell the schema is older.
            if "task_profile" not in s:
                legacy_specs_seen += 1
            s.setdefault("fallback_agent_id", "")
            specs.append(AgentSpec(**s))
        if legacy_specs_seen:
            _debug_log(
                f"[Checkpoint] detected legacy checkpoint without task_profile "
                f"({legacy_specs_seen} agent(s)), defaulting to ''"
            )

        plan = CrewPlan(
            title=plan_data["title"],
            agents=specs,
            synthesis_prompt=plan_data.get("synthesis_prompt", ""),
            max_qa_retry_rounds=plan_data.get("max_qa_retry_rounds", 2),
        )

        agents_snap = data.get("agents", {})
        agents: dict[str, AgentState] = {}
        for spec in specs:
            snap = agents_snap.get(spec.id)
            if snap:
                agents[spec.id] = AgentState(
                    spec=spec,
                    status=AgentStatus(snap["status"]),
                    result=snap.get("result", ""),
                    error=snap.get("error", ""),
                    retry_count=snap.get("retry_count", 0),
                    round_label=snap.get("round_label", ""),
                )
            else:
                agents[spec.id] = AgentState(spec=spec)

        from larkhelm.concurrency import _get_cancel_event
        state = CrewState(
            crew_id=data["crew_id"],
            chat_id=data["chat_id"],
            plan=plan,
            agents=agents,
            card_mid=data.get("card_mid"),
            start_time=data.get("start_time", time.time()),
            cancel_ev=_get_cancel_event(data["chat_id"]),
            phase=data.get("phase", "running"),
            kind=data.get("kind", "crew"),
            is_resuming=True,
            git_head_before=data.get("git_head_before", ""),
            phase_commits=data.get("phase_commits", {}),
            phase_outputs=data.get("phase_outputs", {}),
        )
        return state
    except Exception as e:
        _debug_log(f"[Checkpoint] failed to rebuild state: {e}")
        return None


def resume_interrupted_crews():
    """Called at service startup: scan all chat checkpoints and resume incomplete crews."""
    import glob
    import larkhelm.config as _cfg
    from larkhelm.lark_client import _patch_card_raw, send_card
    from larkhelm.crew_card import _crew_update_card
    from larkhelm.crew._state import _active_crew, _active_crew_states, _active_crew_lock
    from larkhelm.crew._commands import _register_crew_thread, _unregister_crew_thread
    from larkhelm.crew._runner import _execute_from, _synthesize
    from larkhelm.crew._checkpoint import _clear_checkpoint

    # Scan all directories that might contain checkpoints
    data_dir = Path(_cfg.DATA_DIR)
    pattern  = str(data_dir / "**" / _CHECKPOINT_FILE)
    found    = glob.glob(pattern, recursive=True)

    # Also scan default_cwd and its subdirectories
    if _cfg.DEFAULT_CWD:
        pattern2 = str(Path(_cfg.DEFAULT_CWD) / "**" / _CHECKPOINT_FILE)
        found   += glob.glob(pattern2, recursive=True)

    found = list(set(found))  # deduplicate

    if not found:
        return

    _debug_log(f"[Checkpoint] found {len(found)} incomplete crew(s), preparing to resume...")

    for cp_path in found:
        try:
            data = json.loads(Path(cp_path).read_text(encoding="utf-8"))
        except Exception as e:
            _debug_log(f"[Checkpoint] failed to read {cp_path}: {e}")
            continue
        schema_v = data.get("schema_version")
        version_v = data.get("version")
        if schema_v != 2 and version_v != 1:
            continue
        if version_v == 1 and schema_v is None:
            data = _migrate_v1_to_v2(data)
        phase = data.get("phase", "")
        # P2-3a (W4/W6): "timeout" is the new terminal state for breakpoint
        # auto-cancel; treated as terminal here so existing checkpoints with
        # the legacy "cancelled" value still resume identically.
        if phase in ("done", "cancelled", "failed", "user_cancelled", "timeout"):
            # Already finished; clean up
            try:
                Path(cp_path).unlink()
            except Exception as e:
                _debug_log(f"[Checkpoint] cleanup failed: {e}")
            continue

        chat_id = data.get("chat_id", "")
        if not chat_id:
            continue

        was_paused = (phase == "paused")
        _debug_log(f"[Checkpoint] resuming crew={data.get('crew_id','')[:8]} chat={chat_id} phase={phase}")

        def _resume(d=data, cp=cp_path, paused=was_paused):
            state = _rebuild_state_from_checkpoint(d)
            if not state:
                return
            completed_ids = set(d.get("completed_wave_ids", []))
            resume_title = "▶️ Crew 续跑中（从暂停恢复）" if paused else "🔄 Crew 续跑中（从断点恢复）"
            # Update card to indicate resume in progress
            try:
                from larkhelm.card_builder import _make_card
                _patch_card_raw(state.card_mid, _make_card(
                    resume_title,
                    f"服务重启后从断点继续执行。\n\n**任务：** {state.plan.title}",
                    color="blue",
                ))
            except Exception as e:
                _debug_log(f"[Checkpoint] resume card failed: {e}")

            with _active_crew_lock:
                _active_crew[state.chat_id]        = state.crew_id
                _active_crew_states[state.chat_id] = state

            crew_id = state.crew_id
            _register_crew_thread(crew_id, threading.current_thread())
            try:
                total_timeout = _cfg.HARD_TIMEOUT
                # Rebuild wave_queue, skipping already-completed agents
                _execute_from(state, total_timeout, completed_ids)
                # Synthesis
                final = _synthesize(state)
                with state.lock:
                    state.phase        = "done"
                    state.final_output = final
                _crew_update_card(state)
            except HardFailError as e:
                with state.lock:
                    state.phase = "failed"
                _crew_update_card(state)
                send_card(state.chat_id, "❌ Crew 硬失败", str(e), color="red")
            except QueryCancelledError:
                with state.lock:
                    state.phase = "cancelled"
                _crew_update_card(state)
            except Exception as e:
                _debug_log(f"[Checkpoint] resume error: {e}")
                with state.lock:
                    state.phase = "failed"
                _crew_update_card(state)
            finally:
                _clear_checkpoint(state.chat_id)
                _unregister_crew_thread(crew_id)
                with _active_crew_lock:
                    _active_crew.pop(state.chat_id, None)
                    _active_crew_states.pop(state.chat_id, None)

        # REQ-10: write placeholder BEFORE spawning the thread so concurrent
        # /crew commands see this slot as occupied from the start.
        crew_id_val = data.get("crew_id", "")
        with _active_crew_lock:
            _active_crew[chat_id] = crew_id_val

        threading.Thread(target=_resume, daemon=True,
                         name=f"crew-resume-{crew_id_val[:6]}").start()
