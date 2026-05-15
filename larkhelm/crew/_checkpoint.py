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


def _save_checkpoint(state: CrewState, completed_wave_ids: list[str]):
    """Persist checkpoint after each wave completes, recording finished agent results and current progress."""
    from larkhelm.chat_state import _get_cwd
    cwd = _get_cwd(state.chat_id)
    path = Path(cwd) / _CHECKPOINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize AgentSpec (drop non-serializable dataclass fields)
    def _ser_spec(spec: AgentSpec) -> dict:
        return dataclasses.asdict(spec)

    # Serialize completed/failed agent states
    agents_snap: dict = {}
    with state.lock:
        for ag_id, ag in state.agents.items():
            if ag.status in (AgentStatus.DONE, AgentStatus.FAILED, AgentStatus.CANCELLED):
                agents_snap[ag_id] = {
                    "status":      ag.status.value,
                    "result":      ag.result,
                    "error":       ag.error,
                    "retry_count": ag.retry_count,
                    "round_label": ag.round_label,
                }

    checkpoint = {
        "version":    1,
        "crew_id":    state.crew_id,
        "chat_id":    state.chat_id,
        "card_mid":   state.card_mid,
        "start_time": state.start_time,
        "phase":      state.phase,
        "kind":       state.kind,
        "git_head_before": state.git_head_before,
        "phase_commits":   state.phase_commits,
        "plan": {
            "title":           state.plan.title,
            "synthesis_prompt": state.plan.synthesis_prompt,
            "agents":          [_ser_spec(s) for s in state.plan.agents],
        },
        "agents":           agents_snap,
        "completed_wave_ids": completed_wave_ids,
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


def _load_checkpoint(chat_id: str) -> "dict":
    """Read the checkpoint file and return a dict, or None if unavailable."""
    from larkhelm.chat_state import _get_cwd
    cwd  = _get_cwd(chat_id)
    path = Path(cwd) / _CHECKPOINT_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1:
            return None
        return data
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
        if data.get("version") != 1:
            continue
        phase = data.get("phase", "")
        if phase in ("done", "cancelled", "failed", "user_cancelled"):
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
                total_timeout = _cfg.RESPONSE_TIMEOUT * 12
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

        threading.Thread(target=_resume, daemon=True,
                         name=f"crew-resume-{data.get('crew_id','')[:6]}").start()
