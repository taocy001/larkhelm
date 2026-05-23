"""
larkhelm · Crew multi-agent collaboration module (sub-package)

/crew <requirement>   — Manager dynamically plans a DAG, general tasks
/dev  <requirement>   — Fixed software engineering pipeline (PM → Architect → Engineer → QA → Reviewer)

This file provides backward-compatible re-exports so `from larkhelm.crew import X` continues to work.
"""
from __future__ import annotations

# ── Global state variables ────────────────────────────────────
from larkhelm.crew._state import (
    _active_crew,
    _active_crew_states,
    _active_crew_lock,
    signal_breakpoint,
    get_crew_card_context,
    get_recent_crew_context,
    consume_recent_crew_context,
    clear_recent_crew_context,
    _register_crew_card,
)

# ── Command functions ─────────────────────────────────────────
from larkhelm.crew._commands import (
    cmd_crew,
    cmd_dev,
    _cmd_crew_status,
    _run_generic_crew,
    _run_generic_crew_inner,
    _run_dev_crew,
    _run_dev_crew_inner,
    immediate_cancel_crew,
    cancel_all_crews,
    pause_crew,
    wait_crews_done,
    _register_crew_thread,
    _unregister_crew_thread,
)

# ── Checkpoints ───────────────────────────────────────────────
from larkhelm.crew._checkpoint import (
    resume_interrupted_crews,
)

# ── Scheduler ─────────────────────────────────────────────────
from larkhelm.crew._scheduler import (
    _detect_cycle,
    _topo_waves,
    _topo_waves_subset,
    _get_failed_dep,
    _resolve_prompt,
    _workspace_dir,
)

# ── Runner ────────────────────────────────────────────────────
from larkhelm.crew._runner import (
    _run_agent,
    _wait_for_breakpoint,
    _detect_fail_marker,
    _sync_output_file,
    _execute,
    _execute_from,
    _synthesize,
    _run_crew,
)

# ── Dev pipeline ──────────────────────────────────────────────
from larkhelm.crew._pipeline import (
    _make_dev_pipeline,
)

# ── Checkpoint GC (P3 REQ-09) ─────────────────────────────────
from larkhelm.crew._checkpoint_gc import CheckpointGC


def _register_checkpoint_gc() -> None:
    """Build a :class:`CheckpointGC` for ``DATA_DIR/.crew_workspace`` and
    register it with the :class:`MemoryGCRunner` singleton (D5).

    Safe to call before _init_runtime — degrades to a no-op if DATA_DIR
    isn't set yet or the memory_gc module isn't importable. Idempotent:
    repeat calls overwrite the previous attachment.
    """
    try:
        import larkhelm.config as _cfg
        from larkhelm.memory_gc import attach_checkpoint_gc
        from pathlib import Path
        data_dir = getattr(_cfg, "DATA_DIR", None)
        if data_dir is None:
            return
        workspaces_root = Path(data_dir) / ".crew_workspace"
        ttl_days = float(getattr(_cfg, "CREW_CHECKPOINT_TTL_DAYS", 7.0) or 7.0)
        gc = CheckpointGC(workspaces_root, ttl_days=ttl_days)
        attach_checkpoint_gc(gc)
    except Exception as e:
        try:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[CkptGc] register failed: {e}")
        except Exception:
            pass


# Auto-register on import. The actual GC daemon may not be running yet
# (test mode / boot ordering); the attachment is durable across ticks.
_register_checkpoint_gc()

__all__ = [
    # Global state
    "_active_crew",
    "_active_crew_states",
    "_active_crew_lock",
    "signal_breakpoint",
    "get_crew_card_context",
    "get_recent_crew_context",
    "consume_recent_crew_context",
    "clear_recent_crew_context",
    "_register_crew_card",
    # Commands
    "cmd_crew",
    "cmd_dev",
    "_cmd_crew_status",
    "_run_generic_crew",
    "_run_generic_crew_inner",
    "_run_dev_crew",
    "_run_dev_crew_inner",
    "immediate_cancel_crew",
    "cancel_all_crews",
    "pause_crew",
    "wait_crews_done",
    "_register_crew_thread",
    "_unregister_crew_thread",
    # Checkpoints
    "resume_interrupted_crews",
    # Scheduler
    "_detect_cycle",
    "_topo_waves",
    "_topo_waves_subset",
    "_get_failed_dep",
    "_resolve_prompt",
    "_workspace_dir",
    # Runner
    "_run_agent",
    "_wait_for_breakpoint",
    "_detect_fail_marker",
    "_sync_output_file",
    "_execute",
    "_execute_from",
    "_synthesize",
    "_run_crew",
    # Dev pipeline
    "_make_dev_pipeline",
    # Checkpoint GC (P3 REQ-09)
    "CheckpointGC",
    "_register_checkpoint_gc",
]
