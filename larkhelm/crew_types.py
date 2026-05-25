"""
larkhelm · Crew data type definitions

Contains dataclasses and constants for the Crew/Dev multi-agent system, used by crew.py and crew_card.py.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from enum import Enum


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

CREW_MAX_AGENTS     = 6      # Maximum number of agents planned by the Manager
CREW_CARD_INTERVAL  = 10.0   # Heartbeat push interval (slower than normal 5s queries to avoid QPS throttling)
CREW_RESULT_PREVIEW = 500    # Number of characters of an agent result shown in the card preview


class HardFailError(Exception):
    """QA/critical agent final failure; does not proceed to synthesis phase."""


class NoBackendAvailableError(Exception):
    """Raised by ``_backend_resolver.resolve_backend`` when the agent's
    ``task_profile`` rank query returns nothing AND the orchestrator
    fallback is also unavailable.

    Carries the requested ``task_profile`` and a short ``reason`` so the
    failure card can render a targeted hint (e.g. "no QA-capable backend;
    check /status for backend health").
    """

    def __init__(self, task_profile: str, reason: str):
        self.task_profile = task_profile
        self.reason       = reason
        super().__init__(f"no backend for task_profile={task_profile!r}: {reason}")


# ═══════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════

class AgentStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"
    # SKIPPED: agent was intentionally not run. Used by the
    # ``TASK_ALREADY_COMPLETE`` short-circuit when PM decides the codebase
    # already satisfies the user request, and by the partial-delivery rule
    # in the scheduler when an upstream FAILED agent already wrote its
    # declared output_file. Distinguishes "we chose not to run this" from
    # CANCELLED (user-initiated) and FAILED (ran but errored).
    SKIPPED   = "skipped"


class CrewPhase(Enum):
    """Valid values for CrewState.phase, defined centrally to prevent typos."""
    PLANNING     = "planning"    # Planning in progress (Manager generating plan)
    PLANNED      = "planned"     # Plan generated, awaiting execution
    RUNNING      = "running"     # Executing
    BREAKPOINT   = "breakpoint"  # Paused at breakpoint, awaiting human confirmation
    SYNTHESIZING = "synthesizing"# Synthesis phase
    DONE         = "done"        # Completed
    CANCELLED    = "cancelled"   # Cancelled (user-initiated)
    FAILED       = "failed"      # Failed
    PAUSED       = "paused"      # Paused (user-initiated pause)
    # P2-3a (W4/W6): breakpoint timeout now writes "timeout" instead of
    # "cancelled" so the cancel-by-user vs auto-timeout cases stay
    # distinguishable. Readers that haven't been updated will see the new
    # string verbatim (str compare) — there is no enum→enum migration; only
    # the value at the emission site changed.
    TIMEOUT      = "timeout"     # Auto-cancelled because a wait (e.g. breakpoint) expired


@dataclasses.dataclass
class AgentSpec:
    id:           str
    role:         str
    model:        str           # "claude" | "gemini" | "kimi" | "hermes_race" | "hermes_split" | "hermes_review"
    system:       str           # system prompt (used by /dev pipeline)
    prompt:       str           # task prompt, supports {agent_X_result} placeholders
    depends_on:   list[str]
    timeout:      int           # seconds
    # Phase 1: exit codes and retry
    exit_marker:  str           = ""   # success marker: output's last line containing this string is treated as success
    fail_marker:  str           = ""   # failure marker: output's last line containing this string triggers retry
    retry_target: list[str]     = dataclasses.field(default_factory=list)  # list of agent ids to reset on retry
    max_retries:  int           = 0    # maximum retry count (0 = no retry)
    is_gatekeeper: bool         = False  # True = gatekeeper; failure blocks downstream execution
    breakpoint:   bool          = False  # True = pause after completion and wait for human confirmation
    trigger_only:         bool  = False  # True = only triggered by retry; automatically skipped in normal waves
    hard_fail_on_exhaust: bool  = False  # True = raise HardFailError when retries are exhausted; skip synthesis
    retry_system:         str   = ""     # replacement system prompt on retry (engineer fixer mode)
    retry_prompt:         str   = ""     # replacement prompt on retry
    output_file:          str   = ""     # agent's primary output file (relative to cwd/.crew_workspace/)
    # ── Phase C ────────────────────────────────────────────────────────────
    # Resolver hint for ``crew/_backend_resolver.resolve_backend``:
    #   "" (default) → fall back to the legacy ``model``-string dispatch path
    #   one of {"planner","engineer","qa","reviewer","chat"} → query
    #   ``BACKEND_REGISTRY.rank_for_task(TASK_PROFILES[task_profile])``
    # New ``_pipeline.py`` agents only set ``task_profile``; legacy
    # checkpoints / third-party plugins keep working because the field
    # defaults to "" — see design.md §3.5.
    task_profile:         str   = ""


@dataclasses.dataclass
class AgentState:
    spec:        AgentSpec
    status:      AgentStatus  = AgentStatus.PENDING
    result:      str          = ""
    error:       str          = ""
    start_time:  float | None = None
    end_time:    float | None = None
    # Phase 1: retry support
    retry_count: int          = 0     # number of retries so far
    round_label: str          = ""    # display label, e.g. "Round 2"
    needs_retry: bool         = False  # used by _execute to determine whether to trigger a retry
    feedback:    str          = ""    # feedback injected when a downstream agent fails; prepended to prompt on next run
    feishu_doc_url: str       = ""    # Feishu document URL after output_file sync (empty = local only)
    tokens:      dict[str, int | float] = dataclasses.field(default_factory=dict)  # token stats (in-memory, not serialized)
    # The backend id (e.g. ``claude``, ``kimi``, ``hermes_race``) chosen by
    # ``resolve_backend(spec)`` at runtime — written once when the agent
    # starts. Empty string means "not yet resolved" (PENDING / SKIPPED).
    # Replaces the now-empty ``spec.model`` for crew-card display since
    # the Phase-C ``task_profile`` migration left ``spec.model=""`` by
    # default and the card was rendering "[] PM" instead of "[claude] PM".
    actual_backend_id: str    = ""
    # F4 (2026-05-25): backend ids excluded from ``resolve_backend`` for
    # THIS agent only — populated by ``_run_agent_wrapper`` after a
    # validate-failure attempt so the retry picks a different backend
    # instead of re-running the same tool-incapable model (the previous
    # same-backend retry policy was token waste because backend-intrinsic
    # failures are not transient). Scope: WITHIN a single agent run
    # (attempt 0 → attempt 1). Cleared in ``_runner._execute``'s
    # retry-target reset block so a higher-level retry trigger
    # (architect self-check retry / qa→fixer retry / reviewer retry)
    # restarts backend selection from a clean slate — a backend that
    # was transiently unhappy on round 1 may have recovered by round 2.
    excluded_backend_ids: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CrewPlan:
    title:             str
    agents:            list[AgentSpec]
    synthesis_prompt:  str = ""   # synthesis phase prompt template


@dataclasses.dataclass
class CrewState:
    crew_id:    str
    chat_id:    str
    plan:       CrewPlan
    agents:     dict[str, AgentState]   # agent_id → AgentState
    card_mid:   str | None              = None
    start_time: float                   = dataclasses.field(default_factory=time.time)
    cancel_ev:  threading.Event         = dataclasses.field(default_factory=threading.Event)
    lock:               threading.Lock  = dataclasses.field(default_factory=threading.Lock)
    phase:              str             = "planning"
    final_output:       str             = ""
    breakpoint_agent_id: str            = ""  # agent id at which execution is currently paused
    kind:               str             = "crew"  # "crew" | "dev"
    is_resuming:        bool            = False   # True when resuming from a checkpoint
    git_head_before:    str             = ""      # git HEAD hash at task start
    phase_commits:      dict[str, str]  = dataclasses.field(default_factory=dict)  # agent_id → git_commit_hash
    trigger_msg_id:     str | None      = None    # user message id that triggered the task; used to reply with completion notification
    feishu_folder_token: str            = ""      # per-project Feishu folder token (Drive folder or Wiki node)
    feishu_folder_url:   str            = ""      # human-readable URL for the project folder
    output_file_urls:    dict           = dataclasses.field(default_factory=dict)  # output_file → feishu_doc_url
    sender_open_id:      str            = ""      # open_id of the user who triggered this crew task
    # phase_commits: {agent_id → git_commit_hash} (populated in auto_commit mode)
