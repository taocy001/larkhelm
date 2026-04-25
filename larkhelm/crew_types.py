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


# ═══════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════

class AgentStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class CrewPhase(Enum):
    """Valid values for CrewState.phase, defined centrally to prevent typos."""
    PLANNING     = "planning"    # Planning in progress (Manager generating plan)
    PLANNED      = "planned"     # Plan generated, awaiting execution
    RUNNING      = "running"     # Executing
    BREAKPOINT   = "breakpoint"  # Paused at breakpoint, awaiting human confirmation
    SYNTHESIZING = "synthesizing"# Synthesis phase
    DONE         = "done"        # Completed
    CANCELLED    = "cancelled"   # Cancelled
    FAILED       = "failed"      # Failed
    PAUSED       = "paused"      # Paused (user-initiated pause)


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
    # phase_commits: {agent_id → git_commit_hash} (populated in auto_commit mode)
