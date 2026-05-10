"""larkhelm · agent_hub data types

Pure data classes for the intent / agent system. No external dependencies
beyond stdlib (`dataclasses`, `threading`, `typing`). Every other module in
``agent_hub`` imports from this file, so it must stay free of side effects.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Any, Literal


AgentType = Literal["dev", "crew", "plan", "chat", "doc", "search"]
Complexity = Literal["simple", "medium", "complex"]
LatencyTier = Literal["instant", "fast", "medium", "slow"]
SessionPolicy = Literal["inherit", "isolated", "ephemeral"]


@dataclasses.dataclass(frozen=True)
class IntentResult:
    agent_type:          str
    sub_intent:          str = ""
    complexity:          str = "medium"   # Complexity literal
    required_tags:       tuple = ()       # tuple[str, ...]
    confidence:          float = 0.0
    reasoning:           str = ""
    is_explicit_command: bool = False
    layer:               str = "fallback"  # "L1" | "L2" | "fallback" | "override"
    raw_text:            str = ""


@dataclasses.dataclass(frozen=True)
class TaskProfile:
    complexity:            str = "medium"          # Complexity literal
    required_capabilities: dict = dataclasses.field(default_factory=dict)   # dict[str, float]
    latency_pref:          str = "medium"          # LatencyTier literal
    require_tools:         bool = False
    require_vision:        bool = False
    # USD per call. Compared against an estimate computed by
    # backend_registry.rank_for_task using COST_CEILING_ASSUMED_IN/OUT_TOKENS.
    cost_ceiling:          float | None = None


@dataclasses.dataclass(frozen=True)
class AgentDispatch:
    agent_type: str
    backend_id: str = ""
    mode:       str = ""
    task:       str = ""


@dataclasses.dataclass
class AgentContext:
    chat_id:          str
    user_msg_id:      str | None
    text:             str
    images:           list | None
    parent_id:        str | None
    cancel_ev:        threading.Event
    cwd:              str
    session_policy:   str = "inherit"           # SessionPolicy literal
    force_backend_id: str | None = None
    extra:            dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class AgentResult:
    success:      bool
    output:       str = ""
    backend_id:   str = ""
    duration_sec: float = 0.0
    cost_usd:     float = 0.0
    error:        str = ""


__all__ = [
    "AgentType", "Complexity", "LatencyTier", "SessionPolicy",
    "IntentResult", "TaskProfile", "AgentDispatch",
    "AgentContext", "AgentResult",
]
