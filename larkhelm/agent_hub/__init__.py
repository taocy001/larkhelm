"""larkhelm · agent_hub — Phase 5 intent / agent layer.

Importing this package triggers built-in agent registration (via
``builtin/__init__.py``). Plugin loading is performed by ``config._init_runtime``
*after* this module finishes importing so plugins can rely on the registry
already containing the built-ins.
"""
from __future__ import annotations

from larkhelm.agent_hub.agent_audit import write_audit
from larkhelm.agent_hub.agent_base import AGENT_REGISTRY, AgentExecutor, AgentRegistry
from larkhelm.agent_hub.agent_dispatcher import AgentDispatcher
from larkhelm.agent_hub.intent_feedback import record_feedback
from larkhelm.agent_hub.intent_router import resolve_intent
from larkhelm.agent_hub.intent_types import (
    AgentContext, AgentDispatch, AgentResult, IntentResult, TaskProfile,
)
from larkhelm.agent_hub.model_selector import resolve_backend_for_task
from larkhelm.agent_hub.skill_registry import SKILL_REGISTRY
from larkhelm.agent_hub.skill_runner import SkillExecutor, register_injector
from larkhelm.agent_hub.skill_types import KeywordRuleSpec, SkillDef

# Side-effect: builtin agents and skills register themselves.
from larkhelm.agent_hub import builtin  # noqa: F401


__all__ = [
    # Intent / agent types
    "IntentResult", "TaskProfile", "AgentDispatch",
    "AgentContext", "AgentResult",
    # Agent layer
    "AgentExecutor", "AgentRegistry", "AGENT_REGISTRY",
    "resolve_intent", "AgentDispatcher",
    "resolve_backend_for_task",
    # Skill layer (new)
    "SkillDef", "KeywordRuleSpec",
    "SkillExecutor", "SKILL_REGISTRY", "register_injector",
    # Audit / feedback
    "record_feedback", "write_audit",
]
