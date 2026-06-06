"""larkhelm · agent_hub — skill/pipeline registry and plugin loader."""
from __future__ import annotations

from larkhelm.agent_hub.agent_audit import write_audit
from larkhelm.agent_hub.agent_base import AGENT_REGISTRY, AgentExecutor, AgentRegistry
from larkhelm.agent_hub.intent_types import (
    AgentContext, AgentDispatch, AgentResult, IntentResult, TaskProfile,
)
from larkhelm.agent_hub.skill_registry import SKILL_REGISTRY
from larkhelm.agent_hub.skill_runner import SkillExecutor, register_injector
from larkhelm.agent_hub.skill_types import KeywordRuleSpec, SkillDef


__all__ = [
    # Intent / agent types
    "IntentResult", "TaskProfile", "AgentDispatch",
    "AgentContext", "AgentResult",
    # Agent layer
    "AgentExecutor", "AgentRegistry", "AGENT_REGISTRY",
    # Skill layer
    "SkillDef", "KeywordRuleSpec",
    "SkillExecutor", "SKILL_REGISTRY", "register_injector",
    # Audit
    "write_audit",
]
