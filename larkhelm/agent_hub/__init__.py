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

# Side-effect: builtin agents register themselves with AGENT_REGISTRY.
from larkhelm.agent_hub import builtin  # noqa: F401


__all__ = [
    "IntentResult", "TaskProfile", "AgentDispatch",
    "AgentContext", "AgentResult",
    "AgentExecutor", "AgentRegistry", "AGENT_REGISTRY",
    "resolve_intent", "AgentDispatcher",
    "resolve_backend_for_task",
    "record_feedback", "write_audit",
]
