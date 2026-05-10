"""larkhelm · agent_hub.builtin — register the 5 built-in agents.

Importing this sub-package triggers :func:`register_all`. Calling it again
is idempotent (re-registering the same ``agent_type`` simply replaces the
existing executor instance, which is fine for tests that swap mocks in).
"""
from __future__ import annotations

from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
from larkhelm.agent_hub.builtin.chat_agent import ChatAgent
from larkhelm.agent_hub.builtin.crew_agent import CrewAgent
from larkhelm.agent_hub.builtin.dev_agent import DevAgent
from larkhelm.agent_hub.builtin.doc_agent import DocAgent
from larkhelm.agent_hub.builtin.plan_agent import PlanAgent


_BUILTIN_CLASSES = (ChatAgent, DevAgent, CrewAgent, PlanAgent, DocAgent)


def register_all() -> None:
    for cls in _BUILTIN_CLASSES:
        AGENT_REGISTRY.register(cls())


register_all()


__all__ = [
    "ChatAgent", "DevAgent", "CrewAgent", "PlanAgent", "DocAgent",
    "register_all",
]
