"""larkhelm · agent_hub · AgentExecutor ABC + AgentRegistry singleton."""
from __future__ import annotations

import abc
import threading

from larkhelm.agent_hub.intent_types import (
    AgentContext, AgentResult, IntentResult,
)


class AgentExecutor(abc.ABC):
    """Abstract base for built-in and plugin agents.

    Subclasses must override ``agent_type`` and implement :meth:`execute`.
    Default :meth:`abort` triggers the chat-level cancel event so existing
    cancellation paths (``_do_query``) wake up.
    """

    agent_type: str = ""
    description: str = ""
    required_capabilities: tuple = ()

    def can_handle(self, intent: IntentResult) -> float:
        return 1.0 if intent.agent_type == self.agent_type else 0.0

    @abc.abstractmethod
    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        ...

    def abort(self, chat_id: str) -> bool:
        try:
            from larkhelm.concurrency import _trigger_cancel
            _trigger_cancel(chat_id)
            return True
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[AgentExecutor.abort] {self.agent_type}: {e}")
            return False


class AgentRegistry:
    """In-memory registry mapping agent_type → AgentExecutor instance."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentExecutor] = {}
        self._lock = threading.RLock()

    def register(self, executor: AgentExecutor) -> None:
        if not isinstance(executor, AgentExecutor):
            raise TypeError("register() requires an AgentExecutor instance")
        if not executor.agent_type:
            raise ValueError("AgentExecutor must declare a non-empty agent_type")
        with self._lock:
            self._agents[executor.agent_type] = executor

    def unregister(self, agent_type: str) -> bool:
        with self._lock:
            return self._agents.pop(agent_type, None) is not None

    def get(self, agent_type: str) -> AgentExecutor | None:
        with self._lock:
            return self._agents.get(agent_type)

    def list_types(self) -> list[str]:
        with self._lock:
            return sorted(self._agents.keys())

    def match(self, intent: IntentResult) -> AgentExecutor | None:
        with self._lock:
            agents = list(self._agents.values())
        best: AgentExecutor | None = None
        best_score = 0.0
        for agent in agents:
            try:
                score = float(agent.can_handle(intent))
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best = agent
        return best


AGENT_REGISTRY: AgentRegistry = AgentRegistry()


__all__ = ["AgentExecutor", "AgentRegistry", "AGENT_REGISTRY"]
