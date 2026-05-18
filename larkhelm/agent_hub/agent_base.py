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
    cancellation paths (``_do_query``, crew runners, plan runners) wake up.
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
        # REQ-03 (D6): precompute description embedding once at register time
        # so the L2-embed router path can be O(1). Failures here are silent —
        # an embedding backend hiccup must not bounce agent registration.
        self._maybe_attach_embedding(executor)

    @staticmethod
    def _maybe_attach_embedding(executor: AgentExecutor) -> None:
        desc = getattr(executor, "description", "") or ""
        if not desc.strip():
            return
        try:
            from larkhelm.memory_embedding import get_embedding_backend
            backend = get_embedding_backend()
        except Exception:
            return
        if backend is None:
            return
        try:
            vectors = backend.embed([desc])
            executor._description_embedding = vectors[0]
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(
                f"[AgentRegistry] embed description for {executor.agent_type!r} failed: {e}"
            )

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
