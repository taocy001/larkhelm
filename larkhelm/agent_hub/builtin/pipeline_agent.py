"""larkhelm · agent_hub.builtin.pipeline_agent — generic DevPipelineAgent.

Each registered :class:`~larkhelm.agent_hub.pipeline_types.PipelineDef` gets a
corresponding ``DevPipelineAgent`` instance in ``AGENT_REGISTRY`` so that the
existing intent-dispatch path can route to it without any changes.

The agent is intentionally thin: it delegates plan construction to
``PIPELINE_REGISTRY.build_plan()`` and execution to ``run_pipeline()`` in
``larkhelm.crew._commands``.  This keeps the execution infrastructure DRY —
the same crew runner, card updates, milestone logging, and cancellation paths
are reused for all pipeline variants.
"""
from __future__ import annotations

import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class DevPipelineAgent(AgentExecutor):
    """Generic agent executor that runs a named pipeline from PIPELINE_REGISTRY.

    Instantiated by :class:`~larkhelm.agent_hub.pipeline_registry.PipelineRegistry`
    once per registered :class:`~larkhelm.agent_hub.pipeline_types.PipelineDef`.
    Do not instantiate directly.
    """

    required_capabilities = ("code", "tools")

    def __init__(self, pipeline_id: str, description: str) -> None:
        self.pipeline_id = pipeline_id
        self.agent_type: str = pipeline_id        # type: ignore[assignment]
        self.description: str = description       # type: ignore[assignment]

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.agent_hub.pipeline_registry import PIPELINE_REGISTRY
        from larkhelm.crew._commands import run_pipeline

        start = time.monotonic()
        plan = PIPELINE_REGISTRY.build_plan(self.pipeline_id, ctx.text, ctx.cwd)
        if plan is None:
            return AgentResult(
                success=False,
                duration_sec=time.monotonic() - start,
                error=f"pipeline {self.pipeline_id!r} not found in registry",
            )
        try:
            run_pipeline(
                ctx.chat_id, plan, ctx.text, ctx.user_msg_id,
                sender_open_id=ctx.extra.get("sender_open_id", ""),
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            return AgentResult(
                success=False,
                duration_sec=time.monotonic() - start,
                error=str(e),
            )
