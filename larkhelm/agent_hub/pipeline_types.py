"""larkhelm · agent_hub · PipelineDef — data-driven pipeline variant definitions.

A **Pipeline** is an ordered DAG of AgentSpec stages executed by the crew runner.
Unlike a SkillDef (single _do_query call), a pipeline spawns multiple AI subprocesses.

``stages`` is a list of dicts whose keys map 1:1 to AgentSpec fields.  Two template
variables are substituted at build time inside ``prompt`` and ``system``:
    {requirement}  — the user's task description
    {cwd}          — the chat's current working directory

``timeout_factor`` in a stage dict is multiplied by RESPONSE_TIMEOUT to derive the
stage timeout in seconds.  A plain ``timeout`` (int seconds) takes precedence when
present (useful for user-created JSON pipelines with absolute timeouts).

Lifecycle:
    source="builtin"  — shipped with larkhelm; cannot be deleted at runtime.
    source="user"     — created at runtime; persisted to DATA_DIR/pipelines/<id>.json.
    source="plugin"   — loaded from a third-party Python module.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any


@dataclasses.dataclass
class PipelineDef:
    id: str
    name: str
    description: str
    stages: list[dict] = dataclasses.field(default_factory=list)
    synthesis_prompt: str = ""
    l1_keywords: list[dict] = dataclasses.field(default_factory=list)
    source: str = "builtin"
    enabled: bool = True
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineDef":
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    @classmethod
    def from_json(cls, text: str) -> "PipelineDef":
        return cls.from_dict(json.loads(text))


__all__ = ["PipelineDef"]
