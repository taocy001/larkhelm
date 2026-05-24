"""larkhelm · agent_hub · SkillDef — data-driven Skill definitions.

A **Skill** is a lightweight, data-described executor that follows the pattern::

    user text
      → (optional) context injectors pre-fetch external data
      → system prompt template + augmented text
      → _do_query with preferred backend profile

Skills are **pure data** (serializable to JSON/YAML) and can be created,
modified, and deleted at runtime without restarting the bridge.  Complex,
stateful pipelines (dev, crew, plan, chat) remain as Python AgentExecutors.

Key distinction:
  - Agent  = Python class  — stateful, multi-step, can spawn sub-processes
  - Skill  = Data dict     — stateless, single-call, just prompt + backend hint
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any


@dataclasses.dataclass
class KeywordRuleSpec:
    """JSON/YAML-serialisable form of a L1 keyword rule.

    ``pattern`` is either a plain substring match or a Python regex when the
    value starts with the ``re:`` prefix (e.g. ``re:\\bimplement\\s+a``).
    """

    pattern: str                  # literal string or "re:<python-regex>"
    strength: float = 0.80        # 0.0–1.0 confidence when matched
    note: str = ""

    def compile(self) -> re.Pattern[str] | str:
        """Return a compiled regex or the raw string for fast matching."""
        if self.pattern.startswith("re:"):
            return re.compile(self.pattern[3:], re.IGNORECASE)
        return self.pattern.lower()


@dataclasses.dataclass
class SkillDef:
    """Data definition for a LarkHelm Skill.

    Minimal fields needed to execute a skill:

    * ``system_prompt`` — instruction prefix injected before the user text.
    * ``backend_profile`` — which kind of backend to prefer
      (chat / planner / engineer / reviewer).
    * ``context_injectors`` — ordered list of named pre-execution functions
      that fetch external data (e.g. ``web_search``, ``shell_exec``).

    Routing metadata:

    * ``l1_keywords`` — integrated into the L1 intent router at runtime.
    * ``description`` — used by the L2 embedding/LLM classifier.

    Lifecycle:

    * ``source = "builtin"``  — shipped with larkhelm, cannot be deleted via
      ``/skill delete``; can be disabled.
    * ``source = "user"``     — created by operators at runtime; persisted to
      ``DATA_DIR/skills/<id>.json``.
    * ``source = "plugin"``   — loaded from a third-party Python module via
      ``config.agent_plugins``.
    """

    id: str                                            # unique key, e.g. "translate"
    name: str                                          # display name
    description: str                                   # used by L2 classifier
    system_prompt: str = ""                            # injected before user text
    backend_profile: str = "chat"                      # chat/planner/engineer/reviewer
    context_injectors: list[str] = dataclasses.field(default_factory=list)
    strip_trigger_pattern: str = ""                    # regex: remove trigger words
    l1_keywords: list[KeywordRuleSpec] = dataclasses.field(default_factory=list)
    enabled: bool = True
    version: int = 1
    source: str = "builtin"                            # builtin | user | plugin

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = dataclasses.asdict(self)
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_yaml(self) -> str:
        """Return a YAML-formatted string.  Requires PyYAML; raises ImportError
        if not installed (use :meth:`to_json` as a reliable alternative)."""
        try:
            import yaml  # type: ignore
            return yaml.dump(self.to_dict(), allow_unicode=True,
                             default_flow_style=False, sort_keys=False)
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML export.  "
                "Install with: pip install pyyaml"
            ) from exc

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SkillDef":
        """Deserialise from a plain dict (JSON-loaded or hand-crafted)."""
        d = dict(d)           # shallow copy — don't mutate caller's dict
        raw_kws = d.pop("l1_keywords", [])
        kw_specs: list[KeywordRuleSpec] = []
        for k in raw_kws:
            if isinstance(k, dict):
                kw_specs.append(KeywordRuleSpec(**{
                    f.name: k[f.name]
                    for f in dataclasses.fields(KeywordRuleSpec)
                    if f.name in k
                }))
            elif isinstance(k, KeywordRuleSpec):
                kw_specs.append(k)
        d["l1_keywords"] = kw_specs
        # Drop unknown keys for forward-compatibility.
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    @classmethod
    def from_json(cls, text: str) -> "SkillDef":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> "SkillDef":
        """Parse from a YAML string.  Requires PyYAML."""
        try:
            import yaml  # type: ignore
            return cls.from_dict(yaml.safe_load(text))
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML import.  "
                "Install with: pip install pyyaml"
            ) from exc


__all__ = ["KeywordRuleSpec", "SkillDef"]
