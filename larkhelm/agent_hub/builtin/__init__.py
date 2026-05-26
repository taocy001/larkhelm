"""larkhelm · agent_hub.builtin — register all built-in Agents and Skills.

Architecture note
-----------------
This package contains **two kinds** of built-ins:

Agents (Python classes, stateful)
    :class:`ChatAgent`, :class:`DevAgent`, :class:`CrewAgent`,
    :class:`PlanAgent`, :class:`DocAgent`, :class:`FileAgent`,
    :class:`GitHubAgent`, :class:`CalendarAgent`

    These are complex pipelines or stateful executors with their own
    subprocess / session / multi-step logic.  They live as Python classes.

Skills (data-driven, stateless)
    Defined as :class:`~larkhelm.agent_hub.skill_types.SkillDef` dicts in
    ``builtin/skills/_defs.py`` and executed by a generic
    :class:`~larkhelm.agent_hub.skill_runner.SkillExecutor`.

    Because Skills are pure data, operators can:
    * Modify the system prompt or L1 keywords without touching Python code.
    * Add new Skills via ``/skill new`` or by dropping a JSON file in
      ``DATA_DIR/skills/``.
    * Disable Skills at runtime via ``/skill disable <id>``.
"""
from __future__ import annotations

from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
from larkhelm.agent_hub.builtin.calendar_agent import CalendarAgent
from larkhelm.agent_hub.builtin.chat_agent import ChatAgent
from larkhelm.agent_hub.builtin.crew_agent import CrewAgent
from larkhelm.agent_hub.builtin.dev_agent import DevAgent
from larkhelm.agent_hub.builtin.doc_agent import DocAgent
from larkhelm.agent_hub.builtin.file_agent import FileAgent
from larkhelm.agent_hub.builtin.github_agent import GitHubAgent
from larkhelm.agent_hub.builtin.plan_agent import PlanAgent

_AGENT_CLASSES = (
    ChatAgent, DevAgent, CrewAgent, PlanAgent, DocAgent,
    FileAgent, GitHubAgent, CalendarAgent,
)


def register_builtin_agents() -> None:
    """Register stateful Pipeline/Agent classes into AGENT_REGISTRY."""
    for cls in _AGENT_CLASSES:
        AGENT_REGISTRY.register(cls())


def register_builtin_skills() -> None:
    """Register data-driven Skill definitions via SkillRegistry.

    Each SkillDef is automatically mirrored into AGENT_REGISTRY as a
    :class:`~larkhelm.agent_hub.skill_runner.SkillExecutor` so routing works
    transparently.  Also loads user-created skills from DATA_DIR/skills/.
    """
    from pathlib import Path
    from larkhelm.agent_hub.skill_types import SkillDef
    from larkhelm.agent_hub.skill_registry import SKILL_REGISTRY
    from larkhelm.agent_hub.builtin.skills._defs import _BUILTIN_SKILL_DICTS

    # Register built-in skills (source="builtin").
    for d in _BUILTIN_SKILL_DICTS:
        try:
            sk = SkillDef.from_dict(dict(d))  # defensive copy
            SKILL_REGISTRY.register(sk, persist=False)
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[builtin] register skill {d.get('id')!r}: {e}")

    # Load user-created skills from DATA_DIR/skills/.
    try:
        import larkhelm.config as _cfg
        data_dir = Path(getattr(_cfg, "DATA_DIR", ""))
        if data_dir:
            loaded = SKILL_REGISTRY.load_from_dir(data_dir / "skills")
            if loaded:
                from larkhelm.log import lazy_debug_log
                lazy_debug_log(
                    f"[builtin] loaded {len(loaded)} user skill(s): "
                    + ", ".join(s.id for s in loaded)
                )
    except Exception:
        pass


def register_builtin_pipelines() -> None:
    """Register built-in pipeline variants (hotfix, review-only) and load user pipelines."""
    from larkhelm.agent_hub.pipeline_registry import register_builtin_pipelines as _reg
    _reg()


def register_all() -> None:
    """Idempotent: register all built-in agents, skills, and pipeline variants."""
    register_builtin_agents()
    register_builtin_skills()
    register_builtin_pipelines()


register_all()


__all__ = [
    "ChatAgent", "DevAgent", "CrewAgent", "PlanAgent", "DocAgent",
    "FileAgent", "GitHubAgent", "CalendarAgent",
    "register_all", "register_builtin_agents", "register_builtin_skills",
]
