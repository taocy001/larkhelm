"""larkhelm · agent_hub · SkillRegistry — CRUD + JSON persistence.

Thread-safe registry for :class:`~larkhelm.agent_hub.skill_types.SkillDef`
instances.  Skills can be added, modified, and deleted at runtime without
restarting the bridge.

Persistence
-----------
User-created skills are auto-saved to ``DATA_DIR/skills/<id>.json`` whenever
:meth:`SkillRegistry.register` is called with ``source != "builtin"``.
:meth:`SkillRegistry.load_from_dir` scans the same directory at startup.

AGENT_REGISTRY integration
---------------------------
Each registered :class:`SkillDef` automatically gets a corresponding
:class:`~larkhelm.agent_hub.skill_runner.SkillExecutor` entry in the global
``AGENT_REGISTRY`` so the existing intent-dispatch pipeline routes to it
without any changes.  Unregistering a skill removes it from both registries.

L1 keyword rules
-----------------
:meth:`SkillRegistry.get_l1_rules` returns live keyword rules from all enabled
skills.  ``intent_router._resolve_l1`` calls this at routing time so newly
added skills are immediately usable without restarting.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

from larkhelm.agent_hub.skill_types import SkillDef


class SkillRegistry:
    """Thread-safe registry for :class:`SkillDef` instances.

    Usage::

        from larkhelm.agent_hub.skill_registry import SKILL_REGISTRY

        SKILL_REGISTRY.register(my_skill)
        sk = SKILL_REGISTRY.get("translate")
        all_skills = SKILL_REGISTRY.list_all()
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDef] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register(self, skill: SkillDef, *, persist: bool | None = None) -> None:
        """Register (or replace) a SkillDef and mirror it into AGENT_REGISTRY.

        *persist* controls JSON serialisation:
        - ``None`` (default): auto-save only for user/plugin skills.
        - ``True``: always save.
        - ``False``: never save (useful for tests).
        """
        if not skill.id:
            raise ValueError("SkillDef.id must be non-empty")
        with self._lock:
            self._skills[skill.id] = skill
        self._sync_agent_registry(skill)
        if persist is True or (persist is None and skill.source != "builtin"):
            self._maybe_persist(skill)

    def unregister(self, skill_id: str, *, delete_file: bool = False) -> bool:
        """Remove skill from both registries.  Returns False if not found.

        When *delete_file* is True, also removes the persisted JSON file from
        ``DATA_DIR/skills/<skill_id>.json`` (only user-created skills).
        """
        with self._lock:
            sk = self._skills.pop(skill_id, None)
        if sk is None:
            return False
        try:
            from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
            AGENT_REGISTRY.unregister(skill_id)
        except Exception:
            pass
        if delete_file and sk.source == "user":
            self._delete_file(skill_id)
        return True

    def update(self, skill_id: str, **kwargs) -> SkillDef | None:
        """Patch fields on an existing skill; returns updated SkillDef or None.

        Automatically re-registers the executor in AGENT_REGISTRY so routing
        picks up the new description/keywords immediately.
        """
        with self._lock:
            sk = self._skills.get(skill_id)
            if sk is None:
                return None
            d = sk.to_dict()
            d.update(kwargs)
            new_sk = SkillDef.from_dict(d)
            self._skills[skill_id] = new_sk
        self._sync_agent_registry(new_sk)
        if new_sk.source != "builtin":
            self._maybe_persist(new_sk)
        return new_sk

    def get(self, skill_id: str) -> SkillDef | None:
        with self._lock:
            return self._skills.get(skill_id)

    def list_all(self, *, include_disabled: bool = False) -> list[SkillDef]:
        with self._lock:
            skills = list(self._skills.values())
        if not include_disabled:
            skills = [s for s in skills if s.enabled]
        return sorted(skills, key=lambda s: (s.source != "builtin", s.id))

    def __iter__(self) -> Iterator[SkillDef]:
        with self._lock:
            return iter(list(self._skills.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._skills)

    # ------------------------------------------------------------------
    # L1 keyword rule integration for intent_router
    # ------------------------------------------------------------------

    def get_l1_rules(self) -> list[tuple[str, str, float, str]]:
        """Return ``(pattern, skill_id, strength, note)`` tuples for all enabled skills.

        The intent router calls this at L1 routing time to include dynamically
        registered skills in the keyword-matching tier without a restart.
        """
        rules: list[tuple[str, str, float, str]] = []
        with self._lock:
            skills = list(self._skills.values())
        for sk in skills:
            if not sk.enabled:
                continue
            for kw in sk.l1_keywords:
                rules.append((kw.pattern, sk.id, kw.strength, kw.note))
        return rules

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def load_from_file(self, path: Path) -> SkillDef | None:
        """Load a single JSON (or YAML if PyYAML available) file and register the skill."""
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix in (".yaml", ".yml"):
                sk = SkillDef.from_yaml(text)
            else:
                sk = SkillDef.from_json(text)
            self.register(sk, persist=False)
            return sk
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[SkillRegistry] load_from_file {path.name}: {e}")
            return None

    def load_from_dir(self, directory: Path) -> list[SkillDef]:
        """Scan directory for *.json / *.yaml / *.yml and load all skills.

        Call this once at bridge startup from ``DATA_DIR/skills/``.
        """
        loaded: list[SkillDef] = []
        if not directory.is_dir():
            return loaded
        for suffix in ("*.json", "*.yaml", "*.yml"):
            for path in sorted(directory.glob(suffix)):
                sk = self.load_from_file(path)
                if sk is not None:
                    loaded.append(sk)
        return loaded

    def save_to_file(self, skill_id: str, path: Path) -> bool:
        """Serialise a skill to *path* (JSON or YAML based on suffix).

        Returns True on success, False if skill not found or write failed.
        """
        sk = self.get(skill_id)
        if sk is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in (".yaml", ".yml"):
                path.write_text(sk.to_yaml(), encoding="utf-8")
            else:
                path.write_text(sk.to_json(), encoding="utf-8")
            return True
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[SkillRegistry] save_to_file {path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_agent_registry(self, skill: SkillDef) -> None:
        """Create/replace a SkillExecutor in AGENT_REGISTRY for *skill*."""
        try:
            from larkhelm.agent_hub.skill_runner import SkillExecutor
            from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
            AGENT_REGISTRY.register(SkillExecutor(skill))
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[SkillRegistry] _sync_agent_registry {skill.id}: {e}")

    def _maybe_persist(self, skill: SkillDef) -> None:
        """Save skill to DATA_DIR/skills/<id>.json (silent on failure)."""
        try:
            import larkhelm.config as _cfg
            data_dir = Path(getattr(_cfg, "DATA_DIR", ""))
            if not data_dir:
                return
            skills_dir = data_dir / "skills"
            skills_dir.mkdir(parents=True, exist_ok=True)
            path = skills_dir / f"{skill.id}.json"
            path.write_text(skill.to_json(), encoding="utf-8")
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[SkillRegistry] _maybe_persist {skill.id}: {e}")

    def _delete_file(self, skill_id: str) -> None:
        """Remove DATA_DIR/skills/<skill_id>.json (silent on failure)."""
        try:
            import larkhelm.config as _cfg
            data_dir = Path(getattr(_cfg, "DATA_DIR", ""))
            if not data_dir:
                return
            path = data_dir / "skills" / f"{skill_id}.json"
            if path.exists():
                path.unlink()
        except Exception:
            pass


# ── Module-level singleton ─────────────────────────────────────────────────
SKILL_REGISTRY: SkillRegistry = SkillRegistry()

__all__ = ["SkillDef", "SkillRegistry", "SKILL_REGISTRY"]
