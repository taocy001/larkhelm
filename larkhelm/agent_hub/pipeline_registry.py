"""larkhelm · agent_hub · PipelineRegistry — CRUD + JSON persistence.

Thread-safe registry for :class:`~larkhelm.agent_hub.pipeline_types.PipelineDef`
instances.  Pipelines can be added and disabled at runtime without restarting
the bridge.

Persistence
-----------
User-created pipelines are auto-saved to ``DATA_DIR/pipelines/<id>.json``
whenever :meth:`PipelineRegistry.register` is called with ``source != "builtin"``.
:meth:`PipelineRegistry.load_from_dir` scans the same directory at startup.

AGENT_REGISTRY integration
---------------------------
Each registered pipeline gets a corresponding :class:`DevPipelineAgent` in
``AGENT_REGISTRY`` so the intent-dispatch pipeline can route to it.
Unregistering removes it from both registries.
"""
from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from larkhelm.crew_types import CrewPlan

from larkhelm.agent_hub.pipeline_types import PipelineDef


# ── Builtin pipeline stage definitions ────────────────────────────────────────
# Two pre-defined pipeline variants complementing the default 6-stage /dev flow.

_HOTFIX_STAGES: list[dict] = [
    {
        "id": "implementer",
        "role": "工程师",
        "model": "",
        "task_profile": "engineer",
        "system": (
            "你是一个资深工程师，专注于精准修复用户描述的具体问题。\n\n"
            "## 工作步骤\n"
            "1. 用 Read/Grep/Glob 定位问题相关代码，不要大范围扫描无关文件\n"
            "2. 精准修复：只改动与问题直接相关的代码，不要顺手重构\n"
            "3. 修复后运行相关测试验证（如有）\n"
            "4. 将改动摘要写入 .crew_workspace/changes.md：`文件路径 — 改动内容和原因`\n\n"
            "**注意**：修复范围最小化，严格遵循现有代码风格。"
        ),
        "prompt": (
            "请直接修复以下问题，完成后写入 .crew_workspace/changes.md：\n\n"
            "{requirement}"
        ),
        "depends_on": [],
        "timeout_factor": 8,
        "output_file": "changes.md",
    },
    {
        "id": "reviewer",
        "role": "代码审查员",
        "model": "",
        "task_profile": "reviewer",
        "system": (
            "你是一个严格的代码审查员，审查本次修复的所有代码。\n\n"
            "**必须逐条检查以下 8 项，每项给出 ✅ 或 ❌ 及说明：**\n"
            "1. 安全：无注入/XSS，无硬编码密钥，无不安全反序列化\n"
            "2. 错误处理：异常是否被合理捕获和处理\n"
            "3. 边界条件：空值、极大值、并发访问是否正确处理\n"
            "4. 代码规范：命名一致、无重复代码、函数职责单一\n"
            "5. 性能：无 N+1 查询、无不必要循环、无内存泄漏\n"
            "6. 测试覆盖：修复是否有对应测试用例\n"
            "7. 文档：公共接口是否有必要注释\n"
            "8. 完整性：changes.md 中的改动是否真正解决了问题\n\n"
            "将检查结果输出到 .crew_workspace/review.md，不要自行修改代码。\n\n"
            "⚠️ 输出的最后一行必须且只能是以下之一（不含其他字符）：\n"
            "APPROVED\n"
            "REJECTED"
        ),
        "prompt": (
            "请读取 .crew_workspace/changes.md 了解本次修复内容，"
            "对所有改动代码进行 8 项审查，输出到 .crew_workspace/review.md。\n\n"
            "**重要**：请直接输出结果，不要等待用户确认。"
        ),
        "depends_on": ["implementer"],
        "timeout_factor": 4,
        "output_file": "review.md",
        "exit_marker": "APPROVED",
        "fail_marker": "REJECTED",
        "is_gatekeeper": True,
    },
]

_REVIEW_ONLY_STAGES: list[dict] = [
    {
        "id": "reviewer",
        "role": "代码审查员",
        "model": "",
        "task_profile": "reviewer",
        "system": (
            "你是一个严格的代码审查员，直接审查用户指定的代码或变更集。\n\n"
            "**必须逐条检查以下 8 项，每项给出 ✅ 或 ❌ 及说明：**\n"
            "1. 安全：无 SQL 注入/命令注入/XSS，无硬编码密钥\n"
            "2. 错误处理：异常是否被合理捕获和处理\n"
            "3. 边界条件：空值、零值、极大值、并发访问是否正确\n"
            "4. 代码规范：命名一致、无重复代码、函数职责单一、无无用注释\n"
            "5. 性能：无 N+1 查询、无不必要循环、无内存泄漏风险\n"
            "6. 测试覆盖：核心逻辑和边界条件是否有对应测试\n"
            "7. 文档：公共接口和复杂逻辑是否有必要注释\n"
            "8. 设计合理性：架构、接口设计、命名是否符合项目约定\n\n"
            "将检查结果输出到 .crew_workspace/review.md，不要修改被审查的代码。\n\n"
            "⚠️ 输出的最后一行必须且只能是以下之一（不含其他字符）：\n"
            "APPROVED\n"
            "REJECTED"
        ),
        "prompt": (
            "请对以下代码/文件/变更进行完整的 8 项审查。"
            "如需读取相关文件请直接使用 Read/Grep/Glob 工具，"
            "最终将审查报告写入 .crew_workspace/review.md：\n\n"
            "{requirement}"
        ),
        "depends_on": [],
        "timeout_factor": 8,
        "output_file": "review.md",
        "exit_marker": "APPROVED",
        "fail_marker": "REJECTED",
        "is_gatekeeper": True,
    },
]

_BUILTIN_PIPELINE_DEFS: list[dict] = [
    {
        "id": "hotfix",
        "name": "紧急修复",
        "description": "两阶段流水线：工程师直接修复 → 审查员验证。跳过 PM/架构，适合已知定点问题且需要代码审查的场景",
        "stages": _HOTFIX_STAGES,
        "synthesis_prompt": (
            "请综合输出一份简洁的修复报告，包含：\n"
            "1. 修复了什么问题（参考 changes.md）\n"
            "2. 审查结论（参考 review.md 8 项结果）\n"
            "3. 遗留风险（如有）"
        ),
        "l1_keywords": [
            {"pattern": "紧急修复",        "strength": 0.90, "note": "hotfix 触发词"},
            {"pattern": "修复并审查",      "strength": 0.90},
            {"pattern": "修完帮我review",  "strength": 0.90, "note": "强于 reviewer skill 的 0.88"},
            {"pattern": "修完审查一下",    "strength": 0.88},
            {"pattern": "re:\\bhotfix\\b", "strength": 0.90, "note": "英文 hotfix"},
        ],
        "source": "builtin",
    },
    {
        "id": "review-only",
        "name": "完整代码审查",
        "description": "单阶段审查流水线：审查员直接读取代码，输出结构化 review.md，适合需要持久化审查报告的场景",
        "stages": _REVIEW_ONLY_STAGES,
        "synthesis_prompt": (
            "请综合输出审查结论摘要，包含 8 项检查的总体通过/失败情况及主要发现。"
        ),
        "l1_keywords": [
            {"pattern": "写一份review报告",   "strength": 0.88},
            {"pattern": "生成review报告",     "strength": 0.88},
            {"pattern": "做一次完整的代码审查", "strength": 0.85},
            {"pattern": "re:write\\s+a\\s+(?:code\\s+)?review\\s+report\\b", "strength": 0.85},
            {"pattern": "re:full\\s+code\\s+review\\b", "strength": 0.82},
        ],
        "source": "builtin",
    },
]


# ── PipelineRegistry ───────────────────────────────────────────────────────────


class PipelineRegistry:
    """Thread-safe registry for :class:`PipelineDef` instances."""

    def __init__(self) -> None:
        self._pipelines: dict[str, PipelineDef] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register(self, pd: PipelineDef, *, persist: bool | None = None) -> None:
        """Register (or replace) a PipelineDef and mirror it into AGENT_REGISTRY."""
        if not pd.id:
            raise ValueError("PipelineDef.id must be non-empty")
        with self._lock:
            self._pipelines[pd.id] = pd
        self._sync_agent_registry(pd)
        if persist is True or (persist is None and pd.source != "builtin"):
            self._maybe_persist(pd)

    def unregister(self, pipeline_id: str, *, delete_file: bool = False) -> bool:
        with self._lock:
            pd = self._pipelines.pop(pipeline_id, None)
        if pd is None:
            return False
        try:
            from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
            AGENT_REGISTRY.unregister(pipeline_id)
        except Exception:
            pass
        if delete_file and pd.source == "user":
            self._delete_file(pipeline_id)
        return True

    def get(self, pipeline_id: str) -> PipelineDef | None:
        with self._lock:
            return self._pipelines.get(pipeline_id)

    def list_all(self, *, include_disabled: bool = False) -> list[PipelineDef]:
        with self._lock:
            pds = list(self._pipelines.values())
        if not include_disabled:
            pds = [p for p in pds if p.enabled]
        return sorted(pds, key=lambda p: (p.source != "builtin", p.id))

    def __iter__(self) -> Iterator[PipelineDef]:
        with self._lock:
            return iter(list(self._pipelines.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._pipelines)

    # ------------------------------------------------------------------
    # L1 keyword rule integration
    # ------------------------------------------------------------------

    def get_l1_rules(self) -> list[tuple[str, str, float, str]]:
        """Return ``(pattern, pipeline_id, strength, note)`` for all enabled pipelines."""
        rules: list[tuple[str, str, float, str]] = []
        with self._lock:
            pds = list(self._pipelines.values())
        for pd in pds:
            if not pd.enabled:
                continue
            for kw in pd.l1_keywords:
                if isinstance(kw, dict) and kw.get("pattern"):
                    rules.append((
                        str(kw["pattern"]),
                        pd.id,
                        float(kw.get("strength", 0.80)),
                        str(kw.get("note", "")),
                    ))
        return rules

    # ------------------------------------------------------------------
    # Plan builder
    # ------------------------------------------------------------------

    def build_plan(self, pipeline_id: str, requirement: str, cwd: str) -> "CrewPlan | None":
        """Instantiate a :class:`~larkhelm.crew_types.CrewPlan` from a registered pipeline."""
        pd = self.get(pipeline_id)
        if pd is None:
            return None
        try:
            return _build_crew_plan(pd, requirement, cwd)
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[PipelineRegistry] build_plan {pipeline_id!r}: {e}")
            return None

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def load_from_dir(self, directory: Path) -> list[PipelineDef]:
        """Scan ``DATA_DIR/pipelines/`` for ``*.json`` files and load them."""
        loaded: list[PipelineDef] = []
        if not directory.is_dir():
            return loaded
        for path in sorted(directory.glob("*.json")):
            try:
                pd = PipelineDef.from_json(path.read_text(encoding="utf-8"))
                self.register(pd, persist=False)
                loaded.append(pd)
            except Exception as e:
                from larkhelm.log import lazy_debug_log
                lazy_debug_log(f"[PipelineRegistry] load {path.name}: {e}")
        return loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_agent_registry(self, pd: PipelineDef) -> None:
        """Create/replace a DevPipelineAgent in AGENT_REGISTRY for *pd*."""
        try:
            from larkhelm.agent_hub.builtin.pipeline_agent import DevPipelineAgent
            from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
            AGENT_REGISTRY.register(DevPipelineAgent(pd.id, pd.description))
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[PipelineRegistry] _sync_agent_registry {pd.id!r}: {e}")

    def _maybe_persist(self, pd: PipelineDef) -> None:
        try:
            import larkhelm.config as _cfg
            data_dir = Path(getattr(_cfg, "DATA_DIR", "") or "")
            if not data_dir:
                return
            pipelines_dir = data_dir / "pipelines"
            pipelines_dir.mkdir(parents=True, exist_ok=True)
            (pipelines_dir / f"{pd.id}.json").write_text(pd.to_json(), encoding="utf-8")
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[PipelineRegistry] _maybe_persist {pd.id!r}: {e}")

    def _delete_file(self, pipeline_id: str) -> None:
        try:
            import larkhelm.config as _cfg
            data_dir = Path(getattr(_cfg, "DATA_DIR", "") or "")
            if not data_dir:
                return
            path = data_dir / "pipelines" / f"{pipeline_id}.json"
            if path.exists():
                path.unlink()
        except Exception:
            pass


# ── Plan builder helper ────────────────────────────────────────────────────────


def _build_crew_plan(pd: PipelineDef, requirement: str, cwd: str) -> "CrewPlan":
    """Convert a :class:`PipelineDef` to a :class:`~larkhelm.crew_types.CrewPlan`."""
    import larkhelm.config as _cfg
    from larkhelm.crew_types import AgentSpec, CrewPlan

    rt = int(getattr(_cfg, "RESPONSE_TIMEOUT", 300))
    tmpl_vars = {"requirement": requirement, "cwd": cwd}
    known_spec_fields = {f.name for f in dataclasses.fields(AgentSpec)}
    agents: list[AgentSpec] = []

    for stage in pd.stages:
        d = dict(stage)

        # Resolve timeout: explicit int > factor × RESPONSE_TIMEOUT > default 4×
        if d.get("timeout"):
            d.pop("timeout_factor", None)
        else:
            factor = int(d.pop("timeout_factor", 4))
            d["timeout"] = rt * factor

        # Template substitution in prompt and system strings
        for field_name in ("prompt", "system"):
            if isinstance(d.get(field_name), str):
                try:
                    d[field_name] = d[field_name].format(**tmpl_vars)
                except (KeyError, ValueError, IndexError):
                    pass  # leave unsubstituted — don't break on partial templates

        # Ensure required AgentSpec fields have fallback values
        d.setdefault("model", "")
        d.setdefault("system", "")
        d.setdefault("prompt", "")
        d.setdefault("depends_on", [])

        spec_kwargs = {k: v for k, v in d.items() if k in known_spec_fields}
        agents.append(AgentSpec(**spec_kwargs))

    title_line = requirement.split("\n", 1)[0].strip() or requirement[:30]
    return CrewPlan(
        title=f"{pd.name}：{title_line[:25]}",
        agents=agents,
        synthesis_prompt=pd.synthesis_prompt,
    )


# ── Module-level singleton ─────────────────────────────────────────────────────

PIPELINE_REGISTRY: PipelineRegistry = PipelineRegistry()


def register_builtin_pipelines() -> None:
    """Register all built-in pipeline variants and load user pipelines from DATA_DIR."""
    for d in _BUILTIN_PIPELINE_DEFS:
        try:
            pd = PipelineDef.from_dict(dict(d))
            PIPELINE_REGISTRY.register(pd, persist=False)
        except Exception as e:
            from larkhelm.log import lazy_debug_log
            lazy_debug_log(f"[PipelineRegistry] register builtin {d.get('id')!r}: {e}")

    try:
        import larkhelm.config as _cfg
        data_dir = Path(getattr(_cfg, "DATA_DIR", "") or "")
        if data_dir:
            loaded = PIPELINE_REGISTRY.load_from_dir(data_dir / "pipelines")
            if loaded:
                from larkhelm.log import lazy_debug_log
                lazy_debug_log(
                    f"[PipelineRegistry] loaded {len(loaded)} user pipeline(s): "
                    + ", ".join(p.id for p in loaded)
                )
    except Exception:
        pass


__all__ = ["PipelineDef", "PipelineRegistry", "PIPELINE_REGISTRY", "register_builtin_pipelines"]
