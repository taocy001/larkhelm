"""
larkhelm · Crew DAG topological sort algorithms and scheduling helpers
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from larkhelm.crew_types import AgentSpec, AgentState, AgentStatus, CrewState, CREW_RESULT_PREVIEW


def _detect_cycle(agents: list[dict]) -> list[str] | None:
    id_map    = {a["id"]: a for a in agents}
    visited   = set()
    in_stack  = set()
    path: list[str] = []

    def dfs(node: str) -> bool:
        visited.add(node)
        in_stack.add(node)
        path.append(node)
        for dep in id_map.get(node, {}).get("depends_on", []):
            if dep not in visited:
                if dfs(dep):
                    return True
            elif dep in in_stack:
                path.append(dep)
                return True
        in_stack.discard(node)
        path.pop()
        return False

    for a in agents:
        if a["id"] not in visited:
            if dfs(a["id"]):
                return list(path)
    return None


def _topo_waves(agents: list[AgentSpec]) -> list[list[AgentSpec]]:
    """BFS topological layering; agents in the same layer can run in parallel."""
    id_map     = {a.id: a for a in agents}
    in_degree  = {a.id: len(a.depends_on) for a in agents}
    dependents: dict[str, list[str]] = {a.id: [] for a in agents}
    for a in agents:
        for dep in a.depends_on:
            dependents[dep].append(a.id)

    waves: list[list[AgentSpec]] = []
    ready = [aid for aid, deg in in_degree.items() if deg == 0]
    while ready:
        waves.append([id_map[aid] for aid in ready])
        nxt: list[str] = []
        for aid in ready:
            for dep_id in dependents[aid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    nxt.append(dep_id)
        ready = nxt
    return waves


def _topo_waves_subset(all_agents: list[AgentSpec], subset_ids: set[str]) -> list[list[AgentSpec]]:
    """Same as _topo_waves, but only processes agents in subset_ids; dependencies outside the subset are treated as already satisfied."""
    filtered = [
        dataclasses.replace(spec, depends_on=[d for d in spec.depends_on if d in subset_ids])
        for spec in all_agents
        if spec.id in subset_ids
    ]
    return _topo_waves(filtered)


# Phase 4: failure propagation helper
def _get_failed_dep(state: CrewState, spec: AgentSpec,
                    workspace_path: "Path | None" = None) -> str:
    """Return the first failed dependency id if any upstream (transitively) of spec is FAILED/CANCELLED
    and not needs_retry; otherwise return None.

    Partial-delivery rule (when ``workspace_path`` provided)
    --------------------------------------------------------
    A FAILED upstream is treated as "delivered" — and therefore does
    NOT block its downstream — when ALL of:

      * its ``AgentSpec.output_file`` is set,
      * the file at ``workspace_path / output_file`` exists, and
      * the file size is > 0 bytes.

    Real-world case this fixes (commit b3a116f post-mortem): the
    implementer's claude CLI got OOM-killed mid-stream **after** the
    Write tool had atomically committed the file. The agent thread
    caught the subprocess error and marked status=FAILED, even though
    ``changes.md`` was complete on disk. The pre-fix behavior cascaded
    that FAILED into fixer→qa→reviewer = three more skipped agents
    for one real failure. With this rule, fixer (which reads
    changes.md anyway) gets to run on the delivered artefact.

    CANCELLED upstream is **unchanged** — user pressed cancel, so we
    respect that intent even if the file happened to be written first.
    Only FAILED is reinterpreted.

    ``workspace_path=None`` (the legacy signature) preserves the old
    "any FAILED blocks downstream" behavior — keeps non-runner callers
    (scheduler unit tests, possible external dispatchers) unchanged.
    """
    id_map  = {s.id: s for s in state.plan.agents}
    checked: set[str] = set()

    def _has_partial_delivery(dep_spec: AgentSpec) -> bool:
        """True iff the dep's declared ``output_file`` exists and is non-empty."""
        if workspace_path is None or not dep_spec.output_file:
            return False
        try:
            p = workspace_path / dep_spec.output_file
            return p.exists() and p.stat().st_size > 0
        except Exception:
            # Disk weirdness / perms — fall back to "no delivery" so we
            # behave like the conservative legacy path on any IO trouble.
            return False

    def _check(dep_id: str) -> str:
        if dep_id in checked:
            return None
        checked.add(dep_id)
        ag = state.agents.get(dep_id)
        dep_spec = id_map.get(dep_id)
        if ag and ag.status in (AgentStatus.FAILED, AgentStatus.CANCELLED) and not ag.needs_retry:
            # Only FAILED qualifies for the partial-delivery escape;
            # CANCELLED stays blocking (user-intent semantics).
            if ag.status == AgentStatus.FAILED and dep_spec is not None \
                    and _has_partial_delivery(dep_spec):
                # Don't return as a blocking failure, but still recurse
                # upstream — the dep's own dependencies might be the
                # real reason it didn't finish, and those could lack
                # output_file fallbacks.
                pass
            else:
                return dep_id
        if dep_spec:
            for upstream in dep_spec.depends_on:
                r = _check(upstream)
                if r:
                    return r
        return None

    for dep_id in spec.depends_on:
        r = _check(dep_id)
        if r:
            return r
    return None


def _resolve_prompt(template: str, state: CrewState) -> str:
    """Replace {agent_N_result} placeholders with the corresponding agent's output summary and file reference."""
    from larkhelm.chat_state import _get_cwd

    workspace = _workspace_dir(state.chat_id, state.crew_id)

    def _replace(m: re.Match) -> str:
        agent_id  = m.group(1)
        a         = state.agents.get(agent_id)
        if not a:
            return f"[{agent_id} does not exist]"
        if a.status == AgentStatus.DONE:
            result_file = workspace / f"{agent_id}_result.txt"
            summary = a.result[:CREW_RESULT_PREVIEW]
            suffix  = "…（已截断）" if len(a.result) > CREW_RESULT_PREVIEW else ""
            # Prefer showing output_file (if it exists)
            cwd = _get_cwd(state.chat_id)
            out_file_ref = ""
            if a.spec.output_file:
                out_path = Path(cwd) / ".crew_workspace" / a.spec.output_file
                out_file_ref = f"\n主要输出文件：{out_path}（可用 Read 工具读取完整内容）"
                if a.feishu_doc_url:
                    out_file_ref += f"\n飞书文档：{a.feishu_doc_url}"
            ref = f"{out_file_ref}\n完整输出见：{result_file}（可用 Read 工具读取全文）"
            return f"【{a.spec.role} 输出摘要】\n{summary}{suffix}{ref}"
        elif a.status == AgentStatus.FAILED:
            return f"[{agent_id}({a.spec.role}) 执行失败: {a.error[:100]}]"
        return f"[{agent_id} 结果未就绪]"

    return re.sub(r'\{(agent_\d+)_result\}', _replace, template)


def _workspace_dir(chat_id: str, crew_id: str) -> Path:
    import larkhelm.config as _cfg
    d = _cfg.SESSION_DIR / chat_id / f"crew_{crew_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d
