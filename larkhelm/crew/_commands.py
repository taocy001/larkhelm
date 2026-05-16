"""
larkhelm · Crew public command functions and Manager planning
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from larkhelm.log import _debug_log, log_entry
from larkhelm.crew_types import (
    AgentSpec, AgentState, AgentStatus, CrewPlan, CrewState,
    CREW_MAX_AGENTS,
)
from larkhelm.crew_card import _crew_update_card


# ═══════════════════════════════════════════════════════════════
#  Workspace metadata helpers (stale-detection for /dev)
# ═══════════════════════════════════════════════════════════════

# Workspace artifacts older than this are treated as stale and cleared on the
# next /dev invocation, even if the task_hash still matches. The original
# resume-on-interrupt semantics ("same hash + uncompleted → reuse design.md")
# assumed users come back within minutes; after a day the artifacts are almost
# certainly unrelated to whatever they're now asking, and feeding them silently
# into the implementer caused cross-task contamination.
_WORKSPACE_STALE_TTL = 24 * 3600  # 24h


def _task_hash(requirement: str) -> str:
    return hashlib.md5(requirement.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
#  /dev context injection (chat history + memory)
# ═══════════════════════════════════════════════════════════════

# Caps tuned conservatively: PM's prompt budget is mostly spent on the system
# prompt (~1.5K tokens) and project file exploration; injecting up to ~6K
# chars of context still leaves headroom. The chat tail is bigger because
# planning discussions tend to be long; memory is short by construction.
_DEV_CTX_CHAT_TURNS = 12
_DEV_CTX_CHAT_CHARS = 4000
_DEV_CTX_MEMORY_CHARS = 2000


def _augment_requirement_with_context(requirement: str, chat_id: str, cwd: str) -> str:
    """Prepend recent chat turns + memory snippets to ``requirement`` so PM
    agent has the same context the user assumes when typing /dev.

    Returns ``requirement`` unchanged when neither chat history nor memory is
    available. Failures inside helper calls are swallowed via ``_debug_log``;
    /dev must keep running even when context retrieval breaks.

    The returned string is consumed only by ``_make_dev_pipeline`` (PM's
    prompt template). It is NOT used as ``task_key`` for hashing — the
    resume semantics still pin to the user-typed literal so re-running
    ``/dev <same X>`` reliably picks the same workspace_meta.
    """
    chat_ctx = ""
    try:
        from larkhelm.log import _get_recent_turns
        chat_ctx = _get_recent_turns(
            chat_id,
            max_turns=_DEV_CTX_CHAT_TURNS,
            max_chars=_DEV_CTX_CHAT_CHARS,
        )
    except Exception as e:
        _debug_log(f"[Dev] recent-turns load failed: {e}")

    mem_ctx = ""
    try:
        # REQ-22 (Phase C wrap-up): switch to v2 so the layered injection
        # logic (S50 lazy global / S51 project conditional) gets a real
        # ``query`` to gate on — previously the no-arg call hit the
        # fail-open branches and pulled in everything regardless of
        # whether the requirement actually needs global preferences or
        # project memory. ``requirement`` is the user-typed /dev <X>
        # text, which is the right ``query`` here. ``recent_turns`` is
        # intentionally omitted: the PM-context path keeps chat history
        # and memory cleanly separated below, so cross-layer dedup
        # would only complicate the prompt shape without saving tokens.
        from larkhelm.memory import get_memory_context_v2
        mem_ctx, _ = get_memory_context_v2(chat_id, cwd=cwd, query=requirement)
        if len(mem_ctx) > _DEV_CTX_MEMORY_CHARS:
            mem_ctx = mem_ctx[: _DEV_CTX_MEMORY_CHARS] + "\n…(truncated)"
    except Exception as e:
        _debug_log(f"[Dev] memory load failed: {e}")

    if not chat_ctx and not mem_ctx:
        return requirement

    sections = [requirement, "", "---", "",
                "## 任务背景上下文（仅供 PM 阶段理解需求，不要照抄进 PRD）"]
    if mem_ctx:
        sections.append(f"\n### 长期记忆\n{mem_ctx}")
    if chat_ctx:
        sections.append(f"\n### 最近对话\n{chat_ctx}")
    sections.append(
        "\n> 注意：以上仅为背景，**真正要实现的需求是文首那段**。"
        "若背景与需求冲突，以需求为准。"
    )
    return "\n".join(sections)


def _read_workspace_meta(ws_path: Path) -> dict:
    meta_file = ws_path / "workspace_meta.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            return {}
    return {}


def _write_workspace_meta(ws_path: Path, task_hash: str, completed: bool = False) -> None:
    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "workspace_meta.json").write_text(
        json.dumps({"task_hash": task_hash, "completed": completed})
    )


def _clear_workspace(ws_path: Path) -> None:
    if ws_path.exists():
        for f in ws_path.iterdir():
            try:
                f.unlink()
            except Exception:
                pass


_FEISHU_URL_RE = re.compile(r'https://[a-zA-Z0-9-]+\.feishu\.cn/[^\s\]>）]+')
_DEV_DOC_MAX_CHARS = 50_000  # generous limit for requirement docs


def _expand_doc_requirement(requirement: str) -> str:
    """If the requirement contains Feishu doc URLs, read and inline their content.

    The original URL text is preserved at the end so task_hash remains stable.
    Only the first 3 URLs are expanded; failures are silently skipped.
    """
    urls = _FEISHU_URL_RE.findall(requirement)
    if not urls:
        return requirement
    try:
        from larkhelm.lark_client import FeishuDocClient, parse_doc_url
        doc_client = FeishuDocClient()
    except Exception as e:
        _debug_log(f"[Dev] doc client init failed: {e}")
        return requirement

    injections = []
    for url in urls[:3]:
        ref = parse_doc_url(url)
        if ref is None:
            continue
        try:
            result = doc_client.read(ref, max_chars=_DEV_DOC_MAX_CHARS)
            label = result.title or url
            injections.append(f"[任务来源文档：《{label}》]\n{result.content}")
        except Exception as e:
            _debug_log(f"[Dev] failed to read doc {url}: {e}")

    if injections:
        return "\n\n".join(injections) + "\n\n---\n\n" + requirement
    return requirement


# Crew thread registry, used by wait_crews_done
_crew_threads: dict[str, threading.Thread] = {}   # crew_id → thread
_crew_threads_lock = threading.Lock()


def _register_crew_thread(crew_id: str, t: threading.Thread):
    with _crew_threads_lock:
        _crew_threads[crew_id] = t


def _unregister_crew_thread(crew_id: str):
    with _crew_threads_lock:
        _crew_threads.pop(crew_id, None)


# ═══════════════════════════════════════════════════════════════
#  Manager planning (/crew command)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  Automatic mode detection (Hermes orchestrator)
# ═══════════════════════════════════════════════════════════════

_HERMES_MODE_KEYWORDS = {
    "hermes_race": ["紧急", "快速", "尽快", "临时", "修复", "bug", "比选", "对比", "竞争", "最快"],
    "hermes_split": ["全栈", "前后端", "前端", "后端", "API", "页面", "UI", "界面", "网页", "React", "Vue"],
    "hermes_review": ["核心", "关键", "安全", "支付", "认证", "审计", "严格审查", "金融", "交易", "密码"],
}


def _detect_hermes_mode(requirement: str) -> str:
    """Detect if the requirement matches a Hermes orchestrator mode. Returns mode name or None."""
    req_lower = requirement.lower()
    for mode, keywords in _HERMES_MODE_KEYWORDS.items():
        if any(kw.lower() in req_lower for kw in keywords):
            return mode
    return None


_MANAGER_PROMPT_TPL = """\
你是一个任务规划专家。将以下需求分解为多 Agent 协作任务计划。

## 输出格式

只输出一个 JSON 代码块，不要有其他文字。格式如下：

```json
{{
  "title": "任务标题（15字以内）",
  "agents": [
    {{
      "id": "agent_1",
      "role": "角色名称",
      "model": "<必须从下方'可用模型清单'选取一个 id；不要照抄此处占位符>",
      "system": "该 agent 的角色定义和行为风格",
      "prompt": "具体任务描述，可用 {{agent_N_result}} 引用上游输出",
      "output_file": "该 agent 的主要输出文件名（相对于 .crew_workspace/），如无文件输出则留空字符串",
      "depends_on": [],
      "timeout": 300
    }}
  ],
  "synthesis_prompt": "最终整合所有 agent 输出的指令"
}}
```

## 规划规则

1. agents 数量：2 到 {max_agents} 个（根据任务复杂度决定，尽量并行）
2. id 格式：agent_1、agent_2……（从 1 开始）
3. depends_on：列出必须先完成的 agent id，可以为空列表
4. 依赖关系不能成环，无依赖的 agent 会并行执行
5. model：**必须从下方"可用模型清单"中选取 id**，或选 Hermes 编排模式 "hermes_race" / "hermes_split" / "hermes_review"。**禁用的模型不会出现在清单里，绝对不要选**
6. timeout：秒，范围 60 到 {max_timeout}
7. prompt 中可使用 {{agent_N_result}} 引用上游 agent 的输出摘要
8. system 字段：该 agent 的角色定义，指导其行为风格
9. output_file：agent 的主要输出文件名（相对于 .crew_workspace/），下游 agent 应优先读取该文件而非依赖摘要传递

## 可用模型清单（实时，来自 BackendRegistry）

{available_models}

## Hermes 编排模式（高级并行策略）

当任务适合以下场景时，使用 Hermes 编排模式代替单 agent：

- **hermes_race**（竞争模式）：同一任务需要多个 agent 并行尝试，取最快/最好的结果
  - 适用：紧急修复、快速原型、方案比选
  - 示例：{{"task": "修复登录bug", "agents": ["claude", "kimi"]}}

- **hermes_split**（分工模式）：任务可拆分为前后端/多模块并行开发
  - 适用：全栈功能开发、API+UI 同时实现
  - 示例：{{"backend_task": "FastAPI API", "frontend_task": "React 页面"}}

- **hermes_review**（评审模式）：需要实现→审查→测试的完整流水线
  - 适用：核心模块开发、安全关键代码、需要多轮验证
  - 示例：{{"task": "实现支付模块", "agents": ["claude", "kimi-thinking", "deepseek"]}}（具体 id 必须存在于"可用模型清单"中）

使用编排模式时：
- prompt 必须是 JSON 格式（包含 task/agents/context 等字段）
- 该 agent 会自动调度多个子 agent 完成工作
- timeout 应设置为普通 agent 的 2-3 倍

## 并行规划原则（重要）

**尽可能发掘并行机会**，让没有依赖关系的 agent 同时执行：

- 多个独立调研/分析子任务 → 各自独立 agent，depends_on 均为空，全部并行
- 代码实现中相互独立的模块（如前端/后端/脚本）→ 各自独立 implementer，并行执行
- 审查和测试 → 可对不同模块/角度并行审查，最后汇总
- **反例**：不要把所有工作串成一条链——只有真正有数据依赖时才加 depends_on

**两阶段规划模式**（适用于需求复杂、实现量大的任务）：

当任务同时包含"深度分析/设计"和"动手实现/修改"两类工作时，考虑分两阶段：
- **Phase 1（分析）**：调研/需求分析/方案设计 agent，产出文件到 `.crew_workspace/`
- **Phase 2（实现）**：实现 agent 以 depends_on 依赖 Phase 1 的 agent，直接读取其输出文件

这样分析结论自动流入实现阶段，无需人工转抄。

## 模型选择指南（按 tag）

清单里每个 backend 都带 tags，按需求性质从清单选：

- 任务需调用工具（Write/Edit/Bash/WebFetch）→ 选有 **tools** tag 的
- 任务涉及图像/视觉 → 选有 **vision** tag 的（清单里没有就跳过此类任务）
- 简单/便宜/高频小任务（摘要、分类、格式化）→ 优先选 **cheap** + **fast** tag 的
- 硬数学 / 长链推理 → 选名字带 **thinking** 的
- 中文任务、长文档 → kimi 系列通常最稳

**均衡分配原则**：避免所有 agent 都用同一个 backend。优先把**独立调研/分析任务**分给不同的 worker（如 deepseek/kimi），让 orchestrator 角色（claude）专注串联和综合。

**Hermes 编排模式**（与单 backend 平级的选项，model 字段写 hermes_race/split/review）：
- **hermes_race**：紧急任务多 agent 竞争，取最快——清单里 ≥2 个 worker 时可用
- **hermes_split**：全栈开发前后端并行——后端/前端独立子任务
- **hermes_review**：核心模块的实现→审查→测试流水线——多轮验证场景

## 自动 Hermes 触发规则

需求含以下关键词时优先选对应 Hermes 模式而非单 agent：

- **hermes_race**：紧急 / 快速 / 尽快 / 临时 / 修复 / bug / 比选 / 对比
- **hermes_split**：全栈 / 前后端 / 前端 / 后端 / API / 页面 / UI
- **hermes_review**：核心 / 关键 / 安全 / 支付 / 认证 / 审计 / 严格审查

不含以上关键词时，按"模型选择指南"从清单里挑 backend。

## 当前项目目录

{cwd}

## 用户需求

{requirement}
"""


def _build_available_models_section() -> str:
    """Build the live ``available_models`` block injected into the Manager prompt.

    Snapshots BackendRegistry, lists enabled+healthy backends with their tags
    and a short description, and explicitly names disabled backends so the
    planner doesn't pick them. Replaces the previous hardcoded "claude/
    gemini/kimi" menu in ``_MANAGER_PROMPT_TPL`` — that menu didn't reflect
    runtime config (gemini disabled) or new arrivals (deepseek), so the
    planner kept assigning tasks to backends that the dispatch layer would
    immediately reject.
    """
    from larkhelm.backend_registry import BACKEND_REGISTRY
    all_specs = BACKEND_REGISTRY.snapshot()
    enabled_healthy = [s for s in all_specs if s.enabled and s.healthy]
    disabled = [s for s in all_specs if not s.enabled]
    unhealthy = [s for s in all_specs if s.enabled and not s.healthy]

    if not enabled_healthy:
        return "(无可用 backend — 请检查 config.json + 运行时健康状态)"

    lines = []
    for s in sorted(enabled_healthy, key=lambda x: x.id):
        tags = ", ".join(s.tags) if s.tags else "—"
        desc = (s.description or s.capabilities or "").strip().split("\n")[0]
        if len(desc) > 100:
            desc = desc[:97] + "…"
        if desc:
            lines.append(f"- **{s.id}** [{tags}] — {desc}")
        else:
            lines.append(f"- **{s.id}** [{tags}]")

    if disabled:
        ids = ", ".join(sorted(s.id for s in disabled))
        lines.append(f"\n⛔ **已禁用，不要选**：{ids}")
    if unhealthy:
        ids = ", ".join(sorted(f"{s.id} ({s.last_error or '?'})" for s in unhealthy))
        lines.append(f"\n⚠️ **当前不健康，不要选**：{ids}")

    return "\n".join(lines)


def _crew_plan(chat_id: str, requirement: str, cwd: str,
               max_agents: int, cancel_ev: threading.Event) -> CrewPlan:
    """Call the Manager LLM (Claude tool_use) to generate a task plan. Returns None on failure."""
    import larkhelm.config as _cfg
    from larkhelm.crew._scheduler import _detect_cycle
    from larkhelm.backend_registry import BACKEND_REGISTRY

    # Bail out early if there are no enabled+healthy backends — Manager would
    # otherwise emit a plan referencing nonexistent ids, which dispatch would
    # then reject one-by-one. Better to fail fast at planning with a clear
    # message than to consume an entire planning timeout for garbage output.
    _enabled_healthy = [
        s for s in BACKEND_REGISTRY.snapshot()
        if s.enabled and s.healthy
    ]
    if not _enabled_healthy:
        _debug_log(
            "[Crew] Manager: no enabled+healthy backends in registry; "
            "refusing to plan (check config.json probe_models or the "
            "BackendRegistry health log)"
        )
        return None

    mgr_ns   = f"{chat_id}__crew_mgr_{uuid.uuid4().hex[:8]}"
    prompt   = _MANAGER_PROMPT_TPL.format(
        max_agents=max_agents,
        max_timeout=_cfg.HARD_TIMEOUT // 2,
        cwd=cwd,
        requirement=requirement,
        available_models=_build_available_models_section(),
    )

    import os as _os
    # Use get_ai_sem() at the point of acquisition rather than a
    # ``from ... import _ai_proc_sem`` binding — the latter freezes the sem
    # instance at import time and survives ``_init_ai_sem`` rebuilds, which
    # is the P0 bug fixed by the round-3 OOM defense refactor.
    from larkhelm.runner_base import get_ai_sem

    args = [
        _cfg.CLAUDE_CMD, "--print", "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions",
    ]

    env = {
        **_os.environ,
        "DBUS_SESSION_BUS_ADDRESS": "",
        "GCM_CREDENTIAL_STORAGE": "file",
        "FEISHU_CHAT_ID": mgr_ns,
        "FEISHU_PERM_YOLO": "1",
    }

    plan_timeout = min(_cfg.RESPONSE_TIMEOUT, 120)

    # Acquire process semaphore so Manager counts against the global AI
    # subprocess limit. Capture the exact sem instance so the matching
    # release in the finally clause hits the same object even if
    # _init_ai_sem rebuilds the global mid-flight.
    _crew_sem = get_ai_sem()
    if not _crew_sem.acquire(timeout=plan_timeout):
        _debug_log("[Crew] Manager: timed out waiting for AI process slot")
        return None

    try:
        try:
            proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, cwd=cwd, env=env,
            )
        except FileNotFoundError:
            _debug_log("[Crew] Manager: Claude CLI not found")
            return None

        try:
            proc.stdin.write(prompt + "\n")
            proc.stdin.close()
        except OSError:
            proc.kill()
            return None

        _stderr_buf: list[str] = []
        def _drain_stderr():
            for ln in proc.stderr:
                if len(_stderr_buf) < 50:
                    _stderr_buf.append(ln.rstrip())
        threading.Thread(target=_drain_stderr, daemon=True).start()
        _debug_log(f"[Crew] Manager: planning started pid={proc.pid} timeout={plan_timeout}s")

        # Hard deadline enforced by a timer thread — guards against claude producing no output at all,
        # which would cause `for line in proc.stdout` to block indefinitely.
        def _hard_kill():
            proc.kill()
            stderr_preview = " | ".join(_stderr_buf[-3:]) if _stderr_buf else ""
            _debug_log(f"[Crew] Manager: planning timed out (hard kill){'; stderr: ' + stderr_preview if stderr_preview else ''}")
        _timer = threading.Timer(plan_timeout, _hard_kill)
        _timer.daemon = True
        _timer.start()

        # Collect all text output; cap total accumulated chars to avoid large-string mmap pressure
        _TEXT_BUF_MAX_CHARS = 200_000
        text_buf: list[str] = []
        text_buf_len = 0
        try:
            for line in proc.stdout:
                if cancel_ev.is_set():
                    proc.kill()
                    return None
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Collect text content
                if ev.get("type") == "assistant":
                    for block in ev.get("message", {}).get("content", []) or []:
                        if block.get("type") == "text":
                            chunk = block.get("text", "")
                            if chunk and text_buf_len < _TEXT_BUF_MAX_CHARS:
                                text_buf.append(chunk)
                                text_buf_len += len(chunk)
                if ev.get("type") == "result":
                    result_val = ev.get("result", "")
                    if result_val and text_buf_len < _TEXT_BUF_MAX_CHARS:
                        text_buf.append(result_val)
                    break
        finally:
            _timer.cancel()

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()  # always reap zombie regardless of kill outcome

    finally:
        _crew_sem.release()

    full_text = "\n".join(text_buf)

    # Extract ```json ... ``` code block from text
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", full_text)
    if not m:
        # Try to find the outermost JSON object directly
        m = re.search(r"(\{[\s\S]*\"agents\"[\s\S]*\})", full_text)
    if not m:
        stderr_preview = " | ".join(_stderr_buf[-5:]) if _stderr_buf else ""
        _debug_log(f"[Crew] Manager: no JSON plan found, output: {full_text[:200]!r}"
                   + (f", stderr: {stderr_preview}" if stderr_preview else ""))
        return None

    try:
        plan_input = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        _debug_log(f"[Crew] Manager: JSON parse failed: {e}")
        return None

    # Parse and validate
    try:
        raw_agents = plan_input.get("agents", [])
        if not raw_agents:
            raise ValueError("agents is empty")

        # Dependency cycle detection
        cycle = _detect_cycle(raw_agents)
        if cycle:
            _debug_log(f"[Crew] Manager: dependency cycle: {cycle}")
            return None

        agents = [
            AgentSpec(
                id=a["id"], role=a["role"], model=a.get("model", "claude"),
                system=a.get("system", ""),
                prompt=a["prompt"],
                depends_on=a.get("depends_on", []),
                timeout=min(max(int(a.get("timeout", _cfg.RESPONSE_TIMEOUT)), 60),
                            _cfg.HARD_TIMEOUT // 2),
                output_file=a.get("output_file", ""),
            )
            for a in raw_agents[:max_agents]
        ]
        return CrewPlan(
            title=plan_input.get("title", requirement[:30]),
            agents=agents,
            synthesis_prompt=plan_input.get("synthesis_prompt", ""),
        )
    except Exception as e:
        _debug_log(f"[Crew] Manager: plan parse failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  Command entry points (called by bridge.py)
# ═══════════════════════════════════════════════════════════════

def _cmd_crew_status(chat_id: str):
    """Display the current crew/dev task status, or notify that no task is running."""
    from larkhelm.log import _read_logs
    from larkhelm.card_builder import _fmt_elapsed
    from larkhelm.lark_client import send_card
    from larkhelm.crew_card import _STATUS_ICON
    from larkhelm.crew._state import _active_crew_lock, _active_crew_states

    with _active_crew_lock:
        state = _active_crew_states.get(chat_id)
    if state:
        elapsed = _fmt_elapsed(time.time() - state.start_time)
        n_done  = sum(1 for a in state.agents.values() if a.status == AgentStatus.DONE)
        n_total = len(state.agents)
        lines   = [f"**任务：** {state.plan.title}",
                   f"**状态：** {state.phase}  · {n_done}/{n_total} Agent 完成  · 已耗时 {elapsed}"]
        for spec in state.plan.agents:
            ag   = state.agents[spec.id]
            icon = _STATUS_ICON.get(ag.status, "?")
            lines.append(f"{icon} {spec.role}（{spec.model}）")
        send_card(chat_id, "⚙️ Crew 任务进行中", "\n".join(lines), color="blue")
    else:
        # Look for the most recent crew record in the logs
        records = _read_logs(chat_id)
        last_crew = next((r for r in reversed(records)
                          if r.get("model") == "crew" and r["role"] == "assistant"), None)
        if last_crew:
            ts  = last_crew["ts"][:16].replace("T", " ")
            snip = last_crew["content"][:120].replace("\n", " ")
            send_card(chat_id, "📋 最近 Crew 任务",
                      f"**完成时间：** {ts}\n**摘要：** {snip}…\n\n"
                      f"发 `/crew <需求>` 开始新任务",
                      color="blue")
        else:
            send_card(chat_id, "📋 Crew 状态",
                      "当前没有正在运行的 crew 任务。\n\n"
                      "发 `/crew <需求>` 或 `/dev <需求>` 开始。",
                      color="blue")


def cmd_crew(chat_id: str, args_str: str, user_msg_id: str = None):
    """/crew command entry point, runs in a daemon thread."""
    requirement = args_str.strip()
    if not requirement or requirement == "status":
        _cmd_crew_status(chat_id)
        return

    max_agents    = CREW_MAX_AGENTS
    total_timeout = None  # will be computed inside

    import larkhelm.config as _cfg
    total_timeout = _cfg.RESPONSE_TIMEOUT * 12

    # Simple argument parsing
    if requirement.startswith("--agents "):
        parts = requirement.split(None, 2)
        if len(parts) >= 3:
            try:
                max_agents  = max(2, min(int(parts[1]), CREW_MAX_AGENTS))
                requirement = parts[2]
            except ValueError:
                pass

    _run_generic_crew(chat_id, requirement, max_agents, total_timeout, user_msg_id)


def cmd_dev(chat_id: str, args_str: str, user_msg_id: str = None):
    """/dev command entry point, fixed software engineering pipeline."""
    from larkhelm.lark_client import send_card

    args = args_str.strip()
    no_confirm = False
    if args.startswith("--no-confirm"):
        no_confirm = True
        args = args[len("--no-confirm"):].strip()

    requirement = args
    if not requirement:
        send_card(chat_id, "⚠️ 用法",
                  "`/dev <需求描述 或 飞书文档URL>`\n\n"
                  "软件工程流水线：PM **[确认]** → 架构师 → 工程师 → QA（失败重试 2×）→ 审查员（重试 1×）\n\n"
                  "加 `--no-confirm` 可跳过 PM 后的人工确认断点，直接连续执行。",
                  color="orange")
        return
    # If requirement contains a Feishu doc URL, read and inline the document content.
    # task_hash is based on the original `args` string so workspace stale-detection
    # remains stable; if the doc changes, pass --no-confirm to force a fresh plan.
    expanded_requirement = _expand_doc_requirement(requirement)
    _run_dev_crew(chat_id, expanded_requirement, user_msg_id, no_confirm=no_confirm,
                  force_replan=no_confirm,  # --no-confirm always re-runs planning
                  task_key=requirement)     # stale-detection key: original args, not expanded content


def _run_generic_crew(chat_id: str, requirement: str,
                      max_agents: int, total_timeout: int,
                      user_msg_id: str | None):
    """Generic crew: Manager dynamically plans a DAG."""
    crew_id   = uuid.uuid4().hex[:12]
    _register_crew_thread(crew_id, threading.current_thread())
    try:
        _run_generic_crew_inner(chat_id, requirement, max_agents, total_timeout,
                                user_msg_id, crew_id)
    finally:
        _unregister_crew_thread(crew_id)
        try:
            from larkhelm.token_stats import evict_crew_agent_tokens
            evict_crew_agent_tokens(f"{chat_id}__crew_{crew_id}")
        except Exception as e:
            _debug_log(f"[Crew] token eviction failed: {e}")


def _run_generic_crew_inner(chat_id: str, requirement: str,
                             max_agents: int, total_timeout: int,
                             user_msg_id: str, crew_id: str):
    """Actual implementation of _run_generic_crew (crew_id already generated by the outer call).

    Phase C wraps the original body in an outer ``try/except Exception`` that
    forwards any unexpected escape to ``emit_terminal_failure`` so users see
    a ⚠️ card instead of silent failure.
    """
    from larkhelm.crew._failure_card import emit_terminal_failure
    try:
        return _run_generic_crew_inner_impl(
            chat_id, requirement, max_agents, total_timeout, user_msg_id, crew_id,
        )
    except Exception as e:
        _debug_log(f"[Crew] _run_generic_crew_inner uncaught: {e}")
        emit_terminal_failure(chat_id, kind="crew",
                              reason="未捕获异常导致任务终止", exc=e)
        raise


def _run_generic_crew_inner_impl(chat_id: str, requirement: str,
                                  max_agents: int, total_timeout: int,
                                  user_msg_id: str, crew_id: str):
    """Original body of _run_generic_crew_inner; see wrapper above for the
    Phase C terminal-failure plumbing."""
    import larkhelm.config as _cfg
    from larkhelm.concurrency import _get_cancel_event
    from larkhelm.chat_state import _get_cwd
    from larkhelm.lark_client import _reply_card_raw, _send_card_raw, _patch_card_raw, _pin_task_card, send_card
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _active_crew_states,
        _git_head, clear_recent_crew_context, _signal_crew_done,
    )
    from larkhelm.crew._runner import _run_crew

    cancel_ev = _get_cancel_event(chat_id)
    cwd       = _get_cwd(chat_id)

    with _active_crew_lock:
        if chat_id in _active_crew:
            send_card(chat_id, "⚠️ Crew 已在运行",
                      "当前已有 crew 任务在运行，发送 `/cancel` 停止后再试。",
                      color="orange")
            return
        _active_crew[chat_id] = crew_id
        # Clear any stale cancel signal carried over from the *previous*
        # crew on this chat. Abnormal exits (breakpoint timeout, hard fail,
        # SIGTERM) call ``state.cancel_ev.set()`` but the finally cleanup
        # below never clears the per-chat Event — so the next /crew or /dev
        # inherits the set state and ``_execute`` raises QueryCancelledError
        # on the very first wave check, masquerading as an instant cancel.
        # We hold the active_crew slot, so no concurrent crew can race here.
        cancel_ev.clear()

    log_entry(chat_id, "user", requirement, model="crew")
    clear_recent_crew_context(chat_id)

    try:
        # Clear leftover workspace files from previous /crew runs so agents don't read stale data.
        _clear_workspace(Path(cwd) / ".crew_workspace")

        from larkhelm.card_builder import _make_card
        init_card = _make_card(
            "🧠 Crew · 规划中",
            f"**需求：** {requirement[:100]}\n\nManager 正在分析需求，生成任务计划…",
            color="grey",
        )
        if user_msg_id:
            card_mid = _reply_card_raw(user_msg_id, init_card, in_thread=False)
        else:
            card_mid = _send_card_raw(chat_id, init_card)
        if card_mid:
            _pin_task_card(chat_id, card_mid)

        # Manager planning
        plan = _crew_plan(chat_id, requirement, cwd, max_agents, cancel_ev)
        if plan is None:
            if not cancel_ev.is_set():
                # Fallback: single agent
                _debug_log("[Crew] Manager planning failed, falling back to single agent")
                plan = CrewPlan(
                    title=requirement[:30],
                    agents=[AgentSpec(
                        id="agent_1", role="通用助手", model="claude",
                        system="", prompt=requirement,
                        depends_on=[], timeout=_cfg.RESPONSE_TIMEOUT,
                    )],
                    synthesis_prompt="",
                )
            else:
                if card_mid:
                    _patch_card_raw(card_mid, _make_card("🛑 Crew 已取消", "", color="orange"))
                return

        # Initialize CrewState
        state = CrewState(
            crew_id=crew_id, chat_id=chat_id, plan=plan,
            agents={spec.id: AgentState(spec=spec) for spec in plan.agents},
            card_mid=card_mid, cancel_ev=cancel_ev, phase="planned",
            git_head_before=_git_head(cwd),
            trigger_msg_id=user_msg_id,
        )
        _crew_update_card(state)
        with _active_crew_lock:
            _active_crew_states[chat_id] = state

        _run_crew(state, total_timeout)

    finally:
        with _active_crew_lock:
            _active_crew.pop(chat_id, None)
            _active_crew_states.pop(chat_id, None)
            _signal_crew_done(chat_id)
        # Capture this /crew completion into session memory (debounced).
        try:
            from larkhelm.memory import record_milestone
            record_milestone(chat_id, "crew", summary=requirement)
        except Exception as _e:
            _debug_log(f"[Crew] milestone record failed: {_e}")


def _run_dev_crew(chat_id: str, requirement: str, user_msg_id: str,
                  no_confirm: bool = False, force_replan: bool = False,
                  task_key: str | None = None):
    """Fixed software engineering pipeline.
    task_key: the original (unexpanded) requirement string used for stale-detection hash.
    Defaults to requirement itself when not set (e.g. called from /plan).
    """
    crew_id = uuid.uuid4().hex[:12]
    _register_crew_thread(crew_id, threading.current_thread())
    try:
        _run_dev_crew_inner(chat_id, requirement, user_msg_id, no_confirm, crew_id,
                            force_replan=force_replan, task_key=task_key)
    finally:
        _unregister_crew_thread(crew_id)
        try:
            from larkhelm.token_stats import evict_crew_agent_tokens
            evict_crew_agent_tokens(f"{chat_id}__crew_{crew_id}")
        except Exception as e:
            _debug_log(f"[Crew] token eviction failed: {e}")


def _run_dev_crew_inner(chat_id: str, requirement: str, user_msg_id: str,
                        no_confirm: bool, crew_id: str, force_replan: bool = False,
                        task_key: str | None = None,
                        suppress_done_signal: bool = False,
                        suppress_finalize: bool = False):
    """Phase C wrapper for _run_dev_crew_inner; routes uncaught exceptions to
    ``emit_terminal_failure`` so the user sees a ⚠️ card instead of a silent
    background-thread death."""
    from larkhelm.crew._failure_card import emit_terminal_failure
    try:
        return _run_dev_crew_inner_impl(
            chat_id, requirement, user_msg_id, no_confirm, crew_id,
            force_replan=force_replan, task_key=task_key,
            suppress_done_signal=suppress_done_signal,
            suppress_finalize=suppress_finalize,
        )
    except Exception as e:
        _debug_log(f"[Crew] _run_dev_crew_inner uncaught: {e}")
        emit_terminal_failure(chat_id, kind="dev",
                              reason="未捕获异常导致任务终止", exc=e)
        raise


def _run_dev_crew_inner_impl(chat_id: str, requirement: str, user_msg_id: str,
                              no_confirm: bool, crew_id: str, force_replan: bool = False,
                              task_key: str | None = None,
                              suppress_done_signal: bool = False,
                              suppress_finalize: bool = False):
    """Actual implementation of _run_dev_crew (crew_id already generated by the outer call).
    task_key: original (unexpanded) requirement used for workspace stale-detection.
    suppress_finalize: True when invoked as a /plan [dev] step — /plan emits its
        own finalize card after the whole multi-step run; without this flag the
        user would see a workspace summary card per [dev] step PLUS a final one
        from /plan (cosmetic redundancy, no data impact).
    """
    import larkhelm.config as _cfg
    from larkhelm.concurrency import _get_cancel_event
    from larkhelm.chat_state import _get_cwd
    from larkhelm.lark_client import _reply_card_raw, _send_card_raw, _pin_task_card, send_card
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _active_crew_states,
        _git_head, clear_recent_crew_context, _signal_crew_done,
    )
    from larkhelm.crew._pipeline import _make_dev_pipeline
    from larkhelm.crew._runner import _run_crew

    cancel_ev = _get_cancel_event(chat_id)
    cwd       = _get_cwd(chat_id)

    with _active_crew_lock:
        if chat_id in _active_crew:
            send_card(chat_id, "⚠️ Crew 已在运行",
                      "当前已有 crew 任务在运行，发送 `/cancel` 停止后再试。",
                      color="orange")
            return
        _active_crew[chat_id] = crew_id
        # Clear any stale cancel signal carried over from the *previous*
        # crew on this chat. Abnormal exits (breakpoint timeout, hard fail,
        # SIGTERM) call ``state.cancel_ev.set()`` but the finally cleanup
        # below never clears the per-chat Event — so the next /dev or /crew
        # inherits the set state and ``_execute`` raises QueryCancelledError
        # on the very first wave check, masquerading as an instant cancel.
        # We hold the active_crew slot, so no concurrent crew can race here.
        cancel_ev.clear()

    log_entry(chat_id, "user", requirement, model="crew")
    clear_recent_crew_context(chat_id)

    try:
        ws_path   = Path(cwd) / ".crew_workspace"
        # Use task_key (original, unexpanded requirement) for stale-detection so that the hash
        # is stable even when a Feishu doc URL gets its content inlined into `requirement`.
        task_hash = _task_hash(task_key if task_key is not None else requirement)
        meta      = _read_workspace_meta(ws_path)

        # Clear workspace when:
        #   1. task_hash differs (different task entirely), OR
        #   2. previous run completed successfully (APPROVED), OR
        #   3. workspace_meta.json hasn't been touched in >24h (stale-TTL guard).
        # Rationale for (3): the resume-on-interrupt semantics ("same hash + uncompleted
        # → reuse design.md/tasks.json") was assuming the user comes back within minutes
        # to hours. After 24h the workspace artifacts are usually unrelated to whatever
        # the user is now asking, and silently feeding stale prd.md/design.md to the
        # implementer caused real cross-task contamination (see /chat-planning bug).
        meta_path = ws_path / "workspace_meta.json"
        is_stale_age = False
        if meta and meta_path.exists():
            try:
                age_sec = time.time() - meta_path.stat().st_mtime
                is_stale_age = age_sec > _WORKSPACE_STALE_TTL
            except OSError as _e:
                _debug_log(f"[Dev] workspace_meta stat failed: {_e}")
        if meta and (
            meta.get("task_hash") != task_hash
            or meta.get("completed")
            or is_stale_age
        ):
            if is_stale_age:
                _debug_log(f"[Dev] clearing stale workspace (age > {_WORKSPACE_STALE_TTL}s)")
            _clear_workspace(ws_path)
            meta = {}

        # force_replan=True (--no-confirm) always runs full pipeline.
        has_design    = (ws_path / "design.md").exists() and (ws_path / "tasks.json").exists()
        skip_planning = bool(meta) and has_design and not force_replan

        if not meta:
            _write_workspace_meta(ws_path, task_hash=task_hash, completed=False)

        # Build augmented requirement that includes recent chat turns + global/project
        # memory so PM agent isn't context-blind. WITHOUT this, /dev only saw the
        # literal command-line string, and references like "实现刚才讨论的方案" had no
        # anchor — PM would either probe filesystem (and risk reading stale workspace
        # artifacts) or hallucinate a task. ``task_key`` (used for hash above) deliberately
        # stays the original literal so resume semantics stay stable.
        augmented_requirement = _augment_requirement_with_context(requirement, chat_id, cwd)

        plan = _make_dev_pipeline(augmented_requirement, cwd, no_confirm=no_confirm,
                                  skip_planning=skip_planning)

        if skip_planning:
            flow_desc = "工程师 → QA（失败重试 2×）→ 审查员（重试 1×）\n\n💡 检测到已有 design.md + tasks.json，跳过 PM 和架构师"
        else:
            flow_desc = "产品经理 → 架构师 → 工程师 → 测试工程师 → 代码审查员"

        from larkhelm.card_builder import _make_card
        init_card = _make_card(
            f"📐 软件开发 · {plan.title}",
            f"**需求：** {requirement[:200]}\n\n**流程：** {flow_desc}",
            color="blue",
        )
        if user_msg_id:
            card_mid = _reply_card_raw(user_msg_id, init_card, in_thread=False)
        else:
            card_mid = _send_card_raw(chat_id, init_card)
        if card_mid:
            _pin_task_card(chat_id, card_mid)

        state = CrewState(
            crew_id=crew_id, chat_id=chat_id, plan=plan,
            agents={spec.id: AgentState(spec=spec) for spec in plan.agents},
            card_mid=card_mid, cancel_ev=cancel_ev, phase="planned", kind="dev",
            git_head_before=_git_head(cwd),
            trigger_msg_id=user_msg_id,
        )
        _crew_update_card(state)
        with _active_crew_lock:
            _active_crew_states[chat_id] = state

        # /dev total timeout: worst case pm+arch+impl+(fixer+qa)×2+reviewer ≈ 8 roles
        # pm/arch/qa/reviewer ~20min each, impl/fixer ~6h each; estimated 2.5h covers the full pipeline
        total_timeout = _cfg.RESPONSE_TIMEOUT * 28
        _run_crew(state, total_timeout)

        # Mark workspace completed only after APPROVED review so the next /dev for the same
        # requirement starts fresh. On REJECTED or interrupted, completed=False is kept so
        # that the next run can skip PM/architect and continue from where it left off.
        # Shared implementation with /plan (``workspace_finalize`` module): also emits a
        # Feishu summary card with a copy-paste-able ``git add`` / ``git commit`` hint.
        # title=requirement[:80] so the suggested ``git commit -m <title>`` line stays
        # within Feishu's card-title-friendly length.
        # When invoked as a /plan [dev] step, /plan emits the workspace summary
        # card itself after the full multi-step run completes — skip here to
        # avoid one card per step plus a duplicate final one.
        if not suppress_finalize:
            try:
                from larkhelm.workspace_finalize import finalize_workspace
                finalize_workspace(chat_id, requirement[:80], kind="dev")
            except Exception as _fe:
                _debug_log(f"[Dev] workspace finalisation failed: {_fe}")

    finally:
        with _active_crew_lock:
            _active_crew.pop(chat_id, None)
            _active_crew_states.pop(chat_id, None)
            if not suppress_done_signal:
                _signal_crew_done(chat_id)
        # Capture this /dev completion into session memory immediately
        # rather than waiting for the next ``maybe_auto_update`` triggered
        # from a chat turn. Failures swallowed inside the helper.
        try:
            from larkhelm.memory import record_milestone
            record_milestone(chat_id, "dev", summary=requirement)
        except Exception as _e:
            _debug_log(f"[Dev] milestone record failed: {_e}")


def immediate_cancel_crew(chat_id: str) -> bool:
    """Called immediately when the user clicks the cancel button: removes card buttons and shows
    'cancelling' status without waiting for agents to actually stop. _run_crew will update the
    card to the final 'cancelled' state once it receives InterruptedError.
    """
    from larkhelm.card_builder import _fmt_elapsed
    from larkhelm.lark_client import _patch_card_raw
    from larkhelm.crew._state import _active_crew_lock, _active_crew_states

    with _active_crew_lock:
        state = _active_crew_states.get(chat_id)
    if not state or not state.card_mid:
        return False
    try:
        from larkhelm.card_builder import _make_card
        elapsed = _fmt_elapsed(time.time() - state.start_time)
        _label  = "Dev" if state.kind == "dev" else "Crew"
        _patch_card_raw(state.card_mid, _make_card(
            f"🛑 {_label} 取消中… ({elapsed})",
            f"**{state.plan.title}**\n\n正在等待运行中的 Agent 退出…",
            color="orange",
        ))
        return True
    except Exception as e:
        _debug_log(f"[Cancel] immediate card update failed: {e}")
        return False


def cancel_all_crews(reason: str = "服务重启"):
    """Send cancel signals to all active crews and update their cards. Called by the SIGTERM handler."""
    from larkhelm.card_builder import _fmt_elapsed
    from larkhelm.lark_client import _patch_card_raw
    from larkhelm.crew._state import _active_crew_lock, _active_crew_states
    from larkhelm.crew._checkpoint import _save_checkpoint

    with _active_crew_lock:
        states = list(_active_crew_states.values())

    for state in states:
        _debug_log(f"[Shutdown] cancelling crew {state.crew_id[:8]} chat={state.chat_id}")
        # Save checkpoint first (records currently completed agents) so the service can resume after restart
        try:
            with state.lock:
                completed_ids = [
                    ag_id for ag_id, ag in state.agents.items()
                    if ag.status in (AgentStatus.DONE, AgentStatus.FAILED)
                ]
                state.phase = "running"   # Keep as running so resume logic re-runs incomplete parts
            _save_checkpoint(state, completed_ids)
        except Exception as e:
            _debug_log(f"[Shutdown] checkpoint save failed: {e}")

        # Send cancel signal after checkpoint is saved, to avoid persisting cancelled state
        state.cancel_ev.set()
        try:
            from larkhelm.card_builder import _make_card
            elapsed = _fmt_elapsed(time.time() - state.start_time)
            _patch_card_raw(state.card_mid, _make_card(
                f"🛑 Crew 已中断（{elapsed}）",
                (f"⚠️ **{reason}**，服务重启后将自动从断点恢复。\n\n"
                 f"**任务：** {state.plan.title}\n\n"
                 f"已完成 {len(completed_ids)} 个 Agent，断点已保存。"),
                color="orange",
            ))
        except Exception as e:
            _debug_log(f"[Shutdown] cancel card update failed: {e}")


def pause_crew(crew_id: str) -> bool:
    """Called when the user clicks ⏸: pause the specified crew, save checkpoint, update card."""
    from larkhelm.card_builder import _fmt_elapsed
    from larkhelm.lark_client import _patch_card_raw
    from larkhelm.crew._state import _active_crew_lock, _active_crew_states
    from larkhelm.crew._checkpoint import _save_checkpoint

    with _active_crew_lock:
        state = next(
            (s for s in _active_crew_states.values() if s.crew_id == crew_id),
            None,
        )
    if not state:
        _debug_log(f"[Pause] crew_id={crew_id[:8]} not found")
        return False

    _debug_log(f"[Pause] pausing crew={crew_id[:8]} chat={state.chat_id}")

    # Collect completed agents first, then send cancel signal
    with state.lock:
        completed_ids = [
            ag_id for ag_id, ag in state.agents.items()
            if ag.status in (AgentStatus.DONE, AgentStatus.FAILED)
        ]
        state.phase = "paused"

    # Send cancel signal so running agents exit on their own
    state.cancel_ev.set()

    # Save checkpoint (phase="paused", distinct from service-restart "running", but resume logic is the same)
    _save_checkpoint(state, completed_ids)

    # Update card
    from larkhelm.card_builder import _make_card
    _label  = "Dev" if state.kind == "dev" else "Crew"
    elapsed = _fmt_elapsed(time.time() - state.start_time)
    try:
        _patch_card_raw(state.card_mid, _make_card(
            f"⏸ {_label} 已暂停 ({elapsed})",
            (f"**{state.plan.title}**\n\n"
             f"断点已保存（已完成 {len(completed_ids)} 个 Agent）。\n\n"
             f"服务重启后将自动从断点继续执行。"),
            color="yellow",
        ))
    except Exception as e:
        _debug_log(f"[Pause] card update failed: {e}")

    return True


def wait_crews_done(timeout: float = 30.0) -> bool:
    """Wait for all crew threads to exit, up to timeout seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _crew_threads_lock:
            alive = [t for t in _crew_threads.values() if t.is_alive()]
        if not alive:
            return True
        time.sleep(0.5)
    return False
