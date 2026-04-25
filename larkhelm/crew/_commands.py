"""
larkhelm · Crew public command functions and Manager planning
"""
from __future__ import annotations

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
      "model": "claude",
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
5. model：可以是 "claude"、"gemini"、"kimi"，或 Hermes 编排模式 "hermes_race"、"hermes_split"、"hermes_review"
6. timeout：秒，范围 60 到 {max_timeout}
7. prompt 中可使用 {{agent_N_result}} 引用上游 agent 的输出摘要
8. system 字段：该 agent 的角色定义，指导其行为风格
9. output_file：agent 的主要输出文件名（相对于 .crew_workspace/），下游 agent 应优先读取该文件而非依赖摘要传递

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
  - 示例：{{"task": "实现支付模块", "agents": ["claude", "kimi", "gemini"]}}

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

## 模型选择指南

根据任务特征选择合适的模型，尽量均衡分配，避免所有 agent 都用同一模型：

- **claude** 擅长：代码生成与修改（需调用 Write/Edit 工具）、需求分析、复杂多步推理、长上下文理解、设计决策
- **gemini** 擅长：信息搜索与收集（WebSearch/WebFetch）、代码审查与静态分析（只读文件）、内容摘要与整合、并行独立调研子任务
- **kimi** 擅长：大文件处理、长上下文理解、文档生成
- **hermes_race** 擅长：紧急任务多 agent 竞争，取最快结果
- **hermes_split** 擅长：全栈开发前后端并行
- **hermes_review** 擅长：核心模块的完整实现→审查→测试流水线

## 自动模式选择规则

当需求包含以下关键词时，自动选择对应 Hermes 编排模式：

- **hermes_race**（竞争模式）：需求含 "紧急"、"快速"、"尽快"、"临时"、"修复"、"bug"、"比选"、"对比"
- **hermes_split**（分工模式）：需求含 "全栈"、"前后端"、"前端"、"后端"、"API"、"页面"、"UI"
- **hermes_review**（评审模式）：需求含 "核心"、"关键"、"安全"、"支付"、"认证"、"审计"、"严格审查"

如果需求不含以上关键词，使用传统单 agent 模式（claude/gemini）。

典型分配示例：
- 写代码 → claude；审查代码 → gemini 或 hermes_review
- 设计方案 → claude；调研竞品/文档 → gemini
- 多个并行调研子任务 → 各自用 gemini；汇总整合 → claude
- 分析设计（claude）→ 并行实现多模块（claude×N 或 hermes_split）→ 并行审查（gemini×N 或 hermes_review）→ 汇总（claude）
- 紧急修复 → hermes_race（claude+kimi 竞争）

## 当前项目目录

{cwd}

## 用户需求

{requirement}
"""


def _crew_plan(chat_id: str, requirement: str, cwd: str,
               max_agents: int, cancel_ev: threading.Event) -> CrewPlan:
    """Call the Manager LLM (Claude tool_use) to generate a task plan. Returns None on failure."""
    import larkhelm.config as _cfg
    from larkhelm.crew._scheduler import _detect_cycle

    mgr_ns   = f"{chat_id}__crew_mgr_{uuid.uuid4().hex[:8]}"
    prompt   = _MANAGER_PROMPT_TPL.format(
        max_agents=max_agents,
        max_timeout=_cfg.HARD_TIMEOUT // 2,
        cwd=cwd,
        requirement=requirement,
    )

    import os as _os

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

    try:
        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=cwd, env=env,
        )
    except FileNotFoundError:
        _debug_log("[Crew/Manager] Claude CLI not found")
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
            _stderr_buf.append(ln.rstrip())
    threading.Thread(target=_drain_stderr, daemon=True).start()
    _debug_log(f"[Crew/Manager] planning started pid={proc.pid} timeout={min(_cfg.RESPONSE_TIMEOUT, 120)}s")

    # Hard deadline enforced by a timer thread — guards against claude producing no output at all,
    # which would cause `for line in proc.stdout` to block indefinitely.
    plan_timeout = min(_cfg.RESPONSE_TIMEOUT, 120)
    def _hard_kill():
        proc.kill()
        stderr_preview = " | ".join(_stderr_buf[-3:]) if _stderr_buf else ""
        _debug_log(f"[Crew/Manager] planning timed out (hard kill){'; stderr: ' + stderr_preview if stderr_preview else ''}")
    _timer = threading.Timer(plan_timeout, _hard_kill)
    _timer.daemon = True
    _timer.start()

    # Collect all text output
    text_buf: list[str] = []
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
                        text_buf.append(block.get("text", ""))
            if ev.get("type") == "result":
                if ev.get("result"):
                    text_buf.append(ev["result"])
                break
    finally:
        _timer.cancel()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    full_text = "\n".join(text_buf)

    # Extract ```json ... ``` code block from text
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", full_text)
    if not m:
        # Try to find the outermost JSON object directly
        m = re.search(r"(\{[\s\S]*\"agents\"[\s\S]*\})", full_text)
    if not m:
        stderr_preview = " | ".join(_stderr_buf[-5:]) if _stderr_buf else ""
        _debug_log(f"[Crew/Manager] no JSON plan found, output: {full_text[:200]!r}"
                   + (f", stderr: {stderr_preview}" if stderr_preview else ""))
        return None

    try:
        plan_input = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        _debug_log(f"[Crew/Manager] JSON parse failed: {e}")
        return None

    # Parse and validate
    try:
        raw_agents = plan_input.get("agents", [])
        if not raw_agents:
            raise ValueError("agents is empty")

        # Dependency cycle detection
        cycle = _detect_cycle(raw_agents)
        if cycle:
            _debug_log(f"[Crew/Manager] dependency cycle: {cycle}")
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
        _debug_log(f"[Crew/Manager] plan parse failed: {e}")
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
                  "`/dev <需求描述>`\n\n"
                  "软件工程流水线：PM **[确认]** → 架构师 → 工程师 → QA（失败重试 2×）→ 审查员（重试 1×）\n\n"
                  "加 `--no-confirm` 可跳过 PM 后的人工确认断点，直接连续执行。",
                  color="orange")
        return
    _run_dev_crew(chat_id, requirement, user_msg_id, no_confirm=no_confirm)


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


def _run_generic_crew_inner(chat_id: str, requirement: str,
                             max_agents: int, total_timeout: int,
                             user_msg_id: str, crew_id: str):
    """Actual implementation of _run_generic_crew (crew_id already generated by the outer call)."""
    import larkhelm.config as _cfg
    from larkhelm.concurrency import _get_cancel_event
    from larkhelm.chat_state import _get_cwd
    from larkhelm.lark_client import _reply_card_raw, _send_card_raw, _patch_card_raw, _pin_task_card, send_card
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _active_crew_states,
        _git_head, clear_recent_crew_context,
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

    log_entry(chat_id, "user", requirement, model="crew")
    clear_recent_crew_context(chat_id)

    try:
        # Send initial card
        init_card = json.dumps({
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {"template": "grey",
                       "title": {"tag": "plain_text", "content": "🧠 Crew · 规划中"}},
            "body": {"elements": [
                {"tag": "markdown", "content": f"**需求：** {requirement[:100]}"},
                {"tag": "markdown", "content": "Manager 正在分析需求，生成任务计划…"},
            ]},
        }, ensure_ascii=False)
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
                    _patch_card_raw(card_mid, json.dumps({
                        "schema": "2.0",
                        "header": {"template": "orange",
                                   "title": {"tag": "plain_text", "content": "🛑 Crew 已取消"}},
                        "body": {"elements": []},
                    }, ensure_ascii=False))
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


def _run_dev_crew(chat_id: str, requirement: str, user_msg_id: str,
                  no_confirm: bool = False):
    """Fixed software engineering pipeline."""
    crew_id = uuid.uuid4().hex[:12]
    _register_crew_thread(crew_id, threading.current_thread())
    try:
        _run_dev_crew_inner(chat_id, requirement, user_msg_id, no_confirm, crew_id)
    finally:
        _unregister_crew_thread(crew_id)


def _run_dev_crew_inner(chat_id: str, requirement: str, user_msg_id: str,
                        no_confirm: bool, crew_id: str):
    """Actual implementation of _run_dev_crew (crew_id already generated by the outer call)."""
    import larkhelm.config as _cfg
    from larkhelm.concurrency import _get_cancel_event
    from larkhelm.chat_state import _get_cwd
    from larkhelm.lark_client import _reply_card_raw, _send_card_raw, _pin_task_card, send_card
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _active_crew_states,
        _git_head, clear_recent_crew_context,
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

    log_entry(chat_id, "user", requirement, model="crew")
    clear_recent_crew_context(chat_id)

    try:
        # Detect whether design.md + tasks.json already exist in .crew_workspace/
        # If so, skip pm/architect and start directly from implementer (continues from a prior /crew analysis)
        ws_path    = Path(cwd) / ".crew_workspace"
        has_design = (ws_path / "design.md").exists() and (ws_path / "tasks.json").exists()
        skip_planning = has_design and not no_confirm  # --no-confirm forces the full pipeline

        plan = _make_dev_pipeline(requirement, cwd, no_confirm=no_confirm,
                                  skip_planning=skip_planning)

        if skip_planning:
            flow_desc = "工程师 → QA（失败重试 2×）→ 审查员（重试 1×）\n\n💡 检测到已有 design.md + tasks.json，跳过 PM 和架构师"
        else:
            flow_desc = "产品经理 → 架构师 → 工程师 → 测试工程师 → 代码审查员"

        init_card = json.dumps({
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {"template": "blue",
                       "title": {"tag": "plain_text",
                                 "content": f"📐 软件开发 · {plan.title}"}},
            "body": {"elements": [
                {"tag": "markdown",
                 "content": f"**需求：** {requirement[:200]}\n\n**流程：** {flow_desc}"},
            ]},
        }, ensure_ascii=False)
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

    finally:
        with _active_crew_lock:
            _active_crew.pop(chat_id, None)
            _active_crew_states.pop(chat_id, None)


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
        elapsed = _fmt_elapsed(time.time() - state.start_time)
        _label  = "Dev" if state.kind == "dev" else "Crew"
        _patch_card_raw(state.card_mid, json.dumps({
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "orange",
                "title": {"tag": "plain_text",
                          "content": f"🛑 {_label} 取消中… ({elapsed})"},
            },
            "body": {"elements": [{"tag": "markdown",
                "content": (
                    f"**{state.plan.title}**\n\n"
                    "正在等待运行中的 Agent 退出…"
                ),
            }]},
        }, ensure_ascii=False))
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
            elapsed = _fmt_elapsed(time.time() - state.start_time)
            _patch_card_raw(state.card_mid, json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"template": "orange",
                           "title": {"tag": "plain_text",
                                     "content": f"🛑 Crew 已中断（{elapsed}）"}},
                "body": {"elements": [{"tag": "markdown",
                    "content": (
                        f"⚠️ **{reason}**，服务重启后将自动从断点恢复。\n\n"
                        f"**任务：** {state.plan.title}\n\n"
                        f"已完成 {len(completed_ids)} 个 Agent，断点已保存。"
                    ),
                }]},
            }, ensure_ascii=False))
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
    _label  = "Dev" if state.kind == "dev" else "Crew"
    elapsed = _fmt_elapsed(time.time() - state.start_time)
    try:
        _patch_card_raw(state.card_mid, json.dumps({
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {"template": "yellow",
                       "title": {"tag": "plain_text",
                                 "content": f"⏸ {_label} 已暂停 ({elapsed})"}},
            "body": {"elements": [{"tag": "markdown",
                "content": (
                    f"**{state.plan.title}**\n\n"
                    f"断点已保存（已完成 {len(completed_ids)} 个 Agent）。\n\n"
                    f"服务重启后将自动从断点继续执行。"
                ),
            }]},
        }, ensure_ascii=False))
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
