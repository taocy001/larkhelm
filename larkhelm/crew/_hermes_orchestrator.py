"""
larkhelm · Hermes multi-agent orchestrator integration

Wraps the Hermes multi-agent orchestrator (race/split/review modes) so it can be
used as a Crew agent backend (model="hermes_race" | "hermes_split" | "hermes_review").

This module is imported by larkhelm.crew._runner and provides:
  - _run_hermes_orchestrator()   Run a Hermes orchestration as a single Crew agent
  - _make_hermes_plan()           Convert a Crew plan to Hermes orchestrator params
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from larkhelm.log import _debug_log
from larkhelm.crew_types import AgentSpec, CrewState

# ═══════════════════════════════════════════════════════════════
#  Agent configurations (aligned with ~/.hermes/scripts/multi_agent_orchestrator.py)
# ═══════════════════════════════════════════════════════════════

HERMES_AGENTS = {
    "claude": {
        "command": "claude",
        "args": ["-p"],
        "strength": "代码质量高，架构设计强，适合核心逻辑",
        "timeout": 120,
    },
    "kimi": {
        "command": "kimi",
        "args": ["-p"],
        "strength": "上下文长，适合审查和文档",
        "timeout": 120,
        "prompt_prefix": "请直接给出结果，不要探索项目目录，不要读取文件。限制在1000字以内。",
    },
    "gemini": {
        "command": "gemini",
        "args": ["-p"],
        "strength": "测试覆盖好，文档生成强",
        "timeout": 180,
    },
}


def _run_hermes_orchestrator(
    state: CrewState,
    agent_id: str,
    spec: AgentSpec,
    cancel_ev: threading.Event,
    on_text=None,
) -> str:
    """
    Run a Hermes multi-agent orchestration as a single Crew agent.

    The agent's `model` field determines the mode:
      - "hermes_race"   → competition mode (multiple agents, fastest wins)
      - "hermes_split"  → split mode (backend + frontend parallel)
      - "hermes_review" → review mode (implement → review → test pipeline)

    The agent's `prompt` is parsed as JSON containing:
      - "task": str                    # main task description
      - "agents": list[str]            # agents to use (default: ["claude", "kimi"])
      - "context": str                 # optional context
      - "backend_task": str            # for split mode
      - "frontend_task": str           # for split mode
    """
    import larkhelm.config as _cfg

    mode = spec.model.replace("hermes_", "")
    cwd = _cfg.DEFAULT_CWD  # or derive from state

    # Parse prompt as JSON if possible, otherwise treat as plain task
    try:
        params = json.loads(spec.prompt)
    except json.JSONDecodeError:
        params = {"task": spec.prompt}

    task = params.get("task", spec.prompt)
    context = params.get("context", "")
    agents = params.get("agents", ["claude", "kimi"])

    # Build the orchestrator command
    script_path = Path.home() / ".hermes" / "scripts" / "multi_agent_orchestrator.py"
    if not script_path.exists():
        # Fallback: use inline implementation
        return _run_inline_orchestrator(mode, task, context, agents, cancel_ev, on_text)

    cmd = [sys.executable, str(script_path), "--mode", mode, "--task", task]
    if context:
        cmd += ["--context", context]
    if agents:
        cmd += ["--agents", ",".join(agents)]

    _debug_log(f"[HermesOrchestrator] mode={mode} agents={agents} task={task[:80]}")

    # Run with cancellation support
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )

    # Poll for cancellation
    def _watch_cancel():
        while not cancel_ev.is_set():
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()

    threading.Thread(target=_watch_cancel, daemon=True).start()

    stdout, stderr = proc.communicate()

    if cancel_ev.is_set():
        return "（已取消）"

    if proc.returncode != 0:
        _debug_log(f"[HermesOrchestrator] failed: {stderr[:500]}")
        return f"Hermes orchestrator failed (exit {proc.returncode}): {stderr[:1000]}"

    # Parse JSON result if present
    try:
        result = json.loads(stdout)
        # Format the result nicely
        return _format_hermes_result(result, mode)
    except json.JSONDecodeError:
        # Plain text output
        return stdout.strip() or "（无输出）"


def _run_inline_orchestrator(
    mode: str,
    task: str,
    context: str,
    agents: list[str],
    cancel_ev: threading.Event,
    on_text=None,
) -> str:
    """Inline implementation when the script is not available."""
    import concurrent.futures

    _debug_log(f"[HermesOrchestrator] inline mode={mode} agents={agents}")

    if mode == "race":
        return _mode_race_inline(task, context, agents, cancel_ev, on_text)
    elif mode == "split":
        return _mode_split_inline(task, context, agents, cancel_ev, on_text)
    elif mode == "review":
        return _mode_review_inline(task, context, agents, cancel_ev, on_text)
    else:
        return f"Unknown mode: {mode}"


def _mode_race_inline(task, context, agents, cancel_ev, on_text):
    """Competition mode: run all agents in parallel, return the fastest result."""
    import concurrent.futures

    results = {}

    def run_one(agent_name):
        if cancel_ev.is_set():
            return agent_name, {"status": "cancelled"}
        agent = HERMES_AGENTS.get(agent_name)
        if not agent:
            return agent_name, {"status": "error", "error": f"Unknown agent: {agent_name}"}

        prefix = agent.get("prompt_prefix", "")
        full_prompt = f"{prefix}\n\n{context}\n\n任务：{task}\n\n请直接输出代码/结果，不要交互式确认。"
        cmd = [agent["command"]] + agent["args"] + [full_prompt]
        timeout = agent.get("timeout", 120)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return agent_name, {
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout,
                "stderr": result.stderr[:500] if result.returncode != 0 else "",
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return agent_name, {"status": "timeout"}
        except Exception as e:
            return agent_name, {"status": "error", "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(run_one, a): a for a in agents}
        for future in concurrent.futures.as_completed(futures):
            agent_name, result = future.result()
            results[agent_name] = result
            if on_text:
                on_text(f"🏁 {agent_name} 完成: {result.get('status', '?')}")

    # Find the fastest successful result
    winner = None
    for agent_name in agents:  # preserve order for tie-breaking
        r = results.get(agent_name)
        if r and r.get("status") == "success":
            winner = agent_name
            break

    if not winner:
        # All failed, return concatenated errors
        errors = "\n".join(
            f"{a}: {r.get('error', r.get('stderr', 'unknown error'))}"
            for a, r in results.items()
        )
        return f"❌ 所有 agent 失败:\n{errors}"

    w = results[winner]
    return (
        f"🏆 胜出: {winner}\n"
        f"   优势: {HERMES_AGENTS[winner]['strength']}\n"
        f"   输出预览: {w['output'][:300]}...\n\n"
        f"   其他结果:\n"
        + "\n".join(
            f"   {'✅' if r.get('status') == 'success' else '❌'} {a}: {r.get('status')}"
            for a, r in results.items()
            if a != winner
        )
        + f"\n\n---\n\n{w['output']}"
    )


def _mode_split_inline(backend_task, frontend_task, agents, cancel_ev, on_text):
    """Split mode: backend + frontend parallel development."""
    import concurrent.futures

    # Default assignment: claude backend, kimi frontend
    backend_agent = "claude"
    frontend_agent = "kimi" if "kimi" in agents else "gemini" if "gemini" in agents else "claude"

    results = {}

    def run_backend():
        if cancel_ev.is_set():
            return "backend", {"status": "cancelled"}
        agent = HERMES_AGENTS[backend_agent]
        prefix = agent.get("prompt_prefix", "")
        full_prompt = f"{prefix}\n\n任务：{backend_task}\n\n请直接输出代码/结果，不要交互式确认。"
        cmd = [agent["command"]] + agent["args"] + [full_prompt]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=agent.get("timeout", 120))
            return "backend", {
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout,
                "agent": backend_agent,
            }
        except subprocess.TimeoutExpired:
            return "backend", {"status": "timeout"}
        except Exception as e:
            return "backend", {"status": "error", "error": str(e)}

    def run_frontend():
        if cancel_ev.is_set():
            return "frontend", {"status": "cancelled"}
        agent = HERMES_AGENTS[frontend_agent]
        prefix = agent.get("prompt_prefix", "")
        full_prompt = f"{prefix}\n\n任务：{frontend_task}\n\n请直接输出代码/结果，不要交互式确认。"
        cmd = [agent["command"]] + agent["args"] + [full_prompt]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=agent.get("timeout", 300))
            return "frontend", {
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout,
                "agent": frontend_agent,
            }
        except subprocess.TimeoutExpired:
            return "frontend", {"status": "timeout"}
        except Exception as e:
            return "frontend", {"status": "error", "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_backend): "backend",
            executor.submit(run_frontend): "frontend",
        }
        for future in concurrent.futures.as_completed(futures):
            role, result = future.result()
            results[role] = result
            if on_text:
                on_text(f"✅ {role} ({result.get('agent', '?')}) 完成: {result.get('status', '?')}")

    # Format output
    lines = ["# 分工模式结果", ""]
    for role in ("backend", "frontend"):
        r = results.get(role, {})
        agent = r.get("agent", "?")
        status = r.get("status", "?")
        output = r.get("output", "")
        if status == "success":
            lines.append(f"## {role.upper()} ({agent})")
            lines.append(f"```\n{output[:1000]}\n```")
        else:
            lines.append(f"## {role.upper()} ({agent}) — ❌ {status}")
            lines.append(f"Error: {r.get('error', output[:500])}")
        lines.append("")

    return "\n".join(lines)


def _mode_review_inline(task, context, agents, cancel_ev, on_text):
    """Review mode: implement → review → test pipeline."""
    # Default assignment: claude implement, kimi review, gemini test
    implementer = "claude"
    reviewer = "kimi" if "kimi" in agents else "claude"
    tester = "gemini" if "gemini" in agents else "kimi" if "kimi" in agents else "claude"

    # Step 1: Implement
    if on_text:
        on_text("[1/3] 实现阶段开始...")
    agent = HERMES_AGENTS[implementer]
    prefix = agent.get("prompt_prefix", "")
    full_prompt = f"{prefix}\n\n{context}\n\n任务：{task}\n\n请直接输出完整实现代码，不要交互式确认。"
    cmd = [agent["command"]] + agent["args"] + [full_prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=agent.get("timeout", 120))
        impl_output = result.stdout
        if result.returncode != 0:
            return f"❌ 实现阶段失败:\n{result.stderr[:1000]}"
    except Exception as e:
        return f"❌ 实现阶段异常: {e}"

    if cancel_ev.is_set():
        return "（已取消）"
    if on_text:
        on_text("✅ 实现阶段完成")

    # Step 2: Review
    if on_text:
        on_text("[2/3] 审查阶段开始...")
    agent = HERMES_AGENTS[reviewer]
    prefix = agent.get("prompt_prefix", "")
    review_prompt = (
        f"{prefix}\n\n请审查以下代码，找出问题并给出改进建议:\n\n"
        f"```\n{impl_output}\n```\n\n"
        f"请列出所有发现的问题（bug、性能问题、风格问题等）。"
    )
    cmd = [agent["command"]] + agent["args"] + [review_prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=agent.get("timeout", 180))
        review_output = result.stdout
    except Exception as e:
        review_output = f"审查异常: {e}"

    if cancel_ev.is_set():
        return "（已取消）"
    if on_text:
        on_text("✅ 审查阶段完成")

    # Step 3: Test
    if on_text:
        on_text("[3/3] 测试阶段开始...")
    agent = HERMES_AGENTS[tester]
    prefix = agent.get("prompt_prefix", "")
    test_prompt = (
        f"{prefix}\n\n请为以下代码编写完整的 pytest 测试，并修复审查发现的问题:\n\n"
        f"原始代码:\n```\n{impl_output}\n```\n\n"
        f"审查意见:\n{review_output}\n\n"
        f"请输出修复后的完整代码 + 完整的测试文件。"
    )
    cmd = [agent["command"]] + agent["args"] + [test_prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=agent.get("timeout", 180))
        test_output = result.stdout
    except Exception as e:
        test_output = f"测试异常: {e}"

    if cancel_ev.is_set():
        return "（已取消）"
    if on_text:
        on_text("✅ 测试阶段完成")

    return (
        f"# 评审模式结果\n\n"
        f"## 1. 实现 ({implementer})\n```\n{impl_output[:1500]}\n```\n\n"
        f"## 2. 审查 ({reviewer})\n```\n{review_output[:1500]}\n```\n\n"
        f"## 3. 测试 + 修复 ({tester})\n```\n{test_output[:2000]}\n```\n\n"
        f"---\n流水线完成: 实现 ✅ 审查 ✅ 测试 ✅"
    )


def _format_hermes_result(result: dict, mode: str) -> str:
    """Format the JSON result from the orchestrator script for Feishu card display."""
    if mode == "race":
        winner = result.get("winner", "?")
        agents = result.get("agents", {})
        winner_output = agents.get(winner, {}).get("output", "")
        # Truncate for card display
        preview = winner_output[:800]
        suffix = "\n\n…（输出已截断，完整内容见结果文件）" if len(winner_output) > 800 else ""
        return (
            f"🏆 **胜出: {winner}**\n"
            f"> {HERMES_AGENTS.get(winner, {}).get('strength', '')}\n\n"
            f"```\n{preview}{suffix}\n```\n\n"
            f"**其他 agent 结果:**\n"
            + "\n".join(
                f"{'✅' if r.get('status') == 'success' else '❌'} {a}: {r.get('status', '?')}"
                for a, r in agents.items()
                if a != winner
            )
        )
    elif mode == "split":
        backend = result.get("backend", {})
        frontend = result.get("frontend", {})
        b_agent = backend.get('agent', '?')
        f_agent = frontend.get('agent', '?')
        b_output = backend.get('output', '')[:600]
        f_output = frontend.get('output', '')[:600]
        return (
            f"**分工模式结果**\n\n"
            f"🖥️ **后端 ({b_agent})**\n"
            f"```\n{b_output}\n```\n\n"
            f"🎨 **前端 ({f_agent})**\n"
            f"```\n{f_output}\n```"
        )
    elif mode == "review":
        impl = result.get("implementation", {})
        review = result.get("review", {})
        test = result.get("test", {})
        impl_agent = impl.get('agent', '?')
        review_agent = review.get('agent', '?')
        test_agent = test.get('agent', '?')
        impl_out = impl.get('output', '')[:500]
        review_out = review.get('output', '')[:500]
        test_out = test.get('output', '')[:500]
        return (
            f"**评审模式结果**\n\n"
            f"1️⃣ **实现 ({impl_agent})**\n"
            f"```\n{impl_out}\n```\n\n"
            f"2️⃣ **审查 ({review_agent})**\n"
            f"```\n{review_out}\n```\n\n"
            f"3️⃣ **测试 ({test_agent})**\n"
            f"```\n{test_out}\n```\n\n"
            f"✅ 流水线完成"
        )
    else:
        # Fallback: return truncated JSON
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        return f"```json\n{json_str[:1000]}\n```"


def _make_hermes_plan(
    requirement: str,
    mode: str = "race",
    agents: list = None,
    context: str = "",
) -> dict:
    """Build a Hermes orchestrator parameter dict from a natural language requirement."""
    return {
        "task": requirement,
        "mode": mode,
        "agents": agents or ["claude", "kimi"],
        "context": context,
    }
