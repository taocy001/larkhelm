"""larkhelm · agent_hub.builtin.shell_agent — execute shell + AI interpretation.

Unlike /run (which returns raw output), ShellAgent:
  1. Pre-executes the shell command(s) extracted from the user's message.
  2. Injects the stdout/stderr/exit-code into the AI context.
  3. Lets the AI interpret the output and compose a meaningful reply.

This gives a much better UX for questions like "run git status and tell me
what needs committing" — the user gets both the raw facts AND the AI analysis
in one round-trip, without waiting for the AI to decide to call a bash tool.
"""
from __future__ import annotations

import re
import subprocess
import time

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult

# Commands that could cause irreversible damage are rejected up front.
_DANGEROUS_RE = re.compile(
    r"\b(rm\s+-[a-z]*r|mkfs|dd\s+if=|:(){ :|:&};:|chmod\s+777|chown|sudo|shutdown|reboot)\b",
    re.IGNORECASE,
)

_SHELL_TIMEOUT_SEC = 30
_MAX_OUTPUT_CHARS = 3000


def _extract_command(text: str) -> str:
    """Pull a shell command out of a natural-language request.

    Looks for ```...``` code fences first, then falls back to the raw text.
    """
    fence = re.search(r"```(?:bash|sh|shell)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Heuristic: if the text starts with a shell-looking token, use it as-is.
    stripped = text.strip()
    if re.match(r"^[\$>]?\s*([\w./~-])", stripped):
        # Strip leading prompt chars
        return re.sub(r"^[\$>\s]+", "", stripped)
    return stripped


def _run(cmd: str, cwd: str) -> tuple[str, str, int]:
    """Run a single shell command safely (no shell=True)."""
    import shlex, os as _os
    _SENSITIVE = ("SECRET", "TOKEN", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL")
    safe_env = {k: v for k, v in _os.environ.items()
                if not any(s in k.upper() for s in _SENSITIVE)}
    try:
        args = shlex.split(cmd)
    except ValueError as e:
        return "", f"命令格式错误: {e}", 1
    try:
        r = subprocess.run(
            args, shell=False, capture_output=True, text=True,
            timeout=_SHELL_TIMEOUT_SEC, cwd=cwd, env=safe_env,
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", f"命令超时（>{_SHELL_TIMEOUT_SEC}s）", -1
    except FileNotFoundError:
        return "", f"命令未找到: {args[0]}", 127
    except Exception as e:
        return "", str(e), -1


class ShellAgent(AgentExecutor):
    agent_type = "shell"
    description = "执行 shell 命令并用 AI 解读输出，适合「运行 X 并告诉我 Y」类请求"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            cmd = _extract_command(ctx.text)

            if _DANGEROUS_RE.search(cmd):
                from larkhelm.lark_client import send_card_reply
                send_card_reply(
                    ctx.chat_id, ctx.user_msg_id,
                    "⚠️ 命令被拒", f"检测到高危命令，已拒绝执行：\n```\n{cmd[:300]}\n```",
                    color="red",
                )
                return AgentResult(
                    success=False, duration_sec=time.monotonic() - start,
                    error="dangerous command rejected",
                )

            _debug_log(f"[ShellAgent] executing cmd={cmd!r} cwd={ctx.cwd!r}")
            stdout, stderr, rc = _run(cmd, ctx.cwd)

            # Build a context block the AI can reason about.
            parts = [f"以下是 shell 命令的执行结果，请根据用户原始问题给出解答。\n"]
            parts.append(f"**用户原始请求：** {ctx.text}\n")
            parts.append(f"**执行命令：** `{cmd}`\n**目录：** `{ctx.cwd}`\n**退出码：** `{rc}`\n")
            if stdout.strip():
                truncated = stdout.strip()[:_MAX_OUTPUT_CHARS]
                parts.append(f"\n**stdout：**\n```\n{truncated}\n```")
            if stderr.strip():
                truncated = stderr.strip()[:500]
                parts.append(f"\n**stderr：**\n```\n{truncated}\n```")
            if not stdout.strip() and not stderr.strip():
                parts.append("\n_（命令无输出）_")

            augmented_message = "\n".join(parts)
            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=augmented_message,
                model=model,
                user_msg_id=ctx.user_msg_id,
                parent_id=ctx.parent_id,
                force_backend_id=ctx.force_backend_id,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            _debug_log(f"[ShellAgent] execute failed: {e}")
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
