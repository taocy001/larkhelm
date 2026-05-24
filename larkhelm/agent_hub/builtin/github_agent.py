"""larkhelm · agent_hub.builtin.github_agent — GitHub operations via gh CLI or REST API.

Two execution paths (tried in order):
  1. ``gh`` CLI  — pre-authenticated, no extra config needed on dev machines.
  2. GitHub REST API via urllib — requires ``github_token`` in config.json.

Flow:
  1. Detect available path.
  2. Execute a ``gh`` / REST call based on the user's intent keywords.
  3. Inject the structured result into _do_query so the AI can compose a reply.

Supported intents (detected from ctx.text via keyword heuristics):
  - List open PRs / issues
  - Show PR details / diff summary
  - Create an issue
  - Check CI / workflow run status
  - List recent commits
  - Fall-through: open-ended GitHub query answered by AI with fetched context

Config keys (config.json):
  ``github_token``   — Personal access token or fine-grained token (optional if gh CLI authed)
  ``github_repo``    — Default repo in "owner/repo" format (optional; inferred from git remote)
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Optional

from larkhelm.agent_hub.agent_base import AgentExecutor
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult

_GH_TIMEOUT = 20
_MAX_OUTPUT_CHARS = 3000


# ── gh CLI helpers ─────────────────────────────────────────────────────

def _gh_available() -> bool:
    try:
        r = subprocess.run(["gh", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _run_gh(*args: str, cwd: Optional[str] = None) -> tuple[str, int]:
    """Run a ``gh`` subcommand; return (stdout, returncode)."""
    try:
        r = subprocess.run(
            ["gh", *args], capture_output=True, text=True,
            timeout=_GH_TIMEOUT, cwd=cwd,
        )
        out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr.strip() else "")
        return out.strip()[:_MAX_OUTPUT_CHARS], r.returncode
    except subprocess.TimeoutExpired:
        return "gh command timed out", -1
    except FileNotFoundError:
        return "gh CLI not found", 127
    except Exception as e:
        return str(e), -1


# ── GitHub REST API fallback ───────────────────────────────────────────

def _github_rest(path: str, token: str, method: str = "GET",
                 body: Optional[dict] = None) -> tuple[dict | list | None, int]:
    """Call the GitHub REST API using stdlib urllib."""
    import urllib.request
    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_GH_TIMEOUT) as resp:
            return json.loads(resp.read()), resp.status
    except Exception as e:
        return {"error": str(e)}, -1


# ── Intent → gh command mapping ────────────────────────────────────────

def _detect_gh_intent(text: str, repo: str) -> tuple[list[str], str]:
    """Map user text to a gh subcommand and a human-readable label."""
    tl = text.lower()
    repo_args = ["--repo", repo] if repo else []

    # PR list
    if re.search(r"\b(pr|pull.?request|拉取请求|合并请求)\b", tl) and re.search(r"\b(list|open|show|查|列)", tl):
        return ["pr", "list", "--state", "open", "--limit", "10", *repo_args], "Open PRs"

    # PR details (number present)
    pr_num = re.search(r"pr\s*#?(\d+)", tl)
    if pr_num:
        return ["pr", "view", pr_num.group(1), *repo_args], f"PR #{pr_num.group(1)}"

    # Issue list
    if re.search(r"\bissue\b|issue列表|待处理", tl) and re.search(r"\b(list|open|show|查|列)\b", tl):
        return ["issue", "list", "--state", "open", "--limit", "10", *repo_args], "Open Issues"

    # Create issue
    if re.search(r"(create|new|新建|提交).{0,10}(issue|问题|bug)", tl, re.IGNORECASE):
        title_m = re.search(r'["""](.+?)["""]|title[:\s]+(.+)', text)
        title = (title_m.group(1) or title_m.group(2)).strip() if title_m else "Issue from larkhelm"
        return ["issue", "create", "--title", title, "--body", text, *repo_args], f"Create issue: {title}"

    # CI / workflow runs
    if re.search(r"\b(ci|workflow|action|run|流水线|构建|build)\b", tl):
        return ["run", "list", "--limit", "5", *repo_args], "Recent CI runs"

    # Recent commits
    if re.search(r"\b(commit|提交记录|最近提交)\b", tl):
        return ["log", "--oneline", "-10"] if not repo_args else \
               ["api", f"/repos/{repo}/commits?per_page=10"], "Recent commits"

    # Repo status / default
    return ["repo", "view", *repo_args], "Repo overview"


def _get_default_repo(cwd: str) -> str:
    """Infer owner/repo from git remote.origin.url in cwd."""
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        url = r.stdout.strip()
        # github.com:owner/repo.git or https://github.com/owner/repo
        m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
        return m.group(1) if m else ""
    except Exception:
        return ""


# ── Agent ──────────────────────────────────────────────────────────────

class GitHubAgent(AgentExecutor):
    agent_type = "github"
    description = "GitHub 操作：列举 PR/Issue、查 CI 状态、提交 Issue，无需额外配置（使用 gh CLI 或 token）"
    required_capabilities = ()

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        from larkhelm.handlers._query import _do_query
        from larkhelm.chat_state import _get_chat_model
        from larkhelm.log import _debug_log

        start = time.monotonic()
        try:
            import larkhelm.config as _cfg
            github_token = getattr(_cfg, "GITHUB_TOKEN", "")
            config_repo = getattr(_cfg, "GITHUB_REPO", "")
            repo = config_repo or _get_default_repo(ctx.cwd)

            gh_args, label = _detect_gh_intent(ctx.text, repo)
            _debug_log(f"[GitHubAgent] intent={label!r} repo={repo!r} args={gh_args}")

            gh_output = ""
            if _gh_available():
                raw, rc = _run_gh(*gh_args, cwd=ctx.cwd)
                gh_output = raw
                _debug_log(f"[GitHubAgent] gh rc={rc}")
            elif github_token:
                # Simple fallback: list open PRs via REST
                rest_path = f"/repos/{repo}/pulls?state=open&per_page=10" if repo else "/user/repos?per_page=5"
                data, status = _github_rest(rest_path, github_token)
                gh_output = json.dumps(data, ensure_ascii=False, indent=2)[:_MAX_OUTPUT_CHARS]
            else:
                gh_output = (
                    "⚠️ GitHub 访问未配置：请安装 gh CLI (`brew install gh && gh auth login`) "
                    "或在 config.json 中设置 `github_token`。"
                )

            parts = [
                f"[GitHub 数据：{label}]\n",
                f"```\n{gh_output}\n```\n",
                f"\n---\n**用户请求：** {ctx.text}\n",
                "请根据以上 GitHub 数据回答用户请求。",
            ]
            augmented = "\n".join(parts)
            model = _get_chat_model(ctx.chat_id)
            _do_query(
                chat_id=ctx.chat_id,
                message=augmented,
                model=model,
                user_msg_id=ctx.user_msg_id,
                parent_id=ctx.parent_id,
                force_backend_id=ctx.force_backend_id,
            )
            return AgentResult(success=True, duration_sec=time.monotonic() - start)
        except Exception as e:
            _debug_log(f"[GitHubAgent] execute failed: {e}")
            return AgentResult(
                success=False, duration_sec=time.monotonic() - start, error=str(e),
            )
