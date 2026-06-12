"""Workspace snapshot reader.

Reads ``<cwd>/.crew_workspace/workspace_meta.json`` (+ companion artefacts)
into a structured dict for the ``workspace_snapshot`` MCP tool
(``mcp_server.py``). Every external interaction (file read, git invoke) is
wrapped so a failure here never affects the caller — fail-soft contract.
"""
from __future__ import annotations

from pathlib import Path


# ── Tunable constants ────────────────────────────────────────────────────

# git invoke timeout. Bumped to 30s because large mono-repos with thousands
# of untracked files (e.g. node_modules left over by an aborted install) can
# legitimately take >8s.
_GIT_STATUS_TIMEOUT_SEC = 30


# ── Public API ───────────────────────────────────────────────────────────


def _sanitize_file_changes(files: list) -> list:
    """Filter file entries containing '..' or absolute paths."""
    from pathlib import Path as _Path
    result = []
    for f in files:
        s = str(f) if not isinstance(f, str) else f
        if ".." in s:
            continue
        try:
            if _Path(s).is_absolute():
                continue
        except Exception:
            continue
        result.append(f)
    return result


def generate_workspace_snapshot(workspace_dir: "Path") -> dict:
    """Read workspace_dir/workspace_meta.json and workspace_dir/file_changes.json,
    merge into a structured snapshot dict.

    Returns:
    {
        "batch_id":     str,   # workspace_meta.json["batch_id"], default ""
        "task_hash":    str,   # workspace_meta.json["task_hash"], default ""
        "completed":    bool,  # workspace_meta.json["completed"], default False
        "plan_title":   str,   # first line of prd.md stripped of # prefix, default ""
        "agent_results": list, # workspace_meta.json["agent_results"], default []
        "file_changes": list,  # file_changes.json["files"], default []
        "created_at":   float, # workspace_meta.json["finalized_at"], default 0.0
        "snapshot_at":  float, # time.time()
    }

    If workspace_meta.json does not exist: {"error": "workspace_meta not found"}.
    Any other IO or parse error: {"error": "<message>"}.
    Never raises (fail-soft contract).
    """
    import json as _json
    import time as _time

    try:
        meta_path = workspace_dir / "workspace_meta.json"
        if not meta_path.exists():
            return {"error": "workspace_meta not found"}
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))

        plan_title = ""
        try:
            prd_path = workspace_dir / "prd.md"
            if prd_path.exists():
                first_line = prd_path.read_text(encoding="utf-8").splitlines()[0]
                plan_title = first_line.lstrip("#").strip()
        except Exception:
            pass

        file_changes: list = []
        try:
            fc_path = workspace_dir / "file_changes.json"
            if fc_path.exists():
                fc_data = _json.loads(fc_path.read_text(encoding="utf-8"))
                file_changes = fc_data.get("files", []) or []
        except Exception:
            pass

        file_changes = _sanitize_file_changes(file_changes)

        last_commit_desc = ""
        try:
            import subprocess as _sp
            _cwd = str(workspace_dir.parent)
            _proc = _sp.run(
                ["git", "-C", _cwd, "log", "-1", "--format=%s"],
                capture_output=True, text=True, timeout=_GIT_STATUS_TIMEOUT_SEC,
            )
            if _proc.returncode == 0:
                last_commit_desc = _proc.stdout.strip()
        except Exception:
            pass

        return {
            "batch_id":        str(meta.get("batch_id", "") or ""),
            "task_hash":       str(meta.get("task_hash", "") or ""),
            "completed":       bool(meta.get("completed", False)),
            "plan_title":      plan_title,
            "agent_results":   list(meta.get("agent_results", []) or []),
            "file_changes":    file_changes,
            "created_at":      float(meta.get("finalized_at", 0.0) or 0.0),
            "snapshot_at":     _time.time(),
            "last_commit_desc": last_commit_desc,
        }
    except Exception as e:
        return {"error": str(e)}
