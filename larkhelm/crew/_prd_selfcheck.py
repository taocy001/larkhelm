"""PRD self-check gate (B1).

Runs after the ``architect`` agent finishes writing ``tasks.json`` +
``prd_criteria.json`` + ``file_changes.json``. Three independent checks:

  (a) **Anchor grep** — every ``logic_analysis[i][2].anchors[].snippet``
      must hit between 1 and 5 lines in the target file via
      ``grep -F``. Hits of 0 or >5 fail the anchor.
  (b) **AC how_to_verify dry-parse** — every
      ``prd_criteria.json.criteria[].how_to_verify`` must not contain
      unfilled placeholders such as ``<VAR>`` / ``{var}`` / ``$VAR``.
  (c) **file_changes consistency** — for every
      ``file_changes.json.files[i]``, ``action="create"`` must mean the
      target path does NOT exist yet; any other action must mean the
      target path DOES exist.

Failures are concatenated into a Markdown summary written to
``.crew_workspace/prd_selfcheck.md`` — readable by the next architect
run via the standard retry feedback path.

Soft contract: silently skips anchor checks when the logic_analysis
entry is still in the legacy 2-tuple shape, so an old checkpoint won't
trip the gate.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from larkhelm.log import _debug_log


# Single-grep timeout (s) and overall self-check budget (s).
_GREP_TIMEOUT_SEC      = 5
_OVERALL_TIMEOUT_SEC   = 30
_ANCHOR_MIN_LEN        = 30
_ANCHOR_HIT_MIN        = 1
_ANCHOR_HIT_MAX        = 5
# Placeholder shapes considered "obvious unfilled template":
#   <FOO>     — Markdown-ish template var
#   {foo}     — Python-ish format placeholder
#   $VAR      — shell-ish (uppercase identifier only, to avoid false
#               matching on ``$0`` / ``$@`` in legit shell)
_PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]{1,}>|\{[a-zA-Z_][a-zA-Z0-9_]*\}|\$[A-Z][A-Z0-9_]{1,}")
# Regex special chars that disqualify a snippet for plain ``grep -F``
# usage. ``-F`` already handles them, so this is a stricter sanity
# guard for the writer (architect) than a runtime constraint.
_REGEX_META = set(".^$*+?()[]{}|\\")


def _read_json(path: Path) -> "dict | list | None":
    """Best-effort JSON read; returns None on missing file / parse error."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as e:
        _debug_log(f"[PRD-Selfcheck] read {path} failed: {e}")
        return None


def _grep_count(snippet: str, file_path: Path) -> int:
    """Count exact-match occurrences of ``snippet`` in ``file_path``.

    Uses ``grep -c -F`` so the snippet is treated literally. Returns -1
    on any subprocess failure (treated as a hit-failure upstream).
    """
    try:
        r = subprocess.run(
            ["grep", "-c", "-F", snippet, str(file_path)],
            capture_output=True, text=True, timeout=_GREP_TIMEOUT_SEC,
        )
        # grep -c returns 1 (no match) but stdout is "0\n", so we go by
        # stdout rather than returncode. Non-existent files raise via
        # the FileNotFoundError-equivalent OS error which lands in the
        # except below.
        if r.stdout.strip().isdigit():
            return int(r.stdout.strip())
        return 0
    except subprocess.TimeoutExpired:
        _debug_log(f"[PRD-Selfcheck] grep timeout: {file_path} <- {snippet[:40]!r}")
        return -1
    except FileNotFoundError:
        # grep binary missing — degrade open
        return -2
    except Exception as e:
        _debug_log(f"[PRD-Selfcheck] grep error {file_path}: {e}")
        return -1


def _check_anchors(cwd: Path, tasks: dict, deadline: float) -> list[str]:
    """Return a list of FAIL lines for anchor problems (empty = all pass).

    Per-anchor early-return: a single soft fail (e.g. snippet too short)
    is reported but does not abort the rest. Hard limit: stop iterating
    once overall self-check budget is exceeded.
    """
    import time
    fails: list[str] = []
    la = tasks.get("logic_analysis") or []
    if not isinstance(la, list):
        return fails

    for idx, entry in enumerate(la):
        if time.time() > deadline:
            fails.append(f"⏱ anchor check budget exhausted at idx={idx}, remaining items skipped")
            break
        if not isinstance(entry, (list, tuple)) or len(entry) < 1:
            continue
        file_path_str = entry[0]
        # Legacy 2-tuple — no anchors slot; skip silently (compat).
        if len(entry) < 3 or not isinstance(entry[2], dict):
            continue
        anchors = entry[2].get("anchors") or []
        if not isinstance(anchors, list) or not anchors:
            # Author opted out of anchors for this file — not a fail.
            continue

        target = cwd / file_path_str
        for ai, anc in enumerate(anchors):
            if not isinstance(anc, dict):
                fails.append(f"❌ {file_path_str} anchor[{ai}]: not a dict")
                continue
            snippet = (anc.get("snippet") or "").rstrip("\r\n")
            if not snippet:
                fails.append(f"❌ {file_path_str} anchor[{ai}]: empty snippet")
                continue
            if len(snippet) < _ANCHOR_MIN_LEN:
                fails.append(
                    f"❌ {file_path_str} anchor[{ai}]: snippet length "
                    f"{len(snippet)} < {_ANCHOR_MIN_LEN} (too generic)"
                )
                continue
            if any(ch in _REGEX_META for ch in snippet):
                # Soft fail — agent should pick a snippet without regex
                # meta even though ``grep -F`` would handle them — keeps
                # the contract simple for downstream Grep tool usage.
                fails.append(
                    f"⚠ {file_path_str} anchor[{ai}]: snippet contains regex metachar — "
                    f"choose a cleaner line"
                )
            if not target.exists():
                # Action-create paths legitimately don't exist yet.
                # Defer to the file_changes consistency check below.
                continue
            hits = _grep_count(snippet, target)
            if hits == -2:
                fails.append("⚠ system has no `grep` binary — anchor check degraded open")
                # No point hammering: bail.
                return fails
            if hits < 0:
                fails.append(f"❌ {file_path_str} anchor[{ai}]: grep failed (timeout or error)")
                continue
            if hits < _ANCHOR_HIT_MIN:
                fails.append(
                    f"❌ {file_path_str} anchor[{ai}]: snippet not found "
                    f"(grep -c == 0). snippet={snippet[:60]!r}"
                )
            elif hits > _ANCHOR_HIT_MAX:
                fails.append(
                    f"❌ {file_path_str} anchor[{ai}]: snippet ambiguous "
                    f"({hits} matches, max {_ANCHOR_HIT_MAX}). "
                    f"snippet={snippet[:60]!r}"
                )

    return fails


def _check_ac_verifications(criteria: list) -> list[str]:
    """Return FAIL lines for any AC ``how_to_verify`` that smells unfilled."""
    fails: list[str] = []
    if not isinstance(criteria, list):
        return fails
    for idx, c in enumerate(criteria):
        if not isinstance(c, dict):
            continue
        htv = c.get("how_to_verify", "")
        if not isinstance(htv, str) or not htv.strip():
            fails.append(f"❌ {c.get('id', f'AC-{idx}')}: how_to_verify empty")
            continue
        m = _PLACEHOLDER_RE.search(htv)
        if m:
            fails.append(
                f"❌ {c.get('id', f'AC-{idx}')}: unfilled placeholder "
                f"{m.group(0)!r} in how_to_verify"
            )
    return fails


def _check_file_changes(cwd: Path, file_changes: dict) -> list[str]:
    """Return FAIL lines for file_changes.json inconsistencies."""
    fails: list[str] = []
    files = file_changes.get("files") if isinstance(file_changes, dict) else None
    if not isinstance(files, list):
        return fails
    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            continue
        path_str = entry.get("path", "")
        action = (entry.get("action", "") or "").lower()
        if not path_str:
            fails.append(f"❌ files[{idx}]: empty path")
            continue
        target = cwd / path_str
        exists = target.exists()
        if action == "create" and exists:
            fails.append(
                f"❌ {path_str}: action=create but file already exists "
                f"(use action=modify or pick a different path)"
            )
        elif action and action != "create" and not exists:
            fails.append(
                f"❌ {path_str}: action={action} but file does not exist "
                f"(use action=create or fix the path)"
            )
    return fails


def run_prd_selfcheck(cwd: str | Path) -> tuple[bool, str]:
    """Run all three checks against ``<cwd>/.crew_workspace/``.

    Returns ``(passed, report_markdown)``. The report is written to
    ``<cwd>/.crew_workspace/prd_selfcheck.md`` as a side effect so the
    architect's next attempt can Read it via the standard feedback hook.

    Never raises — internal failures are appended to the report and
    counted as soft pass (so the gate degrades open rather than locking
    out a borderline-valid PRD).
    """
    import time
    cwd_p = Path(cwd)
    ws = cwd_p / ".crew_workspace"
    deadline = time.time() + _OVERALL_TIMEOUT_SEC

    tasks   = _read_json(ws / "tasks.json")          or {}
    prd_c   = _read_json(ws / "prd_criteria.json")   or {}
    fc      = _read_json(ws / "file_changes.json")   or {}

    try:
        a_fails = _check_anchors(cwd_p, tasks, deadline)
    except Exception as e:
        _debug_log(f"[PRD-Selfcheck] anchor check raised: {e}")
        a_fails = [f"⚠ anchor check internal error: {e}"]

    try:
        b_fails = _check_ac_verifications(prd_c.get("criteria") or [])
    except Exception as e:
        _debug_log(f"[PRD-Selfcheck] ac check raised: {e}")
        b_fails = [f"⚠ ac how_to_verify check internal error: {e}"]

    try:
        c_fails = _check_file_changes(cwd_p, fc)
    except Exception as e:
        _debug_log(f"[PRD-Selfcheck] file_changes check raised: {e}")
        c_fails = [f"⚠ file_changes consistency check internal error: {e}"]

    # ``⚠`` lines are soft warnings (degraded path, system constraint);
    # only ``❌`` lines count as a real fail.
    hard_count = sum(
        1
        for line in a_fails + b_fails + c_fails
        if line.startswith("❌")
    )
    passed = hard_count == 0

    parts: list[str] = ["# PRD Self-Check Report (B1)\n"]
    parts.append(f"**Result**: {'✅ PASS' if passed else '❌ FAIL'} "
                 f"({hard_count} hard fail(s))\n")
    parts.append("## (a) Anchor grep\n")
    parts.append("\n".join(a_fails) if a_fails else "_All anchors hit 1–5 lines._")
    parts.append("\n\n## (b) AC how_to_verify dry-parse\n")
    parts.append("\n".join(b_fails) if b_fails else "_All criteria look filled-in._")
    parts.append("\n\n## (c) file_changes.json consistency\n")
    parts.append("\n".join(c_fails) if c_fails else "_All paths consistent with action._")
    if not passed:
        parts.append(
            "\n\n---\n"
            "## 修复建议（架构师下一轮 retry 时）\n"
            "1. 上述每个 ❌ 行都必须解决；⚠ 行视为风格提示，建议但不强制\n"
            "2. 锚点失效时：用 Read 重新读源文件，挑一条更独特的整行（≥30 字符）\n"
            "3. AC 占位符未填：把 `<VAR>` / `{var}` 之类替换成具体命令或值\n"
            "4. file_changes 路径不一致：要么改 action，要么改 path\n"
        )
    report = "".join(parts) + "\n"

    try:
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "prd_selfcheck.md").write_text(report)
    except Exception as e:
        _debug_log(f"[PRD-Selfcheck] write report failed: {e}")

    return passed, report
