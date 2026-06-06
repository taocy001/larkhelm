"""
larkhelm · Crew Agent executor
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from larkhelm.log import _debug_log, log_entry, warn
from larkhelm.ai_runner import QueryCancelledError
from larkhelm.crew_types import (
    HardFailError, AgentSpec, AgentState, AgentStatus, CrewState,
    NoBackendAvailableError,
    CREW_RESULT_PREVIEW,
)
from larkhelm.crew_card import _crew_update_card, _start_heartbeat
from larkhelm.crew._backend_resolver import resolve_backend
from larkhelm.crew._failure_card import (
    emit_agent_failure, emit_breakpoint_timeout,
)


def _workspace_dir(chat_id: str, crew_id: str) -> Path:
    import larkhelm.config as _cfg
    d = _cfg.SESSION_DIR / chat_id / f"crew_{crew_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _detect_fail_marker(spec: AgentSpec, result: str) -> bool:
    """Check whether the last non-empty line of the output contains the fail_marker."""
    if not spec.fail_marker or not result:
        return False
    last_line = next((l.strip() for l in reversed(result.split("\n")) if l.strip()), "")
    return spec.fail_marker in last_line


def _parse_qa_verdict(result: str) -> dict:
    """Parse QA_VERDICT protocol line from QA agent output.

    Searches the last 20 non-empty lines of result for a line starting with "QA_VERDICT:".
    Expected format: "QA_VERDICT: PASS|FAIL FAILED=N BLOCKED=N SKIP=N"

    Returns:
        {"verdict": "PASS"|"FAIL"|"UNKNOWN",
         "failed_count": int, "blocked_count": int, "skip_count": int}
    Never raises; returns {"verdict": "UNKNOWN", ...} on any parse error.
    """
    _default = {"verdict": "UNKNOWN", "failed_count": 0, "blocked_count": 0, "skip_count": 0}
    try:
        import re as _re
        lines = [ln for ln in result.split("\n") if ln.strip()][-20:]
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped.startswith("QA_VERDICT:"):
                continue
            rest = stripped[len("QA_VERDICT:"):].strip()
            parts = rest.split()
            if not parts:
                return _default
            verdict = parts[0].upper()
            if verdict not in ("PASS", "FAIL"):
                return _default
            counts = {"failed_count": 0, "blocked_count": 0, "skip_count": 0}
            for p in parts[1:]:
                m = _re.match(r"FAILED=(\d+)", p)
                if m:
                    counts["failed_count"] = int(m.group(1))
                    continue
                m = _re.match(r"BLOCKED=(\d+)", p)
                if m:
                    counts["blocked_count"] = int(m.group(1))
                    continue
                m = _re.match(r"SKIP=(\d+)", p)
                if m:
                    counts["skip_count"] = int(m.group(1))
            return {"verdict": verdict, **counts}
    except Exception:
        pass
    return _default


# Tool-call protocol tokens that some backends emit as plain text when the
# larkhelm runner does not translate them into ``tool_use`` events. Observed
# 2026-05-22 from a DeepSeek-flavoured planner backend writing
# ``<｜｜DSML｜｜tool_calls>...`` verbatim into ``prd.md`` / ``design.md``,
# which silently corrupted the entire downstream pipeline (architect, fixer,
# QA all burned full agents' worth of tokens on an unusable contract).
# Pinned narrow on purpose — broad regexes risk false-positives on docs
# that legitimately quote these tokens.
_OUTPUT_SENTINELS: tuple[str, ...] = (
    "<｜｜DSML｜｜tool_call",   # DeepSeek DSML (zero-width-ish full-width pipes)
    "<｜tool▁call",             # DeepSeek alt tokenizer form
    "<｜tool_call",             # DeepSeek alt tokenizer form
    "<|tool_call",              # generic OpenAI-style sentinel leak
    "<tool_call>",
    "<tool_calls>",
)

# SEC-v2-MED-1 (2026-05-25 review_security_v2): Anthropic XML-style
# markers were briefly added to ``_OUTPUT_SENTINELS`` (LOW3, batch
# e15e2a4) as bare-substring matches. That over-fired on legitimate
# narrative prose that mentioned ``<function_calls>`` or ``<invoke
# name=`` — Claude API docs, agent prompt examples, project README
# tool-use sections, the larkhelm CLAUDE.md itself. The false-positive
# surface was unacceptable now that we deploy on a corpus that talks
# about agent tool-use as a topic.
#
# Replacement: require the FULL structural shape — an opening
# ``<function_calls>`` token, an inner ``<invoke name=`` element, and a
# closing ``</function_calls>`` — all within a bounded window. The shape
# only arises when a non-tool backend emitted unwrapped tool-call
# markup; ambient discussion of any single tag (even both in the same
# document) does not match. ``re.DOTALL`` so the inner ``<invoke …>``
# span may cross lines. The 4 KiB window cap prevents a malformed /
# legitimate document with the two tag names far apart from being
# flagged.
#
# Caveat: a malicious / sloppy backend that produces a legitimately-
# shaped open+invoke+close trio will still trip — that is the desired
# semantic. Discussion documents must wrap the example in fences /
# inline backticks (``_strip_code_evidence`` removes those before the
# scan), same contract as the strict sentinels above.
_ANTHROPIC_LOOSE_SENTINEL_RE = re.compile(
    r"<function_calls>.{0,4096}?<invoke\s+name=.{0,4096}?</function_calls>",
    re.DOTALL,
)


def _strip_code_evidence(text: str) -> str:
    """Remove all "quoted evidence" forms so downstream sentinel scans only
    see narrative prose, not legitimate citations of upstream corruption.

    Three forms scrubbed:

      1. ``` ``` ... ``` ``` triple-backtick fenced blocks
      2. ``` ` ... ` ``` single-backtick inline code spans
      3. ``> `` blockquote lines (often used to quote raw model output)

    Motivation: review / QA / docs agents legitimately quote upstream
    corruption as evidence — e.g. a reviewer writing about a prior
    SecurityExpert failure may put the offending token inside ` ` `
    inline spans (see ``review_security.md.invalid`` line 12 of the
    2026-05-25 failure: ``review_docs.md.invalid 是 DeepSeek 错误吐出
    的 `<｜｜DSML｜｜tool_calls>` 文本``). Pre-F1+F2 only the triple-
    fence form was stripped, so the inline-backtick quote tripped the
    sentinel and the entire 27 KB legitimate report was quarantined.

    Tradeoff: a malicious / sloppy backend could wrap its own leaked
    tool-call output in any of these forms to bypass detection. We
    accept this — the realistic adversary is "an agent quoting evidence
    in prose", not "an attacker crafting fake quote wrappers".
    """
    # Pass 1: drop ``` fenced blocks line-by-line. Tracks the open state
    # across lines because fence markers always live on their own line.
    if "```" in text:
        out: list[str] = []
        in_fence = False
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                out.append(line)
        text = "\n".join(out)
    # Pass 2: drop blockquote lines (``>`` at line start, optional leading
    # whitespace). These often quote raw model outputs in reviews.
    if ">" in text:
        text = "\n".join(
            ln for ln in text.split("\n")
            if not ln.lstrip().startswith(">")
        )
    # Pass 3: strip ``` ` ... ` ``` inline spans. Non-greedy match capped
    # within a single line so an unmatched stray backtick doesn't swallow
    # the rest of the document. ``re.DOTALL`` deliberately NOT set.
    if "`" in text:
        text = re.sub(r"`[^`\n]+`", "", text)
    return text


# Back-compat alias — older tests / external callers import the original
# name. New code should call ``_strip_code_evidence`` directly.
_strip_fenced_code_blocks = _strip_code_evidence


def _sanitize_quarantined_content(raw: str) -> str:
    """Best-effort recovery of prose from a ``<output>.invalid`` artifact
    so :func:`_synthesize` can surface a useful excerpt of an agent's
    work even after quarantine.

    F5 (2026-05-25): two passes —

      1. drop any LINE that still contains a tool-call sentinel (we
         already know the file is poisoned somewhere; this strips the
         specific token occurrence so the synth can never re-leak it
         into ``final_review.md``).
      2. run :func:`_strip_code_evidence` on the remainder so any inline
         backtick / fenced quote that survived line-drop is removed too.

    Returns ``""`` when nothing usable remains — caller falls back to
    the in-memory ``result`` / error-only summary.
    """
    if not raw:
        return ""
    kept: list[str] = []
    for line in raw.splitlines():
        if any(s in line for s in _OUTPUT_SENTINELS):
            continue
        kept.append(line)
    cleaned = _strip_code_evidence("\n".join(kept)).strip()
    # Reject if scrubbing left nothing meaningful — guards against the
    # docs-agent case where the entire .invalid file was sentinel markup
    # and nothing else (537 bytes of pure DSML).
    if len(cleaned) < 50:
        return ""
    return cleaned


def _validate_output_artifact(state: CrewState, agent_id: str, result: str) -> str:
    """Return a short problem description if the agent's output artifact is
    structurally corrupt, or "" if it looks OK.

    Two failure modes covered:
      1. Tool-call protocol tokens leaked into the artifact (see
         ``_OUTPUT_SENTINELS``). Downstream stages would parse these as
         markdown content and silently produce garbage. Sentinels inside
         quoted citations (fenced / inline-backtick / blockquote) are
         scrubbed first — see ``_strip_code_evidence``.
      2. ``.json`` ``output_file`` that is not parseable JSON. Downstream
         stages crash on ``json.load`` mid-wave. JSON validation does NOT
         strip fences (the whole file must parse).

    Scans the on-disk artifact when present (post Write tool) and falls back
    to the in-memory ``result`` otherwise. F1+F2 (2026-05-25): the scrubber
    now strips inline-backtick and blockquote citations in addition to
    fenced blocks, and the scan runs on the FULL prose body rather than
    capping at 8 KiB — the cap created a false sense of safety (the
    SecurityExpert false-positive sat in line 12, well inside the cap)
    and full-scan cost on 30 KB reports is sub-millisecond.
    Never raises — defects in this helper must not crash the crew wrapper.
    """
    try:
        spec = state.agents[agent_id].spec
    except Exception:
        return ""

    candidates: list[tuple[str, str]] = []  # (label, content)
    if spec.output_file:
        try:
            from larkhelm.chat_state import _get_cwd
            cwd = _get_cwd(state.chat_id)
            out_path = Path(cwd) / ".crew_workspace" / spec.output_file
            if out_path.exists():
                content = out_path.read_text(encoding="utf-8", errors="replace")
                if content:
                    candidates.append((spec.output_file, content))
        except Exception as e:
            _debug_log(f"[Crew] {agent_id} validate: read {spec.output_file} failed: {e}")
    if not candidates and result:
        candidates.append(("result", result))

    for label, content in candidates:
        # Sentinel scan: strip all "quoted evidence" forms first so
        # legitimate citations of upstream corruption (in QA, reviewer,
        # docs bail-out reports) do not trip. See `_strip_code_evidence`
        # docstring for the three forms recognised.
        prose = _strip_code_evidence(content)
        for s in _OUTPUT_SENTINELS:
            if s in prose:
                return f"{label} contains tool-call sentinel {s!r}"
        # SEC-v2-MED-1: structural Anthropic-XML check on the scrubbed
        # prose. Single tag mentions are noise (docs / READMEs talk about
        # them); only the full opening + invoke + closing trio is a real
        # leak. Gated by `crew_sentinel_anthropic_loose_enabled` (default
        # true). Metric always emitted so operators can spot misuse even
        # when the gate is off.
        loose_label = _anthropic_loose_check(label, prose)
        if loose_label:
            return loose_label
        # SEC-CRIT-4 layer-2 heuristic (review_security.md). Layer-1
        # above (`s in prose`) is bypassable by wrapping sentinels in
        # fenced / inline-backtick / blockquote citations — the scrubber
        # strips them, prose ends up clean, layer-1 misses. Layer-2
        # counts RAW sentinel occurrences (pre-scrub) and rejects when
        # heavy citation density hides too many of them (drop_ratio
        # heuristic) or when the raw count is paranoid-high regardless
        # of citation form. Gated by `crew_sentinel_layer2_enabled` +
        # traffic dial; metric is always emitted (observe mode) so
        # thresholds can be calibrated against real crew runs before
        # flipping enforcement on.
        l2_label = _layer2_check(state, label, content, prose)
        if l2_label:
            return l2_label
        # JSON validation runs on the full content — the file is supposed
        # to be parseable end-to-end, fences would already invalidate it.
        if spec.output_file and spec.output_file.endswith(".json"):
            try:
                json.loads(content)
            except Exception as e:
                return f"{label} is not valid JSON: {type(e).__name__}: {str(e)[:120]}"
    return ""


def _anthropic_loose_check(label: str, prose: str) -> str:
    """SEC-v2-MED-1 (2026-05-25 review_security_v2): structural check
    for Anthropic-shaped tool-call leakage in already-scrubbed prose.

    Returns a non-empty failure label when the structural pattern fires
    AND the loose tier is enabled, else "". Always observable via
    ``larkhelm_crew_validate_anthropic_loose_total{outcome}`` so
    operators see misuse counts even when the gate is off.

    Outcomes:
      * ``hit_enforced``  — pattern matched AND gate enabled → reject
      * ``hit_observed``  — pattern matched AND gate disabled → metric only
      * ``abstain``       — pattern did NOT match (no metric noise here)

    Never raises — defects in this helper must not crash the wrapper.
    """
    try:
        m = _ANTHROPIC_LOOSE_SENTINEL_RE.search(prose)
    except Exception:
        return ""
    if m is None:
        return ""

    try:
        import larkhelm.config as _cfg
        cfg = getattr(_cfg, "config", {}) or {}
        enabled = bool(cfg.get("crew_sentinel_anthropic_loose_enabled", True))
    except Exception:
        enabled = True

    outcome = "hit_enforced" if enabled else "hit_observed"
    try:
        from larkhelm.metrics import inc_crew_validate_anthropic_loose
        inc_crew_validate_anthropic_loose(outcome)
    except Exception as e:
        _debug_log(f"[Crew] anthropic_loose metric emit failed: {e}")

    if not enabled:
        return ""

    # Trim the match preview to keep the failure label tight for cards.
    snippet = m.group(0)
    if len(snippet) > 120:
        snippet = snippet[:117] + "…"
    return (
        f"{label} contains structural Anthropic tool-call XML "
        f"(<function_calls>…<invoke name=…</function_calls>): {snippet!r}"
    )


def _layer2_check(
    state: CrewState, label: str, content: str, prose: str,
) -> str:
    """SEC-CRIT-4 layer-2 sentinel heuristic. See `_validate_output_artifact`.

    Returns a non-empty failure label when layer-2 enforces (gated on +
    threshold trips), else "". Always observable via
    `larkhelm_crew_validate_layer2_total{outcome, mode}` — when traffic
    bucket misses we still bump the metric with ``mode="observe"`` so
    operators can pre-calibrate thresholds. Never raises.
    """
    try:
        raw_hits = sum(content.count(s) for s in _OUTPUT_SENTINELS)
    except Exception:
        return ""
    if raw_hits == 0:
        return ""  # nothing to evaluate; no metric noise either

    drop_ratio = 1.0 - (len(prose) / max(len(content), 1))

    try:
        import larkhelm.config as _cfg
        cfg = getattr(_cfg, "config", {}) or {}
        raw_threshold = int(cfg.get("crew_sentinel_layer2_raw_threshold", 3))
        ratio_threshold = float(cfg.get("crew_sentinel_layer2_drop_ratio", 0.30))
        paranoid_threshold = int(
            cfg.get("crew_sentinel_layer2_paranoid_threshold", 5)
        )
    except Exception:
        raw_threshold, ratio_threshold, paranoid_threshold = 3, 0.30, 5

    if raw_hits >= paranoid_threshold:
        outcome = "hit_paranoid"
        msg = (
            f"{label} has {raw_hits} sentinels — likely tool-call leak "
            f"[layer-2 paranoid ≥{paranoid_threshold}]"
        )
    elif raw_hits >= raw_threshold and drop_ratio > ratio_threshold:
        outcome = "hit_drop_ratio"
        msg = (
            f"{label} has {raw_hits} sentinels behind heavy quoting "
            f"(drop_ratio={drop_ratio:.0%}) — suspected wrapper bypass "
            f"[layer-2 drop_ratio>{ratio_threshold:.0%}]"
        )
    else:
        outcome = "abstain"
        msg = ""

    try:
        from larkhelm._gating import hash_traffic_active
        enforced = hash_traffic_active(
            str(getattr(state, "chat_id", "") or ""),
            enabled_key="crew_sentinel_layer2_enabled",
            traffic_key="crew_sentinel_layer2_traffic",
            default_enabled=False,
            default_traffic=0.0,
        )
    except Exception:
        enforced = False
    try:
        from larkhelm.metrics import inc_crew_validate_layer2
        inc_crew_validate_layer2(
            outcome=outcome, mode="enforced" if enforced else "observe",
        )
    except Exception:
        pass

    return msg if (enforced and outcome != "abstain") else ""


def _quarantine_invalid_output(state: CrewState, agent_id: str) -> None:
    """Rename a validation-failed ``output_file`` to ``<name>.invalid`` so
    the scheduler's partial-delivery rule (``_get_failed_dep``) does not
    interpret the corrupt file as "delivered" and let downstream agents run.

    Background: ``_get_failed_dep`` treats a FAILED upstream as non-blocking
    when its ``output_file`` exists and is non-empty (designed for the
    OOM-after-Write case where the artifact was atomically committed before
    the agent process died). That rule has no way to know that the file
    contents are structurally invalid — when our validator catches a
    tool-call leak, we have to invalidate the file ourselves, otherwise the
    crew cascades the same corruption through architect → implementer →
    QA → reviewer (observed 2026-05-22 dev failure).

    Atomic ``os.replace`` keeps the original bytes available for user
    inspection at the new ``.invalid`` path instead of deleting them.
    Failures here are logged and swallowed — quarantine is best-effort;
    the FAILED status itself still gets emitted by ``emit_agent_failure``.
    """
    try:
        spec = state.agents[agent_id].spec
    except Exception:
        return
    if not spec.output_file:
        return
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(state.chat_id)
        out_path = Path(cwd) / ".crew_workspace" / spec.output_file
        if not out_path.exists():
            return
        invalid_path = out_path.with_suffix(out_path.suffix + ".invalid")
        import os as _os
        _os.replace(out_path, invalid_path)
        warn(
            f"[Crew] {agent_id} quarantined invalid output: "
            f"{spec.output_file} → {invalid_path.name} "
            f"(inspect this file to see what the backend produced)"
        )
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} quarantine failed: {e}")


def _crew_owner_open_id(state: CrewState) -> str:
    """Return the open_id to transfer ownership to: chat sender > config default."""
    import larkhelm.config as _cfg
    from larkhelm.chat_state import _get_chat_state
    owner = _get_chat_state(state.chat_id).get("sender_open_id", "")
    return owner or _cfg.DEFAULT_OWNER_OPEN_ID


def _ensure_crew_folder(state: CrewState) -> None:
    """Create a per-project Feishu folder at crew start. Sets state.feishu_folder_token/url.
    Falls back silently if Feishu is unavailable or backend is 'local'.
    """
    import larkhelm.config as _cfg
    if _cfg.DOC_WRITE_BACKEND == "local":
        return

    project_name = (state.plan.title or "crew")[:40]
    owner_open_id = _crew_owner_open_id(state)

    from larkhelm.lark_client import FeishuDocClient, DocError
    doc_client = FeishuDocClient()

    try:
        if _cfg.DEFAULT_WIKI_SPACE_ID:
            ref = doc_client.create_wiki_node(
                _cfg.DEFAULT_WIKI_SPACE_ID, project_name, _cfg.DEFAULT_WIKI_PARENT_TOKEN,
                owner_open_id=owner_open_id,
            )
            node_token = ref.raw_url.split("/")[-1]
            state.feishu_folder_token = node_token
            state.feishu_folder_url   = f"https://feishu.cn/wiki/{node_token}"
        else:
            parent = _cfg.DEFAULT_DRIVE_FOLDER or ""
            folder_token = doc_client.create_folder(project_name, parent, owner_open_id=owner_open_id)
            state.feishu_folder_token = folder_token
            state.feishu_folder_url   = f"https://feishu.cn/drive/folder/{folder_token}"
        _debug_log(f"[Crew] 项目文件夹已创建: {state.feishu_folder_url}")
    except DocError as e:
        _debug_log(f"[Crew] 创建项目文件夹失败，将写入本地: {e}")


def _persist_result_to_output_file_if_missing(
    state: CrewState, agent_id: str, result: str,
) -> None:
    """Safety net: persist the in-memory ``result`` to the agent's declared
    ``output_file`` when the agent forgot (or failed) to call the Write tool.

    The agent prompt injection (in :func:`_run_agent`) tells the agent to
    Write its full output to ``.crew_workspace/{spec.output_file}``, but in
    practice ``state.agents[agent_id].result`` often holds only the LAST
    streamed text chunk (a short closing acknowledgement after the Write
    tool call). When the agent honours the contract, this is harmless —
    the on-disk file is the source of truth, and ``result`` is just a
    summary. When the agent skips the Write call (P3 PM did exactly this —
    emitted a ~39 K token PRD that never landed on disk), the architect
    downstream finds an empty / nonexistent file and falls back to
    reverse-engineering from source.

    Heuristic: write the fallback only when
      • ``spec.output_file`` is set
      • the file is missing OR materially smaller than ``result``
      • ``result`` is substantial (≥ 200 chars; below that it's almost
        certainly just a closing marker like "PRD written.")

    Failures are logged and swallowed — the safety net must never raise.
    """
    spec = state.agents[agent_id].spec
    if not spec.output_file:
        return
    if not result or len(result) < 200:
        # Result is just a closing marker; the agent presumably called Write
        # itself with the real payload. Don't risk overwriting a good file.
        return
    # Resolve ``out_path`` up front so both the LOW2 skip-path and the
    # normal write-path can use it (skip-path needs it for the evidence
    # sidecar, see SEC-v2-HIGH-1 below).
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(state.chat_id)
        out_path = Path(cwd) / ".crew_workspace" / spec.output_file
    except Exception as e:  # noqa: BLE001 — safety net must never raise
        _debug_log(
            f"[Crew] {agent_id} safety-net cwd lookup failed: {e}; "
            f"abandoning persist (validator will run on in-memory result)."
        )
        return
    # LOW2 (2026-05-25 review_followup): scan for tool-call sentinels
    # BEFORE atomic-writing. Without this, a corrupt result reaches disk
    # and only ``_validate_output_artifact`` (caller's next step) flags it
    # — at which point we have to ``_quarantine_invalid_output`` and a
    # concurrent reader may have already seen the poisoned file. Skipping
    # the write keeps the agent in its current state and lets the normal
    # FAILED → retry-on-different-backend path take over.
    prose = _strip_code_evidence(result)
    hit_sentinel = next((s for s in _OUTPUT_SENTINELS if s in prose), None)
    if hit_sentinel:
        warn(
            f"[Crew] {agent_id} safety-net skipped persist: result contains "
            f"tool-call sentinel {hit_sentinel!r} after code-evidence strip; "
            f"backend leaked a tool token into prose. Letting validator "
            f"surface the failure on the in-memory result instead."
        )
        # SEC-v2-HIGH-1 (2026-05-25 review_security_v2): preserve the raw
        # in-memory result on disk under ``.invalid.prerun`` so incident
        # response keeps evidence even when we skip the real persist.
        # Without this, the only signal is one ``warn(...)`` line that ages
        # out of LARKHELM debug log after 30 days. ``.invalid`` (no suffix)
        # is reserved for ``_quarantine_invalid_output`` — already-written
        # files moved aside. ``.invalid.prerun`` is the pre-write twin and
        # is never read by synth (sanitizer only globs ``.invalid``), so
        # this is purely a forensic artefact.
        try:
            ev_path = out_path.with_suffix(out_path.suffix + ".invalid.prerun")
            ev_path.parent.mkdir(parents=True, exist_ok=True)
            ev_path.write_text(result, encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — evidence persist is best-effort
            _debug_log(
                f"[Crew] {agent_id} safety-net evidence write failed: {e}; "
                f"in-memory warn above is the only signal."
            )
        return
    try:
        existing_size = 0
        if out_path.exists():
            try:
                existing_size = out_path.stat().st_size
            except OSError:
                existing_size = 0
        # If the on-disk file already covers ≥ 80% of result length, trust
        # the agent's own write — closing summary mismatch is fine.
        if existing_size >= int(len(result) * 0.8):
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: ``tmp`` + ``replace`` so a concurrent reader never
        # sees a torn file. Same pattern as ``memory_io.atomic_write``.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(result, encoding="utf-8")
        import os as _os
        _os.replace(tmp, out_path)
        # Surface at WARN so this never silently becomes "background noise":
        # every safety-net hit means the underlying agent failed to honour
        # the Write-tool contract, which is a real regression to chase
        # (Claude CLI perm denial / agent-prompt drift / token-limit cutoff
        # etc.). Independent review flagged that the original ``_debug_log``
        # alone gave operators zero signal.
        warn(
            f"[Crew] {agent_id} output_file safety-net wrote {len(result)} chars "
            f"to {spec.output_file} — agent skipped Write tool "
            f"(existing on disk: {existing_size}b). Investigate the agent's "
            f"tool-call trace to find why Write was bypassed."
        )
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} output_file safety-net failed: {e}")


def _sync_output_file(state: CrewState, agent_id: str) -> str:
    """Sync an agent's output_file to Feishu, inside the per-project folder.
    Returns the Feishu document URL on success, empty string on failure/local-only.
    Always falls back to local if Feishu is unavailable.
    """
    import larkhelm.config as _cfg
    if _cfg.DOC_WRITE_BACKEND == "local":
        return ""

    spec = state.agents[agent_id].spec
    if not spec.output_file:
        return ""

    from larkhelm.chat_state import _get_cwd
    cwd      = _get_cwd(state.chat_id)
    out_path = Path(cwd) / ".crew_workspace" / spec.output_file
    if not out_path.exists():
        return ""

    try:
        content = out_path.read_text(encoding="utf-8")
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} 读取 output_file 失败: {e}")
        return ""

    _STEM_NAMES = {
        "prd": "产品需求文档", "design": "技术设计方案",
        "changes": "代码变更记录", "qa_report": "测试报告", "review": "代码审查报告",
    }
    display_stem  = _STEM_NAMES.get(out_path.stem, out_path.stem)
    project_prefix = (state.plan.title or "crew")[:30].strip()
    title         = f"{project_prefix} · {display_stem}"
    folder_token  = state.feishu_folder_token
    owner_open_id = _crew_owner_open_id(state)

    from larkhelm.lark_client import FeishuDocClient, DocError, parse_doc_url
    doc_client = FeishuDocClient()

    # Reuse existing Feishu doc if this output_file was already synced (e.g. fixer reusing changes.md)
    existing_url = state.output_file_urls.get(spec.output_file, "")

    try:
        if existing_url:
            # Append to the already-created doc instead of creating a duplicate
            ref = parse_doc_url(existing_url)
            doc_client.append(ref, content)
            _debug_log(f"[Crew] {agent_id} 追加到已有飞书文档: {existing_url}")
            return existing_url
        elif _cfg.DEFAULT_WIKI_SPACE_ID:
            parent_token = folder_token or _cfg.DEFAULT_WIKI_PARENT_TOKEN
            ref        = doc_client.create_wiki_node(_cfg.DEFAULT_WIKI_SPACE_ID, title, parent_token,
                                                     owner_open_id=owner_open_id)
            node_token = ref.raw_url.split("/")[-1]
            doc_url    = f"https://feishu.cn/wiki/{node_token}"
            doc_client.append(ref, content)
        else:
            target = folder_token or _cfg.DEFAULT_DRIVE_FOLDER or ""
            ref     = doc_client.create_doc(title, target, owner_open_id=owner_open_id)
            doc_url = f"https://feishu.cn/docx/{ref.token}"
            doc_client.append(ref, content)

        with state.lock:
            state.output_file_urls[spec.output_file] = doc_url
        _debug_log(f"[Crew] {agent_id} 已同步到飞书: {doc_url}")
        return doc_url

    except DocError as e:
        _debug_log(f"[Crew] {agent_id} 飞书写入失败，保留本地文件: {e}")
        return ""


def _run_agent(state: CrewState, agent_id: str) -> str:
    """Execute a single agent and return the text result. Uses an isolated session namespace."""
    import larkhelm.config as _cfg
    from larkhelm.chat_state import _get_cwd
    from larkhelm.concurrency import _get_cancel_event
    from larkhelm.card_builder import _fmt_elapsed
    from larkhelm.perm import grant_yolo, revoke_yolo
    from larkhelm.ai_runner import query_gemini, query_kimi
    from larkhelm.crew._scheduler import _resolve_prompt
    from larkhelm.crew._state import _git_head
    # ``_run_hermes_orchestrator`` is lazy-imported inside the ``hermes_*``
    # dispatch branch below so a failure to load it cannot poison agents
    # that never call it (regression observed 2026-05-23: a single transient
    # ModuleNotFoundError at this site failed all 4 agents of a /crew run).

    spec      = state.agents[agent_id].spec
    cancel_ev = state.cancel_ev
    cwd       = _get_cwd(state.chat_id)
    workspace = _workspace_dir(state.chat_id, state.crew_id)

    # Session isolation: crew_ns is the namespace, isolated from the main session.
    # Crew agents do not use persistent sessions (sid=None):
    #   1. Carrying over the previous failure history on retry significantly increases token usage.
    #   2. Resumed runs inject context via the is_resuming prompt and do not rely on sessions.
    crew_ns = f"{state.chat_id}__crew_{state.crew_id}_{agent_id}"
    sid     = None  # No session reuse; each run starts a fresh conversation

    # Build the full prompt (system + placeholder resolution).
    # On retry, switch to the retry-specific role prompt (e.g. fixer mode).
    ag_state = state.agents[agent_id]
    active_system = spec.retry_system if (ag_state.retry_count > 0 and spec.retry_system) else spec.system
    active_prompt = spec.retry_prompt if (ag_state.retry_count > 0 and spec.retry_prompt) else spec.prompt
    
    # For Hermes orchestrator agents, the prompt is JSON params
    if spec.model.startswith("hermes_"):
        full_prompt = active_prompt
    else:
        resolved = _resolve_prompt(active_prompt, state)
        if active_system:
            full_prompt = f"{active_system}\n\n---\n\n{resolved}"
        else:
            full_prompt = resolved

    # Inject output_file requirement so the agent knows it must write the file
    if spec.output_file and not spec.model.startswith("hermes_"):
        full_prompt += (
            f"\n\n---\n\n**Output requirement**: When you have finished, write your complete "
            f"output to `.crew_workspace/{spec.output_file}` using the Write tool. "
            f"Do not skip this step."
        )

    # Inject Feishu doc URLs from completed upstream agents so this agent can reference them
    _upstream_doc_refs = []
    for _dep_id in spec.depends_on:
        _dep = state.agents.get(_dep_id)
        if _dep and _dep.feishu_doc_url:
            _upstream_doc_refs.append(f"- {_dep.spec.role}：{_dep.feishu_doc_url}")
    if _upstream_doc_refs and not spec.model.startswith("hermes_"):
        full_prompt += (
            "\n\n---\n\n**上游 Agent 飞书文档（可直接引用，与本地文件内容相同）：**\n"
            + "\n".join(_upstream_doc_refs)
        )

    # Inject downstream feedback (written by _execute on retry, truncated to 3000 chars to prevent token explosion)
    if ag_state.feedback:
        round_info = f" (retry {ag_state.retry_count})" if ag_state.retry_count else ""
        feedback_trimmed = ag_state.feedback[:3000]
        if len(ag_state.feedback) > 3000:
            feedback_trimmed += f"\n\n…（已截断，共 {len(ag_state.feedback)} 字符）"
        full_prompt = (
            f"⚠️ **Feedback from previous iteration{round_info} — please fix the following issues:**\n\n"
            f"{feedback_trimmed}\n\n"
            f"---\n\n"
            + full_prompt
        )

    # Resume context injection: inform the agent this task is being resumed from a checkpoint
    if state.is_resuming:
        _ws_path = Path(cwd) / ".crew_workspace"
        # Only list planning/document files (.md/.json); filter out intermediate artifacts like result.txt (noise)
        _plan_files = sorted(
            f.name for f in _ws_path.iterdir()
            if f.is_file() and f.suffix in (".md", ".json")
        ) if _ws_path.exists() else []
        _git_info = ""
        if state.git_head_before:
            _cur_head = _git_head(cwd)
            if _cur_head and _cur_head != state.git_head_before:
                _git_info = (f"\n- Code has been modified (start commit: {state.git_head_before},"
                             f" current: {_cur_head}). Run `git diff {state.git_head_before}`"
                             f" first to review completed changes and avoid duplicate work")
            elif _cur_head:
                _git_info = f"\n- No code changes detected (HEAD: {_cur_head})"
        # Build phase_outputs summary lines for resume context
        _po_lines: list[str] = []
        if state.phase_outputs:
            _spec_by_id = {s.id: s for s in state.plan.agents}
            for _po_id, _po_data in state.phase_outputs.items():
                _po_spec = _spec_by_id.get(_po_id)
                _po_role = _po_spec.role if _po_spec else _po_id
                _po_file = (_po_data.get("output_file") or "")
                _po_sum  = (_po_data.get("summary") or "")[:200]
                _po_arrow = f" → {_po_file}" if _po_file else ""
                _po_lines.append(f"[{_po_role}{_po_arrow}]: {_po_sum}")

        _resume_prefix = (
            "⚠️ **Resuming task (previous execution was interrupted, continuing from checkpoint)**\n\n"
            "Resume notes:\n"
            "- `.crew_workspace/` contains the planning files from the last run; **use these as the baseline**, do not re-plan"
            + (_git_info or "")
            + ("\n- Existing planning files: " + ", ".join(_plan_files) if _plan_files else "")
            + ("\n\n前序 Agent 摘要：\n" + "\n".join("- " + ln for ln in _po_lines) if _po_lines else "")
            + "\n- Continue directly from the incomplete parts; skip already completed content\n\n---\n\n"
        )
        full_prompt = _resume_prefix + full_prompt

    # Resolve memory context for this agent (B1/B6 + Phase B S44 unification):
    # API backends receive it as extra_system; CLI backends get a [System] prefix.
    # hermes orchestrators have their own context and don't need memory injection.
    #
    # Phase B unifies crew agents with /chat and /plan: all three now call
    # ``get_memory_context`` (global + project + session). Previously crew
    # agents called ``get_project_memory_context`` which dropped the global
    # layer, hiding user-level preferences (style / language) from sub-agents
    # — a frequent source of "agent ignored my preferences" reports.
    # Per-layer S49–S52 budget logic in ``MemoryContextBuilder`` keeps the
    # token footprint bounded; passing ``full_prompt`` as the gating query
    # lets ``should_include_*`` make keyword-aware decisions.
    _crew_mem_ctx = ""
    if not spec.model.startswith("hermes_"):
        try:
            from larkhelm.memory import get_memory_context_v2
            # Phase D: map AgentSpec.task_profile → agent_type so the
            # retriever (when enabled) chooses a sensible per-agent policy.
            # Unknown profiles fall back to ``crew`` policy (large budget,
            # decision-heavy kind priority).
            _tp_to_agent_type = {
                "planner": "plan",
                "engineer": "dev",
                "qa": "dev",
                "reviewer": "dev",
                "chat": "chat",
            }
            _agent_type = _tp_to_agent_type.get(
                (spec.task_profile or ""), "crew",
            )
            _crew_mem_ctx, _ = get_memory_context_v2(
                state.chat_id, cwd=str(cwd), query=full_prompt,
                sender_open_id=state.sender_open_id,
                backend_spec=spec,
            )
        except Exception as e:
            _debug_log(f"[Crew] memory load failed: {e}")

    # Timeout control: start countdown only after acquiring the process slot (semaphore),
    # so waiting time is not counted toward the timeout.
    agent_cancel  = threading.Event()
    _slot_ready   = threading.Event()   # fired when the semaphore slot is acquired

    def _timeout_watcher():
        """Forward crew-level cancellation; emit a soft-timeout log line.

        **No hard kill** — that responsibility belongs to
        ``BaseProcessRunner._watch`` which now measures **idle** time. The
        previous version of this watcher set ``hard_deadline = time.time()
        + max(spec.timeout * 2, HARD_TIMEOUT)`` and unconditionally fired
        ``agent_cancel`` at the wall-clock mark — meaning a long but
        actively-streaming agent (multi-hour /dev pipeline producing
        continuous output) got cancelled at the same instant a wedged
        agent did. The idle-clock fix in ``runner_base`` only takes
        effect if this outer wall-clock kill doesn't preempt it first.
        Forwarding cancel_ev + logging soft is the only remaining job.
        """
        # Phase 1: wait for process slot while monitoring crew-level cancellation.
        # Also honours ``agent_cancel`` — set by ``_run_agent``'s own
        # ``finally`` block at the bottom of this function (line ~472),
        # which fires when the agent body exits early (e.g.
        # ``NoBackendAvailableError`` raised before any subprocess starts).
        # Without this check the watcher would self-spin for the full
        # sem-wait window after each failed agent, accumulating one
        # daemon thread per failure until the crew completes. Cheap fix
        # per review OBS-01; regression test:
        # ``test_timeout_watcher_exits_when_agent_cancel_set``.
        while not _slot_ready.is_set():
            if cancel_ev.is_set() or agent_cancel.is_set():
                agent_cancel.set()
                return
            time.sleep(0.3)
        # Phase 2: process has started; begin counting spec.timeout from now
        soft_deadline = time.time() + spec.timeout
        soft_fired = False
        while True:
            if cancel_ev.is_set():
                agent_cancel.set()
                return
            # Stop polling once the inner runner has signalled completion —
            # otherwise this thread spins until process exit (harmless but
            # wasteful) and may delay agent cleanup.
            if agent_cancel.is_set():
                return
            if not soft_fired and time.time() >= soft_deadline:
                soft_fired = True
                _debug_log(
                    f"[Crew] {agent_id} soft timeout ({spec.timeout}s), "
                    "BaseProcessRunner idle-clock continues to gate hard kill"
                )
            time.sleep(0.3)

    threading.Thread(target=_timeout_watcher, daemon=True).start()

    # Progress card update: only cache the latest text; heartbeat (_start_heartbeat) pushes the card
    def _on_text(text: str, status: str = "typing"):
        with state.lock:
            state.agents[agent_id].result = text

    def _on_start():
        """Callback when the process slot is acquired: update status and start the timeout countdown."""
        _slot_ready.set()
        with state.lock:
            state.agents[agent_id].status     = AgentStatus.RUNNING
            state.agents[agent_id].start_time = time.time()
        _crew_update_card(state)

    # Crew agents automatically grant all permissions (user authorized via /crew command)
    grant_yolo(crew_ns)
    try:
        # Resolve which backend should serve this agent BEFORE entering the
        # dispatch branches. The resolver checks task_profile first
        # (Phase C path), then falls back to legacy model-string handling
        # for backward-compat. The resolved BackendSpec drives the branch
        # selection below via its ``provider`` field.
        # F4 (2026-05-25) + SEC-v2-MED-2 (review_security_v2): honour any
        # per-agent transient exclusions stamped by ``_run_agent_wrapper``
        # after a validate failure, so the retry picks a different
        # backend. The map is keyed by expiry timestamp; only entries
        # whose cool-down has NOT lapsed gate the resolver. Expired
        # entries fall out naturally here and are pruned later in
        # ``_execute``'s retry-target reset block.
        _now_excl = time.time()
        excluded = frozenset(
            bid for bid, exp in (state.agents[agent_id].excluded_backends_until or {}).items()
            if exp > _now_excl
        )
        try:
            resolved = resolve_backend(spec, exclude_backend_ids=excluded)
        except NoBackendAvailableError:
            # Re-raise so the wrapper can record the failure and surface
            # a "无可用 backend" card; not retried because this is a config
            # / health issue, not a transient subprocess error.
            raise

        # Record the actual backend id picked by the resolver so the
        # crew card can show "[claude] PM" / "[kimi] engineer" instead
        # of the now-empty ``[spec.model]`` placeholder. Phase-C left
        # ``spec.model=""`` and ``spec.task_profile="planner"/...`` so
        # the model is only known after this resolve call.
        with state.lock:
            state.agents[agent_id].actual_backend_id = resolved.id

        # Resolved hermes_* synthesizes a provider="hermes" BackendSpec —
        # rebuild the dispatch decision around ``resolved.provider`` so
        # existing branches stay intact.
        if resolved.provider == "hermes":
            _disp_kind = resolved.id        # e.g. "hermes_race"
        elif resolved.provider == "gemini_cli":
            _disp_kind = "gemini"
        elif resolved.provider == "kimi_cli":
            _disp_kind = "kimi"
        elif resolved.provider == "deepseek_api":
            _disp_kind = "deepseek"
        else:
            _disp_kind = "orchestrator"

        if _disp_kind == "gemini":
            _on_start()   # Gemini has no semaphore callback; trigger directly
            _gemini_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                           if _crew_mem_ctx else full_prompt)
            output = query_gemini(
                chat_id=crew_ns,
                message=_gemini_msg,
                cwd=cwd,
                cancel_ev=agent_cancel,
                on_text=_on_text,
                on_tool=None,
                on_tool_result=None,
                use_session=False,          # No session reuse; prevents carrying history into retries
                record_under=state.chat_id, # Token stats recorded under the real chat_id
            )
        elif _disp_kind == "kimi":
            _on_start()   # Kimi has no semaphore callback; trigger directly
            _kimi_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                         if _crew_mem_ctx else full_prompt)
            output = query_kimi(
                chat_id=crew_ns,
                message=_kimi_msg,
                cwd=cwd,
                cancel_ev=agent_cancel,
                on_text=_on_text,
                use_session=False,
                record_under=state.chat_id,
            )
        elif _disp_kind == "deepseek":
            from larkhelm.ai_runner import query_deepseek
            _on_start()   # DeepSeek has no semaphore callback; trigger directly
            _ds_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                       if _crew_mem_ctx else full_prompt)
            output = query_deepseek(
                chat_id=crew_ns,
                message=_ds_msg,
                cwd=cwd,
                cancel_ev=agent_cancel,
                on_text=_on_text,
                use_session=False,
                record_under=state.chat_id,
            )
        elif _disp_kind.startswith("hermes_"):
            # Hermes multi-agent orchestrator modes: race, split, review.
            # ``provider == "hermes"`` was previously checked here too,
            # but ``_disp_kind`` is set to ``resolved.id`` exactly when
            # ``resolved.provider == "hermes"`` (line 356), and every
            # hermes BackendSpec id begins with ``hermes_``, so the
            # second clause was unreachable. Review §4 cleanup.
            from larkhelm.crew._hermes_orchestrator import _run_hermes_orchestrator
            _on_start()
            output = _run_hermes_orchestrator(
                state=state,
                agent_id=agent_id,
                spec=spec,
                cancel_ev=agent_cancel,
                on_text=_on_text,
            )
        else:
            from larkhelm.backend_cli import run_claude as _bc_run_claude, run_gemini as _bc_run_gemini, run_kimi as _bc_run_kimi
            # ``resolved`` is whatever the backend resolver picked — could be
            # the orchestrator default OR a task_profile-ranked candidate.
            # Either way we hand off to one of the backend_cli / backend_api
            # branches below based on its provider, so claude-disabled hosts
            # naturally fall through to kimi/deepseek (PRD AC-06).
            _spec = resolved
            _API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api")
            _cli_msg = (f"[System]\n{_crew_mem_ctx}\n\n[User Query]\n{full_prompt}"
                        if _crew_mem_ctx else full_prompt)
            if _spec.provider in _API_PROVIDERS:
                import larkhelm.backend_api as _bapi
                _fn = {"anthropic_api": _bapi.run_anthropic,
                       "google_api": _bapi.run_google,
                       "openai_compat_api": _bapi.run_openai_compat}[_spec.provider]
                _on_start()
                output, _ = _fn(spec=_spec, chat_id=crew_ns, message=full_prompt,
                                history=[], cancel_ev=agent_cancel, on_text=_on_text,
                                extra_system=_crew_mem_ctx)
            elif _spec.provider == "gemini_cli":
                _on_start()
                output = _bc_run_gemini(spec=_spec, chat_id=crew_ns, message=_cli_msg,
                                        sid=None, cwd=cwd, cancel_ev=agent_cancel,
                                        on_text=_on_text, use_session=False)
            elif _spec.provider == "kimi_cli":
                _on_start()
                output = _bc_run_kimi(spec=_spec, chat_id=crew_ns, message=_cli_msg,
                                      sid=None, cwd=cwd, cancel_ev=agent_cancel,
                                      on_text=_on_text)
            else:
                output = _bc_run_claude(
                    spec=_spec, chat_id=crew_ns, message=_cli_msg,
                    sid=sid, cwd=cwd, cancel_ev=agent_cancel,
                    on_text=_on_text, on_tool=None, on_tool_result=None,
                    allow_retry=False, on_start=_on_start, session_namespace=crew_ns,
                )
    except QueryCancelledError:
        if cancel_ev.is_set():
            raise  # Crew-level cancellation → propagate → wrapper marks CANCELLED
        # Per-agent timeout only → FAILED (not CANCELLED)
        raise RuntimeError(f"Agent timed out (terminated after {_fmt_elapsed(spec.timeout)})")
    finally:
        agent_cancel.set()  # always stop the timeout watcher thread, even on abnormal exit
        revoke_yolo(crew_ns)

    result = output.strip() or "（无输出）"

    # Write result to file so downstream agents can read the full text via the Read tool
    result_file = workspace / f"{agent_id}_result.txt"
    try:
        result_file.write_text(result, encoding="utf-8")
    except Exception as e:
        _debug_log(f"[Crew] result file write failed: {e}")

    # For Hermes orchestrator agents, also write a formatted summary to a markdown file
    if spec.model.startswith("hermes_"):
        _write_hermes_summary(state, agent_id, result, workspace)

    return result


def _write_hermes_summary(state: CrewState, agent_id: str, result: str, workspace: Path) -> None:
    """Write a formatted markdown summary of Hermes orchestrator results for Feishu doc sync."""
    spec = state.agents[agent_id].spec
    mode = spec.model.replace("hermes_", "")
    
    summary_path = workspace / f"{agent_id}_summary.md"
    lines = [
        f"# {spec.role} 结果 ({mode} 模式)",
        "",
        f"**任务:** {state.plan.title}",
        f"**Agent:** {agent_id}",
        f"**模式:** {mode}",
        f"**完成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        result,
        "",
        "---",
        "",
        f"*由 LarkHelm Hermes Orchestrator 自动生成*",
    ]
    try:
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        _debug_log(f"[HermesOrchestrator] summary written to {summary_path}")
    except Exception as e:
        _debug_log(f"[HermesOrchestrator] failed to write summary: {e}")


# OOM detection signature set. The two main sources of OOM in this
# project's history:
#
#   * cgroup OOM-killer killing the node CLI (claude/kimi/gemini)
#     subprocess. ``runner_base._on_kill_signal`` wraps SIGKILL into
#     ``RuntimeError(..."killed by OS (rc=-9)..."``).
#   * V8 internal "FATAL ERROR: Reached heap limit" — the node CLI
#     prints this to stderr and exits non-zero with a different
#     message shape; the runner reports it as ``"abnormal exit
#     rc=N\n<stderr-tail>"``.
#
# Match conservatively against both shapes; over-matching here just
# means an unrelated transient error gets the 8s backoff instead of
# 1s (harmless), while under-matching means an OOM gets 1s and likely
# OOMs again (the bug we're fixing). The substrings are deliberately
# lowercased / case-insensitive checked.
_OOM_ERROR_MARKERS = (
    "killed by os",                   # runner_base._on_kill_signal
    "rc=-9",                          # explicit SIGKILL exit code
    "cgroup oom",                     # explicit kernel cgroup OOM
    "out of memory",                  # generic
    "memorymax",                      # systemd cgroup property name
    "reached heap limit",             # V8 native OOM
    "javascript heap out of memory",  # node default OOM phrase
    "fatal error: ineffective mark-compacts near heap limit",  # V8 specific
)


def _is_likely_oom_error(exc: Exception) -> bool:
    """Return True iff the exception message matches a known OOM
    signature. Used to bias the retry backoff toward giving the
    cgroup time to reclaim memory before re-spawning.

    Conservative + idempotent — never raises; bad exception object
    just falls through to False.
    """
    try:
        msg = str(exc).lower()
    except Exception:
        return False
    return any(marker in msg for marker in _OOM_ERROR_MARKERS)


def _log_oom_diagnostics(agent_id: str) -> None:
    """Snapshot the larkhelm cgroup's memory state into the debug log.

    Sole purpose: forensic. When OOM hits in production it's often
    investigated hours later when the kernel ring buffer / journalctl
    has already rotated. Persisting one line per OOM into
    ``_cfg.DEBUG_LOG`` keeps the signal close to the failure for
    later grep.

    Fail-soft: any disk / IO error is swallowed so this diagnostic
    helper can never compound an already-bad situation.
    """
    try:
        cgroup_root = Path("/sys/fs/cgroup/system.slice/larkhelm.service")
        if not cgroup_root.is_dir():
            return
        snapshot = {}
        for f in ("memory.current", "memory.high", "memory.max",
                  "memory.peak", "memory.swap.current"):
            try:
                snapshot[f] = (cgroup_root / f).read_text(encoding="utf-8").strip()
            except OSError:
                snapshot[f] = "?"
        _debug_log(
            f"[Crew] {agent_id} OOM diagnostic — cgroup state: " +
            " ".join(f"{k}={v}" for k, v in snapshot.items())
        )
    except Exception as e:
        _debug_log(f"[Crew] {agent_id} OOM diagnostic snapshot failed: {e}")


def _preflight_env_check(spec: "AgentSpec", cfg: dict) -> "tuple[bool, str]":
    """Check agent runtime environment against spec requirements.

    Returns (True, "") if the agent can run normally.
    Returns (False, reason) if the agent should be set to SKIPPED.

    Checks (short-circuit):
      1. spec.require_arch non-empty → check sys.platform + platform.machine()
         format: "os/arch", e.g. "linux/amd64"
         linux/amd64 → sys.platform=="linux" and platform.machine()=="x86_64"
      2. spec.require_docker_image non-empty → subprocess.run(
             ["docker", "image", "inspect", spec.require_docker_image],
             shell=False, timeout=5, capture_output=True
         ), returncode != 0 → fail

    Security: docker inspect command strictly uses list form (shell=False);
    spec fields are never interpolated into a shell string.
    """
    import platform
    import sys as _sys

    if getattr(spec, "require_arch", ""):
        parts = spec.require_arch.split("/", 1)
        req_os = parts[0].lower() if len(parts) >= 1 else ""
        req_arch = parts[1].lower() if len(parts) >= 2 else ""
        cur_platform = _sys.platform.lower()
        cur_machine = platform.machine().lower()
        # Normalize arch: x86_64 == amd64
        machine_aliases = {"x86_64": "amd64", "aarch64": "arm64", "aarch64_be": "arm64"}
        cur_arch_norm = machine_aliases.get(cur_machine, cur_machine)
        req_arch_norm = machine_aliases.get(req_arch, req_arch)

        os_ok = True
        if req_os == "linux":
            os_ok = cur_platform == "linux"
        elif req_os == "darwin":
            os_ok = cur_platform == "darwin"
        elif req_os:
            os_ok = cur_platform.startswith(req_os)

        arch_ok = (not req_arch_norm) or (cur_arch_norm == req_arch_norm)

        if not (os_ok and arch_ok):
            reason = (f"需要 {spec.require_arch}，"
                      f"当前 {_sys.platform}/{platform.machine()}")
            return False, reason

    if getattr(spec, "require_docker_image", ""):
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", spec.require_docker_image],
                shell=False, timeout=5, capture_output=True,
            )
            if proc.returncode != 0:
                return False, f"docker image not found: {spec.require_docker_image}"
        except Exception as e:
            return False, f"docker inspect failed: {e}"

    return True, ""


def _run_agent_wrapper(state: CrewState, agent_id: str) -> None:
    """Agent execution shell: catches exceptions, updates state, detects exit markers.
    Process-level failures (subprocess crashes, etc.) are retried at most once.
    """
    from larkhelm.chat_state import _get_cwd
    from larkhelm.crew._state import _git_auto_commit
    from larkhelm.metrics import inc_crew_preflight

    last_exc: Exception = None
    backend_select_failure: bool = False  # set when NoBackendAvailableError surfaces
    validate_failure: bool = False        # set when _validate_output_artifact fires twice

    # Preflight environment check — early exit before the 2-attempt retry loop.
    import larkhelm.config as _cfg_module
    _spec = state.agents[agent_id].spec
    _ok, _reason = _preflight_env_check(_spec, getattr(_cfg_module, "config", {}))
    if not _ok:
        state.agents[agent_id].status = AgentStatus.SKIPPED
        state.agents[agent_id].skip_reason = _reason
        _check_type = "docker" if getattr(_spec, "require_docker_image", "") else "arch"
        _outcome = "fail_docker" if _check_type == "docker" else "fail_arch"
        inc_crew_preflight(_outcome, _check_type)
        _debug_log(f"[Crew] {agent_id} preflight SKIPPED: {_reason}")
        return
    _pass_check_type = ("arch" if getattr(_spec, "require_arch", "")
                        else "docker" if getattr(_spec, "require_docker_image", "")
                        else "arch")
    inc_crew_preflight("pass", _pass_check_type)

    for proc_attempt in range(2):   # attempt 0 = first try, attempt 1 = retry
        try:
            result = _run_agent(state, agent_id)
            needs_retry = _detect_fail_marker(state.agents[agent_id].spec, result)
            if needs_retry:
                _debug_log(f"[Crew] {agent_id} fail marker detected, pending retry")
            # Output-file safety net: agents are instructed (via the prompt
            # injection in _run_agent) to call the Write tool to persist
            # their full output to ``.crew_workspace/{spec.output_file}``.
            # Reality: some agents emit a short closing marker (the result
            # field captures the last text chunk, often ≤100 chars) and
            # rely on Write — but if the Write call was skipped, truncated,
            # or wrote to the wrong path, the full PRD/design/etc. is gone
            # and downstream agents read an empty / missing file. This
            # observably bit P3 (PM emitted a 39196-token PRD that never
            # landed on disk; architect had to reverse-engineer it from
            # source + memory).
            #
            # Fallback: if the expected file is missing or too small AND
            # the in-memory result is substantially longer, persist
            # ``result`` to the expected path before the sync step runs.
            _persist_result_to_output_file_if_missing(state, agent_id, result)
            # Artifact contract check: if the on-disk file (or in-memory
            # result) contains a tool-call protocol leak or malformed JSON,
            # circuit-break before downstream agents read it. Retry once on
            # the first attempt (transient model glitch); on the second
            # failure fall through to ``emit_agent_failure(stage="validate")``
            # so the user gets a targeted card instead of watching the rest
            # of the pipeline burn tokens on an unusable contract.
            artifact_issue = _validate_output_artifact(state, agent_id, result)
            if artifact_issue:
                if proc_attempt == 0:
                    # F4 (2026-05-25): exclude the failing backend from
                    # the retry's resolve_backend() so we don't burn
                    # another 5 min on the same tool-incapable model.
                    # Validate failures are backend-intrinsic (model can't
                    # tool_use → emits protocol tokens as plain text);
                    # same-backend retry was guaranteed-loss. The retry
                    # falls back to the next-ranked healthy backend that
                    # passes the task_profile filter (or the orchestrator
                    # when nothing else qualifies). Quarantine first so
                    # the safety-net rewrite from attempt 1 doesn't see
                    # the corrupt prior file and short-circuit.
                    _quarantine_invalid_output(state, agent_id)
                    failed_backend_id = state.agents[agent_id].actual_backend_id
                    repeat_swing = False
                    with state.lock:
                        state.agents[agent_id].status     = AgentStatus.PENDING
                        state.agents[agent_id].start_time = None
                        state.agents[agent_id].result     = ""
                        if failed_backend_id:
                            excluded_map = state.agents[agent_id].excluded_backends_until
                            # SEC-v2-MED-2: pre-existing entry (even if
                            # it just expired) means we already excluded
                            # this backend earlier in the crew → re-
                            # failure is the swing signal we report.
                            repeat_swing = failed_backend_id in excluded_map
                            # TTL stamp; cool-down sourced from config
                            # (default 60s, floor 0 = disabled). Reading
                            # _cfg.config lazily keeps the runner import-
                            # safe when config has not been initialised
                            # (test bootstrap).
                            try:
                                import larkhelm.config as _cfg_swing
                                cooldown = max(0.0, float(
                                    getattr(_cfg_swing, "config", {}).get(
                                        "crew_backend_exclusion_cooldown_sec", 60.0,
                                    ) or 60.0
                                ))
                            except (TypeError, ValueError):
                                cooldown = 60.0
                            excluded_map[failed_backend_id] = time.time() + cooldown
                    if repeat_swing:
                        try:
                            from larkhelm.metrics import inc_crew_backend_swing
                            inc_crew_backend_swing(
                                agent_id=agent_id,
                                backend_id=failed_backend_id,
                            )
                        except Exception as e:
                            _debug_log(
                                f"[Crew] {agent_id} swing-metric emit failed: {e}"
                            )
                    warn(
                        f"[Crew] {agent_id} output validation failed "
                        f"({artifact_issue}) — retrying on a different backend "
                        f"(excluded: {failed_backend_id or '<unknown>'}"
                        f"{', swing-repeat' if repeat_swing else ''})"
                    )
                    time.sleep(1)
                    continue
                last_exc = RuntimeError(
                    f"output artifact contract violation: {artifact_issue}"
                )
                validate_failure = True
                # Quarantine the corrupt file BEFORE breaking out, so the
                # scheduler's partial-delivery rule does not see a
                # nominally-present output_file and let downstream agents
                # cascade-read the garbage (regression observed 2026-05-22).
                _quarantine_invalid_output(state, agent_id)
                break
            # Sync output_file before updating state to avoid holding the lock during IO
            feishu_url = _sync_output_file(state, agent_id)
            # B1: PRD self-check gate — fires once after architect writes
            # tasks.json / prd_criteria.json / file_changes.json. On any
            # ``❌`` finding we flip ``needs_retry=True`` and stuff a
            # pointer to ``.crew_workspace/prd_selfcheck.md`` into
            # architect's feedback slot so the retry round sees the
            # diagnosis prepended to its prompt. The architect spec has
            # ``retry_target=["architect"], max_retries=1`` so the
            # scheduler reruns it exactly once; a second failure falls
            # through to existing exhaustion handling (no hard-fail —
            # we'd rather ship a borderline PRD than block the user).
            if not needs_retry and agent_id == "architect":
                try:
                    from larkhelm.crew._prd_selfcheck import run_prd_selfcheck
                    cwd_for_check = _get_cwd(state.chat_id)
                    passed, report = run_prd_selfcheck(cwd_for_check)
                    if not passed:
                        needs_retry = True
                        # Write to feedback inside the lock — same field that
                        # ``_run_agent`` will read on the retry pass.
                        with state.lock:
                            state.agents[agent_id].feedback = (
                                "PRD self-check 失败（B1 gate）。请读取 "
                                ".crew_workspace/prd_selfcheck.md 修正 anchors / "
                                "AC how_to_verify 占位符 / file_changes 路径不一致 "
                                "三类问题后重新生成 tasks.json / prd_criteria.json / "
                                "file_changes.json。"
                                + (f"\n\n以下是首轮自检完整报告（供参考）：\n\n{report}"
                                   if len(report) < 4000 else "")
                            )
                        _debug_log("[Crew] architect PRD self-check FAILED, pending retry")
                except Exception as e:
                    # Self-check is a soft gate — never block the pipeline
                    # if its internals raise. Just log and let architect's
                    # output through.
                    _debug_log(f"[Crew] architect PRD self-check hook raised: {e}")
            # B4: reconcile file_changes.json drift after implementer/fixer.
            # The architect's declared list is best-effort; agents
            # routinely modify files outside the declared scope (tests,
            # neighbouring doc syncs). Reconciling here keeps the
            # artifact honest before B2 finalize reads it for its
            # drift-threshold decision. Fail-soft.
            if not needs_retry and agent_id in ("implementer", "fixer"):
                try:
                    from larkhelm.crew._workspace_reconcile import (
                        reconcile_file_changes,
                    )
                    reconcile_file_changes(_get_cwd(state.chat_id))
                except Exception as e:
                    _debug_log(f"[Crew] workspace reconcile failed ({agent_id}): {e}")
            # B4: stamp schema_version=2 on tasks.json once architect has
            # emitted anchor metadata. No-op when anchors are absent.
            if not needs_retry and agent_id == "architect":
                try:
                    from larkhelm.crew._workspace_reconcile import (
                        stamp_schema_version_on_tasks,
                    )
                    stamp_schema_version_on_tasks(_get_cwd(state.chat_id))
                except Exception as e:
                    _debug_log(f"[Crew] tasks.json schema stamp failed: {e}")
            # Auto-commit after code-modification agents succeed (when dev_auto_commit=true)
            commit_hash = ""
            if not needs_retry and agent_id in ("implementer", "fixer"):
                cwd = _get_cwd(state.chat_id)
                commit_hash = _git_auto_commit(cwd, agent_id)
                if commit_hash:
                    with state.lock:
                        state.phase_commits[agent_id] = commit_hash
                    _debug_log(f"[Git] {agent_id} auto-committed: {commit_hash}")
            # Capture per-agent token stats
            crew_ns = f"{state.chat_id}__crew_{state.crew_id}_{agent_id}"
            try:
                from larkhelm.token_stats import get_crew_agent_tokens
                agent_tokens = get_crew_agent_tokens(crew_ns)
            except Exception:
                agent_tokens = {}
            with state.lock:
                state.agents[agent_id].status         = AgentStatus.DONE
                state.agents[agent_id].result         = result
                state.agents[agent_id].end_time       = time.time()
                state.agents[agent_id].needs_retry    = needs_retry
                state.agents[agent_id].feishu_doc_url = feishu_url
                state.agents[agent_id].tokens         = agent_tokens
            _debug_log(f"[Crew] {agent_id} done ({len(result)} chars)")
            _crew_update_card(state)
            return

        except QueryCancelledError:
            with state.lock:
                state.agents[agent_id].status   = AgentStatus.CANCELLED
                state.agents[agent_id].end_time = time.time()
            _crew_update_card(state)
            return

        except NoBackendAvailableError as e:
            # Config / health failure — retrying won't help, so skip the
            # second attempt and fall straight to the failure card branch
            # below. emit_agent_failure tags this as ``stage="backend_select"``
            # so the user sees the targeted "无可用 backend" hint.
            last_exc = e
            backend_select_failure = True
            break

        except Exception as e:
            last_exc = e
            if proc_attempt == 0:
                # OOM-aware retry backoff. A claude/node CLI killed by the
                # cgroup OOM-killer (rc=-9) leaves the cgroup near its
                # memory.max — page cache and V8 native allocations don't
                # release instantly. Retrying after 1s usually re-trips
                # the same OOM. 8s gives the kernel a chance to reclaim
                # + lets MAX_AI_PROCS=2 wave back below the high-water
                # mark before a second large process starts. Other errors
                # (HTTP transient, parse error, etc.) keep the original
                # 1s — no point waiting 8s on a quick recoverable failure.
                _oom = _is_likely_oom_error(e)
                backoff_sec = 8 if _oom else 1
                if _oom:
                    _debug_log(
                        f"[Crew] {agent_id} OOM-class failure on attempt 1/2 — "
                        f"backing off {backoff_sec}s before retry "
                        f"(let cgroup reclaim memory). Error: {str(e)[:200]}"
                    )
                    _log_oom_diagnostics(agent_id)
                else:
                    _debug_log(f"[Crew] {agent_id} process failed (attempt 1/2), retrying in {backoff_sec}s: {e}")
                time.sleep(backoff_sec)
                with state.lock:   # Reset to PENDING for retry
                    state.agents[agent_id].status     = AgentStatus.PENDING
                    state.agents[agent_id].start_time = None
                    state.agents[agent_id].result     = ""
                continue  # proceed to second attempt

    # Both attempts failed (or first attempt was a non-retryable backend-select
    # failure). Emit the user-facing ⚠️ card via _failure_card so the user
    # actually sees what went wrong; the helper handles state mutation +
    # heartbeat push internally and never raises.
    if validate_failure:
        stage = "validate"
    elif backend_select_failure:
        stage = "backend_select"
    else:
        stage = "run"
    if last_exc is None:
        last_exc = RuntimeError("unknown error")
    emit_agent_failure(state, agent_id, stage, last_exc)


# ═══════════════════════════════════════════════════════════════
#  Breakpoint confirmation
# ═══════════════════════════════════════════════════════════════

def _wait_for_breakpoint(state: CrewState, agent_id: str) -> bool:
    """Pause after PM completes, update main card to show 继续/取消 buttons and wait for the user to click. Returns True=continue, False=cancel."""
    import larkhelm.config as _cfg
    from larkhelm.crew._state import (
        _breakpoint_meta, _breakpoint_events, _breakpoint_results,
    )

    crew_id = state.crew_id

    # Register wait event.
    # Note: if signal_breakpoint is called before this function registers the event (theoretical race),
    # _breakpoint_results already has a value and should not be overwritten; fire bp_ev directly instead.
    bp_ev = threading.Event()
    with _breakpoint_meta:
        _breakpoint_events[crew_id] = bp_ev
        if crew_id in _breakpoint_results:
            # Signal arrived early; wake immediately without overwriting the existing result
            bp_ev.set()
        else:
            _breakpoint_results[crew_id] = False  # Default: cancel on timeout

    # Update main card to breakpoint phase — _build_card shows "继续/取消" buttons for this phase
    with state.lock:
        state.phase              = "breakpoint"
        state.breakpoint_agent_id = agent_id
    _crew_update_card(state)

    # Wait for user decision; poll to support /cancel interruption.
    # Phase C: the deadline is sourced from ``_cfg.CREW_BREAKPOINT_TIMEOUT_SEC``
    # (default 1800s, configurable via crew_breakpoint_timeout_sec) instead of
    # the previous hardcoded ``min(RESPONSE_TIMEOUT*2, 3600)``. On timeout we
    # both set ``cancel_ev`` (so the executor breaks out of its wave loop) and
    # emit a dedicated breakpoint-timeout card (so the user sees why the task
    # ended).
    # Trust whatever ``_cfg.CREW_BREAKPOINT_TIMEOUT_SEC`` resolves to —
    # ``_init_runtime`` already floors the configured value at 60s, and
    # tests need to monkeypatch lower for fast assertions.
    bp_timeout = int(getattr(_cfg, "CREW_BREAKPOINT_TIMEOUT_SEC", 1800))
    bp_deadline = time.time() + bp_timeout
    timed_out = False
    while time.time() < bp_deadline:
        if state.cancel_ev.is_set():
            _debug_log("[Crew] breakpoint wait interrupted by /cancel")
            break
        if bp_ev.wait(timeout=min(2.0, max(0.0, bp_deadline - time.time()))):  # clamp to avoid overshooting deadline
            break
    else:
        # Loop exited via condition (no break) — that's the timeout branch.
        timed_out = True

    with _breakpoint_meta:
        confirmed = _breakpoint_results.pop(crew_id, False)
        _breakpoint_events.pop(crew_id, None)

    if timed_out and not confirmed:
        _debug_log(f"[Crew] breakpoint timeout after {bp_timeout}s, auto-cancelling")
        # REQ-11: persist checkpoint BEFORE cancelling so a restart can resume
        # from the last completed wave, not from the beginning.
        try:
            from larkhelm.crew._checkpoint import _save_checkpoint
            with state.lock:
                completed_ids = [
                    spec.id for spec in state.plan.agents
                    if state.agents[spec.id].status in
                    (AgentStatus.DONE, AgentStatus.FAILED)
                ]
            _save_checkpoint(state, completed_ids, phase="timeout")
        except Exception as _cp_err:
            _debug_log(f"[Crew] breakpoint timeout checkpoint failed: {_cp_err}")
        state.cancel_ev.set()
        emit_breakpoint_timeout(state)
        return False

    return confirmed


# ═══════════════════════════════════════════════════════════════
#  Scheduler
# ═══════════════════════════════════════════════════════════════

def _toposort_agents(agents: list[AgentSpec]) -> list[AgentSpec]:
    """DFS topological sort with cycle detection.

    Returns agents ordered so that all dependencies precede dependents.
    Raises ValueError("cycle detected: A → B → A") if any cycle exists.
    Time complexity: O(V+E).
    """
    id_to_spec = {s.id: s for s in agents}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}
    stack: list[str] = []
    result: list[AgentSpec] = []

    def _visit(node_id: str) -> None:
        if color.get(node_id) == BLACK:
            return
        if color.get(node_id) == GRAY:
            idx = stack.index(node_id)
            cycle = stack[idx:] + [node_id]
            raise ValueError("cycle detected: " + " → ".join(cycle))
        color[node_id] = GRAY
        stack.append(node_id)
        spec = id_to_spec.get(node_id)
        if spec:
            for dep_id in (spec.depends_on or []):
                if dep_id in id_to_spec:
                    _visit(dep_id)
        stack.pop()
        color[node_id] = BLACK
        if spec:
            result.append(spec)

    for spec in agents:
        if color.get(spec.id, WHITE) == WHITE:
            _visit(spec.id)

    return result


def _execute(state: CrewState, total_timeout: int):
    """Execute all agents in topological waves, supporting retry fallback when QA/Reviewer fails."""
    import larkhelm.config as _cfg
    from larkhelm.chat_state import _get_cwd
    from larkhelm.crew._scheduler import (
        _topo_waves, _topo_waves_subset, _get_failed_dep,
    )
    from larkhelm.crew._checkpoint import _save_checkpoint

    cancel_ev    = state.cancel_ev
    deadline     = time.time() + total_timeout

    # Validate DAG before running: DFS cycle detection raises ValueError on cycles.
    _toposort_agents(state.plan.agents)

    # Create per-project Feishu folder before running any agents
    _ensure_crew_folder(state)

    wave_queue   = deque(_topo_waves(state.plan.agents))
    # Agents that were run early as fallbacks; skip if seen again in later waves.
    fb_ran_ids: set[str] = set()
    retry_counts: dict[str, int] = {spec.id: 0 for spec in state.plan.agents}
    qa_retry_rounds: int = 0  # total QA-fail → fixer cycles; capped by plan.max_qa_retry_rounds

    # Workspace path resolved once and passed into ``_get_failed_dep`` for
    # the partial-delivery rule (see scheduler docstring). When an upstream
    # FAILED agent has already written its declared ``output_file``, we
    # let downstream agents run instead of cascading the FAILED into N
    # more skipped agents. Real motivator: implementer's claude CLI gets
    # OOM-killed after the Write tool atomically committed changes.md,
    # but fixer/qa/reviewer were all marked "skipped because upstream
    # implementer failed". Now fixer can actually run.
    try:
        _crew_ws_path = Path(_get_cwd(state.chat_id)) / ".crew_workspace"
    except Exception:
        _crew_ws_path = None

    while wave_queue:
        if cancel_ev.is_set():
            raise QueryCancelledError("Crew cancelled")
        if time.time() > deadline:
            with state.lock:
                for spec in state.plan.agents:
                    if state.agents[spec.id].status == AgentStatus.PENDING:
                        state.agents[spec.id].status = AgentStatus.FAILED
                        state.agents[spec.id].error  = "Crew 总超时"
            break

        wave = wave_queue.popleft()

        # ── Phase 4: failure propagation — skip agents whose upstream is FAILED ──
        runnable: list[AgentSpec] = []
        with state.lock:
            for spec in wave:
                failed_dep = _get_failed_dep(state, spec, _crew_ws_path)
                if failed_dep:
                    _debug_log(f"[Crew] {spec.id} skipped because upstream {failed_dep} failed")
                    ag = state.agents[spec.id]
                    ag.status   = AgentStatus.SKIPPED
                    ag.error    = f"upstream {failed_dep} failed"
                    ag.end_time = time.time()
                elif spec.id in fb_ran_ids:
                    pass  # already ran as fallback; do not re-queue
                elif spec.trigger_only and state.agents[spec.id].retry_count == 0:
                    # Skip on first wave: mark as DONE (trigger pending), do not actually execute
                    ag = state.agents[spec.id]
                    ag.status   = AgentStatus.DONE
                    ag.result   = ""   # empty result, sentinel
                    ag.end_time = time.time()
                    _debug_log(f"[Crew] {spec.id} trigger_only, skipping first wave")
                else:
                    runnable.append(spec)

        if not runnable:
            _crew_update_card(state)
            continue

        threads = [
            threading.Thread(
                target=_run_agent_wrapper,
                args=(state, spec.id),
                daemon=True,
                name=f"crew-{spec.id}-{state.crew_id[:6]}",
            )
            for spec in runnable
        ]
        for t in threads:
            t.start()

        # Wait for this wave to complete; check cancel every 0.5s
        done_ev = threading.Event()
        def _waiter(ts=threads):
            for t in ts:
                t.join()
            done_ev.set()
        threading.Thread(target=_waiter, daemon=True).start()
        while not done_ev.is_set():
            if cancel_ev.is_set():
                raise QueryCancelledError("Crew cancelled")
            done_ev.wait(timeout=0.5)

        # ── Check if any agent in this wave needs retry (read needs_retry under lock) ──
        retry_specs: list[AgentSpec] = []
        with state.lock:
            for spec in wave:
                if state.agents[spec.id].needs_retry:
                    retry_specs.append(spec)

        for spec in retry_specs:
            ag = state.agents[spec.id]

            # QA-specific: parse structured verdict, enforce plan-level retry cap
            _qa_verdict_prefix = ""
            if spec.id == "qa":
                _qv = _parse_qa_verdict(ag.result)
                _qv_str = _qv.get("verdict", "UNKNOWN")
                _qv_fc  = _qv.get("failed_count", 0)
                _qv_bc  = _qv.get("blocked_count", 0)
                _qv_sc  = _qv.get("skip_count", 0)
                _qa_verdict_prefix = (
                    f"QA Verdict: {_qv_str} — FAILED={_qv_fc} "
                    f"BLOCKED={_qv_bc} SKIP={_qv_sc}\n"
                )
                qa_retry_rounds += 1
                if qa_retry_rounds > state.plan.max_qa_retry_rounds:
                    _debug_log(
                        f"[Crew] qa exceeded plan max_qa_retry_rounds "
                        f"({state.plan.max_qa_retry_rounds}), forcing FAILED"
                    )
                    with state.lock:
                        ag.needs_retry = False
                        ag.status = AgentStatus.FAILED
                        ag.error  = "超过计划重试上限"
                    wave_queue.clear()
                    break

            if retry_counts[spec.id] >= spec.max_retries:
                _debug_log(f"[Crew] {spec.id} reached max retries ({spec.max_retries}), giving up")
                with state.lock:
                    ag.needs_retry = False
                if spec.is_gatekeeper:
                    # Gatekeeper final rejection: clear remaining queue and go straight to synthesis
                    _debug_log(f"[Crew] gatekeeper {spec.id} final rejection, skipping remaining tasks")
                    wave_queue.clear()
                # Fallback agent: when retries exhausted and fallback_agent_id is configured,
                # run the fallback synchronously instead of hard-failing.
                _fb_id = getattr(spec, "fallback_agent_id", "") or ""
                if _fb_id and _fb_id in state.agents:
                    _debug_log(f"[Crew] {spec.id} retries exhausted, switching to fallback {_fb_id}")
                    with state.lock:
                        _fb = state.agents[_fb_id]
                        if _fb.status != AgentStatus.RUNNING:
                            _fb.status     = AgentStatus.PENDING
                            _fb.result     = ""
                            _fb.error      = ""
                            _fb.start_time = None
                            _fb.end_time   = None
                    _run_agent_wrapper(state, _fb_id)
                    fb_ran_ids.add(_fb_id)
                    continue  # fallback handled; skip hard_fail_on_exhaust
                if spec.hard_fail_on_exhaust:
                    raise HardFailError(f"{spec.role}（{spec.id}）最终失败，已重试 {spec.max_retries} 次")
                continue

            # Not enough remaining time for a retry round (estimate: retry rounds × RESPONSE_TIMEOUT)
            retry_num = retry_counts[spec.id] + 1
            est_needed = (len(spec.retry_target) + 1) * _cfg.RESPONSE_TIMEOUT
            if time.time() + est_needed > deadline:
                _debug_log(f"[Crew] {spec.id} insufficient time remaining (~{est_needed}s needed), skipping retry #{retry_num}")
                with state.lock:
                    ag.needs_retry = False
                if spec.hard_fail_on_exhaust:
                    raise HardFailError(f"{spec.role} ({spec.id}) retries abandoned due to timeout")
                continue

            retry_counts[spec.id] += 1
            _debug_log(f"[Crew] {spec.id} triggering retry #{retry_counts[spec.id]}, resetting: {spec.retry_target}")

            # Build feedback: if the triggering agent has an output_file, pass only the file path reference.
            # (The retry_target agent's system prompt already instructs it to read that file;
            #  passing content directly would double-transmit and waste ~1000-3000 tokens/retry.)
            # If no output_file (/crew dynamic scenario), pass the beginning of the output (errors are typically at the top).
            _cwd_fb = _get_cwd(state.chat_id)
            if ag.spec.output_file:
                _fb_path = Path(_cwd_fb) / ".crew_workspace" / ag.spec.output_file
                feedback = (
                    _qa_verdict_prefix
                    + f"Failure report: {_fb_path}\n"
                    + f"Use the Read tool to read this file for details, then fix each issue."
                )
            else:
                # Take the beginning (bug list is typically first) rather than the end (usually just TESTS_FAILED etc.)
                feedback = _qa_verdict_prefix + (ag.result[:2000] if ag.result else ag.error[:1000])

            # Reset this agent + agents in retry_target
            targets = list(spec.retry_target) + [spec.id]
            with state.lock:
                for tid in targets:
                    if tid not in state.agents:
                        continue
                    ta = state.agents[tid]
                    ta.status     = AgentStatus.PENDING
                    ta.result     = ""
                    ta.error      = ""
                    ta.start_time = None
                    ta.end_time   = None
                    ta.needs_retry = False
                    ta.retry_count += 1
                    ta.round_label = f"Round {ta.retry_count + 1}"
                    # F4 follow-up + SEC-v2-MED-2 (review_security_v2):
                    # prune ONLY entries whose TTL has lapsed. Previously
                    # this was an unconditional ``.clear()`` so a
                    # transiently-unhappy backend could recover by round
                    # 2 — but that also let an attacker swing the pool
                    # (force backend A to validate-fail → A excluded →
                    # B succeeds → QA fails → clear wipes A → next
                    # round A admitted → A fails again, burning tokens
                    # each round). The TTL prune keeps a freshly-failed
                    # backend out across the immediate retry; legitimate
                    # transient recoveries still re-enter once their
                    # cool-down lapses (``crew_backend_exclusion_cooldown_sec``,
                    # default 60s; 0 restores pre-fix unconditional clear).
                    _now_prune = time.time()
                    ta.excluded_backends_until = {
                        bid: exp for bid, exp
                        in ta.excluded_backends_until.items()
                        if exp > _now_prune
                    }
                    # Only inject feedback into upstream retry_target agents (e.g. engineer), not into self
                    if tid != spec.id:
                        ta.feedback = feedback

            # Re-enqueue the retry agent subset
            retry_waves = _topo_waves_subset(state.plan.agents, set(targets))
            wave_queue.extendleft(reversed(retry_waves))
            _crew_update_card(state)
            break  # At most one retry trigger per wave

        # ── Save checkpoint: this wave is complete ────────────────────
        if not retry_specs:
            completed_ids = [spec.id for spec in state.plan.agents
                             if state.agents[spec.id].status in
                             (AgentStatus.DONE, AgentStatus.FAILED)]
            _save_checkpoint(state, completed_ids)

        # ── PM short-circuit: ``TASK_ALREADY_COMPLETE`` marker ─────────
        # When PM (or any planner-tier first-wave agent) detects that the
        # task's acceptance points are already met by the current codebase,
        # it emits this single-line marker. Drain the remaining waves —
        # downstream agents would just waste tokens and time re-discovering
        # "nothing to do". The synthesis phase still runs so the user gets
        # a nice "already complete" card with PM's reason.
        if not retry_specs and _check_task_already_complete(state, wave):
            with state.lock:
                for _spec in state.plan.agents:
                    _ag = state.agents.get(_spec.id)
                    if _ag and _ag.status == AgentStatus.PENDING:
                        _ag.status = AgentStatus.SKIPPED
                        _ag.error = "TASK_ALREADY_COMPLETE — PM 判定任务已在代码中完成"
            wave_queue.clear()
            break

        # ── Phase 3.1: breakpoint — pause after an agent with breakpoint=True completes ──
        if not retry_specs:  # Only check breakpoints when there are no retries (skip during retry)
            for spec in wave:   # Iterate original wave; state guard ensures only DONE agents trigger
                ag = state.agents.get(spec.id)
                if (spec.breakpoint and ag and ag.status == AgentStatus.DONE):
                    confirmed = _wait_for_breakpoint(state, spec.id)
                    if not confirmed:
                        raise QueryCancelledError("User cancelled")
                    # Restore main card to running state
                    with state.lock:
                        state.phase = "running"
                    _crew_update_card(state)
                    break


# ── PM TASK_ALREADY_COMPLETE marker ─────────────────────────────────────

_TASK_COMPLETE_MARKER = "TASK_ALREADY_COMPLETE"


def _extract_task_complete_marker(state: CrewState) -> "tuple[bool, str]":
    """Single source of truth for the marker detection. Returns
    ``(hit, reason)`` where:

      • ``hit`` is True iff PM is DONE, not a retry, AND the marker
        appears at the start of either PM's in-memory result OR the
        first non-blank line of ``.crew_workspace/prd.md``.
      • ``reason`` is the trailing free-text after ``TASK_ALREADY_COMPLETE:``
        (empty if no colon / nothing after).

    Round-2 review caught that ``_check_task_already_complete`` had the
    prd.md fallback but ``_synthesize`` only inspected ``pm_state.result``,
    so PM honouring the system-prompt contract by writing only the file
    (a very common shape — PM emits "PRD written." as result after the
    Write tool call) would correctly short-circuit ``_execute`` but
    silently trigger the LLM synthesis path. Sharing this helper
    guarantees the two phases never disagree.
    """
    pm_state = state.agents.get("pm")
    if not pm_state or pm_state.status != AgentStatus.DONE:
        return False, ""
    if pm_state.retry_count > 0:
        return False, ""

    # Primary source: in-memory result. Fallback: first non-blank line of
    # prd.md (covers the case where the agent honoured the system-prompt
    # contract by writing the marker to the file but the in-memory result
    # is a short closing summary).
    candidates: list[str] = []
    head = (pm_state.result or "").lstrip()
    if head:
        candidates.append(head)
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(state.chat_id)
        prd_path = Path(cwd) / ".crew_workspace" / "prd.md"
        if prd_path.exists():
            file_head = prd_path.read_text(encoding="utf-8").lstrip()
            if file_head:
                candidates.append(file_head)
    except Exception:
        pass

    for cand in candidates:
        if cand.startswith(_TASK_COMPLETE_MARKER):
            first_line = cand.splitlines()[0].strip()
            reason = first_line.removeprefix(_TASK_COMPLETE_MARKER).lstrip(": ").strip()
            return True, reason
    return False, ""


def _check_task_already_complete(state: CrewState, wave) -> bool:
    """Return True iff a just-completed first-wave PM agent emitted the
    ``TASK_ALREADY_COMPLETE: <reason>`` marker, signalling that the
    codebase already satisfies the user's acceptance points.

    Only fires when:
      • the wave contains an agent with id ``"pm"`` (dev pipeline shape;
        /crew dynamic plans are unaffected) — guard so this helper can't
        misfire on a non-dev plan whose first agent happens to be named
        differently.
      • the marker is found in either PM's in-memory result or in
        ``.crew_workspace/prd.md`` (see ``_extract_task_complete_marker``).

    The check is read-only — the caller is responsible for marking the
    remaining agents as SKIPPED.
    """
    if not any(s.id == "pm" for s in wave):
        return False
    hit, _reason = _extract_task_complete_marker(state)
    return hit


def _execute_from(state: CrewState, total_timeout: int, skip_ids: set):
    """Continue execution after a set of already-completed agents, skipping agents in skip_ids."""
    from larkhelm.chat_state import _get_cwd
    from larkhelm.crew._scheduler import _topo_waves, _get_failed_dep
    from larkhelm.crew._checkpoint import _save_checkpoint

    cancel_ev  = state.cancel_ev
    deadline   = time.time() + total_timeout
    all_waves  = _topo_waves(state.plan.agents)
    wave_queue = deque()

    # Same partial-delivery context as ``_execute``; see commentary there.
    try:
        _crew_ws_path = Path(_get_cwd(state.chat_id)) / ".crew_workspace"
    except Exception:
        _crew_ws_path = None

    for wave in all_waves:
        # If all agents in this wave are already completed, skip the wave
        if all(spec.id in skip_ids for spec in wave):
            continue
        wave_queue.append(wave)

    # Skipped agent states are already restored from checkpoint; keep them as-is.
    # Reset incomplete agents to PENDING (they may have been interrupted mid-run last time).
    with state.lock:
        for spec in state.plan.agents:
            if spec.id not in skip_ids:
                ag = state.agents[spec.id]
                if ag.status not in (AgentStatus.DONE, AgentStatus.FAILED):
                    ag.status     = AgentStatus.PENDING
                    ag.result     = ""
                    ag.error      = ""
                    ag.start_time = None
                    ag.end_time   = None

    state.phase = "running"
    _crew_update_card(state)

    while wave_queue:
        if cancel_ev.is_set():
            raise QueryCancelledError("Crew cancelled")
        if time.time() > deadline:
            break

        wave = wave_queue.popleft()
        runnable: list[AgentSpec] = []
        with state.lock:
            for spec in wave:
                if spec.id in skip_ids:
                    continue
                failed_dep = _get_failed_dep(state, spec, _crew_ws_path)
                if failed_dep:
                    ag = state.agents[spec.id]
                    ag.status = AgentStatus.SKIPPED
                    ag.error  = f"upstream {failed_dep} failed"
                    ag.end_time = time.time()
                elif spec.trigger_only and state.agents[spec.id].retry_count == 0:
                    ag = state.agents[spec.id]
                    ag.status  = AgentStatus.DONE
                    ag.result  = ""
                    ag.end_time = time.time()
                else:
                    runnable.append(spec)

        if not runnable:
            _crew_update_card(state)
            continue

        threads = [
            threading.Thread(target=_run_agent_wrapper, args=(state, spec.id),
                             daemon=True, name=f"crew-{spec.id}-{state.crew_id[:6]}")
            for spec in runnable
        ]
        for t in threads:
            t.start()
        done_ev = threading.Event()
        def _waiter(ts=threads):
            for t in ts: t.join()
            done_ev.set()
        threading.Thread(target=_waiter, daemon=True).start()
        while not done_ev.is_set():
            if cancel_ev.is_set():
                raise QueryCancelledError("Crew cancelled")
            done_ev.wait(timeout=0.5)

        # Save checkpoint: this wave is complete
        completed = [spec.id for spec in state.plan.agents
                     if state.agents[spec.id].status in
                     (AgentStatus.DONE, AgentStatus.FAILED)]
        _save_checkpoint(state, completed)


# ═══════════════════════════════════════════════════════════════
#  Synthesis (Manager summary)
# ═══════════════════════════════════════════════════════════════

def _synthesize(state: CrewState) -> str:
    """Manager synthesizes all agent results and produces the final deliverable."""
    from larkhelm.chat_state import _get_cwd
    from larkhelm.perm import grant_yolo, revoke_yolo
    from larkhelm.backend_cli import run_claude as _bc_run_claude
    from larkhelm.backend_registry import BACKEND_REGISTRY as _reg

    cancel_ev = state.cancel_ev
    cwd       = _get_cwd(state.chat_id)

    # ── TASK_ALREADY_COMPLETE short-circuit ─────────────────────────────
    # When PM short-circuited the pipeline (see ``_check_task_already_complete``)
    # the downstream agents are SKIPPED — there's nothing meaningful to
    # synthesise. Return a templated card body so the user sees a clean
    # "已完成" message instead of a synthesis-LLM hallucinating about empty
    # inputs.
    #
    # Round-2 review caught that this branch previously only inspected
    # ``pm_state.result``, but ``_check_task_already_complete`` also
    # falls back to prd.md. The asymmetry meant a PM that wrote the
    # marker only to the file (the more common shape: PM calls Write,
    # then emits "Done." as the in-memory result) correctly short-
    # circuited ``_execute`` but quietly hit this branch's "no" path
    # and ran a full synthesis LLM call. Fixed by sharing
    # ``_extract_task_complete_marker``.
    _marker_hit, _marker_reason = _extract_task_complete_marker(state)
    if _marker_hit:
        skipped_ids = [s.id for s in state.plan.agents
                       if state.agents[s.id].status == AgentStatus.SKIPPED]
        return (
            "**✅ 任务已经在代码中完成**\n\n"
            f"PM 判定：{_marker_reason or '（PM 未给出详细说明）'}\n\n"
            f"已跳过的阶段：{', '.join(skipped_ids) if skipped_ids else '（无）'}\n\n"
            "若你认为这个判断不对，请用 `/dev --no-confirm <更具体的需求>` 强制重跑。"
        )

    # No synthesis_prompt and only one agent → return its result directly
    plan = state.plan
    if not plan.synthesis_prompt and len(plan.agents) == 1:
        only = list(state.agents.values())[0]
        return only.result if only.status == AgentStatus.DONE else only.error

    # Build synthesis prompt.
    # Agents with output_file: pass only the path (Claude will use Read to retrieve it, avoiding double transmission).
    # Agents without output_file: pass a summary (the only way to get their result).
    parts = []
    _cwd_synth = _get_cwd(state.chat_id)
    for spec in plan.agents:
        a = state.agents[spec.id]
        if a.status == AgentStatus.DONE:
            if spec.output_file:
                out_path = Path(_cwd_synth) / ".crew_workspace" / spec.output_file
                feishu_note = f"\n飞书文档：{a.feishu_doc_url}" if a.feishu_doc_url else ""
                parts.append(f"## {spec.role} ({spec.id})\nOutput file: {out_path}{feishu_note}")
            else:
                workspace   = _workspace_dir(state.chat_id, state.crew_id)
                result_file = workspace / f"{spec.id}_result.txt"
                preview     = a.result[:CREW_RESULT_PREVIEW]
                suffix      = "…" if len(a.result) > CREW_RESULT_PREVIEW else ""
                parts.append(
                    f"## {spec.role} ({spec.id})\n{preview}{suffix}\n"
                    f"Full output: {result_file}"
                )
        elif a.status == AgentStatus.FAILED:
            # F5 (2026-05-25): when validate quarantined an output_file to
            # ``<name>.invalid``, try to recover sanitized prose for the
            # synth so good content isn't thrown away on a false-positive
            # quarantine (e.g. SecurityExpert 2026-05-25: 27 KB legitimate
            # report quarantined for an inline-backtick mention of a
            # sentinel token). The recovered text is run through the same
            # ``_strip_code_evidence`` scrubber AND has lines containing
            # sentinels dropped wholesale, so the synth never sees raw
            # protocol tokens.
            recovered = ""
            if spec.output_file:
                inv_path = Path(_cwd_synth) / ".crew_workspace" / f"{spec.output_file}.invalid"
                if inv_path.exists():
                    try:
                        raw_inv = inv_path.read_text(encoding="utf-8", errors="replace")
                        recovered = _sanitize_quarantined_content(raw_inv)
                    except Exception as _re:
                        _debug_log(f"[Crew] {spec.id}: read .invalid failed: {_re}")
            if recovered:
                preview = recovered[:CREW_RESULT_PREVIEW]
                suffix  = "…(truncated)" if len(recovered) > CREW_RESULT_PREVIEW else ""
                parts.append(
                    f"## {spec.role} ({spec.id})\n"
                    f"⚠️ Output quarantined as `{spec.output_file}.invalid` "
                    f"(reason: {a.error[:120]}); sanitized excerpt below:\n\n"
                    f"{preview}{suffix}"
                )
            elif a.result:
                # Partial output from timeout or similar; include in synthesis, annotate truncation
                preview = a.result[:CREW_RESULT_PREVIEW]
                suffix  = "…(truncated due to timeout)" if len(a.result) > CREW_RESULT_PREVIEW else "(truncated due to timeout)"
                parts.append(f"## {spec.role} ({spec.id})\n⚠️ Incomplete execution, partial output:\n{preview}{suffix}")
            else:
                parts.append(f"## {spec.role} ({spec.id})\nExecution failed: {a.error}")

    if not parts:
        return "All agents produced no results."

    synthesis_prompt = (plan.synthesis_prompt or "Please synthesize the outputs of all agents above and produce the final delivery report.")
    full_prompt      = synthesis_prompt + "\n\n" + "\n\n".join(parts)

    synth_ns = f"{state.chat_id}__crew_{state.crew_id}_synth"
    synth_cancel = threading.Event()

    def _watch_cancel():
        while not synth_cancel.is_set():
            if cancel_ev.is_set():
                synth_cancel.set()
                return
            time.sleep(0.3)
    threading.Thread(target=_watch_cancel, daemon=True).start()

    grant_yolo(synth_ns)
    _synth_spec = _reg.get_orchestrator()
    if _synth_spec is None:
        raise RuntimeError("No orchestrator backend available for synthesis")
    _API_PROVIDERS = ("anthropic_api", "google_api", "openai_compat_api")
    usage_holder: dict = {}
    model_label = _synth_spec.model or _synth_spec.id
    try:
        if _synth_spec.provider in _API_PROVIDERS:
            import larkhelm.backend_api as _bapi
            _fn = {"anthropic_api": _bapi.run_anthropic,
                   "google_api": _bapi.run_google,
                   "openai_compat_api": _bapi.run_openai_compat}[_synth_spec.provider]
            result, _ = _fn(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                            history=[], cancel_ev=synth_cancel, on_text=None,
                            suppress_token_recording=True, usage_holder=usage_holder)
        elif _synth_spec.provider == "gemini_cli":
            from larkhelm.backend_cli import run_gemini as _bc_run_gemini
            result = _bc_run_gemini(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                                    sid=None, cwd=cwd, cancel_ev=synth_cancel,
                                    on_text=None, use_session=False,
                                    suppress_token_recording=True, usage_holder=usage_holder)
        elif _synth_spec.provider == "kimi_cli":
            from larkhelm.backend_cli import run_kimi as _bc_run_kimi
            result = _bc_run_kimi(spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                                  sid=None, cwd=cwd, cancel_ev=synth_cancel, on_text=None,
                                  suppress_token_recording=True, usage_holder=usage_holder)
        else:
            result = _bc_run_claude(
                spec=_synth_spec, chat_id=synth_ns, message=full_prompt,
                sid=None, cwd=cwd, cancel_ev=synth_cancel,
                on_text=None, allow_retry=False, session_namespace=synth_ns,
                suppress_token_recording=True, usage_holder=usage_holder,
            )
        if usage_holder:
            from larkhelm.token_stats import record_token_usage
            record_token_usage(state.chat_id, model_label, usage_holder)
    finally:
        synth_cancel.set()
        revoke_yolo(synth_ns)

    return result.strip() or "Synthesis phase produced no output."


# ═══════════════════════════════════════════════════════════════
#  Main flow
# ═══════════════════════════════════════════════════════════════

def _run_crew(state: CrewState, total_timeout: int):
    """Complete crew execution flow (runs in a dedicated thread)."""
    from larkhelm.chat_state import _get_cwd
    from larkhelm.card_builder import _fmt_elapsed, _split_md, _make_card
    from larkhelm.lark_client import _reply_card_raw, send_card
    from larkhelm.crew._state import _register_crew_card
    from larkhelm.crew._checkpoint import _clear_checkpoint

    hb_stop   = threading.Event()
    hb_thread = _start_heartbeat(state, hb_stop)

    try:
        # ── Execute agents ───────────────────────────────
        with state.lock:
            state.phase = "running"
        _crew_update_card(state)

        try:
            _execute(state, total_timeout)
        except QueryCancelledError:
            with state.lock:
                state.phase = "cancelled"
            _crew_update_card(state)
            # Clear checkpoint so the next bridge restart doesn't auto-resume
            # a crew the user already gave up on (e.g. breakpoint timeout,
            # /cancel button, /cancel text command). Exception: during a
            # SIGTERM shutdown, ``cancel_all_crews`` deliberately re-saved
            # the checkpoint with ``phase="running"`` *immediately before*
            # setting ``cancel_ev``, precisely so ``resume_interrupted_crews``
            # can pick up where we left off after the restart. Preserve that.
            from larkhelm.concurrency import is_shutting_down
            if not is_shutting_down():
                _clear_checkpoint(state.chat_id)
            return
        except HardFailError as e:
            with state.lock:
                state.phase = "failed"
                state.final_output = f"❌ Pipeline hard failure: {e}"
            _crew_update_card(state)
            log_entry(state.chat_id, "error", str(e), model="crew")
            # Hard-fail is terminal — never resume.
            _clear_checkpoint(state.chat_id)
            return

        if state.cancel_ev.is_set():
            with state.lock:
                state.phase = "cancelled"
            _crew_update_card(state)
            from larkhelm.concurrency import is_shutting_down
            if not is_shutting_down():
                _clear_checkpoint(state.chat_id)
            return

        # ── Synthesis ────────────────────────────────────
        with state.lock:
            state.phase = "synthesizing"
        _crew_update_card(state)

        try:
            final = _synthesize(state)
        except Exception as e:
            _debug_log(f"[Crew] synthesis failed: {e}")
            # Fall back to concatenating all completed results when synthesis fails
            final = "\n\n---\n\n".join(
                f"**{state.agents[spec.id].spec.role}**\n{state.agents[spec.id].result}"
                for spec in state.plan.agents
                if state.agents[spec.id].status == AgentStatus.DONE
            ) or "Synthesis failed, no results available."

        # ── Git change statistics (appended to delivery report) ──────
        try:
            _cwd = _get_cwd(state.chat_id)
            # Prefer uncommitted changes; if already committed, show the most recent commit's changes
            _diff = subprocess.run(
                ["git", "diff", "--stat", "HEAD"],
                cwd=_cwd, capture_output=True, text=True, timeout=10,
            )
            _diff_out = _diff.stdout.strip()
            if not _diff_out:
                # Engineer may have already committed; check the most recent commit
                _log = subprocess.run(
                    ["git", "log", "-1", "--stat", "--no-merges", "--format="],
                    cwd=_cwd, capture_output=True, text=True, timeout=10,
                )
                _diff_out = _log.stdout.strip()
            if _diff_out:
                final = final + "\n\n---\n\n**📊 Code change statistics**\n```\n" + _diff_out + "\n```"
        except Exception:
            pass  # Silently ignore: non-git repo, git not installed, timeout, etc.

        # Append Feishu folder link, per-agent doc links, and git commit info
        _cwd_val = _get_cwd(state.chat_id)
        from larkhelm.crew._state import _git_head
        _ws_note = "\n\n---\n\n"
        if state.feishu_folder_url:
            _ws_note += f"📁 [飞书项目文件夹]({state.feishu_folder_url})"
        else:
            _ws_note += f"📁 Workspace: `{_cwd_val}/.crew_workspace/`"
        # List per-agent Feishu doc links (deduplicated: same URL only once)
        _seen_urls: set[str] = set()
        _doc_links = []
        for spec in state.plan.agents:
            _ag = state.agents[spec.id]
            if _ag.feishu_doc_url and _ag.feishu_doc_url not in _seen_urls:
                _seen_urls.add(_ag.feishu_doc_url)
                _doc_links.append(f"[{spec.role}]({_ag.feishu_doc_url})")
        if _doc_links:
            _ws_note += "\n📄 " + "  ·  ".join(_doc_links)
        if state.phase_commits:
            _commits_str = "  ".join(
                f"`{aid}:{h}`" for aid, h in state.phase_commits.items()
            )
            _ws_note += f"\n🔖 Auto-committed: {_commits_str}"
        elif state.git_head_before:
            _cur = _git_head(_cwd_val)
            if _cur and _cur != state.git_head_before:
                _ws_note += f"\n📝 Code changed (start: `{state.git_head_before}` → current: `{_cur}`)"
        final = final + _ws_note

        # Stop the heartbeat before writing the terminal state.  Without this,
        # a heartbeat that read the old phase ("synthesizing") just before we set
        # "done" can complete its API call after _crew_update_card and overwrite
        # the final card.
        hb_stop.set()
        hb_thread.join(timeout=10.0)

        with state.lock:
            state.phase        = "done"
            state.final_output = final
        _crew_update_card(state)

        # Completion notification: reply to the original user message so the user receives an alert
        _label   = "Dev" if state.kind == "dev" else "Crew"
        elapsed  = _fmt_elapsed(time.time() - state.start_time)
        n_agents = len(state.plan.agents)
        notify_body = (
            f"**{state.plan.title}** completed\n\n"
            f"{n_agents} agents · elapsed {elapsed}"
        )
        notify_card = _make_card(f"✅ {_label} complete", notify_body, color="green")
        if state.trigger_msg_id:
            notify_mid = _reply_card_raw(state.trigger_msg_id, notify_card, in_thread=False)
        else:
            notify_mid = send_card(state.chat_id, f"✅ {_label} complete", notify_body, color="green")

        # Extra-long output: send additional cards (filter empty chunks to avoid blank cards)
        chunks = _split_md(final)
        if len(chunks) > 1:
            for chunk in chunks[1:]:
                if chunk.strip():
                    send_card(state.chat_id, "📄 Continued (Crew output)", chunk, color="green")

        log_entry(state.chat_id, "assistant", final, model="crew")
        _clear_checkpoint(state.chat_id)
        # Register both progress card and notification card in crew_card_index so replying to either hits context
        if state.card_mid:
            _register_crew_card(state.card_mid, state.chat_id, state.plan.title, final)
        if notify_mid:
            _register_crew_card(notify_mid, state.chat_id, state.plan.title, final)

    finally:
        hb_stop.set()
