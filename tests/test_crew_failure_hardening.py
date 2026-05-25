"""F1-F6 hardening batch (2026-05-25): tests for the five fixes to the
crew agent validate-failure surface.

Context (post-mortem): a Crew run dispatched DocsExpert to DeepSeek (a
backend without tool capability), which emitted DSML protocol tokens as
plain text → safety-net captured those as the file → validator
quarantined. SecurityExpert (also tool-capable Claude) succeeded with
27 KB of legitimate analysis, but mentioned the DSML token inside an
inline backtick code span while citing DocsExpert's failure — old
``_strip_fenced_code_blocks`` only stripped triple-backtick fences,
so the inline citation tripped the sentinel and the entire 27 KB report
was false-positive quarantined. No red banner ever surfaced. This file
pins regression coverage for the five fixes:

  * F1+F2 — :func:`_strip_code_evidence` strips inline `…` and ``>``
    blockquote citations too; sentinel scan runs full-content instead
    of 8 KiB cap.
  * F3   — :func:`crew/_backend_resolver.resolve_backend` Path 2 (legacy
    direct-dispatch) gates ``output_file``-bearing agents against
    backends without ``"tools"`` in tags.
  * F4   — :func:`_run_agent_wrapper` validate retry routes through a
    fresh ``resolve_backend(... exclude_backend_ids=...)`` call so the
    second attempt picks a different backend.
  * F5   — :func:`_synthesize` reads ``<output_file>.invalid`` for
    FAILED agents and includes a sanitized excerpt in the synth prompt.
  * F6   — :func:`emit_agent_failure` sends a standalone red banner
    (throttled per ``(crew_id, agent_id)``) for actionable stages.
"""
from __future__ import annotations

import json
from pathlib import Path


def _patch_cwd(monkeypatch, tmp_path: Path) -> None:
    from larkhelm import chat_state
    monkeypatch.setattr(chat_state, "_get_cwd", lambda chat_id: str(tmp_path))


def _write_workspace_file(tmp_path: Path, name: str, content: str) -> Path:
    ws = tmp_path / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    p = ws / name
    p.write_text(content, encoding="utf-8")
    return p


# ── F1 — inline backtick + blockquote scrubbing ───────────────────────


def test_validate_strips_inline_backtick_citation(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """The 2026-05-25 SecurityExpert false positive: a reviewer mentioned
    ``` `<｜｜DSML｜｜tool_calls>` ``` inline while citing the
    DocsExpert failure. Pre-F1 the scrubber only handled triple-fence
    blocks and the inline mention tripped the sentinel scan; post-F1
    it must NOT trip.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["sec"])
    state.agents["sec"].spec = fake_agent_spec(id="sec", output_file="review_security.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "review_security.md",
        "# Security review\n\n"
        "review_docs.md.invalid 是 DeepSeek 错误吐出的 "
        "`<｜｜DSML｜｜tool_calls>` 文本，不是真实 QA 报告。\n\n"
        "其余检查通过。\n",
    )
    assert _validate_output_artifact(state, "sec", result="ack") == ""


def test_validate_strips_blockquote_citation(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Block-quote (``>``) is a common way reviewers cite raw model
    output. Must not trip sentinel scan.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["qa"])
    state.agents["qa"].spec = fake_agent_spec(id="qa", output_file="qa_report.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "qa_report.md",
        "## Observed failure\n\n"
        "> <｜｜DSML｜｜tool_calls>\n"
        "> <｜｜DSML｜｜invoke name=\"Read\">\n\n"
        "Conclusion: backend can't tool_use; switched to claude.\n",
    )
    assert _validate_output_artifact(state, "qa", result="ack") == ""


def test_validate_still_catches_unquoted_leak_after_8kib_boundary(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """F2: scan must NOT cap at 8 KiB. A legitimate leak past the old
    boundary (large report with DSML at the tail) used to escape — now
    it must trip.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)
    big_prose = ("legit prose line.\n" * 1000)  # ~ 18 KiB
    _write_workspace_file(
        tmp_path, "design.md",
        big_prose + "\n<｜｜DSML｜｜tool_calls>\n",  # leak past 8 KiB
    )
    issue = _validate_output_artifact(state, "arch", result="ack")
    assert issue
    assert "sentinel" in issue


def test_validate_still_catches_unquoted_inline_leak(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Pin: a real leak NOT wrapped in any quote form still trips."""
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "prd.md",
        "I need to read the file first.\n\n"
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"Read\">\n",
    )
    issue = _validate_output_artifact(state, "pm", result="ack")
    assert issue
    assert "sentinel" in issue


# ── F3 — require_tools gate on Path 2 ──────────────────────────────────


def test_resolve_backend_path2_rejects_non_tool_backend_when_output_file(
    init_test_config, fake_agent_spec, monkeypatch,
):
    """When Manager LLM emits ``model="deepseek"`` for an agent with
    ``output_file=...``, Path 2 (legacy direct dispatch) must refuse the
    tool-incapable backend and fall through to the orchestrator. Pre-F3
    this gap let DocsExpert get DeepSeek, which then emitted DSML
    tokens as plain text. Post-F3 the resolver picks the orchestrator
    (Claude in this test fixture) instead.
    """
    from larkhelm.crew._backend_resolver import resolve_backend
    from larkhelm.backend_registry import BackendRegistry, BackendSpec
    import larkhelm.backend_registry as _br_mod

    reg = BackendRegistry()
    reg._specs["claude"] = BackendSpec(
        id="claude", provider="anthropic_api",
        display_name="Claude", role="orchestrator",
        tags=["tools"], healthy=True, enabled=True,
    )
    reg._specs["deepseek"] = BackendSpec(
        id="deepseek", provider="deepseek_api",
        display_name="DeepSeek", role="worker",
        tags=["cheap", "fast"], healthy=True, enabled=True,
    )
    monkeypatch.setattr(_br_mod, "BACKEND_REGISTRY", reg)

    spec = fake_agent_spec(
        id="docs", role="文档专家",
        model="deepseek", task_profile="",   # Path 2 entry condition
        output_file="review_docs.md",
    )
    resolved = resolve_backend(spec)
    assert resolved.id == "claude", (
        f"Path 2 must refuse deepseek (no 'tools' tag) for an agent "
        f"with output_file; got {resolved.id!r}"
    )


def test_resolve_backend_path2_allows_non_tool_backend_when_no_output_file(
    init_test_config, fake_agent_spec, monkeypatch,
):
    """Negative pin: an agent WITHOUT ``output_file`` (e.g. inline-only
    chat helper) is fine on a non-tool backend. The gate only fires for
    artifact-producing agents.
    """
    from larkhelm.crew._backend_resolver import resolve_backend
    from larkhelm.backend_registry import BackendRegistry, BackendSpec
    import larkhelm.backend_registry as _br_mod

    reg = BackendRegistry()
    reg._specs["claude"] = BackendSpec(
        id="claude", provider="anthropic_api",
        display_name="Claude", role="orchestrator",
        tags=["tools"], healthy=True, enabled=True,
    )
    reg._specs["deepseek"] = BackendSpec(
        id="deepseek", provider="deepseek_api",
        display_name="DeepSeek", role="worker",
        tags=["cheap"], healthy=True, enabled=True,
    )
    monkeypatch.setattr(_br_mod, "BACKEND_REGISTRY", reg)

    spec = fake_agent_spec(
        id="chat_only", role="闲聊",
        model="deepseek", task_profile="",
        output_file="",   # no artifact contract
    )
    resolved = resolve_backend(spec)
    assert resolved.id == "deepseek"


def test_resolve_backend_honours_exclude_backend_ids(
    init_test_config, fake_agent_spec, monkeypatch,
):
    """F4 driver contract: the resolver's new ``exclude_backend_ids``
    kwarg removes the named ids from Path 1's ranked list. The runner's
    retry-loop relies on this to switch backends after a validate
    failure.
    """
    from larkhelm.crew._backend_resolver import resolve_backend
    from larkhelm.backend_registry import BackendRegistry, BackendSpec
    import larkhelm.backend_registry as _br_mod

    reg = BackendRegistry()
    reg._specs["claude"] = BackendSpec(
        id="claude", provider="anthropic_api",
        display_name="Claude", role="orchestrator",
        tags=["tools"], healthy=True, enabled=True,
        capability_scores={"coding": 1.0, "tools": 1.0},
    )
    reg._specs["kimi"] = BackendSpec(
        id="kimi", provider="kimi_cli",
        display_name="Kimi", role="worker",
        tags=["tools"], healthy=True, enabled=True,
        capability_scores={"coding": 0.8, "tools": 0.9},
    )
    monkeypatch.setattr(_br_mod, "BACKEND_REGISTRY", reg)

    spec = fake_agent_spec(id="x", task_profile="engineer")
    # First call: top-ranked wins.
    first = resolve_backend(spec)
    # Exclude that backend, expect a different pick.
    second = resolve_backend(spec, exclude_backend_ids=frozenset({first.id}))
    assert second.id != first.id, (
        f"exclude_backend_ids did not filter; got the same backend twice "
        f"(first={first.id}, second={second.id})"
    )


# ── F5 — synth reads .invalid for FAILED agents ────────────────────────


def test_sanitize_quarantined_content_strips_sentinel_lines_and_quotes():
    """The sanitizer must drop any line containing a sentinel AND strip
    inline/fenced/blockquote citations from the survivors. A purely-
    sentinel file (DocsExpert 537 B case) returns "".
    """
    from larkhelm.crew._runner import _sanitize_quarantined_content

    # Mixed: real prose + sentinel line + inline-quote line.
    mixed = (
        "# Security review\n"
        "\n"
        "Body sentence one with enough length to clear the 50-char floor.\n"
        "<｜｜DSML｜｜tool_calls>\n"
        "Inline citation: `<｜｜DSML｜｜tool_calls>` was DeepSeek's leak.\n"
        "Final conclusion sentence with sufficient prose length here too.\n"
    )
    out = _sanitize_quarantined_content(mixed)
    assert out
    assert "DSML" not in out, f"sentinel survived sanitize: {out!r}"
    assert "Body sentence" in out
    assert "Final conclusion" in out

    # Pure-sentinel: < 50 chars after strip → return "".
    pure = "<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name=\"Read\">\n"
    assert _sanitize_quarantined_content(pure) == ""


def test_synthesize_includes_sanitized_invalid_excerpt_for_failed_agent(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """When ``_synthesize`` encounters a FAILED agent with a quarantined
    ``<output_file>.invalid`` file on disk, it must include a sanitized
    excerpt in the parts assembled for the synth backend (so the synth
    LLM can incorporate the false-positive-quarantined report's content
    rather than throw the work away).
    """
    from larkhelm.crew._runner import _synthesize
    from larkhelm.crew_types import AgentStatus, CrewPlan
    state = fake_crew_state(["sec"])
    state.agents["sec"].spec = fake_agent_spec(
        id="sec", role="安全专家",
        output_file="review_security.md",
    )
    state.agents["sec"].status = AgentStatus.FAILED
    state.agents["sec"].error  = "output artifact contract violation: sentinel"
    state.plan = CrewPlan(
        title="t", agents=[state.agents["sec"].spec],
        synthesis_prompt="Synth this.",
    )
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "review_security.md.invalid",
        "# Security review\n\n"
        "Body sentence — the substantive analysis goes here, clearly above\n"
        "the 50-char floor so the sanitizer keeps it.\n"
        "Inline cite: `<｜｜DSML｜｜tool_calls>` was DeepSeek's bad output.\n"
        "Final conclusion: routing fix needed — Path 2 require_tools gate.\n",
    )
    # Stub every possible synth-dispatch entry point so the test captures
    # the prompt regardless of which orchestrator provider the registry
    # resolves to (anthropic_api / gemini_cli / kimi_cli / claude CLI).
    captured = {}

    def _fake_api(*args, **kwargs):
        captured["prompt"] = kwargs.get("message", "")
        return "synthesized-output", {}   # API runners return (text, usage)

    def _fake_cli(*args, **kwargs):
        captured["prompt"] = kwargs.get("message", "")
        return "synthesized-output"

    import larkhelm.backend_api as _bapi
    import larkhelm.backend_cli as _bcli
    monkeypatch.setattr(_bapi, "run_anthropic", _fake_api, raising=False)
    monkeypatch.setattr(_bapi, "run_google", _fake_api, raising=False)
    monkeypatch.setattr(_bapi, "run_openai_compat", _fake_api, raising=False)
    monkeypatch.setattr(_bcli, "run_claude", _fake_cli, raising=False)
    monkeypatch.setattr(_bcli, "run_gemini", _fake_cli, raising=False)
    monkeypatch.setattr(_bcli, "run_kimi", _fake_cli, raising=False)
    monkeypatch.setattr(
        "larkhelm.perm.grant_yolo", lambda chat_id: None,
    )
    monkeypatch.setattr(
        "larkhelm.perm.revoke_yolo", lambda chat_id: None,
    )

    out = _synthesize(state)
    assert out == "synthesized-output"
    prompt = captured.get("prompt", "")
    assert "review_security.md.invalid" in prompt, (
        f"synth prompt didn't mention quarantine path; got {prompt!r}"
    )
    assert "Final conclusion" in prompt, (
        f"synth prompt didn't include sanitized excerpt; got {prompt!r}"
    )
    # Critical: the sentinel itself must NEVER reach the synth backend.
    assert "DSML" not in prompt


# ── F6 — emit_agent_failure red banner + throttle ─────────────────────


def test_emit_agent_failure_sends_red_banner_for_validate_stage(
    init_test_config, fake_crew_state, fake_agent_spec, fake_card_sender,
):
    """Validate-stage failures must surface a standalone red banner so
    the user sees the failure even when the in-place crew card is stale
    / collapsed. Pre-F6 the 2026-05-25 incident emitted ZERO failure
    cards for two failed agents.
    """
    from larkhelm.crew._failure_card import (
        emit_agent_failure, _reset_banner_throttle_for_tests,
    )
    _reset_banner_throttle_for_tests()
    state = fake_crew_state(["docs"])
    state.agents["docs"].spec = fake_agent_spec(
        id="docs", role="文档专家", output_file="review_docs.md",
    )
    fake_card_sender.clear()
    emit_agent_failure(
        state, "docs", stage="validate",
        exc=RuntimeError("output artifact contract violation: sentinel"),
    )
    red_cards = [
        c for c in fake_card_sender
        if c.get("kind") == "send_card" and c.get("color") == "red"
    ]
    assert red_cards, f"no red banner emitted; recorded={fake_card_sender}"
    banner = red_cards[0]
    assert "docs" in banner["title"]
    assert "validate" in banner["title"]
    # Body must surface the .invalid path so the user can manually
    # recover content if the synth's sanitized excerpt is insufficient.
    assert "review_docs.md.invalid" in banner["body"]


def test_emit_agent_failure_throttles_repeated_banners(
    init_test_config, fake_crew_state, fake_agent_spec, fake_card_sender,
):
    """A retry storm or duplicate emit must NOT spam the chat. The
    (crew_id, agent_id) throttle ensures one banner per agent per crew.
    """
    from larkhelm.crew._failure_card import (
        emit_agent_failure, _reset_banner_throttle_for_tests,
    )
    _reset_banner_throttle_for_tests()
    state = fake_crew_state(["sec"])
    state.agents["sec"].spec = fake_agent_spec(
        id="sec", role="安全专家", output_file="review_security.md",
    )
    fake_card_sender.clear()
    for _ in range(3):
        emit_agent_failure(
            state, "sec", stage="validate",
            exc=RuntimeError("validate failed again"),
        )
    red_cards = [
        c for c in fake_card_sender
        if c.get("kind") == "send_card" and c.get("color") == "red"
    ]
    assert len(red_cards) == 1, (
        f"throttle failed — got {len(red_cards)} red banners, expected 1"
    )


def test_emit_agent_failure_skips_banner_for_run_stage(
    init_test_config, fake_crew_state, fake_agent_spec, fake_card_sender,
):
    """Generic ``run`` stage failures keep the legacy in-place-card-only
    behaviour — they're typically transient and the existing retry path
    handles them. Reserve the red banner for terminal/actionable
    failures (validate / backend_select / oom / timeout).
    """
    from larkhelm.crew._failure_card import (
        emit_agent_failure, _reset_banner_throttle_for_tests,
    )
    _reset_banner_throttle_for_tests()
    state = fake_crew_state(["pm"])
    fake_card_sender.clear()
    emit_agent_failure(
        state, "pm", stage="run",
        exc=RuntimeError("transient subprocess error"),
    )
    red_cards = [
        c for c in fake_card_sender
        if c.get("kind") == "send_card" and c.get("color") == "red"
    ]
    assert not red_cards, (
        f"run-stage shouldn't emit red banner; got {red_cards}"
    )


# ── Back-compat alias for _strip_fenced_code_blocks ───────────────────


def test_strip_fenced_alias_still_works():
    """External tests / callers using the old function name must keep
    seeing the same behaviour (now via _strip_code_evidence).
    """
    from larkhelm.crew._runner import _strip_fenced_code_blocks
    out = _strip_fenced_code_blocks("a\n```\ndsml\n```\nb")
    assert "dsml" not in out


# ── LOW2 — safety-net pre-scan sentinels before persisting ────────────


def test_safety_net_skips_persist_when_result_contains_bare_sentinel(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """LOW2 (review_followup): if the in-memory result contains a tool-call
    sentinel after code-evidence strip, ``_persist_result_to_output_file_if_missing``
    must NOT write it to disk. Otherwise the file lands → validator catches
    it → ``_quarantine_invalid_output`` runs, but a concurrent reader may
    have already seen the poisoned file. Skipping the write keeps disk clean.
    """
    from larkhelm.crew._runner import _persist_result_to_output_file_if_missing
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)

    bare_leak = (
        "Here is the PRD.\n\n"
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"Write\">\n"
        + ("padding line.\n" * 30)  # > 200 chars threshold
    )
    _persist_result_to_output_file_if_missing(state, "pm", bare_leak)

    out_path = tmp_path / ".crew_workspace" / "prd.md"
    assert not out_path.exists(), (
        f"safety-net must skip persist when sentinel present; "
        f"file landed: {out_path.read_text() if out_path.exists() else '<n/a>'}"
    )


def test_safety_net_still_persists_when_sentinels_only_in_fenced_quote(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Negative pin for LOW2: a result that quotes sentinels inside a
    fenced block (legitimate post-mortem write-up) must still persist —
    the pre-scan strips fenced/inline/blockquote citations first, just
    like ``_validate_output_artifact``.
    """
    from larkhelm.crew._runner import _persist_result_to_output_file_if_missing
    state = fake_crew_state(["reviewer"])
    state.agents["reviewer"].spec = fake_agent_spec(
        id="reviewer", output_file="review.md",
    )
    _patch_cwd(monkeypatch, tmp_path)

    quoted = (
        "# Post-mortem\n\n"
        "Backend leaked tokens like:\n\n"
        "```\n<｜｜DSML｜｜tool_calls>\n```\n\n"
        "Fix: route output_file agents to tool-capable backends only.\n"
        + ("more analysis.\n" * 20)  # > 200 chars threshold
    )
    _persist_result_to_output_file_if_missing(state, "reviewer", quoted)

    out_path = tmp_path / ".crew_workspace" / "review.md"
    assert out_path.exists(), "fenced-quote citation must not block persist"
    assert "tool_calls" in out_path.read_text(), "original quote preserved"


def test_safety_net_persists_evidence_when_skipping_for_sentinel(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """SEC-v2-HIGH-1 (review_security_v2): when LOW2 pre-scan trips and
    the safety-net skips the real persist, the raw in-memory result must
    still land on disk under ``<output>.invalid.prerun`` for forensic
    inspection. Without it, the only signal is a single ``warn(...)`` line
    that ages out of the debug log after 30 days, and operators can't
    reconstruct what the backend actually emitted.

    Pins:
      * ``<output>`` itself is NOT written (LOW2 contract holds)
      * ``<output>.invalid.prerun`` contains the full raw result verbatim
      * ``.invalid`` (no suffix — owned by ``_quarantine_invalid_output``
        for already-written files) is NOT created — the two suffixes have
        distinct lifecycles
    """
    from larkhelm.crew._runner import _persist_result_to_output_file_if_missing
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)

    bare_leak = (
        "Here is the PRD.\n\n"
        "<｜｜DSML｜｜tool_calls>\n"
        "<｜｜DSML｜｜invoke name=\"Write\">\n"
        + ("padding line.\n" * 30)  # > 200 chars threshold
    )
    _persist_result_to_output_file_if_missing(state, "pm", bare_leak)

    workspace = tmp_path / ".crew_workspace"
    out_path = workspace / "prd.md"
    prerun_path = workspace / "prd.md.invalid.prerun"
    quarantine_path = workspace / "prd.md.invalid"

    assert not out_path.exists(), "real output must not land when sentinel hit"
    assert prerun_path.exists(), (
        f"SEC-v2-HIGH-1: evidence sidecar missing — incident response loses "
        f"the raw payload. workspace contents: "
        f"{sorted(p.name for p in workspace.iterdir())}"
    )
    saved = prerun_path.read_text(encoding="utf-8")
    assert saved == bare_leak, (
        "evidence sidecar must preserve the raw in-memory result verbatim "
        "(no scrubbing); reviewers need the original token stream to debug"
    )
    assert not quarantine_path.exists(), (
        "``.invalid`` (no suffix) is owned by _quarantine_invalid_output for "
        "already-written files; pre-write evidence belongs under .invalid.prerun"
    )


def test_safety_net_evidence_write_failure_does_not_raise(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """SEC-v2-HIGH-1 negative pin: even if the evidence write itself
    fails (e.g. read-only disk), the safety-net must not raise — it is a
    best-effort forensic artefact, not a correctness gate. LOW2's main
    contract (real output suppressed) must still hold.
    """
    from larkhelm.crew._runner import _persist_result_to_output_file_if_missing
    state = fake_crew_state(["pm"])
    state.agents["pm"].spec = fake_agent_spec(id="pm", output_file="prd.md")
    _patch_cwd(monkeypatch, tmp_path)

    # Force evidence write to explode: monkeypatch Path.write_text to raise
    # only when targeting the .prerun sidecar so the existing safety-net
    # write logic stays unaffected for other tests.
    import pathlib
    real_write_text = pathlib.Path.write_text

    def _exploding_write_text(self, *args, **kwargs):
        if str(self).endswith(".invalid.prerun"):
            raise OSError("simulated read-only disk")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", _exploding_write_text)

    bare_leak = (
        "Here is the PRD.\n\n"
        "<｜｜DSML｜｜tool_calls>\n"
        + ("padding line.\n" * 30)
    )
    # Must not raise:
    _persist_result_to_output_file_if_missing(state, "pm", bare_leak)

    # Real output still suppressed (LOW2 contract still holds):
    assert not (tmp_path / ".crew_workspace" / "prd.md").exists()


# ── LOW3 / SEC-v2-MED-1 — Anthropic XML structural sentinel ──────────


def test_validate_catches_anthropic_function_calls_structural_leak(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """LOW3 origin + SEC-v2-MED-1 refinement: a real Anthropic-shaped
    tool-call leak (opening + invoke + closing within window) must
    still trip. The fix narrowed the match to the FULL structural
    shape — this test confirms the shape itself is still detected.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "design.md",
        "# Architecture\n\n"
        "Proceeding with the plan.\n\n"
        "<function_calls>\n<invoke name=\"Read\">\n</invoke>\n</function_calls>\n",
    )
    issue = _validate_output_artifact(state, "arch", result="ack")
    assert issue
    # SEC-v2-MED-1 emits the dedicated "structural Anthropic tool-call
    # XML" label; the legacy "sentinel" label still passes for backward
    # compat callers, but we expect the structural one here.
    assert "Anthropic" in issue or "function_calls" in issue


def test_validate_anthropic_sentinel_in_fenced_quote_does_not_trip(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Negative pin: legitimate docs/reviews quoting the Anthropic tag
    inside a fenced block must still pass (the scrubber removes it
    before either substring or structural scan).
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["docs"])
    state.agents["docs"].spec = fake_agent_spec(id="docs", output_file="docs.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "docs.md",
        "# Docstring\n\n"
        "Claude tool-use tokens look like:\n\n"
        "```\n<function_calls>\n<invoke name=\"X\">\n</invoke>\n</function_calls>\n```\n\n"
        "These are stripped before the sentinel scan.\n",
    )
    assert _validate_output_artifact(state, "docs", result="ack") == ""


# ── SEC-v2-MED-1: structural-only loose tier ──────────────────────────


def test_validate_anthropic_bare_function_calls_mention_no_longer_trips(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """SEC-v2-MED-1: a narrative-prose mention of the tag NAME alone
    (no invoke + close trio) must NOT trigger quarantine. Pre-fix the
    LOW3 bare-substring scan rejected this; post-fix only the full
    structural shape rejects.

    Common legitimate cases: README discussing tool-use protocol,
    CLAUDE.md project notes, agent prompt examples.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["docs"])
    state.agents["docs"].spec = fake_agent_spec(id="docs", output_file="readme.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "readme.md",
        "# Architecture notes\n\n"
        "Claude streams tool calls via <function_calls> elements wrapping\n"
        "individual invocations. The runner translates these into tool_use\n"
        "events before they reach the agent's output_file.\n",
    )
    assert _validate_output_artifact(state, "docs", result="ack") == ""


def test_validate_anthropic_invoke_name_alone_no_longer_trips(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """SEC-v2-MED-1: same as above but with only the inner element
    mention. Pre-fix the LOW3 bare-substring scan caught this too.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["docs"])
    state.agents["docs"].spec = fake_agent_spec(id="docs", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "design.md",
        "# Tool dispatch design\n\n"
        "Each invocation block uses <invoke name=\"...\"> to address the\n"
        "specific tool. The dispatcher routes by the name attribute.\n",
    )
    assert _validate_output_artifact(state, "docs", result="ack") == ""


def test_validate_anthropic_loose_disabled_falls_back_to_metric_only(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """SEC-v2-MED-1 gate flip: with
    ``crew_sentinel_anthropic_loose_enabled=false`` the structural
    regex still runs (for the metric) but the artifact passes — the
    metric records ``hit_observed`` rather than ``hit_enforced``.
    """
    import larkhelm.config as _cfg
    from larkhelm.crew._runner import _validate_output_artifact

    # Snapshot + override the gate for this test scope only.
    original_cfg = getattr(_cfg, "config", {})
    monkeypatch.setattr(
        _cfg, "config",
        {**original_cfg, "crew_sentinel_anthropic_loose_enabled": False},
    )

    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "design.md",
        "# Architecture\n\n"
        "<function_calls>\n<invoke name=\"Read\">\n</invoke>\n</function_calls>\n",
    )
    assert _validate_output_artifact(state, "arch", result="ack") == "", (
        "Gate-off should let the structural leak through (metric-only)"
    )


def test_validate_anthropic_far_apart_tags_do_not_trip(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """SEC-v2-MED-1: the structural regex caps the gap between
    ``<function_calls>`` opening, ``<invoke name=`` inner, and
    ``</function_calls>`` closing at 4 KiB. Documents that mention each
    tag in different sections (far apart) must NOT match — the
    structural pattern only fires for an actual emitted tool-call
    block, not for unrelated discussion that happens to use both
    tag names.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    state = fake_crew_state(["docs"])
    state.agents["docs"].spec = fake_agent_spec(id="docs", output_file="docs.md")
    _patch_cwd(monkeypatch, tmp_path)
    # 8 KiB of unrelated prose between the three tokens to exceed the
    # 4 KiB inner window cap. The cap allows the regex to skip up to
    # 4 KiB between EACH pair, so we space them 5 KiB apart to be sure
    # neither pair fits.
    pad = ("legit prose discussing crew design.\n" * 200)   # ~ 7 KiB
    _write_workspace_file(
        tmp_path, "docs.md",
        f"# Doc\n\nIntro mentions <function_calls> as a concept.\n\n"
        f"{pad}\n"
        f"Later section mentions <invoke name=\"X\"> separately.\n\n"
        f"{pad}\n"
        f"Closing section mentions </function_calls> at the very end.\n",
    )
    assert _validate_output_artifact(state, "docs", result="ack") == ""


# ── SEC-CRIT-4 — layer-2 sentinel heuristic ──────────────────────────
#
# Layer-1 (`_strip_code_evidence` + `s in prose`) is bypassable by
# wrapping tool-call tokens in fenced / inline-backtick / blockquote
# citations. Layer-2 counts RAW sentinel occurrences (pre-scrub) and
# rejects on either:
#   (a) raw_hits >= raw_threshold (3) AND drop_ratio > 30%
#   (b) raw_hits >= paranoid_threshold (5)  [regardless of drop_ratio]
#
# Gated by `crew_sentinel_layer2_enabled` + traffic dial. When the gate
# is off (default), the metric still fires with mode="observe" so
# operators can pre-calibrate thresholds against real crew runs.


def _set_layer2_config(
    *, enabled: bool, traffic: float = 1.0,
    raw_threshold: int = 3, drop_ratio: float = 0.30,
    paranoid_threshold: int = 5,
) -> None:
    """Set layer-2 flags on the in-memory ``larkhelm.config.config`` dict.
    ``init_test_config`` re-runs ``_init_runtime`` per test, so these
    mutations don't bleed into other tests.
    """
    import larkhelm.config as _cfg
    _cfg.config["crew_sentinel_layer2_enabled"] = enabled
    _cfg.config["crew_sentinel_layer2_traffic"] = float(traffic)
    _cfg.config["crew_sentinel_layer2_raw_threshold"] = int(raw_threshold)
    _cfg.config["crew_sentinel_layer2_drop_ratio"] = float(drop_ratio)
    _cfg.config["crew_sentinel_layer2_paranoid_threshold"] = int(paranoid_threshold)


def _wrapper_bypass_payload(n_sentinels: int, *, prose_padding: int = 0) -> str:
    """Build content where N sentinels each sit inside inline backticks
    (layer-1 scrubs them → prose hits == 0). ``prose_padding`` adds N
    extra characters of unrelated prose to dilute drop_ratio.
    """
    sentinel = "<｜｜DSML｜｜tool_call"  # one of the 7 in _OUTPUT_SENTINELS
    sentinel_lines = "\n".join(f"see token `{sentinel}` here" for _ in range(n_sentinels))
    pad = ("more prose. " * (max(prose_padding, 0) // 12 + 1))[:prose_padding]
    return f"# Review\n\n{sentinel_lines}\n\n{pad}\n"


def test_layer2_observe_only_when_disabled(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Default config (enabled=False) → validator must NOT reject even when
    layer-2 thresholds are tripped. Metric fires with mode="observe" so
    operators can see the would-be reject rate before flipping enforcement.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=False)
    state = fake_crew_state(["rv"])
    state.agents["rv"].spec = fake_agent_spec(id="rv", output_file="review.md")
    _patch_cwd(monkeypatch, tmp_path)
    # 6 sentinels (>= paranoid_threshold 5) all in inline-backticks →
    # layer-1 misses, layer-2 in observe mode must NOT reject.
    _write_workspace_file(tmp_path, "review.md", _wrapper_bypass_payload(6))

    assert _validate_output_artifact(state, "rv", result="ack") == ""


def test_layer2_enforces_paranoid_threshold(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Config on (enabled + traffic=1.0) + 5+ wrapped sentinels → reject
    with paranoid label, regardless of drop_ratio.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["rv"])
    state.agents["rv"].spec = fake_agent_spec(id="rv", output_file="review.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(tmp_path, "review.md", _wrapper_bypass_payload(6))

    msg = _validate_output_artifact(state, "rv", result="ack")
    assert msg
    assert "paranoid" in msg
    assert "6 sentinels" in msg


def test_layer2_enforces_drop_ratio_heuristic(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """3 wrapped sentinels (>= raw_threshold) + drop_ratio > 30% → reject
    with drop_ratio label.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["rv"])
    state.agents["rv"].spec = fake_agent_spec(id="rv", output_file="review.md")
    _patch_cwd(monkeypatch, tmp_path)
    # 3 sentinels in inline backticks, minimal padding → high drop_ratio
    # because backtick-wrapping strips ~25 chars per occurrence (the
    # token itself is ~22 chars + 2 backticks).
    _write_workspace_file(tmp_path, "review.md", _wrapper_bypass_payload(3))

    msg = _validate_output_artifact(state, "rv", result="ack")
    assert msg
    assert "drop_ratio" in msg
    assert "3 sentinels" in msg


def test_layer2_abstains_below_raw_threshold(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Exactly 2 wrapped sentinels (< raw_threshold 3) + drop_ratio > 30%
    → abstain. Boundary pin: raw_threshold is exclusive on the low side,
    inclusive on the trigger side (>=3).
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["rv"])
    state.agents["rv"].spec = fake_agent_spec(id="rv", output_file="review.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(tmp_path, "review.md", _wrapper_bypass_payload(2))

    assert _validate_output_artifact(state, "rv", result="ack") == ""


def test_layer2_abstains_when_drop_ratio_low(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """4 wrapped sentinels (>= raw_threshold) BUT diluted with enough
    plain prose that drop_ratio ≤ 30% → abstain on the drop_ratio rule.
    Paranoid threshold (5) also not yet tripped. Boundary pin for the
    AND-condition in the drop_ratio branch.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["rv"])
    state.agents["rv"].spec = fake_agent_spec(id="rv", output_file="review.md")
    _patch_cwd(monkeypatch, tmp_path)
    # 4 sentinels (< paranoid 5), heavily padded prose so backtick-strip
    # removes < 30% of total content. Each sentinel inline occupies ~25
    # chars; with 4 of them ~100 chars get scrubbed, so total must be
    # > ~333 chars for drop_ratio ≤ 30%.
    _write_workspace_file(
        tmp_path, "review.md", _wrapper_bypass_payload(4, prose_padding=2000),
    )

    assert _validate_output_artifact(state, "rv", result="ack") == ""


def test_layer2_legitimate_review_with_few_citations_passes(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Realistic positive: a real reviewer report that cites the DSML
    token 1-2 times inside fenced blocks alongside lots of legitimate
    prose. Must pass under enforced mode (raw_hits < raw_threshold).
    Pins that the heuristic doesn't fire on the original 2026-05-25
    27 KB SecurityExpert false-positive shape.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["sec"])
    state.agents["sec"].spec = fake_agent_spec(id="sec", output_file="review_security.md")
    _patch_cwd(monkeypatch, tmp_path)
    body = (
        "# Security review v2\n\n"
        "## Finding 1\n"
        "DocsExpert emitted `<｜｜DSML｜｜tool_call` once as raw text — "
        "see `review_docs.md.invalid`. Quarantine path worked.\n\n"
        + ("This is legitimate analysis prose. " * 200)  # ~7 KB padding
    )
    _write_workspace_file(tmp_path, "review_security.md", body)

    assert _validate_output_artifact(state, "sec", result="ack") == ""


# ── SEC-v2-MED-1 metric pinning ───────────────────────────────────────
#
# The behavioural tests above pin "file rejected" / "file passes" but
# don't actually verify the ``larkhelm_crew_validate_anthropic_loose_total``
# counter increments correctly. These three cases lock the counter
# contract end-to-end so a future refactor can't silently drop the
# metric or mis-label it. Mirrors the metric-pin style used by
# ``tests/test_cascade_backoff_metric.py`` for the cascade counter.


def _read_anthropic_loose_counter(outcome: str) -> "float | None":
    """Pull the live value of the labelled counter or skip when prom-client
    is absent (CI-without-extras). Returns float on hit, None on skip.
    """
    pytest = __import__("pytest")
    pytest.importorskip("prometheus_client")
    from larkhelm import metrics as _met
    reg = _met.get_registry()
    counter = getattr(reg, "crew_validate_anthropic_loose_total", None)
    if not reg.available or counter is None:
        pytest.skip("prometheus-client not installed in this venv")
    try:
        return counter.labels(outcome=outcome)._value.get()
    except Exception:
        pytest.skip("prometheus_client internal shape changed")


def test_anthropic_loose_case_a_full_shape_enforced_bumps_metric(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Case A (enforced): full ``<function_calls>`` + ``<invoke name=`` +
    ``</function_calls>`` structural shape with the gate ON →
    ``_validate_output_artifact`` rejects the artifact AND
    ``larkhelm_crew_validate_anthropic_loose_total{outcome="hit_enforced"}``
    bumps exactly once. ``hit_observed`` must NOT bump (gate is ON).
    """
    from larkhelm.crew._runner import _validate_output_artifact

    before_enforced = _read_anthropic_loose_counter("hit_enforced")
    before_observed = _read_anthropic_loose_counter("hit_observed")

    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "design.md",
        "# Architecture\n\n"
        "Proceeding with the plan.\n\n"
        "<function_calls>\n<invoke name=\"Read\">\n</invoke>\n</function_calls>\n",
    )
    issue = _validate_output_artifact(state, "arch", result="ack")
    assert issue, "case A should reject the artifact under default gate"
    assert "Anthropic" in issue or "function_calls" in issue

    after_enforced = _read_anthropic_loose_counter("hit_enforced")
    after_observed = _read_anthropic_loose_counter("hit_observed")
    assert after_enforced == before_enforced + 1, (
        f"hit_enforced should bump by 1; "
        f"got {before_enforced!r} → {after_enforced!r}"
    )
    assert after_observed == before_observed, (
        f"hit_observed must not bump while gate is ON; "
        f"got {before_observed!r} → {after_observed!r}"
    )


def test_anthropic_loose_case_b_bare_tag_mention_no_metric_bump(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Case B (no match): a narrative-prose mention of the tag NAME
    alone (no invoke + close trio) must NOT match the structural regex,
    so neither ``hit_enforced`` nor ``hit_observed`` bumps. Mirrors a
    README / CLAUDE.md / agent-doc citation of the protocol.
    """
    from larkhelm.crew._runner import _validate_output_artifact

    before_enforced = _read_anthropic_loose_counter("hit_enforced")
    before_observed = _read_anthropic_loose_counter("hit_observed")

    state = fake_crew_state(["docs"])
    state.agents["docs"].spec = fake_agent_spec(id="docs", output_file="readme.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "readme.md",
        "# Architecture notes\n\n"
        "Claude streams tool calls via <function_calls> elements wrapping\n"
        "individual invocations. The runner translates these into tool_use\n"
        "events before they reach the agent's output_file.\n",
    )
    assert _validate_output_artifact(state, "docs", result="ack") == ""

    after_enforced = _read_anthropic_loose_counter("hit_enforced")
    after_observed = _read_anthropic_loose_counter("hit_observed")
    assert after_enforced == before_enforced, (
        f"hit_enforced must not bump on bare-tag mention; "
        f"got {before_enforced!r} → {after_enforced!r}"
    )
    assert after_observed == before_observed, (
        f"hit_observed must not bump on bare-tag mention; "
        f"got {before_observed!r} → {after_observed!r}"
    )


def test_anthropic_loose_case_c_gate_off_observes_without_quarantine(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Case C (observe-only): full structural shape with the gate OFF →
    ``_validate_output_artifact`` returns "" (file passes) AND
    ``larkhelm_crew_validate_anthropic_loose_total{outcome="hit_observed"}``
    bumps exactly once. ``hit_enforced`` must NOT bump.
    """
    import larkhelm.config as _cfg
    from larkhelm.crew._runner import _validate_output_artifact

    before_enforced = _read_anthropic_loose_counter("hit_enforced")
    before_observed = _read_anthropic_loose_counter("hit_observed")

    # Snapshot + override the gate for this test scope only — pytest's
    # monkeypatch reverts on teardown so other tests still see the
    # default-true config.
    original_cfg = getattr(_cfg, "config", {})
    monkeypatch.setattr(
        _cfg, "config",
        {**original_cfg, "crew_sentinel_anthropic_loose_enabled": False},
    )

    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)
    _write_workspace_file(
        tmp_path, "design.md",
        "# Architecture\n\n"
        "<function_calls>\n<invoke name=\"Read\">\n</invoke>\n</function_calls>\n",
    )
    assert _validate_output_artifact(state, "arch", result="ack") == "", (
        "Gate-off should let the structural leak through (observe-only)"
    )

    after_enforced = _read_anthropic_loose_counter("hit_enforced")
    after_observed = _read_anthropic_loose_counter("hit_observed")
    assert after_observed == before_observed + 1, (
        f"hit_observed should bump by 1 with gate OFF; "
        f"got {before_observed!r} → {after_observed!r}"
    )
    assert after_enforced == before_enforced, (
        f"hit_enforced must not bump while gate is OFF; "
        f"got {before_enforced!r} → {after_enforced!r}"
    )
