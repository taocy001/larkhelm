"""Adversarial fixture for the sentinel-guard chain (2026-05-25).

Context: ``.crew_workspace/review_security_v2.md`` §107 #7 and §134 ⚠️#1
both flagged that the existing F1-F6 hardening tests
(``tests/test_crew_failure_hardening.py``) are all happy-path —

  * unwrapped sentinel → trips layer-1
  * fenced/inline/blockquote citation of a SINGLE sentinel → does not
    trip layer-1
  * synthetic ``_wrapper_bypass_payload`` factory exercises the layer-2
    counter contract

The missing case is the realistic *attack* shape: a backend (or a
prompt-injected one) emits prose that **smuggles multiple sentinels
past layer-1 by wrapping each one in citation syntax**, then layer-2
is the only remaining gate.

This file pins that adversarial chain end-to-end, so future refactors
that loosen layer-2 thresholds, change the citation scrubber, or alter
the safety-net pre-scan all break a regression test instead of silently
re-opening the bypass.

Each test runs against the real ``_persist_result_to_output_file_if_missing``
and ``_validate_output_artifact`` helpers — no in-test stubbing of the
defense path. Mirrors the harness used by
``tests/test_crew_failure_hardening.py`` (same ``init_test_config`` /
``fake_crew_state`` / ``fake_agent_spec`` fixtures, same ``_patch_cwd``
pattern).
"""
from __future__ import annotations

from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────


def _patch_cwd(monkeypatch, tmp_path: Path) -> None:
    """Redirect ``chat_state._get_cwd`` so workspace writes land in
    ``tmp_path/.crew_workspace/`` for the duration of one test.
    """
    from larkhelm import chat_state
    monkeypatch.setattr(chat_state, "_get_cwd", lambda chat_id: str(tmp_path))


def _write_workspace_file(tmp_path: Path, name: str, content: str) -> Path:
    ws = tmp_path / ".crew_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    p = ws / name
    p.write_text(content, encoding="utf-8")
    return p


def _set_layer2_config(
    *, enabled: bool, traffic: float = 1.0,
    raw_threshold: int = 3, drop_ratio: float = 0.30,
    paranoid_threshold: int = 5,
) -> None:
    """Set layer-2 flags on the in-memory ``larkhelm.config.config`` dict.

    ``init_test_config`` re-runs ``_init_runtime`` per test, so these
    mutations cannot bleed across tests. Same shape as the helper
    inside ``test_crew_failure_hardening.py``.
    """
    import larkhelm.config as _cfg
    _cfg.config["crew_sentinel_layer2_enabled"] = enabled
    _cfg.config["crew_sentinel_layer2_traffic"] = float(traffic)
    _cfg.config["crew_sentinel_layer2_raw_threshold"] = int(raw_threshold)
    _cfg.config["crew_sentinel_layer2_drop_ratio"] = float(drop_ratio)
    _cfg.config["crew_sentinel_layer2_paranoid_threshold"] = int(paranoid_threshold)


# One of the strict ``_OUTPUT_SENTINELS`` entries — picked DSML because
# the full-width pipes guarantee it cannot accidentally appear in any
# CI / source-code identifier and the threat model documents this as
# the original 2026-05-22 leak token.
_DSML_TOKEN = "<｜｜DSML｜｜tool_call"


def _adversarial_inline_payload(
    *, n_sentinels: int, prose_padding_chars: int = 0, dense: bool = False,
) -> str:
    """Build a result body that wraps each of ``n_sentinels`` strict
    sentinel tokens inside inline-backtick spans, interleaved with
    `prose_padding_chars` of unrelated narrative prose.

    The shape simulates: "an attacker that has injected the backend
    has it emit a plausible-looking analysis paragraph that *cites*
    the sentinel multiple times to leak it past layer-1's citation
    scrubber". The scrubber removes the ``\\` \\`` wrapping, leaving
    pure prose; layer-1's substring scan then misses all occurrences.

    ``dense=True`` strips the narrative wrapper around each citation
    so the line is essentially ``\\`<token>\\`\\n``. Used to exercise
    layer-2's drop_ratio branch where the scrubber removes a large
    fraction of the content — the "low raw count but high citation
    density" attack shape.
    """
    if dense:
        # Each line is just the wrapped sentinel — no surrounding
        # prose at all. Scrubber strips ~23 code points per line out
        # of ~25, so drop_ratio ≈ 90% with negligible content_len.
        body = "\n".join(f"`{_DSML_TOKEN}`" for _ in range(n_sentinels))
    else:
        body = "\n".join(
            f"Finding {i + 1}: the upstream agent emitted `{_DSML_TOKEN}` as raw text."
            for i in range(n_sentinels)
        )
    if prose_padding_chars > 0:
        pad = ("Legitimate analysis prose continues here. " * 100)[:prose_padding_chars]
        body = f"{body}\n\n{pad}\n"
    return f"# Adversarial review\n\n{body}\n"


def _adversarial_mixed_payload(*, n_each: int = 2) -> str:
    """Build a body that mixes all three citation forms (fenced, inline
    backtick, blockquote) — ``n_each`` sentinels per form, total
    ``3 * n_each`` raw occurrences. Each form gets scrubbed by a
    different pass inside ``_strip_code_evidence``, so this fixture
    catches a regression in any of the three passes (a future refactor
    that drops, say, the blockquote pass would not be caught by a
    single-form payload).
    """
    fenced_block = "\n".join([f"{_DSML_TOKEN}_fenced_{i}>" for i in range(n_each)])
    inline_lines = "\n".join([f"see `{_DSML_TOKEN}_inline_{i}` here." for i in range(n_each)])
    blockquote_lines = "\n".join([f"> {_DSML_TOKEN}_quoted_{i}" for i in range(n_each)])
    return (
        "# Mixed-citation adversarial review\n\n"
        "## Fenced citations\n\n"
        f"```\n{fenced_block}\n```\n\n"
        "## Inline citations\n\n"
        f"{inline_lines}\n\n"
        "## Blockquote citations\n\n"
        f"{blockquote_lines}\n\n"
        "Conclusion: backend repeatedly emits the protocol token.\n"
    )


# ── 1. Pre-scan miss attests the bypass exists when wrapped ───────────


def test_pre_scan_safety_net_misses_inline_wrapped_sentinels(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """SEC-v2-HIGH-1 / SEC-CRIT-4 attestation: the LOW2 pre-scan in
    ``_persist_result_to_output_file_if_missing`` runs the citation
    scrubber BEFORE the substring scan. An adversarial result that
    wraps 4 sentinels in inline backticks therefore writes cleanly
    to disk — the pre-scan is structurally unable to stop the
    wrapper-bypass class of attacks on its own.

    This test deliberately runs with layer-2 disabled (the production
    default) so the only defenses in play are the pre-scan and the
    layer-1 validator. The expected outcome — file lands on disk
    without ``.invalid.prerun`` evidence — pins the documented gap
    so anyone who tries to "fix" it without coordinating with the
    SEC-CRIT-4 layer-2 dial breaks this test and triggers review.
    """
    from larkhelm.crew._runner import _persist_result_to_output_file_if_missing
    _set_layer2_config(enabled=False)
    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)

    payload = _adversarial_inline_payload(n_sentinels=4)
    _persist_result_to_output_file_if_missing(state, "arch", payload)

    workspace = tmp_path / ".crew_workspace"
    out_path = workspace / "design.md"
    prerun_path = workspace / "design.md.invalid.prerun"

    assert out_path.exists(), (
        "Pre-scan must MISS wrapped sentinels (citation scrubber strips "
        "the backticks first) — the file should land. If this assertion "
        "starts failing because the file did not land, the pre-scan has "
        "become stricter; coordinate with SEC-CRIT-4 layer-2 thresholds "
        "before tightening."
    )
    assert not prerun_path.exists(), (
        "No evidence sidecar should be written when the pre-scan abstains; "
        "the .invalid.prerun path is reserved for SEC-v2-HIGH-1's "
        "skip-with-evidence flow which only triggers on bare-sentinel hits."
    )
    # Verify the file actually carries the raw adversarial content
    # (the entire chain was bypassed):
    assert _DSML_TOKEN in out_path.read_text(encoding="utf-8"), (
        "The adversarial payload must reach disk intact — that is the "
        "exact failure mode SEC-CRIT-4 layer-2 was introduced to close."
    )


# ── 2. Layer-1 validate alone also misses the same payload ────────────


def test_layer1_validate_alone_misses_inline_wrapped_sentinels(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Same attestation as above but for ``_validate_output_artifact``
    with layer-2 disabled. Pins that layer-1 alone (substring scan on
    scrubbed prose) cannot catch the wrapper-bypass class of attacks.

    Why this exists separately from the safety-net test: the two
    helpers share the scrubber, but a future refactor could change
    one without the other (e.g. someone adds a paranoia mode only to
    the safety-net). Pinning both ensures the documented bypass shape
    is reproducible on whichever path the adversary lands on.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=False)
    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)

    payload = _adversarial_inline_payload(n_sentinels=4)
    _write_workspace_file(tmp_path, "design.md", payload)

    assert _validate_output_artifact(state, "arch", result="ack") == "", (
        "Layer-1 must not catch the wrapper-bypass payload — layer-2 is "
        "the intended defense. If this starts rejecting, someone tightened "
        "layer-1's scrubber/substring contract; that would also reject "
        "the 27 KB SecurityExpert false-positive that motivated F1+F2."
    )


# ── 3. Layer-2 paranoid closes the bypass ─────────────────────────────


def test_layer2_paranoid_closes_inline_wrapper_bypass(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Same adversarial payload as the two tests above, layer-2 enforced.
    With 6 wrapped sentinels (≥ paranoid_threshold 5), the validator
    must reject the artifact regardless of citation form — paranoid is
    a "raw count only, no drop_ratio gate" rule, so wrapping cannot
    defeat it.

    This is the realistic SEC-CRIT-4 defense: layer-1 stays liberal
    (so legitimate reviewers can quote evidence), and layer-2 catches
    the *quantity* anomaly that distinguishes prose-with-citations from
    leaked-protocol-tokens-being-cited-as-cover.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)

    payload = _adversarial_inline_payload(n_sentinels=6)
    _write_workspace_file(tmp_path, "design.md", payload)

    issue = _validate_output_artifact(state, "arch", result="ack")
    assert issue, "Layer-2 paranoid must close the wrapper-bypass with ≥5 hits"
    assert "paranoid" in issue, (
        f"Expected paranoid-branch label (≥5 raw hits is the paranoid rule); "
        f"got: {issue!r}"
    )
    assert "6 sentinels" in issue, (
        f"Failure label should surface the raw count for operators; got: {issue!r}"
    )


# ── 4. Layer-2 drop_ratio closes the dense-citation bypass ────────────


def test_layer2_drop_ratio_closes_dense_inline_citation_bypass(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """3 wrapped sentinels with minimal padding — below the paranoid
    threshold but the citation scrubbing strips a large fraction of
    the content (each inline-backtick wrap removes ~25 chars), so
    drop_ratio > 30%. Layer-2's drop_ratio branch fires.

    This pins the *other* layer-2 rule: an attacker who keeps the raw
    count below the paranoid threshold still loses if the citation
    density is high. The two rules together cover the "low-noise"
    and "high-noise" wrapper-bypass shapes.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["rv"])
    state.agents["rv"].spec = fake_agent_spec(id="rv", output_file="review.md")
    _patch_cwd(monkeypatch, tmp_path)

    # ``dense=True`` strips the narrative wrapper around each
    # citation so the scrubbed prose is essentially empty —
    # drop_ratio approaches 90% — well clear of the 30% rule.
    # Three sentinels keeps raw_hits below paranoid_threshold (5),
    # so this can only trip via the drop_ratio branch.
    payload = _adversarial_inline_payload(n_sentinels=3, dense=True)
    _write_workspace_file(tmp_path, "review.md", payload)

    issue = _validate_output_artifact(state, "rv", result="ack")
    assert issue, "Layer-2 drop_ratio rule must close the dense-citation bypass"
    assert "drop_ratio" in issue, (
        f"Expected drop_ratio-branch label (raw=3 < paranoid=5); got: {issue!r}"
    )
    assert "3 sentinels" in issue, (
        f"Failure label should surface the raw count; got: {issue!r}"
    )


# ── 5. End-to-end safety-net → validate-quarantine adversarial chain ──


def test_end_to_end_adversarial_persist_then_layer2_validates_and_quarantines(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Full chain on adversarial input, layer-2 enforced:

      1. ``_persist_result_to_output_file_if_missing`` runs with the
         wrapped payload. Pre-scan MISSES (test #1 attests). File
         lands on disk.
      2. Caller invokes ``_validate_output_artifact``. Layer-1 MISSES
         (test #2 attests). Layer-2 paranoid catches.
      3. Caller invokes ``_quarantine_invalid_output``. The poisoned
         file moves to ``<output>.invalid``; the original path no
         longer carries the adversarial bytes.

    This is the realistic defense chain a crew run actually traverses
    when an adversarial backend lands the wrapped payload — the three
    components have to cooperate, and a regression in any one of them
    re-opens the bypass. Without this end-to-end test, individual
    component tests can each pass while the chain silently breaks.
    """
    from larkhelm.crew._runner import (
        _persist_result_to_output_file_if_missing,
        _quarantine_invalid_output,
        _validate_output_artifact,
    )
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["arch"])
    state.agents["arch"].spec = fake_agent_spec(id="arch", output_file="design.md")
    _patch_cwd(monkeypatch, tmp_path)

    payload = _adversarial_inline_payload(n_sentinels=6)

    # Step 1 — safety-net writes (pre-scan misses wrapped sentinels)
    _persist_result_to_output_file_if_missing(state, "arch", payload)
    out_path = tmp_path / ".crew_workspace" / "design.md"
    invalid_path = tmp_path / ".crew_workspace" / "design.md.invalid"
    assert out_path.exists(), "Step 1: pre-scan must let wrapped payload through"

    # Step 2 — validator catches via layer-2 paranoid
    issue = _validate_output_artifact(state, "arch", result="ack")
    assert issue, "Step 2: layer-2 paranoid must reject the on-disk artifact"
    assert "paranoid" in issue

    # Step 3 — quarantine atomically moves the file aside
    _quarantine_invalid_output(state, "arch")
    assert not out_path.exists(), (
        "Step 3: quarantine must move the poisoned file off the canonical path "
        "so downstream agents reading design.md do not pick up the adversarial bytes"
    )
    assert invalid_path.exists(), (
        f"Step 3: quarantine must preserve evidence under .invalid; workspace contents: "
        f"{sorted(p.name for p in (tmp_path / '.crew_workspace').iterdir())}"
    )
    assert _DSML_TOKEN in invalid_path.read_text(encoding="utf-8"), (
        "Step 3: quarantine must preserve the original bytes verbatim — "
        "redacting at this stage destroys evidence that operators need to "
        "tell apart 'truly adversarial backend' from 'prompt-injection victim'."
    )


# ── 6. Layer-2 + mixed citation forms — no false-positive on light citations ─


def test_layer2_mixed_citation_forms_with_high_count_still_caught(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """Adversarial input that interleaves all three citation forms
    (fenced, inline-backtick, blockquote) — exercising every pass of
    ``_strip_code_evidence`` simultaneously. 6 sentinels total
    (2 per form). Layer-2 paranoid must still close it.

    Without this test, a regression that quietly removed one of the
    three scrubber passes (e.g. dropping the blockquote pass) would
    survive every other test in the suite — the single-form fixtures
    in ``test_crew_failure_hardening.py`` would still pass because
    they exercise one pass each.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["rv"])
    state.agents["rv"].spec = fake_agent_spec(id="rv", output_file="review.md")
    _patch_cwd(monkeypatch, tmp_path)

    payload = _adversarial_mixed_payload(n_each=2)
    _write_workspace_file(tmp_path, "review.md", payload)

    issue = _validate_output_artifact(state, "rv", result="ack")
    assert issue, (
        "Layer-2 must reject 6 sentinels even when split across all three "
        "citation forms — paranoid is form-agnostic by design"
    )
    assert "paranoid" in issue or "drop_ratio" in issue, (
        f"Expected layer-2 label (paranoid or drop_ratio); got: {issue!r}"
    )


# ── 7. Realistic legitimate review with layer-2 ON must NOT trip ──────


def test_layer2_realistic_legitimate_27kb_review_passes(
    init_test_config, fake_crew_state, fake_agent_spec, monkeypatch, tmp_path,
):
    """False-positive regression for the 2026-05-25 SecurityExpert
    incident shape: a 27 KB legitimate security review that cites the
    DSML token twice inside fenced blocks. Pre-F1 this was wrongly
    quarantined. Post-F1+F2 layer-1 lets it pass. **Layer-2 enforced
    must also let it pass** — otherwise we have replaced one false-
    positive (layer-1) with another (layer-2).

    Pins the calibration of the default layer-2 thresholds
    (raw_threshold=3, drop_ratio=30%, paranoid=5) against a realistic
    legitimate body: 2 fenced citations + ~27 KB of analysis prose.
    """
    from larkhelm.crew._runner import _validate_output_artifact
    _set_layer2_config(enabled=True, traffic=1.0)
    state = fake_crew_state(["sec"])
    state.agents["sec"].spec = fake_agent_spec(id="sec", output_file="review_security.md")
    _patch_cwd(monkeypatch, tmp_path)

    body = (
        "# Security review v2 (2026-05-25 replay)\n\n"
        "## Finding 1 — DocsExpert tool-use leakage\n\n"
        "DocsExpert (DeepSeek-flavoured backend) emitted the DSML protocol\n"
        "token verbatim into its `output_file`. The on-disk artifact is\n"
        "preserved at `review_docs.md.invalid` for inspection — sample lines:\n\n"
        f"```\n{_DSML_TOKEN}s>\n```\n\n"
        "Root cause: the F3 resolver-side gate (tool-capable backends only\n"
        "for `output_file`-bearing agents) had not yet landed at the time\n"
        "of the incident.\n\n"
        + ("Analysis paragraph documenting downstream impact. " * 400)
        + "\n\n## Finding 2 — Validator scrubber over-strict (now fixed)\n\n"
        "The reviewer's report mentioned the DSML token a second time when\n"
        "summarising the failure mode:\n\n"
        f"```\n{_DSML_TOKEN}\n```\n\n"
        "F1+F2 (this batch) extended `_strip_code_evidence` to cover the\n"
        "inline-backtick and blockquote citation forms in addition to the\n"
        "fenced form, and removed the 8 KiB scan cap.\n\n"
        + ("Further analysis prose elaborating each finding. " * 400)
    )
    assert len(body) > 27_000, f"Fixture must reach realistic 27 KB shape; got {len(body)}"
    _write_workspace_file(tmp_path, "review_security.md", body)

    issue = _validate_output_artifact(state, "sec", result="ack")
    assert issue == "", (
        f"Default layer-2 thresholds must not flag a 27 KB legitimate review "
        f"with only 2 fenced citations of the DSML token. If this fails, "
        f"calibrate {{raw_threshold, drop_ratio, paranoid}} BEFORE landing — "
        f"the 2026-05-25 incident proved the false-positive cost is high. "
        f"Got: {issue!r}"
    )
