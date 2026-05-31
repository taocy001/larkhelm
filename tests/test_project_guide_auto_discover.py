"""AC-04 / AC-05: project_guide auto-discover logic tests."""
import pytest
from unittest.mock import patch


def test_ac04_auto_discover_finds_claude_md(tmp_path, monkeypatch):
    """AC-04: CLAUDE.md in cwd is injected with outcome='auto_discovered'."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "config", {
        "project_guide_enabled": True,
        "project_guide_path": "",
        "project_guide_auto_discover": True,
    }, raising=False)

    # Write CLAUDE.md to tmp_path
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Project Guide\nThis is a test guide.", encoding="utf-8")

    captured_inc = []

    def fake_inc_ig(point, outcome):
        captured_inc.append((point, outcome))

    from larkhelm.handlers._query import _apply_project_guide_gate

    memory_ctx = "original context"
    new_ctx, outcome = _apply_project_guide_gate(str(tmp_path), memory_ctx, is_cli_claude=False)

    assert outcome == "auto_discovered", f"Expected 'auto_discovered', got {outcome!r}"
    assert "[Project Guide]" in new_ctx, f"Expected '[Project Guide]' in memory_ctx"
    assert "This is a test guide." in new_ctx


def test_ac04_auto_discover_finds_larkhelm_project_md(tmp_path, monkeypatch):
    """AC-04b: .larkhelm_project.md in cwd is injected when CLAUDE.md absent."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "config", {
        "project_guide_enabled": True,
        "project_guide_path": "",
        "project_guide_auto_discover": True,
    }, raising=False)

    lpm = tmp_path / ".larkhelm_project.md"
    lpm.write_text("# LarkHelm Project\nProject info.", encoding="utf-8")

    from larkhelm.handlers._query import _apply_project_guide_gate

    new_ctx, outcome = _apply_project_guide_gate(str(tmp_path), "", is_cli_claude=False)

    assert outcome == "auto_discovered"
    assert "Project info." in new_ctx


def test_ac05_auto_discover_not_found(tmp_path, monkeypatch):
    """AC-05: no guide file found → outcome='not_found_auto', memory_ctx unchanged."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "config", {
        "project_guide_enabled": True,
        "project_guide_path": "",
        "project_guide_auto_discover": True,
    }, raising=False)

    # tmp_path is empty — no CLAUDE.md or .larkhelm_project.md
    from larkhelm.handlers._query import _apply_project_guide_gate

    original = "original context"
    new_ctx, outcome = _apply_project_guide_gate(str(tmp_path), original, is_cli_claude=False)

    assert outcome == "not_found_auto", f"Expected 'not_found_auto', got {outcome!r}"
    assert new_ctx == original, "memory_ctx should be unchanged when guide not found"


def test_ac04_skipped_for_cli_claude(tmp_path, monkeypatch):
    """CLI Claude backends are skipped (they read CLAUDE.md themselves)."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "config", {
        "project_guide_enabled": True,
        "project_guide_path": "",
        "project_guide_auto_discover": True,
    }, raising=False)

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Guide content", encoding="utf-8")

    from larkhelm.handlers._query import _apply_project_guide_gate

    original = "ctx"
    new_ctx, outcome = _apply_project_guide_gate(str(tmp_path), original, is_cli_claude=True)

    assert outcome == "skipped_cli"
    assert new_ctx == original


def test_ac04_truncates_at_4000_chars(tmp_path, monkeypatch):
    """Guide content > 4000 chars is truncated with ellipsis."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "config", {
        "project_guide_enabled": True,
        "project_guide_path": "",
        "project_guide_auto_discover": True,
    }, raising=False)

    long_content = "x" * 5000
    (tmp_path / "CLAUDE.md").write_text(long_content, encoding="utf-8")

    from larkhelm.handlers._query import _apply_project_guide_gate

    new_ctx, outcome = _apply_project_guide_gate(str(tmp_path), "", is_cli_claude=False)

    assert outcome == "auto_discovered"
    assert "…" in new_ctx, "Truncated content should end with ellipsis"
    # Content injected should not exceed 4000 + ellipsis chars for the guide body
    guide_section = new_ctx.split("[/Project Guide]")[0]
    assert len(guide_section) < 5500  # sanity upper bound
