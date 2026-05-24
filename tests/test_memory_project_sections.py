"""P2 REQ-05.2: tests for ``larkhelm.memory_project_sections``."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import memory_project_sections as _mps  # noqa: E402


@pytest.fixture
def isolated_memory_home(monkeypatch, tmp_path: Path):
    import larkhelm.memory as _mem
    monkeypatch.setattr(_mem, "MEMORY_HOME_DIR", tmp_path / "memory", raising=False)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    return tmp_path / "memory"


@pytest.fixture
def real_cwd(tmp_path: Path):
    """A real on-disk cwd so the project_memory_file path resolution works."""
    d = tmp_path / "project"
    d.mkdir()
    return str(d)


# ── parse_body ───────────────────────────────────────────────────────────


def test_parse_body_splits_on_h2_headings():
    body = (
        "## TechStack\nPython 3.13\n\n"
        "## Conventions\npreserve byte-compat\n\n"
        "## Architecture\nthree-tier memory\n\n"
        "## Constraints\nno new runtime deps"
    )
    sections = _mps.parse_body(body)
    assert "Python 3.13" in sections["TechStack"]
    assert "byte-compat" in sections["Conventions"]
    assert "three-tier" in sections["Architecture"]
    assert "no new runtime" in sections["Constraints"]


def test_parse_body_legacy_no_headings_to_first_section():
    body = "legacy free-form project memory body"
    out = _mps.parse_body(body)
    assert out["TechStack"] == body  # first section per fallback rule
    for s in ("Conventions", "Architecture", "Constraints"):
        assert out[s] == ""


def test_parse_body_legacy_not_truncated_to_section_budget():
    """P5-OPT6 BLOCKER (reviewer): flipping
    ``memory_project_section_enabled`` to true must NOT silently truncate
    existing free-form project memory from up to 1500 chars down to
    SECTION_BUDGET (400). Legacy path now caps to SECTION_LEGACY_CAP."""
    body = "y" * 1500
    out = _mps.parse_body(body)
    assert len(out["TechStack"]) == 1500, (
        f"legacy free-form body must keep all 1500 chars (was {len(out['TechStack'])});"
        f" if you re-introduce per-section SECTION_BUDGET on this path,"
        f" existing project memory drops by 73% on first read after the"
        f" flag flips."
    )


def test_parse_body_empty_returns_all_empty():
    out = _mps.parse_body("")
    assert out == {s: "" for s in _mps.SECTION_NAMES}


# ── render_for_context ──────────────────────────────────────────────────


def test_render_for_context_omits_empty_sections():
    rendered = _mps.render_for_context({
        "TechStack": "Python", "Conventions": "",
        "Architecture": "tiered", "Constraints": "",
    })
    assert "## TechStack\nPython" in rendered
    assert "## Architecture\ntiered" in rendered
    assert "Conventions" not in rendered
    assert "Constraints" not in rendered


def test_render_for_context_all_empty():
    assert _mps.render_for_context({s: "" for s in _mps.SECTION_NAMES}) == ""


# ── round-trip save/load + cwd check ────────────────────────────────────


def test_save_load_roundtrip_with_cwd_check(isolated_memory_home, real_cwd):
    sections = {
        "TechStack": "Python 3.13",
        "Conventions": "preserve byte-compat",
        "Architecture": "three-tier memory",
        "Constraints": "no new runtime deps",
    }
    _mps.save_project_sections(real_cwd, sections)
    loaded = _mps.load_project_sections(real_cwd)
    assert loaded == sections


def test_load_with_unknown_cwd_returns_empty_dict(isolated_memory_home):
    # No file written yet → all-empty.
    out = _mps.load_project_sections("/nonexistent/path")
    assert out == {s: "" for s in _mps.SECTION_NAMES}


# ── is_enabled() ─────────────────────────────────────────────────────────


def test_is_enabled_default_true():
    """P5-OPT6: runtime default true (``config.setdefault`` truth)."""
    assert _mps.is_enabled() is True


def test_is_enabled_can_be_disabled(monkeypatch):
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "MEMORY_PROJECT_SECTION_ENABLED", False, raising=False)
    assert _mps.is_enabled() is False


def test_is_enabled_honours_flag(monkeypatch):
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "MEMORY_PROJECT_SECTION_ENABLED", True, raising=False)
    assert _mps.is_enabled() is True
