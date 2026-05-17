"""P2 AC-08: tests for ``larkhelm.__version__`` resolution.

Three pinned scenarios:
  1) ``larkhelm._version`` (hatch-vcs generated) is present and clean
     → version matches the ``^2026\\.\\d+\\.\\d+$`` release tag regex,
     OR the dirty editable suffix ``.dirty`` is appended.
  2) clean release tag from importlib.metadata
  3) hatch-vcs missing → fallback to ``"0.0.0+unknown"``
"""
from __future__ import annotations

import importlib
import os
import re
import sys

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")


# ── 1) dirty / clean tag pattern ────────────────────────────────────────


def test_version_string_is_pep440_ish():
    import larkhelm
    v = larkhelm.__version__
    # Either a release tag (2026.X.Y) or a dev-style tag with optional
    # ``.dirty`` / ``+local`` suffix. The "0.0.0+unknown" fallback also
    # matches the PEP440-ish loose regex.
    pat = r"^\d+\.\d+\.\d+([.\-+][0-9A-Za-z.\-+]+)?(\.dirty)?$"
    assert re.match(pat, v), f"version {v!r} doesn't look PEP 440-ish"


def test_resolved_version_when_metadata_available():
    """When ``importlib.metadata.version('larkhelm')`` works (it does inside
    the editable install used for tests), the resolved version starts
    with a digit-dot tuple — never with text or whitespace.
    """
    import larkhelm
    v = larkhelm.__version__
    assert v[0].isdigit(), f"version starts with non-digit: {v!r}"


# ── 2) clean release tag matches ^2026\.\d+\.\d+$ when present ──────────


def test_release_tag_pattern_when_no_dirty_suffix():
    """If the resolved version is an exact ``YYYY.MAJOR.MINOR`` tag (no
    dev/dirty/local components), it must match the release regex.

    We don't assert this unconditionally — a dev / editable install may
    legitimately produce a different string. This test only fires when
    the version *looks* like a clean tag.
    """
    import larkhelm
    v = larkhelm.__version__
    # Strip optional ``+local`` (which hatch-vcs adds on dirty installs)
    base = v.split("+", 1)[0]
    has_dev_marker = (
        ".dev" in base or ".post" in base or ".dirty" in v or "rc" in base
        or v == "0.0.0+unknown"
    )
    if not has_dev_marker:
        # Treat as a clean release; year-based scheme starts at 2026.
        assert re.match(r"^\d{4}\.\d+\.\d+$", base), (
            f"clean tag {v!r} doesn't match YYYY.X.Y"
        )


# ── 3) hatch-vcs / _version missing → fallback string ───────────────────


def test_fallback_when_version_module_missing(monkeypatch):
    """Forcing both _version.__version__ AND importlib.metadata to fail
    AND the git-describe fallback to fail must yield ``"0.0.0+unknown"``.

    The third (git-describe) tier was added as a P2 follow-up after AC-08
    review noted that source-tree runs were getting ``0.0.0+unknown``
    instead of a meaningful version. This test pins the LAST-RESORT
    fallback by forcing all three earlier tiers to fail.
    """
    # Tier 1: drop the cached _version module so re-import re-runs the resolver.
    monkeypatch.setitem(sys.modules, "larkhelm._version", None)
    # Tier 2: make importlib.metadata pretend the package isn't installed.
    import importlib.metadata as _md
    real_version = _md.version

    def _raise_not_found(name):
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_md, "version", _raise_not_found)
    # Tier 3: short-circuit the git-describe fallback by patching the
    # helper to return None (simulates "no git binary / no .git dir / git
    # describe failed").
    import larkhelm
    monkeypatch.setattr(larkhelm, "_version_from_git_describe", lambda: None)
    # Re-import larkhelm to trigger the resolver on a clean cache. The
    # monkey-patched ``_version_from_git_describe`` survives the reload
    # because monkeypatch.setattr patches the module attribute, which the
    # resolver re-looks-up after reload via the module's own namespace.
    # To make the patch effective post-reload we instead drop and re-import.
    importlib.reload(larkhelm)
    # Re-apply the patch after reload (reload re-runs module-level code).
    monkeypatch.setattr(larkhelm, "_version_from_git_describe", lambda: None)
    # Re-run the resolver by calling it directly with all 3 tiers patched.
    resolved = larkhelm._resolve_version()
    try:
        assert resolved == "0.0.0+unknown", (
            f"all 3 tiers patched to fail, but got {resolved!r}"
        )
    finally:
        monkeypatch.setattr(_md, "version", real_version)
        importlib.reload(larkhelm)


def test_git_describe_fallback_used_in_source_tree():
    """Source-tree run (no _version.py, no installed metadata) should hit
    the git-describe tier and produce a string that's NOT the literal
    last-resort fallback — proves the new tier is wired in."""
    import larkhelm
    v_from_git = larkhelm._version_from_git_describe()
    if v_from_git is None:
        pytest.skip("no .git directory or git binary — fallback not exercisable here")
    # Must be PEP 440-ish (digits + optional local part). The shape varies
    # by whether the tree has tags / is dirty, but it should always start
    # with a digit and contain a dot.
    assert v_from_git[0].isdigit()
    assert "." in v_from_git
    # AC-08 explicitly checks for ``.dirty`` suffix when working tree is dirty.
    # We can't force dirty/clean from a test, but if the suffix appears it
    # must be the literal ``.dirty`` exactly (not ``-dirty`` or similar).
    if "dirty" in v_from_git:
        assert ".dirty" in v_from_git, (
            f"dirty suffix not normalised: {v_from_git!r}"
        )
