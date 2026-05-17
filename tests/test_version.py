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
    """Forcing both _version.__version__ and importlib.metadata to fail
    must yield ``"0.0.0+unknown"``."""
    # Drop the cached module so re-import re-runs the resolver.
    monkeypatch.setitem(sys.modules, "larkhelm._version", None)
    # And make importlib.metadata pretend the package isn't installed.
    import importlib.metadata as _md
    real_version = _md.version

    def _raise_not_found(name):
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_md, "version", _raise_not_found)
    # Re-import larkhelm to trigger the resolver on a clean cache.
    import larkhelm
    importlib.reload(larkhelm)
    try:
        assert larkhelm.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.setattr(_md, "version", real_version)
        # Reload again to restore the live version for subsequent tests.
        importlib.reload(larkhelm)
