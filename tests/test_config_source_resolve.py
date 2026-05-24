"""Pin ``larkhelm.config._resolve_source_dir`` contracts.

The resolver decides where ``SOURCE_DIR`` points and whether the install is
editable. ``/upgrade`` (``commands._do_upgrade``) reads these to pick between
``pip install -e`` and ``pip install --force-reinstall`` and to surface a
friendly error when the source repo has gone missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from larkhelm.config import _resolve_source_dir


def _git_init(path: Path) -> None:
    """Create just enough of a `.git` directory to satisfy the ``(p / ".git").exists()`` probe."""
    (path / ".git").mkdir(parents=True, exist_ok=True)


def test_editable_install_returns_repo_root(tmp_path: Path):
    """Editable layout: ``<repo>/larkhelm/config.py`` with ``<repo>/.git``."""
    repo = tmp_path / "repo"
    pkg = repo / "larkhelm"
    pkg.mkdir(parents=True)
    _git_init(repo)
    config_file = pkg / "config.py"
    config_file.write_text("")

    src, editable = _resolve_source_dir(config_file)
    assert src == repo
    assert editable is True


def test_non_editable_with_direct_url_returns_source_path(tmp_path: Path):
    """Non-editable + ``direct_url.json`` points to a live git source dir."""
    # Simulate a real source repo
    real_source = tmp_path / "src" / "larkhelm-source"
    real_source.mkdir(parents=True)
    _git_init(real_source)

    # Simulate a site-packages layout without ``.git``
    site_packages = tmp_path / "venv" / "site-packages"
    pkg = site_packages / "larkhelm"
    pkg.mkdir(parents=True)
    config_file = pkg / "config.py"
    config_file.write_text("")

    dist_info = site_packages / "larkhelm-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "direct_url.json").write_text(json.dumps({
        "dir_info": {},
        "url": f"file://{real_source}",
    }))

    src, editable = _resolve_source_dir(config_file)
    assert src == real_source
    assert editable is False


def test_non_editable_direct_url_path_deleted_falls_back(tmp_path: Path):
    """If ``direct_url.json`` points to a path that no longer exists (or
    isn't a git repo), return the legacy package-dir fallback so callers
    can detect the failure mode and surface a clear error."""
    site_packages = tmp_path / "venv" / "site-packages"
    pkg = site_packages / "larkhelm"
    pkg.mkdir(parents=True)
    config_file = pkg / "config.py"
    config_file.write_text("")

    dist_info = site_packages / "larkhelm-1.2.3.dist-info"
    dist_info.mkdir()
    # Points to a directory that doesn't exist on disk
    (dist_info / "direct_url.json").write_text(json.dumps({
        "dir_info": {},
        "url": "file:///does/not/exist/larkhelm-source",
    }))

    src, editable = _resolve_source_dir(config_file)
    assert src == pkg, "should fall back to the package dir when source path is gone"
    assert editable is False


def test_non_editable_no_dist_info_falls_back(tmp_path: Path):
    """Wheel install from PyPI (no ``direct_url.json``) → fallback."""
    site_packages = tmp_path / "venv" / "site-packages"
    pkg = site_packages / "larkhelm"
    pkg.mkdir(parents=True)
    config_file = pkg / "config.py"
    config_file.write_text("")
    # Note: no dist-info at all

    src, editable = _resolve_source_dir(config_file)
    assert src == pkg
    assert editable is False


def test_direct_url_with_non_file_scheme_falls_back(tmp_path: Path):
    """VCS-installed packages have ``vcs_info`` and a non-``file://`` URL —
    not something ``/upgrade``'s ``git pull`` can use directly, so we fall
    back rather than handing back a bogus path."""
    site_packages = tmp_path / "venv" / "site-packages"
    pkg = site_packages / "larkhelm"
    pkg.mkdir(parents=True)
    config_file = pkg / "config.py"
    config_file.write_text("")

    dist_info = site_packages / "larkhelm-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "direct_url.json").write_text(json.dumps({
        "vcs_info": {"vcs": "git", "commit_id": "abc"},
        "url": "https://github.com/example/larkhelm",
    }))

    src, editable = _resolve_source_dir(config_file)
    assert src == pkg
    assert editable is False


def test_corrupt_direct_url_json_is_swallowed(tmp_path: Path, capsys):
    """Invalid JSON must not crash — fall back to package dir and log to stderr."""
    site_packages = tmp_path / "venv" / "site-packages"
    pkg = site_packages / "larkhelm"
    pkg.mkdir(parents=True)
    config_file = pkg / "config.py"
    config_file.write_text("")

    dist_info = site_packages / "larkhelm-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "direct_url.json").write_text("{this is not json")

    src, editable = _resolve_source_dir(config_file)
    assert src == pkg
    assert editable is False
    err = capsys.readouterr().err
    assert "direct_url.json scan failed" in err


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
