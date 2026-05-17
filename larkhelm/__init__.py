"""larkhelm — Feishu ↔ Claude / Gemini / Kimi bridge.

Version resolution order (P2 AC-08):

1. ``larkhelm._version.__version__`` — written by ``hatch-vcs`` at build
   time. Present in any wheel / sdist / properly built editable install
   triggered by ``pip install -e .``.
2. ``importlib.metadata.version("larkhelm")`` — present when the package
   has been installed (pip / pipx) but the generated ``_version.py`` was
   stripped (rare; happens with vendored copies).
3. ``git describe --tags --always --dirty`` — when neither metadata path
   is available (source-tree run from a freshly-cloned repo without
   ``pip install`` ever firing). Normalised to PEP 440-ish form so
   downstream tooling that parses ``__version__`` keeps working. This is
   the path that lets a developer run ``python -m larkhelm --version``
   straight from a checkout and still see a meaningful version string.
4. ``"0.0.0+unknown"`` — last-resort fallback so ``import larkhelm`` never
   fails on a tarball / submodule without git history.

The hard-coded literal at tier 4 stays as a safety net but should be
unreachable in practice; tiers 1–3 cover wheel install / editable
install / source checkout respectively.
"""
from __future__ import annotations

import re


def _version_from_git_describe() -> str | None:
    """Tier 3 fallback: run ``git describe`` against the repo root.

    Returns ``None`` if git is unavailable, the working tree isn't a
    repo, or describe produced no useful output. Output is normalised:

    * Tagged + clean: ``2026.5.17``                       → ``2026.5.17``
    * Tagged + ahead: ``2026.5.17-12-gabc1234``           → ``2026.5.17.dev12+gabc1234``
    * Tagged + dirty: ``2026.5.17-12-gabc1234-dirty``     → ``2026.5.17.dev12+gabc1234.dirty``
    * No tag, clean:  ``abc1234``                          → ``0.0.0+gabc1234``
    * No tag, dirty:  ``abc1234-dirty``                    → ``0.0.0+gabc1234.dirty``

    The dirty suffix matches the AC-08 contract (regex ``\\.dirty``).
    """
    import shutil
    import subprocess
    from pathlib import Path
    if shutil.which("git") is None:
        return None
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ".git").exists():
        return None
    try:
        raw = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout.strip()
    except Exception:
        return None
    if not raw:
        return None
    # Tagged forms: ``TAG`` or ``TAG-N-gSHA`` (optionally trailing ``-dirty``).
    m = re.match(r"^(\S+?)-(\d+)-g([0-9a-f]+)(-dirty)?$", raw)
    if m:
        tag, ahead, sha, dirty = m.groups()
        suffix = ".dirty" if dirty else ""
        return f"{tag}.dev{ahead}+g{sha}{suffix}"
    m = re.match(r"^(\S+?)(-dirty)?$", raw)
    if m:
        body, dirty = m.groups()
        # Bare SHA (no tag at all): ``git describe --always`` prints just
        # the SHA. Detect by absence of dots.
        if re.match(r"^[0-9a-f]+$", body) and "." not in body:
            suffix = ".dirty" if dirty else ""
            return f"0.0.0+g{body}{suffix}"
        # Tagged + clean: just the tag.
        if dirty:
            return f"{body}+dirty"
        return body
    return None


def _resolve_version() -> str:
    try:
        from larkhelm._version import __version__ as _v  # type: ignore[attr-defined]
        if _v:
            return str(_v)
    except Exception:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version as _md_version
        try:
            return _md_version("larkhelm")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    git_ver = _version_from_git_describe()
    if git_ver:
        return git_ver
    return "0.0.0+unknown"


__version__: str = _resolve_version()
