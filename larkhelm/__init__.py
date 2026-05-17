"""larkhelm — Feishu ↔ Claude / Gemini / Kimi bridge.

Version resolution order (P2 AC-08):

1. ``larkhelm._version.__version__`` — written by ``hatch-vcs`` at build
   time. Present in any wheel / sdist / properly built editable install.
2. ``importlib.metadata.version("larkhelm")`` — present when the package
   has been installed (pip/pipx) but the generated ``_version.py`` was
   stripped (rare; happens with vendored copies).
3. ``"0.0.0+unknown"`` — last-resort fallback so ``import larkhelm`` never
   fails on a source-only checkout without metadata.

The hard-coded literal that used to live here was removed because
``hatch.version.path`` could not express the "dirty editable" suffix
(``.dirty``) AC-08 needs.
"""
from __future__ import annotations


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
    return "0.0.0+unknown"


__version__: str = _resolve_version()
