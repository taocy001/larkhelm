"""larkhelm · project memory section-based layout (P2 REQ-05.2).

Splits the single ``project_{hash16}.md`` body into 4 fixed H2 sections —
``TechStack`` / ``Conventions`` / ``Architecture`` / ``Constraints`` —
each holding domain-specific facts. The cheap-LLM extractor is later
prompted to emit targeted ``## <Section>`` patches instead of rewriting
the entire body, cutting token cost on partial updates.

On-disk layout:

    ---
    updated_at: "..."
    schema_version: "2"
    sections: "TechStack,Conventions,Architecture,Constraints"
    ---
    ## TechStack
    ...

    ## Conventions
    ...

Backwards compatibility: a file without recognised ``## <Section>``
headings is treated as a single chunk under the *first* section
(``TechStack``); the loader returns the expected key set so the renderer
emits the body unchanged.

Gated by ``memory_project_section_enabled`` (default False).
"""
from __future__ import annotations

import re

import larkhelm.config as _cfg


# Fixed ordering — render_for_context emits in this sequence so the LLM
# prompt sees the same arrangement turn-after-turn (prompt caching win).
SECTION_NAMES: tuple[str, ...] = (
    "TechStack",
    "Conventions",
    "Architecture",
    "Constraints",
)

# Per-section soft cap; the four together respect PROJECT_MAX_CHARS=1500.
SECTION_BUDGET: int = 400
# Combined cap used only for the legacy free-form fallback in ``parse_body``
# (matches PROJECT_MAX_CHARS so we don't truncate existing project memory
# below the layer's real budget on first read after P5-OPT6 flag flip).
SECTION_LEGACY_CAP: int = 1500

_SCHEMA_VERSION = "2"

# Case-insensitive heading match. Bare ``## <Section>`` only — a trailing
# decoration like ``## TechStack — pinned`` would not match (intentional:
# free-form trailing text shouldn't be treated as a section header).
_HEADING_RE = re.compile(
    r"(?im)^##\s+(" + "|".join(SECTION_NAMES) + r")\s*$"
)


def is_enabled() -> bool:
    """Read the operator gate; defaults to False so P1 behaviour holds."""
    return bool(getattr(_cfg, "MEMORY_PROJECT_SECTION_ENABLED", False))


def parse_body(body: str) -> dict[str, str]:
    """Split a body on ``## <Section>`` headings; missing sections → empty.

    Stray text before the first heading is appended to the first matched
    section so manual edits at the top don't vanish on next save.

    Returns a dict with every SECTION_NAMES key present (empty string
    when the section is absent) — keeps caller code simpler.
    """
    out: dict[str, str] = {s: "" for s in SECTION_NAMES}
    if not body:
        return out

    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        # Legacy free-form body: cap to SECTION_LEGACY_CAP (== layer
        # budget) rather than SECTION_BUDGET (per-section cap). Applying
        # the per-section cap here would silently truncate existing
        # free-form project memory from up to 1500 chars down to 400 the
        # first time the file is read after ``memory_project_section_enabled``
        # flips to true (P5-OPT6 blocker — reviewed). Subsequent saves
        # re-emit under proper headings, at which point the per-section
        # cap kicks in.
        out[SECTION_NAMES[0]] = body.strip()[:SECTION_LEGACY_CAP]
        return out

    pre_text = body[: matches[0].start()].strip()
    if pre_text:
        first = matches[0].group(1)
        out[first] = (pre_text + "\n" + out[first]).strip()

    for i, m in enumerate(matches):
        sec = m.group(1)
        # Canonical-case lookup: the heading may have arrived as
        # ``## techstack`` from a markdown linter; map back to the
        # SECTION_NAMES canonical spelling.
        canonical = next((s for s in SECTION_NAMES if s.lower() == sec.lower()), sec)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        if not chunk:
            continue
        if out.get(canonical):
            out[canonical] = (out[canonical] + "\n" + chunk).strip()
        else:
            out[canonical] = chunk

    for s in SECTION_NAMES:
        out[s] = out[s][:SECTION_BUDGET]
    return out


def load_project_sections(cwd: str) -> dict[str, str]:
    """Load the section dict for ``cwd``'s project memory.

    Returns the all-empty default when no project file exists for this
    working directory — the cwd hashing rules are owned by ``memory.py``.
    """
    from larkhelm import memory as _mem
    body = _mem.load_project_memory(cwd)
    if body is None:
        return {s: "" for s in SECTION_NAMES}
    return parse_body(body)


def save_project_sections(cwd: str, sections: dict[str, str]) -> None:
    """Persist ``sections`` as a schema_version=2 file via memory.save_project_memory."""
    from larkhelm import memory as _mem

    trimmed: dict[str, str] = {}
    for s in SECTION_NAMES:
        v = (sections.get(s) or "").strip()
        if v:
            trimmed[s] = v[:SECTION_BUDGET]

    body_parts: list[str] = []
    for s in SECTION_NAMES:
        if s in trimmed:
            body_parts.append(f"## {s}\n{trimmed[s]}")
    body = "\n\n".join(body_parts)

    # save_project_memory enforces the cwd→hash→path mapping and runs
    # the same atomic-write + 0600 chmod as global. We piggy-back on its
    # extra_fm_pairs so the schema version sticks to disk.
    extra_pairs = {
        "schema_version": _SCHEMA_VERSION,
        "sections":       ",".join(SECTION_NAMES),
    }
    _mem.save_project_memory(cwd, body, extra_fm_pairs=extra_pairs)


def render_for_context(sections: dict[str, str]) -> str:
    """Render sections back to Markdown for memory_context injection.

    Returns empty string when every section is empty so the caller can
    skip emitting ``[PROJECT MEMORY]`` tags entirely.
    """
    parts: list[str] = []
    for s in SECTION_NAMES:
        v = (sections.get(s) or "").strip()
        if v:
            parts.append(f"## {s}\n{v}")
    return "\n\n".join(parts)


__all__ = [
    "SECTION_NAMES",
    "SECTION_BUDGET",
    "is_enabled",
    "parse_body",
    "load_project_sections",
    "save_project_sections",
    "render_for_context",
]
