"""larkhelm · global memory slot-based layout (P2 REQ-05.1).

Splits the single ``global_{open_id}.md`` body into 4 typed slots
(``style`` / ``format`` / ``domain`` / ``expertise``), each ≤ 200 chars,
total ≤ 800 chars — preserving the existing GLOBAL_MAX_CHARS budget.

On-disk layout (schema_version="2"):

    ---
    updated_at: "..."
    schema_version: "2"
    slots: "style,format,domain,expertise"
    slot_lens: "180,140,90,210"
    ---
    ## style
    ...

    ## format
    ...

Backwards compatibility: when a file has no ``slots:`` frontmatter (or
``schema_version`` is < 2), the entire body is returned as the ``style``
slot so the legacy free-form text keeps influencing AI prompts. New writes
always produce the slotted layout.

Default behaviour: this module's effects are gated by
``memory_global_profile_slot_enabled`` (default False). When the flag is
off, ``memory_context.MemoryContextBuilder._layer_global`` calls
``load_global_memory`` directly — the slot path is dead code.
"""
from __future__ import annotations

import re

import larkhelm.config as _cfg


# Tuple ordering matters: render_for_context emits slots in this order so a
# downstream LLM prompt sees the same arrangement turn-after-turn (better
# prompt caching). Don't reorder without bumping schema_version.
SLOT_NAMES: tuple[str, ...] = ("style", "format", "domain", "expertise")
SLOT_BUDGET: int = 200          # per-slot char cap
SLOT_TOTAL: int = 800           # combined cap; matches GLOBAL_MAX_CHARS

# Schema marker written into frontmatter when this module persists. The
# loader accepts ``"1"`` (legacy, no slots) and ``"2"`` (slotted body).
_SCHEMA_VERSION = "2"

# Heading regex: only ``## <slot>`` at line start, single-word slot name
# matching SLOT_NAMES (case-insensitive). Looser matches risk merging
# accidental ``## TODO`` or markdown headers from user content.
_HEADING_RE = re.compile(
    r"(?im)^##\s+(" + "|".join(SLOT_NAMES) + r")\s*$"
)


def is_enabled() -> bool:
    """Read the operator gate; defaults to False so P1 behaviour holds."""
    return bool(getattr(_cfg, "MEMORY_GLOBAL_PROFILE_SLOT_ENABLED", False))


def parse_body(body: str) -> dict[str, str]:
    """Parse a slot-formatted body into ``{slot_name: text}``.

    Heuristics:
      * No ``## <slot>`` heading present → entire trimmed body goes into
        ``style`` and the other three slots stay empty (legacy fallback).
      * Multiple ``## <slot>`` headings present → split on heading lines,
        whitespace-trimmed values; duplicates are joined with a newline.
      * Stray content before the first heading is appended to the first
        recognised slot (so a hand-edited file's prose doesn't disappear).

    Always returns the full SLOT_NAMES key set so callers don't need
    `.get(slot, "")` ceremony.
    """
    out: dict[str, str] = {s: "" for s in SLOT_NAMES}
    if not body:
        return out

    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        text = body.strip()
        if text:
            out["style"] = text[:SLOT_BUDGET]
        return out

    # Pre-heading prose: stick it on the first slot's bucket so a manual
    # edit ("free text at the top") survives the next load/save cycle.
    pre_text = body[: matches[0].start()].strip()
    if pre_text:
        first_slot = matches[0].group(1).lower()
        out[first_slot] = (pre_text + "\n" + out[first_slot]).strip()

    for i, m in enumerate(matches):
        slot = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        if not chunk:
            continue
        # Concatenate duplicates rather than overwrite — preserves manual
        # edits that produced two ``## style`` sections.
        if out.get(slot):
            out[slot] = (out[slot] + "\n" + chunk).strip()
        else:
            out[slot] = chunk

    for s in SLOT_NAMES:
        out[s] = out[s][:SLOT_BUDGET]
    return out


def load_global_slots(chat_id: str | None, *,
                      sender_open_id: str | None = None) -> dict[str, str]:
    """Load the slot dict for ``chat_id``'s global memory.

    Returns the default-all-empty dict when the file doesn't exist OR is
    older than schema_version=2 (in which case the entire body is mapped
    to ``style`` so the legacy text still reaches the prompt).
    """
    # Lazy import to break the memory.py ↔ memory_global_slots.py cycle.
    from larkhelm import memory as _mem
    path = _mem._global_memory_file(chat_id, sender_open_id=sender_open_id)
    body = _mem._load_md_body(path)
    if body is None:
        return {s: "" for s in SLOT_NAMES}
    return parse_body(body)


def save_global_slots(chat_id: str | None, slots: dict[str, str], *,
                      sender_open_id: str | None = None) -> None:
    """Persist ``slots`` as a schema_version=2 file.

    Enforces SLOT_BUDGET on each slot before writing. Empty slots are
    omitted from the body so the file stays compact (the loader re-fills
    missing slots with empty strings on read).
    """
    from larkhelm import memory as _mem
    path = _mem._global_memory_file(chat_id, sender_open_id=sender_open_id)
    if path is None:
        return

    trimmed: dict[str, str] = {}
    for s in SLOT_NAMES:
        v = (slots.get(s) or "").strip()
        if v:
            trimmed[s] = v[:SLOT_BUDGET]

    body_parts: list[str] = []
    for s in SLOT_NAMES:
        if s in trimmed:
            body_parts.append(f"## {s}\n{trimmed[s]}")
    body = "\n\n".join(body_parts)

    # ``slots`` field is informational; the loader works without it. The
    # ``slot_lens`` field is a quick debug aid for log line "global slots
    # 180/140/90/210" reads.
    slot_lens = ",".join(str(len(trimmed.get(s, ""))) for s in SLOT_NAMES)
    extra_pairs = {
        "schema_version": _SCHEMA_VERSION,
        "slots":          ",".join(SLOT_NAMES),
        "slot_lens":      slot_lens,
    }
    _mem._save_md(path, body, SLOT_TOTAL, extra_fm_pairs=extra_pairs)


def merge_slot_update(
    existing: dict[str, str],
    new_text: str,
    slot: str,
) -> dict[str, str]:
    """Return a fresh dict with ``slot`` replaced by ``new_text`` (capped).

    ``existing`` is NOT mutated. Unknown slot names are silently coerced
    to the closest valid one ("expertice" → "expertise") via a small
    typo-tolerant map; this is conservative — when no match, the update
    is dropped (preserving the existing slots unchanged) so a buggy
    extractor can't corrupt the file.
    """
    out = {s: existing.get(s, "") for s in SLOT_NAMES}
    s = (slot or "").strip().lower()
    if s not in SLOT_NAMES:
        # Tiny typo tolerance — common misspellings the cheap LLM produces.
        typo_map = {"styles": "style", "formats": "format",
                    "domains": "domain", "expertice": "expertise",
                    "expert": "expertise"}
        s = typo_map.get(s, "")
    if s in SLOT_NAMES:
        out[s] = (new_text or "").strip()[:SLOT_BUDGET]
    return out


def render_for_context(slots: dict[str, str]) -> str:
    """Render slots back to Markdown for memory_context injection.

    Returns the empty string when every slot is empty so the caller can
    short-circuit the "[GLOBAL MEMORY]" tag emission.
    """
    parts: list[str] = []
    for s in SLOT_NAMES:
        v = (slots.get(s) or "").strip()
        if v:
            parts.append(f"## {s}\n{v}")
    return "\n\n".join(parts)


__all__ = [
    "SLOT_NAMES",
    "SLOT_BUDGET",
    "SLOT_TOTAL",
    "is_enabled",
    "parse_body",
    "load_global_slots",
    "save_global_slots",
    "merge_slot_update",
    "render_for_context",
]
