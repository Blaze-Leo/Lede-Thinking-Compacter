"""v0.4.2 heading/pin retention helpers (pure, deterministic).

Locates headings by index in a sentence list and renders pinned content.
Reuses lede._headings for all "what is a heading" logic; adds no new
heading detection. The Rust mirror is rust/src/pins.rs.
"""
from __future__ import annotations

from collections.abc import Sequence

from lede._headings import is_structural_heading, md_depth


def nearest_heading_map(sentences: list[str]) -> list[int | None]:
    """For each sentence index, the index of the most recent heading
    sentence at or before it, or None if no heading precedes it.

    A heading's own entry is None (headings are never selected, so the
    value is unused; None keeps the rule "the heading above me")."""
    out: list[int | None] = []
    current: int | None = None
    for i, s in enumerate(sentences):
        if is_structural_heading(s):
            out.append(None)
            current = i
        else:
            out.append(current)
    return out


def document_title_index(sentences: list[str]) -> int | None:
    """Index of the document title: a depth-1 heading appearing before
    any body sentence. None if the first sentence-ish line is body, or
    the first heading is not markdown depth 1."""
    for i, s in enumerate(sentences):
        if is_structural_heading(s):
            return i if md_depth(s) == 1 else None
        return None
    return None


def render_toc_from_list(headings: Sequence[str]) -> str:
    """TOC from caller-supplied headings: deduped (first wins), given order,
    one per line, no indentation."""
    seen: set[str] = set()
    out: list[str] = []
    for h in headings:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return "\n".join(out)


def render_toc(text: str) -> str:
    """Render a table of contents from the document outline. Each
    section name on its own line, indented (depth-1)*2 spaces."""
    from lede.extract.outline import outline

    lines: list[str] = []
    for section in outline(text):
        indent = "  " * max(0, section.depth - 1)
        lines.append(f"{indent}{section.name}")
    return "\n".join(lines)


def _render_keep_headings_override(
    sentences: list[str],
    selected: list[int],
    headings: Sequence[str],
) -> tuple[str, list[str]]:
    """Render body with caller-supplied headings per §4.3 of the spec.

    Returns (body_string, pinned_list) where pinned_list is every heading
    emitted in emission order."""
    seen: set[str] = set()
    H: list[str] = []
    for h in headings:
        if h not in seen:
            seen.add(h)
            H.append(h)

    matched_pos: dict[str, int] = {}
    claimed: set[int] = set()
    for h in H:
        hs = h.strip()
        for i, s in enumerate(sentences):
            if i in claimed:
                continue
            if s.strip() == hs:
                matched_pos[h] = i
                claimed.add(i)
                break
    pos_to_heading = {idx: h for h, idx in matched_pos.items()}
    matched_indices = sorted(matched_pos.values())

    pinned: list[str] = []
    out_lines: list[str] = []
    buf: list[str] = []
    emitted_idx: set[int] = set()
    emitted_text: set[str] = set()

    def flush() -> None:
        if buf:
            out_lines.append(" ".join(buf))
            buf.clear()

    title = H[0] if H else None
    if title is not None:
        out_lines.append(title)
        pinned.append(title)
        emitted_text.add(title)
        if title in matched_pos:
            emitted_idx.add(matched_pos[title])

    for h in H:
        if h in emitted_text:
            continue
        if h not in matched_pos:
            out_lines.append(h)
            pinned.append(h)
            emitted_text.add(h)

    def nearest_matched(s_idx: int) -> int | None:
        best: int | None = None
        for mi in matched_indices:
            if mi <= s_idx:
                best = mi
            else:
                break
        return best

    for s_idx in selected:
        mi = nearest_matched(s_idx)
        if mi is not None and mi not in emitted_idx:
            flush()
            h = pos_to_heading[mi]
            out_lines.append(h)
            pinned.append(h)
            emitted_idx.add(mi)
            emitted_text.add(h)
        buf.append(sentences[s_idx])
    flush()

    # Survival guarantee (§4.3 step 6): any supplied heading not yet emitted —
    # i.e. it matched a body line but that section had no selected sentence —
    # is appended so every heading in H is present in the output.
    for h in H:
        if h not in emitted_text:
            out_lines.append(h)
            pinned.append(h)
            emitted_text.add(h)

    return "\n".join(out_lines), pinned


def render_with_pins(
    sentences: list[str],
    selected: list[int],
    *,
    keep_headings: bool,
    include_toc: bool,
    pin: Sequence[str] | None,
    text: str,
    headings: Sequence[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Weave pinned content around the extractive body.

    Returns (output_string, pinned_headings). `selected` must be sorted
    ascending. Headings interleave in document order; the title (if any)
    is pinned at the top; `pin` lines and the TOC prepend as blocks in
    the order pin -> toc -> body. Deterministic: dedup preserves first
    occurrence; no ordering depends on hashing or locale."""
    pinned_headings: list[str] = []

    if keep_headings:
        if headings:
            body, pinned_list = _render_keep_headings_override(sentences, selected, headings)
            pinned_headings = pinned_list
        else:
            heading_of = nearest_heading_map(sentences)
            title_idx = document_title_index(sentences)
            emitted: set[int] = set()
            out_lines: list[str] = []
            buf: list[str] = []

            def flush() -> None:
                if buf:
                    out_lines.append(" ".join(buf))
                    buf.clear()

            if title_idx is not None:
                out_lines.append(sentences[title_idx])
                emitted.add(title_idx)
                pinned_headings.append(sentences[title_idx])

            for s_idx in selected:
                h = heading_of[s_idx]
                if h is not None and h not in emitted:
                    flush()
                    out_lines.append(sentences[h])
                    emitted.add(h)
                    pinned_headings.append(sentences[h])
                buf.append(sentences[s_idx])
            flush()

            if pinned_headings:
                body = "\n".join(out_lines)
            else:
                # No headings pinned: byte-identical to the plain scorer.
                body = " ".join(sentences[i] for i in selected)
    else:
        body = " ".join(sentences[i] for i in selected)

    blocks: list[str] = []
    if pin:
        blocks.append("\n".join(pin))
    if include_toc:
        toc_text = render_toc_from_list(headings) if headings else render_toc(text)
        if toc_text:
            blocks.append(toc_text)
    blocks.append(body)
    output = "\n\n".join(b for b in blocks if b)
    return output, tuple(pinned_headings)


def prepend_blocks(
    body: str,
    *,
    include_toc: bool,
    pin: Sequence[str] | None,
    text: str,
    headings: Sequence[str] | None = None,
) -> str:
    """Prepend pin and TOC blocks (in that order) to an already-rendered
    body string. Used when keep_headings is off but pin/include_toc are
    on, so the body stays byte-identical to the non-pinned scorer."""
    blocks: list[str] = []
    if pin:
        blocks.append("\n".join(pin))
    if include_toc:
        toc_text = render_toc_from_list(headings) if headings else render_toc(text)
        if toc_text:
            blocks.append(toc_text)
    if body:
        blocks.append(body)
    return "\n\n".join(b for b in blocks if b)
