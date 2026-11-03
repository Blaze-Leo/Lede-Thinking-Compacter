"""Hierarchical outline extractor.

Detects sections via structural heading patterns (markdown, allcaps,
short-colon-label). For each section picks the highest-composite-score
non-heading sentence as the representative.

Note: outline() uses a narrower heading predicate than _headings.is_heading
— it does NOT treat "fewer than 4 content tokens" as a heading signal,
because that heuristic misclassifies short body sentences as headings,
leaving sections with no representative-sentence candidates. Outline
needs real section structure, not title-length prose.

Mirrors rust/src/extract/outline.rs.
"""
import re

from .._headings import heading_name, is_structural_heading, md_depth
from .._types import Section
from ..sentences import split_sentences
from ..tfidf import _composite_score_parts, _separate_heading_lines

# Heading patterns + helpers live in lede._headings — single source of
# truth so a new pattern requires one edit, not two. T13b Class D cases
# intentionally NOT handled (still tracked as follow-ups):
#   - "Meeting: Platform Migration Planning" (Label:Subject inline pattern)
#   - "Held: Section 412(b) does not authorize..." (inline colon-label
#     with body text on the same line)
#   - "Reply from support (Kai T., day 1)" (parenthetical structured
#     heading; too corpus-specific for a general predicate)

# Local aliases so existing call sites keep their names; both come from
# the shared module.
_is_structural_heading = is_structural_heading
_md_depth = md_depth


def outline(text: str) -> tuple[Section, ...]:
    """Extract sections: heading + top-scoring non-heading sentence within."""
    if not text:
        return ()

    # Pre-split heading-only lines so they become standalone sentences —
    # mirrors tfidf.summarize's default-mode preprocessing.
    prepared = _separate_heading_lines(text)
    sentences = split_sentences(prepared)
    if not sentences:
        return ()

    parts = _composite_score_parts(sentences)
    scores = [0.60 * t + 0.25 * p + 0.15 * l for (t, p, l) in parts]

    # Group non-heading sentences under the most recent heading.
    sections: list[tuple[int, str, list[int]]] = []
    current: tuple[int, str, list[int]] | None = None
    for i, s in enumerate(sentences):
        if _is_structural_heading(s):
            if current is not None:
                sections.append(current)
            name = heading_name(s) or ""
            current = (_md_depth(s), name, [])
        else:
            if current is not None:
                current[2].append(i)
    if current is not None:
        sections.append(current)

    out: list[Section] = []
    for depth, name, body in sections:
        if not body or not name:
            continue
        best_idx = max(body, key=lambda i: scores[i])
        out.append(Section(
            depth=depth,
            name=name,
            representative_sentence=sentences[best_idx],
        ))
    return tuple(out)


def toc(text: str) -> tuple[str, ...]:
    """Return section names in document order. Lightweight TOC.

    Equivalent to ``tuple(s.name for s in outline(text))`` but exposed as
    its own primitive for discoverability. Uses the regex heading detector;
    no backend kwarg because the only backend is regex (parity with
    ``outline()``).
    """
    return tuple(s.name for s in outline(text))


# Register 'regex' backend entry for registry symmetry with metadata/phrases/
# correlate_facts. The public `toc()` function does not accept a backend kwarg
# (regex is the only option, same as outline); registration keeps the registry
# complete so any introspection shows toc.
from ._backends import register as _register  # noqa: E402

_register("regex", "toc", toc)
