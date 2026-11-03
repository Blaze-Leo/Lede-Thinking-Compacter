"""Coverage-constrained selector. Activated via summarize(mode='coverage').

Picks the highest-scoring sentence from each paragraph (paragraphs are
blank-line-delimited; fewer-than-20-char paragraphs are skipped). After a
one-per-paragraph pass, the next-best ungrabbed sentences from paragraphs
not yet represented fill remaining budget. Reorder by original document
position.

Coverage mode uses the raw (tfidf, position, length) composite
(0.60/0.25/0.15) WITHOUT the default-mode cue/digit/section boosts —
coverage is about breadth (how many paragraphs the summary touches), not
per-sentence signal density.
"""
import re

from lede._headings import is_heading
from lede.sentences import split_sentences
from lede.tfidf import _composite_score_parts


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def _composite_score_coverage_for_hints(sentences: list[str]) -> list[float]:
    """Per-sentence coverage-mode scores for the hint two-pool path.

    Returns the raw 60/25/15 composite (tfidf, position, length) per
    sentence — the same base scores `summarize_coverage` computes
    internally. Headings get `-inf` so the two-pool selector skips them.

    Note: coverage mode's per-paragraph breadth guarantee is intentionally
    NOT preserved when hints are used. The caller asked for hint-biased
    selection; breadth is a secondary concern.
    """
    if not sentences:
        return []
    parts = _composite_score_parts(sentences)
    return [
        float("-inf") if is_heading(s) else (0.60 * t + 0.25 * p + 0.15 * l)
        for (t, p, l), s in zip(parts, sentences)
    ]


def _split_paragraphs(text: str) -> list[str]:
    """Return paragraphs of at least 20 chars after strip."""
    return [p.strip() for p in _PARA_SPLIT_RE.split(text) if len(p.strip()) >= 20]


def select_coverage_indices(
    text: str, max_length: int = 500
) -> "tuple[list[str], list[int]] | None":
    """Return (sentences, selected_indices_sorted) for coverage-mode selection.

    Returns None when an early-return path applies (empty sentences list).
    The returned ``selected`` list may be empty (no sentence fit the budget).
    """
    sentences = split_sentences(text)
    if not sentences:
        return None
    parts = _composite_score_parts(sentences)
    scores = [0.60 * t + 0.25 * p + 0.15 * l for (t, p, l) in parts]

    # Map each sentence to its source paragraph by occurrence-counting:
    # the K-th occurrence of a repeated sentence goes to the K-th paragraph
    # that contains it. The previous "first containing paragraph wins" rule
    # biased coverage selection on documents with template/FAQ-style
    # repeated sentences. Mirrors `rust/src/coverage.rs`. Closes AAT-021.
    paragraphs = _split_paragraphs(text)
    sent_to_para: list[int] = []
    occurrence_seen: dict[str, int] = {}
    for s in sentences:
        target = occurrence_seen.get(s, 0)
        idx = -1
        seen = 0
        for i, p in enumerate(paragraphs):
            if s in p:
                if seen == target:
                    idx = i
                    break
                seen += 1
        sent_to_para.append(idx)
        occurrence_seen[s] = target + 1

    # Best non-heading sentence per paragraph.
    best_per_para: dict[int, tuple[float, int]] = {}
    for i, sentence in enumerate(sentences):
        if is_heading(sentence):
            continue
        para_idx = sent_to_para[i]
        if para_idx < 0:
            continue
        if para_idx not in best_per_para or scores[i] > best_per_para[para_idx][0]:
            best_per_para[para_idx] = (scores[i], i)

    # Per-paragraph pass — iterate paragraphs in document order, add the
    # paragraph's best sentence if budget allows.
    selected: list[int] = []
    covered_paras: set[int] = set()
    budget_used = 0
    for para_idx in sorted(best_per_para):
        _, sent_idx = best_per_para[para_idx]
        cost = len(sentences[sent_idx]) + 1
        if budget_used + cost <= max_length:
            selected.append(sent_idx)
            covered_paras.add(para_idx)
            budget_used += cost

    # Greedy fill — only add sentences from paragraphs NOT yet represented,
    # to preserve the "breadth" guarantee of coverage mode.
    remaining: list[tuple[float, int]] = []
    for i, sentence in enumerate(sentences):
        if i in selected:
            continue
        if is_heading(sentence):
            continue
        para_idx = sent_to_para[i]
        if para_idx < 0 or para_idx in covered_paras:
            continue
        remaining.append((scores[i], i))
    remaining.sort(key=lambda x: -x[0])

    for _, i in remaining:
        cost = len(sentences[i]) + 1
        if budget_used + cost <= max_length:
            selected.append(i)
            covered_paras.add(sent_to_para[i])
            budget_used += cost
        if budget_used >= max_length:
            break

    selected.sort()
    return sentences, selected


def summarize_coverage(text: str, max_length: int = 500) -> str:
    """C2 coverage selector — paragraph-aware extractive summary."""
    sel = select_coverage_indices(text, max_length)
    if sel is None:
        return text
    sentences, selected = sel
    # Use " " to match default and legacy modes' separator. Earlier coverage
    # used "\n", which surprised callers diffing across modes. AAT-022.
    return " ".join(sentences[i] for i in selected)
