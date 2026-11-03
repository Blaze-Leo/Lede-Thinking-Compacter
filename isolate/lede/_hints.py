"""Internal: hint preprocessing, matching, scoring, two-pool selection.

All names are package-private (underscore module). Public surface is the
hint kwargs on lede.summarize / brief / extract.* primitives.

Must remain byte-identical to rust/src/hints.rs.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

_WHITESPACE_RE = re.compile(r"\s+")

# Cache compiled per-hint regexes. Bounded — most processes use a small
# stable set of hints. If callers churn through thousands of distinct
# hint strings, this cache grows; that's their bandwidth call.
_REGEX_CACHE: dict[str, re.Pattern] = {}

_HINT_BASE_WEIGHT = 0.5
_HINT_MATCH_CAP = 3


def preprocess_hints(
    hints: list[str] | dict[str, float] | None,
) -> list[tuple[str, float]]:
    """Return list of (lowercased_hint, weight) tuples.

    - Strips, collapses internal whitespace, lowercases.
    - Drops empty/whitespace-only entries.
    - List input → weight 1.0 per hint.
    - Dict input → uses values as weights.
    - None or empty → empty list.
    """
    if not hints:
        return []
    out: list[tuple[str, float]] = []
    if isinstance(hints, dict):
        items: Iterable[tuple[str, float]] = hints.items()
    else:
        items = ((h, 1.0) for h in hints)
    for raw, weight in items:
        if not isinstance(raw, str):
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        collapsed = _WHITESPACE_RE.sub(" ", stripped)
        out.append((collapsed.lower(), float(weight)))
    return out


def _compile_hint(hint: str) -> re.Pattern:
    """Compile and cache the word-boundary regex for a preprocessed hint."""
    cached = _REGEX_CACHE.get(hint)
    if cached is not None:
        return cached
    pattern = re.compile(r"\b" + re.escape(hint) + r"\b")
    _REGEX_CACHE[hint] = pattern
    return pattern


def match_count(hint: str, sentence: str) -> int:
    """Non-overlapping match count for one preprocessed hint vs. sentence.

    Caller is responsible for preprocessing the hint via `preprocess_hints`.
    The sentence is lowercased here; preprocessing the hint there.
    """
    if not hint:
        return 0
    return len(_compile_hint(hint).findall(sentence.lower()))


def round_to_int(value: int, focus: float) -> int:
    """Portable integer-math rounding for budget splits.

    Avoids Python `round` (banker's) vs Rust `f64::round` (half-away)
    divergence. Encodes focus as numerator/10000, then does integer
    arithmetic. Identical results in Rust via the mirror in
    rust/src/hints.rs.
    """
    num = int(focus * 10000)
    den = 10000
    return (value * num + den // 2) // den


def hint_bonus(sentence: str, hints: list[tuple[str, float]]) -> float:
    """Sum of per-hint bonuses for one sentence.

    For each (hint, weight): min(count, _HINT_MATCH_CAP) * weight * _HINT_BASE_WEIGHT
    """
    total = 0.0
    for hint, weight in hints:
        count = match_count(hint, sentence)
        if count == 0:
            continue
        capped = count if count < _HINT_MATCH_CAP else _HINT_MATCH_CAP
        total += capped * weight * _HINT_BASE_WEIGHT
    return total


def _rank_indices(scores: list[float]) -> list[int]:
    """Sort indices by (-score, original_index). Deterministic."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def select_by_chars(
    scores: list[float],
    lengths: list[int],
    *,
    budget: int,
    exclude: set[int],
    separator_len: int = 1,
) -> set[int]:
    """Greedy char-budget selector.

    Picks indices in score-descending order (position tiebreak) until
    the budget can't fit another sentence. Skips `exclude`. Treats
    -inf scores as ineligible.
    """
    if budget <= 0:
        return set()
    chosen: set[int] = set()
    used = 0
    for idx in _rank_indices(scores):
        if idx in exclude:
            continue
        if scores[idx] == -math.inf:
            continue
        needed = lengths[idx] + (separator_len if chosen else 0)
        if used + needed <= budget:
            chosen.add(idx)
            used += needed
    return chosen


def select_by_count(
    scores: list[float],
    *,
    count: int,
    exclude: set[int],
) -> set[int]:
    """Top-N count-budget selector for primitives like key_facts/phrases.

    Same ranking as `select_by_chars`. Stops after `count` selections.
    Treats -inf as ineligible.
    """
    if count <= 0:
        return set()
    chosen: set[int] = set()
    for idx in _rank_indices(scores):
        if idx in exclude:
            continue
        if scores[idx] == -math.inf:
            continue
        chosen.add(idx)
        if len(chosen) >= count:
            break
    return chosen
