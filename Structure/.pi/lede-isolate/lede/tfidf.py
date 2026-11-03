"""TF-IDF + position + length extractive summarization pipeline.

Composite score:
  score = 0.60 * tfidf + 0.25 * position + 0.15 * length

All scores normalized to [0, 1] per dimension. The composite is a weighted
sum, also in [0, 1].

v0.2.0 adds mode='default' with four scorer tweaks on top of the legacy
formula: heading filter, cue-phrase boost, digit bonus, and section-position
weighting. mode='legacy' preserves v0.0.1 behavior byte-identically. See
docs/v0-2-design.md for the design contract.
"""
import math
import re
from collections import Counter
from collections.abc import Sequence

from lede.sentences import split_sentences
from lede._headings import is_heading, heading_name
from lede._types import SummaryResult
from lede._pins import render_with_pins, prepend_blocks
from lede._hints import (
    preprocess_hints,
    hint_bonus,
    select_by_chars,
    round_to_int,
)

_TFIDF_WEIGHT = 0.60
_POSITION_WEIGHT = 0.25
_LENGTH_WEIGHT = 0.15

# Basic stopword list — deliberately small and stable across languages.
# Shared with the cross-language fixture corpus; do not add locale-specific terms.
_STOPWORDS = frozenset({
    "the", "and", "that", "this", "with", "for", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "what", "all", "when",
    "who", "will", "can", "from", "they", "each", "which", "their",
    "there", "about", "would", "make", "more", "some", "into",
    "other", "than", "its", "also", "after", "use", "how", "our",
    "any", "these", "most", "may", "should", "could", "does", "did",
    "just", "because", "over", "such", "through", "very", "your",
    "a", "an", "is", "it", "in", "on", "of", "to", "be", "as", "at", "by",
})

_TOKEN_RE = re.compile(r"\b[a-z]{3,}\b")

# --- v0.2 default-mode scorer tweaks ---

_CUE_PHRASE_RE = re.compile(
    r"^\s*(held|resolution|in summary|conclusion|action item|"
    r"decision|finding|key takeaway|outcome|ruling):?\b",
    re.IGNORECASE,
)

_DIGIT_RE = re.compile(r"\d+")

_SECTION_BOOST_HEADINGS = {
    "discussion", "conclusion", "held", "resolution",
    "key findings", "summary", "decision",
}


def _tokenize(sentence: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(sentence.lower()) if t not in _STOPWORDS]


def _normalize(scores: list[float]) -> list[float]:
    hi = max(scores, default=0.0)
    if hi <= 0.0:
        return [0.0] * len(scores)
    return [s / hi for s in scores]


def tfidf_score(sentences: list[str]) -> list[float]:
    """TF-IDF score per sentence, normalized to [0, 1]."""
    tokenized = [_tokenize(s) for s in sentences]
    n = len(sentences)

    # Document frequency: how many sentences contain each term
    df: Counter[str] = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    # IDF — smoothed to avoid log(1) = 0 for universal terms
    idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

    raw: list[float] = []
    for tokens in tokenized:
        if not tokens:
            raw.append(0.0)
            continue
        tf = Counter(tokens)
        # Sum of tf-idf, divided by sentence length — average term importance
        score = sum(tf[term] * idf.get(term, 0.0) for term in tf) / len(tokens)
        raw.append(score)

    return _normalize(raw)


def position_score(n: int) -> list[float]:
    """Position score: first and last sentences score 1.0, middle scores lower.

    Uses a U-shape: score(i) = max(1 - i/n, i/n). For i=0 or i=n-1, this is 1.0.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    scores: list[float] = []
    for i in range(n):
        # Distance from the nearer endpoint, normalized
        d = min(i, n - 1 - i) / max(n - 1, 1)
        # d=0 at endpoints, d=0.5 at middle. Score = 1 - 2d gives 1 at ends, 0 at middle.
        scores.append(max(0.0, 1.0 - 2.0 * d))
    # Endpoints are already 1.0; normalize is a no-op but keeps shape consistent
    return _normalize(scores)


def length_score(sentences: list[str]) -> list[float]:
    """Length score: peaks in the 10-30 word range; penalizes shorter and longer."""
    raw: list[float] = []
    for s in sentences:
        words = len(s.split())
        if words == 0:
            raw.append(0.0)
        elif 10 <= words <= 30:
            raw.append(1.0)
        elif words < 10:
            raw.append(words / 10.0)
        else:  # words > 30
            # Linear decay to 0 by word count 80
            raw.append(max(0.0, 1.0 - (words - 30) / 50.0))
    return _normalize(raw)


def _composite_score_parts(
    sentences: list[str],
) -> list[tuple[float, float, float]]:
    """Return (tfidf, position, length) triples per sentence."""
    if not sentences:
        return []
    t = tfidf_score(sentences)
    p = position_score(len(sentences))
    l = length_score(sentences)
    return [(t[i], p[i], l[i]) for i in range(len(sentences))]


def _composite_score_legacy(sentences: list[str]) -> list[float]:
    """Legacy composite: 0.60 * tfidf + 0.25 * position + 0.15 * length.

    Byte-identical to v0.0.1 behavior.
    """
    parts = _composite_score_parts(sentences)
    return [
        _TFIDF_WEIGHT * t + _POSITION_WEIGHT * p + _LENGTH_WEIGHT * l_
        for (t, p, l_) in parts
    ]


def composite_score(sentences: list[str]) -> list[float]:
    """Composite score: 0.60 * tfidf + 0.25 * position + 0.15 * length.

    Preserved for backward compatibility with callers and tests that import
    this helper directly. Equivalent to _composite_score_legacy.
    """
    return _composite_score_legacy(sentences)


def _separate_heading_lines(text: str) -> str:
    """Ensure heading-looking lines are separated from their following line by
    a blank line, so the sentence splitter treats them as standalone sentences.

    Only touches lines that the heading detector would classify as headings
    when considered in isolation. Idempotent: already-separated headings pass
    through unchanged.
    """
    if not text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        if i == len(lines) - 1:
            continue
        # If current line is a heading candidate and the next line is non-empty,
        # insert a blank line so paragraph-break splitting kicks in.
        if line.strip() and is_heading(line):
            nxt = lines[i + 1]
            if nxt.strip():
                out.append("")
    return "\n".join(out)


def _build_section_map(sentences: list[str]) -> dict[int, str]:
    """Map each sentence index to the lowercase name of the most recent
    heading above it (or "" if none)."""
    section_map: dict[int, str] = {}
    current = ""
    for i, sentence in enumerate(sentences):
        if is_heading(sentence):
            name = heading_name(sentence)
            current = name.lower() if name else ""
        section_map[i] = current
    return section_map


def _composite_score_default(
    sentences: list[str],
    section_map: dict[int, str],
) -> list[float]:
    """v0.2 default scorer: legacy composite + four tweaks.

    - Heading sentences get -inf (dropped from selection).
    - Sentences under a high-signal section (discussion/conclusion/etc.)
      get tfidf *= 1.3.
    - Cue-phrase sentences ("Held:", "Resolution:", ...) get +2.0.
    - Sentences containing digits get +0.3.
    """
    parts = _composite_score_parts(sentences)
    scores: list[float] = []
    for i, sentence in enumerate(sentences):
        if is_heading(sentence):
            scores.append(float("-inf"))
            continue
        t, p, l_ = parts[i]
        if section_map.get(i, "") in _SECTION_BOOST_HEADINGS:
            t *= 1.3
        score = _TFIDF_WEIGHT * t + _POSITION_WEIGHT * p + _LENGTH_WEIGHT * l_
        if _CUE_PHRASE_RE.match(sentence):
            score += 2.0
        if _DIGIT_RE.search(sentence):
            score += 0.3
        scores.append(score)
    return scores


# --- Top-level summarize pipeline ---

_MIN_SENTENCES = 3
_MIN_BUDGET_FOR_SENTENCES = 50  # chars; below this, truncate


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    # Reserve 3 chars for the ellipsis
    body_budget = max(0, max_length - 3)
    return text[:body_budget] + "..."


def _summarize_legacy(text: str, max_length: int = 500) -> str:
    """Extractive summary — v0.0.1 byte-identical behavior.

    Pipeline:
      1. If input fits the budget, return unchanged.
      2. If the budget is too small for sentences, truncate.
      3. Split into sentences; if fewer than 3, truncate.
      4. Score sentences (TF-IDF + position + length, 60/25/15).
      5. Greedily add highest-scoring sentences until the char budget is spent.
      6. Reorder selected sentences by original position.
      7. Fallback to truncation if selection is empty.
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    if max_length < _MIN_BUDGET_FOR_SENTENCES:
        return _truncate(text, max_length)

    sentences = split_sentences(text)
    if len(sentences) < _MIN_SENTENCES:
        return _truncate(text, max_length)

    scores = _composite_score_legacy(sentences)
    # Indices sorted by score descending, then by original position ascending
    # (stable tie-break — deterministic).
    indices_by_score = sorted(
        range(len(sentences)),
        key=lambda i: (-scores[i], i),
    )

    selected: list[int] = []
    used = 0
    separator = " "
    for idx in indices_by_score:
        sentence = sentences[idx]
        needed = len(sentence) + (len(separator) if selected else 0)
        if used + needed <= max_length:
            selected.append(idx)
            used += needed

    if not selected:
        return _truncate(text, max_length)

    selected.sort()
    return separator.join(sentences[i] for i in selected)


def _select_with_hints(
    text: str,
    max_length: int,
    *,
    mode: str,
    hints: list[tuple[str, float]],
    hint_focus: float,
    hint_mode: str,
) -> tuple[list[str], list[int]] | None:
    """Two-pool hint selection returning (sentences, selected_sorted), or
    None for early-return / empty-selection paths."""
    if not text:
        return None
    if max_length < _MIN_BUDGET_FOR_SENTENCES:
        return None

    prepared = _separate_heading_lines(text)
    sentences = split_sentences(prepared)
    if len(sentences) < _MIN_SENTENCES:
        return None

    section_map = _build_section_map(sentences)
    if mode == "coverage":
        from lede.coverage import _composite_score_coverage_for_hints
        plain_scores = _composite_score_coverage_for_hints(sentences)
    else:
        plain_scores = _composite_score_default(sentences, section_map)

    # Hint-pool scores. Soft = plain + bonus. Hard = -inf when no match.
    bonuses = [hint_bonus(s, hints) for s in sentences]
    if hint_mode == "hard":
        hint_scores = [
            (plain_scores[i] + bonuses[i]) if bonuses[i] > 0 else float("-inf")
            for i in range(len(sentences))
        ]
    else:  # soft
        hint_scores = [plain_scores[i] + bonuses[i] for i in range(len(sentences))]

    lengths = [len(s) for s in sentences]
    hint_budget = round_to_int(max_length, hint_focus)
    normal_budget = max_length - hint_budget

    selected_hint = select_by_chars(
        hint_scores, lengths, budget=hint_budget, exclude=set()
    )
    selected_normal = select_by_chars(
        plain_scores, lengths, budget=normal_budget, exclude=selected_hint
    )

    # Rollover: unused hint budget → plain pool.
    # In hard mode, rollover still respects the hard filter (uses hint_scores so
    # -inf sentences stay ineligible). In soft mode, plain_scores allow any sentence.
    used_hint = sum(lengths[i] for i in selected_hint) + max(0, len(selected_hint) - 1)
    unused_hint = hint_budget - used_hint
    if unused_hint > 0:
        rollover_scores = hint_scores if hint_mode == "hard" else plain_scores
        extra = select_by_chars(
            rollover_scores,
            lengths,
            budget=unused_hint,
            exclude=selected_hint | selected_normal,
        )
        selected_normal |= extra

    # Soft-only rollover: unused normal → hint pool.
    used_normal = sum(lengths[i] for i in selected_normal) + max(0, len(selected_normal) - 1)
    unused_normal = normal_budget - used_normal
    if unused_normal > 0 and hint_mode == "soft":
        extra = select_by_chars(
            hint_scores,
            lengths,
            budget=unused_normal,
            exclude=selected_hint | selected_normal,
        )
        selected_hint |= extra

    selected = sorted(selected_hint | selected_normal)
    if not selected:
        return None
    return sentences, selected


def _summarize_with_hints(
    text: str,
    max_length: int,
    *,
    mode: str,
    hints: list[tuple[str, float]],
    hint_focus: float,
    hint_mode: str,
) -> str:
    """Two-pool selection: hint pool (bonus-augmented or hard-filtered) + plain pool."""
    if not text:
        return ""
    if max_length < _MIN_BUDGET_FOR_SENTENCES:
        return _truncate(text, max_length)
    sel = _select_with_hints(
        text, max_length, mode=mode, hints=hints,
        hint_focus=hint_focus, hint_mode=hint_mode,
    )
    if sel is None:
        prepared = _separate_heading_lines(text)
        sentences = split_sentences(prepared)
        if len(sentences) < _MIN_SENTENCES and len(text) <= max_length:
            return text
        return _truncate(text, max_length)
    sentences, selected = sel
    return " ".join(sentences[i] for i in selected)


def _select_default(
    text: str, max_length: int
) -> tuple[list[str], list[int]] | None:
    """Return (sentences, selected_indices_sorted) for default-mode
    selection, or None when an early-return path applies (empty input,
    sub-sentence budget, or fewer than _MIN_SENTENCES sentences, or an
    empty selection)."""
    if not text:
        return None
    if max_length < _MIN_BUDGET_FOR_SENTENCES:
        return None
    prepared = _separate_heading_lines(text)
    sentences = split_sentences(prepared)
    if len(sentences) < _MIN_SENTENCES:
        return None
    section_map = _build_section_map(sentences)
    scores = _composite_score_default(sentences, section_map)
    indices_by_score = sorted(
        range(len(sentences)),
        key=lambda i: (-scores[i], i),
    )
    selected: list[int] = []
    used = 0
    separator = " "
    for idx in indices_by_score:
        if scores[idx] == float("-inf"):
            continue
        sentence = sentences[idx]
        needed = len(sentence) + (len(separator) if selected else 0)
        if used + needed <= max_length:
            selected.append(idx)
            used += needed
    if not selected:
        return None
    selected.sort()
    return sentences, selected


def _select_for_mode(
    text: str,
    max_length: int,
    mode: str,
    processed_hints,
    hint_focus: float,
    hint_mode: str,
) -> tuple[list[str], list[int]] | None:
    """Dispatch to the mode-appropriate index-returning selector:
    hint two-pool when hints are present, coverage when mode is
    'coverage', else the default scorer. Returns None on early-return
    paths, making keep_headings a safe no-op there."""
    if processed_hints:
        return _select_with_hints(
            text, max_length, mode=mode, hints=processed_hints,
            hint_focus=hint_focus, hint_mode=hint_mode,
        )
    if mode == "coverage":
        from lede.coverage import select_coverage_indices
        return select_coverage_indices(text, max_length)
    return _select_default(text, max_length)


def _summarize_default(text: str, max_length: int = 500) -> str:
    """v0.2 default scorer — mirrors _summarize_legacy with the default scorer.

    Headings are dropped from candidates (score = -inf), cue-phrase sentences
    are boosted by +2.0, digit-bearing sentences get +0.3, and sentences
    under high-signal section headings have tfidf * 1.3.
    """
    if not text:
        return ""
    if max_length < _MIN_BUDGET_FOR_SENTENCES:
        return _truncate(text, max_length)
    sel = _select_default(text, max_length)
    if sel is None:
        # Reproduce the original early-return semantics exactly: a short
        # doc (fewer than _MIN_SENTENCES sentences) returns as-is when it
        # fits the budget, else truncates; an empty selection truncates.
        prepared = _separate_heading_lines(text)
        sentences = split_sentences(prepared)
        if len(sentences) < _MIN_SENTENCES and len(text) <= max_length:
            return text
        return _truncate(text, max_length)
    sentences, selected = sel
    return " ".join(sentences[i] for i in selected)


_ATTACH_KEYS = {"stats", "outline", "metadata", "phrases", "correlated_facts"}


def summarize(
    text: str,
    max_length: int = 500,
    *,
    mode: str = "default",
    attach: list[str] | tuple[str, ...] | None = None,
    hints: list[str] | dict[str, float] | None = None,
    hint_focus: float = 0.7,
    hint_mode: str = "soft",
    keep_headings: bool = False,
    include_toc: bool = False,
    pin: Sequence[str] | None = None,
    headings: Sequence[str] | None = None,
) -> SummaryResult:
    """Extractive summary with configurable scoring mode and optional hints.

    Args:
        text: input document text.
        max_length: character budget for the extractive body. Default 500.
            Pinned content (``pin``, TOC, injected headings) is added on
            top of this budget, so total output may exceed ``max_length``.
        mode: scoring mode — ``"default"`` (TF-IDF + position + length,
            heading-filtered), ``"legacy"`` (pre-v0.2 byte-identical scorer),
            or ``"coverage"`` (paragraph-aware selection). Default
            ``"default"``.
        attach: list of extra primitives to attach to the returned
            ``SummaryResult``. Supported names: ``"stats"``, ``"outline"``,
            ``"metadata"``, ``"phrases"``, ``"correlated_facts"``.
        hints: optional list[str] or dict[str, float] of terms to bias
            selection toward. List entries get weight 1.0; dict keys are
            terms and values are numeric weights. Default None (no biasing).
        hint_focus: fraction of the character budget reserved for
            hint-matching sentences. Range [0.0, 1.0]. Default 0.7 — 70 %
            of budget goes to hint-matching sentences first, remaining 30 %
            to the highest-scored non-hint sentences. Unused hint-pool budget
            rolls over to the plain pool (and vice versa in soft mode).
            Ignored when ``hints`` is None.
        hint_mode: biasing strategy — ``"soft"`` (default) or ``"hard"``.
            ``"soft"`` adds a bonus to hint-matching sentences and reorders
            without filtering; non-matching sentences can still appear.
            ``"hard"`` restricts the hint pool to only sentences that match
            at least one hint; unmatched sentences can still appear in the
            plain-pool quota. Ignored when ``hints`` is None.
        keep_headings: when ``True``, pins the document title (a depth-1
            markdown heading at the document start, if present) and the
            nearest enclosing section heading above each selected sentence.
            Headings are deduplicated and interleaved into the body in
            document order. Supported in ``"default"`` and ``"coverage"``
            modes; composes with ``hints``. Default ``False``.
        include_toc: when ``True``, prepends a full outline / table-of-
            contents block (from ``lede.extract.outline``) before the body.
            Default ``False``.
        pin: sequence of caller-supplied lines forced verbatim into the
            output. Lines are prepended as a block in the given order,
            before any TOC or body content. Default ``None``.
        headings: caller-supplied heading lines (e.g. from chunk metadata).
            When non-empty, replaces auto-detection for ``keep_headings``
            and ``include_toc``: the heading detector is not run, and these
            lines are used verbatim instead. Has no effect when both
            ``keep_headings`` and ``include_toc`` are ``False``. Rejected
            in ``mode="legacy"`` (raises ``ValueError``). Default ``None``
            (auto-detection, v0.4.2 behavior).

    Returns:
        ``SummaryResult`` — a frozen dataclass with ``.summary: str`` plus
        optional fields populated by ``attach``.  The ``.pinned_headings``
        field lists the headings auto-detected and injected by
        ``keep_headings`` (empty tuple otherwise; ``pin`` lines and TOC
        entries are not listed there).  ``str(r)`` and ``f"{r}"`` evaluate
        to ``.summary`` so legacy string callers still work.

    Raises:
        ValueError: on unknown ``mode``, on ``mode="legacy"`` with hints
            (not supported), on ``hint_focus`` outside [0.0, 1.0], on
            unknown ``hint_mode``, or on ``mode="legacy"`` with any of
            ``keep_headings=True``, ``include_toc=True``, or ``pin``
            supplied (not supported in legacy mode).

    Hint biasing (v0.4):
        When ``hints`` is None (the default), no new code path executes and
        output is byte-identical to v0.3.0. Matching is case-insensitive
        and word-boundary-delimited (``\\b``); no Unicode normalization or
        stemming. For lemma/synonym expansion, compose with
        ``lede_spacy.expand_hints`` before passing hints here::

            from lede import summarize
            from lede_spacy import expand_hints

            expanded = expand_hints(["counties"], kinds=("lemma",))
            # expanded == ["counties", "county"]
            result = summarize(text, hints=expanded, hint_focus=0.7).summary

        Simple usage without expansion::

            from lede import summarize
            result = summarize(text, hints=["John Smith", "county"], hint_focus=0.7).summary

        See docs/REFERENCE.md "Hint biasing" for the full contract.

    Heading & pin retention (v0.4.2):
        All three kwargs default to off — when none are set, output is
        byte-identical to v0.4.1. When any are active, block order in the
        output is (top to bottom):

        1. ``pin`` block — caller-supplied lines, verbatim, in given order.
        2. TOC block — full outline when ``include_toc=True``.
        3. Body — extractive sentences, with document title and the nearest
           enclosing section heading above each sentence interleaved when
           ``keep_headings=True``.

        Pinned content is added **on top of** ``max_length``. The budget
        governs only the extractive body, so pins always survive even if
        ``max_length`` is tight. ``mode="legacy"`` rejects all three kwargs
        (raises ``ValueError``).

        The ``SummaryResult.pinned_headings`` field is a ``tuple[str, ...]``
        listing the auto-detected headings injected by ``keep_headings``
        (empty when ``keep_headings=False`` or no headings were found).
        ``pin`` lines and TOC entries are not recorded there::

            from lede import summarize

            r = summarize(text, max_length=500, keep_headings=True, pin=["Figure 3: Q3 revenue"])
            r.summary          # pinned lines + headings woven into the body
            r.pinned_headings  # the auto-detected headings that were injected

        Headings are detected via lede's structural heading detector (the
        same one ``extract.outline`` uses): markdown ``#`` lines and
        title-case bare lines. The short-sentence heuristic is not used.
        See docs/REFERENCE.md "Heading & pin retention" for the full contract.

        Caller-supplied headings override (v0.4.3):
            When ``headings`` is non-empty, it **replaces** auto-detection
            entirely. TOC block = the supplied headings, deduped (first
            occurrence wins), one per line, flat (depth unknown). Body
            placement: the first supplied heading is pinned as the document
            title at the top; unmatched headings (no body line equals the
            heading, stripped) are emitted as a leading block so they still
            survive; matched headings interleave before their section's
            selected sentences. ``SummaryResult.pinned_headings`` lists all
            injected headings. Use this path when the heading text lives in
            chunk metadata or when auto-detection misses domain-specific
            heading styles (e.g. SCOTUS opinion captions, caption-style
            chapter labels)::

                from lede import summarize

                r = summarize(
                    text,
                    max_length=500,
                    keep_headings=True,
                    headings=["OPINION OF THE COURT", "HELD"],
                )
                r.pinned_headings  # ("OPINION OF THE COURT", "HELD")
    """
    # Validate hint kwargs first (before any work).
    processed_hints = preprocess_hints(hints)
    pins_active = bool(keep_headings) or bool(include_toc) or bool(pin) or bool(headings)
    if pins_active and mode == "legacy":
        raise ValueError(
            "keep_headings/include_toc/pin/headings not supported in legacy mode"
        )
    if processed_hints:
        if mode not in ("default", "coverage", "legacy"):
            raise ValueError(f"unknown mode: {mode!r}")
        if not 0.0 <= hint_focus <= 1.0:
            raise ValueError(
                f"hint_focus must be in [0.0, 1.0]; got {hint_focus}"
            )
        if hint_mode not in ("soft", "hard"):
            raise ValueError(
                f"hint_mode must be 'soft' or 'hard'; got {hint_mode!r}"
            )
        if mode == "legacy":
            raise ValueError("hints not supported in legacy mode")
        summary_text = _summarize_with_hints(
            text,
            max_length=max_length,
            mode=mode,
            hints=processed_hints,
            hint_focus=hint_focus,
            hint_mode=hint_mode,
        )
    elif mode == "coverage":
        from lede.coverage import summarize_coverage
        summary_text = summarize_coverage(text, max_length=max_length)
    elif mode == "legacy":
        summary_text = _summarize_legacy(text, max_length=max_length)
    elif mode == "default":
        summary_text = _summarize_default(text, max_length=max_length)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    pinned_headings: tuple[str, ...] = ()
    if pins_active:
        sel = (
            _select_for_mode(
                text, max_length, mode, processed_hints, hint_focus, hint_mode
            )
            if keep_headings
            else None
        )
        if sel is not None:
            sentences, selected = sel
            summary_text, pinned_headings = render_with_pins(
                sentences,
                selected,
                keep_headings=True,
                include_toc=include_toc,
                pin=pin,
                text=text,
                headings=headings,
            )
        elif include_toc or pin:
            summary_text = prepend_blocks(
                summary_text, include_toc=include_toc, pin=pin, text=text, headings=headings,
            )
        # else: keep_headings requested but no selection available (early-return
        # / no-op) — summary_text already equals the plain body; nothing to weave.

    kwargs: dict = {}
    if attach:
        unknown = set(attach) - _ATTACH_KEYS
        if unknown:
            raise ValueError(f"unknown attach key(s): {sorted(unknown)}")
        from lede.extract import stats as _stats
        from lede.extract import outline as _outline
        from lede.extract import metadata as _metadata
        from lede.extract import phrases as _phrases
        from lede.extract import correlate_facts as _correlate
        if "stats" in attach:
            kwargs["stats"] = tuple(_stats(text))
        if "outline" in attach:
            kwargs["outline"] = tuple(_outline(text))
        if "metadata" in attach:
            kwargs["metadata"] = _metadata(text)
        if "phrases" in attach:
            kwargs["phrases"] = tuple(_phrases(text))
        if "correlated_facts" in attach:
            kwargs["correlated_facts"] = tuple(_correlate(text))
    return SummaryResult(summary=summary_text, pinned_headings=pinned_headings, **kwargs)
