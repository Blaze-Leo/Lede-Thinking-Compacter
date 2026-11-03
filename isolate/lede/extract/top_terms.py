"""extract.top_terms — top-N salient words and phrases (v0.4).

Combines per-document TF-IDF on single tokens with multi-word phrase
frequency into a unified ranking. Each kind contributes candidates
normalized to a [0, 1] score; the top-N across all candidates is
returned in score-descending order.

No new dependencies beyond the existing TF-IDF and regex phrase machinery.
Python-only for v0.4; Rust mirror deferred to v0.5.
"""
from __future__ import annotations

import math
from collections import Counter

from ..sentences import split_sentences
from ..tfidf import _STOPWORDS, _TOKEN_RE
from .._hints import preprocess_hints, hint_bonus
from .._types import TermScore
from .phrases import _regex_phrases, _runs


_VALID_KINDS = ("words", "phrases")


def _word_scores(text: str) -> dict[str, float]:
    """Per-unique-word TF-IDF scores normalized to [0, 1].

    TF = total occurrences in document.
    IDF = log((num_sentences + 1) / (df + 1)) + 1.0 (matches summarize's IDF).
    Score = TF * IDF, then divided by max for normalization.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {}

    # Per-sentence tokens (lowercase, 3+ letters, no stopwords).
    per_sent_tokens: list[list[str]] = []
    for s in sentences:
        toks = [
            t for t in _TOKEN_RE.findall(s.lower())
            if t not in _STOPWORDS
        ]
        per_sent_tokens.append(toks)

    n = len(sentences)
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for toks in per_sent_tokens:
        tf.update(toks)
        for term in set(toks):
            df[term] += 1

    if not tf:
        return {}

    idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    raw = {term: tf[term] * idf[term] for term in tf}

    hi = max(raw.values())
    if hi <= 0:
        return {term: 0.0 for term in raw}
    return {term: score / hi for term, score in raw.items()}


def _phrase_scores(text: str) -> dict[str, float]:
    """Per-phrase scores from `_regex_phrases`, normalized to [0, 1].

    Score = repetition_count * token_count. Phrases with higher counts and
    longer phrases score higher.
    """
    phrases = _regex_phrases(text)
    if not phrases:
        return {}

    # Recompute raw counts using _runs (same tokenization as _regex_phrases).
    runs = _runs(text)
    counts: Counter[str] = Counter(runs)

    raw: dict[str, float] = {}
    for phrase in phrases:
        cnt = counts.get(phrase, 1)
        token_count = phrase.count(" ") + 1
        raw[phrase] = float(cnt * token_count)

    hi = max(raw.values())
    if hi <= 0:
        return {phrase: 0.0 for phrase in raw}
    return {phrase: score / hi for phrase, score in raw.items()}


def top_terms(
    text: str,
    *,
    n: int = 10,
    kinds: tuple[str, ...] = ("words", "phrases"),
    with_scores: bool = False,
    hints: list[str] | dict[str, float] | None = None,
    hint_focus: float = 0.7,
    hint_mode: str = "soft",
) -> tuple[str, ...] | tuple[TermScore, ...]:
    """Return the top-N salient terms (words and/or phrases) from text (v0.4).

    Composes single-word TF-IDF scores with multi-word phrase frequency into
    a unified ranking. Each kind contributes candidates normalized to a
    comparable [0, 1] score range; the top-N across all candidates is
    returned in score-descending order.

    **Word scoring** (``"words"`` kind): per-document TF-IDF, matching the
    same formula as ``summarize()`` (IDF = log((n_sents+1)/(df+1))+1.0,
    score = TF*IDF normalized by max). 3+ letter, non-stopword tokens only.

    **Phrase scoring** (``"phrases"`` kind): candidates from the same repeated
    n-gram extractor as ``extract.phrases()``; raw score = repetition_count ×
    token_count, normalized by max.

    When both kinds are requested (the default), phrase candidates can shadow
    word candidates if they occupy the same token surface, because the merged
    dict uses phrase-dict insertion order after word-dict.

    Args:
        text: input document text.
        n: cap on returned terms. Default 10.
        kinds: which candidate pools to use. Any non-empty subset of
            ``("words", "phrases")``. Default both. Order within the tuple
            does not affect output — scoring and ranking are independent of
            insertion order.
        with_scores: when True, return ``tuple[TermScore]`` instead of
            ``tuple[str]`` (v0.4.1). Each ``TermScore`` is a NamedTuple
            ``(term, score, kind)`` in the same unified ranked order, where
            ``score`` is the value that drove the ranking and ``kind`` is
            ``"word"`` or ``"phrase"``. Default False (returns bare strings,
            byte-for-byte the v0.4.0 behavior). See ``lede.extract.TermScore``
            for the score-calibration caveat (scores are per-kind-normalized,
            not cross-kind comparable).
        hints: optional list[str] or dict[str, float] of terms to bias
            selection toward. List entries get weight 1.0. Default None.
        hint_focus: accepted for API uniformity but has no behavioral effect
            on this primitive; ``top_terms()`` does not apply a two-pool
            budget split. Validated to be in [0.0, 1.0] when hints is not
            None. Default 0.7.
        hint_mode: ``"soft"`` (default) — adds ``hint_bonus`` to each
            candidate's score before ranking, so hint-matching terms rank
            higher. ``"hard"`` — only hint-matching candidates are kept
            before ranking; non-matching candidates are discarded.

    Returns:
        ``tuple[str, ...]`` of top-N salient terms in score-descending order
        (default), or ``tuple[TermScore, ...]`` when ``with_scores=True``.
        Empty tuple when ``text`` is empty or no candidates survive the
        ``kinds`` filters.

    Raises:
        ValueError: on unknown entry in ``kinds``, on ``hint_focus`` outside
            [0.0, 1.0], or on unknown ``hint_mode``.

    Hint biasing (v0.4):
        When ``hints`` is None (the default), output is determined solely by
        TF-IDF and phrase frequency. Matching is case-insensitive and
        word-boundary-delimited (``\\b``); no Unicode normalization or
        stemming. For lemma/synonym expansion compose with
        ``lede_spacy.expand_hints`` before calling this function.

        Note: ``hint_focus`` is validated but has no two-pool effect here
        (unlike ``summarize()`` or ``key_facts()``). In soft mode, hints
        raise candidate scores; in hard mode, non-matching candidates are
        removed. Python-only for v0.4; Rust mirror deferred to v0.5.

        See docs/REFERENCE.md "Hint biasing" for the full contract.
    """
    if not text:
        return ()

    for k in kinds:
        if k not in _VALID_KINDS:
            raise ValueError(
                f"unknown kind: {k!r}; expected one of {_VALID_KINDS}"
            )

    processed_hints = preprocess_hints(hints)
    if processed_hints:
        if not 0.0 <= hint_focus <= 1.0:
            raise ValueError(f"hint_focus must be in [0.0, 1.0]; got {hint_focus}")
        if hint_mode not in ("soft", "hard"):
            raise ValueError(f"hint_mode must be 'soft' or 'hard'; got {hint_mode!r}")

    # Gather candidates from each requested kind, tracking each term's kind.
    # Words are single tokens (no space); phrases are 2-5 token n-grams
    # (always contain a space) — so the keyspaces never collide.
    candidates: dict[str, float] = {}
    kind_of: dict[str, str] = {}
    if "words" in kinds:
        word_scores = _word_scores(text)
        candidates.update(word_scores)
        for term in word_scores:
            kind_of[term] = "word"
    if "phrases" in kinds:
        phrase_scores = _phrase_scores(text)
        candidates.update(phrase_scores)
        for term in phrase_scores:
            kind_of[term] = "phrase"

    if not candidates:
        return ()

    if processed_hints:
        if hint_mode == "hard":
            candidates = {
                term: score
                for term, score in candidates.items()
                if hint_bonus(term, processed_hints) > 0
            }
            if not candidates:
                return ()
        else:  # soft
            candidates = {
                term: score + hint_bonus(term, processed_hints)
                for term, score in candidates.items()
            }

    # Sort by (-score, term) for deterministic output.
    ranked = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
    top = ranked[:n]
    if with_scores:
        return tuple(
            TermScore(term=term, score=score, kind=kind_of[term])
            for term, score in top
        )
    return tuple(term for term, _ in top)
