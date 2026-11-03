"""Key-facts extractor — top-N fact-bearing sentences as complete strings.

Ranked by composite tfidf + stat density, deduped by normalized
(stat_type, value), returned in document order. Composition primitive:
reuses ``stats()`` and ``_composite_score_parts`` — no new regex patterns.

Mirrors rust/src/extract/key_facts.rs. Registered as ('regex','key_facts')
for registry symmetry with the other primitives. No spaCy backend (spaCy
composition path would be a different primitive, not a different backend).
"""
from __future__ import annotations

import re

from ..sentences import split_sentences
from ..tfidf import _composite_score_parts, _TFIDF_WEIGHT, _POSITION_WEIGHT, _LENGTH_WEIGHT
from .._hints import preprocess_hints, hint_bonus, round_to_int
from ._backends import register
from .stats import stats as _stats

# Normalization rule shared with benchmarks/extraction_eval.py::_norm — kept
# in sync by convention (same semantics, local copy to avoid bench dep).
_HYPHEN_RE = re.compile(r"[-_/]+")


def _norm_value(s: str) -> str:
    """Lowercase, hyphen/underscore/slash → space, collapse whitespace.

    Used for dedup key on stat values so "five-day" and "five day" are
    considered the same fact.
    """
    return re.sub(r"\s+", " ", _HYPHEN_RE.sub(" ", s.lower())).strip()


# Stat-density contribution weight in the final score. Kept small so the
# composite tfidf signal still dominates.
_STAT_DENSITY_WEIGHT = 0.15


def key_facts(
    text: str,
    *,
    max_facts: int = 10,
    convert_word_names: bool = False,
    hints: list[str] | dict[str, float] | None = None,
    hint_focus: float = 0.7,
    hint_mode: str = "soft",
) -> tuple[str, ...]:
    """Return top-N sentences containing numeric/named facts.

    Candidates are any sentence that produces at least one ``Stat`` from
    ``stats(text, convert_word_names=convert_word_names)``. Each candidate
    is scored as:

        0.60*tfidf + 0.25*position + 0.15*length
        + 0.15 * (num_stats_in_sentence / sentence_len_in_tokens)

    Candidates are deduped by the normalized ``(stat_type, value)`` key:
    if two sentences share any such key, only the higher-scored one is
    kept. Ties broken by earlier document position (deterministic).

    The top ``max_facts`` survivors are returned in **document order**
    (not score order) — readers want reading flow.

    Args:
        text: input document text.
        max_facts: cap on number of returned sentences. Default 10.
        convert_word_names: forwarded to ``stats()``; requires the
            ``[wordforms]`` extra when True.
        hints: optional list[str] or dict[str, float] of terms to bias
            selection toward. List entries get weight 1.0. Default None.
        hint_focus: fraction of ``max_facts`` quota reserved for
            hint-matching candidates. Range [0.0, 1.0]. Default 0.7.
            Ignored when ``hints`` is None.
        hint_mode: ``"soft"`` (default) or ``"hard"``. ``"soft"`` reorders
            hint-bearing candidates to fill the hint quota first; ``"hard"``
            restricts the hint pool to candidates that contain at least one
            hint. Ignored when ``hints`` is None.

    Returns:
        Tuple of complete sentence strings (not ``Stat`` tuples). Empty
        tuple when no sentence contains a recognized fact.

    Raises:
        ValueError: on ``hint_focus`` outside [0.0, 1.0] or unknown
            ``hint_mode``.

    Hint biasing (v0.4):
        When ``hints`` is None (the default), no new code path executes and
        output is byte-identical to v0.3.0. The two-pool selection uses
        ``max_facts`` as the count budget (not chars): ``hint_quota =
        round(max_facts * hint_focus)``, ``normal_quota = max_facts -
        hint_quota``. Deduplication by ``(stat_type, normalized_value)``
        runs across both pools combined — a high-scoring hint-bearing
        sentence can block a duplicate from appearing in the plain pool.

        Matching is case-insensitive and word-boundary-delimited (``\\b``);
        no Unicode normalization or stemming. See docs/REFERENCE.md "Hint
        biasing" for the full contract.
    """
    if not text:
        return ()

    processed_hints = preprocess_hints(hints)
    if processed_hints:
        if not 0.0 <= hint_focus <= 1.0:
            raise ValueError(f"hint_focus must be in [0.0, 1.0]; got {hint_focus}")
        if hint_mode not in ("soft", "hard"):
            raise ValueError(f"hint_mode must be 'soft' or 'hard'; got {hint_mode!r}")

    all_stats = _stats(text, convert_word_names=convert_word_names)
    if not all_stats:
        return ()

    sentences = split_sentences(text)
    if not sentences:
        return ()

    # Group stats by their context_sentence. We walk sentences in order and
    # match on string equality (context_sentence is a direct slice of the
    # input, same as the one split_sentences returns).
    stats_by_sentence: dict[int, list] = {}
    # Build a sentence-string → index lookup for O(1) grouping. When the same
    # sentence appears twice in a document the first occurrence wins — this
    # is deterministic and matches how stats() emits context_sentence.
    sent_to_idx: dict[str, int] = {}
    for i, s in enumerate(sentences):
        sent_to_idx.setdefault(s, i)
    for stat in all_stats:
        idx = sent_to_idx.get(stat.context_sentence)
        if idx is None:
            continue
        stats_by_sentence.setdefault(idx, []).append(stat)

    if not stats_by_sentence:
        return ()

    parts = _composite_score_parts(sentences)

    candidates: list[tuple[float, int, list]] = []
    for idx, stat_list in stats_by_sentence.items():
        t, p, l_ = parts[idx]
        composite = _TFIDF_WEIGHT * t + _POSITION_WEIGHT * p + _LENGTH_WEIGHT * l_
        # Sentence length in whitespace-separated tokens; mirrors how
        # length_score counts words. Guarded against zero for safety.
        tok_count = max(1, len(sentences[idx].split()))
        density = len(stat_list) / tok_count
        score = composite + _STAT_DENSITY_WEIGHT * density
        candidates.append((score, idx, stat_list))

    if processed_hints:
        # Two-pool hint-aware selection.
        # Compute per-candidate hint bonuses.
        bonus_for: list[float] = [
            hint_bonus(sentences[idx], processed_hints)
            for _score, idx, _stat_list in candidates
        ]

        # Build sorted orderings for each pool.
        # hint_keyed: rank by (score + bonus); hard mode sends bonus==0 to bottom.
        hint_keyed: list[tuple[float, int, int]] = []
        for i, ((score, idx, _stat_list), bonus) in enumerate(
            zip(candidates, bonus_for)
        ):
            if hint_mode == "hard" and bonus == 0:
                sort_key = float("inf")
            else:
                sort_key = -(score + bonus)
            hint_keyed.append((sort_key, idx, i))
        hint_keyed.sort()

        plain_keyed: list[tuple[float, int, int]] = sorted(
            [(-c[0], c[1], i) for i, c in enumerate(candidates)]
        )

        hint_quota = round_to_int(max_facts, hint_focus)
        normal_quota = max_facts - hint_quota

        kept_keys: set[tuple[str, str]] = set()
        kept: list[tuple[float, int]] = []
        selected_cand_indices: set[int] = set()

        def _try_keep(cand_idx: int) -> bool:
            """Keep candidate if it passes the dedup filter. Returns True on success."""
            c_score, c_doc_idx, c_stat_list = candidates[cand_idx]
            keys = {(s.stat_type, _norm_value(s.value)) for s in c_stat_list}
            if keys & kept_keys:
                return False
            kept_keys.update(keys)
            kept.append((c_score, c_doc_idx))
            selected_cand_indices.add(cand_idx)
            return True

        # Hint pool: walk hint_keyed until hint_quota fact-deduped sentences kept.
        hint_kept_count = 0
        for sort_key, _doc, cand_idx in hint_keyed:
            if hint_mode == "hard" and sort_key == float("inf"):
                # bonus == 0 in hard mode — ineligible for hint pool.
                continue
            if cand_idx in selected_cand_indices:
                continue
            if hint_kept_count >= hint_quota:
                break
            if _try_keep(cand_idx):
                hint_kept_count += 1

        # Plain pool: walk plain_keyed until normal_quota MORE kept.
        normal_kept_count = 0
        for _neg_score, _doc, cand_idx in plain_keyed:
            if cand_idx in selected_cand_indices:
                continue
            if normal_kept_count >= normal_quota:
                break
            if _try_keep(cand_idx):
                normal_kept_count += 1

        # Rollover: hint pool came up short → take more.
        # In hard mode, stay restricted to hint-bearing candidates (walk hint_keyed,
        # skipping the float("inf") sentinel entries that mark non-matching items).
        # In soft mode, draw from plain_keyed (any candidate).
        if hint_kept_count < hint_quota:
            extra_needed = hint_quota - hint_kept_count
            rollover_pool = hint_keyed if hint_mode == "hard" else plain_keyed
            for sort_key, _doc, cand_idx in rollover_pool:
                if hint_mode == "hard" and sort_key == float("inf"):
                    # non-matching candidate in hard-mode hint pool — skip
                    continue
                if cand_idx in selected_cand_indices:
                    continue
                if extra_needed <= 0:
                    break
                if _try_keep(cand_idx):
                    extra_needed -= 1

        # Soft-only rollover: plain pool came up short → hint pool takes more.
        if hint_mode == "soft" and normal_kept_count < normal_quota:
            extra_needed = normal_quota - normal_kept_count
            for sort_key, _doc, cand_idx in hint_keyed:
                if cand_idx in selected_cand_indices:
                    continue
                if extra_needed <= 0:
                    break
                if _try_keep(cand_idx):
                    extra_needed -= 1

        # Return in document order.
        kept.sort(key=lambda c: c[1])
        return tuple(sentences[i] for _, i in kept)

    # No hints: original single-pool dedup path (byte-identical to pre-T7).
    # Dedup by (stat_type, normalized value). Walk candidates in
    # score-descending order; for each, if any of its (stat_type, norm_value)
    # keys collide with an already-kept candidate, drop the later one.
    # Ties on score broken by earlier document position.
    candidates.sort(key=lambda c: (-c[0], c[1]))
    kept_keys_sp: set[tuple[str, str]] = set()
    kept_sp: list[tuple[float, int]] = []
    for score, idx, stat_list in candidates:
        keys = {(s.stat_type, _norm_value(s.value)) for s in stat_list}
        if keys & kept_keys_sp:
            continue
        kept_keys_sp.update(keys)
        kept_sp.append((score, idx))
        if len(kept_sp) >= max_facts:
            break

    # Return in document order.
    kept_sp.sort(key=lambda c: c[1])
    return tuple(sentences[i] for _, i in kept_sp)


register("regex", "key_facts", key_facts)
