"""Heuristic phrase extractor. Repeated 2-5 token n-grams between stopwords."""
import re
from ._backends import register, resolve, get_default_backend
from .._hints import hint_bonus


_STOP = frozenset(
    "the a an and or but if then with by of to in on at for from as is are was "
    "were be been being has have had do does did will would could should may "
    "might can must this that these those it its they them their there here "
    "about up down into out over off just also not no our we you your he she "
    "his her him over under more most less least all any some both each every "
    "one two three per".split()
)


def _ngrams(buf: list[str]) -> list[str]:
    """Emit all contiguous 2-5 token n-grams from a non-stopword run."""
    out: list[str] = []
    n = len(buf)
    upper = min(5, n)
    for size in range(2, upper + 1):
        for i in range(0, n - size + 1):
            out.append(" ".join(buf[i:i + size]))
    return out


def _runs(text: str) -> list[str]:
    """Return all 2-5 token n-grams from non-stopword runs (lowercased)."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    runs: list[str] = []
    buf: list[str] = []
    for w in words:
        if w in _STOP:
            runs.extend(_ngrams(buf))
            buf = []
        else:
            buf.append(w)
    runs.extend(_ngrams(buf))
    return runs


def _subsumed(candidates: list[str], counts: dict[str, int]) -> set[str]:
    """Identify sub-ngrams that are fully absorbed by a longer emitted ngram.

    T13h: drop ngram A if there exists a longer emitted ngram B such that A's
    tokens appear contiguously within B's tokens AND counts[A] == counts[B].
    The equal-count signal means A only ever appears inside B's context and
    adds no independent information. Sub-ngrams with counts[A] > counts[B]
    (i.e. A has standalone occurrences) are preserved.
    """
    tok_map = {p: p.split() for p in candidates}
    drop: set[str] = set()
    for a in candidates:
        a_toks = tok_map[a]
        a_len = len(a_toks)
        for b in candidates:
            if a == b:
                continue
            b_toks = tok_map[b]
            if len(b_toks) <= a_len:
                continue
            if counts[b] != counts[a]:
                continue
            # is a contiguous inside b?
            for i in range(len(b_toks) - a_len + 1):
                if b_toks[i:i + a_len] == a_toks:
                    drop.add(a)
                    break
            if a in drop:
                break
    return drop


def _regex_phrases(
    text: str,
    keywords: str | None = None,
    *,
    hints: list[tuple[str, float]] | None = None,
    hint_mode: str = "soft",
) -> tuple[str, ...]:
    """Regex/heuristic phrase extractor. Registered as the 'regex' backend."""
    if not text:
        return ()
    runs = _runs(text)
    counts: dict[str, int] = {}
    order: list[str] = []
    for r in runs:
        counts[r] = counts.get(r, 0) + 1
        if r not in order:
            order.append(r)
    repeated = [r for r in order if counts[r] >= 2]
    # T13h: subsumption — drop sub-ngrams that only appear inside a longer
    # emitted ngram (same count). Tightens precision by eliminating n-gram
    # explosion noise like "environmental protection" / "protection agency"
    # when "environmental protection agency" is also emitted.
    drop = _subsumed(repeated, counts)
    out = [r for r in repeated if r not in drop]
    if keywords:
        kw_set = {k.lower() for k in re.findall(r"[a-z]{3,}", keywords.lower())}
        for r in order:
            if counts.get(r, 0) == 1 and any(k in r.split() for k in kw_set):
                out.append(r)
    # Dedupe keeping first-appearance order
    seen: set[str] = set()
    deduped: list[str] = []
    for r in out:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    if not hints:
        return tuple(deduped)

    # Partition by hint match. Match target is the phrase string itself.
    hint_bearing: list[str] = []
    non_hint: list[str] = []
    for phrase in deduped:
        if hint_bonus(phrase, hints) > 0:
            hint_bearing.append(phrase)
        else:
            non_hint.append(phrase)

    if hint_mode == "hard":
        return tuple(hint_bearing)
    # soft: hint-bearing first (original sub-order), then non-hint (original sub-order)
    return tuple(hint_bearing + non_hint)


register("regex", "phrases", _regex_phrases)


def phrases(
    text: str,
    keywords: str | None = None,
    *,
    backend: str | None = None,
    hints: list[str] | dict[str, float] | None = None,
    hint_focus: float = 0.7,
    hint_mode: str = "soft",
) -> tuple[str, ...]:
    """Extract repeated multi-word phrases (2-5 token n-grams from non-stopword runs).

    backend='regex' (default): the heuristic stdlib implementation above.
    backend='spacy': requires lede-spacy installed (noun_chunks-based impl, future T10b).
    backend='auto': spacy if registered else regex.
    backend=None: uses the global default (see `lede.set_default_backend`).

    Hint biasing (v0.4):
        Pass ``hints`` to bias which phrases are returned or how they are
        ordered. When ``hints`` is None (the default), output is byte-identical
        to v0.3.0.

        - ``"soft"`` (default): hint-bearing phrases reordered to the front
          in their original relative order; all phrases retained.
        - ``"hard"``: only hint-bearing phrases returned (hard filter).
        - ``hint_focus`` is validated but has no behavioral effect on this
          primitive because ``phrases()`` returns all candidates rather than
          a fixed-size count. Use ``hint_mode`` to control filtering.

        Matching is case-insensitive and word-boundary-delimited (``\\b``);
        the match target is the phrase string itself. See REFERENCE.md
        "Hint biasing" for the full contract and validation rules.

    Raises:
        ValueError: on ``hint_focus`` outside [0.0, 1.0] or unknown
            ``hint_mode``.
    """
    from .._hints import preprocess_hints

    processed_hints = preprocess_hints(hints)
    if processed_hints:
        if not 0.0 <= hint_focus <= 1.0:
            raise ValueError(f"hint_focus must be in [0.0, 1.0]; got {hint_focus}")
        if hint_mode not in ("soft", "hard"):
            raise ValueError(f"hint_mode must be 'soft' or 'hard'; got {hint_mode!r}")

    if backend is None:
        backend = get_default_backend()
    impl = resolve(backend, "phrases")

    # Only the regex backend accepts hints today. For other backends (yake, spacy),
    # forward without hints to preserve backward compat.
    if backend == "regex" and processed_hints:
        return impl(text, keywords=keywords, hints=processed_hints, hint_mode=hint_mode)
    return impl(text, keywords=keywords)
