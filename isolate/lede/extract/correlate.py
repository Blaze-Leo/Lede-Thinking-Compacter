"""extract.correlate_facts — pair repeated entities with nearby numbers."""
import re
from collections import Counter
from .._types import PhraseFact
from ._backends import register, resolve, get_default_backend
from .._hints import hint_bonus
from .stats import stats as _stats
from .phrases import phrases as _phrases, _STOP


_GROWTH_WORDS = frozenset({
    "grew", "grow", "increased", "increase", "rose", "up",
    "higher", "gained", "added",
})
_DECLINE_WORDS = frozenset({
    "fell", "fall", "declined", "decline", "decreased", "decrease",
    "dropped", "down", "lower", "lost",
})


def _polarity(sentence: str) -> str:
    """Infer polarity from cue words in the sentence."""
    lower = sentence.lower()
    toks = set(re.findall(r"[a-z]+", lower))
    if toks & _GROWTH_WORDS:
        return "growth"
    if toks & _DECLINE_WORDS:
        return "decline"
    return "absolute"


def _regex_correlate_facts(
    text: str,
    *,
    convert_word_names: bool = False,
    hints: list[tuple[str, float]] | None = None,
    hint_mode: str = "soft",
) -> tuple[PhraseFact, ...]:
    """Regex-backed entity-number correlator. Registered as ('regex','correlate_facts').

    `convert_word_names` is forwarded to the internal stats() call so
    word-form numbers surface as pairing candidates (T13e).
    """
    if not text:
        return ()
    stats_list = _stats(text, convert_word_names=convert_word_names)
    if not stats_list:
        return ()

    # Build single-word frequency map — repeated single-word entity candidates
    single_word_counts: dict[str, int] = {}
    for w in re.findall(r"[a-zA-Z]{3,}", text.lower()):
        single_word_counts[w] = single_word_counts.get(w, 0) + 1
    repeated_words = {
        w for w, c in single_word_counts.items()
        if c >= 2 and w not in _STOP
    }

    # Prefer multi-word phrases from the regex phrases() backend.
    # Pinning backend='regex' keeps this impl internally consistent —
    # a spaCy-backed correlate (T11b) would build its own dep-parsed version.
    # NB: keep phrases() insertion order (do NOT wrap in set()) — when a
    # sentence contains two repeated phrases, the first-match below must be
    # deterministic. set() iterates in PYTHONHASHSEED order, which made this
    # nondeterministic and broke Python<->Rust parity (Rust uses an ordered Vec).
    repeated_phrases = _phrases(text, backend="regex")

    out: list[PhraseFact] = []
    for st in stats_list:
        sent_lower = st.context_sentence.lower()
        candidates = [
            w for w in re.findall(r"[a-zA-Z]{3,}", sent_lower)
            if w in repeated_words
        ]
        # Prefer a multi-word phrase whose every token appears in the sentence
        phrase_match = next(
            (p for p in repeated_phrases if all(w in sent_lower for w in p.split())),
            None,
        )
        entity = phrase_match if phrase_match else (
            max(candidates, key=lambda w: single_word_counts[w]) if candidates else None
        )
        if not entity:
            continue
        out.append(PhraseFact(
            entity=entity,
            number=st.value,
            polarity=_polarity(st.context_sentence),
            sentence=st.context_sentence,
        ))

    # Filter: require each entity to appear with >= 2 distinct facts
    entity_counts = Counter(pf.entity for pf in out)
    final = tuple(pf for pf in out if entity_counts[pf.entity] >= 2)

    if not hints:
        return final

    # Partition by hint match. Match target is OR of entity name and fact value
    # (the PhraseFact fields). Build a combined target: "{entity} {number}".
    hint_bearing: list[PhraseFact] = []
    non_hint: list[PhraseFact] = []
    for pf in final:
        target = f"{pf.entity} {pf.number}"
        if hint_bonus(target, hints) > 0:
            hint_bearing.append(pf)
        else:
            non_hint.append(pf)

    if hint_mode == "hard":
        return tuple(hint_bearing)
    # soft: hint-bearing first (original sub-order), then non-hint (original sub-order)
    return tuple(hint_bearing + non_hint)


register("regex", "correlate_facts", _regex_correlate_facts)


def correlate_facts(
    text: str,
    *,
    backend: str | None = None,
    convert_word_names: bool = False,
    hints: list[str] | dict[str, float] | None = None,
    hint_focus: float = 0.7,
    hint_mode: str = "soft",
) -> tuple[PhraseFact, ...]:
    """Pair repeated entities with their numeric facts.

    backend='regex' (default): composition over stats() + regex phrases() + cue-word polarity.
    backend='spacy': requires lede-spacy installed (dep-parser-based impl, future T11b).
    backend='auto': spacy if registered else regex.
    backend=None: uses the global default (see `lede.set_default_backend`).

    convert_word_names (T13e): forwarded to the internal stats() call so
    spelled-out numbers like "eight days" or "five thousand" surface as
    pairing candidates. Ignored by backends that don't support it — the
    regex backend accepts it; the spacy backend currently does not and
    will raise TypeError if you pass True. Default False preserves the
    zero-runtime-dep contract.

    Hint biasing (v0.4):
        Pass ``hints`` to bias which facts are returned or how they are
        ordered. When ``hints`` is None (the default), output is byte-identical
        to v0.3.0.

        Match target is OR semantics: a ``PhraseFact`` matches if **either**
        its ``entity`` field **or** its ``number`` field contains a hint term
        (case-insensitive, ``\\b``-delimited). Internally, this is checked by
        constructing the combined target ``"{entity} {number}"``.

        - ``"soft"`` (default): hint-bearing facts reordered to the front in
          their original relative order; all facts retained.
        - ``"hard"``: only hint-bearing facts returned (hard filter).
        - ``hint_focus`` is validated but has no behavioral effect on this
          primitive because ``correlate_facts()`` returns all candidates rather
          than a fixed-size count. Use ``hint_mode`` to control filtering.

        See REFERENCE.md "Hint biasing" for the full contract and validation
        rules.

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
    impl = resolve(backend, "correlate_facts")

    # Only the regex backend accepts hints and convert_word_names today. For
    # other backends, forward only the kwargs they support.
    if backend == "regex" and processed_hints:
        return impl(
            text,
            convert_word_names=convert_word_names,
            hints=processed_hints,
            hint_mode=hint_mode,
        )
    if convert_word_names:
        return impl(text, convert_word_names=True)
    return impl(text)
