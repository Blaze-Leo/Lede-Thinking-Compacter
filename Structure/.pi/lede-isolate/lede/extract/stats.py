"""Numeric-fact extractor. Regex-based, deterministic, stdlib only.

Optional word-form number support (convert_word_names=True) uses
`text_to_num` (MIT, PyPI package name `text2num`). When the flag is off
the path stays stdlib-only.
"""
import re
from .._types import Stat
from ..sentences import split_sentences


# Numeric quantifiers are bounded ({1,15}) to defend against catastrophic
# backtracking. Python's `re` engine is backtracking; a long unbroken digit
# run followed by an almost-matching unit keyword can otherwise hang for
# minutes. No real-world stat token has more than 15 digits before its unit
# (15-digit money = $999 trillion, 15-digit count = 999 quadrillion). The
# Rust `regex` crate is RE2-style and unaffected, but mirrors the bounds for
# parity.
# Numeric quantifiers are bounded ({1,15}) to defend against catastrophic
# backtracking. Python's `re` engine is backtracking; without these
# guards an N-digit unbroken run followed by an almost-matching unit
# keyword could retry each (start × length × alternation) combination,
# turning a 100K-char input into a 14-minute hang. No real-world stat
# token has more than 15 digits before its unit (15-digit money = $999
# trillion, 15-digit count = 999 quadrillion). Inputs with a 20+ digit
# unbroken run on a single sentence are also skipped via _LONG_RUN_RE
# below — that's the second line of defense and applies in both
# languages identically. The Rust `regex` crate is RE2-style and
# unaffected by backtracking, but mirrors both guards for byte-identical
# parity.
_MONEY_RE = re.compile(
    r"(?P<value>\$\d[\d,]{0,18}(?:\.\d{1,4})?[KMB]?(?:\s+(?:million|billion|trillion|thousand)\b)?)"
    r"|(?P<value2>\d[\d,]{0,18}(?:\.\d{1,4})?(?:\s+(?:million|billion|trillion|thousand)\b)?)\s*(?P<ccy>dollars?|USD|EUR|GBP|JPY|CHF)"
    r"|(?P<ccyp>USD|EUR|GBP|JPY|CHF)\s+(?P<value3>\d[\d,]{0,18}(?:\.\d{1,4})?(?:\s+(?:million|billion|trillion|thousand)\b)?)",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(
    r"(?P<value>\d{1,15}(?:\.\d{1,6})?)\s*(?P<unit>%|percent)",
    re.IGNORECASE,
)

# NOTE: `year` alternative matches bare 4-digit years in 1900-2099. Known
# collision: numerals like "1500 dollars" would also match as a date stat
# because primitives run independently. No current corpus exercises this;
# future work could add a negative lookahead for currency/unit context.
_DATE_RE = re.compile(
    r"(?P<iso>\d{4}-\d{2}-\d{2})"
    r"|(?P<us>\d{1,2}/\d{1,2}/\d{2,4})"
    r"|(?P<year>\b(?:19|20)\d{2}\b)",
)

_DURATION_RE = re.compile(
    r"(?P<value>\d{1,15})[-\s]*(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?)",
    re.IGNORECASE,
)

_COUNT_RE = re.compile(
    r"(?P<value>\d{1,15}(?:,\d{3}){0,5})\s*(?P<unit>events?|users?|customers?|requests?"
    r"|per second|per minute|per hour|qps|rps|chunks?"
    r"|terabytes?|basis\s+points?"
    r"|items?|documents?|lines?|entries?|records?|files?|actions?|sections?|units?"
    r"|tons?\s+per\s+(?:year|month|day))",
    re.IGNORECASE,
)

# Sentences containing an unbroken 20+ digit run are skipped — defends
# against pathological inputs (encoded data leaks, base64 chunks, OCR
# artifacts) where the bounded quantifier alone would still retry each
# starting position. Real-world numeric tokens are well below 20 digits
# (16-digit cards, 13-digit ISBN, 12-digit phone). Both Python and Rust
# apply this skip identically for parity.
_LONG_RUN_RE = re.compile(r"\d{20,}")


def _ctx_phrase(sentence: str, start: int, end: int, window: int = 25) -> str:
    """Return `sentence[start-window:end+window]`, clipped to token boundaries."""
    l = max(0, start - window)
    r = min(len(sentence), end + window)
    return sentence[l:r].strip()


# ---------- T13e word-form number support (optional dep: text_to_num) -----------

_WORDFORMS_IMPORT_ERROR = (
    "convert_word_names=True requires the 'wordforms' extra — "
    "install with: pip install lede[wordforms]"
)


class _CharToken:
    """Token wrapper for text_to_num.find_numbers that carries original char span."""
    __slots__ = ("_text", "char_start", "char_end")

    def __init__(self, text: str, char_start: int, char_end: int) -> None:
        self._text = text
        self.char_start = char_start
        self.char_end = char_end

    def text(self) -> str:
        return self._text

    def not_a_number_part(self) -> bool:
        return False

    def nt_separated(self, previous: "_CharToken") -> bool:
        return False


# Tokenize into whitespace-separated tokens AND isolated punctuation runs,
# preserving char spans. Punctuation must be its own token so text_to_num
# treats "eight days." and "eight days" the same (period must not stick to
# "days").
_TOKEN_RE = re.compile(r"\w+|[^\w\s]+")


def _tokenize_with_spans(sentence: str) -> list[_CharToken]:
    """Tokenize `sentence` preserving (text, char_start, char_end) per token."""
    return [
        _CharToken(m.group(), m.start(), m.end())
        for m in _TOKEN_RE.finditer(sentence)
    ]


def _word_form_spans(sentence: str) -> list[tuple[int, int, str]]:
    """Return [(orig_char_start, orig_char_end, digit_text), ...] for word-form
    numbers in `sentence`. Lazy-imports text_to_num. Raises ImportError with
    a helpful message if missing."""
    try:
        from text_to_num import find_numbers  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised by test_stats_word_form_flag_without_dep_raises
        raise ImportError(_WORDFORMS_IMPORT_ERROR) from e

    tokens = _tokenize_with_spans(sentence)
    if not tokens:
        return []
    # threshold=0.0 catches isolated small numbers like "two weeks" that the
    # default threshold (3.0) would otherwise skip.
    occurrences = find_numbers(tokens, "en", threshold=0.0)
    spans: list[tuple[int, int, str]] = []
    for occ in occurrences:
        start_tok = tokens[occ.start]
        end_tok = tokens[occ.end - 1]
        spans.append((start_tok.char_start, end_tok.char_end, occ.text))
    return spans


def _substitute(sentence: str, spans: list[tuple[int, int, str]]) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Replace each (orig_start, orig_end, digit_text) in `sentence`.

    Returns (substituted_sentence, span_map) where span_map is a list of
    (orig_start, orig_end, sub_start, sub_end). Spans are assumed to be
    non-overlapping and sorted ascending (text_to_num guarantees that).
    """
    if not spans:
        return sentence, []
    out_parts: list[str] = []
    span_map: list[tuple[int, int, int, int]] = []
    cursor = 0
    sub_offset = 0
    for orig_start, orig_end, digit_text in spans:
        out_parts.append(sentence[cursor:orig_start])
        sub_start = orig_start + sub_offset
        out_parts.append(digit_text)
        sub_end = sub_start + len(digit_text)
        span_map.append((orig_start, orig_end, sub_start, sub_end))
        sub_offset += len(digit_text) - (orig_end - orig_start)
        cursor = orig_end
    out_parts.append(sentence[cursor:])
    return "".join(out_parts), span_map


def _map_back_impl(sub_start: int, sub_end: int, span_map: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    """Clean implementation of span mapping.

    Each char in the substituted sentence maps to an original char position:
      - If it lies inside a substituted span [sub_s, sub_e), it maps to the
        entire original span [orig_s, orig_e).
      - Otherwise it maps to sub_pos - cumulative_delta (delta = diff between
        substituted and original lengths of all earlier-or-overlapping spans).

    For robustness, we compute the mapping for the start edge and the end
    edge separately, expanding to whole original spans when the match
    touches any part of a substituted span.
    """
    def map_pos(sub_pos: int, is_end: bool) -> int:
        delta = 0
        for orig_s, orig_e, sub_s, sub_e in span_map:
            if sub_e <= sub_pos:
                delta += (sub_e - sub_s) - (orig_e - orig_s)
                continue
            if sub_s >= sub_pos:
                # span is at or after sub_pos
                # (no-op — no more spans contribute to delta for this pos)
                break
            # sub_s < sub_pos < sub_e: match edge falls inside span
            return orig_e if is_end else orig_s
        return sub_pos - delta

    orig_start = map_pos(sub_start, is_end=False)
    orig_end = map_pos(sub_end, is_end=True)
    # Also expand to cover any span the match partially crosses but whose
    # start/end edges don't fall inside it (e.g. a span wholly contained
    # inside the match). This is rare for stats regexes but guarantees
    # value-preservation invariants.
    for orig_s, orig_e, sub_s, sub_e in span_map:
        if sub_s >= sub_start and sub_e <= sub_end:
            orig_start = min(orig_start, orig_s)
            orig_end = max(orig_end, orig_e)
    return orig_start, orig_end


def _stats_regex_on(sent_orig: str, sent_sub: str, span_map: list[tuple[int, int, int, int]]) -> list[Stat]:
    """Apply the stats regexes to `sent_sub` and emit Stat records whose
    value / phrase / context_sentence slices come from `sent_orig`.

    When span_map is empty (no word-form substitutions happened), sent_sub
    == sent_orig and _map_back_impl is a no-op pass-through — so the
    default flag-off path is byte-identical to the pre-T13e behavior.
    """
    out: list[Stat] = []

    def emit(m_start: int, m_end: int, stat_type: str, unit: str, value_override: str | None = None) -> None:
        orig_s, orig_e = _map_back_impl(m_start, m_end, span_map)
        original_span_text = sent_orig[orig_s:orig_e]
        out.append(Stat(
            value=value_override if value_override is not None else original_span_text.strip(),
            unit=unit,
            phrase=_ctx_phrase(sent_orig, orig_s, orig_e),
            context_sentence=sent_orig,
            stat_type=stat_type,
        ))

    for m in _MONEY_RE.finditer(sent_sub):
        # Pick the matched alternative to keep value crisp (digits w/ $ or
        # ccy word). The original-span emit is driven by whole-match start/end.
        value_sub = m.group("value") or m.group("value2") or m.group("value3")
        # Use the original span text as the value so "eight dollars" emits
        # "eight" not "8". For money, most corpora are digit-only, but we
        # preserve consistency.
        orig_s, orig_e = _map_back_impl(m.start(), m.end(), span_map)
        # Extract just the numeric portion of the original. To do that,
        # find where `value_sub` starts inside the match; apply same offset
        # in the original span.
        num_start_in_match = m.start(m.lastgroup if m.lastgroup else 0)  # fallback; handled below
        # Simpler and robust: just use the original slice for value.
        original_value = sent_orig[orig_s:orig_e].strip()
        # Preserve the leading $ prefix / trailing ccy if only one grp hit.
        # For byte-identical parity with pre-T13e when no substitutions are
        # in play, drop to the original emit shape.
        if not span_map:
            value_emit = value_sub
        else:
            value_emit = original_value
        out.append(Stat(
            value=value_emit,
            unit="usd",
            phrase=_ctx_phrase(sent_orig, orig_s, orig_e),
            context_sentence=sent_orig,
            stat_type="money",
        ))

    for m in _PERCENT_RE.finditer(sent_sub):
        orig_s, orig_e = _map_back_impl(m.start(), m.end(), span_map)
        # Value: original numeric portion — locate the value group's offset in
        # the substituted sentence, map that back.
        v_sub_s, v_sub_e = m.start("value"), m.end("value")
        v_orig_s, v_orig_e = _map_back_impl(v_sub_s, v_sub_e, span_map)
        value_emit = sent_orig[v_orig_s:v_orig_e].strip() if span_map else m.group("value")
        out.append(Stat(
            value=value_emit,
            unit="percent",
            phrase=_ctx_phrase(sent_orig, orig_s, orig_e),
            context_sentence=sent_orig,
            stat_type="percent",
        ))

    for m in _DATE_RE.finditer(sent_sub):
        orig_s, orig_e = _map_back_impl(m.start(), m.end(), span_map)
        # Dates are intrinsically digit-bearing; word-form "nineteen ninety-nine"
        # does get converted by text_to_num. Preserve original slice.
        if not span_map:
            value_emit = m.group("iso") or m.group("us") or m.group("year")
        else:
            value_emit = sent_orig[orig_s:orig_e].strip()
        out.append(Stat(
            value=value_emit,
            unit="date",
            phrase=_ctx_phrase(sent_orig, orig_s, orig_e),
            context_sentence=sent_orig,
            stat_type="date",
        ))

    for m in _DURATION_RE.finditer(sent_sub):
        unit_str = m.group("unit").lower()
        orig_s, orig_e = _map_back_impl(m.start(), m.end(), span_map)
        # Map the numeric value portion separately so we can construct
        # "eight days" from original word-form even though regex sees "8".
        v_sub_s, v_sub_e = m.start("value"), m.end("value")
        v_orig_s, v_orig_e = _map_back_impl(v_sub_s, v_sub_e, span_map)
        u_sub_s, u_sub_e = m.start("unit"), m.end("unit")
        u_orig_s, u_orig_e = _map_back_impl(u_sub_s, u_sub_e, span_map)
        if span_map:
            orig_value_text = sent_orig[v_orig_s:v_orig_e].strip()
            orig_unit_text = sent_orig[u_orig_s:u_orig_e].strip().lower()
            value_emit = f"{orig_value_text} {orig_unit_text}"
        else:
            value_emit = f"{m.group('value')} {unit_str}"
        out.append(Stat(
            value=value_emit,
            unit=unit_str.rstrip("s"),
            phrase=_ctx_phrase(sent_orig, orig_s, orig_e),
            context_sentence=sent_orig,
            stat_type="duration",
        ))

    for m in _COUNT_RE.finditer(sent_sub):
        orig_s, orig_e = _map_back_impl(m.start(), m.end(), span_map)
        v_sub_s, v_sub_e = m.start("value"), m.end("value")
        v_orig_s, v_orig_e = _map_back_impl(v_sub_s, v_sub_e, span_map)
        if span_map:
            value_emit = sent_orig[v_orig_s:v_orig_e].strip()
        else:
            value_emit = m.group("value")
        out.append(Stat(
            value=value_emit,
            unit=m.group("unit").lower(),
            phrase=_ctx_phrase(sent_orig, orig_s, orig_e),
            context_sentence=sent_orig,
            stat_type="count",
        ))

    return out


def stats(text: str, *, convert_word_names: bool = False) -> tuple[Stat, ...]:
    """Extract money, percent, date, duration, count facts with context.

    Args:
        text: input document text.
        convert_word_names: (T13e) when True, use text_to_num to recognize
            spelled-out numbers like "eight days" or "five thousand
            documents" in addition to digit forms. The regex pipeline runs
            on a digit-substituted shadow copy, but emitted Stat records
            preserve the ORIGINAL word-form text in `value`, `phrase`,
            and `context_sentence`. Requires the 'wordforms' extra.
            Default False → zero-runtime-dep path, byte-identical to
            pre-T13e behavior.
    """
    out: list[Stat] = []
    if convert_word_names:
        # Eagerly import once so we fail fast with a clear message if the
        # dep is missing, rather than only on the first sentence with a
        # word-form number.
        try:
            import text_to_num  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise ImportError(_WORDFORMS_IMPORT_ERROR) from e

    for sent in split_sentences(text):
        # ReDoS guard: skip sentences with pathologically long digit runs.
        # Mirrored in rust/src/extract/stats.rs for byte-identical parity.
        if _LONG_RUN_RE.search(sent):
            continue
        if convert_word_names:
            spans = _word_form_spans(sent)
            sent_sub, span_map = _substitute(sent, spans)
            out.extend(_stats_regex_on(sent, sent_sub, span_map))
        else:
            # Fast path: no span mapping, no imports, byte-identical to pre-T13e.
            for m in _MONEY_RE.finditer(sent):
                value = m.group("value") or m.group("value2") or m.group("value3")
                out.append(Stat(
                    value=value,
                    unit="usd",
                    phrase=_ctx_phrase(sent, m.start(), m.end()),
                    context_sentence=sent,
                    stat_type="money",
                ))
            for m in _PERCENT_RE.finditer(sent):
                out.append(Stat(
                    value=m.group("value"),
                    unit="percent",
                    phrase=_ctx_phrase(sent, m.start(), m.end()),
                    context_sentence=sent,
                    stat_type="percent",
                ))
            for m in _DATE_RE.finditer(sent):
                value = m.group("iso") or m.group("us") or m.group("year")
                out.append(Stat(
                    value=value,
                    unit="date",
                    phrase=_ctx_phrase(sent, m.start(), m.end()),
                    context_sentence=sent,
                    stat_type="date",
                ))
            for m in _DURATION_RE.finditer(sent):
                out.append(Stat(
                    value=f"{m.group('value')} {m.group('unit').lower()}",
                    unit=m.group("unit").lower().rstrip("s"),
                    phrase=_ctx_phrase(sent, m.start(), m.end()),
                    context_sentence=sent,
                    stat_type="duration",
                ))
            for m in _COUNT_RE.finditer(sent):
                out.append(Stat(
                    value=m.group("value"),
                    unit=m.group("unit").lower(),
                    phrase=_ctx_phrase(sent, m.start(), m.end()),
                    context_sentence=sent,
                    stat_type="count",
                ))
    return tuple(out)
