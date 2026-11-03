"""Readable combined reports over lede core and optional spaCy extraction."""
from __future__ import annotations

from collections.abc import Sequence
import re

from ._types import FactRecord, PromotionCandidate, ReadableReport, ReportAttribute
from .tfidf import summarize
from .extract import correlate_facts, key_facts, metadata, phrases, stats


_BOLD_KV_RE = re.compile(
    r"\*\*\s*(?P<label>[^:\n*][^:\n]{0,80}?)\s*:\s*\*\*\s*"
    r"(?P<value>.*?)(?=(?:\s+\*\*\s*[^:\n*][^:\n]{0,80}?\s*:\s*\*\*)|\n|$)",
    re.S,
)
_PLAIN_KV_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<label>[A-Za-z][A-Za-z0-9 ._/\-&()]{1,80})\s*:\s*(?P<value>.+?)\s*$"
)


def _register_spacy_if_requested(backend: str) -> bool:
    if backend not in ("spacy", "auto"):
        return False
    try:
        import lede_spacy  # noqa: F401  # side-effect: registers backend
        return True
    except ImportError as e:
        if backend == "spacy":
            raise ImportError(
                "readable_report(backend='spacy') requires lede-spacy. Install with:\n"
                "    pip install lede-spacy\n"
                "    python -m spacy download en_core_web_sm"
            ) from e
        return False


def _normalize_key(label: str) -> str:
    key = label.strip().lower().replace("&", " and ")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return re.sub(r"_+", "_", key).strip("_")


def _compact_evidence(text: str, *, max_chars: int = 260) -> str:
    value = " ".join(text.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def _value_type(key: str, value: str) -> str:
    lower = value.lower()
    if re.fullmatch(r"\d{4}", value):
        return "year"
    if re.search(r"\$|usd|eur|gbp", lower):
        return "money"
    if re.search(r"https?://|www\.", lower):
        return "url"
    if "citation" in key or re.search(r"\b\d+\s+u\.s\.", lower):
        return "citation"
    if "docket" in key or re.fullmatch(r"\d{1,4}-\d+[A-Za-z]?", value):
        return "identifier"
    if "," in value or ";" in value:
        return "list"
    return "text"


def _attribute_from_pair(label: str, value: str, evidence: str, source: str) -> ReportAttribute | None:
    clean_label = " ".join(label.strip(" *").split())
    clean_value = " ".join(value.strip(" *").split())
    if not clean_label or not clean_value:
        return None
    key = _normalize_key(clean_label)
    if not key or len(key) > 80:
        return None
    return ReportAttribute(
        key=key,
        label=clean_label,
        value=clean_value,
        value_type=_value_type(key, clean_value),
        source=source,
        evidence=_compact_evidence(evidence),
        confidence=0.99 if source == "header_kv" else 0.9,
    )


def _extract_attributes(text: str) -> tuple[ReportAttribute, ...]:
    found: dict[tuple[str, str], ReportAttribute] = {}

    for match in _BOLD_KV_RE.finditer(text):
        label = match.group("label")
        value = match.group("value")
        attr = _attribute_from_pair(label, value, match.group(0), "header_kv")
        if attr:
            found.setdefault((attr.key, attr.value), attr)

    for line in text.splitlines():
        if "**" in line:
            continue
        match = _PLAIN_KV_RE.match(line)
        if not match:
            continue
        attr = _attribute_from_pair(
            match.group("label"),
            match.group("value"),
            line,
            "line_kv",
        )
        if attr:
            found.setdefault((attr.key, attr.value), attr)

    return tuple(found.values())


def _build_fact_records(
    attributes: tuple[ReportAttribute, ...],
    lede_stats,
    spacy_facts,
) -> tuple[FactRecord, ...]:
    records: list[FactRecord] = []
    for attr in attributes:
        records.append(
            FactRecord(
                subject="document",
                predicate=attr.key,
                object=attr.value,
                fact_type="attribute",
                evidence=attr.evidence,
                confidence=attr.confidence,
            )
        )
    for stat in lede_stats:
        records.append(
            FactRecord(
                subject="document",
                predicate=stat.stat_type,
                object=stat.value,
                fact_type="numeric",
                evidence=_compact_evidence(stat.context_sentence),
                confidence=0.8,
            )
        )
    for fact in spacy_facts:
        records.append(
            FactRecord(
                subject=fact.entity,
                predicate=fact.polarity,
                object=fact.number,
                fact_type="entity_number",
                evidence=_compact_evidence(fact.sentence),
                confidence=0.7,
            )
        )
    return tuple(records)


def _build_promotion_candidates(
    attributes: tuple[ReportAttribute, ...],
) -> tuple[PromotionCandidate, ...]:
    return tuple(
        PromotionCandidate(
            path=f"lede_report.attributes.{attr.key}.value",
            key=attr.key,
            value_type=attr.value_type,
            promote=True,
            confidence=attr.confidence,
        )
        for attr in attributes
    )


def _build_search_text(
    summary_text: str,
    key_fact_rows: tuple[str, ...],
    attributes: tuple[ReportAttribute, ...],
    lede_metadata,
    spacy_metadata,
    spacy_phrases: tuple[str, ...],
) -> str:
    parts: list[str] = [summary_text]
    parts.extend(key_fact_rows)
    parts.extend(f"{attr.label}: {attr.value}" for attr in attributes)
    parts.extend(lede_metadata.dates)
    parts.extend(lede_metadata.amounts)
    parts.extend(lede_metadata.urls)
    if spacy_metadata:
        parts.extend(spacy_metadata.entities)
    parts.extend(spacy_phrases)
    return "\n".join(part for part in parts if part)


def readable_report(
    text: str,
    *,
    max_length: int = 2000,
    max_facts: int = 40,
    mode: str = "default",
    backend: str = "regex",
    include_toc: bool = True,
    keep_headings: bool = True,
    headings: Sequence[str] | None = None,
    pin: Sequence[str] | None = None,
    hints: list[str] | dict[str, float] | None = None,
    hint_focus: float = 0.7,
    hint_mode: str = "soft",
    convert_word_names: bool = False,
) -> ReadableReport:
    """Build a readable report combining lede summary/facts and optional spaCy details.

    The default is tuned for human and agent inspection: a 2000-character
    lede summary with headings/TOC retained, up to 40 lede key facts, regex
    metadata/stats, and spaCy-backed entities / entity-number correlations
    only when requested with backend="spacy" or backend="auto".
    """
    spacy_available = _register_spacy_if_requested(backend)

    summary = summarize(
        text,
        max_length=max_length,
        mode=mode,
        keep_headings=keep_headings,
        include_toc=include_toc,
        headings=headings,
        pin=pin,
        hints=hints,
        hint_focus=hint_focus,
        hint_mode=hint_mode,
    )

    lede_stats = stats(text, convert_word_names=convert_word_names)
    lede_metadata = metadata(text, backend="regex")
    lede_facts = key_facts(
        text,
        max_facts=max_facts,
        convert_word_names=convert_word_names,
        hints=hints,
        hint_focus=hint_focus,
        hint_mode=hint_mode,
    )

    spacy_metadata = None
    spacy_facts = ()
    spacy_phrases = ()
    if spacy_available:
        try:
            spacy_metadata = metadata(text, backend="spacy")
            spacy_facts = correlate_facts(text, backend="spacy")
            spacy_phrases = phrases(text, backend="spacy")
        except (ImportError, TypeError):
            if backend == "spacy":
                raise

    attributes = _extract_attributes(text)
    fact_records = _build_fact_records(attributes, lede_stats, spacy_facts)
    promotion_candidates = _build_promotion_candidates(attributes)
    search_text = _build_search_text(
        summary.summary,
        lede_facts,
        attributes,
        lede_metadata,
        spacy_metadata,
        spacy_phrases,
    )

    return ReadableReport(
        summary=summary,
        key_facts=lede_facts,
        stats=lede_stats,
        metadata=lede_metadata,
        spacy_metadata=spacy_metadata,
        spacy_facts=spacy_facts,
        spacy_phrases=spacy_phrases,
        attributes=attributes,
        fact_records=fact_records,
        promotion_candidates=promotion_candidates,
        search_text=search_text,
    )
