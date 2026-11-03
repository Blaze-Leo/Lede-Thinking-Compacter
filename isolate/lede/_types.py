"""v0.2.0 return types (frozen dataclasses)."""
from dataclasses import dataclass
from typing import NamedTuple


class TermScore(NamedTuple):
    """One scored term from ``extract.top_terms(with_scores=True)`` (v0.4.1).

    A ``NamedTuple`` (not a frozen dataclass like the other types here) so
    downstream consumers can both unpack positionally
    (``for term, score, kind in result``) and access by name
    (``ts.term`` / ``ts.score`` / ``ts.kind``).

    Fields:
        term: the word or phrase.
        score: per-kind-normalized salience in ``[0.0, 1.0]`` (plain mode);
            in soft-hint mode the ``hint_bonus`` is added on top so a
            matching term may exceed 1.0. Scores are normalized within each
            kind independently — a word at 1.0 and a phrase at 1.0 are each
            top-of-their-kind, not equal on a shared cross-kind scale.
        kind: ``"word"`` (single token, TF-IDF scored) or ``"phrase"``
            (multi-word n-gram, repetition×length scored).
    """

    term: str
    score: float
    kind: str


@dataclass(frozen=True)
class Stat:
    value: str
    unit: str
    phrase: str
    context_sentence: str
    stat_type: str  # "money" | "percent" | "count" | "date" | "duration"


@dataclass(frozen=True)
class Section:
    depth: int
    name: str
    representative_sentence: str


@dataclass(frozen=True)
class Metadata:
    dates: tuple[str, ...] = ()
    amounts: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()  # populated only when lede[ner] installed (Task 9)


@dataclass(frozen=True)
class PhraseFact:
    entity: str
    number: str
    polarity: str  # "absolute" | "growth" | "decline" | "unknown"
    sentence: str


@dataclass(frozen=True)
class ReportAttribute:
    key: str
    label: str
    value: str
    value_type: str
    source: str
    evidence: str
    confidence: float = 1.0


@dataclass(frozen=True)
class FactRecord:
    subject: str
    predicate: str
    object: str
    fact_type: str
    evidence: str
    confidence: float = 1.0


@dataclass(frozen=True)
class PromotionCandidate:
    path: str
    key: str
    value_type: str
    promote: bool = True
    confidence: float = 1.0


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    stats: tuple[Stat, ...] | None = None
    outline: tuple[Section, ...] | None = None
    metadata: Metadata | None = None
    phrases: tuple[str, ...] | None = None
    correlated_facts: tuple[PhraseFact, ...] | None = None
    pinned_headings: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.summary

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        from .format import to_data

        return to_data(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return a JSON string representation."""
        from .format import to_json

        return to_json(self, indent=indent)

    def to_markdown(self) -> str:
        """Return a Markdown rendering of the summary and attachments."""
        from .format import summary_markdown

        return summary_markdown(self)


@dataclass(frozen=True)
class ReadableReport:
    """Combined human-readable lede + spaCy report."""

    summary: SummaryResult
    key_facts: tuple[str, ...] = ()
    stats: tuple[Stat, ...] = ()
    metadata: Metadata = Metadata()
    spacy_metadata: Metadata | None = None
    spacy_facts: tuple[PhraseFact, ...] = ()
    spacy_phrases: tuple[str, ...] = ()
    attributes: tuple[ReportAttribute, ...] = ()
    fact_records: tuple[FactRecord, ...] = ()
    promotion_candidates: tuple[PromotionCandidate, ...] = ()
    search_text: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        from .format import to_data

        data = to_data(self)
        data["attributes"] = {
            attr.key: {
                "label": attr.label,
                "value": attr.value,
                "type": attr.value_type,
                "source": attr.source,
                "evidence": attr.evidence,
                "confidence": attr.confidence,
            }
            for attr in self.attributes
        }
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return a JSON string representation."""
        from .format import to_json

        return to_json(self, indent=indent)

    def to_text(self) -> str:
        """Return a readable plain-text rendering."""
        from .format import report_text

        return report_text(self)

    def to_markdown(self) -> str:
        """Return a readable Markdown rendering."""
        from .format import report_markdown

        return report_markdown(self)
