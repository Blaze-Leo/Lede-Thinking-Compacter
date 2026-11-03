"""Output formatting helpers for lede results.

The core primitives stay data-first: ``summarize()`` returns
``SummaryResult`` and extraction primitives return tuples/dataclasses.
This module provides the presentation layer used by both the public API
and the CLI.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any


def to_data(value: Any) -> Any:
    """Convert lede dataclasses / NamedTuples into JSON-serializable data."""
    if dataclasses.is_dataclass(value):
        return {field.name: to_data(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {name: to_data(getattr(value, name)) for name in value._fields}
    if isinstance(value, (list, tuple)):
        return [to_data(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_data(v) for k, v in value.items()}
    return value


def to_json(value: Any, *, indent: int | None = 2) -> str:
    """Serialize lede data to JSON with UTF-8 characters preserved."""
    if hasattr(value, "to_dict"):
        return json.dumps(value.to_dict(), ensure_ascii=False, indent=indent)
    return json.dumps(to_data(value), ensure_ascii=False, indent=indent)


def markdown_list(title: str, rows: list[str]) -> str:
    """Render a simple Markdown section with bullet rows."""
    if not rows:
        return f"## {title}\n"
    return "\n".join([f"## {title}", "", *[f"- {row}" for row in rows]])


def stats_text(rows) -> str:
    """Render ``extract.stats`` rows as tab-separated text."""
    return "\n".join(
        f"{s.stat_type}\t{s.value}\t{s.unit}\t{s.context_sentence}"
        for s in rows
    )


def stats_markdown(rows) -> str:
    """Render ``extract.stats`` rows as a Markdown table."""
    if not rows:
        return "## Stats\n"
    lines = ["## Stats", "", "| Type | Value | Unit | Context |", "|---|---|---|---|"]
    for s in rows:
        lines.append(f"| {s.stat_type} | {s.value} | {s.unit} | {s.context_sentence} |")
    return "\n".join(lines)


def metadata_text(m) -> str:
    """Render ``extract.metadata`` as labeled text lists."""
    parts: list[str] = []
    for label, values in (
        ("Dates", m.dates),
        ("Amounts", m.amounts),
        ("URLs", m.urls),
        ("Entities", m.entities),
    ):
        if values:
            parts.append(f"{label}:")
            parts.extend(f"  - {v}" for v in values)
    return "\n".join(parts)


def metadata_markdown(m) -> str:
    """Render ``extract.metadata`` as Markdown sections."""
    lines = ["## Metadata"]
    for label, values in (
        ("Dates", m.dates),
        ("Amounts", m.amounts),
        ("URLs", m.urls),
        ("Entities", m.entities),
    ):
        if values:
            lines.extend(["", f"### {label}", "", *[f"- {v}" for v in values]])
    return "\n".join(lines)


def outline_text(rows) -> str:
    """Render ``extract.outline`` rows as indented text."""
    return "\n".join(
        f"{'  ' * max(s.depth - 1, 0)}- {s.name}: {s.representative_sentence}"
        for s in rows
    )


def outline_markdown(rows) -> str:
    """Render ``extract.outline`` rows as Markdown bullets."""
    return "\n".join(
        [
            "## Outline",
            "",
            *[
                f"{'  ' * max(s.depth - 1, 0)}- **{s.name}**: {s.representative_sentence}"
                for s in rows
            ],
        ]
    )


def correlate_text(rows) -> str:
    """Render ``extract.correlate_facts`` rows as grouped plain text.

    JSON keeps the row-oriented shape for machines. Plain text groups by
    source sentence so long structured-document blocks are not repeated once
    per entity-number pair.
    """
    if not rows:
        return ""
    grouped: dict[str, list[str]] = {}
    for r in rows:
        grouped.setdefault(r.sentence, []).append(
            f"{r.entity}\t{r.number}\t{r.polarity}"
        )

    parts: list[str] = []
    for sentence, facts in grouped.items():
        parts.append(sentence)
        parts.extend(f"  {fact}" for fact in facts)
        parts.append("")
    return "\n".join(parts).rstrip()


def correlate_markdown(rows) -> str:
    """Render ``extract.correlate_facts`` rows as Markdown bullets."""
    if not rows:
        return "## Correlated Facts\n"
    grouped: dict[str, list[str]] = {}
    for r in rows:
        grouped.setdefault(r.sentence, []).append(
            f"`{r.entity}` -> `{r.number}` ({r.polarity})"
        )

    lines = ["## Correlated Facts"]
    for sentence, facts in grouped.items():
        lines.extend(["", f"- {sentence}"])
        for fact in facts:
            lines.append(f"  - {fact}")
    return "\n".join(lines)


def summary_markdown(result) -> str:
    """Render a ``SummaryResult`` and any attachments as Markdown."""
    lines = ["## Summary", "", result.summary.rstrip()]
    if result.stats:
        lines.extend(["", stats_markdown(result.stats)])
    if result.outline:
        lines.extend(["", outline_markdown(result.outline)])
    if result.metadata:
        lines.extend(["", metadata_markdown(result.metadata)])
    if result.phrases:
        lines.extend(["", markdown_list("Phrases", list(result.phrases))])
    if result.correlated_facts:
        lines.extend(["", correlate_markdown(result.correlated_facts)])
    return "\n".join(lines)


def _stat_brief(row) -> str:
    unit = f" {row.unit}" if row.unit and row.unit != row.stat_type else ""
    return f"{row.stat_type}: {row.value}{unit} - {_brief_context(row.context_sentence, max_chars=140)}"


def _brief_context(text: str, *, max_chars: int = 280) -> str:
    """Return a single-line context snippet for human-readable reports."""
    one_line = " ".join(str(text).split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 1].rstrip() + "..."


def _compact_fact_records(report, *, limit: int = 12):
    """Return non-duplicative fact records for compact human report output."""
    rows = []
    seen: set[tuple[str, str, str, str]] = set()
    attribute_keys = {attr.key for attr in report.attributes}
    for fact in report.fact_records:
        if fact.fact_type == "attribute":
            continue
        if fact.fact_type == "numeric" and fact.predicate in attribute_keys and fact.object:
            continue
        key = (fact.subject, fact.predicate, fact.object, fact.fact_type)
        if key in seen:
            continue
        seen.add(key)
        rows.append(fact)
        if len(rows) >= limit:
            break
    return rows


def report_text(report) -> str:
    """Render a combined ``ReadableReport`` as plain text."""
    parts: list[str] = ["Summary", "=======", report.summary.summary.rstrip()]

    parts.extend(["", "Facts and Important Details", "==========================="])

    if report.key_facts:
        parts.extend(["", "Lede key facts:"])
        parts.extend(f"- {_brief_context(fact, max_chars=360)}" for fact in report.key_facts)

    if report.stats:
        parts.extend(["", "Lede numeric/date facts:"])
        parts.extend(f"- {_stat_brief(stat)}" for stat in report.stats)

    if report.metadata and (report.metadata.dates or report.metadata.amounts or report.metadata.urls):
        parts.extend(["", "Lede metadata:"])
        if report.metadata.dates:
            parts.append("  Dates: " + ", ".join(report.metadata.dates))
        if report.metadata.amounts:
            parts.append("  Amounts: " + ", ".join(report.metadata.amounts))
        if report.metadata.urls:
            parts.append("  URLs: " + ", ".join(report.metadata.urls))

    if report.attributes:
        parts.extend(["", "Structured metadata candidates:"])
        for attr in report.attributes:
            parts.append(
                f"- {attr.key}: {attr.value} "
                f"({attr.value_type}, {attr.source}, {attr.confidence:.2f})"
            )

    compact_facts = _compact_fact_records(report)
    if compact_facts:
        parts.extend(["", "Important detail records:"])
        for fact in compact_facts:
            parts.append(
                f"- {fact.subject}: {fact.predicate} = {fact.object} "
                f"({fact.fact_type})"
            )

    if report.spacy_metadata and report.spacy_metadata.entities:
        parts.extend(["", "spaCy entities:"])
        parts.extend(f"- {entity}" for entity in report.spacy_metadata.entities)

    if report.spacy_phrases:
        parts.extend(["", "spaCy noun phrases:"])
        parts.extend(f"- {phrase}" for phrase in report.spacy_phrases)

    if report.spacy_facts:
        parts.extend(["", "spaCy entity-fact links:"])
        grouped: dict[str, list[str]] = {}
        for fact in report.spacy_facts:
            grouped.setdefault(fact.sentence, []).append(
                f"{fact.entity}\t{fact.number}\t{fact.polarity}"
            )
        for sentence, facts in grouped.items():
            parts.append(_brief_context(sentence))
            parts.extend(f"  {fact}" for fact in facts)
            parts.append("")

    return "\n".join(parts).rstrip()


def report_markdown(report) -> str:
    """Render a combined ``ReadableReport`` as Markdown."""
    lines: list[str] = ["## Summary", "", report.summary.summary.rstrip()]
    lines.extend(["", "## Facts and Important Details"])

    if report.key_facts:
        lines.extend(["", "### Lede Key Facts", ""])
        lines.extend(f"- {_brief_context(fact, max_chars=360)}" for fact in report.key_facts)

    if report.stats:
        lines.extend(["", "### Lede Numeric/Date Facts", ""])
        lines.extend(f"- {_stat_brief(stat)}" for stat in report.stats)

    if report.metadata and (report.metadata.dates or report.metadata.amounts or report.metadata.urls):
        lines.extend(["", "### Lede Metadata", ""])
        if report.metadata.dates:
            lines.append("- **Dates:** " + ", ".join(report.metadata.dates))
        if report.metadata.amounts:
            lines.append("- **Amounts:** " + ", ".join(report.metadata.amounts))
        if report.metadata.urls:
            lines.append("- **URLs:** " + ", ".join(report.metadata.urls))

    if report.attributes:
        lines.extend(["", "### Structured Metadata Candidates", ""])
        for attr in report.attributes:
            lines.append(
                f"- **{attr.key}:** {attr.value} "
                f"({attr.value_type}, {attr.source}, confidence {attr.confidence:.2f})"
            )

    compact_facts = _compact_fact_records(report)
    if compact_facts:
        lines.extend(["", "### Important Detail Records", ""])
        for fact in compact_facts:
            lines.append(
                f"- **{fact.subject}:** {fact.predicate} = {fact.object} "
                f"({fact.fact_type})"
            )

    if report.spacy_metadata and report.spacy_metadata.entities:
        lines.extend(["", "### spaCy Entities", ""])
        lines.extend(f"- {entity}" for entity in report.spacy_metadata.entities)

    if report.spacy_phrases:
        lines.extend(["", "### spaCy Noun Phrases", ""])
        lines.extend(f"- {phrase}" for phrase in report.spacy_phrases)

    if report.spacy_facts:
        lines.extend(["", "### spaCy Entity-Fact Links", ""])
        grouped: dict[str, list[str]] = {}
        for fact in report.spacy_facts:
            grouped.setdefault(fact.sentence, []).append(
                f"`{fact.entity}` -> `{fact.number}` ({fact.polarity})"
            )
        for sentence, facts in grouped.items():
            lines.append(f"- {_brief_context(sentence)}")
            lines.extend(f"  - {fact}" for fact in facts)

    return "\n".join(lines).rstrip()


def format_extract(
    kind: str,
    value: Any,
    *,
    output: str = "text",
    scores: bool = False,
) -> str:
    """Render a standalone extraction primitive result.

    Args:
        kind: primitive name, e.g. ``"stats"``, ``"metadata"``,
            ``"key_facts"``, ``"top_terms"``.
        value: result returned by the primitive.
        output: one of ``"text"``, ``"markdown"``, or ``"json"``.
        scores: for ``top_terms`` text/Markdown output, render
            ``TermScore`` records with score and kind.
    """
    if output == "json":
        return to_json(value)
    if output not in ("text", "markdown"):
        raise ValueError(f"output must be 'text', 'markdown', or 'json'; got {output!r}")

    if kind == "stats":
        return stats_markdown(value) if output == "markdown" else stats_text(value)
    if kind in ("facts", "key_facts"):
        rows = list(value)
        return markdown_list("Key Facts", rows) if output == "markdown" else "\n".join(rows)
    if kind == "metadata":
        return metadata_markdown(value) if output == "markdown" else metadata_text(value)
    if kind == "outline":
        return outline_markdown(value) if output == "markdown" else outline_text(value)
    if kind == "toc":
        rows = list(value)
        return markdown_list("Table of Contents", rows) if output == "markdown" else "\n".join(rows)
    if kind == "phrases":
        rows = list(value)
        return markdown_list("Phrases", rows) if output == "markdown" else "\n".join(rows)
    if kind == "correlate_facts":
        return correlate_markdown(value) if output == "markdown" else correlate_text(value)
    if kind == "top_terms":
        if scores:
            rows = [f"{t.term}\t{t.kind}\t{t.score:.6g}" for t in value]
            md_rows = [f"`{t.term}` ({t.kind}, {t.score:.3f})" for t in value]
        else:
            rows = [str(t) for t in value]
            md_rows = rows
        return markdown_list("Top Terms", md_rows) if output == "markdown" else "\n".join(rows)
    raise ValueError(f"unknown extraction kind: {kind!r}")


def format_result(value: Any, *, output: str = "text") -> str:
    """Render any lede value as ``"text"``, ``"markdown"``, or ``"json"``.

    ``SummaryResult`` values use their rich Markdown renderer. Other values
    fall back to ``str(value)`` for text/Markdown unless a specialized CLI
    formatter is used.
    """
    if output == "json":
        return to_json(value)
    if output == "markdown" and hasattr(value, "to_markdown"):
        return value.to_markdown()
    if output == "text" and hasattr(value, "to_text"):
        return value.to_text()
    if output == "text":
        return str(value)
    if output == "markdown":
        return str(value)
    raise ValueError(f"output must be 'text', 'markdown', or 'json'; got {output!r}")
