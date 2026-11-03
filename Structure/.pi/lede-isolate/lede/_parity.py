"""Canonical text formatters for v0.2 extract primitives — used by the
cross-runtime parity walker (`rust/tests/fixtures.rs`) and the Python
generator (`benchmarks/gen_parity_fixtures.py`).

Each formatter takes the primitive's structured output and emits a
deterministic, line-oriented text representation. Python and Rust
have byte-identical implementations so a fixture generated from
Python output can be byte-compared against the Rust output for the
same input.

Format choices:
- One record per line.
- Fields within a record separated by ` | ` (space-pipe-space).
- Trailing newline at end of file (mirror of how Python's `\\n`.join
  + final write_text behaves).
- Empty-output → empty file (no trailing newline either).

Mirrors: rust/src/_parity.rs.
"""
from __future__ import annotations

from ._types import Metadata, PhraseFact, Section, Stat


def _join_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def format_stats(stats: tuple[Stat, ...]) -> str:
    lines = [
        f"{s.stat_type} | {s.value} | {s.unit} | {s.phrase} | {s.context_sentence}"
        for s in stats
    ]
    return _join_lines(lines)


def format_outline(outline: tuple[Section, ...]) -> str:
    lines = [
        f"{sec.depth} | {sec.name} | {sec.representative_sentence}"
        for sec in outline
    ]
    return _join_lines(lines)


def format_toc(toc: tuple[str, ...]) -> str:
    return _join_lines(list(toc))


def format_metadata(md: Metadata) -> str:
    return _join_lines([
        "DATES: " + ", ".join(md.dates),
        "AMOUNTS: " + ", ".join(md.amounts),
        "URLS: " + ", ".join(md.urls),
        "ENTITIES: " + ", ".join(md.entities),
    ])


def format_phrases(phrases: tuple[str, ...]) -> str:
    return _join_lines(list(phrases))


def format_top_terms(terms: tuple[str, ...]) -> str:
    return _join_lines(list(terms))


def format_key_facts(facts: tuple[str, ...]) -> str:
    return _join_lines(list(facts))


def format_correlate_facts(facts: tuple[PhraseFact, ...]) -> str:
    lines = [
        f"{f.entity} | {f.number} | {f.polarity} | {f.sentence}"
        for f in facts
    ]
    return _join_lines(lines)
