"""Structured extractive primitives for lede v0.2.

Each primitive can be called standalone:

    from lede.extract import stats, outline, metadata, phrases, correlate_facts
    s = stats(text)

Or via summarize(attach=[...]):

    from lede import summarize
    r = summarize(text, attach=["stats", "outline"])
    r.stats      # tuple[Stat, ...]
    r.outline    # tuple[Section, ...]

Stub primitives returning empty collections land in this task (T4).
Real implementations land in T6-T11.
"""
from .._types import Stat, Section, Metadata, PhraseFact, TermScore
from .stats import stats
from .outline import outline, toc
from .metadata import metadata
from .phrases import phrases
from .correlate import correlate_facts
from .key_facts import key_facts
from .top_terms import top_terms

__all__ = [
    "Stat", "Section", "Metadata", "PhraseFact", "TermScore",
    "stats", "outline", "toc", "metadata", "phrases", "correlate_facts",
    "key_facts", "top_terms",
]
