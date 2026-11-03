"""Shared pipeline state for enrichment primitives.

When summarize(attach=[...]) is called, the summary step and each
enrichment primitive each parse the input independently for now. T4
lands the bare pipeline; future tasks may thread shared state to cut
latency under the 250ms SC-B budget.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Pipeline:
    text: str
    sentences: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "Pipeline":
        # Late import to avoid a circular dep with tfidf.py importing Pipeline
        from ..sentences import split_sentences
        return cls(text=text, sentences=tuple(split_sentences(text)))
