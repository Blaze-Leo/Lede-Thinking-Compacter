"""At-a-glance document brief — composes summarize + key_facts + toc.

Top-level convenience primitive for "reader brief" use cases: email
digests, file-browser previews, ingest-pipeline pre-summarization.
Agnostic of document type — no per-doc heuristics. Callers who want
different composition should call the underlying primitives directly.

Mirrors rust/src/brief.rs. Keep the three output formats (string,
markdown, dict) byte-identical across Python and Rust for the regex
backend.
"""
from __future__ import annotations

from .tfidf import summarize
from .extract.key_facts import key_facts
from .extract.outline import toc
from .extract.phrases import phrases

# Auto-detect the wordforms extra at import time. When available,
# brief() forwards convert_word_names=True to its internal key_facts()
# call so spelled-out numbers ("five thousand documents", "twelve lines")
# surface in the key facts section. Mirrors the same pattern used in
# benchmarks/extraction_eval.py::_STATS_WORDFORMS.
try:
    import text_to_num  # noqa: F401

    _HAS_WORDFORMS = True
except ImportError:
    _HAS_WORDFORMS = False


# Overview budget clamps. Floor keeps very short inputs readable; ceiling
# keeps very long inputs actually brief (a 30 KB doc at 0.35 would produce
# a 10 KB "overview" which defeats the purpose).
_OVERVIEW_MIN_FRAC = 0.05
_OVERVIEW_MAX_FRAC = 0.50
_OVERVIEW_MIN_CHARS = 250
_OVERVIEW_MAX_CHARS = 1500


def brief(
    text: str,
    *,
    overview_max: float = 0.35,
    max_facts: int = 10,
    include_phrases: bool = False,
    convert_word_names: bool | None = None,
    format: str = "string",
    hints: list[str] | dict[str, float] | None = None,
    hint_focus: float = 0.7,
    hint_mode: str = "soft",
) -> str | dict:
    """Produce an at-a-glance brief of a document.

    Composes ``summarize()`` (overview) + ``extract.key_facts()`` +
    ``extract.toc()`` into a single caller-friendly artifact. Optional
    ``extract.phrases()`` via ``include_phrases=True``.

    Args:
        text: input document text.
        overview_max: fraction of source length to budget for the overview.
            Clamped to ``[0.05, 0.50]``. Default 0.35. The resolved
            char budget is further clamped to ``[250, 1500]`` so short
            docs still get a readable overview and long docs stay brief.
        max_facts: cap on key-facts sentences. Default 10.
        include_phrases: when True, append a key-phrases section (regex
            backend). Default False.
        convert_word_names: forward to ``key_facts()`` so spelled-out
            numbers ("five thousand documents") surface in the
            key-facts section. ``None`` (default) auto-detects whether
            ``text2num`` is importable. Pass ``True`` / ``False``
            explicitly to lock the behavior — useful when you need
            output to match a Rust binary built with or without the
            ``wordforms`` cargo feature regardless of which Python
            extras happen to be installed.
        format: output shape. One of:
            - ``"string"`` (default) — plain text with section labels.
            - ``"markdown"`` — ``##`` headers + bullet lists.
            - ``"dict"`` — structured dict with overview/key_facts/toc/phrases.
        hints: optional list or dict of keywords to bias toward in the overview,
            key facts, and phrases sections. When a list, each hint is weighted 1.0.
            When a dict, keys are hints and values are numeric weights. Default None.
        hint_focus: fraction of each primitive's budget reserved for hint-matching
            content. Range [0.0, 1.0]. Default 0.7. Forwarded to summarize(),
            key_facts(), and phrases().
        hint_mode: biasing strategy. One of ``"soft"`` (default, reorder and score)
            or ``"hard"`` (only hint-matching content up to the quota). Forwarded
            to summarize(), key_facts(), and phrases().

    Returns:
        ``str`` for ``"string"`` / ``"markdown"`` formats, ``dict`` for
        ``"dict"`` format.

    Raises:
        ValueError: when ``format`` is not one of the three supported values, or
            when hint arguments are invalid (out of range, invalid mode).

    Hint biasing (v0.4):
        ``hints``, ``hint_focus``, and ``hint_mode`` are forwarded unchanged to
        the internal ``summarize()``, ``key_facts()``, and ``phrases()`` calls.
        Each primitive applies its own budget split independently using the same
        ``hint_focus`` value. When ``hints`` is None (the default), output is
        byte-identical to v0.3.0.

        Matching is case-insensitive and word-boundary-delimited; no Unicode
        normalization or stemming in core. For lemma/synonym expansion, compose
        with ``lede_spacy.expand_hints`` before calling ``brief()``.

        See ``lede.summarize`` and docs/REFERENCE.md "Hint biasing" for full
        argument semantics and validation rules.

    Notes on cross-runtime parity:
        Python's auto-detect (``convert_word_names=None``) is at import
        time; Rust's equivalent is the compile-time ``wordforms`` cargo
        feature. The two can disagree silently when (a) the Python
        process has ``text2num`` installed for an unrelated tool and
        (b) the Rust binary was built without ``--features wordforms``
        — same input, different brief bytes. Pass an explicit
        ``convert_word_names=True`` (or ``False``) on both sides to
        lock parity.
    """
    # Clamp overview_max to sane fraction bounds.
    overview_max = max(_OVERVIEW_MIN_FRAC, min(_OVERVIEW_MAX_FRAC, overview_max))
    budget = int(len(text) * overview_max)
    budget = max(_OVERVIEW_MIN_CHARS, min(_OVERVIEW_MAX_CHARS, budget))

    overview_result = summarize(
        text,
        max_length=budget,
        hints=hints,
        hint_focus=hint_focus,
        hint_mode=hint_mode,
    )
    overview_text = overview_result.summary.rstrip()

    use_wordforms = _HAS_WORDFORMS if convert_word_names is None else convert_word_names
    facts = key_facts(
        text,
        max_facts=max_facts,
        convert_word_names=use_wordforms,
        hints=hints,
        hint_focus=hint_focus,
        hint_mode=hint_mode,
    )

    sections = toc(text)

    phrases_list: tuple[str, ...] = (
        phrases(text, hints=hints, hint_focus=hint_focus, hint_mode=hint_mode)
        if include_phrases
        else ()
    )

    if format == "dict":
        return {
            "overview": overview_text,
            "key_facts": list(facts),
            "toc": list(sections),
            "phrases": list(phrases_list) if include_phrases else None,
        }
    if format == "markdown":
        parts: list[str] = ["## Overview\n", overview_text, ""]
        if facts:
            parts.append("\n## Key facts\n")
            for f in facts:
                parts.append(f"- {f}")
            parts.append("")
        if sections:
            parts.append("\n## Also in this doc\n")
            for s in sections:
                parts.append(f"- {s}")
            parts.append("")
        if include_phrases and phrases_list:
            parts.append("\n## Key phrases\n")
            parts.append("  ·  ".join(phrases_list))
            parts.append("")
        return "\n".join(parts)
    if format == "string":
        parts = ["Overview:", overview_text, ""]
        if facts:
            parts.append("Key facts:")
            for f in facts:
                parts.append(f"  - {f}")
            parts.append("")
        if sections:
            parts.append("Also in this doc:")
            for s in sections:
                parts.append(f"  - {s}")
            parts.append("")
        if include_phrases and phrases_list:
            parts.append("Key phrases:")
            parts.append("  " + ", ".join(phrases_list))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"
    raise ValueError(
        f"format must be 'string', 'markdown', or 'dict'; got {format!r}"
    )
