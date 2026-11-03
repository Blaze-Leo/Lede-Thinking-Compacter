"""lede CLI.

Usage:
  lede [FILE] --mode MODE [OPTIONS]

Reads FILE or stdin, writes text, Markdown, or JSON to stdout.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any

from lede import brief, clean_text, extract_keyword, readable_report, set_default_backend, strip_think, summarize
from lede.extract import (
    correlate_facts,
    key_facts,
    metadata,
    outline,
    phrases,
    stats,
    toc,
    top_terms,
)
from lede.format import (
    format_extract,
    summary_markdown,
    to_json,
)


_SUMMARY_MODES = {"tfidf": "default", "summarize": "default", "default": "default", "coverage": "coverage", "legacy": "legacy"}
_EXTRACT_MODES = {
    "stats",
    "facts",
    "key_facts",
    "metadata",
    "outline",
    "toc",
    "phrases",
    "correlate_facts",
    "top_terms",
}
_ALL_MODES = sorted(
    set(_SUMMARY_MODES)
    | {
        "brief",
        "report",
        "keyword",
        "clean_text",
        "strip_think",
    }
    | _EXTRACT_MODES
)


def _read_input(path: str | None) -> str:
    # Force UTF-8 — relying on locale gives Windows users mojibake on
    # non-ASCII content and breaks the byte-identical-runtime claim
    # (the Rust CLI reads files as bytes and decodes UTF-8 explicitly).
    if path and path != "-":
        return Path(path).read_text(encoding="utf-8")
    if sys.stdin.isatty():
        print("Reading from stdin — paste text and press Ctrl-D (Ctrl-Z on Windows) to run, Ctrl-C to cancel.", file=sys.stderr)
    if hasattr(sys.stdin, "buffer"):
        return sys.stdin.buffer.read().decode("utf-8")
    return sys.stdin.read()


def _csv_items(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _parse_attach(values: list[str] | None) -> list[str] | None:
    items = _csv_items(values)
    if not items:
        return None
    if "all" in items:
        return ["stats", "outline", "metadata", "phrases", "correlated_facts"]
    return items


def _parse_hints(args: argparse.Namespace) -> list[str] | dict[str, float] | None:
    weighted: dict[str, float] = {}
    plain: list[str] = []

    plain.extend(_csv_items(args.hint))
    plain.extend(_csv_items(args.hints))

    for item in _csv_items(args.hint_weight):
        if "=" not in item:
            raise ValueError("--hint-weight entries must use TERM=WEIGHT")
        term, weight_s = item.split("=", 1)
        term = term.strip()
        if not term:
            raise ValueError("--hint-weight entries must include a non-empty term")
        try:
            weighted[term] = float(weight_s)
        except ValueError as e:
            raise ValueError(f"invalid --hint-weight value for {term!r}: {weight_s!r}") from e

    if weighted:
        for term in plain:
            weighted.setdefault(term, 1.0)
        hints: list[str] | dict[str, float] = weighted
    else:
        hints = plain

    if not hints:
        return None

    kinds = tuple(_csv_items(args.expand_hints))
    if kinds:
        try:
            from lede_spacy import expand_hints
        except ImportError as e:
            raise ImportError(
                "hint expansion requires lede-spacy. Install it and a spaCy model:\n"
                "    pip install lede-spacy\n"
                "    python -m spacy download en_core_web_sm"
            ) from e
        hints = expand_hints(
            hints,
            kinds=kinds,
            top_k=args.expand_top_k,
            expand_weight=args.expand_weight,
        )

    return hints


def _configure_backend(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.spacy_model:
        os.environ["LEDE_SPACY_MODEL"] = args.spacy_model
    if args.vector_model:
        os.environ["LEDE_SPACY_VECTOR_MODEL"] = args.vector_model

    if args.backend in ("spacy", "auto") or args.use_spacy:
        try:
            import lede_spacy  # noqa: F401  # side-effect: registers backend
        except ImportError:
            if args.backend == "spacy" or args.use_spacy:
                parser.error(
                    "spaCy backend requested but lede-spacy is not installed. "
                    "Install with: pip install lede-spacy && python -m spacy download en_core_web_sm"
                )
            # backend=auto can safely fall back to regex.

    if args.use_spacy and args.backend is None:
        args.backend = "spacy"

    if args.backend is not None:
        try:
            set_default_backend(args.backend)
        except ValueError as e:
            parser.error(str(e))

    if args.warmup_spacy:
        try:
            from lede_spacy import warmup
        except ImportError:
            parser.error(
                "--warmup-spacy requires lede-spacy. Install with: pip install lede-spacy"
            )
        try:
            warmup()
        except ImportError as e:
            parser.error(str(e))


def _emit(value: Any, output: str) -> str:
    if output == "json":
        return to_json(value)
    if isinstance(value, str):
        return value
    return str(value)


def _render_extract(mode: str, text: str, args: argparse.Namespace, hints) -> Any:
    backend = args.backend
    if mode == "stats":
        return stats(text, convert_word_names=args.convert_word_names)
    if mode in ("facts", "key_facts"):
        return key_facts(
            text,
            max_facts=args.max_facts or 10,
            convert_word_names=args.convert_word_names,
            hints=hints,
            hint_focus=args.hint_focus,
            hint_mode=args.hint_mode,
        )
    if mode == "metadata":
        return metadata(text, backend=backend)
    if mode == "outline":
        return outline(text)
    if mode == "toc":
        return toc(text)
    if mode == "phrases":
        return phrases(
            text,
            keywords=args.keywords or None,
            backend=backend,
            hints=hints,
            hint_focus=args.hint_focus,
            hint_mode=args.hint_mode,
        )
    if mode == "correlate_facts":
        return correlate_facts(
            text,
            backend=backend,
            convert_word_names=args.convert_word_names,
            hints=hints,
            hint_focus=args.hint_focus,
            hint_mode=args.hint_mode,
        )
    if mode == "top_terms":
        kinds = tuple(_csv_items(args.kinds)) or ("words", "phrases")
        return top_terms(
            text,
            n=args.top,
            kinds=kinds,
            with_scores=args.scores or args.output == "json",
            hints=hints,
            hint_focus=args.hint_focus,
            hint_mode=args.hint_mode,
        )
    raise AssertionError(mode)


def _format_extract(mode: str, value: Any, output: str, scores: bool = False) -> str:
    return format_extract(mode, value, output=output, scores=scores)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lede",
        description="Deterministic extractive summarization and fact extraction.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Input file path. Reads stdin if omitted.",
    )
    parser.add_argument(
        "--mode",
        choices=_ALL_MODES,
        default="tfidf",
        help=(
            "Operation to run. Summary aliases: tfidf, summarize, default, "
            "coverage, legacy. Extraction modes: stats, facts/key_facts, "
            "metadata, outline, toc, phrases, correlate_facts, top_terms. "
            "Other modes: report, brief, keyword, clean_text, strip_think."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Character budget for summary modes (default: 500; report defaults to 2000 unless set).",
    )
    parser.add_argument(
        "--output",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--attach",
        action="append",
        help=(
            "Attach structured fields to summary output. Repeat or comma-separate. "
            "Choices: stats, outline, metadata, phrases, correlated_facts, all."
        ),
    )
    parser.add_argument(
        "--keywords",
        default="",
        help="Space-separated keywords for keyword mode, or phrase keyword inclusion.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of sentences/terms to return in keyword or top_terms mode (default: 10).",
    )
    parser.add_argument(
        "--max-facts",
        type=int,
        default=None,
        help="Maximum facts for brief/key_facts/report mode (defaults: 10, report: 40).",
    )
    parser.add_argument(
        "--format",
        choices=["string", "markdown", "dict"],
        default=None,
        help="brief() format override. Usually prefer --output.",
    )
    parser.add_argument(
        "--include-phrases",
        action="store_true",
        help="Include phrases in brief output.",
    )
    parser.add_argument(
        "--convert-word-names",
        action="store_true",
        help="Recognize spelled-out numbers where supported; requires lede[wordforms].",
    )
    parser.add_argument(
        "--hint",
        action="append",
        help="Hint term or comma-separated hint terms. Repeatable.",
    )
    parser.add_argument(
        "--hints",
        action="append",
        help="Alias for --hint; accepts comma-separated terms.",
    )
    parser.add_argument(
        "--hint-weight",
        action="append",
        help="Weighted hint as TERM=WEIGHT. Repeatable; comma-separated entries allowed.",
    )
    parser.add_argument(
        "--hint-focus",
        type=float,
        default=0.7,
        help="Fraction of budget reserved for hint-matching candidates (default: 0.7).",
    )
    parser.add_argument(
        "--hint-mode",
        choices=["soft", "hard"],
        default="soft",
        help="Hint behavior: soft bias or hard filter for hint pool (default: soft).",
    )
    parser.add_argument(
        "--expand-hints",
        action="append",
        help="Expand hints with lede-spacy. Comma choices: lemma, synonyms, similar.",
    )
    parser.add_argument(
        "--expand-top-k",
        type=int,
        default=5,
        help="Max synonym/similar expansion terms per hint (default: 5).",
    )
    parser.add_argument(
        "--expand-weight",
        type=float,
        default=0.5,
        help="Expansion weight multiplier for weighted hints (default: 0.5).",
    )
    parser.add_argument(
        "--keep-headings",
        action="store_true",
        help="Weave detected or supplied headings into summary output.",
    )
    parser.add_argument(
        "--include-toc",
        action="store_true",
        help="Prepend a table of contents to summary output.",
    )
    parser.add_argument(
        "--pin",
        action="append",
        help="Line to pin before summary output. Repeatable.",
    )
    parser.add_argument(
        "--heading",
        action="append",
        help="Caller-supplied heading line. Repeatable; replaces auto-detection when used.",
    )
    parser.add_argument(
        "--headings-file",
        help="UTF-8 file containing one caller-supplied heading per line.",
    )
    parser.add_argument(
        "--backend",
        choices=["regex", "spacy", "auto"],
        default=None,
        help="Backend for metadata/phrases/correlate_facts and summary attachments.",
    )
    parser.add_argument(
        "--use-spacy",
        action="store_true",
        help="Shortcut for --backend spacy; imports lede_spacy to register the backend.",
    )
    parser.add_argument(
        "--spacy-model",
        help="spaCy model name for lede-spacy (sets LEDE_SPACY_MODEL).",
    )
    parser.add_argument(
        "--vector-model",
        help="spaCy vector model for --expand-hints similar (sets LEDE_SPACY_VECTOR_MODEL).",
    )
    parser.add_argument(
        "--warmup-spacy",
        action="store_true",
        help="Pre-load the configured spaCy model before processing.",
    )
    parser.add_argument(
        "--kinds",
        action="append",
        help="top_terms candidate kinds. Repeat or comma-separate: words, phrases.",
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Include scores in top_terms text/markdown output.",
    )
    args = parser.parse_args(argv)

    _configure_backend(args, parser)

    try:
        hints = _parse_hints(args)
    except (ImportError, RuntimeError, ValueError) as e:
        parser.error(str(e))

    text = _read_input(args.path)
    output = ""

    if args.mode in _SUMMARY_MODES:
        headings = list(args.heading or [])
        if args.headings_file:
            headings.extend(
                line.strip()
                for line in Path(args.headings_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        result = summarize(
            text,
            max_length=args.max_chars,
            mode=_SUMMARY_MODES[args.mode],
            attach=_parse_attach(args.attach),
            hints=hints,
            hint_focus=args.hint_focus,
            hint_mode=args.hint_mode,
            keep_headings=args.keep_headings,
            include_toc=args.include_toc,
            pin=args.pin,
            headings=headings or None,
        )
        if args.output == "json":
            output = _emit(result, "json")
        elif args.output == "markdown":
            output = summary_markdown(result)
        else:
            output = result.summary
    elif args.mode == "brief":
        fmt = args.format
        if fmt is None:
            fmt = "dict" if args.output == "json" else ("markdown" if args.output == "markdown" else "string")
        value = brief(
            text,
            max_facts=args.max_facts or 10,
            include_phrases=args.include_phrases,
            convert_word_names=args.convert_word_names,
            format=fmt,
            hints=hints,
            hint_focus=args.hint_focus,
            hint_mode=args.hint_mode,
        )
        output = _emit(value, "json") if args.output == "json" else str(value)
    elif args.mode == "report":
        headings = list(args.heading or [])
        if args.headings_file:
            headings.extend(
                line.strip()
                for line in Path(args.headings_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        try:
            value = readable_report(
                text,
                max_length=args.max_chars if args.max_chars != 500 else 2000,
                max_facts=args.max_facts or 40,
                backend=args.backend or "regex",
                keep_headings=True if not args.keep_headings else args.keep_headings,
                include_toc=True if not args.include_toc else args.include_toc,
                headings=headings or None,
                pin=args.pin,
                hints=hints,
                hint_focus=args.hint_focus,
                hint_mode=args.hint_mode,
                convert_word_names=args.convert_word_names,
            )
        except (ImportError, TypeError, ValueError, RuntimeError) as e:
            parser.error(str(e))
        if args.output == "json":
            output = value.to_json()
        elif args.output == "markdown":
            output = value.to_markdown()
        else:
            output = value.to_text()
    elif args.mode == "keyword":
        if not args.keywords:
            parser.error("--mode keyword requires --keywords")
        output = extract_keyword(text, args.keywords, num_sentences=args.top)
    elif args.mode == "clean_text":
        output = clean_text(text)
    elif args.mode == "strip_think":
        output = strip_think(text)
    elif args.mode in _EXTRACT_MODES:
        try:
            value = _render_extract(args.mode, text, args, hints)
        except (ImportError, TypeError, ValueError, RuntimeError) as e:
            parser.error(str(e))
        output = _format_extract(args.mode, value, args.output, scores=args.scores or args.output == "json")
    else:  # pragma: no cover
        parser.error(f"unknown mode: {args.mode}")

    try:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
