# isolate/ — Code Analysis

A file-by-file analysis of how the algorithm in `isolate/` works.

## System overview (from README.md)

`isolate/` is a self-contained blackbox that compresses an AI agent's chain-of-thought (CoT). Two CLIs:

- **`plain_compress.py`** — deterministic compressor. No LLM, no network. Pure stdlib + the bundled `lede/` sentence segmenter. Byte-identical across runs.
- **`llm_compress.py`** — runs the same compression, then makes exactly one `pi -p` call to resume work from the compressed thinking.

Both take raw thinking text in (via `--text` or stdin) and write text out to stdout.

The compression strategy (per README):
1. Strip agent scaffolding / pure filler first.
2. Use a **reasoning-event taxonomy** (13 labels: GAP, REJECTION, OBJECTION, REVISION, RECONSIDERATION, ACCEPTANCE, PROPOSAL, OBSERVATION, ACTION, SUPPORT, CONCLUSION, UNCERTAINTY, CONTRAST).
3. Reserve the **head** (task setup) and **tail** (current state / what's done / what's blocked).
4. Guarantee at least one kept sentence per event category present in the input.
5. Fill remaining budget with novelty-penalized (MMR) selection.
6. Deduplicate only near-identical sentences with compatible polarity (negation-aware) and modality.

The `lede/` package is a bundled dependency; the compressor reuses `lede.sentences.split_sentences`.

---

## File index

| File | Lines | Role |
|---|---|---|
| `isolate/plain_compress.py` | 1385 | Deterministic CoT compressor (core algorithm) |
| `isolate/llm_compress.py` | 143 | Compress + ONE `pi -p` call to resume |
| `isolate/lede/sentences.py` | 78 | Sentence segmenter (reused by compressor) |
| `isolate/lede/*.py` | ~3346 | Bundled lede package (segmentation, tfidf, extract, etc.) |
| `isolate/lede/extract/*.py` | ~1587 | Extraction subpackage (outline, key_facts, stats, ...) |

---

## Analysis

### `isolate/plain_compress.py` (1385 lines) — the core deterministic compressor

This is the heart of `isolate/`. It compresses one model-thinking text block with **no ML and no network** — pure stdlib + the bundled `lede.sentences.split_sentences` (with a regex fallback if `lede` isn't importable). Deterministic: same bytes in, same bytes out.

#### Pipeline (5 stages)

The module docstring states the pipeline; the code implements it exactly:

1. **Strip scaffolding + filler** — remove agent/extractor plumbing and pure conversational filler.
2. **Segment** — split into sentences via `lede` (or regex fallback).
3. **Classify** — multi-label, rule-based reasoning-event classification per sentence.
4. **Deduplicate** — merge only near-identical statements with compatible polarity/state.
5. **Select** — event coverage + novelty (MMR) + chronological head/tail preservation.

The docstring also makes a deliberate design decision explicit: it does **not** infer claim identity or derive claim status (that needs reliable coreference resolution). The optional JSON ledger therefore records sentence-level events tied to their original position, not speculative claim clusters.

#### Preprocessing regexes

- `_THINKING_HEADER` — matches `# Thinking N` headers; stripped via `.sub("\n", ...)` in `compress()`.
- `_META` — agent/extractor scaffolding: `AGENTS.md`, `eth_progress.md`, `progress log`, "the user is/said/says/told/pushing", "I'll write/say/mention/mark/keep", "let me be honest", "to be safe", "for this round", "what I should do", "the right move now". Applied at **line** level (drops whole lines) and again at **sentence** level.
- `_FILLER_ONLY` — pure conversational filler that is the *entire* sentence: `wait/okay/hmm/um` interjections plus "let me think", "give me a moment", "let's see", "thinking", etc. Only removed on `fullmatch` (so a real sentence containing "let me think" survives).
- `_TOKEN_RE` — tokenizer `[a-z0-9]+(?:'[a-z]+)?` (case-insensitive).
- `_STOP` — a hand-written stopword set for content-token extraction.

#### Negation and modality (for polarity-aware dedup)

- `_NEGATION` / `_NEGATION_EXCEPTIONS` — negation detection with carve-outs so "not only", "cannot rule out", "no problem" are not treated as negations.
- `_polarity(text)` ➜ `"NEGATIVE"` or `"POSITIVE"`.
- `_modality(text)` ➜ `"HEDGED"` (might/may/perhaps/uncertain) | `"EMPHATIC"` (definitely/certainly/clearly) | `"ASSERTED"`.

These feed the dedup key so that **"X works" is never collapsed into "X doesn't work"** (different polarity) or "X might work" (different modality).

#### Reasoning-event taxonomy (the 13 labels)

`_RULES: dict[str, tuple[re.Pattern, ...]]` — each label maps to a tuple of explicit regexes. A sentence can get **multiple** labels (multi-label). The labels and their intent:

| Label | Weight | Intent |
|---|---|---|
| GAP | 10 | Work that remains / information not yet verified |
| REJECTION | 10 | Explicit abandonment of an approach |
| ACCEPTANCE | 10 | Explicit selection / successful verification / working solution |
| REVISION | 9 | Corrections or changes to an existing conclusion |
| RECONSIDERATION | 9 | Reconsidering an earlier direction |
| OBJECTION | 9 | Technical failures and concrete objections |
| UNCERTAINTY | 9 | Explicit uncertainty (not any "might") |
| CONCLUSION | 9 | Explicit conclusions or decisions |
| PROPOSAL | 7 | Suggested / intended / planned work |
| OBSERVATION | 7 | Reasons, mechanisms, dependencies, causal explanations |
| ACTION | 6 | Actual operations already performed (distinct from PROPOSAL) |
| SUPPORT | 6 | Reported evidence / observations |
| CONTRAST | 5 | Alternatives, contrasts, changes in direction |

Note the docstring comment: patterns are **intentionally explicit** — they cover common coding-agent language without labeling every occurrence of words like "problem", "maybe", "might", or "I think". This is a precision-over-recall design: fewer false labels.

- `_LABEL_WEIGHT` — importance weight per label (used as the MMR relevance term).
- `_COVERAGE_ORDER` — priority order for the coverage guarantee (ACCEPTANCE, REJECTION, GAP first — the "what's the state" labels; CONTRAST last).

#### Classification

- `_tokens(text)` — set of lowercase tokens.
- `_content_tokens(text)` — tokens minus stopwords.
- `_classify(position, sentence)` ➜ `{position, text, labels, polarity, modality, score}` where `score = max(label weights, default=1)`. A sentence with no matching label still gets `score=1` (so it's selectable, just low-priority).

#### Similarity and deduplication

- `_jaccard(a, b)` — Jaccard similarity of content-token sets.
- `_dedup_key(event)` ➜ `(tuple(labels), polarity, modality)` — **only** compare statements with compatible reasoning states.
- `_deduplicate(events, threshold)` ➜ `(kept, removed_count)`. Walks in order; a sentence is a duplicate iff it shares **>= 3 content tokens** AND Jaccard >= threshold with an already-kept sentence that has the **same dedup key**. The "at least three shared content terms" rule means a single shared technical noun is never sufficient. Keeps the first occurrence.

#### Selection: coverage-first + incremental MMR (`_select`)

This is the most involved function. Signature: `_select(events, target_chars, head_frac, tail_frac) ➜ (selected_events, overrun)`. The budget is in **characters** (not tokens).

**Edge cases handled up front:**
- empty events ➜ `[], 0`
- `target_chars is None` ➜ return all events (no cap)
- `target_chars <= 0` ➜ keep the single highest-scoring sentence (tie-break: later position wins via `-position`)

**Performance structure** (the comment block explains each optimization):
- Precomputes `token_sets` and `dedup_keys` once.
- Builds an **inverted index** `postings[(key, token)] ➜ [event indices]`. Candidates sharing no content tokens have zero Jaccard similarity, so they never need a similarity update when a new event is selected. This is what makes the MMR loop incremental instead of O(n) per selection.
- `max_similarity[i]` — max Jaccard similarity of candidate `i` to any **selected** compatible event.
- `versions[i]` — version counter for **lazy heap deletion**: when a candidate's score changes, push a new heap entry with an incremented version; stale entries are discarded on pop by comparing versions.

**MMR score:** `mmr_score(i) = lam * relevance - (1 - lam) * redundancy`, with `lam = 0.75`, `relevance = score / 10` (so weights normalize to [0,1]), `redundancy = max_similarity[i]`. Higher is better; the heap stores `-mmr_score` for a min-heap.

**`incremental_cost(event)`** = `len(text) + (1 if already selected else 0)` — exact character cost including the single-space separator that will join sentences. The comment notes this removes a previous estimated-cost/actual-cost discrepancy.

**`add(index)`** — adds an event only if `used + cost <= target_chars`; on success it registers the selection and calls `update_similarities(index)` to incrementally refresh affected candidates' redundancy.

**The 6 selection phases (in order):**

1. **Reserve head context** — walk from the start, adding sentences until `head_frac * target_chars` is consumed. This preserves task setup.
2. **Reserve tail context** — walk from the end backward, adding until `tail_frac * target_chars` is consumed. Preserves current state / what's done / what's blocked.
3. **Guarantee first and last** — `add(0)` and `add(n-1)` so the very beginning and very end always survive when they fit.
4. **Coverage guarantee** — for each label in `_COVERAGE_ORDER` not yet represented among selected events, pick the highest-scoring (then earliest) candidate carrying that label and add it. This is the structural guarantee that every reasoning-event category present in the input appears in the output.
5. **Fill with incremental MMR** — pop from the heap; skip already-selected or stale-version entries; skip if over budget (the budget only decreases, so an oversized event can't become affordable later); otherwise `add`. This greedily balances relevance vs. redundancy.
6. **Tiny-budget floor** — if nothing was selected (e.g. a positive but sub-sentence budget), force-keep the single highest-scoring sentence.

After selection, events are re-sorted by original `position` to restore chronology. The returned `overrun` is `max(0, actual - target_chars)` where `actual` includes separators — an honest measure of how far over budget the result is.

#### `compress()` — the orchestrator

`compress(text, *, dedup_threshold=0.85, target_chars=None, head_frac=0.15, tail_frac=0.25) ➜ (output_str, stats_dict)`

Order of operations:
1. Validate `dedup_threshold` in [0,1] and non-negative fracs.
2. Strip `_THINKING_HEADER` headers.
3. Drop whole lines matching `_META` (counted as meta).
4. Segment remaining text with `split_sentences`.
5. Per sentence: drop if `_META` matches (meta), drop if `_FILLER_ONLY.fullmatch` (filler), else `_classify` into an event.
6. `_deduplicate` ➜ distinct events + duplicate count.
7. `_select` ➜ selected events + overrun.
8. `output = " ".join(event["text"] for event in selected)`.
9. Build `stats` (see below) and a `ledger` (a copy of every classified event, position-tied).

**Stats dict keys:** `segmenter`, `input_chars`, `total_sentences`, `meta_stripped` (line + sentence meta), `filler_removed`, `usable_sentences`, `duplicates_removed`, `distinct_sentences`, `kept_sentences`, `capped_dropped` (= distinct - kept, i.e. dropped purely by the budget cap), `target_chars`, `budget_overrun_chars`, `output_chars`, `label_counts` (Counter over all events), `ledger`.

#### CLI (`main`)

- `--text` (else stdin), `--target` (0.05), `--dedup` (0.85), `--head-frac` (0.15), `--tail-frac` (0.25), `--debug`, `--ledger-json FILE`.
- Validates ranges (`parser.error` on bad values).
- `target_chars = int(len(source) * target)` — the fraction is applied to the **raw input length** (before any stripping), so the budget is predictable relative to what you fed in.
- Writes the ledger JSON if requested; prints stats to **stderr** under `--debug` (never stdout); writes compressed text to stdout.

#### Key design properties (worth remembering)

- **Deterministic.** No randomness, no time, no dict-ordering hazards that matter (heap ties broken by position). Byte-identical across runs.
- **Character-budgeted, not token-budgeted.** The target is a fraction of raw input characters.
- **Coverage guarantee is structural.** Every event category present in the input yields at least one kept sentence (when it fits the budget).
- **Polarity/modality-aware dedup** prevents "works" from collapsing into "doesn't work".
- **Precision-over-recall labeling.** Explicit regexes; a sentence with no label is still kept as low-priority filler-of-last-resort rather than dropped.

### `isolate/llm_compress.py` (143 lines) — compress + ONE `pi -p` call to resume

A thin wrapper around `plain_compress.compress`. It is the "resume" half of the blackbox: deterministically compress the thinking, then hand the compressed text to a live model and ask it to continue the work.

#### Pipeline

1. **Deterministic compression** — calls `plain_compress.compress(text, dedup_threshold=a.dedup, target_chars=int(len(text) * a.target))`. No LLM here; reuses the exact same engine as `plain_compress.py` (imported, not reimplemented). Note it uses the default `head_frac`/`tail_frac` (0.15/0.25) — those flags are **not** exposed on this CLI.
2. **Build ONE prompt** — `CONTINUATION_PROMPT.format(thinking=compressed)`. If `--ground-truth FILE` is given, the file's text is read and `JUDGE_SUFFIX` is appended to the **same** prompt (so scoring happens in the same single call, never a second call).
3. **Exactly one LLM call** — `run_pi(prompt, model, provider)`.

#### The two prompts

- `CONTINUATION_PROMPT` — generic, no project/git/scenario hardcoded. Frames the input as "internal thinking of an AI coding agent from earlier in a session" and demands an exact reply format:
  - `ACTIONS:` numbered list (max 8) of the exact next actions/commands, in order.
  - `REPORT:` <= 3 sentences on what's done / blocked / open.
- `JUDGE_SUFFIX` (only when `--ground-truth` is present) — adds a third section asking the model to classify each action as `redundant` / `necessary` / `neutral` and emit `RAR: <redundant>/<total>`. "Redundant" is defined concretely (re-investigating, re-reading files from scratch, re-editing already-fixed files, re-running verification that already passed). This is the RAR (Redundancy-Acceleration Ratio) self-scoring mentioned in the README.

#### `run_pi(prompt, model, provider)`

- Builds `pi -p` command. If a `provider` is given **and** the model has no `/` prefix, adds `--provider`; always adds `--model`.
- Pipes the prompt via **stdin** (comment: prompts can exceed ARG_MAX) — never as an argv string.
- `subprocess.run(..., capture_output=True, text=True, timeout=60, cwd=HERE)`.
- Returns `(stdout.strip(), "")` on success; `(None, error)` on non-zero return code (first 500 chars of stderr/stdout) or timeout (`"timeout after 60s"`).

#### CLI flags

`--text` (else stdin), `--target` (0.05), `--dedup` (0.85), `--model` (default `"Summary"`), `--provider` (default `llama.relay`, ignored if model has a `/` prefix), `--ground-truth FILE` (optional, enables self-scoring in the same call), `--debug`.

#### Behavior / properties

- **Exactly one LLM call, always.** Ground truth changes *what* that single call does (adds scoring), never *how many* calls are made. This is a hard invariant called out in both the docstring and README.
- **Deterministic compression, non-deterministic resume.** The compression step is byte-identical across runs; the LLM continuation is a live model call and is not.
- **The compressed text is intermediate** — only shown under `--debug`; stdout carries only the LLM's reply.
- **Failure handling:** if `run_pi` returns `None`, prints `LLM call FAILED: <err>` to stderr and exits 1 (so a failed model call is distinguishable from an empty-but-successful one).
- **60 s timeout** (`PI_TIMEOUT`) on the pi call.

### `isolate/lede/sentences.py` (78 lines) — the sentence segmenter the compressor depends on

This is the **only** part of the `lede` package that `plain_compress.py` actually uses (`from lede.sentences import split_sentences`). Everything else in `lede/` is bundled library code not on the compression path. So this file is the one dependency the algorithm has on the bundle.

#### What it does

`split_sentences(text) -> list[str]` splits text into sentences. Empty input ➜ `[]`. It splits on:
- **Paragraph breaks** (2+ newlines: `\n\s*\n`), OR
- **Terminal punctuation + whitespace + uppercase start**: `(?<=[.!?])\s+(?=[A-Z])` — a lookbehind keeps the terminator attached to the preceding sentence.

The combined pattern is `r"(?<=[.!?])\s+(?=[A-Z])|\n\s*\n"`, applied to the **masked** text, and each piece is stripped; empty pieces dropped.

#### The masking trick (the interesting part)

The naive "split on `. ` + uppercase" rule breaks on abbreviations and decimals. `_mask`/`_unmask` solve it by temporarily replacing protected periods with a NUL sentinel (`\x00`, which won't appear in real text):

- **Decimals** — `digit.digit` ➜ `digit\x00digit` (so `3.14` isn't split).
- **"No." before a number** — `\b(No)\.(?=\s*\d)` ➜ `No\x00` (protects "No. 5", "No. 17"). Bare "no." at end of sentence is deliberately **not** protected ("He said no. Then he left." still splits).
- **Known abbreviations** — a frozenset (`mr, mrs, ms, dr, st, sr, jr, inc, ltd, co, corp, vs, etc, eg, ie, cf, u.s, u.k, u.n, e.u, fig, vol`). A regex `\b([A-Za-z]+(?:\.[A-Za-z]+)*)\.` matches a word (possibly dotted like `U.S`) + period; if the word is in the set, its period becomes `word\x00`. This covers both `Dr.` and multi-dot `U.S.`.
- **NUL stripping** — any NUL already in the input is removed first (comment: NUL bytes can appear in PDF-extracted/ETL text; crashing on them would be hostile). It notes a Rust mirror exists for parity (`rust/src/sentences.rs::mask`).

After splitting, `_unmask` restores `\x00` ➜ `.`.

#### Why it matters to the compressor

The compressor's entire pipeline is **sentence-granular**: classification, dedup, coverage, and MMR all operate on the sentence list. So segmentation quality directly bounds output quality — a mis-split merges two reasoning events (undercounting categories) or splits one across a decimal/abbreviation (creating junk sentences that get classified as low-priority filler). The abbreviation/decimal protection is what keeps technical thinking text (full of `3.14`, `No. 5`, `U.S.`, `etc.`) from fragmenting.

#### Note on the fallback

In `plain_compress.py`, if `lede` can't be imported, a regex fallback (`re.split(r"(?<=[.!?])\s+|\n+", text)`) is used instead. That fallback is cruder — it splits on any terminal punctuation + whitespace (no uppercase requirement) or newlines, and has **no** abbreviation/decimal protection. So the `lede` segmenter is strictly better; the fallback just keeps the script runnable without the bundle.

### `isolate/lede/__init__.py` (43 lines) — package public API

Declares `lede` as a **deterministic extractive summarization** library (`__version__ = "0.5.1"`) and re-exports its public API:

- `summarize(text, max_length, mode='default') -> SummaryResult` — TFIDF-based extractive summary (from `lede.tfidf`).
- `brief(text, *, overview_max, max_facts, include_phrases, format) -> str | dict` (from `lede.brief`).
- `readable_report(text, *, max_length=2000, backend='regex') -> ReadableReport` (from `lede.report`).
- `clean_text(text)`, `strip_think(text)` (from `lede.clean`).
- `extract_keyword(text, keywords, num_sentences=10) -> str` (from `lede.keyword`).
- `set_default_backend(name)` — enrichment backend selector (from `lede.extract._backends`).
- `format_extract(kind, value, output='text')`, `format_result(value, output='text')` (from `lede.format`).
- Type records: `ReportAttribute`, `FactRecord`, `PromotionCandidate` (from `lede._types`).

**Key point for the isolate blackbox:** the compressor does **not** use any of this public API. It imports `lede.sentences.split_sentences` directly, bypassing the package's summarization/report machinery. So `lede/` is a general-purpose extractive-summarization library that happens to be bundled here; the compression algorithm only borrows its sentence segmenter. The rest (`tfidf.summarize`, `brief`, `report`, `extract/*`) is available but not invoked by `plain_compress.py` or `llm_compress.py`. The files below document what the bundled library actually contains.

> **Scope note for the rest of this section:** everything under `lede/` (except `sentences.py`) is **not on the compression path**. These analyses describe the bundled library so the blackbox is fully understood, but none of it changes how `plain_compress.py` behaves.

### `isolate/lede/clean.py` (100 lines) — text cleaners

Two deterministic, stdlib-only cleaners:

- **`strip_think(text)`** — removes `<think>...</think>` blocks (reasoning-model thinking) via `re.sub(r"<think>.*?</think>\s*", ...)` with `DOTALL`, then trims. Returns `""` for `None`. This is the inverse of what the compressor does with its input: it strips *thinking* rather than compressing it.
- **`clean_text(text)`** — a 7-step normalization pipeline, in order: (1) strip markdown (`*`, `_`, `#` headings, `---` rules, bullet/numbered prefixes), (2) remove filler phrases, (3) remove filler words, (4) remove CRM boilerplate, (5) lowercase, (6) collapse whitespace/blank lines, (7) trim. Returns `""` for `None`/empty.

The filler/boilerplate regexes (`_FILLER_PHRASES`, `_FILLER_WORDS`, `_CRM_PATTERNS`) are the tell that this library was built partly for **CRM/sales-note summarization**, not just coding text: patterns like "hope this helps", "let me know if you have any questions", "No updates", "Calendar invite sent", "Meeting confirmed for next week". So `lede` is a general extractive-summarization tool with a sales/CRM provenance; the isolate blackbox only reuses its sentence segmenter.

### `isolate/lede/_types.py` (174 lines) — frozen return-type records

The data model for the library's outputs. All are frozen dataclasses except `TermScore` (a `NamedTuple`, deliberately, so it can be unpacked positionally and accessed by name):

- `TermScore(term, score, kind)` — one scored term; `kind` is `"word"` (TF-IDF) or `"phrase"` (n-gram). Scores are normalized per-kind in [0,1] (soft-hint mode may exceed 1.0).
- `Stat(value, unit, phrase, context_sentence, stat_type)` — a quantitative fact; `stat_type` is money/percent/count/date/duration.
- `Section(depth, name, representative_sentence)` — an outline section.
- `Metadata(dates, amounts, urls, entities)` — extracted metadata; `entities` populated only if the optional `lede[ner]` extra is installed.
- `PhraseFact(entity, number, polarity, sentence)` — a correlated phrase+number fact.
- `ReportAttribute`, `FactRecord`, `PromotionCandidate` — report/fact/promotion metadata records.
- `SummaryResult` — the main summary object (`summary` + optional stats/outline/metadata/phrases/correlated_facts/pinned_headings), with `to_dict` / `to_json` / `to_markdown` / `__str__`.
- `ReadableReport` — a combined lede + spaCy report (adds spacy_metadata/spacy_facts/spacy_phrases/attributes/fact_records/promotion_candidates/search_text), with its own renderers.

These types are what `summarize()` and `readable_report()` return. They're irrelevant to the compressor, which produces a plain string + stats dict, not these records.

### `isolate/lede/keyword.py` (79 lines) — query-driven keyword extractor

`extract_keyword(text, keywords, num_sentences=10)` — a **query-driven** extractive selector (contrast with the compressor's query-less, event-taxonomy-driven selection):

- Parses `keywords` into unique lowercase tokens of length > 2; if none remain, returns `""` (the docstring notes the original SQL reference returned `LEFT(text, 2000)` here, but that silent chop was a footgun so Python/Rust now return empty).
- Splits sentences with `_split_sql_style` (a **separate** splitter from `lede.sentences.split_sentences`: it first replaces `\n+` with `. ` then splits on `[.!?]\s+`, keeping pieces > 20 chars). This is kept for parity with the original PL/pgSQL reference.
- Scores each sentence: `(# keyword matches) + 0.5 (if len > 200) + 0.3 (if contains a digit) + 1.0 (if causal/analytical language per `_CAUSAL_RE`)`.
- Sorts by `(-score, original_position)` for a deterministic tie-break and returns the top-N sentences newline-joined in score order.

Relevance to isolate: this is an *alternative* extraction strategy (keyword/query-driven) that the compressor does **not** use. The compressor instead scores by reasoning-event label weight + MMR novelty. Useful contrast: `extract_keyword` needs a query; `plain_compress` needs none.

### `isolate/lede/_headings.py` (111 lines) — shared heading detection

Detects structural headings so they can be dropped from candidate selection (in `mode='default'`) and reused by `extract.outline` to discover section names. Six patterns:

- `_MD_HEADING_RE` — markdown ATX (`#+ text`).
- `_ALLCAPS_RE` — all-caps short line (4-80 chars, no lowercase; upper bound extended 28➜80 for long titles like "SUPREME COURT OF THE UNITED STATES").
- `_SHORT_LABEL_RE` — short label ending in colon (<= 30 chars).
- `_BARE_TITLE_RE` — bare title-case single line (starts uppercase, ~60 chars, **no terminal punctuation** — that constraint is load-bearing to exclude body sentences like "Costs declined.").
- `_NUMBERED_SECTION_RE` — `\d+. Section Name` on its own line.
- `_TITLE_WITH_DASH_RE` — `Title — Metadata` (em/en-dash or hyphen, space both sides so hyphenated words don't match; no terminal punct for the same body-sentence reason).

Functions:
- `is_structural_heading(sentence)` — True if any of the six explicit patterns match (narrower: drops the "< 4 content tokens" heuristic so short body sentences don't masquerade as headings).
- `md_depth(heading_line)` — markdown heading depth (# = 1, ## = 2); 1 for non-markdown.
- `is_heading(sentence)` — broader: structural OR the heuristic "fewer than 4 content-word tokens" (`[A-Za-z]{3,}` count).
- `heading_name(sentence)` — extracts the name portion (strips markdown `#`, numeric prefix, em-dash metadata suffix, trailing colon), or `None`.

Relevance to isolate: the compressor does **not** call any of this. It strips `# Thinking N` headers with its own `_THINKING_HEADER` regex and treats everything else as candidate sentences. So heading detection here is a `lede`-library feature (used by `summarize(mode='default')` / `extract.outline`), not part of the compression algorithm.

### `isolate/lede/tfidf.py` (743 lines) — the main lede extractive summarizer

This is `lede`'s headline feature: classic **extractive summarization** via a composite sentence score. It is *not* used by the isolate compressor, but it's the largest lede module and worth understanding as the "general" counterpart to the domain-specific `plain_compress`.

#### The composite score (the core formula)

```
score = 0.60 * tfidf + 0.25 * position + 0.15 * length
```
All three dimensions normalized to [0,1]; the weighted sum is also in [0,1].

- **`tfidf_score(sentences)`** — per-sentence TF-IDF. Tokenizes with `\b[a-z]{3,}\b` minus a small stable `_STOPWORDS`. Document frequency counts how many *sentences* contain each term; IDF is smoothed as `log((n+1)/(freq+1)) + 1.0` (the `+1` avoids `log(1)=0` for universal terms). Per-sentence score = `sum(tf * idf) / len(tokens)` (average term importance), then min-max normalized.
- **`position_score(n)`** — a **U-shape**: first and last sentences score 1.0, the middle scores low. Computed as `d = min(i, n-1-i)/(n-1)`, `score = max(0, 1 - 2d)` (1 at both ends, 0 at center). This encodes the "lead + conclusion matter most" prior.
- **`length_score(sentences)`** — peaks in the 10-30 word range (=1.0); shorter scales up linearly (`words/10`), longer decays linearly to 0 by word 80. Encodes "a good summary sentence is a full but not bloated sentence."

#### Scoring modes

- **`_composite_score_legacy`** / `composite_score` — the raw 60/25/15 weighted sum. Byte-identical to v0.0.1.
- **`_composite_score_default`** (v0.2) — legacy + four tweaks on top:
  1. Heading sentences ➜ `-inf` (dropped from selection).
  2. Sentences under a high-signal section heading (`discussion/conclusion/held/resolution/key findings/summary/decision`) ➜ `tfidf *= 1.3`.
  3. Cue-phrase sentences (`"Held:", "Resolution:", "In summary", ...` via `_CUE_PHRASE_RE`) ➜ `+2.0`.
  4. Digit-bearing sentences ➜ `+0.3`.

Supporting helpers: `_separate_heading_lines(text)` inserts a blank line after any heading candidate so the splitter treats it as a standalone sentence; `_build_section_map(sentences)` maps each sentence index to the most recent heading name above it (for the section boost).

#### Selection pipelines

- **`_summarize_legacy(text, max_length=500)`** — the v0.0.1 pipeline: (1) if input fits budget, return unchanged; (2) if budget < 50 chars, truncate; (3) split into sentences, if < 3 truncate; (4) score with legacy composite; (5) greedily add highest-scoring sentences (tie-break by original position) until the char budget is spent; (6) reorder selected by original position; (7) fallback to truncation if nothing selected. `_truncate` reserves 3 chars for `...`.
- **`_select_default` / `_summarize_default`** — same greedy structure but with the v0.2 default scorer (skips `-inf` heading sentences).
- **`_select_with_hints` / `_summarize_with_hints`** — a **two-pool** selection when `hints` are given: a hint pool (bonus-augmented in soft mode, or hard-filtered to only hint-matching sentences) gets `hint_focus` (default 0.7) of the budget, the plain pool gets the rest, with **rollover** both directions (unused hint ➜ plain; unused normal ➜ hint in soft mode). Matching is case-insensitive, word-boundary (`\b`), no stemming.
- **`_select_for_mode`** — dispatches to the right selector (hints / coverage / default).

#### Top-level API: `summarize(...)`

```
summarize(text, max_length=500, *, mode="default", attach=None,
          hints=None, hint_focus=0.7, hint_mode="soft",
          keep_headings=False, include_toc=False, pin=None, headings=None)
    -> SummaryResult
```

- **`mode`** — `"default"` (TF-IDF + position + length, heading-filtered), `"legacy"` (pre-v0.2 byte-identical), or `"coverage"` (paragraph-aware, delegated to `lede.coverage`).
- **`attach`** — optional list of primitives to attach to the result: `stats`, `outline`, `metadata`, `phrases`, `correlated_facts` (each lazily imported from `lede.extract`).
- **`hints` / `hint_focus` / `hint_mode`** — term biasing (see two-pool selection above). List entries weight 1.0; dict keys➜weights.
- **`keep_headings` / `include_toc` / `pin` / `headings`** — heading & pin retention (v0.4.2/0.4.3). Output block order: (1) `pin` block verbatim, (2) TOC block if `include_toc`, (3) body with document title + nearest enclosing section heading interleaved when `keep_headings`. Pinned content is added **on top of** `max_length` (the budget governs only the extractive body, so pins always survive). `headings` (non-empty) replaces auto-detection entirely. All rejected in `mode="legacy"` (raises `ValueError`).
- Returns a frozen `SummaryResult`; `str(r)` / `f"{r}"` evaluate to `.summary` for legacy string callers.

#### How it contrasts with `plain_compress.py` (the important takeaway)

These are two *different* extractive algorithms that share only the sentence segmenter:

| | `lede.tfidf.summarize` | `isolate/plain_compress` |
|---|---|---|
| Scoring signal | TF-IDF + position U-shape + length | reasoning-event label weight + MMR novelty |
| Domain assumption | generic documents (legal/CRM/news) | coding-agent chain-of-thought specifically |
| Selection | greedy by composite score, char budget | head/tail reserve + per-category coverage guarantee + incremental MMR |
| Guarantees | "lead+conclusion matter" prior | every event category present ➜ at least one kept sentence |
| Dedup | none (pure scoring) | polarity/modality-aware Jaccard dedup |
| Output type | `SummaryResult` dataclass (+optional attachments) | plain string + stats dict |

So the isolate blackbox deliberately does **not** use lede's TF-IDF summarizer; it rolls its own CoT-specific selection because the reasoning-event taxonomy (what was accepted/rejected/left open/where things stand) is a stronger prior for resuming a coding session than generic TF-IDF salience.

### `isolate/lede/coverage.py` (143 lines) — paragraph-coverage selector (`mode="coverage"`)

A third selection strategy, activated via `summarize(mode="coverage")`. Unlike default mode (which chases per-sentence signal density), coverage mode optimizes for **breadth** — how many distinct paragraphs the summary touches. It uses the raw `0.60*tfidf + 0.25*position + 0.15*length` composite **without** the default-mode cue/digit/section boosts.

- **`_split_paragraphs(text)`** — blank-line-delimited paragraphs (`\n\s*\n+`), keeping only those >= 20 chars after strip.
- **`select_coverage_indices(text, max_length=500)` ➔ `(sentences, selected_sorted) | None`.** Two-pass selection:
  1. **Per-paragraph pass:** map each sentence to its source paragraph (by occurrence-counting — the K-th occurrence of a repeated sentence maps to the K-th paragraph containing it; the comment notes the old "first containing paragraph wins" rule biased FAQ/template docs, closing AAT-021). Pick the best non-heading sentence per paragraph, then iterate paragraphs in document order adding each best if budget allows.
  2. **Greedy fill:** only add sentences from paragraphs **not yet represented**, preserving the breadth guarantee, until the budget is full.
  - Reorder selected by original position; return `None` on empty input.
- **`summarize_coverage(text, max_length=500)` ➔ `str`.** Wraps the selector and joins with `" "` (the comment notes an earlier version used `"\n"`, which surprised callers diffing across modes — AAT-022).
- **`_composite_score_coverage_for_hints(sentences)`** — raw 60/25/15 composite per sentence with headings set to `-inf`, for the hint two-pool path. Note: coverage's per-paragraph breadth guarantee is intentionally **not** preserved when hints are used (the caller asked for hint-biased selection; breadth is secondary).

Relevance to isolate: none on the compression path. It's lede's "touch every paragraph" summarization mode — a generic-document strategy, again distinct from the compressor's event-coverage guarantee (which guarantees *reasoning categories*, not *paragraphs*, are represented).

### `isolate/lede/_hints.py` (160 lines) — hint biasing internals

Package-private helpers powering the `hints` kwarg on `summarize`. The docstring stresses it **must remain byte-identical to `rust/src/hints.rs`** — lede has a Rust mirror and these are shared contract code.

- `preprocess_hints(hints)` ➜ list of `(lowercased_hint, weight)`. Strips, collapses internal whitespace, lowercases; drops empty. List input ➜ weight 1.0 each; dict input ➜ values are weights.
- `_compile_hint(hint)` ➜ cached word-boundary regex `\b<escaped>\b` (a module-level `_REGEX_CACHE`; unbounded but fine for stable hint sets).
- `match_count(hint, sentence)` ➔ non-overlapping match count (sentence lowercased here; hint preprocessed by caller).
- `round_to_int(value, focus)` ➔ **portable integer rounding** to avoid Python banker's-rounding vs Rust half-away divergence: encodes `focus` as `numerator/10000`, then `(value*num + den//2)//den`. This is a deliberate cross-runtime determinism measure.
- `hint_bonus(sentence, hints)` ➔ sum over hints of `min(count, 3) * weight * 0.5` (`_HINT_MATCH_CAP=3`, `_HINT_BASE_WEIGHT=0.5`). Capping at 3 means a sentence stuffed with the same hint doesn't dominate.
- `_rank_indices(scores)` ➔ sort by `(-score, original_index)` (deterministic tie-break).
- `select_by_chars(scores, lengths, budget, exclude, separator_len=1)` ➔ greedy char-budget selector (skips excluded and `-inf` scores).
- `select_by_count(scores, count, exclude)` ➔ top-N count-budget selector (for primitives like key_facts/phrases).

These are the building blocks `_select_with_hints` in `tfidf.py` composes into its two-pool hint selection.

### `isolate/lede/_pins.py` (246 lines) — heading/pin retention rendering

Pure, deterministic helpers for the v0.4.2/0.4.3 `keep_headings` / `include_toc` / `pin` / `headings` features on `summarize`. Reuses `lede._headings` for all "what is a heading" logic (adds no new detection). Rust mirror: `rust/src/pins.rs`.

- `nearest_heading_map(sentences)` ➔ for each index, the most recent structural-heading index at or before it (a heading's own entry is `None`).
- `document_title_index(sentences)` ➔ index of the document title: a **depth-1** heading appearing before any body sentence; else `None`.
- `render_toc_from_list(headings)` ➔ TOC from caller-supplied headings (deduped first-wins, given order, one/line).
- `render_toc(text)` ➔ TOC from the document outline (`lede.extract.outline`), indented `(depth-1)*2` spaces.
- `_render_keep_headings_override(sentences, selected, headings)` ➔ renders the body with caller-supplied headings per spec ¨4.3: first heading pinned as title, matched headings interleaved before their section's selected sentences, unmatched headings emitted as a leading block, plus a **survival guarantee** that every supplied heading appears in the output.
- `render_with_pins(sentences, selected, keep_headings, include_toc, pin, text, headings)` ➔ weaves pinned content around the extractive body; returns `(output_string, pinned_headings)`. Final block order: **pin ➜ TOC ➜ body**, joined by blank lines. When no headings are pinned it's byte-identical to the plain scorer.
- `prepend_blocks(body, include_toc, pin, text, headings)` ➔ prepends pin + TOC blocks to an already-rendered body (used when `keep_headings` is off but pin/TOC are on).

All deterministic: dedup preserves first occurrence; no ordering depends on hashing or locale.

### `isolate/lede/_parity.py` (77 lines) — cross-runtime parity formatters

Canonical text formatters for the v0.2 extract primitives, used by the **cross-runtime parity walker** (`rust/tests/fixtures.rs`) and the Python fixture generator (`benchmarks/gen_parity_fixtures.py`). Each takes a primitive's structured output and emits a deterministic line-oriented text representation so a Python-generated fixture can be byte-compared against the Rust output for the same input. Mirrors `rust/src/_parity.rs`.

Format conventions: one record per line; fields separated by ` | ` (space-pipe-space); trailing newline at EOF; empty output ➔ empty file (no trailing newline).

- `_join_lines(lines)` ➔ `"\n".join(lines) + "\n"`, or `""` if empty.
- `format_stats(stats)` ➔ `stat_type | value | unit | phrase | context_sentence` per line.
- `format_outline(outline)` ➔ `depth | name | representative_sentence` per line.
- `format_toc` / `format_phrases` / `format_top_terms` / `format_key_facts` ➔ one string per line.
- `format_metadata(md)` ➔ `DATES: ...`, `AMOUNTS: ...`, `URLS: ...`, `ENTITIES: ...` lines.
- `format_correlate_facts(facts)` ➔ `entity | number | polarity | sentence` per line.

Relevance to isolate: none of this is on the compression path. It exists purely so the Python and Rust lede implementations can be proven byte-identical via shared fixtures — a property of the *library*, not the blackbox compressor.

### `isolate/lede/brief.py` (197 lines) — at-a-glance document brief

A top-level convenience primitive that **composes** other lede primitives into one caller-friendly artifact: `summarize()` (overview) + `extract.key_facts()` + `extract.toc()`, with optional `extract.phrases()`. Target use cases: email digests, file-browser previews, ingest-pipeline pre-summarization. Document-type-agnostic (no per-doc heuristics). Mirrors `rust/src/brief.rs`; the three output formats must stay byte-identical across Python/Rust for the regex backend.

- **Wordforms auto-detect** — at import time it tries `import text_to_num` and sets `_HAS_WORDFORMS`. When true (or when `convert_word_names=True`), spelled-out numbers ("five thousand documents") are converted so they surface in key facts. The docstring flags a parity footgun: Python auto-detect is runtime, Rust's is a compile-time cargo feature, so the two can silently disagree unless you pass an explicit `convert_word_names` on both sides.
- **Overview budget clamps** — `overview_max` clamped to `[0.05, 0.50]`, then the resolved char budget clamped to `[250, 1500]` (floor keeps short docs readable; ceiling keeps a 30 KB doc from producing a 10 KB "overview").
- **`brief(text, *, overview_max=0.35, max_facts=10, include_phrases=False, convert_word_names=None, format="string", hints=None, hint_focus=0.7, hint_mode="soft")` ➜ `str | dict`.** The `hints`/`hint_focus`/`hint_mode` are forwarded unchanged to all three primitives (each applies its own budget split independently). `format` is `"string"` (plain text with section labels), `"markdown"` (`##` headers + bullets), or `"dict"` (structured `{overview, key_facts, toc, phrases}`).

Relevance to isolate: none on the compression path. It's a higher-level composition over the same TF-IDF summarizer the compressor avoids.

### `isolate/lede/report.py` (277 lines) — readable combined report

Builds a `ReadableReport` that combines lede core output with **optional spaCy** extraction. The default is tuned for human/agent inspection: a 2000-char lede summary with headings/TOC retained, up to 40 lede key facts, regex metadata/stats, and spaCy-backed entities/entity-number correlations only when `backend="spacy"` or `"auto"`.

Key helpers:
- `_BOLD_KV_RE` / `_PLAIN_KV_RE` — key-value regexes (bold `**Label:** value` and plain `Label: value` lines) for attribute extraction.
- `_register_spacy_if_requested(backend)` — imports `lede_spacy` (side-effect registers the backend) when backend is `spacy`/`auto`; raises a helpful install error if `backend="spacy"` but it's missing; returns False otherwise.
- `_normalize_key(label)` — label ➔ snake-case key (`&`➜" and ", non-alnum➜`_`).
- `_value_type(key, value)` — classifies a value as year / money / url / citation / identifier / list / text via regex heuristics.
- `_attribute_from_pair(...)` ➔ a `ReportAttribute` (confidence 0.99 for header_kv, 0.9 for line_kv).
- `_extract_attributes(text)` — scans bold-KV then plain-KV lines into a deduped tuple of attributes.
- `_build_fact_records(attributes, lede_stats, spacy_facts)` ➔ `FactRecord`s (attribute / numeric / entity_number types with confidences 0.99/0.8/0.7).
- `_build_promotion_candidates(attributes)` ➔ `PromotionCandidate`s (path `lede_report.attributes.<key>.value`).
- `_build_search_text(...)` ➔ a newline-joined blob of summary + facts + attributes + metadata + phrases for downstream search.

**`readable_report(text, *, max_length=2000, max_facts=40, mode="default", backend="regex", include_toc=True, keep_headings=True, headings=None, pin=None, hints=None, hint_focus=0.7, hint_mode="soft", convert_word_names=False)` ➔ `ReadableReport`.** Calls `summarize` (with heading/TOC/pin retention), `stats`, `metadata(backend="regex")`, `key_facts`; if spaCy is available also `metadata/correlate_facts/phrases(backend="spacy")`; then assembles attributes, fact records, promotion candidates, and search text into the `ReadableReport` dataclass.

Relevance to isolate: none on the compression path. This is the richest lede feature (a full structured report), again built on the TF-IDF summarizer + extract primitives rather than the CoT event taxonomy.

### `isolate/lede/format.py` (392 lines) — output formatting / presentation layer

The presentation layer for lede results. The core primitives are data-first (`summarize()` ➜ `SummaryResult`, extract primitives ➜ tuples/dataclasses); this module turns that structured data into text / Markdown / JSON, used by both the public API and the CLI.

- **Serialization:** `to_data(value)` recursively converts dataclasses/NamedTuples/lists/dicts to JSON-serializable data; `to_json(value, indent=2)` serializes with `ensure_ascii=False` (UTF-8 preserved), using `value.to_dict()` when available.
- **Per-primitive renderers** (each has a `_text` and `_markdown` variant): `stats_text/stats_markdown`, `metadata_text/metadata_markdown`, `outline_text/outline_markdown`, `correlate_text/correlate_markdown` (the latter groups rows by source sentence so long structured blocks aren't repeated once per entity-number pair).
- **`summary_markdown(result)`** — renders a `SummaryResult` plus any attachments (stats/outline/metadata/phrases/correlated_facts) as Markdown.
- **Human-report helpers:** `_brief_context(text, max_chars=280)` (single-line snippet), `_stat_brief(row)`, `_compact_fact_records(report, limit=12)` (drops attribute-type records and numeric records that duplicate an attribute key, dedupes by `(subject, predicate, object, fact_type)`, caps at 12).
- **`report_text(report)` / `report_markdown(report)`** — render a full `ReadableReport` (summary + key facts + stats + metadata + structured attributes + compact fact records + spaCy entities/phrases/entity-fact links) as plain text or Markdown.
- **Dispatchers:** `format_extract(kind, value, output="text", scores=False)` renders a standalone extraction primitive by name (`stats`, `facts`/`key_facts`, `metadata`, `outline`, `toc`, `phrases`, `correlate_facts`, `top_terms`); `format_result(value, output="text")` renders any lede value (uses rich Markdown for objects with `to_markdown`, falls back to `str`).

Relevance to isolate: none on the compression path. Pure presentation; the compressor emits a plain string and never touches this.

### `isolate/lede/cli.py` (526 lines) — the lede command-line interface

The CLI for the whole lede library (not the isolate blackbox). Usage: `lede [FILE] --mode MODE [OPTIONS]` — reads FILE or stdin, writes text/Markdown/JSON to stdout.

- **Modes** (`_ALL_MODES`): summary aliases `tfidf`/`summarize`/`default`/`coverage`/`legacy` (mapped via `_SUMMARY_MODES`); extraction modes `stats`, `facts`/`key_facts`, `metadata`, `outline`, `toc`, `phrases`, `correlate_facts`, `top_terms`; plus `report`, `brief`, `keyword`, `clean_text`, `strip_think`.
- **Input:** `_read_input(path)` reads a file or stdin, **forcing UTF-8** (comment: relying on locale gives Windows mojibake and breaks the byte-identical-runtime claim; the Rust CLI reads bytes and decodes UTF-8 explicitly).
- **Hint parsing:** `_parse_hints(args)` merges `--hint`/`--hints` (plain) and `--hint-weight` (`TERM=WEIGHT`) into a list or weighted dict; optionally expands via `lede_spacy.expand_hints` when `--expand-hints` is given (lemma/synonyms/similar).
- **Backend config:** `_configure_backend(args, parser)` sets `LEDE_SPACY_MODEL`/`LEDE_SPACY_VECTOR_MODEL` env vars, imports `lede_spacy` when backend is spacy/auto (with a helpful install error if missing), calls `set_default_backend`, and optionally warms up the spaCy model.
- **`_render_extract(mode, text, args, hints)`** — dispatches to the right extract primitive with the relevant kwargs.
- **`main(argv)`** — builds the full argparse surface (`--mode`, `--max-chars` (500), `--output` (text/markdown/json), `--attach`, `--keywords`, `--top`, `--max-facts`, `--format`, `--include-phrases`, `--convert-word-names`, hint flags, `--keep-headings`, `--include-toc`, `--pin`, `--heading`/`--headings-file`, backend/spacy flags, `--kinds`, `--scores`), then routes by mode: summary ➜ `summarize`; brief ➜ `brief`; report ➜ `readable_report` (max_chars defaults to 2000 for report unless explicitly set); keyword ➜ `extract_keyword`; clean_text/strip_think; extract modes ➜ `_render_extract`. Output is written to stdout with a trailing newline; `BrokenPipeError` is swallowed (returns 0).

Relevance to isolate: this is the CLI for the **lede library**, distinct from the two isolate CLIs (`plain_compress.py`, `llm_compress.py`). It's bundled but not invoked by the blackbox. Notably it has no reasoning-event-taxonomy concept — it's the generic-document summarization/extract front-end.

### `isolate/lede/extract/` subpackage — structured extractive primitives

The `extract/` subpackage holds lede's **structured extraction** primitives (stats, outline, metadata, phrases, correlate_facts, key_facts, top_terms). Each can be called standalone or attached to a summary via `summarize(attach=[...])`. None are on the isolate compression path; they're the "fact extraction" half of the lede library.

#### `extract/__init__.py` (31 lines)

Re-exports the extract primitives and their type records: `Stat, Section, Metadata, PhraseFact, TermScore` (from `_types`) and `stats, outline, toc, metadata, phrases, correlate_facts, key_facts, top_terms`. The docstring notes the task history: stub primitives returning empty collections landed first (T4), real implementations in T6-T11.

#### `extract/_backends.py` (69 lines) — pluggable backend registry

The dispatch mechanism that lets enrichment primitives (metadata, phrases, correlate_facts) run on different backends:

- `_REGISTRY: dict[backend, dict[primitive, fn]]` starts as `{"regex": {}}`; `_DEFAULT = "regex"`.
- `register(backend, primitive, fn)` — a backend (e.g. `spacy` via the `lede-spacy` companion) registers its implementations at import time by calling this.
- `resolve(backend, primitive)` — returns the registered callable. `backend="auto"` falls through `spacy` ➜ `regex` (best available without hard-coding).
- `get_default_backend()` / `set_default_backend(name)` — process-wide default; per-call `backend=` still wins.

The `'regex'` backend is **always** registered by lede itself, preserving lede's zero-dependency identity. Users opt into `'auto'`/`'spacy'` explicitly. This is how lede stays stdlib-only by default while supporting an optional neural (spaCy) path.

#### `extract/_common.py` (20 lines) — shared pipeline state

A frozen `Pipeline(text, sentences)` dataclass with a `from_text(text)` classmethod that late-imports `split_sentences` (to avoid a circular dep with `tfidf.py`). The docstring is candid: when `summarize(attach=[...])` runs, the summary step and each enrichment primitive currently parse the input **independently**; shared state is a future optimization to cut latency under a 250 ms budget. So this is scaffolding for a not-yet-realized optimization.

#### `extract/metadata.py` (80 lines) — dates / amounts / URLs (+ optional entities)

Stdlib core + pluggable backend dispatch:

- `_DATE_RE` — ISO `yyyy-mm-dd`, US `m/d/yyyy`, and bare years 1900-2099 (mirrors `extract.stats._DATE_RE` for parity).
- `_AMOUNT_RE` — money patterns (`$1,234.56K`, `5 million`, `USD 100`, etc.); quantifiers bounded `{0,18}` **for ReDoS parity** with `extract.stats._MONEY_RE`.
- `_URL_RE` — `http(s)://...`.
- `_collect_unique(regex, text)` — all matches preserving first-appearance order, deduped.
- `_regex_metadata(text)` ➔ `Metadata(dates, amounts, urls, entities=())` (entities always empty in the regex path).
- Self-registers the `'regex'` backend at module load; `metadata(text, *, backend=None)` resolves the backend (None ➜ process default) and dispatches. `backend="spacy"` would populate entities via `en_core_web_sm`.

#### `extract/outline.py` (97 lines) — hierarchical section outline

Detects sections via structural heading patterns and, for each, picks the **highest-composite-score non-heading sentence** as the representative:

- Uses `_separate_heading_lines` + `split_sentences` (same preprocessing as `tfidf.summarize` default mode), then scores with the raw `0.60*tfidf + 0.25*position + 0.15*length` composite (`_composite_score_parts`).
- Deliberately uses the **narrower** `is_structural_heading` predicate (not `is_heading`) — it does *not* treat "fewer than 4 content tokens" as a heading, because that heuristic misclassifies short body sentences as headings and would leave sections with no representative candidates. The docstring lists intentionally-unhandled Class D cases (inline `Label:Subject`, inline colon-label with body text, parenthetical structured headings) as follow-ups.
- Groups non-heading sentences under the most recent heading; emits a `Section(depth, name, representative_sentence)` per section that has both a name and a body.
- `outline(text)` ➔ `tuple[Section, ...]`; `toc(text)` ➔ `tuple[str, ...]` (just the section names in document order). Registers the `'regex'` backend for `toc` for registry symmetry.

#### `extract/stats.py` (413 lines) — numeric-fact extractor

Regex-based, deterministic, stdlib-only extraction of money / percent / date / duration / count facts, each with a context phrase. The largest extract primitive; its notable complexity is **ReDoS defense** and **optional word-form number support**.

- **Patterns:** `_MONEY_RE` (`$1,234.56K`, `5 million`, `USD 100`), `_PERCENT_RE`, `_DATE_RE` (ISO / US / bare year 1900-2099), `_DURATION_RE` (`8 days`), `_COUNT_RE` (`1,000 users`, `42 qps`, `3 basis points`, ...). All numeric quantifiers are **bounded** (`{1,15}` / `{0,18}`) to defend against catastrophic backtracking in Python's backtracking `re` engine (the comment walks through a 100K-char input ➜ 14-min hang scenario). The Rust `regex` crate is RE2-style and unaffected but mirrors the bounds for parity.
- **`_LONG_RUN_RE`** (`\d{20,}`) — second line of defense: sentences with an unbroken 20+ digit run are skipped entirely (encoded-data leaks, base64, OCR artifacts). Applied identically in both languages.
- **`_ctx_phrase(sentence, start, end, window=25)`** — the context phrase, clipped to a ~25-char window each side.
- **Word-form support (T13e, `convert_word_names=True`, optional `text_to_num` dep):** `_CharToken` / `_tokenize_with_spans` (tokens + isolated punctuation runs with char spans), `_word_form_spans` (uses `text_to_num.find_numbers` at threshold 0.0 to catch small numbers like "two weeks"), `_substitute` (replaces word-form numbers with digits in a **shadow copy**, tracking a span map), `_map_back_impl` (maps substituted positions back to original char spans). The regexes run on the digit-substituted shadow, but emitted `Stat` records preserve the **original** word-form text in `value`/`phrase`/`context_sentence`. When the flag is off, `span_map` is empty and `_map_back_impl` is a no-op ➔ byte-identical to pre-T13e.
- **`stats(text, *, convert_word_names=False)`** ➔ `tuple[Stat, ...]`. Fast path (flag off) applies the five regexes directly per sentence; flag-on path does the shadow substitution first. Emits `Stat(value, unit, phrase, context_sentence, stat_type)`.

#### `extract/key_facts.py` (272 lines) — top-N fact-bearing sentences

A **composition** primitive: reuses `stats()` + `_composite_score_parts`, no new regex patterns. Returns complete fact-bearing sentence strings (not `Stat` tuples).

- A candidate is any sentence producing at least one `Stat`. Score = `0.60*tfidf + 0.25*position + 0.15*length + 0.15*(num_stats_in_sentence / sentence_len_in_tokens)` (the stat-density term is kept small so the TF-IDF signal dominates).
- **Dedup** by normalized `(stat_type, value)` key (`_norm_value` lowercases and maps hyphen/underscore/slash ➔ space, so "five-day" == "five day"): if two sentences share any such key, only the higher-scored is kept (ties ➔ earlier position).
- **Two-pool hint selection** with rollover when `hints` are given: `hint_quota = round(max_facts * hint_focus)`, `normal_quota = max_facts - hint_quota`; dedup runs across both pools combined; unused quota rolls over (hard mode stays restricted to hint-bearing, soft mode draws from either).
- The top `max_facts` survivors are returned in **document order** (not score order) — "readers want reading flow".

#### `extract/phrases.py` (186 lines) — repeated n-gram phrase extractor

Heuristic stdlib extractor of repeated 2-5 token n-grams from non-stopword runs:

- `_runs(text)` — tokenize `[a-z]{3,}` lowercased; within each maximal run of non-stopwords, emit all contiguous 2-5 token n-grams (`_ngrams`).
- Count each n-gram; keep those with count >= 2 (repeated).
- **Subsumption (T13h):** `_subsumed` drops a sub-ngram A if some longer emitted n-gram B contains A's tokens contiguously **and** `counts[A] == counts[B]` (equal-count means A only ever appears inside B ➔ adds no independent info). This kills n-gram explosion noise like "environmental protection" / "protection agency" when "environmental protection agency" is also emitted. Sub-ngrams with standalone occurrences (`counts[A] > counts[B]`) are preserved.
- Optional `keywords` inclusion (single-count phrases containing a keyword); dedupe keeping first-appearance order; optional hint soft/hard partitioning.

#### `extract/correlate.py` (190 lines) — entity-number correlator

Pairs **repeated entities** with their **numeric facts** (composition over `stats()` + regex `phrases()` + cue-word polarity):

- `_polarity(sentence)` — infers growth / decline / absolute from cue words (`_GROWTH_WORDS` / `_DECLINE_WORDS`).
- Builds a single-word frequency map; `repeated_words` = words appearing >= 2 times (non-stopwords). Prefers multi-word phrases from the regex `phrases()` backend, **preserving insertion order** (the comment is explicit: wrapping in `set()` would iterate in `PYTHONHASHSEED` order and break Python↔Rust parity, since Rust uses an ordered Vec).
- For each stat, the entity is a phrase match (every token present in the sentence) if available, else the most-frequent repeated word in that sentence. Emits `PhraseFact(entity, number, polarity, sentence)`.
- **Filter:** require each entity to appear with >= 2 distinct facts (drops one-off pairings).
- Optional hint soft/hard partitioning (match target = `{entity} {number}`).

#### `extract/top_terms.py` (229 lines) — top-N salient words + phrases (v0.4)

Combines per-document TF-IDF on single tokens with multi-word phrase frequency into a unified ranking. Python-only for v0.4 (Rust mirror deferred to v0.5).

- **`_word_scores(text)`** — per-unique-word TF-IDF: `TF = total occurrences`, `IDF = log((num_sentences+1)/(df+1)) + 1.0` (matches `summarize`'s IDF), score = `TF * IDF` normalized by max ➔ [0,1]. 3+ letter non-stopword tokens only.
- **`_phrase_scores(text)`** — candidates from `_regex_phrases`; raw score = `repetition_count * token_count`, normalized by max.
- **`top_terms(text, *, n=10, kinds=("words","phrases"), with_scores=False, hints=None, hint_focus=0.7, hint_mode="soft")`** — gathers candidates per kind (word keyspace = single tokens, phrase keyspace = n-grams with a space, so they never collide); applies hint soft (add bonus) / hard (filter) modes; sorts by `(-score, term)` for deterministic output; returns top-N. `with_scores=True` returns `tuple[TermScore]` (`(term, score, kind)`) instead of bare strings. Note: scores are **per-kind-normalized**, not cross-kind comparable; and when both kinds are requested, phrase candidates can shadow word candidates on the same token surface.

---

## Synthesis — how the isolate algorithm actually works

Stepping back from the file-by-file detail, here is the whole picture.

### What `isolate/` is

A **self-contained blackbox** for compressing an AI agent's chain-of-thought so it can be re-injected into a prompt (or resumed by a model). It ships two CLIs plus a bundled copy of the `lede` extractive-summarization library.

### The two things that matter (the compression path)

1. **`plain_compress.py`** — the deterministic compressor. This is *the* algorithm:
   - strip scaffolding/filler ➜ segment sentences (via `lede.sentences.split_sentences`) ➔ classify each sentence with a 13-label reasoning-event taxonomy ➔ polarity/modality-aware dedup ➔ coverage-first + incremental-MMR selection under a **character budget**.
   - Its distinctive guarantees: head/tail reservation, **every event category present in the input yields at least one kept sentence**, and "works" never collapses into "doesn't work" (polarity-aware dedup).
2. **`llm_compress.py`** — a thin wrapper that runs `plain_compress.compress` and then makes **exactly one** `pi -p` call to resume the work from the compressed thinking (optionally self-scoring via RAR when ground truth is given).

### What the bundled `lede/` library is (and isn't)

- The compressor uses **only** `lede.sentences.split_sentences` from the bundle. Everything else in `lede/` is a **general-purpose extractive-summarization library** that is *not* on the compression path:
  - `tfidf.summarize` — classic TF-IDF + position-U-shape + length composite scorer (generic documents; legal/CRM/news provenance), with modes, hints, heading/pin retention, and attachable extract primitives.
  - `extract/*` — structured fact extraction: `stats` (money/percent/date/duration/count, ReDoS-defended, optional word-form numbers), `key_facts`, `phrases` (repeated n-grams with subsumption), `correlate_facts` (entity-number pairing), `outline`/`toc`, `metadata` (dates/amounts/urls/entities), `top_terms`.
  - `brief` / `report` — higher-level compositions of the above.
  - `clean` (`clean_text`/`strip_think`), `keyword` (query-driven extractor), `format` (presentation layer), `cli` (the lede CLI).
- The library has a strong **Python↔Rust byte-parity** contract (mirrors in `rust/src/*`, `_parity.py` formatters, bounded regexes, integer rounding to avoid banker's-vs-half-away divergence). That's a property of the *library*, not the blackbox.

### Why two different extractive algorithms coexist

`lede.tfidf.summarize` and `plain_compress` are deliberately **different**:

- lede scores by **generic salience** (TF-IDF + position + length) — good for "what are the important sentences of this document."
- plain_compress scores by **reasoning-event semantics** (label weight + MMR novelty) with a **coverage guarantee over reasoning categories** — good for "what did the agent decide, reject, leave open, and where does it stand now," which is exactly what you need to resume a coding session.

The blackbox borrows lede's *sentence segmenter* (a solved problem: abbreviation/decimal-safe splitting) but rolls its own *selection* because the CoT event taxonomy is a stronger prior for this task than generic TF-IDF. That division of labor — reuse the segmenter, own the selection — is the core design decision of `isolate/`.

### Determinism & safety properties (across the codebase)

- **Deterministic compression:** no randomness, no time, heap ties broken by position ➔ byte-identical across runs. (The LLM resume step is the only non-deterministic part.)
- **ReDoS defense:** bounded numeric quantifiers + long-run skip in `extract.stats`; conservative dedup thresholds; capped hint matching.
- **Zero-dependency core:** the compression path and the regex backends are stdlib-only; spaCy / text_to_num are optional extras that opt in explicitly.
- **Cross-runtime parity** (library): shared formatters, bounded regexes, integer rounding, ordered iteration — so Python and Rust outputs byte-compare.









