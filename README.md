# tier-A-code — compress a model's thinking, then resume it

Two small CLIs for working with an AI agent's chain-of-thought. Both take
**raw thinking text in** (via `--text` or stdin) and write **text out** to
stdout. No preset data ships in this folder — you pipe your own thinking in.

- **`plain_compress.py`** — deterministic compressor. No LLM, no network.
  Pure Python stdlib + the bundled `lede/` sentence segmenter. Same bytes every
  run.
- **`llm_compress.py`** — runs the same compression, then makes **exactly one**
  LLM call (`pi -p`) to resume the work from the compressed thinking. Needs the
  `pi` CLI on PATH with a configured model.

```
tier-A-code/
├ plain_compress.py   # deterministic CoT compressor (text in, text out)
├ llm_compress.py     # compress + ONE pi -p call to resume (needs pi)
├ lede/               # summarizer package — compressor reuses lede.sentences.split_sentences
└ README.md           # this file
```

---

## 1. `plain_compress.py` — deterministic, no LLM

Shrinks a thinking trace to a target size using a **reasoning-event taxonomy**
(13 labels: GAP, REJECTION, OBJECTION, REVISION, RECONSIDERATION, ACCEPTANCE,
PROPOSAL, OBSERVATION, ACTION, SUPPORT, CONCLUSION, UNCERTAINTY, CONTRAST).
Selection reserves the **head** (task setup) and **tail** (current state /
what's done / what's blocked), guarantees at least one kept sentence per event
category present in the input, then fills the remaining budget with
novelty-penalized (MMR) selection. Agent scaffolding and pure filler are
stripped first; deduplication only merges near-identical sentences with
compatible polarity (negation-aware) and modality, so "X works" is never
collapsed into "X doesn't work".

```bash
# stdin -> stdout (the realistic path for large thinking)
cat thinking.md | python3 plain_compress.py --target 0.05 > compressed.md

# or pass text directly
python3 plain_compress.py --text "$(cat thinking.md)" --target 0.10

# optional: dump the sentence-level event ledger (labels/polarity/modality per sentence)
cat thinking.md | python3 plain_compress.py --target 0.05 --ledger-json events.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--text TEXT` | *(stdin if absent)* | raw thinking text |
| `--target F` | `0.05` | output size as a fraction of input (0.05 = 5%) |
| `--dedup F` | `0.85` | Jaccard threshold for conservative deduplication |
| `--head-frac F` | `0.15` | fraction of the target reserved for beginning context |
| `--tail-frac F` | `0.25` | fraction of the target reserved for ending context |
| `--debug` | off | print compression stats to stderr (sents kept / meta stripped / dups removed / label counts / output %) |
| `--ledger-json FILE` | *(none)* | also write a sentence-level event ledger (position, text, labels, polarity, modality, score) as JSON |

Use this alone when you just want a smaller thinking blob to re-inject into a
prompt (e.g. replacing a model's own long thinking in a chat).

---

## 2. `llm_compress.py` — compress + ONE LLM call

Same compression as above, then a single `pi -p` call that hands the compressed
thinking to a model and asks it to resume: plan its next actions and report the
state. **Exactly one LLM call, always** — whether or not ground truth is given.

```bash
# stdin -> stdout; default model "Summary" on provider llama.relay
cat thinking.md | python3 llm_compress.py --target 0.05

# with a completed-work file -> the SAME single call also self-scores (RAR)
cat thinking.md | python3 llm_compress.py --ground-truth done.md --target 0.05

# choose a different model / provider
cat thinking.md | python3 llm_compress.py --model "Qwen3.8 27B" --provider llama.relay
```

| Flag | Default | Meaning |
|---|---|---|
| `--text TEXT` | *(stdin if absent)* | raw thinking text |
| `--target F` | `0.05` | compression target as a fraction of input |
| `--dedup F` | `0.85` | Jaccard dedup threshold |
| `--head-frac F` | `0.15` | fraction of the target reserved for beginning context |
| `--tail-frac F` | `0.25` | fraction of the target reserved for ending context |
| `--model M` | `"Summary"` | pi model to use |
| `--provider P` | `llama.relay` | pi provider (ignored if `--model` already has a `provider/` prefix) |
| `--ground-truth FILE` | *(none)* | optional file describing work already completed; enables self-scoring in the same call |
| `--debug` | off | print compression stats + the exact prompt sent to pi (to stderr) |

### Output

The LLM's reply on stdout:

```
ACTIONS: a numbered list of what it would do next, in order.
REPORT: <=3 sentences — what is done, what is blocked or open.
```

If `--ground-truth` is given, a third section is added in the same call:

```
SCORE: per-action label (redundant / necessary / neutral) + RAR <redundant>/<total>.
```

No ground truth — no scoring, just the continuation. Either way it's one `pi -p`
call (60 s timeout). The compressed text is intermediate and only shown under
`--debug`.

---

## Requirements

- **Python 3.10+** (stdlib only for compression).
- **`pi` CLI on PATH with a working model** — needed only for `llm_compress.py`.
  Default is `--model "Summary"` on `--provider llama.relay`; override with
  `--model` / `--provider` if your setup uses different names. The prompt is
  piped via stdin (it can exceed ARG_MAX).

## Notes

- **Deterministic compression, non-deterministic resume.** `plain_compress.py`
  and the compression step of `llm_compress.py` share one engine and are
  byte-identical across runs; the LLM continuation is not (it's a live model call).
- **Coverage guarantee.** Every reasoning-event category present in the input
  appears in the output (at least one kept sentence per category), so the
  compressed thinking structurally retains what was accepted, rejected, left
  open, and where things stand.
- **Auditability.** `--ledger-json` writes a position-tied event ledger — useful
  for debugging selections or running deterministic fact-coverage checks without
  any LLM.
- **Generic by design.** No project, git, or scenario is hardcoded in the prompt
  — it works on any coding-session thinking.
- **One LLM call, always.** Ground truth changes *what* that single call does
  (adds self-scoring), never *how many* calls are made.
