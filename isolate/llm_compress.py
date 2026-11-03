#!/usr/bin/env python3
"""Compress model thinking, then resume it through a SINGLE LLM call.

Pipeline:
  1. Deterministically compress the raw thinking to --target (reuses
     plain_compress.compress — no LLM).
  2. Make ONE pi -p call: hand the compressed thinking + a generic "resume"
     prompt; the model plans its next actions and reports state.

Optionally, with --ground-truth FILE, that SAME single call also receives the
completed-work description and self-classifies each action as redundant /
necessary / neutral (plus an RAR). No ground truth — no scoring, just the
continuation. Either way: exactly one LLM call.

Input is raw thinking text (via --text or stdin); output is the LLM result on
stdout. The compressed text is intermediate and only shown under --debug.

Usage:
    cat thinking.md | llm_compress.py --target 0.05
    cat thinking.md | llm_compress.py --ground-truth done.md --model "Summary"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plain_compress import compress  # noqa: E402

PI_TIMEOUT = 60

# Generic continuation prompt — no project, no git, no hardcoded scenario.
CONTINUATION_PROMPT = """\
Below is the internal thinking of an AI coding agent from earlier in a coding \
session. The session ended immediately after the agent finished its work.

=== THINKING FROM EARLIER IN THE SESSION ===
{thinking}
=== END THINKING ===

The user now asks you to resume this session and continue the work. You have \
NOT yet performed any actions in this turn.

Reply in exactly this format:
ACTIONS: a numbered list (max 8) of the exact actions/commands you would perform next, in order.
REPORT: what you would tell the user, in at most 3 sentences (what is done, what is blocked or open).
"""

# Appended only when --ground-truth is given; scored in the SAME single call.
JUDGE_SUFFIX = """

You are also given the work that was ALREADY COMPLETED in the original session.
After ACTIONS and REPORT, add a third section:

SCORE: for each numbered action above, classify it as exactly one of
- "redundant": re-does work already completed (re-investigating, re-reading \
files to understand from scratch, re-editing files already fixed, re-running \
verification that already passed)
- "necessary": advances the user's request
- "neutral": neither
Then give RAR: <redundant count>/<total actions>.

=== WORK ALREADY COMPLETED (ground truth) ===
{ground_truth}
=== END GROUND TRUTH ===
"""


def run_pi(prompt: str, model: str, provider: str | None) -> tuple[str | None, str]:
    """One pi -p call; prompt piped via stdin (prompts can exceed ARG_MAX)."""
    cmd = ["pi", "-p"]
    if provider and "/" not in model:
        cmd += ["--provider", provider]
    cmd += ["--model", model]
    try:
        p = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=PI_TIMEOUT, cwd=str(HERE),
        )
        if p.returncode != 0:
            return None, (p.stderr or p.stdout)[:500]
        return p.stdout.strip(), ""
    except subprocess.TimeoutExpired:
        return None, f"timeout after {PI_TIMEOUT}s"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", help="raw thinking text (or pass via stdin)")
    ap.add_argument("--target", type=float, default=0.05,
                    help="compression target as a fraction of input (default 0.05)")
    ap.add_argument("--dedup", type=float, default=0.85,
                    help="Jaccard threshold for collapsing repeated reasoning (default 0.85)")
    ap.add_argument("--model", default="Summary",
                    help='pi model to use (default "Summary")')
    ap.add_argument("--provider", default="llama.relay",
                    help="pi provider (default llama.relay; ignored if --model has a '/' prefix)")
    ap.add_argument("--ground-truth", default=None,
                    help="optional path to a file describing work already completed; "
                         "enables self-scoring in the same single LLM call")
    ap.add_argument("--debug", action="store_true",
                    help="print compression stats + the full prompt sent to pi (to stderr)")
    a = ap.parse_args(argv)

    text = a.text if a.text is not None else sys.stdin.read()

    # 1) deterministic compression (no LLM)
    compressed, st = compress(
        text, dedup_threshold=a.dedup, target_chars=int(len(text) * a.target)
    )
    if a.debug:
        pct = 100 * st["output_chars"] / max(1, st["input_chars"])
        print(f"[compress] kept {st['kept_sentences']} sents -> {st['output_chars']} chars "
              f"({pct:.1f}% of input)", file=sys.stderr)

    # 2) build the single prompt (ground truth folds into the SAME call)
    prompt = CONTINUATION_PROMPT.format(thinking=compressed)
    if a.ground_truth:
        gt = Path(a.ground_truth).read_text(encoding="utf-8")
        prompt += JUDGE_SUFFIX.format(ground_truth=gt)

    if a.debug:
        print(f"[prompt] {len(prompt)} chars, model={a.model} provider={a.provider}", file=sys.stderr)
        print("[prompt] ---- begin ----", file=sys.stderr)
        print(prompt, file=sys.stderr)
        print("[prompt] ---- end ----", file=sys.stderr)

    # 3) exactly one LLM call
    out, err = run_pi(prompt, a.model, a.provider)
    if out is None:
        print(f"LLM call FAILED: {err}", file=sys.stderr)
        return 1
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
