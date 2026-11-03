#!/usr/bin/env python3
"""Deterministically compress one model-thinking text block (no ML).

Pipeline:
  1. Remove known agent/extractor scaffolding and pure conversational filler.
  2. Split into sentences using Lede, with a standard-library fallback.
  3. Apply multi-label, rule-based reasoning-event classification.
  4. Deduplicate only near-identical statements with compatible polarity/state.
  5. Select by event coverage, novelty, and chronological head/tail preservation.

The script deliberately does NOT infer claim identity or derive claim status.
Those tasks require reliable semantic/coreference resolution; weak lexical
clusters create misleading histories. The optional JSON ledger therefore
records sentence-level events, each tied to its original position.

Usage:
    cat thinking.md | plain_compress.py --target 0.05 > compressed.md
    plain_compress.py --text "..." --target 0.10 --debug
    cat thinking.md | plain_compress.py --ledger-json events.json --debug
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys

from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    from lede.sentences import split_sentences
    SEGMENTER = "lede"

except ImportError:
    SEGMENTER = "regex-fallback"

    def split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [part.strip() for part in parts if part.strip()]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_THINKING_HEADER = re.compile(
    r"^#{1,6}\s+Thinking\s+\d+\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_META = re.compile(
    r"""AGENTS\.md
    | eth_progress\.md
    | progress\s+log
    | (?:note|notes?)\s+in\s+the\s+log
    | the\s+user\s+(?:is|said|says|told|pushing)
    | i'?ll\s+(?:write|say|mention|mark|keep)
    | let\s+me\s+(?:write|say|be\s+honest|also\s+honestly)
    | to\s+be\s+safe
    | (?:for|in)\s+this\s+round
    | what\s+i\s+should\s+do
    | the\s+right\s+move\s+now
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FILLER_ONLY = re.compile(
    r"^(?:(?:wait|okay|ok|alright|right|hmm+|um+|uh+)[,;:\s]*)*"
    r"(?:let\s+me\s+(?:think|reason)(?:\s+(?:about\s+this|through\s+this|like\s+this))?"
    r"|let(?:'?s|\s+us)\s+(?:think|reason)(?:\s+(?:about\s+this|through\s+this|like\s+this))?"
    r"|i\s+need\s+to\s+(?:think|reason)(?:\s+about\s+this)?"
    r"|give\s+me\s+(?:a\s+)?moment|one\s+moment|let'?s\s+see|thinking)"
    r"[,;:\s.!?]*$",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(
    r"[a-z0-9]+(?:'[a-z]+)?",
    re.IGNORECASE,
)

_STOP = set("""
 a an the and or but if then else when while although though however yet to of
 in on at by for from with without as into onto about over under is are was were
 be been being am do does did doing have has had having it its this that these
 those there here i me my mine we us our ours you your yours he she they them
 their theirs who whom which what where why how can could may might must shall
 should will would just also very more most much many some any each every another
 other same such than too not no nor never n't actually really basically generally
 simply therefore thus hence so maybe perhaps probably possibly likely
""".split())


# ---------------------------------------------------------------------------
# Negation and modality
# ---------------------------------------------------------------------------

_NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|cannot|can't|won't|wouldn't|shouldn't|"
    r"couldn't|doesn't|don't|didn't|isn't|aren't|wasn't|weren't|hasn't|"
    r"haven't|hadn't|without|fails? to)\b",
    re.IGNORECASE,
)

_NEGATION_EXCEPTIONS = re.compile(
    r"\b(?:not\s+(?:only|just|merely|sure|necessarily|impossible)|"
    r"(?:cannot|can't|can not)\s+rule\s+out|"
    r"no\s+(?:problem|issue|risk|limitation|drawback))\b",
    re.IGNORECASE,
)


def _polarity(text: str) -> str:
    cleaned = _NEGATION_EXCEPTIONS.sub(" ", text)

    if _NEGATION.search(cleaned):
        return "NEGATIVE"

    return "POSITIVE"


def _modality(text: str) -> str:
    low = text.lower()

    if re.search(
        r"\b(?:might|may|could|perhaps|possibly|probably|"
        r"uncertain|unclear|not sure)\b",
        low,
    ):
        return "HEDGED"

    if re.search(
        r"\b(?:definitely|certainly|clearly|undoubtedly)\b",
        low,
    ):
        return "EMPHATIC"

    return "ASSERTED"


# ---------------------------------------------------------------------------
# Reasoning-event classification
#
# These patterns are intentionally explicit.
# They cover common coding-agent language without labeling every occurrence
# of words such as "problem", "maybe", "might", or "I think".
# ---------------------------------------------------------------------------

_RULES: dict[str, tuple[re.Pattern[str], ...]] = {

    # Work that remains to be done or information that has not been verified.
    "GAP": (
        re.compile(
            r"\b(?:open question|unresolved|still unknown|not yet verified|"
            r"not yet tested|not yet computed|not yet implemented|"
            r"untested|unverified|pending verification|pending test|"
            r"todo|fixme)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:haven't|have not|hasn't|has not)\s+(?:yet\s+)?"
            r"(?:verified|tested|computed|checked|measured|confirmed|"
            r"determined|run|executed|inspected|reviewed|reproduced)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:still\s+need(?:s)?\s+to|need(?:s)?\s+to|remains?\s+to)\s+"
            r"(?:verify|test|run|compute|check|measure|confirm|investigate|"
            r"inspect|review|trace|reproduce)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:not yet|still haven't|still have not)\s+"
            r"(?:resolved|fixed|validated|confirmed|checked)\b",
            re.IGNORECASE,
        ),
    ),

    # Explicit abandonment or rejection of an approach.
    "REJECTION": (
        re.compile(
            r"\b(?:reject(?:ed|s|ing)?|ruled out|discard(?:ed|s|ing)?|"
            r"abandon(?:ed|s|ing)?|dropped the approach|"
            r"decided against|backed out of)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:will not work|won't work|does not work|doesn't work|"
            r"cannot work|can't work|not viable|unsuitable|not suitable|"
            r"not worth using|shouldn't use|should not use|"
            r"won't use|will not use|not going to use)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:doesn't|does not|cannot|can't|won't|will not)\s+"
            r"(?:solve|address|fix|satisfy|meet|scale|handle)\s+"
            r"(?:the\s+)?(?:requirement|problem|issue|constraint|case)\b",
            re.IGNORECASE,
        ),
    ),

    # Technical failures and concrete objections.
    "OBJECTION": (
        re.compile(
            r"\b(?:does not support|doesn't support|cannot handle|can't handle|"
            r"does not handle|doesn't handle|fails? under|fails? to|"
            r"breaks? when|incompatible with|conflicts? with|unable to)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:introduces?|creates?|causes?)\s+"
            r"(?:a|an|the)\s+"
            r"(?:problem|issue|risk|limitation|drawback|overhead)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:tests?|test suite|build|compilation|type check|lint)\s+"
            r"(?:all\s+)?(?:fail|failed|fails|failing|"
            r"did not pass|does not pass)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:fails?|breaks?|crashes?)\s+"
            r"(?:when|if|under|with|on|during)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:throws?|raises?)\s+(?:an?\s+)?"
            r"(?:[\w.]*?(?:error|exception))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:error|exception|traceback|stack trace)\s+"
            r"(?:occurs?|appears?|is raised|is thrown|shows?|reports?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:race condition|deadlock|timeout|segmentation fault|"
            r"memory leak|crash(?:es|ed)?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:not found|undefined reference|unresolved reference|"
            r"type mismatch|invalid state|incorrect result|unexpected output)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:returns?|produces?)\s+"
            r"(?:none|null|false|an empty result|empty output|"
            r"unexpected|incorrect)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:doesn't|does not|cannot|can't|won't|will not)\s+"
            r"(?:compile|build|pass|work|handle|support|resolve|"
            r"match|parse|load|connect|authenticate|validate)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:unsupported|incompatible|not compatible|not supported)\b",
            re.IGNORECASE,
        ),
    ),

    # Corrections or changes to an existing implementation or conclusion.
    "REVISION": (
        re.compile(
            r"^\s*(?:correction|scratch that|rather than that|"
            r"i was wrong|that was incorrect)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+(?:have\s+)?"
            r"(?:changed|updated|modified|replaced|renamed|moved|"
            r"refactored|rewrote|patched|fixed|reverted)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:changed|updated|modified|replaced|renamed|moved|"
            r"refactored|rewrote|patched|fixed|reverted)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:replace|replacing|replaced)\b.{1,100}\bwith\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:switch|switched|switching)\s+from\b.{1,80}\bto\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i initially|i previously|we initially|we previously)\s+"
            r".{1,100}\b(?:was wrong|was incorrect|needs? changing)\b",
            re.IGNORECASE,
        ),
    ),

    # Reconsidering an earlier direction.
    "RECONSIDERATION": (
        re.compile(
            r"\b(?:on second thought|reconsider(?:ing|ed)?|"
            r"revisit(?:ing|ed)?|re-evaluate|re-evaluating|"
            r"re-examine|re-examining)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:actually|after all)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:looking at|checking|testing|reading|reviewing)\b"
            r".{0,50}\b(?:again|more closely)\b",
            re.IGNORECASE,
        ),
    ),

    # Explicit selection, successful verification, or a working solution.
    "ACCEPTANCE": (
        re.compile(
            r"\b(?:accepted|selected|chosen|settled on|went with)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:let's use|we should use|we will use|we'll use|"
            r"the (?:best|better) choice is|is the (?:best|better) choice)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:tests?|test suite|build|type check|lint)\s+"
            r"(?:now\s+)?(?:pass|passes|passed|are passing|"
            r"is green|succeeded|succeeds)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:passed|passes|succeeded|succeeds)\s+"
            r"(?:all\s+)?(?:the\s+)?"
            r"(?:tests?|build|type checks?|lint)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:compiles?|builds?)\s+(?:successfully|cleanly)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:works?|worked)\s+(?:as expected|correctly|now)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:issue|bug|failure|regression)\s+"
            r"(?:is|was)\s+(?:fixed|resolved)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:fix|patch|change|solution)\s+"
            r"(?:works|worked|passes|is correct|is valid)\b",
            re.IGNORECASE,
        ),
    ),

    # Suggested, intended, or planned work.
    "PROPOSAL": (
        re.compile(
            r"\b(?:could|should|would|might|may)\s+"
            r"(?:try|use|choose|consider|add|remove|change|update|"
            r"inspect|check|test|run|implement|create|modify|"
            r"rewrite|replace|refactor|move|rename|verify)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+(?:could|should|will|would|might|may)\s+"
            r"(?:try|use|choose|consider|add|remove|change|update|"
            r"inspect|check|test|run|implement|create|modify|"
            r"rewrite|replace|refactor|move|rename|verify)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+(?:plan|intend|want|need)\s+to\s+"
            r"(?:try|use|choose|consider|add|remove|change|update|"
            r"inspect|check|test|run|implement|create|modify|"
            r"rewrite|replace|refactor|move|rename|verify)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:propose|suggest|consider|evaluate|explore)\s+"
            r"(?:using\s+)?\w+",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:one option|another option|an alternative approach|what if)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:let's|next,?)\s+"
            r"(?:try|use|add|remove|change|update|inspect|check|"
            r"test|run|implement|create|modify|rewrite|replace|refactor)\b",
            re.IGNORECASE,
        ),
    ),

    # Reasons, mechanisms, dependencies, and causal explanations.
    "SUPPORT": (
        re.compile(
            r"\b(?:because|given that|due to|the reason is|"
            r"as a result of|caused by|stems from|comes from|"
            r"root cause)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:this|that|which)\s+"
            r"(?:means|allows|enables|prevents|ensures|avoids|"
            r"requires|explains|causes|indicates)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:prevents?|avoids?|ensures?|allows?|enables?|"
            r"depends on|results in|leads to|is responsible for)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:so that|in order to|that's why|this is why|"
            r"which is why)\b",
            re.IGNORECASE,
        ),
    ),

    # Explicit conclusions or decisions.
    "CONCLUSION": (
        re.compile(
            r"^\s*(?:therefore|thus|hence|consequently|"
            r"in conclusion|bottom line|in short)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*so\s+(?:we|i)\s+"
            r"(?:should|will|must|can conclude|should choose|should use)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the conclusion is|the key insight is|"
            r"the bottom line is|for that reason)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:this means we should|that means we should|"
            r"the simplest solution is|the best approach is|"
            r"the correct approach is)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:we'll go with|we will go with|"
            r"i'll go with|i will go with)\b",
            re.IGNORECASE,
        ),
    ),

    # Explicit uncertainty rather than arbitrary occurrences of "might".
    "UNCERTAINTY": (
        re.compile(
            r"\b(?:i am|i'm|we are|we're)\s+"
            r"(?:not sure|uncertain|unsure|not confident|not convinced)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:unclear|uncertain|unverified|"
            r"not confident|not established)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:cannot|can't|don't know|do not know)\s+"
            r"(?:whether|if|how|why|what)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i suspect|it appears that|it seems that|"
            r"it looks like|this appears to|this seems to|"
            r"likely cause|probably caused by)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:might|may|could)\s+"
            r"(?:fail|not work|be incorrect|cause a failure|"
            r"introduce a problem|be caused by|be due to|"
            r"indicate|explain|require)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:possibly|probably|likely)\s+"
            r"(?:the|this|that|it|a|an|we|i)\b",
            re.IGNORECASE,
        ),
    ),

    # Alternatives, contrasts, and changes in direction.
    "CONTRAST": (
        re.compile(
            r"^\s*(?:but|however|nevertheless|"
            r"on the other hand|alternatively|instead)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:whereas|although|even though|"
            r"on the contrary|in contrast|rather than)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"[,;]\s*but\b",
            re.IGNORECASE,
        ),
    ),

    # Direct observations and reported evidence.
    "OBSERVATION": (
        re.compile(
            r"\b(?:i|we)\s+"
            r"(?:found|observed|noticed|saw|identified|discovered)\s+"
            r"(?:that\b)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:log|traceback|stack trace|output|"
            r"compiler|test result|test output)\s+"
            r"(?:shows?|reports?|indicates?|reveals?|contains?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:function|method|endpoint|handler|"
            r"parser|query|command|script|class|module)\s+"
            r"(?:returns?|raises?|throws?|calls?|accepts?|rejects?|"
            r"reads?|writes?|creates?|updates?|imports?|defines?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:is defined in|is implemented in|"
            r"is located in|is stored in|is imported from)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the current implementation|currently uses|"
            r"currently calls|currently returns|currently contains|"
            r"currently imports|currently defines)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the error message says|the exception says|"
            r"the test output says|the log reports|"
            r"the traceback shows)\b",
            re.IGNORECASE,
        ),
    ),

    # Actual operations already performed.
    # Kept distinct from PROPOSAL, which represents intended future work.
    "ACTION": (
        re.compile(
            r"^\s*(?:i|we)\s+(?:have\s+)?"
            r"(?:added|implemented|changed|updated|modified|removed|"
            r"deleted|created|renamed|moved|refactored|rewrote|"
            r"patched|fixed|reverted|ran|executed|tested|checked|"
            r"inspected|opened|read|searched|traced|verified|"
            r"confirmed|validated)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:added|implemented|updated|modified|removed|"
            r"deleted|created|renamed|moved|refactored|rewrote|"
            r"patched|fixed|reverted)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:ran|executed)\s+(?:the\s+)?"
            r"(?:tests?|test suite|build|script|command|"
            r"type checker|linter|benchmark)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+(?:just\s+)?"
            r"(?:ran|executed|tested|inspected|checked|"
            r"opened|read|searched|traced)\b",
            re.IGNORECASE,
        ),
    ),
}


_LABEL_WEIGHT = {
    "GAP": 10,
    "REJECTION": 10,
    "ACCEPTANCE": 10,
    "REVISION": 9,
    "RECONSIDERATION": 9,
    "OBJECTION": 9,
    "UNCERTAINTY": 9,
    "CONCLUSION": 9,
    "PROPOSAL": 7,
    "OBSERVATION": 7,
    "ACTION": 6,
    "SUPPORT": 6,
    "CONTRAST": 5,
}


_COVERAGE_ORDER = [
    "ACCEPTANCE",
    "REJECTION",
    "GAP",
    "REVISION",
    "RECONSIDERATION",
    "OBJECTION",
    "UNCERTAINTY",
    "CONCLUSION",
    "PROPOSAL",
    "OBSERVATION",
    "ACTION",
    "SUPPORT",
    "CONTRAST",
]


# ---------------------------------------------------------------------------
# Tokenization and classification
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
    }


def _content_tokens(text: str) -> set[str]:
    return _tokens(text) - _STOP


def _classify(position: int, sentence: str) -> dict:
    labels = sorted(
        label
        for label, patterns in _RULES.items()
        if any(pattern.search(sentence) for pattern in patterns)
    )

    return {
        "position": position,
        "text": sentence.strip(),
        "labels": labels,
        "polarity": _polarity(sentence),
        "modality": _modality(sentence),
        "score": max(
            (_LABEL_WEIGHT[label] for label in labels),
            default=1,
        ),
    }


# ---------------------------------------------------------------------------
# Similarity and deduplication
# ---------------------------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _dedup_key(event: dict) -> tuple:
    """Only compare statements with compatible reasoning states."""

    return (
        tuple(event["labels"]),
        event["polarity"],
        event["modality"],
    )


def _deduplicate(
    events: list[dict],
    threshold: float,
) -> tuple[list[dict], int]:

    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    kept_keys: list[tuple] = []

    removed = 0

    for event in events:

        current = _content_tokens(event["text"])
        current_key = _dedup_key(event)

        duplicate = False

        for previous_tokens, previous_key in zip(
            kept_tokens,
            kept_keys,
        ):

            if current_key != previous_key:
                continue

            shared = current & previous_tokens

            # At least three shared content terms are required.
            # A shared technical noun alone is never sufficient.
            if len(shared) < 3:
                continue

            if _jaccard(current, previous_tokens) >= threshold:
                duplicate = True
                break

        if duplicate:
            removed += 1

        else:
            kept.append(event)
            kept_tokens.append(current)
            kept_keys.append(current_key)

    return kept, removed


# ---------------------------------------------------------------------------
# Optimized coverage-first + MMR selection
# ---------------------------------------------------------------------------

def _select(
    events: list[dict],
    target_chars: int | None,
    head_frac: float,
    tail_frac: float,
) -> tuple[list[dict], int]:

    if not events:
        return [], 0

    if target_chars is None:
        return list(events), 0

    # Non-positive targets retain one useful sentence.
    if target_chars <= 0:

        best = max(
            events,
            key=lambda event: (
                event["score"],
                -event["position"],
            ),
        )

        return [best], len(best["text"])

    n = len(events)

    # -----------------------------------------------------------------------
    # Precompute token sets and compatibility keys exactly once.
    # -----------------------------------------------------------------------

    token_sets = [
        _content_tokens(event["text"])
        for event in events
    ]

    dedup_keys = [
        _dedup_key(event)
        for event in events
    ]

    # Inverted index:
    # (reasoning-state key, token) -> event indices containing that token.
    #
    # Candidates that share no content tokens have zero Jaccard similarity
    # and do not require an update when another event is selected.
    postings: dict[tuple, list[int]] = {}

    for index, tokens in enumerate(token_sets):

        key = dedup_keys[index]

        for token in tokens:
            postings.setdefault(
                (key, token),
                [],
            ).append(index)

    # Maximum similarity to any selected, compatible event.
    max_similarity = [0.0] * n

    # Heap versions allow stale priority entries to be discarded cheaply.
    versions = [0] * n

    # Selected event indices.
    selected_indices: set[int] = set()

    # Maintain selected events in original position order at the end.
    selected: dict[int, dict] = {}

    used = 0
    lam = 0.75

    def mmr_score(index: int) -> float:

        relevance = events[index]["score"] / 10.0
        redundancy = max_similarity[index]

        return (
            lam * relevance
            - (1.0 - lam) * redundancy
        )

    # -----------------------------------------------------------------------
    # Initialize heap.
    #
    # Entries:
    # (-score, original_position, event_index, version)
    #
    # Earlier sentences win ties, preserving deterministic behavior.
    # -----------------------------------------------------------------------

    heap: list[tuple[float, int, int, int]] = []

    for index, event in enumerate(events):

        heap.append((
            -mmr_score(index),
            event["position"],
            index,
            0,
        ))

    heapq.heapify(heap)

    # -----------------------------------------------------------------------
    # Incrementally update candidate similarities when selecting an event.
    # -----------------------------------------------------------------------

    def update_similarities(new_index: int) -> None:

        new_tokens = token_sets[new_index]

        if not new_tokens:
            return

        key = dedup_keys[new_index]

        affected: set[int] = set()

        # Only candidates sharing at least one token can have non-zero
        # Jaccard similarity with the newly selected sentence.
        for token in new_tokens:

            affected.update(
                postings.get((key, token), ())
            )

        for candidate in affected:

            if candidate in selected_indices:
                continue

            similarity = _jaccard(
                token_sets[candidate],
                new_tokens,
            )

            if similarity <= max_similarity[candidate]:
                continue

            max_similarity[candidate] = similarity
            versions[candidate] += 1

            heapq.heappush(
                heap,
                (
                    -mmr_score(candidate),
                    events[candidate]["position"],
                    candidate,
                    versions[candidate],
                ),
            )

    # -----------------------------------------------------------------------
    # Exact incremental output cost.
    #
    # Used by add(), head reservation, and tail reservation.
    # This removes the previous estimated-cost/actual-cost discrepancy.
    # -----------------------------------------------------------------------

    def incremental_cost(event: dict) -> int:

        return len(event["text"]) + (
            1 if selected_indices else 0
        )

    # -----------------------------------------------------------------------
    # Add an event while maintaining an exact character budget.
    # -----------------------------------------------------------------------

    def add(index: int) -> bool:
        nonlocal used

        if index in selected_indices:
            return True

        event = events[index]
        cost = incremental_cost(event)

        if used + cost > target_chars:
            return False

        selected_indices.add(index)
        selected[index] = event

        used += cost

        # Update only affected MMR candidates.
        update_similarities(index)

        return True

    # -----------------------------------------------------------------------
    # 1. Reserve beginning context.
    # -----------------------------------------------------------------------

    head_budget = int(
        target_chars * max(0.0, head_frac)
    )

    head_used = 0

    for index, event in enumerate(events):

        cost = incremental_cost(event)

        if head_used + cost > head_budget:
            break

        if add(index):
            head_used += cost

    # -----------------------------------------------------------------------
    # 2. Reserve ending context.
    # -----------------------------------------------------------------------

    tail_budget = int(
        target_chars * max(0.0, tail_frac)
    )

    tail_used = 0

    for index in range(n - 1, -1, -1):

        if index in selected_indices:
            continue

        event = events[index]
        cost = incremental_cost(event)

        if tail_used + cost > tail_budget:
            break

        if add(index):
            tail_used += cost

    # -----------------------------------------------------------------------
    # 3. Guarantee beginning and ending context when they fit.
    # -----------------------------------------------------------------------

    add(0)
    add(n - 1)

    # -----------------------------------------------------------------------
    # 4. Ensure representation of reasoning-event categories.
    # -----------------------------------------------------------------------

    covered = {
        label
        for event in selected.values()
        for label in event["labels"]
    }

    for label in _COVERAGE_ORDER:

        if label in covered:
            continue

        candidates = [
            index
            for index, event in enumerate(events)
            if (
                label in event["labels"]
                and index not in selected_indices
            )
        ]

        candidates.sort(
            key=lambda index: (
                -events[index]["score"],
                events[index]["position"],
            )
        )

        for candidate in candidates:

            if add(candidate):

                covered.update(
                    events[candidate]["labels"]
                )

                break

    # -----------------------------------------------------------------------
    # 5. Fill remaining budget using incremental MMR.
    #
    # Avoids:
    #   - Rebuilding the selected-events list every iteration.
    #   - Scanning every event to rebuild the available list.
    #   - Recalculating content-token sets.
    #   - Recalculating similarities to all selected sentences.
    #
    # Similarities update only when a newly selected sentence can affect
    # a candidate's maximum redundancy.
    # -----------------------------------------------------------------------

    while heap:

        negative_score, position, index, version = heapq.heappop(heap)

        # Ignore events already selected through head/tail/coverage.
        if index in selected_indices:
            continue

        # Ignore old heap entries after an MMR score update.
        if version != versions[index]:
            continue

        event = events[index]
        cost = incremental_cost(event)

        # The budget only decreases, so an oversized event cannot become
        # affordable later.
        if used + cost > target_chars:
            continue

        add(index)

    # -----------------------------------------------------------------------
    # 6. Guarantee at least one sentence for tiny positive budgets.
    # -----------------------------------------------------------------------

    if not selected:

        best_index = max(
            range(n),
            key=lambda index: (
                events[index]["score"],
                -events[index]["position"],
            ),
        )

        event = events[best_index]

        selected[best_index] = event
        selected_indices.add(best_index)

        used = len(event["text"])

    # Restore original chronology.
    result = sorted(
        selected.values(),
        key=lambda event: event["position"],
    )

    # Exact final output length, including separators.
    actual = (
        sum(len(event["text"]) for event in result)
        + max(0, len(result) - 1)
    )

    overrun = max(
        0,
        actual - target_chars,
    )

    return result, overrun


# ---------------------------------------------------------------------------
# Main compression function
# ---------------------------------------------------------------------------

def compress(
    text: str,
    *,
    dedup_threshold: float = 0.85,
    target_chars: int | None = None,
    head_frac: float = 0.15,
    tail_frac: float = 0.25,
) -> tuple[str, dict]:
    """Return compressed text and statistics."""

    if not 0.0 <= dedup_threshold <= 1.0:
        raise ValueError(
            "dedup_threshold must be between 0 and 1"
        )

    if head_frac < 0 or tail_frac < 0:
        raise ValueError(
            "head_frac and tail_frac must be non-negative"
        )

    original = text

    text = _THINKING_HEADER.sub("\n", text)

    # -----------------------------------------------------------------------
    # Remove complete agent-plumbing lines.
    # -----------------------------------------------------------------------

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    filtered_lines = []
    meta_lines = 0

    for line in lines:

        if _META.search(line):
            meta_lines += 1

        else:
            filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines)

    # -----------------------------------------------------------------------
    # Segment sentences.
    # -----------------------------------------------------------------------

    sentences = [
        sentence.strip()
        for sentence in split_sentences(cleaned)
        if sentence and sentence.strip()
    ]

    # -----------------------------------------------------------------------
    # Remove remaining scaffolding/filler and classify events.
    # -----------------------------------------------------------------------

    events = []

    meta_sentences = 0
    filler_removed = 0

    for position, sentence in enumerate(sentences):

        if _META.search(sentence):
            meta_sentences += 1
            continue

        if _FILLER_ONLY.fullmatch(sentence):
            filler_removed += 1
            continue

        events.append(
            _classify(position, sentence)
        )

    # -----------------------------------------------------------------------
    # Conservative deduplication.
    # -----------------------------------------------------------------------

    distinct, duplicates = _deduplicate(
        events,
        dedup_threshold,
    )

    # -----------------------------------------------------------------------
    # Coverage-first and MMR selection.
    # -----------------------------------------------------------------------

    selected, overrun = _select(
        distinct,
        target_chars,
        head_frac,
        tail_frac,
    )

    output = " ".join(
        event["text"]
        for event in selected
    )

    # -----------------------------------------------------------------------
    # Statistics and sentence-level ledger.
    # -----------------------------------------------------------------------

    label_counts: Counter[str] = Counter()

    for event in events:
        label_counts.update(event["labels"])

    # Sentence-level event ledger only.
    # No speculative claim clusters or inferred statuses.
    ledger = [
        dict(event)
        for event in events
    ]

    stats = {
        "segmenter": SEGMENTER,
        "input_chars": len(original),
        "total_sentences": len(sentences),
        "meta_stripped": meta_lines + meta_sentences,
        "filler_removed": filler_removed,
        "usable_sentences": len(events),
        "duplicates_removed": duplicates,
        "distinct_sentences": len(distinct),
        "kept_sentences": len(selected),
        "capped_dropped": len(distinct) - len(selected),
        "target_chars": target_chars,
        "budget_overrun_chars": overrun,
        "output_chars": len(output),
        "label_counts": dict(sorted(label_counts.items())),
        "ledger": ledger,
    }

    return output, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )

    parser.add_argument(
        "--text",
        help="raw thinking text (or pass via stdin)",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=0.05,
        help="target output size as a fraction of input (default: 0.05)",
    )

    parser.add_argument(
        "--dedup",
        type=float,
        default=0.85,
        help="Jaccard threshold for conservative deduplication (default: 0.85)",
    )

    parser.add_argument(
        "--head-frac",
        type=float,
        default=0.15,
        help="fraction of target reserved for beginning context",
    )

    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.25,
        help="fraction of target reserved for ending context",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="print compression statistics to stderr",
    )

    parser.add_argument(
        "--ledger-json",
        type=Path,
        help="optional path for sentence-level event ledger JSON",
    )

    args = parser.parse_args(argv)

    if not 0.0 <= args.target <= 1.0:
        parser.error("--target must be between 0 and 1")

    if not 0.0 <= args.dedup <= 1.0:
        parser.error("--dedup must be between 0 and 1")

    if args.head_frac < 0 or args.tail_frac < 0:
        parser.error(
            "--head-frac and --tail-frac must be non-negative"
        )

    source = (
        args.text
        if args.text is not None
        else sys.stdin.read()
    )

    output, stats = compress(
        source,
        dedup_threshold=args.dedup,
        target_chars=int(len(source) * args.target),
        head_frac=args.head_frac,
        tail_frac=args.tail_frac,
    )

    # -----------------------------------------------------------------------
    # Optional JSON ledger.
    # -----------------------------------------------------------------------

    if args.ledger_json:

        args.ledger_json.write_text(
            json.dumps(
                stats["ledger"],
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    # -----------------------------------------------------------------------
    # Debug output goes to stderr, never stdout.
    # -----------------------------------------------------------------------

    if args.debug:

        pct = (
            100.0
            * stats["output_chars"]
            / max(1, stats["input_chars"])
        )

        for key in (
            "segmenter",
            "input_chars",
            "total_sentences",
            "meta_stripped",
            "filler_removed",
            "usable_sentences",
            "duplicates_removed",
            "distinct_sentences",
            "kept_sentences",
            "capped_dropped",
            "target_chars",
            "budget_overrun_chars",
            "output_chars",
            "label_counts",
        ):

            print(
                f"{key:22} {stats[key]}",
                file=sys.stderr,
            )

        print(
            f"compression           {pct:.1f}% of input",
            file=sys.stderr,
        )

    sys.stdout.write(
        output + ("\n" if output else "")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
