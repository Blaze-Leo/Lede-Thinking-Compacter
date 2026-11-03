"""Text cleaners: clean_text (markdown + filler + boilerplate) and strip_think
(reasoning-model <think>...</think> blocks).

Both functions are deterministic and stdlib-only.
"""
import re

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and trim surrounding whitespace."""
    if text is None:
        return ""
    return _THINK_RE.sub("", text).strip()


# --- clean_text — pipeline ---
#
# Steps in order:
#   1. Strip markdown (*, _, #, ---, bullets, numbered list prefixes)
#   2. Remove filler phrases
#   3. Remove filler words
#   4. Remove CRM boilerplate
#   5. Lowercase
#   6. Collapse whitespace and blank lines
#   7. Trim

_FILLER_PHRASES = re.compile(
    r"\b(just wanted to|i just wanted to|wanted to follow up|as discussed"
    r"|per our conversation|as mentioned|going forward|at the end of the day"
    r"|in terms of|with respect to|in regards to|please find attached"
    r"|hope this helps|let me know if you have any questions"
    r"|looking forward to hearing from you)\b",
    re.IGNORECASE,
)

_FILLER_WORDS = re.compile(
    r"\b(basically|essentially|actually|literally|honestly|frankly"
    r"|obviously|clearly|simply|really|very|quite|rather|pretty much"
    r"|kind of|sort of|in order to|due to the fact that"
    r"|at this point in time|for all intents and purposes)\b",
    re.IGNORECASE,
)

_CRM_PATTERNS = [
    re.compile(r"\bNo update[s]?\b\.?", re.IGNORECASE),
    re.compile(r"\bCalendar invite sent[.]?\b", re.IGNORECASE),
    re.compile(
        r"\bSent (proposal|case study|documentation|overview|pricing)"
        r" (documentation |via email|as requested)?\.?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bWaiting (on|for) callback\.?\b", re.IGNORECASE),
    re.compile(r"\bUpdated CRM with latest info\.?\b", re.IGNORECASE),
    re.compile(r"\bMeeting confirmed for next week\.?\b", re.IGNORECASE),
    re.compile(r"\bFollowing standard sales process\.?\b", re.IGNORECASE),
    re.compile(r"\bMeeting went as expected\.?\b", re.IGNORECASE),
]


def clean_text(text: str | None) -> str:
    """Strip markdown, filler phrases, filler words, and CRM boilerplate;
    lowercase and normalize whitespace.

    Returns the empty string for None or empty input.
    """
    if not text:
        return ""

    result = text

    # 1. Markdown formatting
    result = re.sub(r"\*{1,3}", "", result)
    result = re.sub(r"_{1,3}", "", result)
    result = re.sub(r"^#{1,6}\s*", "", result, flags=re.MULTILINE)
    result = re.sub(r"^-{3,}$", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*[-*+]\s+", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*\d+\.\s+", "", result, flags=re.MULTILINE)

    # 2-3. Filler
    result = _FILLER_PHRASES.sub("", result)
    result = _FILLER_WORDS.sub("", result)

    # 4. CRM boilerplate
    for pattern in _CRM_PATTERNS:
        result = pattern.sub("", result)

    # 5. Lowercase
    result = result.lower()

    # 6. Whitespace normalization
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n\s*\n+", "\n", result)
    result = re.sub(r"^\s+", "", result, flags=re.MULTILINE)
    result = re.sub(r"\s+$", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*$\n?", "", result, flags=re.MULTILINE)

    # 7. Trim
    return result.strip()
