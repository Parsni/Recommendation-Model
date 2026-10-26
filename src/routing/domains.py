"""Shared prompt-domain detection for the routing engine.

Single source of truth used by:
  - the classifier ensemble (domain affinity features),
  - the context bridge (one-hot domain features),
  - the accuracy evaluation (empirical best-model-per-category metric).

Order matters: the first matching domain wins, most specific buckets first.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Canonical domain vocabulary (append-only; changing order changes detection).
DOMAINS: List[str] = [
    "math",
    "code",
    "creative",
    "translation",
    "doc",
    "comm",
    "qa",
    "general",
]

_PATTERNS: Dict[str, re.Pattern] = {
    "math": re.compile(
        r"\b(integral|derivative|solve|calculus|proof|theorem|equation|eigenvalue"
        r"|rational|irrational|riemann|combinatorics|probability|statistics"
        r"|algebra|geometry|matrix|vector|matrixes|matrices|factorial|modular"
        r"|math|numerical|polynomial|differentiate|integrate)\b",
        re.IGNORECASE,
    ),
    "code": re.compile(
        r"\b(python|script|function|def |class |sql|react|component|typescript"
        r"|javascript|refactor|debug|api|docker|kubernetes|redis|algorithm"
        r"|multiprocessing|code|java|c\+\+|regex|bash|shell|linux|git|css|html"
        r"|node|compiler|stack trace|exception|unit test|pytest|pandas|numpy"
        r"|database|query|endpoint|backend|frontend|compile|bug)\b",
        re.IGNORECASE,
    ),
    "creative": re.compile(
        r"\b(story|poem|haiku|scifi|sci-fi|fiction|narrative|novel|character"
        r"|creative|essay|metaphor|lyrics|screenplay|plot|protagonist|worldbuild"
        r"|dialogue|song|poetry|sonnet|villain|hero|fable|myth)\b",
        re.IGNORECASE,
    ),
    "translation": re.compile(
        r"\b(translate|translation|translating|localize|localisation|localization"
        r"|keigo|in japanese|in spanish|in french|in german|in hindi|in chinese"
        r"|to japanese|to spanish|to french|to german|to hindi|to chinese"
        r"|english to|from english|bilingual)\b",
        re.IGNORECASE,
    ),
    "doc": re.compile(
        r"\b(50-page|100-page|regulatory|compliance|pdf|financial report"
        r"|annual report|large document|summarize|summary|summarise|tl;dr"
        r"|transcript|whitepaper|white paper|report|bullet points|key takeaways"
        r"|abstract|synopsis|documentation)\b",
        re.IGNORECASE,
    ),
    "comm": re.compile(
        r"\b(email|diplomatic|contract|vendor|formal letter|greeting|corporate"
        r"|cover letter|apology|memo|outreach|follow-up|follow up|resignation"
        r"|complaint|announcement|newsletter|linkedin|dear (sir|madam|team|all))\b",
        re.IGNORECASE,
    ),
    "qa": re.compile(
        r"\b(capital|population|what is|what are|how do i|how does|workout"
        r"|nutrition|fat loss|quick facts|who is|who was|when did|where is"
        r"|why do|explain|define|difference between|list of)\b",
        re.IGNORECASE,
    ),
}


def detect_domain(text: str) -> str:
    """Classify a prompt into exactly one of ``DOMAINS`` (default: general)."""
    if not text:
        return "general"
    for domain in ("math", "code", "creative", "translation", "doc", "comm", "qa"):
        if _PATTERNS[domain].search(text):
            return domain
    return "general"


def domain_one_hot(domain: str) -> List[float]:
    """Return a 0/1 vector over :data:`DOMAINS` for the given domain label."""
    return [1.0 if d == domain else 0.0 for d in DOMAINS]
