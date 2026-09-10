"""
Fast-path speed router — item 3 of the 2026-09-10 four-part roadmap (see
'Local Mary Clone' Session 38 in Mark's Mary vault, 06 - Resources).

Goal: right now every request, simple or complex, hits qwen3-coder:30b.
Voice UX lives and dies on latency, so a genuinely simple command ("what
time is it") pays the same cost as a real coding question. This module is
step one of the fix — a cheap, zero-model-call heuristic that classifies a
request as "fast" (route to a small, near-instant model) or "full" (route
to the 30B daily driver), built and bench-tested BEFORE reaching for a
second model or a learned classifier, per the roadmap's explicit
sequencing: start dumb and cheap, only escalate if the heuristic proves
too sloppy in real use.

*** NOT YET WIRED INTO agent.py. *** This is deliberate — actually routing
requests needs a second, small model pulled and benched on AI-Server first
(folds into the local-LLM-watch item in Active Priorities), and wiring
`classify()` into call_ollama's model selection is a real .py edit to a
live production file, which needs Mark's explicit confirm per the
source-edit rule, done live on AI-Server where it can be verified against
real hardware and a real second model. This file is the scaffolding for
that session to review, bench, and wire in - not a finished feature.

Real risk this heuristic exists to manage, stated plainly rather than
glossed over: a false "fast" classification sends a genuinely complex
request to a weak model - which is the exact fabrication risk this whole
roadmap exists to reduce, not add. That's why this errs toward "full" on
anything ambiguous rather than trying to be clever - a wrongly-escalated
simple request only costs a little latency; a wrongly-fast-tracked complex
one risks a bad or fabricated answer.
"""

import re

# Any of these appearing anywhere in the request is a strong signal it
# needs real reasoning, generation, or multi-step work - never fast-path
# these, no matter how short the message is. Deliberately broad/loose
# (a false positive here just costs a little latency on the 30B; a false
# negative risks a real fabrication-class failure).
_COMPLEX_KEYWORDS_RE = re.compile(
    r"\b("
    r"write|code|debug|fix|refactor|implement|build|design|architect|"
    r"why|explain|analyz\w*|compare|evaluate|review|"
    r"plan|strategy|migrate|"
    r"summar\w*|draft|"
    r"script|function|class|algorithm|regex|query|"
    r"and then|after that|step by step|multi-step"
    r")\b",
    re.IGNORECASE,
)

# Common short, bounded command shapes worth fast-pathing even if they're
# not trivially short - covers the ops-layer monitoring commands from item
# 1 of this same roadmap (status checks, restarts) as well as simple
# assistant asks.
_SIMPLE_PATTERN_RE = re.compile(
    r"^\s*(what(’|'| i)?s|what is|what time|turn (on|off)|set a? ?(timer|reminder|alarm)|"
    r"check (the|your|my)?\s*(gpu|disk|temp|status|backup|task)|"
    r"restart|reboot your|stop listening|list (faces|open tasks|"
    r"tasks)|show (your|me) (face|status)|how (much|many))",
    re.IGNORECASE,
)

# Above this many words, treat as complex by default even with no keyword
# hit - a long message is very rarely a genuinely simple command.
_LONG_MESSAGE_WORD_THRESHOLD = 20

FAST = "fast"
FULL = "full"


def classify(text: str) -> str:
    """Return FAST or FULL for a given request. Pure function, no model
    call, sub-millisecond - the entire point is to decide BEFORE paying
    for any model's time. Errs toward FULL when unsure (see module
    docstring for why)."""
    stripped = text.strip()
    if not stripped:
        return FAST  # empty/no-op input, nothing to reason about either way

    if _COMPLEX_KEYWORDS_RE.search(stripped):
        return FULL

    word_count = len(stripped.split())
    if word_count > _LONG_MESSAGE_WORD_THRESHOLD:
        return FULL

    if _SIMPLE_PATTERN_RE.match(stripped):
        return FAST

    # Short (<=6 words) and no complexity signal at all - very likely a
    # bare simple command ("turn off the lights", "what's the time").
    # Between 7 and _LONG_MESSAGE_WORD_THRESHOLD words with no explicit
    # simple-pattern match and no complex keyword: genuinely ambiguous:
    # err toward FULL per the module's stated risk tradeoff.
    if word_count <= 6:
        return FAST
    return FULL
