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

*** WIRED INTO agent.py's run_task() since 2026-09-12. *** The wiring
happened live on AI-Server with Mark's explicit confirm per the
source-edit rule, after llama3.2:3b was pulled and benched on the isolated
GPU (127.0.0.1:11435). The sequencing caution in the next paragraph still
applies to any future heuristic change: bench against realistic traffic
first.

Real risk this heuristic exists to manage, stated plainly rather than
glossed over: a false "fast" classification sends a genuinely complex
request to a weak model - which is the exact fabrication risk this whole
roadmap exists to reduce, not add. That's why this errs toward "full" on
anything ambiguous rather than trying to be clever - a wrongly-escalated
simple request only costs a little latency; a wrongly-fast-tracked complex
one risks a bad or fabricated answer.
"""

import re
import pathlib

# Real incident, 2026-09-12: a plain "open the X face" command went FAST
# (short, no complex keyword) and llama3.2:3b picked show_face (just
# refocus the window) over set_face (actually switch) - twice in one
# night, even after the fast-lane prompt was clarified. set_face on the
# full 30B model got it right both times it ran instead. A bare "show me
# your face"/"show your status" (no switch implied) is left alone here -
# that's genuinely fine on the fast lane, and is the existing simple
# pattern below. Only a request that names/implies switching to a
# specific face is forced FULL, regardless of length.
_FACE_SWITCH_RE = re.compile(
    r"\b(open|switch(?:\s+to)?|change|set|bring up|pull up)\b.{0,30}\bface\b",
    re.IGNORECASE,
)

# Real gap flagged 2026-09-16: a request that names a face but drops the
# literal word "face" entirely ("pull up circuit", "bring up aether") never
# matched _FACE_SWITCH_RE above and slipped to the fast lane, which is the
# same unreliable-tool-pick failure the module docstring already warns
# about. Fix: read the real face names straight off visualizer/faces/ (same
# trick tools.py's list_faces() already uses) so this stays correct as the
# face library grows, instead of hardcoding a name list that goes stale.
_VISUALIZER_FACES_DIR = pathlib.Path(__file__).parent / "visualizer" / "faces"


def _face_names() -> list[str]:
    if not _VISUALIZER_FACES_DIR.is_dir():
        return []
    return sorted(p.name for p in _VISUALIZER_FACES_DIR.iterdir() if p.is_dir())


def _face_name_switch_pattern() -> "re.Pattern | None":
    names = _face_names()
    if not names:
        return None
    # Allow a face's own separator (_ or -) to also match a spoken space,
    # since "blue_board" is far more likely spoken as "blue board".
    alternation = "|".join(
        re.escape(name).replace(r"\_", "[_ ]").replace(r"\-", "[\\- ]")
        for name in names
    )
    return re.compile(
        r"\b(open|switch(?:\s+to)?|change|set|bring up|pull up)\b.{0,15}\b("
        + alternation
        + r")\b",
        re.IGNORECASE,
    )


_FACE_NAME_SWITCH_RE = _face_name_switch_pattern()

# 2026-09-25: face EDITS (restyle, add a feature, tweak a value) are a
# different job from face SWITCHES above, and this project's own incident
# history shows granite4:tiny-h is specifically bad at them - the circuit
# clobber, the aether->blue_board restyle-vs-switch confusion, the
# swarm-doubling edit that hit the wrong file. Route this shape to
# qwen3-coder:30b instead: face edits aren't latency-sensitive spoken
# answers, they're deliberate, occasional customization Mark can afford
# to wait ~20-30s longer for in exchange for a model that's actually good
# at precise file edits. Deliberately loose (verb + style/content noun
# anywhere in the message, no strict ordering) since a false positive here
# just means qwen3-coder answers a request that wasn't really a face
# edit - a mild cost, nothing like the FAST/FULL fabrication risk.
_FACE_EDIT_VERBS_RE = re.compile(
    r"\b(add|change|edit|restyle|modify|update|give|make|increase|"
    r"decrease|double|adjust)\b",
    re.IGNORECASE,
)
_FACE_EDIT_NOUNS_RE = re.compile(
    r"\b(color|colour|motif|palette|scheme|hue|tint|lines?|feature|"
    r"swarm|particles?|density|font|trails?|orbs?)\b",
    re.IGNORECASE,
)


def _mentions_a_face(text: str) -> bool:
    if re.search(r"\bface\b", text, re.IGNORECASE):
        return True
    return any(re.search(rf"\b{re.escape(n)}\b", text, re.IGNORECASE) for n in _face_names())


def _is_face_edit(text: str, active_face: str | None = None) -> bool:
    """`active_face`, when given, is whatever face is actually live right
    now (tools._active_face_name()). Real incident, 2026-09-25: "make the
    main orbs have more of a trail" is exactly EDIT-shaped (verb + visual
    noun) but never names a face or says the word "face" at all - it
    leans on Mark having just been talking about the orbit face, which
    this router can't see since it only ever looks at one utterance at a
    time. Rather than teach classify() to read conversation history
    itself, the caller (agent.py's run_task, which already tracks the
    live face) passes it in - a verb+noun match with a known active face
    is treated as an edit even with no explicit face reference, since
    "no face named and no face implied" is the only genuinely ambiguous
    case left once both signals are missing."""
    if not (_FACE_EDIT_VERBS_RE.search(text) and _FACE_EDIT_NOUNS_RE.search(text)):
        return False
    if _mentions_a_face(text):
        return True
    return bool(active_face)

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

# Memory questions must never fast-path: the fast lane deliberately loads
# no vault/task/lesson context (see FAST_SYSTEM_PROMPT in agent.py), so a
# memory question answered there reads as the assistant "forgetting" Mark
# (reported 2026-09-13). A false positive here just costs a little latency
# on the 30B; a false negative is a confident answer from nothing.
_MEMORY_KEYWORDS_RE = re.compile(
    r"\b("
    r"remember|remembers|remembered|recall|recalled|"
    r"forgot|forgotten|forget|"
    r"yesterday|earlier|previously|"
    r"last (night|week|time)|"
    r"\btodo\b|to-?do list|my tasks?|my notes?|"
    r"what did i|what have i"
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
EDIT = "edit"


def classify(text: str, active_face: str | None = None) -> str:
    """Return FAST, FULL, or EDIT for a given request. Pure function, no
    model call, sub-millisecond - the entire point is to decide BEFORE
    paying for any model's time. Errs toward FULL when unsure (see module
    docstring for why). EDIT (see _is_face_edit above) is checked first
    since it's the most specific signal - a request that both edits a
    face AND happens to contain a switch verb (e.g. "change aether's
    color") is an edit, not a switch.

    `active_face`, when given, only ever helps an already verb+noun-shaped
    edit request through when it names no face itself - it can't turn an
    unrelated sentence into an edit on its own. See _is_face_edit."""
    stripped = text.strip()
    if not stripped:
        return FAST  # empty/no-op input, nothing to reason about either way

    if _is_face_edit(stripped, active_face):
        return EDIT

    if _FACE_SWITCH_RE.search(stripped):
        return FULL

    if _FACE_NAME_SWITCH_RE is not None and _FACE_NAME_SWITCH_RE.search(stripped):
        return FULL

    if _COMPLEX_KEYWORDS_RE.search(stripped):
        return FULL

    if _MEMORY_KEYWORDS_RE.search(stripped):
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
