"""
Tool implementations for the from-scratch agent.

Every tool function takes a single dict of arguments (as parsed from the
model's tool call) and returns a string — that string goes straight back
into the conversation as the tool's result, so keep output readable, not a
raw repr.
"""

import datetime
import difflib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid

import httpx

import ops_monitor

# Everything the agent touches is scoped under this folder so a stray tool
# call can't wander the whole filesystem. Paths passed by the model are
# resolved relative to here unless already absolute.
WORKSPACE = pathlib.Path(__file__).parent.resolve()

# Code-level gate on write_file/edit_file for source files, added
# 2026-09-09 after a real incident (a NameError crash in voice.py) needed
# Mary, not Jarvis, to root-cause and fix - Mark asked to give Jarvis that
# same troubleshooting ability. Matches the run_shell soft-block pattern
# already proven necessary on this model: a prompt-only "confirm before
# editing code" rule doesn't reliably hold, so the friction has to live in
# code. Scoped to .py for now (the concrete case that prompted this); widen
# this set if another source-file type needs the same protection later.
_SOURCE_CODE_SUFFIXES = {".py"}


def _source_edit_block_message(path: pathlib.Path, tool_name: str) -> str:
    return (
        f"BLOCKED: {path} is source code (.py). Writing to it is never "
        "allowed on the first call - same rule as run_shell's "
        "disruptive-action gate, see the 2026-09-09 troubleshooting-gate "
        "entry in INDEX.md. Answer directly now (no more tool calls this "
        "turn) stating the exact change and what it does, then wait for "
        f"Mark to actually reply yes. Only if he does, call {tool_name} "
        "again with the same arguments plus confirmed=true - not before."
    )

# The vault: a lightweight version of Mary's own memory pattern (an
# always-loaded index + dated daily notes) instead of one flat file.
# Scaled down on purpose - no folder-per-domain taxonomy, no wikilinks,
# no frontmatter typing. Those solve a scale problem (many machines, many
# months, many notes) this project doesn't have yet.
VAULT_DIR = WORKSPACE / "vault"
INDEX_FILE = VAULT_DIR / "INDEX.md"
DAILY_DIR = VAULT_DIR / "daily"
LESSONS_FILE = VAULT_DIR / "LESSONS.md"
TASKS_FILE = VAULT_DIR / "TASKS.md"

DAILY_TEMPLATE = "# {date}\n\n"
LESSONS_TEMPLATE = (
    "# Lessons Learned\n\n"
    "A running chronicle of real fixes, working methods, and dead ends to\n"
    "skip - separate from the daily notes, which log what happened. This\n"
    "file logs what was learned, so a later run doesn't pay the discovery\n"
    "tax twice. Append with `append_lesson`; a tail of this file is "
    "auto-loaded into every run, same as INDEX.md.\n"
)
TASKS_TEMPLATE = (
    "# Tasks\n\n"
    "Real persistent tracking of open and completed work across "
    "sessions - not something to ever claim you don't have. The Open "
    "section auto-loads into every run's system prompt, same as "
    "INDEX.md and LESSONS.md's tail. Add an item with `add_task` when "
    "something's left open or worth tracking; close it with "
    "`complete_task` when it's actually done, don't leave it stale. "
    "Check `list_open_tasks` if asked what's outstanding.\n\n"
    "## Open\n\n"
    "## Completed\n"
)


# --- Write-time spam/duplicate guard for append_daily_note and
# append_lesson ---
#
# Root cause this exists for (2026-09-09): the model, stuck making no
# real progress, repeatedly logged near-identical "still blocked"
# status entries to LESSONS.md and the daily note instead of trying
# something different - 20+ entries in one case, 6 in 3 minutes in the
# other. Since a tail of both files auto-loads into every run's system
# prompt, that noise crowded out the real answer and got parroted back
# as fact. This guard stops the write itself instead of relying on a
# human to notice and clean it up after the fact.

_NEAR_DUP_THRESHOLD = 0.6  # SequenceMatcher ratio on normalized text
_SPAM_WINDOW_MINUTES = 5
_SPAM_MAX_IN_WINDOW = 3


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _split_entry_bodies(text: str) -> list:
    """Split a LESSONS.md/daily-note body into a list of entry bodies, one
    per '## ' header block, in file order. Content before the first
    header (title/intro) is dropped."""
    parts = re.split(r"(?m)^## .*$", text)
    return [p.strip() for p in parts[1:] if p.strip()]


def _extract_headers(text: str) -> list:
    return re.findall(r"(?m)^## (.*)$", text)


def _write_guard(existing_text: str, new_text: str, parse_ts) -> str:
    """Returns a BLOCKED message string if the write should be refused, or
    '' if it's fine to proceed. parse_ts(header_str) -> datetime|None is
    file-format-specific (daily note headers are bare 'HH:MM', LESSONS.md
    headers are 'YYYY-MM-DD HH:MM — topic')."""
    bodies = _split_entry_bodies(existing_text)
    if not bodies:
        return ""

    new_norm = _normalize_for_compare(new_text)
    for body in bodies[-_SPAM_MAX_IN_WINDOW:]:
        ratio = difflib.SequenceMatcher(None, new_norm, _normalize_for_compare(body)).ratio()
        if ratio >= _NEAR_DUP_THRESHOLD:
            return (
                f"BLOCKED: this is a near-duplicate ({ratio:.0%} similar) of "
                "an entry you already logged. Don't log the same status "
                "again - either try a genuinely different approach, check "
                "if the answer's already recorded (search_vault / "
                "read_file), or just answer directly instead of logging "
                "another status update."
            )

    headers = _extract_headers(existing_text)
    now = datetime.datetime.now()
    recent_count = 0
    for h in headers[-_SPAM_MAX_IN_WINDOW:]:
        ts = parse_ts(h)
        if ts and (now - ts) <= datetime.timedelta(minutes=_SPAM_WINDOW_MINUTES):
            recent_count += 1
    if recent_count >= _SPAM_MAX_IN_WINDOW:
        return (
            f"BLOCKED: {recent_count} entries already logged to this file "
            f"in the last {_SPAM_WINDOW_MINUTES} minutes - that's spinning, "
            "not progress. Stop logging status updates and either answer "
            "the question directly or try something concretely different."
        )
    return ""


# --- Self-incapacity claim guard for append_daily_note and append_lesson ---
#
# Root cause this exists for (2026-09-10): a self-diagnosed "I can't do X"
# claim about the agent's own tool-calling ability got logged straight to
# LESSONS.md via a normal, successful append_lesson call - no malformed
# tool call happened first, so the existing give-up retry gate in
# agent.py (which only watches the spoken final answer) never saw it.
# The claim was false - tool calls work fine, append_lesson itself proves
# it - but because LESSONS.md's tail auto-loads into every later run, it
# sat there being treated as fact until Mary caught and corrected it.
# Same root problem as the 2026-09-08 fabricated-facts-about-Mark
# incident, just landing through a different tool. Fix: a claim shaped
# like "tool calls / the interface can't do X" requires verified=true -
# proof the caller actually read the relevant code or a real traceback
# first (see INDEX.md's "Debugging a crash" playbook), not just a
# plausible-sounding guess.

_CAPABILITY_PHRASE = (
    r"(?:tool[\s\-]?calls?|tool[\s\-]?execution|function[\s\-]?calls?|"
    r"the interface|this interface)"
)
_NEGATION_PHRASE = (
    r"(?:can'?t|cannot|unable|persistent failure|limitation|doesn'?t work|"
    r"not (?:something|possible)|prevents? (?:\w+\s+)?from|"
    r"misinterpret\w*|interpretation issue|isn'?t (?:being )?"
    r"(?:recognized|received|processed|going through))"
)
_GAP = r"(?:(?!\.\s|\n).){0,150}?"

# Matches either word order - "tool calls ... can't" (the original
# shape) or "unable ... tool calls" (found live 2026-09-11: a false
# "unable to execute tool calls" lesson got past the original
# capability-first-only pattern, poisoned LESSONS.md, and triggered a
# real fabrication spiral on the very next turn - the negation word
# coming first was never covered before).
_SELF_INCAPACITY_RE = re.compile(
    rf"(?:{_CAPABILITY_PHRASE}{_GAP}{_NEGATION_PHRASE}"
    rf"|{_NEGATION_PHRASE}{_GAP}{_CAPABILITY_PHRASE})",
    re.IGNORECASE | re.DOTALL,
)


def _capability_claim_guard(text: str, verified: bool) -> str:
    """Returns a BLOCKED message if `text` claims the agent itself (its
    tool-calling, its interface) can't do something, and that claim
    hasn't been marked verified. '' if fine to proceed."""
    if verified:
        return ""
    if _SELF_INCAPACITY_RE.search(text):
        return (
            "BLOCKED: this reads as a claim that tool calls or the "
            "interface itself is broken/limited - that exact kind of "
            "claim has been fabricated before (2026-09-10) and turned "
            "out false; tool calls work. Before logging this, actually "
            "read the relevant code or a real traceback (see INDEX.md's "
            "'Debugging a crash' playbook) - if you've genuinely done "
            "that and it's grounded in real evidence, call this again "
            "with verified=true. If you haven't verified it, don't log "
            "it as a lesson at all - just say plainly you don't know "
            "why something didn't work."
        )
    return ""


def _parse_daily_header_ts(header: str) -> object:
    m = re.fullmatch(r"(\d{2}):(\d{2})", header.strip())
    if not m:
        return None
    today = datetime.date.today()
    try:
        return datetime.datetime(today.year, today.month, today.day, int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None


def _parse_lesson_header_ts(header: str) -> object:
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", header.strip())
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _today_daily_path() -> pathlib.Path:
    today = datetime.date.today().isoformat()
    return DAILY_DIR / f"{today}.md"


def _ensure_today_daily_note() -> pathlib.Path:
    """Auto-create today's daily note from a minimal template if it
    doesn't exist yet - mirrors Mary's own 'create from template if
    missing' habit."""
    path = _today_daily_path()
    if not path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            DAILY_TEMPLATE.format(date=datetime.date.today().isoformat()),
            encoding="utf-8",
        )
    return path


def _resolve(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else (WORKSPACE / p)


def _memory_file_block_message(path: pathlib.Path) -> str | None:
    """BLOCKED message if `path` is one of Jarvis's memory files, None
    otherwise. Real incident, 2026-09-20: asked to double the swarm in the
    orbit face, the model never looked at the face; it searched his notes,
    found an old LESSONS.md entry, replaced the whole paragraph with the
    text 'swarm scaling (double)' via edit_file, and reported the swarm
    doubled. The vault isn't in git, so that write destroyed the entry
    (restored from the backup mirror). Each memory file has its own
    purpose-built tool (append_lesson, add_task/complete_task,
    append_daily_note) that adds entries safely; the generic write/edit
    tools have no business rewriting them. No confirmed=true override, same
    reasoning as the shrink block: the model can set that flag itself, so it
    can't be what protects a file. INDEX.md is deliberately NOT covered
    here (it has no dedicated append tool, and Mark may want Jarvis able to
    update it); widen this if that turns out to need the same protection."""
    try:
        resolved = os.path.normcase(str(path.resolve()))
    except OSError:
        return None
    daily = os.path.normcase(str(DAILY_DIR.resolve())) + os.sep
    if resolved == os.path.normcase(str(LESSONS_FILE.resolve())):
        tool = "append_lesson"
    elif resolved == os.path.normcase(str(TASKS_FILE.resolve())):
        tool = "add_task / complete_task"
    elif resolved.startswith(daily):
        tool = "append_daily_note"
    else:
        return None
    return (
        f"BLOCKED: {path} is one of your own memory files. It is "
        f"append-only through its own tool ({tool}); write_file and "
        "edit_file can never change it, because a replace destroys "
        "entries and the vault isn't in git. This block has NO "
        "confirmed=true override. If an entry there is wrong and needs "
        "removing or rewording, tell Mark and have Mary do it. If you "
        "meant to change something else (a face, a config), you are "
        "looking in the wrong place: read that file directly first."
    )


def read_file(args: dict) -> str:
    path = _resolve(args["path"])
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: no such file: {path}"
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def _python_syntax_error(content: str, path: pathlib.Path) -> str | None:
    """Returns a human-readable syntax error if `content` isn't valid
    Python, None if it compiles clean. Real incident, 2026-09-17: the
    model wrote a syntactically broken function straight over the real
    tools.py (a stray colon inside an `if`) and then claimed it
    'installed successfully' without ever running it. This makes that
    exact failure structurally impossible for .py writes - broken code
    never reaches disk in the first place, so there's nothing left to
    falsely claim success about."""
    try:
        compile(content, str(path), "exec")
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    return None


_SHRINK_GUARD_MIN_BYTES = 2000
_SHRINK_GUARD_RATIO = 0.2


def _shrink_block_message(path: pathlib.Path, content: str) -> str | None:
    """Returns a BLOCKED message if this write would replace a substantial
    existing file with something under 20% of its size. Real incident,
    2026-09-18: asked to work on the circuit face, the model overwrote the
    39KB index.html with a 504-byte CSS snippet via write_file, then
    claimed the face was edited. A genuine rewrite rarely shrinks a file
    that hard; a clobber always does."""
    try:
        old_size = path.stat().st_size
    except OSError:
        return None
    new_size = len(content.encode("utf-8"))
    if old_size < _SHRINK_GUARD_MIN_BYTES or new_size >= old_size * _SHRINK_GUARD_RATIO:
        return None
    return (
        f"BLOCKED: {path} is currently {old_size} bytes and this write "
        f"would replace it with only {new_size} bytes - that wipes out "
        "almost everything in it. write_file overwrites the WHOLE file. "
        "To change part of a file, use edit_file with a small exact "
        "old_string/new_string instead. This block has NO confirmed=true "
        "override - the model can set that flag itself, and did (aether "
        "face, 2026-09-19), so it can't be what protects a file. If a "
        "near-total rewrite really is intended, tell Mark and have Mary "
        "do it."
    )


def write_file(args: dict) -> str:
    path = _resolve(args["path"])
    content = args.get("content", "")
    confirmed = bool(args.get("confirmed", False))
    memory_block = _memory_file_block_message(path)
    if memory_block:
        return memory_block
    if path.suffix in _SOURCE_CODE_SUFFIXES and not confirmed:
        return _source_edit_block_message(path, "write_file")
    shrink = _shrink_block_message(path, content)
    if shrink:
        return shrink
    if path.suffix in _SOURCE_CODE_SUFFIXES:
        err = _python_syntax_error(content, path)
        if err:
            return (
                f"ERROR: not writing {path} - the content doesn't parse "
                f"as valid Python ({err}). Nothing was written. Fix the "
                "syntax before trying again."
            )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


# Code-level gate on run_shell, added 2026-09-09 after the prompt-only
# "warn before disruptive actions" rule was live-tested and failed twice
# in a row (a throwaway file, asked to be deleted directly, got deleted
# immediately with zero warning both in one-shot and interactive chat
# mode). Matches the pattern already proven this same day with the
# log-spam guard: this model does not reliably hold a "state it, then
# wait" instruction on its own - the friction has to live in code, not
# just the prompt.
_HARD_BLOCKED_SHELL_RE = re.compile(
    r"\b(shutdown|restart-computer|reboot)\b", re.IGNORECASE
)
_SOFT_BLOCKED_SHELL_RE = re.compile(
    r"\b(rm|del|erase|remove-item|rd|rmdir|taskkill|stop-process|format)\b",
    re.IGNORECASE,
)


def _log_confirmed_disruptive_action(command: str, returncode: int) -> None:
    """Unconditional audit trail for any run_shell call that passed the
    soft-block gate with confirmed=true. Writes directly, bypassing
    _write_guard on purpose - an audit entry must never be silently
    dropped as a 'duplicate', and this is a distinct action each time
    even if the command text repeats."""
    try:
        path = _ensure_today_daily_note()
        timestamp = datetime.datetime.now().strftime("%H:%M")
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"\n## {timestamp} - CONFIRMED DISRUPTIVE ACTION\n"
                f"Ran with confirmed=true: `{command}` (exit code {returncode})\n"
            )
    except Exception:
        pass  # never let audit logging break the actual tool result


def run_shell(args: dict) -> str:
    command = args["command"]
    confirmed = bool(args.get("confirmed", False))

    if _HARD_BLOCKED_SHELL_RE.search(command):
        return (
            "BLOCKED: this command can reboot, shut down, or restart this "
            "machine. That is never allowed through run_shell, confirmed "
            "or not - see the 2026-09-09 unwarned-reboot incident in "
            "INDEX.md. If Mark genuinely needs this machine rebooted, tell "
            "him to do it himself or ask Mary. Don't retry this command."
        )

    if _SOFT_BLOCKED_SHELL_RE.search(command) and not confirmed:
        return (
            "BLOCKED: this command has a real, hard-to-reverse side effect "
            "(deleting a file, killing a process, or similar). It was NOT "
            "run. Answer directly now (no more tool calls this turn) "
            "stating the exact command and exactly what it will do, then "
            "wait for Mark to actually reply yes. Only if he does, call "
            "run_shell again with this exact same command and "
            "confirmed=true - not before."
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        pieces = [f"exit code: {result.returncode}"]
        if out:
            pieces.append(f"stdout:\n{out}")
        if err:
            pieces.append(f"stderr:\n{err}")
        if confirmed and _SOFT_BLOCKED_SHELL_RE.search(command):
            _log_confirmed_disruptive_action(command, result.returncode)
        return "\n".join(pieces)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30s"
    except Exception as e:
        return f"ERROR running command: {e}"


# Confirm gate on ops_monitor.restart_hung_process, added 2026-09-10 as
# part of wiring the four-part roadmap's ops-monitoring item into the live
# agent. Same soft-block pattern as run_shell's own disruptive-action
# gate: killing a real process is a real, hard-to-reverse action, so it
# doesn't get exposed unconfirmed even though it's a narrower, named tool
# rather than an arbitrary shell command.
def restart_hung_process(args: dict) -> str:
    process_name = args.get("process_name") if args else None
    confirmed = bool(args.get("confirmed", False)) if args else False
    if process_name and not confirmed:
        return (
            f"BLOCKED: this would kill every '{process_name}' process and "
            "relaunch it - a real, hard-to-reverse action. It was NOT run. "
            "Answer directly now (no more tool calls this turn) stating "
            "exactly what you're restarting and why, then wait for Mark to "
            "actually reply yes. Only if he does, call restart_hung_process "
            "again with the same arguments plus confirmed=true - not before."
        )
    return ops_monitor.restart_hung_process(args)


def append_daily_note(args: dict) -> str:
    text = args["text"]
    path = _ensure_today_daily_note()
    existing = path.read_text(encoding="utf-8")
    blocked = _write_guard(existing, text, _parse_daily_header_ts)
    if blocked:
        return blocked
    blocked = _capability_claim_guard(text, bool(args.get("verified", False)))
    if blocked:
        return blocked
    timestamp = datetime.datetime.now().strftime("%H:%M")
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n{text.rstrip(chr(10))}\n")
        return f"OK: appended {len(text)} chars to {path.relative_to(WORKSPACE)}"
    except Exception as e:
        return f"ERROR appending to daily note: {e}"


def _ensure_lessons_file() -> pathlib.Path:
    """Auto-create LESSONS.md from a minimal template if it doesn't exist
    yet - mirrors the daily note's own 'create from template if
    missing' habit."""
    if not LESSONS_FILE.exists():
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        LESSONS_FILE.write_text(LESSONS_TEMPLATE, encoding="utf-8")
    return LESSONS_FILE


def append_lesson(args: dict) -> str:
    """Log a real fix, working method, or dead end to skip - not a daily
    journal entry, a durable lesson meant to change future behavior. A
    tail of this file is auto-loaded into every run's system prompt, so
    what gets logged here actually gets used, not just archived."""
    lesson = args["lesson"]
    topic = args.get("topic", "").strip()
    path = _ensure_lessons_file()
    existing = path.read_text(encoding="utf-8")
    blocked = _write_guard(existing, lesson, _parse_lesson_header_ts)
    if blocked:
        return blocked
    blocked = _capability_claim_guard(lesson, bool(args.get("verified", False)))
    if blocked:
        return blocked
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"## {timestamp}" + (f" — {topic}" if topic else "")
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n{header}\n{lesson.rstrip(chr(10))}\n")
        return f"OK: logged a lesson to {path.relative_to(WORKSPACE)}"
    except Exception as e:
        return f"ERROR appending lesson: {e}"


def recent_lessons(max_chars: int = 2000) -> str:
    """Return the tail of LESSONS.md, or '' if nothing's been logged yet.
    Not a model-callable tool - used by the agent loop to auto-load past
    lessons at the start of every run, same principle as
    recent_daily_notes: don't rely on the model remembering to go look."""
    if not LESSONS_FILE.exists():
        return ""
    text = LESSONS_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def _ensure_tasks_file() -> pathlib.Path:
    """Auto-create TASKS.md from a minimal template if it doesn't exist
    yet - mirrors the daily note's and LESSONS.md's own 'create from
    template if missing' habit."""
    if not TASKS_FILE.exists():
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        TASKS_FILE.write_text(TASKS_TEMPLATE, encoding="utf-8")
    return TASKS_FILE


def _split_tasks(text: str) -> tuple[str, str, str]:
    """Split TASKS.md's raw text into (before_open, open_body,
    rest_from_completed_on). `open_body` is everything between the
    '## Open' and '## Completed' headers - the actual list of open
    items, with its surrounding blank lines intact so re-joining is
    lossless."""
    open_marker = "## Open"
    done_marker = "## Completed"
    open_idx = text.index(open_marker) + len(open_marker)
    done_idx = text.index(done_marker)
    return text[:open_idx], text[open_idx:done_idx], text[done_idx:]


def add_task(args: dict) -> str:
    """Log a new open task - real cross-session tracking, not a daily
    journal entry. Appends a checkbox line under TASKS.md's Open
    section, dated today."""
    text = args["text"].strip()
    today = datetime.date.today().isoformat()
    path = _ensure_tasks_file()
    try:
        content = path.read_text(encoding="utf-8")
        before, open_body, rest = _split_tasks(content)
        new_line = f"- [ ] {text} (opened {today})\n"
        open_body = open_body.rstrip("\n") + "\n" + new_line + "\n"
        path.write_text(before + open_body + rest, encoding="utf-8")
        return f"OK: added open task to {path.relative_to(WORKSPACE)}"
    except ValueError:
        return "ERROR: TASKS.md is missing its ## Open/## Completed headers - don't guess, ask Mark before hand-editing the structure."
    except Exception as e:
        return f"ERROR adding task: {e}"


def complete_task(args: dict) -> str:
    """Mark an open task done - moves its line from the Open section to
    Completed, dated today, with an outcome. Refuses to guess when the
    match isn't unique, same safety property as edit_file."""
    match = args["match"].strip().lower()
    outcome = args.get("outcome", "").strip()
    path = _ensure_tasks_file()
    try:
        content = path.read_text(encoding="utf-8")
        before, open_body, rest = _split_tasks(content)
    except ValueError:
        return "ERROR: TASKS.md is missing its ## Open/## Completed headers - don't guess, ask Mark before hand-editing the structure."
    except Exception as e:
        return f"ERROR reading tasks: {e}"

    lines = open_body.splitlines()
    hits = [i for i, line in enumerate(lines) if line.strip().startswith("- [ ]") and match in line.lower()]
    if not hits:
        return f"ERROR: no open task matching {args['match']!r}."
    if len(hits) > 1:
        return f"ERROR: {len(hits)} open tasks match {args['match']!r} - not unique, give more specific text."

    i = hits[0]
    task_line = lines[i].strip()[len("- [ ]"):].strip()
    task_line = re.sub(r"\s*\(opened \d{4}-\d{2}-\d{2}\)\s*$", "", task_line)
    today = datetime.date.today().isoformat()
    del lines[i]
    open_body = ("\n".join(lines).rstrip("\n") + "\n\n") if any(l.strip() for l in lines) else "\n\n"

    done_line = f"- [x] {task_line} (closed {today})"
    if outcome:
        done_line += f" — {outcome}"
    rest = rest.rstrip("\n") + "\n" + done_line + "\n"

    try:
        path.write_text(before + open_body + rest, encoding="utf-8")
    except Exception as e:
        return f"ERROR writing tasks: {e}"
    return f"OK: closed task in {path.relative_to(WORKSPACE)}: {task_line}"


def open_tasks_section(max_chars: int = 2000) -> str:
    """Return TASKS.md's Open section (header included), or '' if empty
    or the file doesn't exist yet. Not a model-callable tool - used by
    the agent loop to auto-load open tasks at the start of every run,
    same principle as recent_daily_notes/recent_lessons: don't rely on
    the model remembering to go look."""
    if not TASKS_FILE.exists():
        return ""
    try:
        _, open_body, _ = _split_tasks(TASKS_FILE.read_text(encoding="utf-8"))
    except ValueError:
        return ""
    text = ("## Open" + open_body).strip()
    if text == "## Open":
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def list_open_tasks(args: dict) -> str:
    """Model-callable version of open_tasks_section - lets the model
    check on demand (e.g. if asked what's outstanding mid-conversation)
    instead of only relying on what auto-loaded at session start."""
    section = open_tasks_section(max_chars=6000)
    return section if section else "No open tasks tracked right now."


def search_vault(args: dict) -> str:
    query = args["query"].lower()
    if not VAULT_DIR.exists():
        return "Vault is empty, no matches."
    hits = []
    for path in sorted(VAULT_DIR.rglob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if query in line.lower():
                hits.append(f"{path.relative_to(VAULT_DIR)}: {line}")
    if not hits:
        return f"No lines in the vault matched {query!r}."
    return "\n".join(hits)


def glob_files(args: dict) -> str:
    pattern = args["pattern"]
    directory = args.get("directory", ".")
    base = _resolve(directory)
    if not base.exists():
        return f"ERROR: no such directory: {base}"
    matches = sorted(
        str(p.relative_to(WORKSPACE)) for p in base.glob(pattern) if p.is_file()
    )
    if not matches:
        return f"No files matched pattern {pattern!r} under {directory}"
    LIMIT = 200
    if len(matches) > LIMIT:
        return "\n".join(matches[:LIMIT]) + f"\n... [{len(matches) - LIMIT} more, truncated]"
    return "\n".join(matches)


# Folders whose contents are noise for a content search - build artifacts,
# the venv, and git internals, not source or notes.
_SEARCH_SKIP_DIRS = {".venv", "__pycache__", ".git"}


def _is_binary(path: pathlib.Path) -> bool:
    """True if `path` looks like binary content rather than text - the
    same NUL-byte heuristic grep -I/git use. Real incident, 2026-09-17: a
    search for 'MCP' matched raw bytes inside a compiled .pyd, and that
    garbage got handed to the model as if it were readable file content -
    it then confidently hallucinated a whole fabricated analysis of it.
    Skipping binary files here means search_files can never hand the
    model text that was never text to begin with."""
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
    except Exception:
        return False
    return b"\x00" in chunk


def search_files(args: dict) -> str:
    query = args["query"]
    glob_pattern = args.get("glob", "**/*")
    case_sensitive = args.get("case_sensitive", False)
    needle = query if case_sensitive else query.lower()

    hits = []
    LIMIT = 200
    for path in sorted(WORKSPACE.glob(glob_pattern)):
        if not path.is_file() or _SEARCH_SKIP_DIRS & set(path.parts):
            continue
        if _is_binary(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                hits.append(f"{path.relative_to(WORKSPACE)}:{i}: {line.strip()}")
                if len(hits) >= LIMIT:
                    break
        if len(hits) >= LIMIT:
            break

    if not hits:
        return f"No matches for {query!r}."
    return "\n".join(hits)


def _closest_lines_hint(content: str, old: str) -> str:
    """When edit_file's old_string isn't found, points at the real lines
    that look most like it, with line numbers. Real incident, 2026-09-18:
    asked to double the traces on the circuit face, the model never read
    the file and twice guessed an old_string (`#accent { color: ... }`)
    that doesn't exist in it, so every edit failed with a bare 'not found'
    and gave it nothing to correct from. Handing back the actual nearest
    lines turns a dead end into a retry it can succeed at."""
    import difflib
    first = next((ln.strip() for ln in old.splitlines() if ln.strip()), "")
    if not first:
        return ""
    lines = content.splitlines()
    stripped = [ln.strip() for ln in lines]
    close = difflib.get_close_matches(first, list(dict.fromkeys(stripped)), n=3, cutoff=0.4)
    if not close:
        return (
            ". Nothing in the file resembles it - you are guessing. Use "
            "read_file or search_files on this file first, then copy the "
            "exact text you want to change."
        )
    shown = []
    for c in close:
        i = stripped.index(c)
        shown.append(f"  line {i + 1}: {lines[i].strip()[:200]}")
    return (
        ". Closest real lines in the file (copy exact text from the "
        "file, do not guess):\n" + "\n".join(shown)
    )


def edit_file(args: dict) -> str:
    """Exact old_string -> new_string replacement, same safety property as
    Mary's own Edit tool: refuses to guess when old_string isn't unique,
    instead of write_file's blind overwrite."""
    path = _resolve(args["path"])
    old = args["old_string"]
    new = args["new_string"]
    replace_all = bool(args.get("replace_all", False))
    confirmed = bool(args.get("confirmed", False))

    memory_block = _memory_file_block_message(path)
    if memory_block:
        return memory_block
    if path.suffix in _SOURCE_CODE_SUFFIXES and not confirmed:
        return _source_edit_block_message(path, "edit_file")

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: no such file: {path}"
    except Exception as e:
        return f"ERROR reading {path}: {e}"

    count = content.count(old)
    if count == 0:
        return f"ERROR: old_string not found in {path}{_closest_lines_hint(content, old)}"
    if count > 1 and not replace_all:
        return (
            f"ERROR: old_string appears {count} times in {path} - not "
            "unique. Pass replace_all=true, or give more surrounding "
            "context to make it unique."
        )

    new_content = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    shrink = _shrink_block_message(path, new_content)
    if shrink:
        return shrink
    if path.suffix in _SOURCE_CODE_SUFFIXES:
        err = _python_syntax_error(new_content, path)
        if err:
            return (
                f"ERROR: not writing {path} - the result doesn't parse "
                f"as valid Python ({err}). Nothing was written. Fix the "
                "replacement text before trying again."
            )
    try:
        path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"ERROR writing {path}: {e}"
    n = count if replace_all else 1
    return f"OK: replaced {n} occurrence(s) in {path}"


def web_fetch(args: dict) -> str:
    """Fetch a URL and return its text content, HTML tags stripped. No API
    key needed - unlike web search, a plain HTTP GET doesn't require an
    account. Truncated to keep one page from blowing out the context."""
    url = args["url"]
    try:
        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR fetching {url}: {e}"

    text = resp.text
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    MAX_CHARS = 6000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"... [truncated, {len(text)} chars total]"
    return text or "(page fetched but had no readable text content)"


# Caps how many subagent levels can nest, so a subagent that itself calls
# run_subagent can't spawn an unbounded recursion tree. MAX_STEPS already
# bounds a runaway loop within one level; this bounds depth across levels.
_subagent_depth = {"n": 0}
MAX_SUBAGENT_DEPTH = 2


def run_subagent(args: dict) -> str:
    """Spin off a second, independent Ollama conversation to work a
    sub-task, then return its final answer as this tool's result. Reuses
    the same agentic loop the main conversation runs on - not a separate
    implementation to keep in sync."""
    task = args["task"]
    if _subagent_depth["n"] >= MAX_SUBAGENT_DEPTH:
        return (
            "ERROR: subagent depth limit reached - refusing to spawn "
            "another level to avoid runaway recursion."
        )

    import agent  # lazy import: agent.py imports this module at top level

    messages = [
        {"role": "system", "content": agent._build_system_prompt()},
        {"role": "user", "content": task},
    ]
    _subagent_depth["n"] += 1
    try:
        agent._agentic_turn(messages, speak_answer=False)
    except agent.OllamaUnavailable as e:
        return f"ERROR: subagent couldn't reach Ollama: {e}"
    finally:
        _subagent_depth["n"] -= 1

    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return "ERROR: subagent finished without a final answer."


RELAY_CONFIG_FILE = WORKSPACE / "relay_config.json"


def relay_send(args: dict) -> str:
    """Send a message to another machine's Mary over the LAN relay - the
    exact same wire protocol Mary's own machines already use for this
    (see start_relay_server() in backtalk/main.py, mary-relay-send.ps1):
    an HTTP POST to /relay with an X-Relay-Secret header and a JSON
    {"text": ...} body. Reads this machine's own relay_config.json for
    the shared secret and the target's address."""
    to = args["to"]
    text = args["text"]
    if not RELAY_CONFIG_FILE.exists():
        return "ERROR: relay_config.json not found - relay isn't set up on this machine."
    try:
        cfg = json.loads(RELAY_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return f"ERROR reading relay_config.json: {e}"
    if not cfg.get("secret"):
        return "ERROR: relay isn't configured (no secret set)."
    peers = cfg.get("peers", {})
    target = peers.get(to)
    if not target:
        known = ", ".join(peers) or "(none)"
        return f"ERROR: no peer named {to!r}. Known peers: {known}"
    try:
        resp = httpx.post(
            f"http://{target}/relay",
            json={"text": text},
            headers={"X-Relay-Secret": cfg["secret"]},
            timeout=10,
        )
    except Exception as e:
        return f"ERROR sending to {to} ({target}): {e}"
    return f"OK: sent to {to} ({target}), HTTP {resp.status_code}"


def schedule_task(args: dict) -> str:
    """Register a real one-time Windows Scheduled Task that fires Jarvis
    (one-shot, --speak mode) with the given task text at the given date and
    time. Routes through a generated .bat wrapper rather than an inline
    quoted command line - an inline string silently fails under Task
    Scheduler on this machine (banked finding from the Ollama Auto-Start
    build, see This AI-Server's maintenance log)."""
    date_str = args["date"]  # "YYYY-MM-DD"
    time_str = args["time"]  # "HH:MM", 24h
    task_text = args["task"]

    try:
        when = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "ERROR: date must be YYYY-MM-DD and time must be HH:MM (24-hour)."
    if when <= datetime.datetime.now():
        return "ERROR: that date/time is already in the past."

    task_name = f"Jarvis-{when.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    bat_dir = WORKSPACE / "scheduled_tasks"
    bat_dir.mkdir(exist_ok=True)
    bat_path = bat_dir / f"{task_name}.bat"
    log_path = WORKSPACE / "scheduled-task.log"
    agent_path = WORKSPACE / "agent.py"

    escaped_task = task_text.replace('"', '""')
    bat_path.write_text(
        "@echo off\n"
        f'"{sys.executable}" "{agent_path}" --speak "{escaped_task}" '
        f'>> "{log_path}" 2>&1\n',
        encoding="utf-8",
    )

    win_date = when.strftime("%m/%d/%Y")
    win_time = when.strftime("%H:%M")
    cmd = [
        "schtasks", "/Create", "/TN", task_name, "/TR", str(bat_path),
        "/SC", "ONCE", "/SD", win_date, "/ST", win_time, "/F",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        return f"ERROR creating scheduled task: {e}"
    if result.returncode != 0:
        return f"ERROR: schtasks failed: {(result.stderr or result.stdout).strip()}"

    return (
        f"OK: scheduled to run at {win_date} {win_time} "
        f"(task name {task_name}). Its output will be spoken live if "
        f"someone's around, and always logged to scheduled-task.log."
    )


def read_index() -> str:
    """Return INDEX.md's content, or '' if it doesn't exist yet. Not a
    model-callable tool - used by the agent loop to auto-load identity
    and the vault map at the start of every run."""
    if not INDEX_FILE.exists():
        return ""
    return INDEX_FILE.read_text(encoding="utf-8").strip()


# --- Live tone control (tone_control.py's popup writes here) ---
#
# Five sliders, not tools.py's own decision - see tone_control.py's own
# docstring for why the original 20-attribute rubric got collapsed to
# five. Each is bucketed into low/mid/high rather than used as a raw
# number: a local model can act on "be concise" vs "be elaborate" in a
# way it almost certainly can't distinguish "6.3 concise" from "7.1
# concise," so the fine slider position only matters for picking which
# of three clearly-different sentences gets used.
TONE_CONFIG_PATH = WORKSPACE / "tone_config.json"
_TONE_DEFAULTS = {
    "formality": 6,
    "warmth": 6,
    "conciseness": 8,
    "humor": 2,
    "confidence": 7,
}
_TONE_DESCRIPTIONS = {
    "formality": {
        "low": "Speak casually and informally - contractions are fine, keep it relaxed.",
        "mid": "Speak in a moderately polished, professional register - not stiff, not casual.",
        "high": "Speak formally and precisely - full grammar, no casual contractions, a polished technical register.",
    },
    "warmth": {
        "low": "Stay pragmatic and detached - focus on the facts and the task, skip warmth or reassurance.",
        "mid": "Be moderately warm - acknowledge the person without dwelling on it, then get to the point.",
        "high": "Be warm and empathetic - acknowledge how the person might feel alongside the answer.",
    },
    "conciseness": {
        "low": "Be elaborate - explain your reasoning and give full context, don't rush the answer.",
        "mid": "Balance brevity and detail - enough context to be useful, without padding.",
        "high": "Be concise - answer directly, minimal words, no padding or restating the question.",
    },
    "humor": {
        "low": "Stay serious - no jokes, no sarcasm, straightforward delivery.",
        "mid": "A little dry wit is fine occasionally, but don't force it.",
        "high": "Lean into dry humor and light sarcasm where it fits naturally.",
    },
    "confidence": {
        "low": "Be suggestive and open-ended - offer options rather than a single directive answer.",
        "mid": "Give a clear recommendation but leave room for the user's judgment.",
        "high": "Be confident, decisive, and authoritative - give one clear directive answer, not a menu of options.",
    },
}


def _tone_bucket(value: float) -> str:
    if value <= 3:
        return "low"
    if value <= 6:
        return "mid"
    return "high"


def _load_tone_values() -> dict:
    """Read tone_config.json (written live by tone_control.py's popup, or
    by the model itself via set_tone) and return the five current values,
    falling back to defaults for anything missing or if the file is
    absent/malformed - a broken config should never crash a turn."""
    values = dict(_TONE_DEFAULTS)
    if TONE_CONFIG_PATH.exists():
        try:
            data = json.loads(TONE_CONFIG_PATH.read_text(encoding="utf-8"))
            for k in _TONE_DEFAULTS:
                if k in data:
                    values[k] = data[k]
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return values


def tone_instruction() -> str:
    """Return the current tone sliders (raw numbers, so the model can do
    its own math for a relative request like 'less silly by 50%') plus a
    short natural-language paragraph translating them into instructions.
    Not a model-callable tool itself - read fresh by _build_system_prompt()
    on every turn, so a slider moved mid-conversation (by the popup or by
    the model's own set_tone call) takes effect on the very next reply,
    no restart needed."""
    values = _load_tone_values()
    numbers = ", ".join(f"{k}={values[k]}" for k in _TONE_DEFAULTS)
    lines = [
        _TONE_DESCRIPTIONS[k][_tone_bucket(values[k])] for k in _TONE_DEFAULTS
    ]
    return f"Current sliders (0-10 scale): {numbers}. " + " ".join(lines)


def set_tone(args: dict) -> str:
    """Model-callable: adjust one or more of the five tone sliders. Takes
    absolute target values (0-10), not deltas - for a relative request
    like 'less silly by 50%', read the current value from this turn's own
    system prompt (it's always shown) and compute the target yourself
    before calling this. Writes straight to tone_config.json, the same
    file tone_control.py's popup uses, so the popup will show the new
    position next time it's opened. Clamps out-of-range values rather
    than rejecting the call outright - a model doing its own arithmetic
    can overshoot."""
    values = _load_tone_values()
    changed = {}
    for key in _TONE_DEFAULTS:
        if key in args and args[key] is not None:
            try:
                new_val = max(0, min(10, round(float(args[key]))))
            except (TypeError, ValueError):
                continue
            if new_val != values[key]:
                changed[key] = (values[key], new_val)
            values[key] = new_val
    if not changed:
        return "No sliders changed - either no valid values were given, or they matched what's already set."
    try:
        TONE_CONFIG_PATH.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        return f"ERROR writing tone_config.json: {e}"
    summary = ", ".join(f"{k} {old}->{new}" for k, (old, new) in changed.items())
    return f"OK: tone updated ({summary}). This takes effect starting with your next reply."


# --- Visual face (visualizer/, a separate copy of Mary's ai-visualizer
# engine - see Local Mary Clone Session 32) ---
VISUALIZER_DIR = WORKSPACE / "visualizer"
VISUALIZER_CONFIG_PATH = VISUALIZER_DIR / "ai-visualizer.json"
VISUALIZER_RUN_BAT = VISUALIZER_DIR / "run.bat"


def _visualizer_port() -> int:
    try:
        cfg = json.loads(VISUALIZER_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cfg = {}
    return int(cfg.get("port", 8790))


def _kill_visualizer_processes(port: int, max_rounds: int = 5) -> int:
    """Kill whatever's actually listening on the visualizer's port, in a
    loop rather than once. Windows lets more than one process bind the
    same port without erroring (confirmed live 2026-09-09: three
    separate server.py instances were all bound to 8798 at once), so a
    single query-and-kill can miss zombies that only surface as "the"
    owner once the one ahead of them is gone. Returns how many were
    actually killed."""
    killed = 0
    find_ps = (
        f"Get-NetTCPConnection -LocalPort {port} -State Listen "
        "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
    )
    for _ in range(max_rounds):
        pids = [p.strip() for p in _run_ps(find_ps).splitlines() if p.strip()]
        if not pids:
            break
        for pid in set(pids):
            _run_ps(f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue")
            killed += 1
        time.sleep(0.4)
    return killed


def _launch_visualizer_server() -> None:
    """Start exactly one fresh instance via run.bat (same interpreter-
    selection logic it already has), detached so it outlives this call
    and keeps its own console hidden."""
    subprocess.Popen(
        ["cmd", "/c", "start", "/min", "", str(VISUALIZER_RUN_BAT)],
        cwd=str(VISUALIZER_DIR),
    )


def _wait_for_face(port: int, name: str, timeout_s: float = 8.0) -> str | None:
    """Poll /config until the fresh server reports the face we just set,
    or give up. Returns the face /config actually reports (possibly
    still the old one, if we timed out), or None if /config never
    answered at all."""
    deadline = time.time() + timeout_s
    last_face = None
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/config", timeout=1.0)
            last_face = resp.json().get("face")
            if last_face == name:
                return last_face
        except Exception:
            pass
        time.sleep(0.5)
    return last_face


def list_faces(args: dict | None = None) -> str:
    """List the face folders actually available in your own gallery
    (visualizer/faces/), and which one is active right now. You do
    have a real visual face - see the 'Your visual face' section in
    this file's own INDEX.md if you're ever unsure."""
    faces_dir = VISUALIZER_DIR / "faces"
    if not faces_dir.is_dir():
        return "No faces/ folder found under visualizer/ - something's wrong with the visualizer install."
    names = sorted(p.name for p in faces_dir.iterdir() if p.is_dir() and (p / "index.html").exists())
    if not names:
        return "No faces exist yet in visualizer/faces/."
    active = ""
    try:
        active = json.loads(VISUALIZER_CONFIG_PATH.read_text(encoding="utf-8")).get("face", "")
    except (OSError, json.JSONDecodeError):
        pass
    return "Available faces: " + ", ".join(names) + f". Active: {active or '(none set)'}"


def set_face(args: dict) -> str:
    """Switch your active visual face to one already present in
    visualizer/faces/ (call list_faces first if you don't know what's
    there). server.py reads ai-visualizer.json once into memory at
    startup and never re-reads it, so just editing the file does
    nothing on its own (learned the hard way 2026-09-09 - see
    LESSONS.md) - this function handles that itself: writes the new
    face, kills whatever's actually running the visualizer (looping
    since Windows can let more than one process end up bound to the
    same port without erroring), starts exactly one fresh instance, and
    polls /config to confirm the new face is really live before
    reporting success. If verification times out, says so honestly
    instead of claiming it worked - no restart needed on your end
    either way, but don't assume "no restart needed" means "definitely
    worked" without reading the return value."""
    name = str(args.get("face", "")).strip()
    if not name:
        return "ERROR: no face name given."
    face_dir = VISUALIZER_DIR / "faces" / name
    if not (face_dir / "index.html").exists():
        return f"ERROR: no face named '{name}' in visualizer/faces/ - call list_faces to see what's actually there."
    try:
        cfg = json.loads(VISUALIZER_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cfg = {}
    old = cfg.get("face", "")
    cfg["face"] = name
    port = int(cfg.get("port", 8790))
    try:
        VISUALIZER_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        return f"ERROR writing ai-visualizer.json: {e}"

    killed = _kill_visualizer_processes(port)
    _launch_visualizer_server()
    live_face = _wait_for_face(port, name)

    if live_face == name:
        return (f"OK: active face switched {old!r} -> {name!r}, server restarted "
                f"({killed} old process(es) cleared), confirmed live via /config.")
    if live_face is None:
        return (f"WROTE {name!r} to ai-visualizer.json and restarted the server, but /config "
                f"never answered within the timeout - check manually, don't assume it worked.")
    return (f"WROTE {name!r} to ai-visualizer.json and restarted the server, but /config still "
            f"reports {live_face!r} - the restart didn't pick up the change, check manually.")


# Window titles your own face pages actually use - "Jarvis Visualizer"
# for your own hand-built face, "<name> Orbit" for the copied Orbit
# face (core.js sets document.title to "{name} Orbit" using
# ai-visualizer.json's own "name" field, which is "Jarvis" here) - a
# new face you write yourself should keep "Jarvis" somewhere in its
# <title> so this can still find it. Matched against msedge/chrome
# specifically (not just any window titled "Jarvis" - your own chat
# console window is titled that too).
_FACE_WINDOW_TITLE_RE = "Jarvis|J\\.A\\.R\\.V\\.I\\.S|Orbit|Precision Facade"


def _run_ps(script: str) -> str:
    """Run a fixed PowerShell script - not model-generated text, so this
    doesn't go through run_shell's safety gate. Used only by show_face's
    own hardcoded window-management logic below."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


_FIND_FACE_WINDOW_PS = (
    "Get-Process msedge,chrome -ErrorAction SilentlyContinue | "
    f"Where-Object {{ $_.MainWindowTitle -match '{_FACE_WINDOW_TITLE_RE}' }} | "
    "Select-Object -First 1 -ExpandProperty Id"
)

_FOCUS_WINDOW_PS = """
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class JarvisFaceWin {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int ht, bool r);
    [DllImport("user32.dll")] public static extern int GetSystemMetrics(int n);
    [DllImport("user32.dll")] public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
}
"@ -ErrorAction SilentlyContinue
$p = Get-Process -Id __PID__ -ErrorAction SilentlyContinue
if ($p) {
    $h = $p.MainWindowHandle
    $screenW = [JarvisFaceWin]::GetSystemMetrics(0)
    $screenH = [JarvisFaceWin]::GetSystemMetrics(1)
    $w = 900; $ht = 900
    $x = [int](($screenW - $w) / 2); $y = [int](($screenH - $ht) / 2)
    [JarvisFaceWin]::MoveWindow($h, $x, $y, $w, $ht, $true) | Out-Null
    # A plain SetForegroundWindow from a background process is blocked by
    # Windows (proven live 2026-09-09 - the window moved/resized correctly
    # but stayed behind other windows). Tapping Alt first gives the calling
    # thread "real recent input," which is what Windows actually checks
    # before granting the foreground-focus steal.
    [JarvisFaceWin]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero)
    [JarvisFaceWin]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 50
    [JarvisFaceWin]::ShowWindow($h, 9) | Out-Null
    [JarvisFaceWin]::SetForegroundWindow($h) | Out-Null
    Write-Output "ok"
}
"""


def show_face(args: dict | None = None) -> str:
    """Bring your own face's browser window to the front and center of
    the screen (roughly 900x900, centered on the primary display). If no
    browser window is currently showing your face, opens one first at
    whichever face is currently active, then focuses it."""
    face_pid = _run_ps(_FIND_FACE_WINDOW_PS)
    if not face_pid:
        try:
            cfg = json.loads(VISUALIZER_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
        face = cfg.get("face", "")
        port = cfg.get("port", 8798)
        # The visualizer server only comes up when set_face() runs it or
        # via the desktop launcher - nothing keeps it alive across a
        # reboot. Don't open a browser at a dead URL if it's not
        # actually answering (found live 2026-09-11 - this was the real
        # bug behind "can't bring up his face").
        try:
            httpx.get(f"http://127.0.0.1:{port}/config", timeout=1.0)
        except Exception:
            _launch_visualizer_server()
            _wait_for_face(port, face)
        url = f"http://127.0.0.1:{port}/faces/{face}/" if face else f"http://127.0.0.1:{port}/"
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "msedge", "--new-window", url])
        except OSError as e:
            return f"ERROR: couldn't open a browser for your face: {e}"
        time.sleep(2)
        face_pid = _run_ps(_FIND_FACE_WINDOW_PS)
        if not face_pid:
            return "Opened a browser window for your face, but couldn't confirm which window it is - check the screen."

    result = _run_ps(_FOCUS_WINDOW_PS.replace("__PID__", face_pid))
    if result == "ok":
        return "OK: your face window is now in front, centered on screen."
    return "Found your face window but couldn't confirm the move/focus succeeded - check the screen."


def recent_daily_notes(max_chars: int = 2000) -> str:
    """Return the tail of today's daily note (creating it from the
    template if this is the first run of the day), capped at max_chars.
    Not a model-callable tool - used by the agent loop to auto-load
    recent context at the start of every run, instead of relying on the
    model to remember to call search_vault."""
    path = _ensure_today_daily_note()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


# Name -> callable, used by the agent loop to dispatch a tool call.
# --- Face tunables, added 2026-09-20 ---
#
# Why this exists: Mark asked Jarvis to "double the swarm objects" on the
# orbit face. A dry-run against granite (0/8) and qwen3-coder:30b (0/4)
# showed neither model ever does the grounding steps a person would (find
# the active face, find the term inside that face's file, read the
# constant, edit it): they wrote files at invented paths, edited the wrong
# face, or gave up. Prompt guidance never held on this model, so the search
# lives in code: face_tunables shows what a face lets you adjust with plain
# descriptions taken from the face's own comments, and set_face_tunable
# changes one value, validated, and verifies it landed.
#
# What counts as tunable, by convention already used across the faces: a
# top-level `const NAME = <number>;` with an ALL-CAPS name, and the color
# custom properties in the first `:root{...}` block.
_TUNABLE_NUM_RE = re.compile(
    r"^([ \t]{0,2})const[ \t]+([A-Z][A-Z0-9_]*)[ \t]*=[ \t]*(-?\d*\.?\d+)[ \t]*;", re.M
)
_TUNABLE_COLOR_RE = re.compile(r"(--[a-z][a-z0-9-]*)[ \t]*:[ \t]*(#[0-9a-fA-F]{3,8})\b")
_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _tidy_comment(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^(/\*+|//+)", "", line)
    line = re.sub(r"\*+/$", "", line)
    line = re.sub(r"^[-=*\s]+|[-=*\s]+$", "", line)
    return line.strip()


def _tunable_description(text: str, match_start: int, match_end: int) -> str:
    """Plain-language description for a numeric constant: its own trailing
    comment if it has one, else the nearest comment line just above it."""
    line_end = text.find("\n", match_end)
    line_end = len(text) if line_end == -1 else line_end
    trailing = text[match_end:line_end]
    m = re.search(r"//\s*(.+)$|/\*\s*(.+?)\s*\*/", trailing)
    if m:
        return (m.group(1) or m.group(2)).strip()[:140]
    above = text[:match_start].split("\n")[-6:-1]
    for raw in reversed(above):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("//", "/*", "*")) or stripped.endswith("*/"):
            cleaned = _tidy_comment(stripped)
            if cleaned:
                return cleaned[:140]
        break
    return ""


def _parse_tunables(text: str) -> list[dict]:
    """Every adjustable value in a face's index.html: {name, kind, value,
    line, span, desc}. `span` is the exact (start, end) of the value in
    `text`, so a change touches nothing else."""
    found = []
    for m in _TUNABLE_NUM_RE.finditer(text):
        found.append({
            "name": m.group(2), "kind": "number", "value": m.group(3),
            "line": text.count("\n", 0, m.start()) + 1, "span": m.span(3),
            "desc": _tunable_description(text, m.start(), m.end()),
        })
    root = re.search(r":root\s*\{(.*?)\}", text, re.S)
    if root:
        for m in _TUNABLE_COLOR_RE.finditer(root.group(1)):
            start = root.start(1) + m.start(2)
            found.append({
                "name": m.group(1), "kind": "color", "value": m.group(2),
                "line": text.count("\n", 0, start) + 1,
                "span": (start, root.start(1) + m.end(2)),
                "desc": "color (CSS custom property)",
            })
    return found


def _active_face_name() -> str:
    try:
        return json.loads(VISUALIZER_CONFIG_PATH.read_text(encoding="utf-8")).get("face", "") or ""
    except (OSError, json.JSONDecodeError):
        return ""


def _face_names() -> list[str]:
    faces_dir = VISUALIZER_DIR / "faces"
    if not faces_dir.is_dir():
        return []
    return sorted(p.name for p in faces_dir.iterdir() if p.is_dir() and (p / "index.html").exists())


def _resolve_face_for_tunables(face: str | None, request: str = "") -> tuple[str | None, str | None]:
    """(face_name, error). No name given means the active face, so 'this
    face' / 'this space' work. A near-miss name (a speech-recognition slip
    like 'acer' for 'aether') gets a suggestion instead of a guess."""
    names = _face_names()
    name = (face or "").strip() or _active_face_name()
    if request and face and name in names and name != _active_face_name():
        spoken = name.lower().replace("_", " ").replace("-", " ")
        if spoken not in request.lower().replace("_", " ").replace("-", " "):
            # Mark never named that face; models pick one up from earlier
            # turns in the history. 'this face' means the active one.
            name = _active_face_name() or name
    if not name:
        return None, "ERROR: no face named and no active face is set. Call list_faces."
    if name in names:
        return name, None
    close = difflib.get_close_matches(name, names, n=2, cutoff=0.5)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    return None, f"ERROR: no face named {name!r}.{hint} Available faces: {', '.join(names)}."


def face_tunables(args: dict | None = None) -> str:
    args = args or {}
    name, err = _resolve_face_for_tunables(args.get("face"), str(args.get("_request") or ""))
    if err:
        return err
    path = VISUALIZER_DIR / "faces" / name / "index.html"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"ERROR reading {path}: {e}"
    items = _parse_tunables(text)
    if not items:
        return (
            f"The {name!r} face has no tunable values (no ALL-CAPS numeric "
            "constants or :root color variables). Adjusting it would mean "
            "editing its index.html by hand; tell Mark instead of guessing."
        )
    lines = [f"Tunable values in the {name!r} face (visualizer/faces/{name}/index.html):"]
    for it in items:
        desc = f"  - {it['desc']}" if it["desc"] else ""
        lines.append(f"  {it['name']} = {it['value']}{desc}")
    lines.append(
        "Change one with set_face_tunable(name, value). The page reads these "
        "when it loads, so Mark has to refresh the face to see a change."
    )
    return "\n".join(lines)


def _format_number(new: float, old_text: str) -> str:
    if "." not in old_text and float(new).is_integer():
        return str(int(new))
    return f"{new:.6f}".rstrip("0").rstrip(".") or "0"


# One change per value per turn. Live dry-run, 2026-09-20: granite called
# set_face_tunable up to eight times in one turn, and each 'double' compounded
# (14 -> 28 -> ... -> 1792). The agent hands each turn a unique id; a second
# change to the same value within the same turn is refused.
_TUNABLE_DONE: dict = {"turn": None, "done": {}}


_TUNABLE_STOPWORDS = {
    "the", "a", "an", "of", "to", "its", "this", "that", "in", "on", "for",
    "objects", "object", "things", "thing", "items", "item", "count",
    "number", "amount", "value", "level", "setting", "face", "space",
}
_SCALE_WORDS = {"double": 2.0, "twice": 2.0, "triple": 3.0, "quadruple": 4.0,
                "half": 0.5, "halve": 0.5}


def _split_words(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w for w in re.split(r"[^A-Za-z0-9]+", text.lower()) if w]


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


# Words in Mark's sentence that say nothing about WHICH value he means.
_REQUEST_FILLER = _TUNABLE_STOPWORDS | {
    "make", "edit", "change", "set", "please", "would", "like", "you", "i",
    "me", "my", "want", "wanna", "can", "could", "it", "up", "down", "more",
    "less", "bigger", "smaller", "some", "just", "okay", "ok", "hey",
    "jarvis", "and", "also", "then", "now", "by", "with", "from", "is",
    "are", "be", "need", "adjust", "increase", "decrease", "raise", "lower",
    "turn", "put", "get", "give", "do", "double", "twice", "triple", "half",
    "quadruple", "halve", "was", "just", "so", "what", "how", "about",
}


def _request_tokens(request: str) -> set[str]:
    return {_stem(w) for w in _split_words(request)
            if w not in _REQUEST_FILLER and not w.isdigit()}


def _request_scale(request: str) -> float | None:
    """A scale word in Mark's own sentence ('double') and no explicit
    number: the factor he asked for. Returns None otherwise."""
    if re.search(r"\d", request):
        return None
    for w in _split_words(request):
        if w in _SCALE_WORDS:
            return _SCALE_WORDS[w]
    return None


def _item_overlap(item: dict, tokens: set[str]) -> int:
    name_words = {_stem(w) for w in _split_words(item["name"])}
    desc_words = {_stem(w) for w in _split_words(item["desc"])}
    return 2 * len(tokens & name_words) + len(tokens & desc_words)


def _match_tunable(items: list[dict], phrase: str) -> tuple[dict | None, str | None]:
    """Resolve what Mark said ('swarm objects') or an exact name to one
    tunable. Exact name first; otherwise score the phrase's meaningful words
    against each tunable's name and description. Only a unique best match is
    accepted: a tie or nothing gets an error that lists the candidates with
    their descriptions, so a wrong guess never silently edits the wrong
    thing (a model picking SQUASH for 'swarm' is exactly what this exists
    to stop)."""
    exact = [i for i in items if i["name"] == phrase or i["name"] == "--" + phrase
             or i["name"].lower() == phrase.lower()]
    if len(exact) == 1:
        return exact[0], None
    tokens = [_stem(w) for w in _split_words(phrase) if w not in _TUNABLE_STOPWORDS]
    scored = []
    for it in items:
        name_words = {_stem(w) for w in _split_words(it["name"])}
        desc_words = {_stem(w) for w in _split_words(it["desc"])}
        score = sum(2 for t in tokens if t in name_words) + sum(1 for t in tokens if t in desc_words)
        if score:
            scored.append((score, it))
    if scored:
        scored.sort(key=lambda x: -x[0])
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return scored[0][1], None
        cands = [it for sc, it in scored if sc == scored[0][0]]
    else:
        cands = items
    listing = "; ".join(
        f"{it['name']} = {it['value']}" + (f" ({it['desc']})" if it["desc"] else "")
        for it in cands[:8]
    )
    return None, (
        f"ERROR: could not tell which value {phrase!r} means. Candidates: {listing}. "
        "Nothing changed. Ask Mark which one, or pick from these."
    )


def _resolve_new_number(raw_value: str, old_text: str) -> tuple[float | None, str | None]:
    """A plain number is the new absolute value. 'double', 'half', 'x3',
    '3x' scale the CURRENT value, so the model never has to do arithmetic."""
    v = raw_value.strip().lower()
    old = float(old_text)
    if v in _SCALE_WORDS:
        return old * _SCALE_WORDS[v], None
    m = re.fullmatch(r"(?:[x\u00d7*]\s*(\d*\.?\d+)|(\d*\.?\d+)\s*[x\u00d7])", v)
    if m:
        return old * float(m.group(1) or m.group(2)), None
    try:
        return float(v), None
    except ValueError:
        return None, f"ERROR: {raw_value!r} is not a number, or double/half/x3. Nothing changed."


def set_face_tunable(args: dict) -> str:
    request = str(args.get("_request") or "")
    name, err = _resolve_face_for_tunables(args.get("face"), request)
    if err:
        return err
    phrase = str(args.get("what") or args.get("name") or "").strip()
    raw_value = str(args.get("value", "")).strip()
    if not phrase or raw_value == "":
        return "ERROR: say what to change (what) and the new value. Call face_tunables to see the real options."
    path = VISUALIZER_DIR / "faces" / name / "index.html"
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
    except OSError as e:
        return f"ERROR reading {path}: {e}"
    items = _parse_tunables(text)
    if not items:
        return face_tunables({"face": name})
    item, why = _match_tunable(items, phrase)
    notes = ""
    if request:
        rt = _request_tokens(request)
        if rt:
            ranked = sorted(((_item_overlap(it, rt), it) for it in items), key=lambda x: -x[0])
            top = ranked[0][0]
            unique_top = top > 0 and (len(ranked) == 1 or ranked[1][0] < top)
            if item is None or _item_overlap(item, rt) == 0:
                # The model's pick has nothing to do with what Mark said:
                # use the one value his own words point at, or refuse.
                if unique_top:
                    item = ranked[0][1]
                    notes = f" [used {item['name']}, the value your words point at]"
                else:
                    cands = "; ".join(f"{it['name']} = {it['value']}" for sc, it in ranked[:6])
                    return (
                        f"ERROR: nothing here clearly matches what Mark said ({request.strip()[:80]!r}). "
                        f"Options: {cands}. Nothing changed. Ask Mark which one he means."
                    )
    if item is None:
        return why
    if item["kind"] == "color":
        if not _HEX_COLOR_RE.match(raw_value):
            return f"ERROR: {raw_value!r} is not a hex color like #0000ff. Nothing changed."
        new_text_value = raw_value
    else:
        new, bad = _resolve_new_number(raw_value, item["value"])
        if bad:
            return bad
        scale = _request_scale(request) if request else None
        if scale is not None and abs(new - float(item["value"]) * scale) > 1e-9:
            new = float(item["value"]) * scale
            notes += f" [value set to x{scale:g} of the current one, as Mark asked]"
        old = float(item["value"])
        limit = max(50 * abs(old), 100)
        if not (new == new) or new in (float("inf"), float("-inf")) or new < 0 or new > limit:
            return (
                f"ERROR: {raw_value} is outside the sane range for {item['name']} "
                f"(0 to {limit:g}, current {item['value']}). Nothing changed."
            )
        new_text_value = _format_number(new, item["value"])
    turn = str(args.get("_turn") or "")
    if turn:
        if _TUNABLE_DONE["turn"] != turn:
            _TUNABLE_DONE["turn"] = turn
            _TUNABLE_DONE["done"] = {}
        prev = _TUNABLE_DONE["done"].get((name, item["name"]))
        if prev:
            return (
                f"ALREADY DONE this turn: {name} {item['name']} was changed "
                f"{prev[0]} -> {prev[1]}. Not changing it again. It is finished: "
                "tell Mark it's done and to refresh the face."
            )
    start, end = item["span"]
    new_text = text[:start] + new_text_value + text[end:]
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        with open(path, "r", encoding="utf-8", newline="") as f:
            check = f.read()
    except OSError as e:
        return f"ERROR writing {path}: {e}"
    after = [i for i in _parse_tunables(check) if i["name"] == item["name"]]
    if len(after) != 1 or after[0]["value"] != new_text_value:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return f"ERROR: the change to {item['name']} did not verify after writing, so it was rolled back. Nothing changed."
    if turn:
        _TUNABLE_DONE["done"][(name, item["name"])] = (item["value"], new_text_value)
    matched = f" (matched {phrase!r} to {item['name']}" + (f": {item['desc']})" if item["desc"] else ")") + notes
    return (
        f"OK: {name} {item['name']} {item['value']} -> {new_text_value}{matched}, "
        f"line {item['line']} of visualizer/faces/{name}/index.html. Done: do not "
        "call this again for the same request. The page reads values when it "
        "loads, so tell Mark to refresh the face."
    )


REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
    "append_daily_note": append_daily_note,
    "append_lesson": append_lesson,
    "add_task": add_task,
    "complete_task": complete_task,
    "list_open_tasks": list_open_tasks,
    "search_vault": search_vault,
    "glob_files": glob_files,
    "search_files": search_files,
    "edit_file": edit_file,
    "web_fetch": web_fetch,
    "run_subagent": run_subagent,
    "schedule_task": schedule_task,
    "relay_send": relay_send,
    "set_tone": set_tone,
    "list_faces": list_faces,
    "set_face": set_face,
    "show_face": show_face,
    "face_tunables": face_tunables,
    "set_face_tunable": set_face_tunable,
    "check_disk": ops_monitor.check_disk,
    "check_scheduled_tasks": ops_monitor.check_scheduled_tasks,
    "check_gpu_nvidia": ops_monitor.check_gpu_nvidia,
    "check_gpu_amd": ops_monitor.check_gpu_amd,
    "restart_hung_process": restart_hung_process,
}

# Ollama/OpenAI-style function schemas, sent to the model so it knows what's
# available and how to call it.
SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file's full contents from the agent's workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace unless absolute."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a text file in the agent's workspace, creating parent folders as needed. Writing a .py file is blocked on the first call - state the exact change to Mark and wait for a yes, then retry with confirmed=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace unless absolute."},
                    "content": {"type": "string", "description": "Full text content to write."},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set true only on a retry, only after Mark has explicitly replied yes to a source-code change you already stated to him. Never set true on a first attempt. Only relevant for .py files - ignored otherwise.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the agent's workspace and return its exit code, stdout, and stderr. Rebooting/shutting down the machine is never allowed, no exceptions. A command that deletes a file, kills a process, or has another real hard-to-reverse side effect gets blocked on the first call - state the exact consequence to Mark as your answer (no more tool calls that turn) and wait for him to actually say yes. Only if he does, call this again with the same command and confirmed=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set true only on a retry, only after Mark has explicitly replied yes to a consequence you already stated to him. Never set true on a first attempt.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_daily_note",
            "description": "Append a timestamped line or block of text to today's daily note in the vault (auto-created if this is the first entry today). Use this to log anything worth remembering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to append."},
                    "verified": {
                        "type": "boolean",
                        "description": "Only set true if this text claims tool calls, the interface, or your own capabilities are broken/limited AND you've actually confirmed that by reading the real code or a real traceback - not a guess. Omit or leave false otherwise; a claim like that without verified=true will be blocked.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_lesson",
            "description": "Log a real fix, working method, or dead end to skip - a durable lesson meant to change future behavior, not a daily journal entry. Use this whenever you discover something worth remembering next time, not just what happened today.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lesson": {"type": "string", "description": "The lesson itself: what worked, what didn't, and why."},
                    "topic": {"type": "string", "description": "Optional short topic label, e.g. 'scheduling' or 'ollama'."},
                    "verified": {
                        "type": "boolean",
                        "description": "Only set true if this lesson claims tool calls, the interface, or your own capabilities are broken/limited AND you've actually confirmed that by reading the real code or a real traceback - not a guess. Omit or leave false otherwise; a claim like that without verified=true will be blocked.",
                    },
                },
                "required": ["lesson"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Log a new open task in TASKS.md - real persistent tracking across sessions, not a daily journal entry. Use this whenever something is left open, blocked, or worth following up on later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The task itself, plain description."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark an open task done - moves it from TASKS.md's Open section to Completed, dated today. Fails if the match isn't unique among open tasks (give more specific text).",
            "parameters": {
                "type": "object",
                "properties": {
                    "match": {"type": "string", "description": "Substring that uniquely identifies the open task to close."},
                    "outcome": {"type": "string", "description": "Optional short note on how it was resolved."},
                },
                "required": ["match"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_open_tasks",
            "description": "Return the current list of open tasks from TASKS.md. Use this if asked what's outstanding, on your to-do list, or still in progress - don't claim you have no memory of past or current tasks.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": "Search the whole vault (INDEX.md and every daily note) for lines containing a query string (case-insensitive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find files in the workspace matching a glob pattern (e.g. '*.py', '**/*.md').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match, e.g. '**/*.py'."},
                    "directory": {"type": "string", "description": "Directory to search under, relative to the workspace. Defaults to the workspace root."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents in the workspace for a substring, optionally restricted to files matching a glob. Returns matching lines with file:line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for."},
                    "glob": {"type": "string", "description": "Glob pattern restricting which files to search, e.g. '**/*.py'. Defaults to every file."},
                    "case_sensitive": {"type": "boolean", "description": "Whether the search is case-sensitive. Defaults to false."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in a file with new text. Fails if old_string isn't found or isn't unique (unless replace_all is set) - safer than write_file for a small change to an existing file. Editing a .py file is blocked on the first call - state the exact change to Mark and wait for a yes, then retry with confirmed=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace unless absolute."},
                    "old_string": {"type": "string", "description": "Exact text to find and replace."},
                    "new_string": {"type": "string", "description": "Text to replace it with."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring old_string to be unique. Defaults to false."},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set true only on a retry, only after Mark has explicitly replied yes to a source-code change you already stated to him. Never set true on a first attempt. Only relevant for .py files - ignored otherwise.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its readable text content (HTML tags stripped). No search - only works if you already have the exact URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_subagent",
            "description": "Spin off an independent sub-task in a fresh conversation and return its final answer. Use for a self-contained side task that would clutter the main conversation, not for anything that needs the current conversation's context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The self-contained task for the subagent to work."},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "relay_send",
            "description": "Send a message to another machine's Mary over the LAN relay. Use this to pass information or a request to Mark's other assistant instances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Name of the peer to send to, e.g. 'VR2026' or 'AI-Server'. Must be a name already known in relay_config.json's peers."},
                    "text": {"type": "string", "description": "The message text to send."},
                },
                "required": ["to", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_tone",
            "description": "Adjust your own tone sliders (formality, warmth, conciseness, humor, confidence - each 0-10) - the same ones Mark's tone_control.py popup controls. Every reply already shows you the current value of each slider. For a relative request like 'be less silly by 50%', read the current humor value from that line and compute the target yourself (e.g. current 4 -> pass humor=2) - this tool takes the final absolute value, not a delta or percentage. Only pass the sliders that should change. Takes effect starting with your very next reply, no restart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formality": {"type": "number", "description": "0 (casual) to 10 (formal). Omit to leave unchanged."},
                    "warmth": {"type": "number", "description": "0 (detached) to 10 (warm). Omit to leave unchanged."},
                    "conciseness": {"type": "number", "description": "0 (elaborate) to 10 (concise). Omit to leave unchanged."},
                    "humor": {"type": "number", "description": "0 (serious) to 10 (playful/sarcastic). Omit to leave unchanged."},
                    "confidence": {"type": "number", "description": "0 (suggestive) to 10 (authoritative/decisive). Omit to leave unchanged."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_faces",
            "description": "List the visual faces actually available in your own gallery (visualizer/faces/) and which one is active. You have a real visual face - never claim you're text-only or have no visual presence, check here first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_face",
            "description": "Switch your active visual face to one already in visualizer/faces/ (call list_faces first to see real options). Handles the full switch itself - writes the config, restarts the visualizer server, and confirms the new face is actually live via /config - so no separate restart step is needed. Read the return value: it says plainly if verification timed out or failed rather than assuming success.",
            "parameters": {
                "type": "object",
                "properties": {
                    "face": {"type": "string", "description": "Name of an existing folder under visualizer/faces/, e.g. 'jarvis' or 'orbit'."},
                },
                "required": ["face"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "face_tunables",
            "description": "Mark asks to change how a face looks or behaves (more or fewer objects, faster, bigger, a different color, 'this space', 'this face')? Use the face tools for it, never write_file or edit_file and never search for files. This one lists what can be adjusted on a face with current values and plain descriptions, read-only. Leave face empty for the face that's active right now. To actually change something, call set_face_tunable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "face": {"type": "string", "description": "Face name, e.g. 'orbit'. Omit for the active face."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_face_tunable",
            "description": "THE way to change a face: more or fewer objects, faster or slower, bigger or smaller, a different color. Never use write_file or edit_file for that. Pass 'what' in Mark's own words (copy his phrase) and 'value': a number, or 'double', 'half' or 'x3' to scale the current value, or a hex color like #0000ff. It finds the matching value itself, edits only that one value, reads it back, and tells you what it matched, so trust its OK or ERROR and do not call it again once it says OK. Omit face for the face that's active right now. The page reads values when it loads, so tell Mark to refresh the face to see it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "what": {"type": "string", "description": "What to change, in Mark's own words, copied from his request."},
                    "value": {"type": "string", "description": "A number, or 'double' / 'half' / 'x3', or a hex color like #0000ff."},
                    "face": {"type": "string", "description": "Face name. Omit for the active face."},
                    "name": {"type": "string", "description": "Optional exact name from face_tunables, only if you have one."},
                },
                "required": ["what", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_face",
            "description": "Bring your own face's browser window to the front and center of the screen. Opens one first if none is currently open.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_disk",
            "description": "Report free/total disk space on a drive. Use for a 'how much disk space is left' style question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drive": {"type": "string", "description": "Drive letter with trailing backslash, e.g. 'C:\\\\'. Defaults to C:\\ if omitted."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_scheduled_tasks",
            "description": "Report last-run time and result for scheduled tasks matching a name pattern - use for 'check your backup status' or similar status questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern matched against task names, e.g. 'Jarvis' or 'Jarvis|Mary'. Defaults to 'Jarvis' if omitted."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_gpu_nvidia",
            "description": "Report the Nvidia GPU's name, temperature, utilization, and memory use via nvidia-smi.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_gpu_amd",
            "description": "Report total GPU 3D-engine utilization across all GPUs on this machine (AMD 7900 XT + Nvidia A1000 combined - no per-vendor split, no temperature available). Use for a general 'check the gpu' question; use check_gpu_nvidia instead if specifically asked about the Nvidia card.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_hung_process",
            "description": "Kill every process matching a name and relaunch it with a given command - use to restart something stuck or unresponsive (e.g. the visualizer). A real, hard-to-reverse action: blocked on the first call - state exactly what you're restarting and why, wait for Mark to say yes, then retry with confirmed=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "process_name": {"type": "string", "description": "Process name to kill, e.g. 'ollama' (no .exe suffix needed)."},
                    "relaunch_command": {"type": "string", "description": "Shell command to relaunch it, e.g. the path to its .bat or .exe."},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set true only on a retry, only after Mark has explicitly replied yes to a restart you already stated to him. Never set true on a first attempt.",
                    },
                },
                "required": ["process_name", "relaunch_command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule a one-time future run of Jarvis with a given task, via a real Windows Scheduled Task. The result will be spoken aloud when it fires and logged to scheduled-task.log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date to run on, format YYYY-MM-DD."},
                    "time": {"type": "string", "description": "Time to run at, 24-hour format HH:MM."},
                    "task": {"type": "string", "description": "The task text to run at that time."},
                },
                "required": ["date", "time", "task"],
            },
        },
    },
]
