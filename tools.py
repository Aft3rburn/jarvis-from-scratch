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
import pathlib
import re
import subprocess
import sys
import uuid

import httpx

# Everything the agent touches is scoped under this folder so a stray tool
# call can't wander the whole filesystem. Paths passed by the model are
# resolved relative to here unless already absolute.
WORKSPACE = pathlib.Path(__file__).parent.resolve()

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


def read_file(args: dict) -> str:
    path = _resolve(args["path"])
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: no such file: {path}"
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def write_file(args: dict) -> str:
    path = _resolve(args["path"])
    content = args.get("content", "")
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


def append_daily_note(args: dict) -> str:
    text = args["text"]
    path = _ensure_today_daily_note()
    existing = path.read_text(encoding="utf-8")
    blocked = _write_guard(existing, text, _parse_daily_header_ts)
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


def edit_file(args: dict) -> str:
    """Exact old_string -> new_string replacement, same safety property as
    Mary's own Edit tool: refuses to guess when old_string isn't unique,
    instead of write_file's blind overwrite."""
    path = _resolve(args["path"])
    old = args["old_string"]
    new = args["new_string"]
    replace_all = bool(args.get("replace_all", False))

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: no such file: {path}"
    except Exception as e:
        return f"ERROR reading {path}: {e}"

    count = content.count(old)
    if count == 0:
        return f"ERROR: old_string not found in {path}"
    if count > 1 and not replace_all:
        return (
            f"ERROR: old_string appears {count} times in {path} - not "
            "unique. Pass replace_all=true, or give more surrounding "
            "context to make it unique."
        )

    new_content = content.replace(old, new) if replace_all else content.replace(old, new, 1)
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


def tone_instruction() -> str:
    """Read tone_config.json (written live by tone_control.py's popup)
    and return a short natural-language paragraph translating the five
    sliders into instructions. Not a model-callable tool - read fresh by
    _build_system_prompt() on every turn, so a slider moved mid-
    conversation takes effect on the model's very next reply, no
    restart needed. Falls back to the defaults if the file is missing
    or malformed - a broken config should never crash a turn."""
    values = dict(_TONE_DEFAULTS)
    if TONE_CONFIG_PATH.exists():
        try:
            data = json.loads(TONE_CONFIG_PATH.read_text(encoding="utf-8"))
            for k in _TONE_DEFAULTS:
                if k in data:
                    values[k] = data[k]
        except (json.JSONDecodeError, OSError, TypeError):
            pass  # fall back to defaults rather than break the turn

    lines = [
        _TONE_DESCRIPTIONS[k][_tone_bucket(values[k])] for k in _TONE_DEFAULTS
    ]
    return " ".join(lines)


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
            "description": "Write (overwrite) a text file in the agent's workspace, creating parent folders as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace unless absolute."},
                    "content": {"type": "string", "description": "Full text content to write."},
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
            "description": "Replace an exact substring in a file with new text. Fails if old_string isn't found or isn't unique (unless replace_all is set) - safer than write_file for a small change to an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace unless absolute."},
                    "old_string": {"type": "string", "description": "Exact text to find and replace."},
                    "new_string": {"type": "string", "description": "Text to replace it with."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring old_string to be unique. Defaults to false."},
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
