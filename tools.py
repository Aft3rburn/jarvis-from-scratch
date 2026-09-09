"""
Tool implementations for the from-scratch agent.

Every tool function takes a single dict of arguments (as parsed from the
model's tool call) and returns a string — that string goes straight back
into the conversation as the tool's result, so keep output readable, not a
raw repr.
"""

import datetime
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

DAILY_TEMPLATE = "# {date}\n\n"


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


def run_shell(args: dict) -> str:
    command = args["command"]
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
        return "\n".join(pieces)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30s"
    except Exception as e:
        return f"ERROR running command: {e}"


def append_daily_note(args: dict) -> str:
    text = args["text"]
    timestamp = datetime.datetime.now().strftime("%H:%M")
    path = _ensure_today_daily_note()
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n{text.rstrip(chr(10))}\n")
        return f"OK: appended {len(text)} chars to {path.relative_to(WORKSPACE)}"
    except Exception as e:
        return f"ERROR appending to daily note: {e}"


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
    "search_vault": search_vault,
    "glob_files": glob_files,
    "search_files": search_files,
    "edit_file": edit_file,
    "web_fetch": web_fetch,
    "run_subagent": run_subagent,
    "schedule_task": schedule_task,
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
            "description": "Run a shell command in the agent's workspace and return its exit code, stdout, and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
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
