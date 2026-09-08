"""
Tool implementations for the from-scratch agent.

Four real tools, each doing one plain thing. Every tool function takes a
single dict of arguments (as parsed from the model's tool call) and returns
a string — that string goes straight back into the conversation as the
tool's result, so keep output readable, not a raw repr.
"""

import datetime
import pathlib
import subprocess

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
]
