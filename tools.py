"""
Tool implementations for the from-scratch agent.

Four real tools, each doing one plain thing. Every tool function takes a
single dict of arguments (as parsed from the model's tool call) and returns
a string — that string goes straight back into the conversation as the
tool's result, so keep output readable, not a raw repr.
"""

import pathlib
import subprocess

# Everything the agent touches is scoped under this folder so a stray tool
# call can't wander the whole filesystem. Paths passed by the model are
# resolved relative to here unless already absolute.
WORKSPACE = pathlib.Path(__file__).parent.resolve()
MEMORY_FILE = WORKSPACE / "memory.md"


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


def append_memory(args: dict) -> str:
    text = args["text"]
    try:
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")
        return f"OK: appended {len(text)} chars to memory.md"
    except Exception as e:
        return f"ERROR appending to memory: {e}"


def search_memory(args: dict) -> str:
    query = args["query"].lower()
    if not MEMORY_FILE.exists():
        return "memory.md is empty, no matches."
    lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
    hits = [line for line in lines if query in line.lower()]
    if not hits:
        return f"No lines in memory.md matched {query!r}."
    return "\n".join(hits)


# Name -> callable, used by the agent loop to dispatch a tool call.
REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
    "append_memory": append_memory,
    "search_memory": search_memory,
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
            "name": "append_memory",
            "description": "Append a line or block of text to the agent's persistent memory.md scratch file.",
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
            "name": "search_memory",
            "description": "Search memory.md for lines containing a query string (case-insensitive).",
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
