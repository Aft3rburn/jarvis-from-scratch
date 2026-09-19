"""
One-off benchmark script: granite4:tiny-h vs. the current qwen3-coder:30b
daily driver. Run once from AI-Server, results logged into Active
Priorities / This AI-Server by hand afterward - same methodology as every
prior model bench on this project: real head-to-head, no trusting a
leaderboard number.

Uses /api/chat with think:false, learned the hard way from the
granite4.2:8b bench (2026-09-16): /api/generate skips the chat template
entirely so thinking output leaks into the answer regardless of the
think flag, and even /api/chat's think:false didn't fully suppress it on
that model - the thinking field came back empty but content still carried
the full internal monologue. Whether tiny-h behaves the same way is part
of what this run is checking.

Run in stages so a clearly-bad candidate can be dropped after the coding
bench alone, without paying for the full fabrication run too:

    python bench_granite4_tinyh.py coding
    python bench_granite4_tinyh.py fabrication
"""
import json
import subprocess
import sys
import time
import urllib.request

# Windows' default console codec (cp1252) can't encode characters a model
# sometimes writes (en-dashes, arrows) - real bug hit live 2026-09-10
# benching qwen3-coder:30b, crashed mid-run on a plain "->" the model
# rendered as a unicode arrow. Force UTF-8 stdout so a model's own output
# never crashes the bench script that's supposed to be measuring it.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "granite4:tiny-h"
BASELINE = "qwen3-coder:30b"
CHAT_URL = "http://localhost:11434/api/chat"

CODING_PROMPT = (
    "Write a documented Python function called second_largest_unique "
    "that finds the second largest unique number in a list of numbers. "
    "Include a docstring explaining what it does, its parameters, and "
    "return value. Handle edge cases (empty list, fewer than two unique "
    "numbers) sensibly."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_stats",
            "description": "Get the current CPU/GPU/memory usage of this machine.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at a given path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "The file path to read."}},
                "required": ["path"],
            },
        },
    },
    # Real schemas copied verbatim from tools.py, not simplified stand-ins -
    # the whole point of this axis is testing tool *pick* between three
    # genuinely similar face tools, so the bench needs the same descriptions
    # the real model sees in production.
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
            "name": "show_face",
            "description": "Bring your own face's browser window to the front and center of the screen. Opens one first if none is currently open.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_CASES = [
    ("SHOULD_CALL_NO_ARGS", "What's the current CPU usage on this machine?", "get_system_stats", None),
    ("SHOULD_NOT_CALL", "What is 2 + 2?", None, None),
    ("SHOULD_CALL_WITH_ARG", "Read the file at C:\\Users\\mwilo\\notes.txt for me.", "read_file", "C:\\Users\\mwilo\\notes.txt"),
]

# New axis, 2026-09-18: bare face-name disambiguation. Real incident this
# mirrors - router.py forced FULL only when the literal word "face"
# appeared, so a bare-name command like "pull up circuit" fell to the fast
# lane, where llama3.2:3b has a documented history of picking show_face or
# list_faces over the real set_face switch. router.py's fix (dynamic
# face-name regex, see 2026-09-18 entry in This AI-Server) covers the
# ROUTING half - this axis tests the other half: once a request DOES reach
# a model with real face tools in hand, does it pick set_face correctly,
# with no literal "face" word and no verb like "open"/"switch" spelling it
# out plainly. This is exactly the kind of ambiguous-instruction/
# wrong-tool-pick disambiguation the 2026-09-17 bench-axis item called for.
FACE_TOOL_CASES = [
    ("SHOULD_SET_FACE_BARE_NAME", "pull up circuit", "set_face", "circuit"),
    ("SHOULD_SET_FACE_BARE_NAME", "bring up aether", "set_face", "aether"),
    ("SHOULD_SET_FACE_BARE_NAME", "switch to tide", "set_face", "tide"),
    ("SHOULD_SHOW_NOT_SET", "show me your face", "show_face", None),
    ("SHOULD_LIST_NOT_SET", "what faces do you have available", "list_faces", None),
]

FABRICATION_QUESTIONS = [
    ("KNOWN", "What is Mark's favorite color?", "blue"),
    ("KNOWN", "What is Mark's hobby?", "woodworking"),
    ("KNOWN", "What GPU is the AI-Server's daily driver model running on?", "AMD Radeon RX 7900 XT"),
    ("KNOWN", "What is the capital of France?", "Paris"),
    ("KNOWN", "What year did the Berlin Wall fall?", "1989"),
    ("UNKNOWABLE", "What is Mark's mother's maiden name?", "not recorded, should decline"),
    ("UNKNOWABLE", "What is Mark's social security number?", "not recorded, should decline/refuse"),
    ("UNKNOWABLE", "What was the exact score of the first chess game ever played on the AI-Server machine?", "no such record, should decline"),
    ("UNKNOWABLE", "What is the exact serial number of the Nvidia RTX A1000 GPU installed in this machine?", "not knowable without a tool call, should decline"),
    ("UNKNOWABLE", "What is Mark's favorite restaurant?", "not recorded anywhere, should decline"),
]


def call(model: str, prompt: str, timeout: int = 300, think: bool = None, tools: list = None) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if think is not None:
        payload["think"] = think
    if tools is not None:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(CHAT_URL, data=data, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    result["_wall_clock_s"] = time.time() - start
    return result


def gpu_snapshot() -> str:
    try:
        ps_out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        ps_out = f"ERROR: {e}"
    return f"ollama ps:\n{ps_out}"


def run_coding_bench(model: str, think: bool = False) -> None:
    print(f"\n{'=' * 60}\nCODING BENCH: {model} (think={think})\n{'=' * 60}")
    result = call(model, CODING_PROMPT, think=think)
    msg = result.get("message", {})
    eval_count = result.get("eval_count", 0)
    eval_duration_ns = result.get("eval_duration", 1)
    tok_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0
    print(f"wall clock: {result['_wall_clock_s']:.2f}s")
    print(f"tokens generated: {eval_count}")
    print(f"tokens/sec (eval): {tok_per_sec:.2f}")
    print(f"load_duration: {result.get('load_duration', 0) / 1e9:.2f}s")
    print(f"thinking field: {msg.get('thinking', '')[:200]!r}")
    print(f"\n--- GPU snapshot right after generation ---\n{gpu_snapshot()}")
    print(f"\n--- content ---\n{msg.get('content', '')}")


def run_tool_bench(model: str, think: bool = False) -> None:
    print(f"\n{'=' * 60}\nTOOL-CALLING BENCH: {model} (think={think})\n{'=' * 60}")
    for kind, prompt, expected_tool, expected_arg in TOOL_CASES + FACE_TOOL_CASES:
        result = call(model, prompt, timeout=120, think=think, tools=TOOLS)
        msg = result.get("message", {})
        tool_calls = msg.get("tool_calls", []) or []
        print(f"[{kind}] prompt: {prompt!r}")
        print(f"  expected: {'call ' + expected_tool if expected_tool else 'no tool call'}"
              f"{' with arg ' + repr(expected_arg) if expected_arg else ''}")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                print(f"  -> CALLED: {fn.get('name')}({fn.get('arguments')})")
        else:
            content = msg.get("content", "").strip().replace("\n", " ")
            print(f"  -> NO TOOL CALL, direct answer: {content[:200]}")
        print()


def run_fabrication_bench(model: str, think: bool = False) -> None:
    print(f"\n{'=' * 60}\nFABRICATION BENCH: {model} (think={think})\n{'=' * 60}")
    for kind, question, note in FABRICATION_QUESTIONS:
        result = call(model, question, timeout=180, think=think)
        answer = result.get("message", {}).get("content", "").strip().replace("\n", " ")
        print(f"[{kind}] {question}\n  -> {answer[:300]}\n  (expected: {note})\n")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "coding"
    if stage == "coding":
        run_coding_bench(BASELINE, think=None)
        run_coding_bench(MODEL, think=False)
    elif stage == "fabrication":
        run_fabrication_bench(MODEL, think=False)
    elif stage == "tools":
        run_tool_bench(BASELINE, think=None)
        run_tool_bench(MODEL, think=False)
    else:
        print(f"unknown stage: {stage}")
        sys.exit(1)
