"""
Retest: qwen3.8:27b vs. the current qwen3-coder:30b daily driver.
Original bench (2026-09-08) used the raw /api/generate endpoint with no
think control and found qwen3.8:27b reasons out loud, padding the answer -
900 tokens at 31.6 tok/s vs qwen3-coder:30b's 225 tokens at 86.4 tok/s.
Retesting with the corrected /api/chat + think:false methodology (learned
2026-09-16 benching granite4.2:8b) before treating that old verdict as
final - if it was purely a thinking-leak artifact like granite4.2:8b, the
model itself might be more competitive than the original number suggested.

Run in stages so a clearly-bad candidate can be dropped after the coding
bench alone, without paying for the full fabrication/tool run too:

    python bench_qwen38_27b_retest.py coding
    python bench_qwen38_27b_retest.py tools
    python bench_qwen38_27b_retest.py fabrication
"""
import json
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "qwen3.8:27b"
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
]

TOOL_CASES = [
    ("SHOULD_CALL_NO_ARGS", "What's the current CPU usage on this machine?", "get_system_stats", None),
    ("SHOULD_NOT_CALL", "What is 2 + 2?", None, None),
    ("SHOULD_CALL_WITH_ARG", "Read the file at C:\\Users\\mwilo\\notes.txt for me.", "read_file", "C:\\Users\\mwilo\\notes.txt"),
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
    for kind, prompt, expected_tool, expected_arg in TOOL_CASES:
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
