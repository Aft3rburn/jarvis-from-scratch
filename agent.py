"""
Jarvis From Scratch — v0

A minimal agentic tool-calling loop, hand-rolled against Ollama's REST API
(stdlib only, no SDK) running qwen3-coder:30b locally. No Claude, no API
cost, no framework — the point is to prove the loop itself: send messages,
let the model request tools, execute them, feed results back, repeat until
the model gives a final answer with no more tool calls.

Usage:
    python agent.py "your task here"
"""

import json
import sys
import urllib.request

import tools

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-coder:30b"
MAX_STEPS = 10

SYSTEM_PROMPT = (
    "You are a local agent running entirely offline. You have tools to "
    "read and write files, run shell commands, and read/append a "
    "persistent memory file. Use tools whenever a task needs real "
    "information or a real action instead of guessing. Chain multiple "
    "tool calls when a task needs more than one step. When you have "
    "enough information to fully answer, respond with plain text and no "
    "further tool calls."
)


def call_ollama(messages: list) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools.SCHEMAS,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_task(task: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- step {step}: asking {MODEL} ---")
        response = call_ollama(messages)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            print(f"\n=== final answer ===\n{message.get('content', '')}")
            return

        for call in tool_calls:
            name = call["function"]["name"]
            raw_args = call["function"]["arguments"]
            # Ollama returns arguments as a dict already in most cases, but
            # some models emit a JSON string - handle both.
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            print(f"tool call: {name}({args})")
            handler = tools.REGISTRY.get(name)
            if handler is None:
                result = f"ERROR: no such tool: {name}"
            else:
                result = handler(args)
            print(f"tool result: {result[:500]}")

            messages.append({"role": "tool", "content": result, "name": name})

    print("\n=== stopped: hit MAX_STEPS without a final answer ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your task here"')
        sys.exit(1)
    run_task(" ".join(sys.argv[1:]))
