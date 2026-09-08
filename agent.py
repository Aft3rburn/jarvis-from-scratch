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
import re
import sys
import urllib.error
import urllib.request

import tools
import voice

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-coder:30b"
MAX_STEPS = 10

# Sometimes the model writes a tool call out as plain text (e.g.
# "<function=append_memory>...") instead of using Ollama's real structured
# tool_calls field - a known local-model flakiness, hit twice in testing
# 2026-09-08. Caught by this pattern so it can be nudged to retry instead
# of silently treating the garbled text as a real final answer.
_FAKE_TOOL_CALL_RE = re.compile(r"<function[=\s]", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a local agent running entirely offline. You have tools to "
    "read and write files, run shell commands, and append to or search a "
    "persistent vault (your identity and daily notes). Use tools "
    "whenever a task needs real information or a real action instead of "
    "guessing. Chain multiple tool calls when a task needs more than one "
    "step. When you have enough information to fully answer, respond "
    "with plain text and no further tool calls."
)


def _build_system_prompt() -> str:
    """Base system prompt plus the vault's INDEX.md and a tail of today's
    daily note, auto-loaded so identity and recent context are there from
    the first message instead of depending on the model remembering to
    call search_vault."""
    index = tools.read_index()
    recent = tools.recent_daily_notes()

    parts = [SYSTEM_PROMPT]
    if index:
        parts.append("Vault index (identity and map):\n\n" + index)
    if recent:
        parts.append(
            "Today's daily note so far. If it already answers the "
            "user's question, answer directly from it - do not call "
            "search_vault for something already shown here. Only use "
            "search_vault for older facts not covered below.\n\n" + recent
        )
    return "\n\n".join(parts)


class OllamaUnavailable(Exception):
    """Raised when Ollama can't be reached at all, so callers can print a
    plain message instead of a raw connection-refused traceback."""


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
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaUnavailable(
            f"Can't reach Ollama at {OLLAMA_URL} ({e.reason}). "
            "Is the Ollama server running? Try 'ollama serve' or check the "
            "Ollama Auto-Start scheduled task."
        ) from e


def _agentic_turn(messages: list, speak_answer: bool = False) -> None:
    """Run the ask-model / execute-tools loop, appending to `messages` in
    place, until the model gives a final text answer (or MAX_STEPS is
    hit). Shared by one-shot tasks, the chat REPL, and voice mode - the
    only difference between those three is how `messages` gets built and
    what happens to the final answer."""
    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- step {step}: asking {MODEL} ---")
        response = call_ollama(messages)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            answer = message.get("content", "")
            if _FAKE_TOOL_CALL_RE.search(answer) and step < MAX_STEPS:
                print(
                    "\n(model wrote a tool call as plain text instead of "
                    "the real structured format - nudging it to retry)"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That wasn't a valid tool call - you wrote it "
                            "out as plain text instead of using the actual "
                            "tool-calling mechanism. Please make the tool "
                            "call again properly."
                        ),
                    }
                )
                continue
            print(f"\n=== answer ===\n{answer}")
            if speak_answer:
                voice.speak(answer)
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
    if speak_answer:
        voice.speak("I stopped without a final answer, I hit my step limit.")


def run_task(task: str, speak_answer: bool = False) -> None:
    """One-shot: fresh conversation, single task, then exit."""
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": task},
    ]
    try:
        _agentic_turn(messages, speak_answer)
    except OllamaUnavailable as e:
        print(f"\n=== error ===\n{e}")
        if speak_answer:
            voice.speak("I can't reach Ollama right now. Is it running?")


def run_chat() -> None:
    """Multi-turn text chat: one conversation that keeps growing across
    turns, like typing into this very terminal - not a fresh start every
    message the way run_task is."""
    messages = [{"role": "system", "content": _build_system_prompt()}]
    print("Chat mode. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            return
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting chat.")
            return
        messages.append({"role": "user", "content": user_input})
        try:
            _agentic_turn(messages, speak_answer=True)
        except OllamaUnavailable as e:
            print(f"\n=== error ===\n{e}")
            voice.speak("I can't reach Ollama right now. Is it running?")
            messages.pop()  # drop the unanswered user turn, retry cleanly next time


def run_voice_loop() -> None:
    """Push-to-talk loop: hold the PTT key, speak your task, release, get a
    spoken answer, repeat. Ctrl+C to exit."""
    print(f"Voice mode (push-to-talk, {voice.PTT_KEY}). Ctrl+C to exit.")
    while True:
        text = voice.listen_ptt()
        if not text:
            continue
        print(f"\nheard: {text}")
        if text.strip().lower() in ("stop listening", "stop", "exit", "quit"):
            voice.speak("Stopping.")
            return
        run_task(text, speak_answer=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--voice":
        run_voice_loop()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--chat":
        run_chat()
    elif len(sys.argv) < 2:
        print('Usage: python agent.py "your task here"')
        print('       python agent.py --chat')
        print('       python agent.py --voice')
        sys.exit(1)
    else:
        run_task(" ".join(sys.argv[1:]))
