# Jarvis From Scratch

Portfolio piece #4: a minimal agentic tool-calling loop, built from scratch
against a fully local model stack — Ollama running `qwen3-coder:30b` on an
AMD RX 7900 XT. No Claude, no API cost, no agent framework. The point is to
demonstrate the orchestration loop itself: a model that can chain multiple
real tool calls to complete a multi-step task, hand-rolled against Ollama's
REST API using nothing but the Python standard library.

This is a different category of portfolio demo than the memory-vault
pattern (Scout) or physics simulation (the cloth demo) — this one is about
agent architecture and tool-calling.

## Stack

- **Model:** `qwen3-coder:30b` via Ollama (`localhost:11434`), MoE
  architecture, ~19GB at Q4_K_M.
- **Language:** Python 3.12, stdlib only (`urllib`, `subprocess`,
  `pathlib`, `json`) — no `requests`, no `ollama` SDK, no LangChain.
- **Interface:** text-first CLI. No voice — backtalk already proves that
  wiring; repeating it here demonstrates nothing new.

## Tools

Four real tools, registered in `tools.py` and described to the model via
OpenAI/Ollama-style function schemas:

1. `read_file(path)` — read a text file from the workspace.
2. `write_file(path, content)` — write/overwrite a text file.
3. `run_shell(command)` — run a shell command, capture stdout/stderr/exit code.
4. `append_memory(text)` / `search_memory(query)` — a flat `memory.md`
   scratch file the agent can read from and write to.

All file/shell tools are scoped to this project folder by default (a
relative path resolves against the workspace root) — a deliberate guardrail
so a wrong tool call from the model can't wander the filesystem.

## Memory

`memory.md` is a flat markdown file, not a vault-style structured memory
system — that level of structure is explicitly out of scope for v0. The
agent appends to it and can search it, nothing more.

## Usage

```
python agent.py "your task here"
```

Each step prints the tool call the model made and the result it got back,
so the reasoning chain is visible, not just the final answer.

## v0 scope

Locked 2026-09-04 — see `Jarvis From Scratch` in the vault
(`06 - Resources`) for the full scoping conversation. Explicit non-goals
for v0: voice, a real vault-style memory system, publishing anywhere.

## Status

Built and verified end-to-end on AI-Server, 2026-09-04. Not published.
