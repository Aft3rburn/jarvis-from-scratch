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
- **Interface:** text-first CLI, or push-to-talk voice via `--voice`
  (v1 addition, see below).

## Tools

Four real tools, registered in `tools.py` and described to the model via
OpenAI/Ollama-style function schemas:

1. `read_file(path)` — read a text file from the workspace.
2. `write_file(path, content)` — write/overwrite a text file.
3. `run_shell(command)` — run a shell command, capture stdout/stderr/exit code.
4. `append_daily_note(text)` / `search_vault(query)` — a small local
   vault (`vault/INDEX.md` + `vault/daily/YYYY-MM-DD.md`), a scaled-down
   version of Mary's own memory pattern. See the Memory section below.

All file/shell tools are scoped to this project folder by default (a
relative path resolves against the workspace root) — a deliberate guardrail
so a wrong tool call from the model can't wander the filesystem.

## Memory

v1 addition (2026-09-08): a small local vault, deliberately scaled down
from Mary's own vault pattern rather than a full clone of it — no
folder-per-domain taxonomy, no wikilinks, no frontmatter typing, since
those solve a scale problem (many machines, many months, many notes)
this one-box project doesn't have yet.

- `vault/INDEX.md` — identity and a map of the vault, auto-loaded into
  the system prompt on every run.
- `vault/daily/YYYY-MM-DD.md` — one file per day, auto-created from a
  template on first use each day. The tail of today's note is also
  auto-loaded into the system prompt, so recent context doesn't depend
  on the model remembering to search for it.
- `append_daily_note` / `search_vault` — the model's tools for writing
  to and searching the vault.

## Chat mode (v1 addition, 2026-09-08)

`python agent.py --chat` — a persistent, multi-turn text conversation
(type `exit` or `quit` to leave), instead of the one-shot
`python agent.py "task"` which starts a fresh conversation every call.

## Voice (v1 addition, 2026-09-08)

`python agent.py --voice` — push-to-talk (hold Left Ctrl, speak,
release) via `voice.py`: faster-whisper (CUDA, on this rig's Nvidia
RTX A1000) for speech-to-text, Piper (CPU) for text-to-speech.

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
Both were picked back up as v1 additions on 2026-09-08 once this became
the base for the [[Local Mary Clone]] project.

## Status

v0 built and verified end-to-end on AI-Server, 2026-09-04, then
published. v1 (voice + vault memory) built 2026-09-08 as part of the
Local Mary Clone project — see that note in the Mary vault
(`06 - Resources`) for the full build/test log.
