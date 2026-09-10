# Handoff: four-part roadmap, AI-Server session

Branch: `roadmap/four-part-scaffolding-2026-09-10`. Built and tested
remotely (ADLAPTOPMAX, 2026-09-10) on everything that didn't need this
machine's actual hardware. Full narrative log in the Mary vault,
`06 - Resources/Local Mary Clone.md`, Session 39, if you want the "why"
behind any of this — this file is just the "what to do next," so you
don't have to go dig for it.

## 0. Pull and sanity-check first

```
git fetch origin
git checkout roadmap/four-part-scaffolding-2026-09-10
python test_regressions.py
python test_router.py
```

Both suites passed clean on ADLAPTOPMAX's Python 3.12 with this repo's
real dependencies already importable there. They SHOULD pass identically
here since none of the logic they touch is hardware-specific - if
anything fails here that passed there, that's itself a real finding
(an actual environment difference), not something to just re-run past.

## 1. Ops-layer monitoring (`ops_monitor.py`) - verify the GPU checks

`check_disk` and `check_scheduled_tasks` are already proven against real
live data (ADLAPTOPMAX's own disk and scheduled tasks) - trust those,
just confirm `check_scheduled_tasks({"pattern": "Jarvis"})` finds the
real task names on this box (e.g. "Jarvis Vault Backup").

`check_gpu_amd`/`check_gpu_nvidia` are UNVERIFIED. Run these two
commands directly first and paste the raw output before trusting either
function's parsing:

```
rocm-smi --showtemp --showuse --csv
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

If the real column order/format differs from what the functions assume,
fix the parsing against the real output, not the docs.

## 2. Speed router (`router.py`) - pick and bench a fast-lane model

`classify()` (the fast/full decision) is done and tested - 6/6 pass,
don't need to touch it unless real traffic proves it too sloppy per the
roadmap's own "start cheap, escalate only if needed" rule.

What's still missing: an actual small model for the FAST lane to
generate the simple answers `classify()` routes to it. Two real,
current (2026) candidates worth a pull-and-bench - same methodology as
every model decision on this project, don't trust the number on paper,
run the actual head-to-head:

- `ollama pull llama3.2:3b` - ~2GB, strong instruction-following, the
  safer default.
- `ollama pull gemma3:2b` (or whatever the current Gemma-4-class 2B tag
  actually is at pull time - check `ollama search gemma` first, tags
  shift) - reportedly the fastest of the two on CPU, worth it if the
  ROCm GPU is already saturated by the 30B daily driver and this needs
  to run CPU-side to avoid contention.

Bench both against ~10 real simple-command prompts (the kind
`test_router.py` already has as examples) for latency and answer
quality before picking one. Whichever wins, note it in
`Active Priorities.md`'s local-LLM-watch item, same as every other
model decision this project's made.

## 3. Wire it all into the live agent

Once 1 and 2 are verified: add the ops_monitor functions and (once a
fast model's chosen) the router-based dispatch into `tools.py` /
`agent.py`'s real `REGISTRY`/`SCHEMAS` and `call_ollama` model
selection. This is a real edit to live production files - state the
exact change to Mark and get his yes first, same rule as every other
`.py` edit this project has made (see `_source_edit_block_message` in
`tools.py` if you need the reminder of why).

`restart_hung_process` in particular should route through the same
`confirmed=true` soft-block pattern `run_shell` already uses when it
gets wired in - it kills a real process, don't expose it unconfirmed.

## 4. Memory hygiene (item 4)

Not scaffolded remotely - needs to actually look at the real current
`vault/INDEX.md` and `LESSONS.md` content to know what's worth
tightening. Read them fresh, don't assume last week's mess is still
today's mess.

## 5. Merge

Once verified end to end (tests pass here, GPU checks confirmed, fast
model chosen and wired in, memory pass done), merge this branch to
master and delete it. Don't leave it dangling once it's done its job.
