"""
Ops-layer monitoring — item 1 of the 2026-09-10 four-part roadmap (see
'Local Mary Clone' Session 38 in Mark's Mary vault, 06 - Resources).

Jarvis already runs on AI-Server 24/7 - this gives him real monitoring
duties there instead of trying to make him clever. Bounded, routine
checks are exactly what a smaller model handles reliably, and it takes
real work off Mary's own sitrep-checking plate (she currently checks
these same scheduled tasks by hand every session).

Wired into tools.py's REGISTRY/SCHEMAS 2026-09-10, verified live on real
AI-Server hardware. disk/scheduled-tasks/nvidia GPU checks all confirmed
against real output that session. AMD GPU check is a real, honest
exception - see check_gpu_amd's own docstring.
"""

import re
import shutil
import subprocess


def _run_ps(script: str, timeout: int = 10) -> str:
    """Run a fixed PowerShell script and return its stdout, stripped.
    Mirrors tools.py's own private _run_ps used by show_face - same
    pattern, kept local here since this module isn't merged into tools.py
    yet. Never raises: a monitoring check failing to run is itself
    useful information, not a crash."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"__ERROR__:{e}"


# --- Disk space --------------------------------------------------------
# Verified for real this session (ADLAPTOPMAX, 2026-09-10) - shutil is
# stdlib and disk_usage works identically on any Windows drive, nothing
# AI-Server-specific here.

def check_disk(args: dict) -> str:
    """Report free/total space on a drive. args: {"drive": "C:\\\\"}
    (optional, defaults to C:\\). Real bug caught live 2026-09-10: the
    fast-lane 3B model garbled the schema's escaped example and passed
    'C\\' (letter + backslash, no colon) instead of 'C:\\' - normalize
    a bare-letter-plus-backslash drive argument instead of trusting a
    small model to reproduce JSON escaping exactly."""
    drive = args.get("drive", "C:\\") if args else "C:\\"
    m = re.fullmatch(r"([A-Za-z])\\?", drive)
    if m:
        drive = f"{m.group(1)}:\\"
    try:
        usage = shutil.disk_usage(drive)
    except OSError as e:
        return f"ERROR checking disk {drive}: {e}"
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    pct_free = usage.free / usage.total * 100
    flag = " - LOW DISK SPACE" if pct_free < 10 else ""
    return (
        f"{drive} has {free_gb:.1f}GB free of {total_gb:.1f}GB "
        f"({pct_free:.1f}% free){flag}"
    )


# --- Scheduled tasks -----------------------------------------------------
# Verified for real this session (ADLAPTOPMAX, 2026-09-10) - same
# Get-ScheduledTaskInfo mechanism already used live in this session's own
# sitrep to check Mary Local Backup / Mary Face Auto-Pull. Task NAMES will
# differ on AI-Server (e.g. "Jarvis Vault Backup") - pass the real names
# at call time, the mechanism itself is what's proven, not any specific
# task name.

_CHECK_TASKS_PS_TEMPLATE = (
    "Get-ScheduledTask | Where-Object {{ $_.TaskName -match '{pattern}' }} | "
    "Get-ScheduledTaskInfo | Select-Object TaskName, "
    "@{{Name='LastRunTime';Expression={{ if ($_.LastRunTime) "
    "{{ $_.LastRunTime.ToString('yyyy-MM-dd HH:mm') }} else {{ 'never' }} }}}}, "
    "LastTaskResult | ConvertTo-Json -Compress"
)


def check_scheduled_tasks(args: dict) -> str:
    """Report LastRunTime/LastTaskResult for scheduled tasks matching a
    name pattern. args: {"pattern": "Jarvis|Mary"} (regex, matched against
    TaskName). A LastTaskResult of 0 means success; SCHED_S_TASK_RUNNING
    (267009 / 0x41301) means it's currently running, not a failure -
    flag anything else as a real problem worth surfacing."""
    pattern = args.get("pattern", "Jarvis") if args else "Jarvis"
    out = _run_ps(_CHECK_TASKS_PS_TEMPLATE.format(pattern=pattern))
    if out.startswith("__ERROR__"):
        return f"ERROR checking scheduled tasks: {out}"
    if not out:
        return f"No scheduled tasks matched pattern {pattern!r}."

    import json
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return f"Couldn't parse scheduled task info, raw output: {out}"
    if isinstance(data, dict):
        data = [data]

    KNOWN_OK = {0, 267009}  # success, or currently running
    lines = []
    for task in data:
        name = task.get("TaskName", "?")
        last_run = task.get("LastRunTime", "?")
        result = task.get("LastTaskResult", "?")
        flag = "" if result in KNOWN_OK else " - CHECK THIS, unexpected result code"
        lines.append(f"{name}: last ran {last_run}, result {result}{flag}")
    return "\n".join(lines)


# --- GPU (AMD ROCm / Nvidia) ---------------------------------------------
# Verified live on real AI-Server hardware, 2026-09-10 - real, honest
# finding along the way: rocm-smi doesn't exist anywhere on this box.
# AMD's Windows ROCm 7.2 install here is the HIP/compiler dev toolkit
# (hipcc, rocgdb, clang...), not a monitoring CLI - rocm-smi/amd-smi are
# Linux-only tools AMD doesn't ship for Windows at all. Confirmed by
# searching the entire ROCm install tree for any *smi* binary - none
# exist. check_gpu_amd below is a real replacement using Windows' own
# built-in GPU Engine performance counters instead, not a rocm-smi
# wrapper.

def check_gpu_amd(args: dict) -> str:
    """Report total GPU 3D-engine utilization via Windows' built-in
    performance counters, since rocm-smi doesn't exist on this Windows
    box (see module note above). Real, honest limitation: this counter
    is system-wide across every GPU (AMD 7900 XT + the Nvidia A1000
    both), not filtered to the AMD card alone - Windows doesn't expose a
    per-vendor breakdown without vendor SDK code, which wasn't worth
    pulling in for this pass. Cross-check against check_gpu_nvidia's
    number if you need to know how much of the total is the Nvidia
    card. No temperature - Windows has no built-in GPU temp counter, and
    without rocm-smi there's no AMD-provided way to read it on this box;
    that's a real gap, not an oversight."""
    ps = (
        "$s = (Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage' "
        "-ErrorAction SilentlyContinue).CounterSamples | "
        "Measure-Object -Property CookedValue -Sum; "
        "if ($s) { [math]::Round($s.Sum, 1) } else { 'ERROR' }"
    )
    out = _run_ps(ps)
    if not out or out == "ERROR" or out.startswith("__ERROR__"):
        return "ERROR: couldn't read GPU utilization via Windows performance counters."
    return (
        f"Total GPU 3D-engine utilization (all GPUs combined, AMD 7900 XT "
        f"+ Nvidia A1000): {out}%. No per-vendor split and no temperature "
        f"available without rocm-smi, which isn't installed on this "
        f"Windows box - see check_gpu_nvidia for the Nvidia card's own "
        f"temp/utilization/memory specifically."
    )


def check_gpu_nvidia(args: dict) -> str:
    """Report Nvidia GPU temp/utilization/memory via nvidia-smi. Verified
    live on real AI-Server hardware, 2026-09-10 - confirmed real output
    against the RTX A1000 actually installed there. Real bug caught in
    the same live test: nvidia-smi's raw CSV row (bare numbers, no
    labels) got the small fast-lane model to answer "running at 75%
    temperature" - it mixed up which column was which. Fixed by labeling
    every field explicitly instead of returning the bare CSV."""
    out = _run_ps(
        "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )
    if out.startswith("__ERROR__") or "not recognized" in out.lower():
        return "ERROR: nvidia-smi not found or failed to run - check it's on PATH."
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != 5:
        return f"ERROR: unexpected nvidia-smi output format: {out}"
    name, temp_c, util_pct, mem_used, mem_total = parts
    return (
        f"{name}: temperature {temp_c}C, utilization {util_pct}%, "
        f"memory {mem_used}MiB used of {mem_total}MiB total"
    )


# --- Hung process restart -------------------------------------------------
# Mechanism verified this session against the existing _kill_visualizer_
# processes pattern already proven live in tools.py (kills every PID
# owning a port, loops since Windows can leave more than one process
# bound to the same port) - this is a generalization of that same proven
# approach to "kill by process name" instead of "kill by port", not a new
# untested technique. Not re-run live this session since there's no
# actual hung Jarvis process on ADLAPTOPMAX to test against.

def restart_hung_process(args: dict) -> str:
    """Kill every process matching a name (e.g. 'ollama') and relaunch it
    via the given command. Deliberately requires BOTH process_name and
    relaunch_command - this is a real disruptive action (killing a live
    process), so wiring this into tools.py's REGISTRY should route it
    through the same confirmed=true soft-block gate run_shell already
    uses, not expose it unconfirmed."""
    process_name = args.get("process_name") if args else None
    relaunch_command = args.get("relaunch_command") if args else None
    if not process_name or not relaunch_command:
        return "ERROR: both process_name and relaunch_command are required."

    kill_out = _run_ps(
        f"Get-Process -Name '{process_name}' -ErrorAction SilentlyContinue | "
        "Stop-Process -Force -ErrorAction SilentlyContinue; "
        "Write-Output done"
    )
    if kill_out.startswith("__ERROR__"):
        return f"ERROR killing {process_name}: {kill_out}"

    try:
        subprocess.Popen(relaunch_command, shell=True)
    except OSError as e:
        return f"ERROR relaunching with {relaunch_command!r}: {e}"

    return f"OK: killed {process_name}, relaunched via: {relaunch_command}"
