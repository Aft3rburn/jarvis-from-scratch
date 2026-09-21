@echo off
rem ============================================================
rem  Jarvis crash-log launcher  (double-click version)
rem  - Stops any stale Jarvis voice instances first (one only)
rem  - Starts Jarvis minimized, detached
rem  - Writes ALL output to logs\jarvis-YYYYMMDD-HHMMSS.log
rem  If Jarvis crashes, the traceback will be in the newest log.
rem  Mary / my-agent is never touched by this script.
rem ============================================================

cd /d "%~dp0"

rem  Remote granite (2026-09-20): granite runs on ADLAPTOP over the LAN, with an automatic
rem  fallback to local granite if the laptop is unreachable. Remove the next line to go fully local.
set JARVIS_REMOTE_GRANITE_URL=http://192.168.131.203:11434/api/chat

echo [Jarvis] Stopping any existing Jarvis voice instances...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*jarvis-from-scratch*agent.py*' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }"
powershell -NoProfile -Command "Start-Sleep -Seconds 3"

if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set TS=%%i
set LOGFILE=logs\jarvis-%TS%.log

echo [Jarvis] Starting voice mode, logging to %LOGFILE% ...
start "JarvisVoice" /min cmd /c ".venv\Scripts\python.exe -u agent.py --voice >> %LOGFILE% 2>&1"

echo [Jarvis] Launched. If it crashes, open the newest file in the logs folder.
