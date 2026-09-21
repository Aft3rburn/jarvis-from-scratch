@echo off
rem ============================================================
rem  Jarvis task launcher (runs from Windows Task Scheduler)
rem  Same as start_jarvis_logged.bat but WITHOUT 'start':
rem  a scheduled task is already detached, and 'start' needs
rem  an interactive desktop. Direct launch + logging instead.
rem  Mary / my-agent is never touched by this script.
rem ============================================================

cd /d "%~dp0"

rem  Remote granite (2026-09-20): granite runs on ADLAPTOP over the LAN, with an automatic
rem  fallback to local granite if the laptop is unreachable. Remove the next line to go fully local.
set JARVIS_REMOTE_GRANITE_URL=http://192.168.131.203:11434/api/chat

powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*jarvis-from-scratch*agent.py*' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }"
powershell -NoProfile -Command "Start-Sleep -Seconds 3"

if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set TS=%%i

.venv\Scripts\python.exe -u agent.py --voice >> "logs\jarvis-%TS%.log" 2>&1
