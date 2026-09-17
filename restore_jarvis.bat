@echo off
echo.
echo ================================
echo    Jarvis Agent Restore Script
echo ================================
echo.
echo This script will restore your Jarvis agent from a backup file.
echo.
echo Make sure you have a backup zip file named like "jarvis_backup_*.zip"
echo in the same directory as this script.
echo.
echo Do NOT run this script while Jarvis is actively running.
echo.
pause

rem Get the backup file name
set "BACKUP_FILE="

for %%f in (jarvis_backup_*.zip) do (
    if "%BACKUP_FILE%"=="" set "BACKUP_FILE=%%f"
)

if "%BACKUP_FILE%"=="" (
    echo No backup file found!
    echo Please place a jarvis_backup_*.zip file in this directory.
    pause
    exit /b 1
)

echo Found backup: %BACKUP_FILE%
echo.
echo Restoring to current directory...
powershell -Command "Expand-Archive -Path '%BACKUP_FILE%' -DestinationPath '.' -Force"

echo.
echo Restore complete!
echo.
echo Please verify the following:
echo 1. All Python files are present
echo 2. Vault directory is restored
echo 3. Visualizer directory is restored
echo.
pause