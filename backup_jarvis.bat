@echo off
echo Backing up Jarvis agent...
set "BACKUP_NAME=jarvis_backup_%date:~10,4%-%date:~4,2%-%date:~7,2%.zip"
echo Creating backup: %BACKUP_NAME%

rem Create a temporary directory for backup
set "TEMP_DIR=%temp%\jarvis_backup_temp"
if exist "%TEMP_DIR%" (
    echo Cleaning up existing temp directory...
    rmdir /s /q "%TEMP_DIR%"
)
mkdir "%TEMP_DIR%"

rem Copy all necessary files to temp directory
echo Copying agent files...
xcopy /E /I /H "*.py" "%TEMP_DIR%\"
xcopy /E /I /H "vault" "%TEMP_DIR%\vault\"
xcopy /E /I /H "visualizer" "%TEMP_DIR%\visualizer\"

rem Create the zip archive
echo Creating zip file...
powershell -Command "Compress-Archive -Path '%TEMP_DIR%' -DestinationPath '%BACKUP_NAME%' -Force"

rem Clean up temp directory
rmdir /s /q "%TEMP_DIR%"

echo Backup complete: %BACKUP_NAME%
pause