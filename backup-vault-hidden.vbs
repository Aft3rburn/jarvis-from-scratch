Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""C:\Users\mwilo\jarvis-from-scratch\backup-vault.ps1""", 0, False
