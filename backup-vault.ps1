# Backs up Jarvis's local vault (vault\INDEX.md + vault\daily\*) to
# D:\MaryBackup\JarvisVault, the same physical drive Mary's own vault
# mirror already lives on (this machine absorbed the old NAS's drives).
# Mirrors Mary's own backup-mary.ps1 pattern: robocopy /MIR, logged.

$source = "C:\Users\mwilo\jarvis-from-scratch\vault"
$dest = "D:\MaryBackup\JarvisVault"
$log = "C:\Users\mwilo\jarvis-from-scratch\backup-vault.log"

"[$(Get-Date)] Jarvis vault backup starting" | Out-File -Append -Encoding ascii $log

robocopy $source $dest /MIR /R:2 /W:5 /LOG+:$log /TEE

"[$(Get-Date)] Jarvis vault backup finished, exit code $LASTEXITCODE" | Out-File -Append -Encoding ascii $log
