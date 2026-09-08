# Waits for real evidence that Mary has finished booting - a live
# backtalk.main process - before letting Jarvis's autostart continue.
# Evidence over a blind delay, per the troubleshooting discipline this
# project now carries in vault/INDEX.md.

$deadline = (Get-Date).AddMinutes(3)
Write-Host "Waiting for Mary (backtalk.main) to finish booting..."

while ((Get-Date) -lt $deadline) {
    $found = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*backtalk.main*' }
    if ($found) {
        Write-Host "Mary is up. Starting Jarvis..."
        exit 0
    }
    Start-Sleep -Seconds 5
}

Write-Host "Timed out after 3 minutes waiting for Mary - starting Jarvis anyway."
exit 0
