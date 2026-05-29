# ALS Control Interface — one-click launcher
# Kills any existing query server, starts a fresh one, opens the browser.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = "python"
$port      = 5175
$url       = "http://localhost:$port"

# ── Kill existing query servers ───────────────────────────────────────────────
$existing = Get-WmiObject Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -like "*query.py*" }
foreach ($p in $existing) {
  Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($existing) { Start-Sleep -Seconds 1 }

# ── Start server ──────────────────────────────────────────────────────────────
$logOut = Join-Path $scriptDir "query_server.log"
$logErr = Join-Path $scriptDir "query_server_err.log"

$proc = Start-Process -FilePath $pythonExe `
  -ArgumentList "-u", "query.py", "--serve", "--port", $port `
  -WorkingDirectory $scriptDir `
  -RedirectStandardOutput $logOut `
  -RedirectStandardError  $logErr `
  -WindowStyle Hidden -PassThru

# ── Wait for ready ────────────────────────────────────────────────────────────
$deadline = (Get-Date).AddSeconds(20)
$ready    = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 600
  try {
    $r = Invoke-WebRequest -Uri "$url/api/status" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($r.StatusCode -eq 200) { $ready = $true; break }
  } catch {}
}

if (-not $ready) {
  Write-Host "Server may still be loading — opening browser anyway."
}

# ── Open browser ──────────────────────────────────────────────────────────────
Start-Process $url
Write-Host "ALS Control Interface → $url  (PID $($proc.Id))"
