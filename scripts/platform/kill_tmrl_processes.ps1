# Stop TrackmaniaRL / AITrackmania Python trees and free distributed ports (55555-55558).
# Broader than Makefile's old taskkill /IM python.exe (misses python3.12.exe, uv children, wandb).

param(
    [int] $ServerPort = 55555,
    [switch] $AllPython
)

$ErrorActionPreference = "Continue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$killPortScript = Join-Path $PSScriptRoot "kill_tcp_port.ps1"

function Stop-ProcessTree {
    param([int] $ProcessId)
    if ($ProcessId -le 0) { return }
    $null = & taskkill /T /F /PID $ProcessId 2>&1
}

Write-Host "=== kill_trackmaniarl_processes ==="
Write-Host "Repo: $repoRoot"

# 1) Free relay ports (server + tlspyo local ports).
for ($port = $ServerPort; $port -lt ($ServerPort + 4); $port++) {
    if (Test-Path $killPortScript) {
        & $killPortScript -Port $port
    }
}

# 2) Kill processes whose command line references this repo or trackmaniarl.
$repoLeaf = Split-Path -Leaf $repoRoot
$patterns = @(
    [regex]::Escape($repoRoot),
    [regex]::Escape($repoLeaf),
    '-m trackmaniarl',
    'trackmaniarl\.tools',
    'trackmaniarl\\',
    'trackmaniarl/'
)

$matched = @()
try {
    Get-CimInstance Win32_Process -ErrorAction Stop |
        Where-Object {
            $_.Name -match '^(python|pythonw|uv)' -and
            $_.CommandLine
        } |
        ForEach-Object {
            $cmd = $_.CommandLine
            $hit = $false
            foreach ($p in $patterns) {
                if ($cmd -match $p) { $hit = $true; break }
            }
            if ($hit) {
                $matched += $_
            }
        }
} catch {
    Write-Warning "Win32_Process query failed (try running PowerShell as Administrator): $_"
}

$seen = [System.Collections.Generic.HashSet[int]]::new()
foreach ($proc in $matched) {
    if ($seen.Add($proc.ProcessId)) {
        Write-Host "Stopping PID $($proc.ProcessId) ($($proc.Name)): $($proc.CommandLine.Substring(0, [Math]::Min(120, $proc.CommandLine.Length)))..."
        Stop-ProcessTree -ProcessId $proc.ProcessId
    }
}

# 3) Optional: all Python interpreters (destructive - other projects too).
if ($AllPython) {
    Write-Host "AllPython: stopping remaining python*/uv* processes..."
    Get-Process -Name "python", "python3*", "pythonw*", "uv" -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($seen.Add($_.Id)) {
                Write-Host "Stopping $($_.ProcessName) PID $($_.Id)..."
                Stop-ProcessTree -ProcessId $_.Id
            }
        }
}

Start-Sleep -Seconds 2

$remaining = @()
try {
    $remaining = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '^(python|pythonw|uv)' -and
            $_.CommandLine -and (
                $_.CommandLine -match [regex]::Escape($repoRoot) -or
                $_.CommandLine -match 'trackmaniarl'
            )
        }
} catch { }

if ($remaining.Count -gt 0) {
    Write-Warning "Still running ($($remaining.Count) TrackmaniaRL-related process(es)). Retry as Administrator or close TrackMania terminals, then:"
    Write-Warning "  powershell -File scripts/platform/kill_trackmaniarl_processes.ps1 -AllPython"
    exit 1
}

Write-Host "Done. No TrackmaniaRL-related Python processes detected."
exit 0
