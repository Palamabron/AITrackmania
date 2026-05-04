param(
    [Parameter(Mandatory = $true)]
    [int] $Port
)
# Free TCP listeners on $Port. Get-NetTCPConnection sometimes misses processes or needs
# elevation; netstat -ano is a reliable fallback (same idea as `lsof` on Unix).

function Add-Pid {
    param([int] $Id)
    if ($Id -gt 0) { [void]$script:Pids.Add($Id) }
}

$script:Pids = [System.Collections.Generic.HashSet[int]]::new()

try {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Add-Pid -Id $_.OwningProcess }
    Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        ForEach-Object { Add-Pid -Id $_.OwningProcess }
} catch {
    # ignore; netstat below still runs
}

# netstat output is English (LISTENING) even on localized Windows.
foreach ($line in (netstat -ano 2>$null)) {
    if ($line -notmatch "LISTENING") { continue }
    if ($line -notmatch "[:.]$Port\s") { continue }
    if ($line -match "\s+(\d+)\s*$") {
        Add-Pid -Id ([int]$Matches[1])
    }
}

if ($Pids.Count -eq 0) {
    Write-Host "No process found listening on TCP port $Port."
    exit 0
}

foreach ($targetPid in $Pids) {
    try {
        $p = Get-Process -Id $targetPid -ErrorAction Stop
        Write-Host "Stopping PID $targetPid ($($p.ProcessName)) holding port $Port..."
        Stop-Process -Id $targetPid -Force -ErrorAction Stop
    } catch {
        Write-Warning "Could not stop PID ${targetPid}: $_"
        exit 1
    }
}

exit 0
