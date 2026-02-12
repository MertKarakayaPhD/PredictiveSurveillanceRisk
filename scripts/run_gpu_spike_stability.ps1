param(
    [double]$DurationHours = 2.0,
    [string]$MetroIds = "philadelphia_pa,chicago_il",
    [string]$Backends = "cpu,torch-cpu,torch-cuda",
    [int]$NVehicles = 50,
    [int]$NTrips = 4,
    [int]$WorkersCpu = 12,
    [int]$WorkersGpu = 1,
    [int]$MpChunksize = 2,
    [int]$Seed = 42,
    [string]$OutputRoot = "results/gpu_spike",
    [string]$LogsRoot = "logs/gpu_spike_stability",
    [int]$PauseSeconds = 15,
    [switch]$StopOnFailure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $RepoRoot (Join-Path $LogsRoot $RunId)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$logPath = Join-Path $RunDir "stability.log"
$summaryCsv = Join-Path $RunDir "stability_summary.csv"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts][$Level] $Message"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

$start = Get-Date
$end = $start.AddHours($DurationHours)
$cycle = 0

Write-Log "GPU spike stability run started. RunId=$RunId"
Write-Log "DurationHours=$DurationHours MetroIds=$MetroIds Backends=$Backends"
Write-Log "NVehicles=$NVehicles NTrips=$NTrips WorkersCpu=$WorkersCpu WorkersGpu=$WorkersGpu MpChunksize=$MpChunksize"
Write-Log "Artifacts: $RunDir"

if (-not (Test-Path $summaryCsv)) {
    "cycle,start_utc,end_utc,exit_code,csv_path" | Set-Content -Path $summaryCsv -Encoding UTF8
}

while ((Get-Date) -lt $end) {
    $cycle += 1
    $cycleStart = Get-Date
    $benchCsv = Join-Path $RunDir ("benchmark_cycle_{0:000}.csv" -f $cycle)

    $cmd = @(
        "python",
        "scripts/benchmark_gpu_spike.py",
        "--metro-ids", $MetroIds,
        "--backends", $Backends,
        "--n-vehicles", "$NVehicles",
        "--n-trips", "$NTrips",
        "--workers-cpu", "$WorkersCpu",
        "--workers-gpu", "$WorkersGpu",
        "--mp-chunksize", "$MpChunksize",
        "--seed", "$Seed",
        "--output-root", $OutputRoot,
        "--csv-out", $benchCsv
    )

    Write-Log ("Cycle $cycle started.")
    Add-Content -Path $logPath -Value ("CMD: " + ($cmd -join " "))

    & $cmd[0] $cmd[1..($cmd.Count - 1)] 2>&1 |
        ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ } } |
        Tee-Object -FilePath $logPath -Append | Out-Host
    $rc = $LASTEXITCODE

    $cycleEnd = Get-Date
    $row = "{0},{1},{2},{3},{4}" -f $cycle, $cycleStart.ToUniversalTime().ToString("o"), $cycleEnd.ToUniversalTime().ToString("o"), $rc, $benchCsv
    Add-Content -Path $summaryCsv -Value $row

    if ($rc -ne 0) {
        Write-Log "Cycle $cycle failed (exit=$rc)." "ERROR"
        if ($StopOnFailure) {
            Write-Log "StopOnFailure is set; terminating stability run." "ERROR"
            exit $rc
        }
    } else {
        Write-Log "Cycle $cycle completed successfully."
    }

    if ((Get-Date) -lt $end) {
        Start-Sleep -Seconds ([Math]::Max(0, $PauseSeconds))
    }
}

Write-Log "Stability run finished."
Write-Log "Summary CSV: $summaryCsv"
exit 0

