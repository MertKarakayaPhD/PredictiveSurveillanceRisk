param(
    [string[]]$Regions = @("atlanta", "memphis", "richmond", "charlotte", "lehigh_valley", "maine"),
    [int]$NVehicles = 5000,
    [int]$NTrips = 10,
    [int]$Seed = 42,
    [int]$Workers = [Math]::Max(1, [Environment]::ProcessorCount - 2),
    [string]$PythonExe = "python",
    [string]$LogsRoot = "logs/weekend",
    [int]$KeepLastRuns = 8,
    [switch]$ForceRerun,
    [switch]$SkipPreflight,
    [switch]$RequireFreshData
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$LogsRootAbs = (Join-Path $RepoRoot $LogsRoot)
$RunDir = Join-Path $LogsRootAbs $RunId
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$RunLog = Join-Path $RunDir "weekend_run.log"
$StatePath = Join-Path $RunDir "state.json"

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts][$Level] $Message"
    Write-Host $line
    Add-Content -Path $RunLog -Value $line
}

function Rotate-RunLogs {
    param(
        [string]$RootDir,
        [int]$Keep
    )
    if (-not (Test-Path $RootDir)) { return }

    $dirs = @(Get-ChildItem -Path $RootDir -Directory | Sort-Object LastWriteTime -Descending)
    if ($dirs.Length -le $Keep) { return }

    $toArchive = $dirs[$Keep..($dirs.Length - 1)]
    foreach ($d in $toArchive) {
        $zipPath = "$($d.FullName).zip"
        try {
            if (-not (Test-Path $zipPath)) {
                Compress-Archive -Path (Join-Path $d.FullName "*") -DestinationPath $zipPath -CompressionLevel Optimal -Force
            }
            Remove-Item -Path $d.FullName -Recurse -Force
        } catch {
            Write-Log "Log rotation failed for $($d.FullName): $($_.Exception.Message)" "WARN"
        }
    }
}

function Save-State {
    param([hashtable]$State)
    $json = $State | ConvertTo-Json -Depth 8
    Set-Content -Path $StatePath -Value $json
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string[]]$Command,
        [string]$LogPath,
        [switch]$IgnoreFailure
    )
    Write-Log "Running step: $Name"
    Add-Content -Path $LogPath -Value ("`n===== " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " | $Name =====")

    $exe = $Command[0]
    $args = @()
    if ($Command.Count -gt 1) {
        $args = $Command[1..($Command.Count - 1)]
    }

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $exe @args 2>&1 |
            ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
            } |
            Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    Write-Log "Step '$Name' exit code: $exitCode"

    if ($exitCode -ne 0 -and -not $IgnoreFailure) {
        throw "Step '$Name' failed with exit code $exitCode"
    }
    return $exitCode
}

Rotate-RunLogs -RootDir $LogsRootAbs -Keep $KeepLastRuns

$state = @{
    run_id = $RunId
    started_at = (Get-Date).ToString("o")
    status = "running"
    params = @{
        regions = $Regions
        n_vehicles = $NVehicles
        n_trips = $NTrips
        seed = $Seed
        workers = $Workers
        force_rerun = [bool]$ForceRerun
        skip_preflight = [bool]$SkipPreflight
        require_fresh_data = [bool]$RequireFreshData
    }
    regions = @{}
}
foreach ($r in $Regions) {
    $state.regions[$r] = @{
        status = "pending"
        started_at = $null
        ended_at = $null
        output = "results/traffic_weighted/road_trajectories_${r}.pkl"
        log = "region_${r}.log"
        exit_code = $null
        note = $null
    }
}
Save-State -State $state

Write-Log "Weekend run started. RunDir: $RunDir"
Write-Log "Repo root: $RepoRoot"
Write-Log "Regions: $($Regions -join ', ')"
Write-Log "n_vehicles=$NVehicles n_trips=$NTrips seed=$Seed workers=$Workers"

# Reduce thread over-subscription and memory pressure.
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
Write-Log "Set OMP/MKL/OPENBLAS/NUMEXPR threads to 1"

if (-not $SkipPreflight) {
    $preflightLog = Join-Path $RunDir "preflight.log"
    $freshJson = Join-Path $RunDir "data_freshness.json"

    $freshCmd = @($PythonExe, "scripts/check_data_freshness.py", "--strict-missing", "--output-json", $freshJson)
    if ($RequireFreshData) {
        $freshCmd += "--require-fresh"
    }
    $null = Invoke-LoggedCommand -Name "data_freshness" -Command $freshCmd -LogPath $preflightLog

    $null = Invoke-LoggedCommand -Name "aadt_validate" -Command @($PythonExe, "scripts/download_aadt_data.py", "--validate") -LogPath $preflightLog
    $null = Invoke-LoggedCommand -Name "tests" -Command @($PythonExe, "-m", "pytest", "-q") -LogPath $preflightLog
} else {
    Write-Log "Preflight skipped by flag."
}

$anyFailure = $false

foreach ($region in $Regions) {
    $regionLog = Join-Path $RunDir "region_${region}.log"
    $outputPath = Join-Path $RepoRoot "results/traffic_weighted/road_trajectories_${region}.pkl"

    if ((Test-Path $outputPath) -and -not $ForceRerun) {
        Write-Log "Skipping $region (output already exists): $outputPath"
        $state.regions[$region].status = "skipped_existing"
        $state.regions[$region].note = "Output exists and -ForceRerun not set"
        $state.regions[$region].ended_at = (Get-Date).ToString("o")
        Save-State -State $state
        continue
    }

    $state.regions[$region].status = "running"
    $state.regions[$region].started_at = (Get-Date).ToString("o")
    Save-State -State $state

    Write-Log "Starting region: $region"

    if ((Test-Path $outputPath) -and $ForceRerun) {
        Remove-Item -Path $outputPath -Force
        Write-Log "Removed existing output for force rerun: $outputPath" "WARN"
    }

    $cmd = @(
        $PythonExe, "-u", "scripts/run_traffic_weighted_simulation.py",
        "--region", $region,
        "--n-vehicles", "$NVehicles",
        "--n-trips", "$NTrips",
        "--seed", "$Seed",
        "--workers", "$Workers",
        "--no-archive"
    )

    try {
        $code = Invoke-LoggedCommand -Name "simulate_$region" -Command $cmd -LogPath $regionLog
        $state.regions[$region].exit_code = $code
        if ($code -eq 0 -and (Test-Path $outputPath)) {
            $state.regions[$region].status = "completed"
            $state.regions[$region].note = "OK"
            Write-Log "Completed region: $region"
        } else {
            $state.regions[$region].status = "failed"
            $state.regions[$region].note = "Process exited without expected output artifact"
            $anyFailure = $true
            Write-Log "Region failed (missing output): $region" "ERROR"
        }
    } catch {
        $state.regions[$region].status = "failed"
        $state.regions[$region].exit_code = $LASTEXITCODE
        $state.regions[$region].note = $_.Exception.Message
        $anyFailure = $true
        Write-Log "Region failed: $region | $($_.Exception.Message)" "ERROR"
    } finally {
        $state.regions[$region].ended_at = (Get-Date).ToString("o")
        Save-State -State $state
    }
}

$summaryLog = Join-Path $RunDir "summary.log"
Add-Content -Path $summaryLog -Value "Run ID: $RunId"
Add-Content -Path $summaryLog -Value ("Started: " + $state.started_at)
Add-Content -Path $summaryLog -Value ("Ended:   " + (Get-Date).ToString("o"))
Add-Content -Path $summaryLog -Value ""
Add-Content -Path $summaryLog -Value "Per-region status:"
foreach ($region in $Regions) {
    $rstate = $state.regions[$region]
    $line = "{0,-15} {1,-16} exit={2,-4} note={3}" -f $region, $rstate.status, $rstate.exit_code, $rstate.note
    Add-Content -Path $summaryLog -Value $line
}

$state.ended_at = (Get-Date).ToString("o")
if ($anyFailure) {
    $state.status = "completed_with_failures"
    Save-State -State $state
    Write-Log "Weekend run finished with failures. See $summaryLog" "WARN"
    exit 1
}

$state.status = "completed"
Save-State -State $state
Write-Log "Weekend run completed successfully. Summary: $summaryLog"
exit 0
