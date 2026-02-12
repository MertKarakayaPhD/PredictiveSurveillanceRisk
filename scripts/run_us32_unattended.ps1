param(
    [string]$Config = "data/external/metro_batch/metros_us_32.json",
    [string]$CameraCatalogCsv = "data/external/camera_catalog/cameras_us_active.csv.gz",
    [string[]]$MetroIds = @(),
    [string[]]$ExcludeMetroIds = @(),
    [switch]$IncludeChicago,
    [int]$Workers = 14,
    [int]$MinWorkers = 8,
    [int]$WorkerStepDown = 2,
    [int]$MaxRetriesPerMetro = 3,
    [int]$CooldownMinutes = 3,
    [int]$BlasThreads = 1,
    [int]$MpChunksize = 2,
    [switch]$DisableRouteCache,
    [switch]$DisableNodeCameraCache,
    [int]$RouteCacheSize = 200000,
    [int]$NodeCameraCacheSize = 200000,
    [string]$PythonExe = "python",
    [string]$LogsRoot = "logs/us32_unattended",
    [int]$KeepLastRuns = 12,
    [switch]$SkipPreflight,
    [switch]$RequireFreshData,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$ConfigAbs = (Resolve-Path (Join-Path $RepoRoot $Config)).Path
$CatalogAbs = (Resolve-Path (Join-Path $RepoRoot $CameraCatalogCsv)).Path

if (-not (Test-Path $ConfigAbs)) { throw "Config not found: $ConfigAbs" }
if (-not (Test-Path $CatalogAbs)) { throw "Camera catalog not found: $CatalogAbs" }
if ($Workers -le 0 -or $MinWorkers -le 0 -or $WorkerStepDown -le 0 -or $MaxRetriesPerMetro -le 0) {
    throw "Workers/MinWorkers/WorkerStepDown/MaxRetriesPerMetro must be >= 1"
}
if ($MinWorkers -gt $Workers) { $MinWorkers = $Workers }
if ($BlasThreads -le 0) { throw "BlasThreads must be >= 1" }
if ($MpChunksize -le 0) { throw "MpChunksize must be >= 1" }
if ($RouteCacheSize -le 0) { throw "RouteCacheSize must be >= 1" }
if ($NodeCameraCacheSize -le 0) { throw "NodeCameraCacheSize must be >= 1" }

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$LogsRootAbs = Join-Path $RepoRoot $LogsRoot
$RunDir = Join-Path $LogsRootAbs $RunId
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$RunLog = Join-Path $RunDir "unattended_run.log"
$StatePath = Join-Path $RunDir "state.json"
$SummaryLog = Join-Path $RunDir "summary.log"

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
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
    $State["updated_at_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    $json = $State | ConvertTo-Json -Depth 10
    Set-Content -Path $StatePath -Value $json -Encoding UTF8
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
        $outputLines = & $exe @args 2>&1 |
            ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
            }
        if ($outputLines) {
            $outputLines | Tee-Object -FilePath $LogPath -Append | Out-Host
        }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    Write-Log "Step '$Name' exit code: $exitCode"

    if ($exitCode -ne 0 -and -not $IgnoreFailure) {
        throw "Step '$Name' failed with exit code $exitCode"
    }
    return [int]$exitCode
}

Rotate-RunLogs -RootDir $LogsRootAbs -Keep $KeepLastRuns

$payload = Get-Content -Path $ConfigAbs -Raw | ConvertFrom-Json
$allMetroIds = @($payload.metros | ForEach-Object { [string]$_.id })
if (-not $allMetroIds -or $allMetroIds.Count -eq 0) {
    throw "No metros found in config: $ConfigAbs"
}

$queue = @()
if ($MetroIds.Count -gt 0) {
    $queue = @($MetroIds | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ } | Select-Object -Unique)
} else {
    $queue = @($allMetroIds | ForEach-Object { $_.Trim().ToLower() } | Select-Object -Unique)
}

$exclude = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
if (-not $IncludeChicago) {
    foreach ($m in $ExcludeMetroIds) {
        if ($m -and $m.Trim()) { $null = $exclude.Add($m.Trim().ToLower()) }
    }
}
$queue = @($queue | Where-Object { -not $exclude.Contains($_) })

$known = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($m in $allMetroIds) { $null = $known.Add($m.Trim().ToLower()) }
$unknown = @($queue | Where-Object { -not $known.Contains($_) })
if ($unknown.Count -gt 0) {
    throw "Unknown metro ids requested: $($unknown -join ', ')"
}
if ($queue.Count -eq 0) {
    throw "Metro queue is empty after filters."
}

$state = @{
    run_id = $RunId
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    status = "running"
    config_path = $Config
    camera_catalog_csv = $CameraCatalogCsv
    queue = $queue
    params = @{
        workers = $Workers
        min_workers = $MinWorkers
        worker_step_down = $WorkerStepDown
        max_retries_per_metro = $MaxRetriesPerMetro
        cooldown_minutes = $CooldownMinutes
        blas_threads = $BlasThreads
        mp_chunksize = $MpChunksize
        disable_route_cache = [bool]$DisableRouteCache
        disable_node_camera_cache = [bool]$DisableNodeCameraCache
        route_cache_size = $RouteCacheSize
        node_camera_cache_size = $NodeCameraCacheSize
        skip_preflight = [bool]$SkipPreflight
        require_fresh_data = [bool]$RequireFreshData
        include_chicago = [bool]$IncludeChicago
        dry_run = [bool]$DryRun
    }
    metros = @{}
}
foreach ($m in $queue) {
    $state.metros[$m] = @{
        status = "pending"
        attempts = 0
        last_workers = $null
        last_exit_code = $null
        started_at_utc = $null
        ended_at_utc = $null
        note = $null
    }
}
Save-State -State $state

Write-Log "Run ID: $RunId"
Write-Log "Queue size: $($queue.Count)"
Write-Log "Queue order: $($queue -join ', ')"
Write-Log "Workers=$Workers MinWorkers=$MinWorkers StepDown=$WorkerStepDown Retries=$MaxRetriesPerMetro"
Write-Log (
    "BLAS threads=$BlasThreads, mp_chunksize=$MpChunksize, " +
    "route_cache=$([bool](-not $DisableRouteCache))($RouteCacheSize), " +
    "node_camera_cache=$([bool](-not $DisableNodeCameraCache))($NodeCameraCacheSize)"
)

$env:OMP_NUM_THREADS = "$BlasThreads"
$env:MKL_NUM_THREADS = "$BlasThreads"
$env:OPENBLAS_NUM_THREADS = "$BlasThreads"
$env:NUMEXPR_NUM_THREADS = "$BlasThreads"

if (-not $SkipPreflight) {
    $preflightLog = Join-Path $RunDir "preflight.log"
    $freshJson = Join-Path $RunDir "data_freshness.json"
    $freshCmd = @(
        $PythonExe,
        "scripts/check_data_freshness.py",
        "--config", $Config,
        "--metro-ids", ($queue -join ","),
        "--camera-catalog-csv", $CameraCatalogCsv,
        "--strict-missing",
        "--output-json", $freshJson
    )
    if ($RequireFreshData) { $freshCmd += "--require-fresh" }
    $null = Invoke-LoggedCommand -Name "preflight_freshness" -Command $freshCmd -LogPath $preflightLog
} else {
    Write-Log "Preflight skipped by flag."
}

$fallbackMetros = @("minneapolis_st_paul_mn", "st_louis_mo", "las_vegas_nv", "kansas_city_mo")
$fallbackSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($m in $fallbackMetros) { $null = $fallbackSet.Add($m) }

$anyFailure = $false

foreach ($metro in $queue) {
    $state.metros[$metro].status = "running"
    $state.metros[$metro].started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    Save-State -State $state

    $metroSucceeded = $false
    $metroLog = Join-Path $RunDir "metro_${metro}.log"
    $baseWorkers = $Workers
    if ($fallbackSet.Contains($metro)) {
        $baseWorkers = [Math]::Min($Workers, 10)
    }

    for ($attempt = 1; $attempt -le $MaxRetriesPerMetro; $attempt++) {
        $workersThisAttempt = [Math]::Max($MinWorkers, $baseWorkers - (($attempt - 1) * $WorkerStepDown))

        $state.metros[$metro].attempts = $attempt
        $state.metros[$metro].last_workers = $workersThisAttempt
        $state.metros[$metro].note = "Attempt $attempt/$MaxRetriesPerMetro"
        Save-State -State $state

        $cmd = @(
            $PythonExe,
            "scripts/run_metro_batch.py",
            "--config", $Config,
            "--metro-ids", $metro,
            "--camera-catalog-csv", $CameraCatalogCsv,
            "--workers", "$workersThisAttempt",
            "--mp-chunksize", "$MpChunksize",
            "--route-cache-size", "$RouteCacheSize",
            "--node-camera-cache-size", "$NodeCameraCacheSize",
            "--blas-threads", "$BlasThreads",
            "--keep-last-runs", "$KeepLastRuns",
            "--skip-preflight"
        )
        if ($DisableRouteCache) { $cmd += "--disable-route-cache" }
        if ($DisableNodeCameraCache) { $cmd += "--disable-node-camera-cache" }
        if ($DryRun) { $cmd += "--dry-run" }

        $rc = 0
        try {
            $rc = Invoke-LoggedCommand -Name "metro_${metro}_attempt_${attempt}" -Command $cmd -LogPath $metroLog -IgnoreFailure
        } catch {
            $rc = if ($LASTEXITCODE) { $LASTEXITCODE } else { 1 }
        }

        $state.metros[$metro].last_exit_code = $rc
        Save-State -State $state

        if ($rc -eq 0) {
            $metroSucceeded = $true
            break
        }

        if ($attempt -lt $MaxRetriesPerMetro) {
            Write-Log "Metro $metro failed on attempt $attempt (exit=$rc). Cooling down for $CooldownMinutes minute(s), then retry."
            Start-Sleep -Seconds ([Math]::Max(1, $CooldownMinutes * 60))
        }
    }

    $state.metros[$metro].ended_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    if ($metroSucceeded) {
        $state.metros[$metro].status = "completed"
        $state.metros[$metro].note = "OK"
        Write-Log "Completed metro: $metro"
    } else {
        $state.metros[$metro].status = "failed"
        $state.metros[$metro].note = "Failed after $MaxRetriesPerMetro attempt(s)"
        Write-Log "Metro failed after retries: $metro" "ERROR"
        $anyFailure = $true
    }
    Save-State -State $state
}

Add-Content -Path $SummaryLog -Value "Run ID: $RunId"
Add-Content -Path $SummaryLog -Value ("Started: " + $state.started_at_utc)
Add-Content -Path $SummaryLog -Value ("Ended:   " + (Get-Date).ToUniversalTime().ToString("o"))
Add-Content -Path $SummaryLog -Value ""
Add-Content -Path $SummaryLog -Value "Per-metro status:"
foreach ($metro in $queue) {
    $mstate = $state.metros[$metro]
    $line = "{0,-30} {1,-12} attempts={2,-3} workers={3,-3} exit={4,-4} note={5}" -f $metro, $mstate.status, $mstate.attempts, $mstate.last_workers, $mstate.last_exit_code, $mstate.note
    Add-Content -Path $SummaryLog -Value $line
}

$state.ended_at_utc = (Get-Date).ToUniversalTime().ToString("o")
if ($anyFailure) {
    $state.status = "completed_with_failures"
    Save-State -State $state
    Write-Log "Unattended queue finished with failures. See $SummaryLog" "WARN"
    exit 1
}

$state.status = "completed"
Save-State -State $state
Write-Log "Unattended queue completed successfully. Summary: $SummaryLog"
exit 0
