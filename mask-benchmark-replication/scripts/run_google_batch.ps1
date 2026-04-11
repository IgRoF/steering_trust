<#
.SYNOPSIS
Run the Google batch generation and replay-scoring workflow.

.DESCRIPTION
This wrapper calls `run_google_batch.py`, checks the required local environment, and prints the exact Python command before launching.

.NOTES
The main knobs to tune are `-Mode`, `-ThinkingLevel`, `-ThinkingBudget`, and the optional artifact-root overrides.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$BatchModel,

    [Parameter(Mandatory = $true)]
    [string]$InspectModel,

    [ValidateSet("build", "smoke", "live-smoke")]
    [string]$Mode = "smoke",

    [string]$RunTag = "",

    [int]$Limit = 10,

    [int]$RequestConcurrency = 4,

    [int]$PollSeconds = 30,

    [int]$TimeoutMinutes = 1440,

    [string]$ThinkingLevel = "",

    [int]$ThinkingBudget = -1,

    [bool]$IncludeThoughts = $false,

    [string]$ArtifactRoot = "",

    [string]$ScoreLogRoot = "",

    [string]$Registry = "",

    [string]$InspectAppdataDir = "",

    [switch]$NoScore,

    [switch]$NoSubmit,

    [switch]$NoWait,

    [string]$BatchJobName = "",

    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found at $PythonExe. Create the virtual environment first or pass -PythonExe explicitly."
}
if ([string]::IsNullOrWhiteSpace($env:GOOGLE_API_KEY)) {
    throw "GOOGLE_API_KEY is required for the Google batch path."
}
if ([string]::IsNullOrWhiteSpace($env:HF_TOKEN) -and [string]::IsNullOrWhiteSpace($env:HUGGINGFACE_TOKEN)) {
    throw "HF_TOKEN or HUGGINGFACE_TOKEN is required for real MASK runs."
}

$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $safeModel = ($BatchModel -replace "[^A-Za-z0-9]+", "_").Trim("_").ToLowerInvariant()
    $RunTag = "${safeModel}_$(Get-Date -Format 'yyyyMMdd')"
}

$args = @(
    ".\scripts\run_google_batch.py"
    $Mode
    "--run-tag"
    $RunTag
    "--batch-model"
    $BatchModel
    "--inspect-model"
    $InspectModel
    "--limit"
    $Limit
)

if (-not [string]::IsNullOrWhiteSpace($ThinkingLevel)) {
    $args += @("--thinking-level", $ThinkingLevel)
}
if ($ThinkingBudget -ge 0) {
    $args += @("--thinking-budget", $ThinkingBudget)
}
if (-not [string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $args += @("--artifact-root", $ArtifactRoot)
}
if (-not [string]::IsNullOrWhiteSpace($ScoreLogRoot)) {
    $args += @("--score-log-root", $ScoreLogRoot)
}
if (-not [string]::IsNullOrWhiteSpace($Registry)) {
    $args += @("--registry", $Registry)
}
if (-not [string]::IsNullOrWhiteSpace($InspectAppdataDir)) {
    $args += @("--inspect-appdata-dir", $InspectAppdataDir)
}
if ($IncludeThoughts) {
    $args += "--include-thoughts"
}
else {
    $args += "--no-include-thoughts"
}

if ($Mode -eq "smoke") {
    if ($NoSubmit) { $args += "--no-submit" }
    if ($NoWait) { $args += "--no-wait" }
    if ($NoScore) { $args += "--no-score" }
    if (-not [string]::IsNullOrWhiteSpace($BatchJobName)) {
        $args += @("--batch-job-name", $BatchJobName)
    }
    $args += @("--poll-seconds", $PollSeconds)
    $args += @("--timeout-minutes", $TimeoutMinutes)
}
elseif ($Mode -eq "live-smoke") {
    if ($NoScore) { $args += "--no-score" }
    $args += @("--request-concurrency", $RequestConcurrency)
}

Write-Host ("PYTHON_COMMAND=" + ($PythonExe + " " + ($args -join " ")))
& $PythonExe @args
if ($LASTEXITCODE -ne 0) {
    throw "Google batch workflow failed for $RunTag with exit code $LASTEXITCODE"
}
