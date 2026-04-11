<#
.SYNOPSIS
Run a MASK evaluation with one public entrypoint.

.DESCRIPTION
This wrapper creates an output folder, prints the exact Inspect command, checks
the most important environment variables, and then runs
`inspect eval inspect_evals/mask`.

Use `-Mode smoke` for the lighter 10-sample defaults or `-Mode full` for the
1,000-sample defaults. You can still override `-Limit`, `-MaxConnections`, and
`-MaxSamples` yourself when you want a custom run size.

.NOTES
The first knobs to tune are `-Mode`, `-Limit`, `-MaxConnections`,
`-MaxSamples`, and `-Timeout`.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",

    [Nullable[int]]$Limit = $null,

    [string]$RunTag = "",

    [int]$MaxRetries = 15,

    [int]$Timeout = 180,

    [string]$InspectExe = ".\.venv\Scripts\inspect.exe",

    [Nullable[int]]$MaxConnections = $null,

    [Nullable[int]]$MaxSamples = $null,

    [int]$RetryOnError = 3,

    [int]$FailOnError = 25,

    [bool]$ContinueOnFail = $true,

    [int]$LogBuffer = 1,

    [string]$Display = "plain",

    [string]$OutputRoot = ".\outputs\runs",

    [switch]$LogModelApi,

    [switch]$LogRefusals,

    [switch]$NoLogRealtime,

    [switch]$NoScoreDisplay
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-SafeTag {
    param([string]$Text)
    $safe = $Text.ToLowerInvariant()
    $safe = $safe -replace '[\\/:\.\s]+', '_'
    $safe = $safe -replace '[^a-z0-9_]+', ''
    return $safe.Trim('_')
}

function Set-HuggingFaceAliases {
    if (-not [string]::IsNullOrWhiteSpace($env:HF_TOKEN) -and [string]::IsNullOrWhiteSpace($env:HUGGINGFACE_TOKEN)) {
        $env:HUGGINGFACE_TOKEN = $env:HF_TOKEN
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:HUGGINGFACE_TOKEN) -and [string]::IsNullOrWhiteSpace($env:HF_TOKEN)) {
        $env:HF_TOKEN = $env:HUGGINGFACE_TOKEN
    }
}

function Get-RequiredApiKeyName {
    param([string]$ModelName)
    switch -Regex ($ModelName) {
        '^openai/' { return 'OPENAI_API_KEY' }
        '^anthropic/' { return 'ANTHROPIC_API_KEY' }
        '^google/' { return 'GOOGLE_API_KEY' }
        '^openai-api/deepseek/' { return 'DEEPSEEK_API_KEY' }
        '^grok/' { return 'XAI_API_KEY' }
        '^openrouter/' { return 'OPENROUTER_API_KEY' }
        default { return $null }
    }
}

function Get-ModeDefaults {
    param([string]$SelectedMode)
    switch ($SelectedMode) {
        "smoke" {
            return @{
                Limit = 10
                MaxConnections = 32
                MaxSamples = 32
            }
        }
        "full" {
            return @{
                Limit = 1000
                MaxConnections = 128
                MaxSamples = 160
            }
        }
        default {
            throw "Unsupported mode: $SelectedMode"
        }
    }
}

$modeDefaults = Get-ModeDefaults -SelectedMode $Mode
if (-not $PSBoundParameters.ContainsKey("Limit")) {
    $Limit = $modeDefaults.Limit
}
if (-not $PSBoundParameters.ContainsKey("MaxConnections")) {
    $MaxConnections = $modeDefaults.MaxConnections
}
if (-not $PSBoundParameters.ContainsKey("MaxSamples")) {
    $MaxSamples = $modeDefaults.MaxSamples
}

$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

Set-HuggingFaceAliases

if (-not (Test-Path -LiteralPath $InspectExe)) {
    throw "inspect executable not found at $InspectExe. Create the virtual environment first or pass -InspectExe explicitly."
}
if ([string]::IsNullOrWhiteSpace($env:HF_TOKEN)) {
    throw "HF_TOKEN or HUGGINGFACE_TOKEN is required for real MASK runs."
}
$requiredApiKey = Get-RequiredApiKeyName -ModelName $Model
if ($null -ne $requiredApiKey -and [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($requiredApiKey))) {
    throw "$requiredApiKey is required for model $Model."
}

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = Get-SafeTag $Model
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resolvedOutputRoot = Join-Path $projectRoot $OutputRoot
$logDir = Join-Path $resolvedOutputRoot "${RunTag}_n${Limit}_$stamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "PROJECT_ROOT=$projectRoot"
Write-Host "MODE=$Mode"
Write-Host "LIMIT=$Limit"
Write-Host "MAX_CONNECTIONS=$MaxConnections"
Write-Host "MAX_SAMPLES=$MaxSamples"
Write-Host "LOG_DIR=$logDir"
Write-Host "INSPECT_EXE=$InspectExe"

$traceFile = Join-Path $logDir "trace.log"
$env:INSPECT_TRACE_FILE = $traceFile
Write-Host "INSPECT_TRACE_FILE=$traceFile"

$inspectArgs = @(
    "eval"
    "inspect_evals/mask"
    "--model"
    $Model
    "--display"
    $Display
    "--log-dir"
    $logDir
    "--max-retries"
    $MaxRetries
    "--timeout"
    $Timeout
    "--limit"
    $Limit
    "--max-connections"
    $MaxConnections
    "--max-samples"
    $MaxSamples
    "--retry-on-error"
    $RetryOnError
    "--fail-on-error"
    $FailOnError
    "--log-buffer"
    $LogBuffer
)

if ($ContinueOnFail) {
    $inspectArgs += "--continue-on-fail"
}
if ($LogModelApi) {
    $inspectArgs += "--log-model-api"
}
if ($LogRefusals) {
    $inspectArgs += "--log-refusals"
}
if ($NoLogRealtime) {
    $inspectArgs += "--no-log-realtime"
}
if ($NoScoreDisplay) {
    $inspectArgs += "--no-score-display"
}

$renderedArgs = $inspectArgs | ForEach-Object {
    if ($_ -match '\s') {
        '"' + $_ + '"'
    }
    else {
        $_
    }
}
Write-Host ("INSPECT_COMMAND=" + ($InspectExe + " " + ($renderedArgs -join " ")))

$inspectExitCode = 0
$previousErrorActionPreference = $ErrorActionPreference
$hadNativeErrorPreference = Test-Path variable:PSNativeCommandUseErrorActionPreference
if ($hadNativeErrorPreference) {
    $previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
}

try {
    $ErrorActionPreference = "Continue"
    if ($hadNativeErrorPreference) {
        $PSNativeCommandUseErrorActionPreference = $false
    }
    & $InspectExe @inspectArgs 2>&1 | Tee-Object -FilePath (Join-Path $logDir "console.txt")
    $inspectExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
    if ($hadNativeErrorPreference) {
        $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
    }
}

if ($inspectExitCode -ne 0) {
    throw "inspect eval failed with exit code $inspectExitCode"
}
