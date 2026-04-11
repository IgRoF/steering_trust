<#
.SYNOPSIS
Watch Google batch jobs from the local registry.

.DESCRIPTION
This wrapper polls the batch registry, prints status to the console, and optionally sends Telegram notifications when jobs reach terminal states.
#>
param(
    [int]$PollSeconds = 30,

    [switch]$Once,

    [switch]$NoNotify,

    [switch]$ExitWhenAllTerminal,

    [string]$Registry = ".\outputs\google-batch\registry.json",

    [string]$State = ".\outputs\google-batch\watch_state.json",

    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found at $PythonExe. Create the virtual environment first or pass -PythonExe explicitly."
}

$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

$args = @(
    ".\scripts\watch_google_batch_jobs.py"
    "--registry"
    $Registry
    "--state"
    $State
    "--poll-seconds"
    $PollSeconds
)

if ($Once) { $args += "--once" }
if ($NoNotify) { $args += "--no-notify" }
if ($ExitWhenAllTerminal) { $args += "--exit-when-all-terminal" }

Write-Host ("PYTHON_COMMAND=" + ($PythonExe + " " + ($args -join " ")))
& $PythonExe @args
if ($LASTEXITCODE -ne 0) {
    throw "Google batch monitor failed with exit code $LASTEXITCODE"
}
