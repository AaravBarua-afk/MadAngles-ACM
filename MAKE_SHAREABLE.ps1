# Creates a clean ZIP from this folder, leaving your working environment/logs intact.
$ErrorActionPreference = 'Stop'
$v13SharePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $v13SharePython)) { throw 'Run setup.ps1 first.' }
$v13ShareOutput = Join-Path $PSScriptRoot ('V13_participant_kit_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.zip')
& $v13SharePython -I -u -B (Join-Path $PSScriptRoot 'tools\kit.py') archive --output $v13ShareOutput
if ($LASTEXITCODE -ne 0) { throw 'Archive verification failed.' }
