# One-time dependency setup only. Does not launch RLBot, a bot or a match.
$ErrorActionPreference = 'Stop'
$v13Root = $PSScriptRoot
& py -3.12 -I -B (Join-Path $v13Root 'tools\kit.py') verify
if ($LASTEXITCODE -ne 0) { throw 'Package verification failed. Use a clean extraction or seal your participant upgrade first.' }
if (-not (Test-Path -LiteralPath (Join-Path $v13Root '.venv\Scripts\python.exe'))) {
    & py -3.12 -m venv (Join-Path $v13Root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
}
$v13Python = Join-Path $v13Root '.venv\Scripts\python.exe'
& $v13Python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'
if ($LASTEXITCODE -ne 0) { throw 'Pinned CPU PyTorch installation failed.' }
& $v13Python -m pip install -r (Join-Path $v13Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Pinned RLBot dependencies installation failed.' }
& $v13Python -I -u -B (Join-Path $v13Root 'src\preflight.py')
if ($LASTEXITCODE -ne 0) { throw 'V13 non-live preflight failed.' }
Write-Host 'Setup complete. Next: RLBot GUI -> Add/Remove -> Add File -> V13 bot.toml.'
Write-Host 'No match launched. GUI gameplay acceptance is still required.'
