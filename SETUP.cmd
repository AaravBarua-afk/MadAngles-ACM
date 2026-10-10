@echo off
setlocal
cd /d "%~dp0"
echo Setting up THIS extracted V13 folder. Wait for Setup complete before opening a match.
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo SETUP FAILED. Save the output above and contact the organiser. Do not start a match.
) else (
  echo SETUP PASSED. Open the official RLBot launcher and add this folder's bot.toml.
)
pause
