@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo SETUP REQUIRED: run SETUP.cmd or setup.ps1 in this folder first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -I -u -B "%~dp0tools\kit.py" verify
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -I -u -B "%~dp0src\preflight.py"
if errorlevel 1 goto failed
echo LOCAL CHECK PASSED. You still need a real GUI match to verify gameplay.
pause
exit /b 0
:failed
echo CHECK FAILED. Save the error output above; do not ignore it.
pause
exit /b 1
