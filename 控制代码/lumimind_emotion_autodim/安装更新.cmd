@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_update.ps1" %*
if errorlevel 1 (
  echo Installation failed. Read the error above. Do not force overwrite different agent code.
) else (
  echo Installation complete. Read the instructions before starting the agent.
)
pause
