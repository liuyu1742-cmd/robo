@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Project Python not found: %CD%\.venv\Scripts\python.exe
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "detailed_action_random_visual_test.py" --device auto
if errorlevel 1 pause
endlocal
