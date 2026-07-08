@echo off
setlocal EnableExtensions
REM Wipes registry + all experiments/configs, experiments/analysis, experiments/logs.
REM Prefers .venv-windows\Scripts\python.exe so "uv run" does not try to recreate a broken .venv.
cd /d "%~dp0.."

echo === Preview (dry run) ===
if exist ".venv-windows\Scripts\python.exe" (
  echo Using .venv-windows\Scripts\python.exe
  ".venv-windows\Scripts\python.exe" -m tmrl.tools.experiment_manager reset all --dry-run
  if errorlevel 1 exit /b 1
) else (
  echo WARNING: .venv-windows not found; using uv run with UV_PROJECT_ENVIRONMENT=.venv-windows
  set "UV_PROJECT_ENVIRONMENT=.venv-windows"
  uv run python -m tmrl.tools.experiment_manager reset all --dry-run
  if errorlevel 1 exit /b 1
)

echo.
set /p OK="Type DELETE ALL to empty registry and delete ALL experiment configs, analysis, and logs: "
if /I not "%OK%"=="DELETE ALL" (
  echo Cancelled.
  exit /b 0
)

if exist ".venv-windows\Scripts\python.exe" (
  ".venv-windows\Scripts\python.exe" -m tmrl.tools.experiment_manager reset all --yes
) else (
  set "UV_PROJECT_ENVIRONMENT=.venv-windows"
  uv run python -m tmrl.tools.experiment_manager reset all --yes
)
if errorlevel 1 exit /b 1
echo Done. Restore decisions.md from git if you also need a clean decisions log.
pause
