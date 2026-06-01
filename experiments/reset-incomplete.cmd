@echo off
setlocal EnableExtensions
REM Double-click from Explorer, or run from repo root: experiments\reset-incomplete.cmd
REM Prefers .venv-windows\Scripts\python.exe so "uv run" does not try to recreate a broken .venv.
cd /d "%~dp0.."

echo === Preview (dry run) ===
if exist ".venv-windows\Scripts\python.exe" (
  echo Using .venv-windows\Scripts\python.exe
  ".venv-windows\Scripts\python.exe" -m tmrl.tools.experiment_manager reset incomplete --dry-run
) else (
  echo WARNING: .venv-windows not found; using uv run with UV_PROJECT_ENVIRONMENT=.venv-windows
  set "UV_PROJECT_ENVIRONMENT=.venv-windows"
  uv run python -m tmrl.tools.experiment_manager reset incomplete --dry-run
)
if errorlevel 1 exit /b 1

echo.
set /p OK="Type YES to remove failed/planned/running from registry and delete their configs/analysis/logs: "
if /I not "%OK%"=="YES" (
  echo Cancelled.
  exit /b 0
)

if exist ".venv-windows\Scripts\python.exe" (
  ".venv-windows\Scripts\python.exe" -m tmrl.tools.experiment_manager reset incomplete --yes
) else (
  set "UV_PROJECT_ENVIRONMENT=.venv-windows"
  uv run python -m tmrl.tools.experiment_manager reset incomplete --yes
)
if errorlevel 1 exit /b 1
echo Done.
pause
