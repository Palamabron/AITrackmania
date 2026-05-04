.PHONY: fmt lint types check test tests install-dev server trainer worker record-episode \
	record-reward record-track-boundaries extend-boundaries build-centerline-reward \
	interpolate-reward plot-boundaries check-env explain-config import-player-runs

fmt:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

lint:
	.venv/bin/ruff check .

types:
	.venv/bin/mypy tmrl

check: lint types

test:
	uv run pytest

tests:
	uv run pytest tests/ -v

install-dev:
	uv sync --group dev

# Default TMRL server port (see tmrl/config/defaults/distributed/default.yaml).
TMRL_SERVER_PORT ?= 55555

# --- Track geometry (reward + boundaries; paths default from TmrlData config when empty) ---
# Reward pickle for interpolate-reward (empty → tmrl.config.REWARD_PATH).
REWARD_INPUT ?=
# Arc-length upsampling factor for interpolate-reward (scripts/interpolate_reward_trajectory.py).
INTERP_FACTOR ?= 10
# Straight extension length (m) for extend-boundaries (tmrl/tools/record_track.py extend).
EXTEND_METERS ?= 100
# One or two boundary .pkl paths for extend-boundaries (space-separated).
BOUNDARY_PKLS ?=
# Extra CLI args for build-centerline-reward / plot scripts (e.g. --debug-plot, --html-out file.html).
CENTERLINE_ARGS ?=
PLOT_ARGS ?=
# Comma-separated .pkl paths for import-player-runs (tmrl --import-player-runs).
PLAYER_RUNS_PATHS ?=

# OS autodetection: OS=Windows_NT, COMSPEC, or uname for Git Bash/MSYS/WSL/Linux.
# Do not run `uname` in $(shell) on Windows: Make may use cmd.exe, where `2>/dev/null ||` fails and prints "path not found".
ifneq ($(filter Windows_NT,$(OS)),)
UNAME_S := MSWin
else
UNAME_S := $(shell uname -s 2>/dev/null || echo unknown)
endif
COMSPEC_NORM := $(subst \,/,$(COMSPEC))
# findstring is case-sensitive; Windows paths may be ...\CMD.EXE
TMRL_CMD_IN_PATH = $(findstring cmd,$(1))$(findstring CMD,$(1))
TMRL_IS_WINDOWS := $(strip $(filter Windows_NT,$(OS)) $(call TMRL_CMD_IN_PATH,$(COMSPEC_NORM)) $(findstring MINGW,$(UNAME_S)) $(findstring MSYS,$(UNAME_S)) $(findstring CYGWIN,$(UNAME_S)))

# uv venv: prefer an existing dir for this OS; POSIX env prefix works in sh, Git Bash, MSYS, WSL, Linux, macOS.
ifndef TMRL_UV_ENV
ifneq ($(TMRL_IS_WINDOWS),)
ifneq ($(wildcard .venv-windows),)
TMRL_UV_ENV := .venv-windows
else
ifneq ($(wildcard .venv),)
TMRL_UV_ENV := .venv
else
TMRL_UV_ENV := .venv-windows
endif
endif
else
ifneq ($(wildcard .venv-linux),)
TMRL_UV_ENV := .venv-linux
else
ifneq ($(wildcard .venv),)
TMRL_UV_ENV := .venv
else
TMRL_UV_ENV := .venv
endif
endif
endif
endif

# On Windows, Make normally uses cmd.exe to run recipe lines. Ctrl+C in cmd.exe triggers the
# "Terminate batch job (Y/N)?" prompt. Override SHELL to PowerShell so cmd.exe is never involved.
TMRL_WIN_ROOT := $(subst \,/,$(or $(WINDIR),$(SystemRoot)))
ifneq ($(TMRL_WIN_ROOT),)
TMRL_PWSH_EXE := $(TMRL_WIN_ROOT)/System32/WindowsPowerShell/v1.0/powershell.exe
else
TMRL_PWSH_EXE := powershell.exe
endif
TMRL_KILL_PS1 := $(subst \,/,$(dir $(abspath $(lastword $(MAKEFILE_LIST))))scripts/platform/kill_tcp_port.ps1)

ifneq ($(TMRL_IS_WINDOWS),)
SHELL := $(TMRL_PWSH_EXE)
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
endif

# Free the port before starting the server. Leading '-' ignores errors (port already free, etc.).
ifneq ($(TMRL_IS_WINDOWS),)
kill-server:
	@Write-Host "Checking for zombie processes on port $(TMRL_SERVER_PORT)..."; & "$(TMRL_PWSH_EXE)" -NoProfile -ExecutionPolicy Bypass -File "$(TMRL_KILL_PS1)" -Port $(TMRL_SERVER_PORT)
else
kill-server:
	@echo "Checking for zombie processes on port $(TMRL_SERVER_PORT)..."
	-@sh -c 'pids=$$(lsof -ti:$(TMRL_SERVER_PORT) 2>/dev/null); [ -n "$$pids" ] && kill -9 $$pids 2>/dev/null || true'
endif

ifneq ($(TMRL_IS_WINDOWS),)
server: kill-server
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --server

trainer:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --explain-active-config; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }; uv run python -m tmrl --trainer

worker:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --explain-active-config; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }; uv run python -m tmrl --worker

record-episode:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --record-episode --record-episode-count $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),2)

record-reward:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --record-reward

# Interactive: choose left/right boundary; spline interpolation runs on lap finish (see tmrl/tools/record_track.py).
record-track-boundaries:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python tmrl/tools/record_track.py

extend-boundaries:
	@if ('$(strip $(BOUNDARY_PKLS))' -eq '') { Write-Error 'Set BOUNDARY_PKLS to one or more track_*_boundary.pkl paths (space-separated). For parallel L/R use exactly two.'; exit 1 }; $$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python tmrl/tools/record_track.py extend $(BOUNDARY_PKLS) --meters $(EXTEND_METERS)

build-centerline-reward:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python scripts/build_centerline_reward.py $(CENTERLINE_ARGS)

interpolate-reward:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; $$in = '$(REWARD_INPUT)'; if ($$in -eq '') { $$in = (uv run python -c "import tmrl.config as c; print(c.REWARD_PATH)") }; uv run python scripts/interpolate_reward_trajectory.py --input $$in --factor $(INTERP_FACTOR)

plot-boundaries:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python scripts/plotTrackPoints.py $(PLOT_ARGS)

check-env:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --check-env

explain-config:
	@$$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --explain-active-config

import-player-runs:
	@if ('$(strip $(PLAYER_RUNS_PATHS))' -eq '') { Write-Error 'Set PLAYER_RUNS_PATHS to a comma-separated list of player-run .pkl files'; exit 1 }; $$env:UV_PROJECT_ENVIRONMENT = '$(strip $(TMRL_UV_ENV))'; uv run python -m tmrl --import-player-runs --player-runs-paths $(PLAYER_RUNS_PATHS)
else
server: kill-server
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --server

trainer:
	@export UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV); uv run python -m tmrl --explain-active-config && uv run python -m tmrl --trainer

worker:
	@export UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV); uv run python -m tmrl --explain-active-config && uv run python -m tmrl --worker

record-episode:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --record-episode --record-episode-count $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),2)

record-reward:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --record-reward

record-track-boundaries:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python tmrl/tools/record_track.py

extend-boundaries:
	@test -n "$(strip $(BOUNDARY_PKLS))" || (echo "Set BOUNDARY_PKLS to one or more track_*_boundary.pkl paths (space-separated). For parallel L/R use exactly two." >&2; exit 1)
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python tmrl/tools/record_track.py extend $(BOUNDARY_PKLS) --meters $(EXTEND_METERS)

build-centerline-reward:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python scripts/build_centerline_reward.py $(CENTERLINE_ARGS)

interpolate-reward:
	@in="$(REWARD_INPUT)"; \
	if [ -z "$$in" ]; then in=$$(UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -c "import tmrl.config as c; print(c.REWARD_PATH)"); fi; \
	UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python scripts/interpolate_reward_trajectory.py --input "$$in" --factor $(INTERP_FACTOR)

plot-boundaries:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python scripts/plotTrackPoints.py $(PLOT_ARGS)

check-env:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --check-env

explain-config:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --explain-active-config

import-player-runs:
	@test -n "$(strip $(PLAYER_RUNS_PATHS))" || (echo "Set PLAYER_RUNS_PATHS to a comma-separated list of player-run .pkl files" >&2; exit 1)
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --import-player-runs --player-runs-paths $(PLAYER_RUNS_PATHS)
endif

# Allow syntax like: make record-episode 5
%:
	@:
