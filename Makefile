.PHONY: fmt lint types check test tests install-dev server trainer worker record-episode

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

types:
	uv run mypy tmrl

check: lint types

test:
	uv run pytest

tests:
	uv run pytest tests/ -v

install-dev:
	uv sync --group dev

# Default TMRL server port (see tmrl/config/defaults/distributed/default.yaml).
TMRL_SERVER_PORT ?= 55555

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

# On Windows, Make often runs recipes with cmd.exe even when SHELL is set to /bin/bash (WSL/Git/Cursor).
# Always set UV_PROJECT_ENVIRONMENT via cmd /C so it works regardless of SHELL.
TMRL_WIN_ROOT := $(subst \,/,$(or $(WINDIR),$(SystemRoot)))
TMRL_COMSPEC := $(subst \,/,$(or $(COMSPEC),cmd.exe))
ifneq ($(TMRL_WIN_ROOT),)
TMRL_PWSH_EXE := $(TMRL_WIN_ROOT)/System32/WindowsPowerShell/v1.0/powershell.exe
else
TMRL_PWSH_EXE := powershell.exe
endif
TMRL_KILL_PS1 := $(subst \,/,$(dir $(abspath $(lastword $(MAKEFILE_LIST))))scripts/kill_tcp_port.ps1)

# Free the port before starting the server. Leading '-' ignores errors (port already free, etc.).
ifneq ($(TMRL_IS_WINDOWS),)
kill-server:
	@echo "Checking for zombie processes on port $(TMRL_SERVER_PORT)..."
	-@"$(TMRL_PWSH_EXE)" -NoProfile -ExecutionPolicy Bypass -File "$(TMRL_KILL_PS1)" -Port $(TMRL_SERVER_PORT)
else
kill-server:
	@echo "Checking for zombie processes on port $(TMRL_SERVER_PORT)..."
	-@sh -c 'pids=$$(lsof -ti:$(TMRL_SERVER_PORT) 2>/dev/null); [ -n "$$pids" ] && kill -9 $$pids 2>/dev/null || true'
endif

ifneq ($(TMRL_IS_WINDOWS),)
server: kill-server
	@"$(TMRL_COMSPEC)" /C "set UV_PROJECT_ENVIRONMENT=$(strip $(TMRL_UV_ENV))&& uv run python -m tmrl --server"

trainer:
	@"$(TMRL_COMSPEC)" /C "set UV_PROJECT_ENVIRONMENT=$(strip $(TMRL_UV_ENV))&& uv run python -m tmrl --trainer"

worker:
	@"$(TMRL_COMSPEC)" /C "set UV_PROJECT_ENVIRONMENT=$(strip $(TMRL_UV_ENV))&& uv run python -m tmrl --worker"

record-episode:
	@"$(TMRL_COMSPEC)" /C "set UV_PROJECT_ENVIRONMENT=$(strip $(TMRL_UV_ENV))&& uv run python -m tmrl --record-episode --record-episode-count $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),2)"
else
server: kill-server
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --server

trainer:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --trainer

worker:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --worker

record-episode:
	@UV_PROJECT_ENVIRONMENT=$(TMRL_UV_ENV) uv run python -m tmrl --record-episode --record-episode-count $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),2)
endif

# Allow syntax like: make record-episode 5
%:
	@:
