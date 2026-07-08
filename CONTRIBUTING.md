# Contributing to TMRL

Thank you for your interest in contributing to `tmrl`. This guide covers the two main
contribution paths: **extending tmrl via its plugin system** (no fork required) and
**contributing directly to the core library** (PR workflow).

---

## Overview — ways to contribute

| Path | When to use |
|------|-------------|
| [Plugin system](#plugin-system) | New algorithm, model, interface, or memory that lives in your own package |
| [Core contributions](#core-contributions) | Bug fixes, performance improvements, documentation, tests |
| [Discussions](https://github.com/trackmania-rl/tmrl/discussions) | Design proposals, questions, sharing results |

---

## Plugin System

`tmrl` exposes four extension registries that third-party packages can populate at
install time via standard Python
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/).
Your package does not need to be part of the `tmrl` repository — just declare the
entry point in your own `pyproject.toml` and `tmrl` will discover it automatically on
the next startup.

The four entry-point groups are:

| Group | Registry | Base class / decorator |
|-------|----------|------------------------|
| `tmrl.algorithms` | `tmrl.registry.ALGORITHMS` | `tmrl.training_offline.training.TrainingAgent` |
| `tmrl.models` | `tmrl.registry.MODELS` | — |
| `tmrl.interfaces` | `tmrl.registry.INTERFACES` | `tmrl.actor.ActorModule` |
| `tmrl.memories` | `tmrl.registry.MEMORIES` | `tmrl.memory.base.Memory` |

### Adding a New Algorithm

1. Inherit from `tmrl.training_offline.training.TrainingAgent` and implement the
   required `train()` method.

2. Decorate your class with `@ALGORITHMS.register`:

   ```python
   # mypackage/my_algorithm.py
   from tmrl.registry import ALGORITHMS
   from tmrl.training_offline.training import TrainingAgent

   @ALGORITHMS.register("MY_ALGO")
   class MyAlgorithm(TrainingAgent):
       def train(self, batch):
           ...
   ```

3. Declare the entry point in your package's `pyproject.toml`:

   ```toml
   [project.entry-points."tmrl.algorithms"]
   MY_ALGO = "mypackage.my_algorithm:MyAlgorithm"
   ```

4. Ship algorithm-specific YAML config defaults alongside your package and point
   `tmrl` at them with `TMRL_EXTRA_CONFIG_PATH`:

   ```bash
   export TMRL_EXTRA_CONFIG_PATH=/path/to/mypackage/config/my_algo_defaults.yaml
   ```

   `TMRL_EXTRA_CONFIG_PATH` is merged after the user's `~/TmrlData/config/local.yaml`
   and before `TMRL_CONFIG_OVERRIDES`, so your defaults can still be overridden by the
   end user.

5. Select your algorithm at runtime:

   ```yaml
   # ~/TmrlData/config/local.yaml
   algorithm:
     name: MY_ALGO
   ```

### Adding a New Model

1. Implement your model (any `torch.nn.Module` subclass) and decorate it:

   ```python
   # mypackage/my_model.py
   from tmrl.registry import MODELS
   import torch.nn as nn

   @MODELS.register("MY_MODEL")
   class MyModel(nn.Module):
       ...
   ```

2. Declare the entry point:

   ```toml
   [project.entry-points."tmrl.models"]
   MY_MODEL = "mypackage.my_model:MyModel"
   ```

3. Select your model via config:

   ```yaml
   model:
     actor_module: MY_MODEL
   ```

### Adding a New Interface

Interfaces define how `tmrl` communicates with the game or robot environment.

1. Inherit from `tmrl.actor.ActorModule` and implement `act()`. Optionally implement
   `save()` and `load()` for checkpoint support.

   ```python
   # mypackage/my_interface.py
   from tmrl.registry import INTERFACES
   from tmrl.actor import ActorModule

   @INTERFACES.register("MY_INTERFACE")
   class MyInterface(ActorModule):
       def __init__(self, observation_space, action_space):
           super().__init__(observation_space, action_space)

       def act(self, obs, test=False):
           ...
   ```

2. Declare the entry point:

   ```toml
   [project.entry-points."tmrl.interfaces"]
   MY_INTERFACE = "mypackage.my_interface:MyInterface"
   ```

3. Select your interface via config:

   ```yaml
   environment:
     rtgym_interface: MY_INTERFACE
   ```

### Adding a New Memory

Replay memories control how experience is stored and sampled.

1. Inherit from `tmrl.memory.base.Memory`. Subclasses must implement:
   - `append_buffer(buffer)` — add a batch of transitions from a worker buffer.
   - `__len__()` — return the current number of stored transitions.

   The base class handles n-step return windowing, CRC debug checks, and dataset
   preloading. Do not override `sample()` unless you have a specific reason to.

   ```python
   # mypackage/my_memory.py
   from tmrl.registry import MEMORIES
   from tmrl.memory.base import Memory

   @MEMORIES.register("MY_MEMORY")
   class MyMemory(Memory):
       def append_buffer(self, buffer):
           ...

       def __len__(self):
           ...
   ```

2. Declare the entry point:

   ```toml
   [project.entry-points."tmrl.memories"]
   MY_MEMORY = "mypackage.my_memory:MyMemory"
   ```

3. Select your memory via config:

   ```yaml
   memory:
     memory_type: MY_MEMORY
   ```

---

## Core Contributions

### PR workflow

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Install the development environment:
   ```bash
   make install-dev
   ```

3. Make your changes. After any significant change, run the full check suite:
   ```bash
   make fmt       # auto-format and fix lint issues
   make check     # ruff check + mypy
   make tests     # pytest -v
   ```

4. Add an entry to `CHANGELOG.md` under `[Unreleased]` in the appropriate section
   (Added / Changed / Fixed / Removed).

5. Open a PR against `main`. The CI workflow runs `lint`, `types`, and `tests`
   automatically on every push; all three jobs must pass before merge.

6. Add your name to the contributors list in `README.md` with a short caption.

### Commit style

Use short imperative subject lines (`fix: ...`, `feat: ...`, `docs: ...`,
`refactor: ...`, `test: ...`). Reference issues where applicable.

---

## Development Commands

All commands use [`uv`](https://github.com/astral-sh/uv). The Makefile auto-detects
the OS and selects the correct virtual environment (`.venv-linux`, `.venv-windows`, or
`.venv`).

| Target | Description |
|--------|-------------|
| `make install-dev` | `uv sync --group dev` — install all dev dependencies |
| `make fmt` | `ruff format` + `ruff check --fix` — auto-format and fix lint |
| `make lint` | `ruff check` — lint without auto-fix |
| `make types` | `mypy tmrl/` — type-check the library |
| `make check` | `lint` + `types` |
| `make test` | `pytest` — run tests |
| `make tests` | `pytest tests/ -v` — run tests with verbose output |
| `make server` | Start the relay server (kills the port first) |
| `make trainer` | Print active config, then start the trainer |
| `make worker` | Print active config, then start the rollout worker |
| `make record-episode` | Record episodes (optional count: `make record-episode 5`) |
| `make record-reward` | Drive a lap to record a reward trajectory |
| `make record-track-boundaries` | Interactive left/right boundary recording |
| `make extend-boundaries` | Append straight extensions to boundary files |
| `make build-centerline-reward` | Build centerline reward from boundary pickles |
| `make interpolate-reward` | Arc-length upsample the reward polyline |
| `make plot-boundaries` | Visualise track boundary points |
| `make plot-reward` | Visualise reward trajectory points |
| `make check-env` | Verify environment setup before training |
| `make explain-config` | Print which config keys are active for the current algorithm/interface |
| `make import-player-runs` | Bootstrap the replay buffer from recorded human demonstrations |
| `make orchestrator` | Start the autonomous experiment-tuning loop |
| `make kill-all` | Kill all running tmrl processes |
| `make kill-all-python` | Kill every Python/uv process on the machine (nuclear option) |

Single test file or test function:

```bash
uv run pytest tests/test_buffer.py
uv run pytest tests/test_buffer.py::test_name
```

---

## Code Style

- **Formatter / linter**: [Ruff](https://docs.astral.sh/ruff/) (`make fmt` /
  `make lint`). Configuration is in `pyproject.toml` under `[tool.ruff]`.
- **Type checker**: [mypy](https://mypy.readthedocs.io/) (`make types`). Strict
  return-type and unused-config warnings are enabled.
- **Logging**: use [loguru](https://loguru.readthedocs.io/) (`from loguru import
  logger`). Do not use the standard `logging` module or bare `print()` statements for
  library code.
- **No hidden numeric defaults**: magic numbers must be named constants or config keys
  with explicit documentation. Do not hard-code learning rates, buffer sizes, or
  architectural dimensions inline.
- **Docstrings**: NumPy style for public functions and classes.
- **Tests**: place new tests under `tests/`. Name files `test_*.py` and functions
  `test_*`. Use `pytest` fixtures; avoid global state between tests.
