# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands use `uv`. The Makefile auto-detects OS and selects the right venv (`.venv-linux`, `.venv-windows`, or `.venv`).

```bash
make install-dev          # uv sync --group dev
make fmt                  # ruff format + ruff check --fix
make lint                 # ruff check
make types                # mypy tmrl/
make check                # lint + types
make test                 # pytest
make tests                # pytest tests/ -v
uv run pytest tests/test_buffer.py   # single test file
uv run pytest tests/test_buffer.py::test_name  # single test
```

**After making major changes, always run `make fmt` then `make types` and fix any errors before committing.**

**Distributed training** (three terminals):
```bash
make server               # relay server (kills port first)
make trainer              # prints active config, then trains
make worker               # prints active config, then collects rollouts
```

**Config verification:**
```bash
uv run python -m tmrl --print-config          # full merged config
make explain-config                            # which model.* keys are active for current algorithm/interface
```

**Experiment orchestrator:**
```bash
make orchestrator                              # autonomous tuning loop
uv run python -m tmrl.tools.experiment_manager status
uv run python -m tmrl.tools.experiment_manager briefing
```

## Architecture

### Distributed training pipeline

Three separate processes communicate via `tlspyo` (TCP or TLS):

- **Server** (`tmrl/networking/server.py`): central relay. Collects experience from workers, buffers it, forwards to trainer. Broadcasts updated model weights back to workers.
- **RolloutWorker** (`tmrl/networking/worker.py`): runs the current policy in the game environment, sends buffered transitions to server, periodically receives new weights.
- **TrainingOffline** (`tmrl/training_offline/training.py`): pulls samples from server into a replay memory, trains the agent, broadcasts weights back.

### Config system (Hydra + Pydantic, 6-layer precedence)

```
defaults/config.yaml  →  TMRL_HYDRA_OVERRIDES  →  ~/TmrlData/config/local.yaml
  →  env secrets  →  TMRL_CONFIG_OVERRIDES (JSON)  →  Pydantic MainConfig
```

| Layer | File | Purpose |
|---|---|---|
| Composition | `tmrl/config/defaults/` | YAML group defaults (algorithm, model, environment, …) |
| Merge | `tmrl/config/loader.py` | Applies override precedence; produces merged dict |
| Validation | `tmrl/config/schema/` | `MainConfig` Pydantic model; fails fast on bad combos |
| Flat constants | `tmrl/config/constants.py` | Convenience flags (`USE_LIDAR`, `BATCH_SIZE`, …) derived from `MAIN_CONFIG` |
| Runtime objects | `tmrl/config/config_objects.py` | Wires flags → concrete interface/memory/model/trainer classes via side-effect `@register` imports |
| Paths | `tmrl/config/paths.py` | All filesystem paths (`CHECKPOINTS_FOLDER`, `REWARD_PATH`, …) |

`config_objects.py` triggers all `@register` decorators by importing the implementation modules at module load time. Any new algorithm/memory/model must be imported there.

**How to consume config in code:**
- Flat scalars: `import tmrl.config as cfg; cfg.BATCH_SIZE`
- Typed tree: `from tmrl.config import MAIN_CONFIG`
- Runtime objects (trainer, memory, agent): `import tmrl.config.config_objects as cfg_obj`

### Component registry

`tmrl/registry.py` provides `Registry[T]` — a string-keyed decorator registry. Four global instances: `ALGORITHMS`, `INTERFACES`, `MEMORIES`, `MODELS`. Registrations live in `tmrl/custom/custom_algorithms/`, `tmrl/custom/interfaces/`, `tmrl/custom/memories/`, `tmrl/custom/models/`.

### Memory / replay buffer

Abstract base: `tmrl/memory/base.py::Memory`. Concrete implementations under `tmrl/custom/memories/`. Key contract: subclasses implement `append_buffer()` and `__len__()`; the base class handles n-step return windowing, CRC debug, and dataset preloading.

### Algorithms and models

Algorithms in `tmrl/custom/custom_algorithms/`: SAC, REDQSAC, TQC, IQN, SDSAC.

Models in `tmrl/custom/models/` organized by input modality:
- `vector_input/` — MLP and residual MLP actors/critics (boundary LIDAR)
- `image_input/` — vanilla CNN, EfficientNet, IMPALA (vision)
- `hybrid_input/` — Sophy and GNN+EffNet+Sophy (track + telemetry + optional image)
- `discrete_actions/` — IQN Q-network and DQN actor

IQN uses discrete actions (`DQNActor` on workers, `IQNQNetwork` in trainer). All other algorithms use continuous actions. Algorithm–model pairings are validated by Pydantic at startup; invalid combos raise `ValueError` immediately.

### Actor interface

`tmrl/actor.py::ActorModule` — implement this for RolloutWorker to use your policy. Must accept `observation_space` and `action_space` in `__init__`, implement `act()`, and optionally `save()`/`load()`.

### Experiment orchestration

`tmrl/tools/orchestrator.py` runs an autonomous loop: launch experiment → monitor via W&B → stop/continue decision → propose next experiment. State lives in `experiments/registry.jsonl` (gitignored). Configs are versioned YAML overrides under `experiments/configs/`. Manage experiments with `tmrl.tools.experiment_manager`.

Do not open `experiments/analysis/` JSON or `registry.jsonl` unless the user explicitly asks for metrics from those files.

### Key data paths (at runtime)

All runtime data lands under `~/TmrlData/`:
- `~/TmrlData/config/local.yaml` — user overrides (not tracked by git)
- `~/TmrlData/checkpoints/` — saved model weights
- `~/TmrlData/reward/` — recorded reward trajectories (`.pkl`)
- `~/TmrlData/track/` — recorded boundary files (`{map}_left.pkl`, `{map}_right.pkl`)
