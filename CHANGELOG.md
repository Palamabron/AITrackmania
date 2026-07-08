# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Plugin system: third-party packages can register algorithms, models, interfaces, and
  memories via `pyproject.toml` entry points (`tmrl.algorithms`, `tmrl.models`,
  `tmrl.interfaces`, `tmrl.memories`).
- `TMRL_EXTRA_CONFIG_PATH` environment variable: points to an additional YAML file that
  is merged into the config after the user's `local.yaml`, enabling plugin-shipped
  config defaults without touching `~/TmrlData/`.
- `py.typed` marker file — `tmrl` now declares itself as a typed package (PEP 561).
- Continuous-integration workflow (`.github/workflows/ci.yml`): lint, type-check, and
  test jobs run on every push and pull request.
- `CONTRIBUTING.md`: contributor guide covering the plugin system, PR workflow, code
  style, and all Makefile targets.
- Stable public API surface documented in `tmrl/__init__.py`; symbols are now part of
  the supported contract and will follow semver.

### Changed
- `google-genai` moved from core dependencies to optional extras (`pip install
  tmrl[orchestrator]`); the experiment orchestrator still works as before when the
  extra is installed, but vanilla installations no longer pull in the Google AI SDK.
- Namespace cleanup: internal helpers moved to `tmrl._internal`; previously accessible
  but undocumented private names are now gated behind `__all__`.

---

## [0.8.0] - 2025-07-01

### Added
- **IQN** (Implicit Quantile Networks) algorithm for discrete action spaces:
  - Double DQN-style action selection (online net selects, target net evaluates).
  - Dueling network architecture in `IQNQNetwork`.
  - Munchausen RL regularisation (`munchausen_enabled`, `munchausen_tau`,
    `munchausen_alpha` config keys).
  - NoisyNet exploration layers as an alternative to epsilon-greedy.
  - Demo-guided learning via a behavioral cloning loss (`bc_lambda`).
- **SimbaV2** residual-MLP backbone for vector-observation actors and critics.
- **Gradient stabilizer**: optional per-layer gradient clipping and norm logging via
  `algorithm.grad_clip_norm` and `algorithm.log_grad_norms`.
- **LR warmup + cosine annealing**: `algorithm.lr_warmup_steps` and
  `algorithm.lr_schedule` config keys, compatible with all continuous-action algorithms.
- **R2D2 memory**: recurrent replay buffer that stores full episode sequences and
  samples contiguous subsequences for RNN training.
- **Experiment orchestrator** (`tmrl/tools/orchestrator.py`): autonomous loop that
  launches experiments, monitors them via Weights & Biases, and proposes the next
  hyperparameter configuration. State is tracked in `experiments/registry.jsonl`.
- **Player-run injection** (`--import-player-runs` / `make import-player-runs`):
  bootstrap the replay buffer from recorded human demonstrations.
- **Pydantic v2 + Hydra config system**: six-layer config precedence
  (YAML defaults → `TMRL_HYDRA_OVERRIDES` → `local.yaml` → env secrets →
  `TMRL_CONFIG_OVERRIDES` JSON → Pydantic `MainConfig`); invalid option combinations
  raise `ValueError` at startup rather than silently producing wrong behaviour.
- `make explain-config` / `--explain-active-config`: prints which `model.*` config keys
  are active or ignored for the current algorithm + interface combination.
- `make orchestrator`, `make kill-all`, `make kill-all-python` Makefile targets.
- `tmrl.tools.experiment_manager` CLI for inspecting orchestrator state.

### Changed
- Config loading moved from ad-hoc `constants.py` globals to the Hydra + Pydantic
  pipeline; `tmrl/config/constants.py` now derives flat scalars from `MAIN_CONFIG`
  rather than defining them independently.
- IQN requires `memory.memory_type: generic` when `algorithm.n_steps > 1`.
- Algorithm–model pairings validated at startup; previously invalid combos would fail
  silently at training time.

### Fixed
- N-step return accumulation was incorrect when the episode ended mid-window; the
  generic memory now truncates the window and bootstraps correctly.
- Reward-shaping recovery terms were applied even when the car had not made forward
  progress; the crash penalty and drift reward now gate on `reward_progress > 0`.
- Pipeline hardening: workers no longer crash on a malformed weight payload; they log
  the error and continue with the previous weights.

---

## [0.7.x]

This release series introduced the modular registry system, TQC and SDSAC algorithms,
the EfficientNet and IMPALA vision backbones, TLS support via `tlspyo`, and the
Real-Time Gym (`rtgym`) integration.

For a full history see `git log --oneline v0.7.0..v0.8.0` or the
[GitHub release page](https://github.com/trackmania-rl/tmrl/releases/tag/v0.7.0).

---

## Comparison links

[Unreleased]: https://github.com/trackmania-rl/tmrl/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/trackmania-rl/tmrl/compare/v0.7.0...v0.8.0
[0.7.x]: https://github.com/trackmania-rl/tmrl/releases/tag/v0.7.0
