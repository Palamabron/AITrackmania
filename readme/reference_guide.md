# TMRL quick reference guide

## Quick links
- [Configuration](#configuration)
- [Command line interface](#command-line-interface)
- [Documentation (Python library)](https://tmrl.readthedocs.io/en/latest/)
- [In-repo config notes](../tmrl/config/README.md) (precedence, Hydra groups, schema)

## Configuration

Runtime settings are **not** read from `config.json`. They are built as follows:

1. **Hydra defaults** — YAML under `tmrl/config/defaults/` (see `tmrl/config/defaults/config.yaml` and group files such as `environment/`, `algorithm/`, `model/`, `distributed/`, …).
2. **`TMRL_HYDRA_OVERRIDES`** — Optional. JSON array of Hydra overrides at compose time, e.g. `'["model=vanilla_cnn_actor_critic","algorithm=sac"]'`.
3. **`~/TmrlData/config/local.yaml`** — Deep-merged on top (your machine-specific and experiment overrides). Create this file if it does not exist.
4. **Secrets** — `WANDB_API_KEY` or `WANDB_KEY` → `wandb.api_key`; `TMRL_PASSWORD` → `distributed.password`.
5. **`TMRL_CONFIG_OVERRIDES`** — Optional. JSON object deep-merged last (e.g. quick ablations without editing files).

The result is validated into a single **`MainConfig`** tree (Pydantic, **snake_case** keys). `schema_version` must be at least **`0.6.0`**.

### Useful CLI helpers

- **`python -m tmrl --print-config`** — Print the fully merged config (secrets redacted) as readable YAML and exit.
- **`python -m tmrl --explain-active-config`** — Print which `model.*` fields affect the current algorithm + interface routing and which are ignored, then exit.

### Example `local.yaml` (fragment)

Adjust paths and values for your setup. Comments are allowed in YAML.

```yaml
schema_version: "0.6.0"

run:
  name: my_experiment

wandb:
  project: tmrl
  entity: tmrl
  api_key: ""  # or set WANDB_API_KEY in the environment

distributed:
  localhost_worker: true
  localhost_trainer: true
  public_ip_server: "0.0.0.0"
  server_port: 55555
  password: change_me
  use_tls: false

environment:
  rtgym_interface: TM20LIDAR
  window_width: 640
  window_height: 480
  img_width: 64
  img_height: 64
  img_grayscale: true
  use_images: false
  sleep_time_at_reset: 1.5
  img_hist_len: 4
  rtgym:
    time_step_duration: 0.05
    start_obs_capture: 0.04
    time_step_timeout_factor: 1.0
    act_buf_len: 2
    benchmark: false
    wait_on_done: true
    ep_max_length: 5000
    interface_kwargs:
      save_replays: false
  reward:
    end_of_track_reward: 10.0
    constant_penalty: 0.0
    check_forward: 500
    check_backward: 10
    min_seconds_before_failure: 3.5
    max_stray: 50.0

algorithm:
  name: SAC  # SAC, REDQSAC, TQC, IQN, SDSAC (subject to model/interface compatibility)

training:
  max_epochs: 10000
  rounds_per_epoch: 100
  training_steps_per_round: 200
  memory_size: 1000000
  batch_size: 256
```

For the full field set, see `tmrl/config/schema/` and the default YAML groups. Preset bundles (e.g. IQN-friendly reward shaping) are documented in `tmrl/config/README.md`.

**Note:** The resource zip may still ship a legacy `config.json` under `~/TmrlData/resources`. The Python stack does **not** load it; use `local.yaml` and/or env overrides instead.

---

## Command line interface

The entrypoint is Tyro-based: `python -m tmrl <mode> [options]` (or the `tmrl` console script after install). Exactly **one** mode flag should be used per run.

### General

- **Show where `TmrlData` lives** (and ensure the package can load config):
  ```bash
  python -m tmrl --install
  ```

- **Test / inference only** (standalone worker, no training samples to the server):
  ```bash
  python -m tmrl --test
  ```

- **Record a reward** (TrackMania 2020; follow terminal prompts):
  ```bash
  python -m tmrl --record-reward
  ```

- **Sanity-check the environment** (reward, observations, camera / lidar):
  ```bash
  python -m tmrl --check-env
  ```

- **Distributed training triplet**:
  ```bash
  python -m tmrl --server
  python -m tmrl --trainer
  python -m tmrl --worker
  ```

- **Weights & Biases** — Logging is **on by default** for the trainer. Disable with:
  ```bash
  python -m tmrl --trainer --no-wandb
  ```
  Configure `wandb.*` in `local.yaml` or set `WANDB_API_KEY` / `WANDB_KEY`.

### Advanced

- **Expert rollout** (ignores model updates from the server):
  ```bash
  python -m tmrl --expert
  ```

- **Benchmark** — Set `environment.rtgym.benchmark: true` in `local.yaml` (or merge via `TMRL_CONFIG_OVERRIDES`), then:
  ```bash
  python -m tmrl --benchmark
  ```

- **`--config '<json>'`** — JSON object merged into the **real-time Gym env dict** passed to workers (`CONFIG_DICT` copy), e.g. rt-gym timestep keys:
  ```bash
  python -m tmrl --worker --config '{"time_step_duration": 0.1}'
  ```
  This does **not** replace `local.yaml` for trainer/server settings. Use `TMRL_CONFIG_OVERRIDES` or `local.yaml` for algorithm, model, distributed, etc.

- **`--record-track`** / **`--record-track-side`** — Record track boundary (default side `left`; use `right` when needed).

- **`--record-episode`**, **`--import-player-runs`** — Demo / dataset tooling (see `python -m tmrl --help`).

- **`--wsl-ip`** — Print this machine’s IP (helper for `distributed.public_ip_server` when mixing WSL and Windows workers).

## API reference

Read the [TMRL documentation](https://tmrl.readthedocs.io/en/latest/).
