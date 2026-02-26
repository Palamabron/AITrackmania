# Player Runs to Replay Buffer

This workflow lets you reuse previously recorded runs to reduce training warmup.

## 1) Record runs

Record one or more episodes into standalone `.pkl` files. **You** drive (human control); the model does not. Use a **physical gamepad** connected to the game. Press **Enter** to start each episode, **Del** to end it early.

`python -m tmrl --record-episode --record-episode-count 5`

Optional flags:
- `--record-episode-output-dir "/path/to/player_runs"`
- `--record-episode-max-samples 2000`

By default, files are written under `~/TmrlData/player_runs`.

## 2) Import runs into replay dataset

Import run files into `data.pkl` using the currently configured memory class:

`python -m tmrl --import-player-runs --player-runs-paths "/path/run1.pkl,/path/run2.pkl"`

Optional flags:
- `--player-runs-overwrite` to replace existing dataset content
- `--player-runs-max-samples 300000` to keep only the newest raw samples
- `--player-runs-dry-run` to validate compatibility without writing

Imported data is saved to `cfg.DATASET_PATH/data.pkl` and loaded automatically by replay memory at trainer startup.

## 3) Enable optional online injection during training

Add this section to `TmrlData/config/config.json`:

```json
"PLAYER_RUNS": {
  "ONLINE_INJECTION": true,
  "SOURCE_PATH": "/home/<user>/TmrlData/player_runs",
  "CONSUME_ON_READ": true,
  "MAX_FILES_PER_UPDATE": 1,
  "DEMO_INJECTION_REPEAT": 1,
  "DEMO_SAMPLING_WEIGHT": 1.0
}
```

When enabled, the trainer periodically scans `SOURCE_PATH` and injects unseen run files into replay memory.
`DEMO_INJECTION_REPEAT` duplicates imported demo buffers in-memory to increase early influence.
`DEMO_SAMPLING_WEIGHT` increases the chance of selecting demo episodes during replay sampling (R2D2 memories).

**Important – `SOURCE_PATH`:** The path is read on the machine where the **trainer** runs. Use a path that exists there:
- **Trainer on WSL:** e.g. `"/mnt/c/Users/<WindowsUser>/TmrlData/player_runs"` (Windows `C:\Users\...` is under `/mnt/c/` in WSL).
- **Trainer on Windows:** e.g. `"C:\\Users\\<You>\\TmrlData\\player_runs"`.
If the path is wrong, the trainer will log `Player runs SOURCE_PATH does not exist` and keep waiting for samples. At startup it also logs `Player runs online injection: SOURCE_PATH=... (exists=true/false)`.

**`CONSUME_ON_READ: true`:** After a file is injected it is renamed to `.pkl.imported`. On a fresh trainer start, only `.pkl` files are read; already-imported files are skipped. To reuse the same recordings after a restart, either add new recordings or set `CONSUME_ON_READ: false` (then the same files are re-injected every time they are polled).

## Compatibility and safety checks

- A run sample must follow `(act, obs, rew, terminated, truncated, info)`.
- `terminated` and `truncated` must be booleans and `info` must be a dictionary.
- Import rejects schema mismatches to avoid silent replay corruption.

## Suggested usage ratio

Keep player-run data as a bootstrap, not the full training signal (for example 5-30% depending on map diversity), and continue mixing with fresh worker samples.
