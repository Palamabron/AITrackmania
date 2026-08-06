# TrackMania IQN + lidar release workflow

Install the game integration only on a machine that has TrackMania, OpenPlanet,
the compatible telemetry plugin, and a virtual gamepad driver:

```bash
uv sync
uv run tmrl init --template trackmania my-agent
cd my-agent
uv sync
uv run tmrl validate run.yaml
uv run tmrl smoke run.yaml
uv run tmrl train run.yaml
```

Generated TrackMania agents select the project's tested PyTorch CUDA runtime by
default on Windows and Linux; a newer NVIDIA driver stays compatible. macOS
falls back to its normal PyPI/MPS Torch wheel. ROCm hosts require the matching
AMD Torch build.

The generated `.npz` is a structural placeholder only. Before training, record
the two map boundaries by hand and build a UID-bound asset. Do not reuse an
asset from another map:

```bash
uv run tmrl track record-boundary left assets/tmrl-test-left.npy
uv run tmrl track record-boundary right assets/tmrl-test-right.npy
uv run tmrl track build-geometry assets/tmrl-test.geometry.npz \
  --left assets/tmrl-test-left.npy --right assets/tmrl-test-right.npy \
  --map-uid <tmrl-test-map-uid> --map-path maps/tmrl-test.Map.Gbx
```

Set that same UID in both `feature_pipeline.kwargs.expected_map_uid` and
`evaluation.maps[].expected_map_uid`. Before a live evaluation, load that local
`.Map.Gbx` manually in TrackMania. The documented OpenPlanet API exposes the
active map UID but not a safe API to load an arbitrary local map. The bundled
plugin's second local command port (default `9001`) therefore verifies the
already loaded UID before every episode, then confirms an active player after
the controller reset using protocol version `2`; a timeout, disconnect or UID
mismatch aborts the run.

The default baseline is a 78-action dueling IQN (`13` steering levels × `2`
gas levels × continuous brake, full brake, or brake tap), not TQC. Its
observation is a fixed 20-feature projection of the documented 33-field
`TMRL_GrabData` packet, plus 15 left + 15 right car-local boundary samples.
The local frame comes from `api.Position` and `vis.Dir`; it does not require
aim-yaw telemetry. TQC remains an optional example only.

Every run writes `manifest.json`, versioned `events.jsonl`, compressed episode
artifacts, checkpoints and study records. Resume a stopped run with:

```bash
uv run tmrl resume run.yaml artifacts/<run-id>/checkpoints/distributed-update-XXXXXXXX.pt
```

`tmrl smoke` is the required Windows preflight. It collects a bounded number of
real actions, completes at least one update, verifies a live policy refresh,
and restores the produced checkpoint. It will operate the virtual gamepad:

```bash
uv run tmrl smoke run.yaml --transitions 100
```

The release benchmark is deterministic only in the sense that it repeats the
same local map and assets. It does not claim game-engine seed control. It runs
exactly 20 `tmrl-test` trials, writes `evaluation.json` with per-trial status,
latency/FPS and map UID, and passes only with at least 18 finishes, median
completed time below 37 seconds, and no telemetry/controller errors:

```bash
uv run tmrl benchmark run.yaml artifacts/trackmania-iqn-lidar/checkpoints/distributed-update-XXXXXXXX.pt
```

The remaining manual release gate is a four-hour Windows soak on the real game,
with periodic checkpoints and at least one successful `tmrl resume`. A failed
benchmark or soak blocks release.
