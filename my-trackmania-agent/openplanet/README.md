# OpenPlanet adapter

Copy `TMRL_GrabData_IQN.as` to `%USERPROFILE%\OpenplanetNext\Scripts`, reload
OpenPlanet and enable the script. The plugin is intentionally a plain `.as`
script; it does not need an `info.toml` or a plugin-manager entry.

It exposes two localhost services:

- `127.0.0.1:9000` — the 33-field float32 telemetry stream used by the Python actor;
- `127.0.0.1:9001` — the map/session JSONL readiness channel.

Verify the telemetry before recording or training:

```powershell
uv run tmrl track check
```

The plugin cannot safely load an arbitrary `.Map.Gbx` itself. Put the map at
`maps/tmrl-test.Map.Gbx`, open it manually in TrackMania, and start/restart a run
before recording boundaries or running the smoke test. The expected map UID in
`run.yaml` and in the geometry asset must match the loaded map.

## Build the geometry asset

Drive the complete map once for each side. Recording waits for movement and
finishes on the game finish state; it is not a fixed-duration capture:

```powershell
uv run poe record-left
uv run poe record-right
uv run poe build-geometry
```

The Poe tasks write `assets/tmrl-test-left.npy`,
`assets/tmrl-test-right.npy` and `assets/tmrl-test.geometry.npz`. If a recording says
that no run is active, restart the map and begin from the start line.

## Async training

With TrackMania and the plugin running, validate the connection and run a short
live check:

```powershell
uv run tmrl track check
uv run tmrl smoke run.yaml --transitions 100
```

Normal local training starts the learner and actor in separate Windows
`spawn` processes and prints the W&B URL:

```powershell
uv run tmrl train run.yaml
```

For a remote actor, set the same `TMRL_DISTRIBUTED_TOKEN` in `.env` on both
machines, run the learner on the training computer, and connect the game
computer with a stable actor ID:

```powershell
# learner machine
uv run tmrl learner run.yaml --bind 0.0.0.0:8787

# game machine
uv run tmrl actor run.yaml --connect LEARNER_IP:8787 --actor-id PC-1
```
