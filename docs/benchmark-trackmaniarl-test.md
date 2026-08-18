# IQN + lidar benchmark: `trackmaniarl-test`, version 1

This optional smoke benchmark is the release performance gate for the bundled
IQN + lidar baseline. It is not part of CPU CI because it requires a configured
TrackMania installation and the `trackmaniarl-test` map.

- Run deterministic policy evaluation (`argmax`, no epsilon exploration) on
  the fixed, manually loaded local map. The game engine seed is not controlled
  or reported.
- Use the local-frame, arc-length-sampled lidar pipeline with finite-value masks.
- Record every trial's finish time, crash state, action latency, throughput,
  map UID, protocol version and checkpoint in `evaluation.json`.
- Acceptance: exactly 20 `trackmaniarl-test` trials, at least 18 completed runs, median
  completed time below 37.0 seconds, and no telemetry or controller errors.
  A failed benchmark blocks release; CPU-only CI does not replace this gate.
