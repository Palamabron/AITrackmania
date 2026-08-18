# Contributing to TrackmaniaRL

TrackmaniaRL is an SDK, not a collection of hard-coded training modes. New behavior belongs behind a small explicit contract in `trackmaniarl.core`, a Trackmania adapter in `trackmaniarl.trackmania`, or an optional observability/experiment adapter.

Before opening a change:

```bash
uv run poe fmt
uv run poe types
uv run poe test
```

Keep the public CLI portable: `trackmaniarl init` and `trackmaniarl validate` must behave the same on Windows and Linux. Avoid reading local files, environment secrets or optional packages during `import trackmaniarl`. Validate configuration at the boundary; do not introduce Pydantic models into rollout or sampling hot paths.

Each built-in extension needs contract coverage and a deterministic synthetic test. Game-dependent smoke tests remain optional.
