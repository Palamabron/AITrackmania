# Contributing to TMRL

TMRL is an SDK, not a collection of hard-coded training modes. New behavior belongs behind a small explicit contract in `tmrl.core`, a Trackmania adapter in `tmrl.trackmania`, or an optional observability/experiment adapter.

Before opening a change:

```bash
uv run poe fmt
uv run poe types
uv run poe test
```

Keep the public CLI portable: `tmrl init` and `tmrl validate` must behave the same on Windows and Linux. Avoid reading local files, environment secrets or optional packages during `import tmrl`. Validate configuration at the boundary; do not introduce Pydantic models into rollout or sampling hot paths.

Each built-in extension needs contract coverage and a deterministic synthetic test. Game-dependent smoke tests remain optional.
