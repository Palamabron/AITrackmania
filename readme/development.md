# Developing TrackmaniaRL

## Repository setup

Requirements are Git, Python 3.12 or newer and
[uv](https://docs.astral.sh/uv/). From a clone:

```bash
git clone https://github.com/Palamabron/AITrackmania.git
cd AITrackmania
uv sync --group dev
uv run trackmaniarl --help
```

`uv sync` creates the managed environment and installs this checkout in editable
mode. Do not activate the environment manually and do not install dependencies
with `pip`.

Run the complete local gate before and after a change:

```bash
uv run poe fmt
uv run poe types
uv run poe test
```

For a quick iteration, run one file or test first, then the complete gate:

```bash
uv run pytest tests/integration/runtime/test_core_runtime.py -q
uv run pytest tests/integration/runtime/test_core_runtime.py::test_resolved_run_writes_manifest_and_smoke_checkpoint -q
```

## Repository map

```text
trackmaniarl/              published library
  core/                    stable contracts, data, replay, spec and runtime
  algorithms/              learners and optimization utilities
    value_based/            unified scalar/quantile learner, targets and losses
  models/                  reusable neural-network building blocks
    encoders/               frame-only MLP/CNN encoders
    temporal/               Identity, GRU and portable Mamba cores
    heads/                  scalar, quantile, actor and critic heads
    strategies/             scalar/fixed/random/learned value support
  builtins/                supported component catalogue
  distributed/             actor/learner protocol and durability
  trackmania/              game-only adapter
  observability/           manifests, events and artifacts
  experiments/             evaluation/study orchestration
  project/                 `trackmaniarl init` templates
tests/                     deterministic unit and integration tests
readme/                    user and developer guides
docs/diagrams/             reproducible architecture diagrams and previews
```

`trackmaniarl init <project-name>` creates an application project outside the
library boundary. Generated projects and their artifacts are ignored source-tree
state, not release contents.

## Change workflow

<p align="center">
  <img src="../docs/diagrams/extension-workflow-preview.svg" alt="TrackmaniaRL extension ownership, contract and promotion gates" width="900">
</p>

[Editable diagram](../docs/diagrams/extension-workflow.excalidraw) ·
[local preview](../docs/diagrams/extension-workflow-preview.html)

1. Decide whether the behavior is reusable library code or application-specific
   experiment code. Prefer the generated agent project for the latter.
2. Identify the owning package from the architecture document.
3. Add or update the smallest explicit contract only if existing contracts
   cannot express the behavior.
4. Implement the component without import-time I/O or optional dependency
   requirements in the core path.
5. Add a deterministic test beside the nearest existing contract tests.
6. Run formatting, strict typing and the full test suite.
7. Update user/developer documentation and editable diagram sources whenever a
   public flow, ownership boundary or checkpoint contract changes.
8. Regenerate each changed diagram from its `.spec.json`, validate the
   `.excalidraw` scene and visually inspect its SVG/PNG preview at documentation
   width.
9. For game changes, run `trackmaniarl track check` and the bounded smoke test
   on Windows with Trackmania before release.

## Adding a dependency

Use `uv add <package>` for a required dependency or place the package in the
correct optional-dependency group. Keep optional integrations optional at
import time. Commit both `pyproject.toml` and `uv.lock`, then verify:

```bash
uv lock --check
uv sync --locked --group dev
```

Dependencies used only by generated projects belong in the generated
`pyproject.toml`, not automatically in the library runtime.

## Adding a public component

A public component should have:

- one clear responsibility and full type annotations;
- an existing `trackmaniarl.core` protocol, or a justified minimal new one;
- a stable import path documented in the SDK guide or built-in catalogue;
- deterministic contract and state round-trip tests;
- no hidden global configuration or mandatory network tracker;
- checkpoint state for everything required to resume correctly.

Value-model additions must also declare their representation contract, validate
encoder/temporal/head/strategy dimensions, cover `[B,...]` and `[B,T,...]`, and
prove that selected-action paths do not materialize unnecessary all-action
tensors. A learned value strategy needs optimizer-isolation and finite-difference
gradient tests.

Temporal-core additions need `unroll`, `step`, initial-state and burn-in tests.
Backend substitutions may change kernels but must not silently change model
parameters or architecture fingerprints.

Add experimental algorithms or encoders as importable, opt-in blocks. Compare
one variable at a time against an identical seeded baseline before promoting a
new default.

## Validation levels

| Level | Command | Purpose |
| --- | --- | --- |
| configuration | `uv run trackmaniarl validate run.yaml` | imports components and performs a synthetic update without the game |
| unit/integration | `uv run poe test` | deterministic core, algorithm and fake distributed behavior |
| game connection | `uv run trackmaniarl track check` | verifies one compatible OpenPlanet telemetry frame |
| bounded live gate | `uv run trackmaniarl smoke run.yaml --transitions 100` | real actor/learner path, policy refresh and checkpoint |
| training | `uv run trackmaniarl train run.yaml` | full configured run |

For behavior cloning, `validate` uses the offline-supervised validation hook.
The bounded workflow additionally includes deterministic split tests, exact
`bc-latest.pt` resume and a real `bc-benchmark` closed-loop release gate.

`validate` imports configured Python components, so it is safe for the game but
not a sandbox for untrusted projects.

## Release checklist

Run the quality gate, build both distributions and inspect their contents:

```bash
uv run poe fmt
uv run poe types
uv run poe test
uv build
uv run python scripts/check_distribution.py
```

Then verify a clean installation of the wheel, the generated project, the
Windows Trackmania smoke test and at least one checkpoint resume. Update the
changelog for user-visible behavior and the security policy/audit when a trust
boundary changes.
