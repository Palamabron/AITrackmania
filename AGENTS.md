# TrackmaniaRL development guide

## Commands

```bash
uv sync --group dev
uv run poe fmt
uv run poe types
uv run poe test
uv run trackmaniarl init my-trackmania-agent
uv run trackmaniarl validate run.yaml
uv run trackmaniarl track check
uv run trackmaniarl smoke run.yaml --transitions 100
uv run trackmaniarl train run.yaml
```

These commands deliberately use `uv` without platform-specific virtualenv or shell branches. They must work unchanged on Windows, Linux, WSL and CI.

## Architecture

The public flow is `RunSpec 2.0 -> actor/spool -> authenticated WAL/replay -> learner -> checkpoint/policy snapshot`. `trackmaniarl.models` composes frame-only encoders, temporal cores, heads and value strategies; `trackmaniarl.algorithms.value_based` trains scalar Q, QR-DQN, IQN and FQF through one learner. `trackmaniarl.core` owns contracts, data, replay and runtime; `trackmaniarl.distributed` owns transport and durability; `trackmaniarl.trackmania` owns the game adapter. Behavior cloning uses the offline-supervised validation lifecycle and shares encoders/temporal cores without entering the RL WAL/replay path.

`trackmaniarl train` uses Windows-safe multiprocessing `spawn` and starts one local
actor by default. Remote deployments use `trackmaniarl learner --bind` and
`trackmaniarl actor --connect`; all participants must use the same run fingerprint,
map UID, geometry hash, feature/action contract and `TRACKMANIARL_DISTRIBUTED_TOKEN`.
Rollouts are flushed every 128 transitions or 2 seconds, policy snapshots are
published at most every 5 seconds, and actor policy state is transferred with
safetensors rather than pickle. Actor exploration profiles are assigned by
stable actor ID; epsilon is never overwritten globally on the learner.

User components are loaded by explicit `module:attribute` paths from an installable local project. Configuration is Pydantic at the CLI/runtime boundary. Transitions and batches are slot dataclasses and PyTrees in hot paths. Do not add global configuration, import-time side effects, feature flags or a mandatory external tracker.

All built-ins and generated user components need deterministic contract tests. TrackMania is exercised in the bounded live smoke test; unit tests use fake actors and a slow learner to verify asynchronous behavior without the game.

## Experimental Modules

Keep experimental model and optimization components opt-in. Add them as reusable, importable blocks first; wire them into a RunSpec only for a named experiment after its baseline is recorded.

- `trackmaniarl.models.SimbaV2Backbone` is an experimental encoder. After each optimizer step, call `project_hyperspherical_weights(model)` and compare it against the identical baseline model with the same seed, replay, update budget, and evaluation suite.
- `trackmaniarl.algorithms.AdaptiveGradientClipper` runs after AMP gradient unscaling and before `optimizer.step()`. Persist its `state_dict()` in the learner checkpoint so resume behavior preserves its EMA and warmup state.
- Change one experimental variable at a time. Record the exact component arguments, seed, training budget, return, finish rate, pace, gradient metrics, and failure modes in the experiment artifacts.
- Promote an experimental component to a built-in default only after deterministic unit coverage and at least one bounded live smoke comparison show a measurable benefit without a regression in resume or distributed behavior.

## Python Code Rules

### Tooling

- Package manager: `uv` only. Never `pip`, `poetry`, or `conda`. Use `uv add`, `uv sync`, `uv run <tool>`.
- Format and lint: `ruff` only. Use `uv run ruff format` and `uv run ruff check --fix`. Do not use `black`, `flake8`, `isort`, or `pylint`.
- Type checker: `pyright` or `mypy` in strict mode.
- Test runner: `pytest` via `uv run pytest`.

### Style

- Add type annotations to all function signatures. Do not leave bare `def foo(x)`.
- Prefer `str | None` over `Optional[str]`. Use modern Python syntax such as `match`, `tomllib`, and `Self` where appropriate.
- Prefer dataclasses or Pydantic models over plain dicts for structured data.
- Never use mutable default arguments.
- Use absolute imports only. Group imports as stdlib, third-party, then local.

### Functions and structure

- Keep functions small, ideally under 20 lines, with one job and one level of abstraction.
- Prefer 0 to 2 arguments. Three is the maximum. Avoid flag arguments and hidden side effects.
- Use intention-revealing names. Classes should be nouns, methods verbs.
- Do not return `None` when an empty collection or an exception is better.

### Errors

- Raise specific exceptions with context, not error codes or flags.
- Never swallow exceptions silently.

### Quality

- Follow `DRY`, `YAGNI`, and `KISS`. Delete dead code instead of commenting it out.
- Prefer self-documenting code. Comments should explain why, not what.
- Write fast, independent tests with one concept per test and an Arrange-Act-Assert shape.

### Security

- Never hardcode secrets. Use environment variables or a gitignored `.env`.
- Never log secrets, tokens, or PII.
- Before invoking an external integration, inspect `.env` variable names only; this workspace may
  provide `GEMINI_API_KEY`, `TRACKMANIARL_DISTRIBUTED_TOKEN`, and `WANDB_API_KEY`. Never read, print, or record
  their values in code, rules, logs, or chat.

### Do Not Do

#### Comments and docstrings

- Do not add comments that restate the code.
- Do not add docstrings that repeat the function name or signature.
- Do not add `Parameters` or `Args` sections that only restate annotated types.
- Do not put implementation details in docstrings.
- Do not add banner or divider comments.
- Use one-line docstrings for trivial methods, or none.
- Do not add `TODO`, `FIXME`, or placeholder comments unless explicitly asked.

#### Over-engineering

- Do not add abstractions for a single use case.
- Do not add config options, parameters, or hooks nobody asked for.
- Do not reimplement stdlib features.
- Do not add backwards-compat shims, deprecation wrappers, or version checks unless requested.

#### Error handling

- Do not use bare `except:` or `except Exception: pass`.
- Do not swallow errors and return `None`, `{}`, or `-1` to hide bugs.
- Do not wrap contract violations in defensive `.get()` or `try/except` blocks that mask the real issue.

#### Noise and filler

- Do not add `if __name__ == "__main__"` demo blocks, example usage, or print debugging.
- Do not use emojis in code, comments, or log messages.
- Do not create extra files unless asked.
- Do not add self-congratulatory or hedging comments.
- Do not restate the task at the top of the file.

#### General

- Match the existing style of the file and codebase.
- Change only what is asked. Avoid opportunistic refactors.
- Do not use `# type: ignore` or `# noqa` to silence checkers.
- Prefer deleting code over adding it.
